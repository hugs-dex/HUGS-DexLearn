import csv
import json
import os
import sys
from copy import deepcopy
from glob import glob
from os.path import join as pjoin

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, open_dict
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dexlearn.dataset import GRASP_TYPES, create_test_dataloader
from dexlearn.network.models import *
from dexlearn.utils.config import flatten_multidex_data_config
from dexlearn.utils.logger import Logger
from dexlearn.utils.util import load_json, set_seed


REAL_GRASP_TYPE_IDS = tuple(range(1, len(GRASP_TYPES)))
REAL_GRASP_TYPE_NUM = len(REAL_GRASP_TYPE_IDS)
DEFAULT_EXCLUDED_GRASP_TYPES = ()


def _grasp_type_id_from_token(token: str) -> int | None:
    """Map a path token or file token to a grasp-type id.

    Args:
        token: Token such as ``right_two`` or ``1_right_two``.

    Returns:
        Integer grasp-type id when the token matches ``GRASP_TYPES``;
        otherwise ``None``.
    """
    token = str(token)
    for type_id, grasp_type in enumerate(GRASP_TYPES):
        if token == grasp_type or grasp_type.endswith(token):
            return type_id
    return None


def _resolve_eval_type_ids(task_config: DictConfig) -> tuple[int, ...]:
    """Resolve real grasp types that should contribute to evaluation metrics.

    Args:
        task_config: Hydra task config for ``robot_type_eval``.

    Returns:
        Tuple of real grasp-type ids that are included in macro metrics and
        output tables.
    """
    raw_excluded = getattr(task_config, "exclude_grasp_types", DEFAULT_EXCLUDED_GRASP_TYPES)
    excluded_type_ids: set[int] = set()
    for raw_token in raw_excluded or []:
        token = str(raw_token)
        type_id = int(token) if token.isdigit() else _grasp_type_id_from_token(token)
        if type_id is None:
            raise ValueError(f"Unknown excluded grasp type: {raw_token!r}")
        if type_id == 0:
            continue
        if type_id not in REAL_GRASP_TYPE_IDS:
            raise ValueError(f"Excluded grasp type must be a real type id, got {raw_token!r}")
        excluded_type_ids.add(type_id)

    eval_type_ids = tuple(type_id for type_id in REAL_GRASP_TYPE_IDS if type_id not in excluded_type_ids)
    if not eval_type_ids:
        raise ValueError("At least one real grasp type must remain after exclude_grasp_types")
    return eval_type_ids


def _scene_id_from_grasp_path(grasp_path: str, grasp_root: str, obj_id: str) -> str:
    """Infer the scene id used by RobotMultiDex grasp records.

    Args:
        grasp_path: Absolute path to one robot grasp ``.npy`` file.
        grasp_root: Root directory that contains grasp-type subfolders.
        obj_id: Object id from the test split.

    Returns:
        Scene id relative to the object id, without the ``.npy`` suffix.
    """
    relative_path = os.path.relpath(grasp_path, grasp_root)
    path_parts = relative_path.split(os.sep)
    obj_idx = path_parts.index(obj_id)
    return os.path.splitext(os.path.join(*path_parts[obj_idx:]))[0]


def _grasp_type_id_from_grasp_path(grasp_path: str, grasp_root: str) -> int:
    """Infer the real grasp-type id from a robot grasp path or payload.

    Args:
        grasp_path: Absolute path to one robot grasp ``.npy`` file.
        grasp_root: Root directory that contains grasp-type subfolders.

    Returns:
        Integer grasp-type id aligned with ``GRASP_TYPES``.
    """
    relative_path = os.path.relpath(grasp_path, grasp_root)
    path_parts = relative_path.split(os.sep)
    if path_parts:
        type_id = _grasp_type_id_from_token(path_parts[0])
        if type_id is not None:
            return type_id

    grasp_data = np.load(grasp_path, allow_pickle=True).item()
    raw_grasp_type = grasp_data.get("grasp_type")
    if raw_grasp_type is None:
        raise KeyError(f"Could not infer grasp_type from path or file: {grasp_path}")
    raw_grasp_type = np.asarray(raw_grasp_type).reshape(-1)[0]
    type_id = _grasp_type_id_from_token(raw_grasp_type)
    if type_id is None:
        raise ValueError(f"Unsupported grasp_type={raw_grasp_type!r} in {grasp_path}")
    return type_id


def _availability_from_counts(type_counts: np.ndarray, min_count: int, min_ratio: float) -> np.ndarray:
    """Convert real-type counts into a binary availability target.

    Args:
        type_counts: Count vector with shape ``(5,)`` for real types ``1..5``.
        min_count: Minimum grasp count required for an available type.
        min_ratio: Minimum scene-level type ratio required for availability.

    Returns:
        Float availability vector with shape ``(5,)``.
    """
    type_counts = np.asarray(type_counts, dtype=np.float32).reshape(REAL_GRASP_TYPE_NUM)
    total_count = float(type_counts.sum())
    if total_count <= 0.0:
        return np.zeros(REAL_GRASP_TYPE_NUM, dtype=np.float32)
    ratios = type_counts / total_count
    return ((type_counts >= int(min_count)) & (ratios >= float(min_ratio))).astype(np.float32)


def _build_robot_type_ground_truth(data_config: DictConfig, split_name: str) -> dict[str, dict]:
    """Build scene-level grasp-type availability targets from full grasp data.

    Args:
        data_config: Flattened RobotMultiDex data config.
        split_name: Object split name, usually ``test`` for this task.

    Returns:
        Mapping ``scene_id -> {availability, counts}`` for real grasp types
        ``1..5``.
    """
    object_path = str(data_config.object_path)
    split_path = str(data_config.split_path)
    grasp_root = str(data_config.grasp_path)
    min_count = int(getattr(data_config, "type_availability_min_count", 1))
    min_ratio = float(getattr(data_config, "type_availability_min_ratio", 0.05))

    object_ids = load_json(pjoin(object_path, split_path, f"{split_name}.json"))
    counts_by_scene: dict[str, np.ndarray] = {}
    for obj_id in tqdm(object_ids, desc="Indexing GT type availability"):
        grasp_paths = sorted(glob(pjoin(grasp_root, "*", obj_id, "**", "*.npy"), recursive=True))
        for grasp_path in grasp_paths:
            scene_id = _scene_id_from_grasp_path(grasp_path, grasp_root, obj_id)
            type_id = _grasp_type_id_from_grasp_path(grasp_path, grasp_root)
            if type_id == 0:
                continue
            if type_id not in REAL_GRASP_TYPE_IDS:
                raise ValueError(f"Unexpected grasp_type_id={type_id} in {grasp_path}")
            if scene_id not in counts_by_scene:
                counts_by_scene[scene_id] = np.zeros(REAL_GRASP_TYPE_NUM, dtype=np.float32)
            counts_by_scene[scene_id][type_id - 1] += 1.0

    return {
        scene_id: {
            "availability": _availability_from_counts(counts, min_count=min_count, min_ratio=min_ratio),
            "counts": counts.astype(np.float32),
        }
        for scene_id, counts in counts_by_scene.items()
    }


def _scene_id_from_scene_path(scene_path: str, object_path: str) -> str:
    """Infer the scene id for a test scene config path.

    Args:
        scene_path: Path returned by the test dataloader.
        object_path: Robot object dataset root.

    Returns:
        Scene id compatible with the grasp-data ground-truth index.
    """
    scene_root = os.path.abspath(pjoin(object_path, "scene_cfg"))
    abs_scene_path = os.path.abspath(str(scene_path))
    if abs_scene_path.startswith(scene_root + os.sep):
        return os.path.splitext(os.path.relpath(abs_scene_path, scene_root))[0]

    scene_cfg = np.load(scene_path, allow_pickle=True).item()
    scene_id = scene_cfg.get("scene_id")
    if scene_id is None and "scene" in scene_cfg:
        scene = scene_cfg["scene"]
        scene_id = scene.get("id", scene.get("scene_id"))
    if scene_id is None:
        raise KeyError(f"Could not infer scene id from scene config: {scene_path}")
    return str(scene_id)


def _predict_robot_type_scores(model: torch.nn.Module, data: dict) -> torch.Tensor:
    """Predict five real-type availability scores from the model type head.

    Args:
        model: Robot hierarchical model with ``backbone`` and ``type_classifier``.
        data: Batch dictionary already moved to the configured device.

    Returns:
        Tensor with shape ``(B, 5)`` for real grasp types ``1..5``.
    """
    global_feature, _ = model.backbone(data)
    type_logits = model.type_classifier(global_feature)
    type_objective = str(getattr(model, "type_objective", "")).lower()
    if type_objective == "availability" and hasattr(model, "_availability_scores"):
        return model._availability_scores(type_logits)
    if type_logits.shape[-1] == len(GRASP_TYPES):
        return F.softmax(type_logits, dim=-1)[:, 1:]
    if type_logits.shape[-1] == REAL_GRASP_TYPE_NUM:
        return F.softmax(type_logits, dim=-1)
    raise ValueError(f"Unsupported type head output shape: {tuple(type_logits.shape)}")


def _aggregate_scene_predictions(records: list[dict]) -> dict[str, dict]:
    """Average repeated predictions for the same scene id.

    Args:
        records: Per-point-cloud prediction rows with ``scene_id`` and
            ``scores`` fields.

    Returns:
        Mapping ``scene_id -> {scores, scene_path, row_count}``.
    """
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(record["scene_id"], []).append(record)

    aggregated = {}
    for scene_id, scene_records in grouped.items():
        scores = np.stack([row["scores"] for row in scene_records], axis=0).mean(axis=0)
        aggregated[scene_id] = {
            "scores": scores,
            "scene_path": scene_records[0]["scene_path"],
            "row_count": len(scene_records),
        }
    return aggregated


def _compute_precision_recall(
    predictions: dict[str, dict],
    ground_truth: dict[str, dict],
    threshold: float,
    eval_type_ids: tuple[int, ...],
) -> dict:
    """Compute per-grasp-type precision, recall and F1.

    Args:
        predictions: Mapping ``scene_id -> {scores}`` from model inference.
        ground_truth: Mapping ``scene_id -> {availability, counts}``.
        threshold: Score threshold used to binarize predicted availability.
        eval_type_ids: Real grasp-type ids included in metrics.

    Returns:
        Summary dictionary with per-type confusion counts and metrics.
    """
    metrics = {}
    matched_scene_ids = sorted(set(predictions) & set(ground_truth))
    precision_values = []
    recall_values = []
    f1_values = []
    for type_id in eval_type_ids:
        type_offset = type_id - 1
        tp = fp = fn = tn = 0
        for scene_id in matched_scene_ids:
            pred_available = bool(predictions[scene_id]["scores"][type_offset] >= threshold)
            gt_available = bool(ground_truth[scene_id]["availability"][type_offset] > 0.5)
            if pred_available and gt_available:
                tp += 1
            elif pred_available and not gt_available:
                fp += 1
            elif not pred_available and gt_available:
                fn += 1
            else:
                tn += 1
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        metrics[str(type_id)] = {
            "grasp_type": GRASP_TYPES[type_id],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "pred_positive": int(tp + fp),
            "gt_positive": int(tp + fn),
        }
    return {
        "threshold": float(threshold),
        "scene_count": int(len(matched_scene_ids)),
        "missing_ground_truth_scene_count": int(len(set(predictions) - set(ground_truth))),
        "unused_ground_truth_scene_count": int(len(set(ground_truth) - set(predictions))),
        "macro_precision": float(np.mean(precision_values)) if precision_values else 0.0,
        "macro_recall": float(np.mean(recall_values)) if recall_values else 0.0,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "included_type_ids": [int(type_id) for type_id in eval_type_ids],
        "included_grasp_types": [GRASP_TYPES[type_id] for type_id in eval_type_ids],
        "excluded_type_ids": [int(type_id) for type_id in REAL_GRASP_TYPE_IDS if type_id not in eval_type_ids],
        "excluded_grasp_types": [GRASP_TYPES[type_id] for type_id in REAL_GRASP_TYPE_IDS if type_id not in eval_type_ids],
        "per_type": metrics,
    }


def _resolve_thresholds(task_config: DictConfig) -> list[float]:
    """Resolve the threshold sweep grid from task config.

    Args:
        task_config: Hydra task config for ``robot_type_eval``.

    Returns:
        Sorted unique threshold list. Explicit ``thresholds`` values take
        priority; otherwise ``threshold_min/max/step`` define the grid.
    """
    explicit_thresholds = getattr(task_config, "thresholds", None)
    if explicit_thresholds:
        thresholds = [float(value) for value in explicit_thresholds]
    else:
        threshold_min = float(getattr(task_config, "threshold_min", 0.0))
        threshold_max = float(getattr(task_config, "threshold_max", 1.0))
        threshold_step = float(getattr(task_config, "threshold_step", 0.05))
        if threshold_step <= 0.0:
            raise ValueError("task.threshold_step must be positive")
        if threshold_min > threshold_max:
            raise ValueError("task.threshold_min must be <= task.threshold_max")
        thresholds = []
        value = threshold_min
        while value <= threshold_max + threshold_step * 0.5:
            thresholds.append(float(round(value, 10)))
            value += threshold_step

    cleaned = sorted({float(value) for value in thresholds})
    for value in cleaned:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"All thresholds must be in [0, 1], got {value}")
    if not cleaned:
        raise ValueError("Threshold sweep is empty")
    return cleaned


def _compute_threshold_sweep(
    predictions: dict[str, dict],
    ground_truth: dict[str, dict],
    thresholds: list[float],
    eval_type_ids: tuple[int, ...],
) -> list[dict]:
    """Evaluate all configured thresholds.

    Args:
        predictions: Mapping ``scene_id -> {scores}`` from model inference.
        ground_truth: Mapping ``scene_id -> {availability, counts}``.
        thresholds: Threshold values used to binarize predicted scores.
        eval_type_ids: Real grasp-type ids included in metrics.

    Returns:
        List of metric summaries, one per threshold.
    """
    return [
        _compute_precision_recall(predictions, ground_truth, threshold=value, eval_type_ids=eval_type_ids)
        for value in thresholds
    ]


def _best_threshold_report(threshold_summaries: list[dict], eval_type_ids: tuple[int, ...]) -> dict:
    """Find convenient best-threshold suggestions from a sweep.

    Args:
        threshold_summaries: List of summaries returned by
            ``_compute_threshold_sweep``.
        eval_type_ids: Real grasp-type ids included in metrics.

    Returns:
        Dictionary containing the best macro-F1 threshold and best per-type F1
        threshold suggestions.
    """
    if not threshold_summaries:
        return {"best_macro_f1": None, "best_per_type_f1": {}}

    best_macro = max(threshold_summaries, key=lambda row: (row["macro_f1"], row["macro_recall"], -row["threshold"]))
    best_per_type = {}
    for type_id in eval_type_ids:
        type_key = str(type_id)
        best_row = max(
            threshold_summaries,
            key=lambda row: (
                row["per_type"][type_key]["f1"],
                row["per_type"][type_key]["recall"],
                -row["threshold"],
            ),
        )
        best_metric = best_row["per_type"][type_key]
        best_per_type[type_key] = {
            "grasp_type": GRASP_TYPES[type_id],
            "threshold": float(best_row["threshold"]),
            "precision": float(best_metric["precision"]),
            "recall": float(best_metric["recall"]),
            "f1": float(best_metric["f1"]),
        }

    return {
        "best_macro_f1": {
            "threshold": float(best_macro["threshold"]),
            "macro_precision": float(best_macro["macro_precision"]),
            "macro_recall": float(best_macro["macro_recall"]),
            "macro_f1": float(best_macro["macro_f1"]),
        },
        "best_per_type_f1": best_per_type,
    }


def _write_threshold_sweep_outputs(
    output_dir: str,
    threshold_summaries: list[dict],
    eval_type_ids: tuple[int, ...],
) -> None:
    """Write threshold sweep tables to disk.

    Args:
        output_dir: Directory where report files should be saved.
        threshold_summaries: List of summaries returned by
            ``_compute_threshold_sweep``.
        eval_type_ids: Real grasp-type ids included in metrics.

    Returns:
        None.
    """
    summary_fields = ["threshold", "macro_precision", "macro_recall", "macro_f1", "scene_count"]
    with open(pjoin(output_dir, "threshold_sweep_summary.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for summary in threshold_summaries:
            writer.writerow({key: summary[key] for key in summary_fields})

    per_type_fields = [
        "threshold",
        "type_id",
        "grasp_type",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
        "tn",
        "pred_positive",
        "gt_positive",
    ]
    with open(pjoin(output_dir, "threshold_sweep_per_type.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_type_fields)
        writer.writeheader()
        for summary in threshold_summaries:
            for type_id in eval_type_ids:
                metric = summary["per_type"][str(type_id)]
                writer.writerow(
                    {
                        "threshold": summary["threshold"],
                        "type_id": type_id,
                        "grasp_type": metric["grasp_type"],
                        "precision": metric["precision"],
                        "recall": metric["recall"],
                        "f1": metric["f1"],
                        "tp": metric["tp"],
                        "fp": metric["fp"],
                        "fn": metric["fn"],
                        "tn": metric["tn"],
                        "pred_positive": metric["pred_positive"],
                        "gt_positive": metric["gt_positive"],
                    }
                )


def _write_outputs(
    output_dir: str,
    predictions: dict[str, dict],
    ground_truth: dict[str, dict],
    summary: dict,
    threshold_summaries: list[dict] | None = None,
    eval_type_ids: tuple[int, ...] = REAL_GRASP_TYPE_IDS,
) -> None:
    """Write robot type evaluation reports to disk.

    Args:
        output_dir: Directory where report files should be saved.
        predictions: Aggregated model predictions by scene id.
        ground_truth: Ground-truth availability and counts by scene id.
        summary: Metric summary returned by ``_compute_precision_recall``.
        threshold_summaries: Optional sweep summaries for additional CSV
            reports.
        eval_type_ids: Real grasp-type ids included in metrics and tables.

    Returns:
        None.
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(pjoin(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if threshold_summaries is not None:
        _write_threshold_sweep_outputs(output_dir, threshold_summaries, eval_type_ids=eval_type_ids)

    row_fields = ["scene_id", "scene_path", "row_count"]
    row_fields += [f"pred_score_{type_id}_{GRASP_TYPES[type_id]}" for type_id in eval_type_ids]
    row_fields += [f"pred_available_{type_id}_{GRASP_TYPES[type_id]}" for type_id in eval_type_ids]
    row_fields += [f"gt_available_{type_id}_{GRASP_TYPES[type_id]}" for type_id in eval_type_ids]
    row_fields += [f"gt_count_{type_id}_{GRASP_TYPES[type_id]}" for type_id in eval_type_ids]
    with open(pjoin(output_dir, "per_scene.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row_fields)
        writer.writeheader()
        for scene_id in sorted(predictions):
            if scene_id not in ground_truth:
                continue
            pred = predictions[scene_id]
            gt = ground_truth[scene_id]
            row = {
                "scene_id": scene_id,
                "scene_path": pred["scene_path"],
                "row_count": pred["row_count"],
            }
            for type_id in eval_type_ids:
                offset = type_id - 1
                row[f"pred_score_{type_id}_{GRASP_TYPES[type_id]}"] = float(pred["scores"][offset])
                row[f"pred_available_{type_id}_{GRASP_TYPES[type_id]}"] = int(
                    pred["scores"][offset] >= summary["threshold"]
                )
                row[f"gt_available_{type_id}_{GRASP_TYPES[type_id]}"] = int(gt["availability"][offset] > 0.5)
                row[f"gt_count_{type_id}_{GRASP_TYPES[type_id]}"] = float(gt["counts"][offset])
            writer.writerow(row)

    with open(pjoin(output_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("# Robot Type Availability Evaluation\n\n")
        f.write(f"- threshold: {summary['threshold']:.4f}\n")
        f.write(f"- matched scenes: {summary['scene_count']}\n")
        f.write(f"- missing GT scenes: {summary['missing_ground_truth_scene_count']}\n")
        f.write(f"- unused GT scenes: {summary['unused_ground_truth_scene_count']}\n\n")
        f.write(f"- included grasp types: {', '.join(summary['included_grasp_types'])}\n")
        f.write(f"- excluded grasp types: {', '.join(summary['excluded_grasp_types']) or 'none'}\n\n")
        f.write(
            f"- macro precision/recall/F1: "
            f"{summary['macro_precision']:.6f} / {summary['macro_recall']:.6f} / {summary['macro_f1']:.6f}\n\n"
        )
        best_report = summary.get("threshold_selection", {})
        best_macro = best_report.get("best_macro_f1")
        if best_macro is not None:
            f.write("## Threshold Sweep Suggestions\n\n")
            f.write(
                f"- best macro-F1 threshold: {best_macro['threshold']:.4f} "
                f"(P={best_macro['macro_precision']:.6f}, R={best_macro['macro_recall']:.6f}, "
                f"F1={best_macro['macro_f1']:.6f})\n\n"
            )
            f.write("| type | best threshold | precision | recall | F1 |\n")
            f.write("| --- | ---: | ---: | ---: | ---: |\n")
            for type_id in eval_type_ids:
                row = best_report["best_per_type_f1"][str(type_id)]
                f.write(
                    f"| {row['grasp_type']} | {row['threshold']:.4f} | "
                    f"{row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f} |\n"
                )
            f.write("\n")

        f.write("## Selected Threshold Metrics\n\n")
        f.write("| type | precision | recall | F1 | tp | fp | fn | tn |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for type_id in eval_type_ids:
            row = summary["per_type"][str(type_id)]
            f.write(
                f"| {row['grasp_type']} | {row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f} | "
                f"{row['tp']} | {row['fp']} | {row['fn']} | {row['tn']} |\n"
            )


def _print_summary(summary: dict) -> None:
    """Print a compact metric table to stdout.

    Args:
        summary: Metric summary returned by ``_compute_precision_recall``.

    Returns:
        None.
    """
    print(
        f"Robot type eval: threshold={summary['threshold']:.4f}, "
        f"matched_scenes={summary['scene_count']}, "
        f"missing_gt={summary['missing_ground_truth_scene_count']}"
    )
    best_macro = summary.get("threshold_selection", {}).get("best_macro_f1")
    if best_macro is not None:
        print(
            f"best_macro_f1_threshold={best_macro['threshold']:.4f}, "
            f"macro_f1={best_macro['macro_f1']:.6f}"
        )
    print(f"included_types={','.join(summary['included_grasp_types'])}")
    print(f"excluded_types={','.join(summary['excluded_grasp_types']) or 'none'}")
    print("type\tprecision\trecall\tf1\ttp\tfp\tfn\ttn")
    for type_id in summary["included_type_ids"]:
        row = summary["per_type"][str(type_id)]
        print(
            f"{row['grasp_type']}\t{row['precision']:.6f}\t{row['recall']:.6f}\t{row['f1']:.6f}\t"
            f"{row['tp']}\t{row['fp']}\t{row['fn']}\t{row['tn']}"
        )


def task_robot_type_eval(config: DictConfig):
    """Evaluate predicted robot grasp-type availability against GT labels.

    Args:
        config: Full Hydra config. Expected launch style:
            ``task=robot_type_eval algo=robotMultiHierar data=leapspMulti
            test_data=leapspMulti exp_name=...``.

    Returns:
        None.
    """
    set_seed(config.seed)
    config.wandb.mode = "disabled"
    logger = Logger(config)

    if config.ckpt is None:
        raise FileNotFoundError("Could not find a checkpoint. Set ckpt=... or enable resume with existing ckpts.")
    ckpt = torch.load(config.ckpt, map_location="cpu")
    ckpt_iter = int(ckpt.get("iter", 0))
    model = eval(config.algo.model.name)(config.algo.model)
    model.load_state_dict(ckpt["model"])
    model.to(config.device)
    model.eval()
    print("loaded ckpt from", config.ckpt)

    data_config = deepcopy(config.data)
    flatten_multidex_data_config(data_config)
    gt_data_config = deepcopy(config.test_data)
    flatten_multidex_data_config(gt_data_config)
    with open_dict(gt_data_config):
        if not hasattr(gt_data_config, "type_availability_min_count"):
            gt_data_config.type_availability_min_count = int(getattr(data_config, "type_availability_min_count", 1))
        if not hasattr(gt_data_config, "type_availability_min_ratio"):
            gt_data_config.type_availability_min_ratio = float(getattr(data_config, "type_availability_min_ratio", 0.05))
    split_name = str(getattr(config.task, "gt_split", getattr(config.test_data, "test_split", "test")))
    ground_truth = _build_robot_type_ground_truth(gt_data_config, split_name=split_name)

    test_loader = create_test_dataloader(config)
    object_path = str(gt_data_config.object_path)
    max_batches = int(getattr(config.task, "max_batches", 0))
    records = []
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader, desc="Predicting type availability")):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            scores = _predict_robot_type_scores(model, data).detach().cpu().numpy()
            for row_idx, scene_path in enumerate(data["scene_path"]):
                scene_id = _scene_id_from_scene_path(scene_path, object_path)
                records.append(
                    {
                        "scene_id": scene_id,
                        "scene_path": str(scene_path),
                        "scores": scores[row_idx].astype(np.float32),
                    }
                )

    predictions = _aggregate_scene_predictions(records)
    eval_type_ids = _resolve_eval_type_ids(config.task)
    thresholds = _resolve_thresholds(config.task)
    threshold_summaries = _compute_threshold_sweep(
        predictions,
        ground_truth,
        thresholds=thresholds,
        eval_type_ids=eval_type_ids,
    )
    threshold = float(getattr(config.task, "threshold", 0.5))
    summary = _compute_precision_recall(
        predictions,
        ground_truth,
        threshold=threshold,
        eval_type_ids=eval_type_ids,
    )
    summary["checkpoint"] = str(config.ckpt)
    summary["checkpoint_iter"] = ckpt_iter
    summary["raw_prediction_row_count"] = int(len(records))
    summary["prediction_scene_count"] = int(len(predictions))
    summary["sweep_thresholds"] = thresholds
    summary["threshold_selection"] = _best_threshold_report(threshold_summaries, eval_type_ids=eval_type_ids)

    default_output_dir = pjoin(logger.save_test_dir, f"step_{str(ckpt_iter).zfill(6)}", "robot_type_eval")
    output_dir = str(getattr(config.task, "output_dir", "") or default_output_dir)
    _write_outputs(
        output_dir,
        predictions,
        ground_truth,
        summary,
        threshold_summaries=threshold_summaries,
        eval_type_ids=eval_type_ids,
    )
    _print_summary(summary)
    print(f"Saved robot type evaluation to {output_dir}")
