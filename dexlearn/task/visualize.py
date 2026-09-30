import time
import os
import json
import random
import re
from glob import escape as glob_escape
from glob import glob

import hydra
import numpy as np
import trimesh
from omegaconf import DictConfig, OmegaConf
from dexlearn.utils.config import flatten_multidex_data_config
from dexlearn.dataset.grasp_types import GRASP_TYPES

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import viser

    VISER_AVAILABLE = True
except ImportError:
    viser = None
    VISER_AVAILABLE = False


VISUALIZE_MODE_OPTIONS = ("random_objects", "one_scene", "one_object", "one_object_multi_seq", "grasp_type")
ROBOT_VISER_MODE_OPTIONS = ("random_objects", "one_scene", "grasp_type")
HUMAN_VISER_MODE_OPTIONS = ("random_objects", "one_object")
HUMAN_VISER_SPLIT_OPTIONS = ("all", "test", "train")
HUMAN_SCORE_SUMMARY_MAX_RECORDS = 20
CAPTION_ASPECTS = (
    ("scene_id", "Scene ID"),
    ("object_id", "Object ID"),
    ("file", "File"),
    ("given_grasp_type", "Given Grasp Type"),
    ("pred_grasp_type", "Pred Grasp Type"),
    ("pred_grasp_type_prob", "Pred Grasp Type Prob"),
    ("position_source", "Position Source"),
    ("point_cloud", "Point Cloud"),
    ("error", "Error"),
    ("ik_status", "IK Status"),
    ("other", "Other"),
)


def progress_iter(iterable, desc: str, total=None):
    """Wrap an iterable with a progress bar when tqdm is available.

    Args:
        iterable: Iterable to wrap.
        desc: Progress bar description.
        total: Optional total item count for iterables without ``len``.

    Returns:
        An iterable that yields the same items, with a tqdm progress bar when
        the dependency is installed.
    """
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total)


def get_task_value(config: DictConfig, key: str, default):
    value = OmegaConf.select(config, f"task.{key}")
    return default if value is None else value


def normalize_visualize_mode(mode: str):
    aliases = {
        "random": "random_objects",
        "random_object": "random_objects",
        "scene": "one_scene",
        "single_scene": "one_scene",
        "object": "one_object",
        "single_object": "one_object",
        "multi_seq": "one_object_multi_seq",
        "multi_sequence": "one_object_multi_seq",
        "one_object_multi_sequence": "one_object_multi_seq",
        "grasp_type_id": "grasp_type",
        "type": "grasp_type",
    }
    mode = str(mode).lower()
    return aliases.get(mode, mode)


def list_sample_files(output_dir: str):
    return sorted(glob(os.path.join(output_dir, "**/*.npy"), recursive=True))


def is_configured_path(value) -> bool:
    """Return whether a config value names a real user-provided path.

    Args:
        value: Raw OmegaConf value that may be ``None`` or an empty string.

    Returns:
        ``True`` when the value should be interpreted as a configured path.
    """
    if value is None:
        return False
    return str(value).strip().lower() not in {"", "none", "null"}


def infer_sample_group_from_sample_path(output_dir: str, sample_file: str):
    """Infer the first-level saved sample group from a sample path.

    Args:
        output_dir: Root directory that contains saved visualization samples.
        sample_file: Path to one saved sample file.

    Returns:
        The first relative path component under ``output_dir`` such as
        ``0_any`` or ``3_right_full``. Returns an empty string when the path is
        too short to contain a saved sample group.
    """
    try:
        rel_path = os.path.relpath(sample_file, output_dir)
    except ValueError:
        rel_path = sample_file
    parts = rel_path.split(os.sep)
    return parts[0] if len(parts) >= 2 else ""


def infer_object_id_from_sample_path(output_dir: str, sample_file: str):
    """Infer a scene-like object id from the saved sample path.

    Args:
        output_dir: Root directory that contains saved visualization samples.
        sample_file: Path to one saved sample file.

    Returns:
        A slash-separated id inferred from the relative sample path. The first
        relative component is the logger group such as ``0_any`` and is skipped.
    """
    try:
        rel_path = os.path.relpath(sample_file, output_dir)
    except ValueError:
        rel_path = sample_file
    parts = rel_path.split(os.sep)
    if len(parts) >= 3:
        return "/".join(parts[1:-1])
    return os.path.splitext(os.path.basename(sample_file))[0]


def sample_group_to_grasp_type_id(sample_group: str):
    """Map a saved sample group name to a grasp type id when possible.

    Args:
        sample_group: First relative path component under one sample root.

    Returns:
        Integer grasp type id such as ``0`` or ``3`` when the name starts with
        a numeric prefix, otherwise ``None``.
    """
    match = re.match(r"^(\d+)_", str(sample_group))
    return int(match.group(1)) if match else None


def normalize_human_score_vector(scores, grasp_type_names):
    """Normalize human type scores to real grasp types ``1..5``.

    Args:
        scores: Raw score/probability vector saved by the sampler.
        grasp_type_names: Ordered grasp type names including ``0_any``.

    Returns:
        A float64 vector aligned with real grasp type ids ``1..5``, or
        ``None`` when the raw vector shape is not recognized.
    """
    score_values = np.asarray(scores, dtype=np.float64).reshape(-1)
    real_type_num = len(grasp_type_names) - 1
    if score_values.size == len(grasp_type_names):
        return score_values[1:].astype(np.float64)
    if score_values.size == real_type_num:
        return score_values.astype(np.float64)
    return None


def get_human_score_vector_from_data(data: dict, grasp_type_names):
    """Read a normalized human type-score vector from one saved sample.

    Args:
        data: Saved visualization sample payload.
        grasp_type_names: Ordered grasp type names including ``0_any``.

    Returns:
        A float64 vector aligned with real grasp type ids ``1..5``, or
        ``None`` when the sample does not contain compatible scores.
    """
    if "pred_grasp_type_prob" not in data:
        return None
    return normalize_human_score_vector(data["pred_grasp_type_prob"], grasp_type_names)


def extract_grasp_type_id(data: dict):
    if "pred_grasp_type_id" in data:
        return int(np.asarray(data["pred_grasp_type_id"]).reshape(-1)[0])
    if "grasp_type_id" in data:
        return int(np.asarray(data["grasp_type_id"]).reshape(-1)[0])
    return None


def infer_object_id_from_scene_cfg(scene_cfg: dict, scene_path: str):
    if "scene_id" in scene_cfg:
        return str(scene_cfg["scene_id"])
    if "object" in scene_cfg and isinstance(scene_cfg["object"], dict):
        obj_name = scene_cfg["object"].get("name")
        if obj_name is not None:
            return str(obj_name)
    task_obj_name = scene_cfg.get("task", {}).get("obj_name") if isinstance(scene_cfg.get("task"), dict) else None
    if task_obj_name is not None:
        return str(task_obj_name)
    return os.path.splitext(os.path.basename(scene_path))[0]


def load_visualization_record_payload(record: dict, scene_path_resolver=None, load_scene_cfg: bool = True):
    """Load sample data and optional scene config into a lightweight record.

    Args:
        record: Sample record created by ``build_visualization_sample_index``.
        scene_path_resolver: Optional callable that maps saved scene paths to
            paths that are valid in the current workspace.
        load_scene_cfg: Whether to load the referenced scene config.

    Returns:
        The same record dictionary after adding loaded payload fields.
    """
    if record.get("data") is not None and (not load_scene_cfg or record.get("scene_cfg")):
        return record

    sample_file = record["sample_file"]
    data = np.load(sample_file, allow_pickle=True).item()
    scene_path = str(data.get("scene_path", ""))
    if scene_path_resolver is not None and scene_path:
        scene_path = scene_path_resolver(scene_path)

    scene_cfg = record.get("scene_cfg") or {}
    if load_scene_cfg and scene_path:
        scene_cfg = np.load(scene_path, allow_pickle=True).item()

    record["data"] = data
    record["scene_path"] = scene_path
    record["scene_cfg"] = scene_cfg
    record["grasp_type_id"] = extract_grasp_type_id(data)
    if scene_cfg:
        record["object_id"] = infer_object_id_from_scene_cfg(scene_cfg, scene_path)
    return record


def make_visualization_sample_record(output_dir: str, sample_file: str):
    """Create a lightweight visualization record for one saved sample file.

    Args:
        output_dir: Directory that contains saved visualization samples.
        sample_file: Path to one saved ``.npy`` sample file.

    Returns:
        A sample record with path-derived metadata. The heavy sample payload is
        loaded later only if this record is selected for display.
    """
    return {
        "sample_file": sample_file,
        "data": None,
        "scene_path": "",
        "scene_cfg": {},
        "sample_group": infer_sample_group_from_sample_path(output_dir, sample_file),
        "object_id": infer_object_id_from_sample_path(output_dir, sample_file),
        "grasp_type_id": None,
    }


def build_visualization_sample_records(
    output_dir: str,
    sample_files,
    scene_path_resolver=None,
    load_payload: bool = True,
    load_scene_cfg: bool = True,
    log_prefix: str = "Indexing saved visualization samples",
):
    """Build visualization records from an explicit saved-sample file list.

    Args:
        output_dir: Directory that contains saved visualization samples.
        sample_files: Iterable of saved ``.npy`` sample paths to index.
        scene_path_resolver: Optional callable used to resolve saved scene
            paths when payloads are loaded.
        load_payload: Whether to load every sample file while indexing.
        load_scene_cfg: Whether to load every scene config while indexing.
        log_prefix: Progress-bar description for the indexing pass.

    Returns:
        A list of visualization sample records.
    """
    sample_records = []
    sample_files = sorted(sample_files, key=natural_sort_key)
    file_iter = progress_iter(sample_files, desc=log_prefix, total=len(sample_files))
    for idx, sample_file in enumerate(file_iter, 1):
        record = make_visualization_sample_record(output_dir, sample_file)
        if not load_payload:
            sample_records.append(record)
            continue
        try:
            load_visualization_record_payload(
                record,
                scene_path_resolver=scene_path_resolver,
                load_scene_cfg=load_scene_cfg,
            )
        except Exception as exc:
            print(f"[visualize] Skipping unreadable sample {sample_file}: {exc}")
            continue
        if idx % 50000 == 0:
            print(f"[visualize] Indexed {idx}/{len(sample_files)} sample file(s).")
        sample_records.append(record)
    return sample_records


def build_visualization_sample_index(
    output_dir: str,
    scene_path_resolver=None,
    load_payload: bool = True,
    load_scene_cfg: bool = True,
):
    """Build an index for saved visualization samples.

    Args:
        output_dir: Directory containing saved ``.npy`` sample files.
        scene_path_resolver: Optional callable used to resolve saved scene paths.
        load_payload: Whether to load every sample file while indexing.
        load_scene_cfg: Whether to load every scene config while indexing.

    Returns:
        A list of sample records. With ``load_payload=False``, records contain
        only path-derived metadata and are hydrated later for selected samples.
    """
    sample_records = []
    sample_files = list_sample_files(output_dir)
    print(
        f"[visualize] Found {len(sample_files)} saved sample file(s) in {output_dir}. "
        f"load_payload={load_payload}, load_scene_cfg={load_scene_cfg}"
    )
    sample_records = build_visualization_sample_records(
        output_dir,
        sample_files,
        scene_path_resolver=scene_path_resolver,
        load_payload=load_payload,
        load_scene_cfg=load_scene_cfg,
    )
    return sample_records


def limit_records(records, max_grasps: int):
    if max_grasps <= 0 or len(records) <= max_grasps:
        return records
    return random.sample(records, k=max_grasps)


def randomize_records(records, max_grasps: int):
    """Return a randomized subset of records.

    Args:
        records: Candidate records or ids.
        max_grasps: Maximum number of records to return. Non-positive values
            return all candidates in randomized order.

    Returns:
        A list sampled from the candidate pool without preferring path-sorted
        prefixes.
    """
    records = list(records)
    if max_grasps <= 0 or len(records) <= max_grasps:
        random.shuffle(records)
        return records
    return random.sample(records, k=max_grasps)


def slice_records_batch(records, max_grasps: int, batch_index: int):
    """Return one circular batch from an ordered record list.

    Args:
        records: Ordered records or ids to batch.
        max_grasps: Maximum number of entries in each batch. Non-positive
            values keep all records in one batch.
        batch_index: Zero-based batch index to show.

    Returns:
        A list containing the selected batch. When ``batch_index`` exceeds the
        number of available batches, selection wraps around.
    """
    records = list(records)
    if max_grasps <= 0 or len(records) <= max_grasps:
        return records
    batch_index = max(0, int(batch_index))
    start = (batch_index * max_grasps) % len(records)
    end = start + max_grasps
    if end <= len(records):
        return records[start:end]
    return records[start:] + records[: end - len(records)]


def object_id_matches(record_object_id: str, target_object_id: str):
    record_object_id = str(record_object_id)
    target_object_id = str(target_object_id)
    return (
        record_object_id == target_object_id
        or record_object_id.startswith(f"{target_object_id}/")
        or os.path.basename(record_object_id) == target_object_id
    )


def natural_sort_key(value):
    return [int(token) if token.isdigit() else token for token in re.split(r"(\d+)", str(value))]


def canonical_object_id(object_id):
    return str(object_id).strip().strip("/")


def base_object_id_from_sequence(object_id):
    object_id = canonical_object_id(object_id)
    if "/" in object_id:
        return object_id.split("/", 1)[0]
    match = re.match(r"^(.+)_seq_[^/]+$", object_id)
    return match.group(1) if match else object_id


def sequence_object_id_matches(record_object_id: str, target_object_id: str):
    return base_object_id_from_sequence(record_object_id) == base_object_id_from_sequence(target_object_id)


def is_our_human_grasp_format_test_data(config: DictConfig):
    try:
        object_path = (
            OmegaConf.select(config, "test_data.object_path")
            or OmegaConf.select(config, "test_data.paths.object_path")
        )
    except Exception:
        object_path = None
    return object_path is not None and "OurHumanGraspFormat" in str(object_path)


def select_evenly_across_sequences(records, max_grasps: int):
    grouped = {}
    for record in sorted(records, key=lambda item: item["sample_file"]):
        grouped.setdefault(canonical_object_id(record["object_id"]), []).append(record)

    selected = []
    sequence_ids = sorted(grouped.keys(), key=natural_sort_key)
    while sequence_ids:
        next_sequence_ids = []
        for sequence_id in sequence_ids:
            if not grouped[sequence_id]:
                continue
            selected.append(grouped[sequence_id].pop(0))
            if max_grasps > 0 and len(selected) >= max_grasps:
                return selected
            if grouped[sequence_id]:
                next_sequence_ids.append(sequence_id)
        sequence_ids = next_sequence_ids
    return selected


def select_random_across_sequences(records, max_grasps: int):
    """Select records randomly while spreading them across sequence ids.

    Args:
        records: Candidate records with ``object_id`` metadata.
        max_grasps: Maximum number of records to return.

    Returns:
        Randomly selected records, round-robin across sequence ids when
        multiple sequence ids are present.
    """
    grouped = {}
    for record in records:
        grouped.setdefault(canonical_object_id(record["object_id"]), []).append(record)
    if not grouped:
        return []

    for group_records in grouped.values():
        random.shuffle(group_records)
    sequence_ids = list(grouped.keys())
    random.shuffle(sequence_ids)

    selected = []
    while sequence_ids:
        next_sequence_ids = []
        for sequence_id in sequence_ids:
            if not grouped[sequence_id]:
                continue
            selected.append(grouped[sequence_id].pop())
            if max_grasps > 0 and len(selected) >= max_grasps:
                return selected
            if grouped[sequence_id]:
                next_sequence_ids.append(sequence_id)
        random.shuffle(next_sequence_ids)
        sequence_ids = next_sequence_ids
    return selected


def object_variant_bucket_id(object_id: str):
    """Return the coarse scale bucket for a scene-like object id.

    Args:
        object_id: Full object id inferred from a saved sample path.

    Returns:
        A bucket id used to interleave scene variants. DGN ids end with tokens
        like ``scale002_pose000_0``; grouping by ``scale002`` prevents the first
        one-object batch from showing only the smallest scale.
    """
    leaf = canonical_object_id(object_id).split("/")[-1]
    match = re.match(r"^(scale\d+)_pose", leaf)
    return match.group(1) if match else leaf


def interleave_object_variant_ids(object_ids):
    """Interleave full object ids across coarse variant buckets.

    Args:
        object_ids: Full object ids, usually one per scene/scale/pose variant.

    Returns:
        A list of object ids ordered round-robin across buckets.
    """
    buckets = {}
    for object_id in sorted(object_ids, key=natural_sort_key):
        buckets.setdefault(object_variant_bucket_id(object_id), []).append(object_id)

    ordered = []
    bucket_ids = sorted(buckets.keys(), key=natural_sort_key)
    while bucket_ids:
        next_bucket_ids = []
        for bucket_id in bucket_ids:
            if not buckets[bucket_id]:
                continue
            ordered.append(buckets[bucket_id].pop(0))
            if buckets[bucket_id]:
                next_bucket_ids.append(bucket_id)
        bucket_ids = next_bucket_ids
    return ordered


def select_one_object_variant_batch(records, max_grasps: int, batch_index: int):
    """Randomly select a one-object batch across full scene variants.

    Args:
        records: Records matched by the selected object id.
        max_grasps: Maximum number of scenes to show.
        batch_index: Zero-based batch index. This is accepted for compatibility
            with the Selection panel state; the selected batch is randomized
            rather than sliced by index.

    Returns:
        Selected records. When a selected base object expands to many DGN
        scale/pose variants, the batch samples scene variants randomly instead
        of taking the first lexicographic scale only.
    """
    del batch_index
    records = list(records)
    if max_grasps <= 0:
        return randomize_records(records, max_grasps)
    grouped = {}
    for record in records:
        grouped.setdefault(canonical_object_id(record["object_id"]), []).append(record)
    if len(grouped) <= 1:
        return randomize_records(records, max_grasps)

    selected_object_ids = randomize_records(grouped.keys(), max_grasps)
    selected = []
    for object_id in selected_object_ids:
        object_records = grouped[object_id]
        selected.append(random.choice(object_records))
    if len(selected) < max_grasps and len(selected) < len(records):
        used_paths = {record["sample_file"] for record in selected}
        remaining = [record for record in records if record["sample_file"] not in used_paths]
        selected.extend(randomize_records(remaining, max_grasps - len(selected)))
    return selected


def annotate_scene_labels(records, mode: str):
    for record in records:
        record.pop("viser_all_label", None)
        record.pop("viser_spatial_group", None)
        record.pop("visualize_mode", None)

    for idx, record in enumerate(records):
        record["visualize_mode"] = mode
        record["viser_all_label"] = f"{idx} | {canonical_object_id(record['object_id'])}"
        if mode == "one_object_multi_seq":
            record["viser_spatial_group"] = canonical_object_id(record["object_id"])
    return records


def prefix_caption_with_viser_label(sample_record, caption: str):
    label = sample_record.get("viser_all_label")
    return f"{label} | {caption}" if label else caption


def copy_viser_record_metadata(sample_record, scene_record):
    for key in (
        "viser_all_label",
        "viser_spatial_group",
        "viser_grid_row",
        "viser_grid_col",
        "viser_grid_rows",
        "viser_grid_cols",
    ):
        if key in sample_record:
            scene_record[key] = sample_record[key]
    return scene_record


def robot_type_score_for_record(record, grasp_type_id: int):
    """Read the availability score for one robot grasp type from a scene record.

    Args:
        record: Robot scene record that may contain ``pred_grasp_type_prob``.
        grasp_type_id: Real grasp type id. Type ids ``1..5`` map to score
            vector entries ``0..4``.

    Returns:
        The matching availability score as a float, or ``None`` when no
        compatible score is available.
    """
    if "pred_grasp_type_prob" not in record:
        return None
    scores = np.asarray(record["pred_grasp_type_prob"]).reshape(-1)
    score_index = int(grasp_type_id) - 1
    if score_index < 0 or score_index >= scores.shape[0]:
        return None
    return float(scores[score_index])


def format_robot_type_score_label(grasp_type_id: int, score, grasp_type_names=None):
    """Format a compact robot row label with grasp type and availability score.

    Args:
        grasp_type_id: Real grasp type id.
        score: Optional availability score for the requested grasp type.
        grasp_type_names: Optional ordered grasp type names including
            ``0_any``.

    Returns:
        Compact label text suitable for the Viser scene label.
    """
    if grasp_type_names is not None and 0 <= int(grasp_type_id) < len(grasp_type_names):
        type_text = grasp_type_names[int(grasp_type_id)]
    else:
        type_text = f"type_{int(grasp_type_id)}"
    if score is None:
        return f"{type_text} | score=n/a"
    return f"{type_text} | score={float(score):.3f}"


def format_robot_availability_markdown(score_vector, grasp_type_names=None):
    """Format robot scene availability scores for the Viser side panel.

    Args:
        score_vector: Availability scores aligned with real grasp types
            ``1..5``. Vectors that include ``0_any`` are also accepted.
        grasp_type_names: Optional ordered grasp type names including
            ``0_any``.

    Returns:
        Markdown text with one compact line per real grasp type, or an empty
        string when the score vector is unavailable or incompatible.
    """
    if score_vector is None:
        return ""
    scores = np.asarray(score_vector, dtype=np.float64).reshape(-1)
    if grasp_type_names is not None and scores.shape[0] == len(grasp_type_names):
        scores = scores[1:]
    real_type_num = (len(grasp_type_names) - 1) if grasp_type_names is not None else 5
    if scores.shape[0] < real_type_num:
        return ""

    lines = ["**Availability Score**"]
    for type_offset in range(real_type_num):
        grasp_type_id = type_offset + 1
        if grasp_type_names is not None and grasp_type_id < len(grasp_type_names):
            type_name = grasp_type_names[grasp_type_id]
        else:
            type_name = f"type_{grasp_type_id}"
        lines.append(f"- `{type_name}`: `{float(scores[type_offset]):.3f}`")
    return "\n".join(lines)


def find_robot_availability_score_vector(scene_records):
    """Find one predicted robot availability score vector from scene records.

    Args:
        scene_records: Built Viser scene records for one selected robot scene
            page.

    Returns:
        The first ``pred_grasp_type_prob`` vector found in the records, or
        ``None`` when no score vector is available.
    """
    for record in scene_records:
        if "pred_grasp_type_prob" in record:
            return record["pred_grasp_type_prob"]
    return None


def robot_mesh_link_prefixes_for_grasp_type(grasp_type_id, hide_inactive_single_hand_mesh: bool = True):
    """Choose robot mesh link prefixes for a predicted grasp type.

    Args:
        grasp_type_id: Robot grasp type id aligned with ``GRASP_TYPES``.
            Types ``1..3`` are right-hand-only and types ``4..5`` are
            bimanual.
        hide_inactive_single_hand_mesh: Whether to skip inactive left-hand
            mesh for right-hand-only grasp types.

    Returns:
        Tuple of link-name prefixes to export, or ``None`` to export all robot
        visual links.
    """
    if not hide_inactive_single_hand_mesh:
        return None
    try:
        grasp_type_id = int(grasp_type_id)
    except (TypeError, ValueError):
        return None
    if grasp_type_id in {1, 2, 3}:
        return ("rh_",)
    return None


def robot_mesh_filter_for_grasp_type(
    grasp_type_id,
    hide_inactive_single_hand_mesh: bool = True,
    visual_scope: str = "full",
    fingertip_link_names=None,
):
    """Choose robot visual mesh filters for one predicted grasp type.

    Args:
        grasp_type_id: Robot grasp type id aligned with ``GRASP_TYPES``.
        hide_inactive_single_hand_mesh: Whether right-hand-only grasp types
            should skip left-hand visual links.
        visual_scope: Mesh scope for Viser upload. ``full`` keeps the selected
            hand/full bimanual mesh, while ``fingertips`` keeps only LEAP
            fingertip visual links for the selected active hands.
        fingertip_link_names: Robot-specific fingertip visual link names
            loaded from the data/test_data config.

    Returns:
        Dictionary with ``include_link_prefixes`` and ``include_link_names``.
        Values may be ``None`` when no filtering should be applied.
    """
    visual_scope = str(visual_scope or "full").lower()
    include_link_prefixes = robot_mesh_link_prefixes_for_grasp_type(
        grasp_type_id,
        hide_inactive_single_hand_mesh=hide_inactive_single_hand_mesh,
    )
    include_link_names = None
    if visual_scope == "fingertips":
        if not fingertip_link_names:
            raise ValueError(
                "task.robot_mesh_visual_scope=fingertips requires "
                "test_data.robot_fingertip_link_names or data.robot_fingertip_link_names."
            )
        include_link_names = frozenset(str(link_name) for link_name in fingertip_link_names)
    elif visual_scope not in {"full", "hand"}:
        raise ValueError(
            f"Unsupported task.robot_mesh_visual_scope={visual_scope}. "
            "Expected one of ['full', 'fingertips']."
        )
    return {
        "include_link_prefixes": include_link_prefixes,
        "include_link_names": include_link_names,
    }


def get_robot_fingertip_link_names(config: DictConfig):
    """Read robot-specific fingertip visual link names from config.

    Args:
        config: Hydra config containing optional ``test_data`` and ``data``
            robot visualization fields.

    Returns:
        A tuple of fingertip link names. ``test_data`` takes priority because
        robot visualization is driven by sampled test-data records.
    """
    link_names = (
        OmegaConf.select(config, "test_data.robot_fingertip_link_names")
        or OmegaConf.select(config, "data.robot_fingertip_link_names")
        or ()
    )
    return tuple(str(link_name) for link_name in link_names)


def apply_one_scene_grasp_type_grid(scene_records, real_type_num: int = 5, grasp_type_names=None):
    """Arrange one-scene robot samples into grasp-type rows for Viser.

    Args:
        scene_records: Scene records built from samples in a single saved
            scene. Records may contain ``pred_grasp_type_id`` or
            ``grasp_type_id`` metadata.
        real_type_num: Number of real grasp types. The current robot model
            uses type ids ``1..5``; id ``0`` is an any-type request.
        grasp_type_names: Optional ordered grasp type names including
            ``0_any``. Used for compact row labels.

    Returns:
        The same scene record list after adding Viser grid row/column metadata.
    """
    if not scene_records:
        return scene_records

    column_counts = {row: 0 for row in range(real_type_num)}
    for record in scene_records:
        grasp_type_id = record.get("pred_grasp_type_id", record.get("grasp_type_id", 1))
        try:
            grasp_type_id = int(grasp_type_id)
            row = grasp_type_id - 1
        except (TypeError, ValueError):
            grasp_type_id = 1
            row = 0
        row = int(np.clip(row, 0, real_type_num - 1))
        col = column_counts[row]
        column_counts[row] += 1
        type_score = robot_type_score_for_record(record, grasp_type_id)
        record["viser_grid_row"] = row
        record["viser_grid_col"] = col
        record["viser_grid_rows"] = real_type_num
        record["label_caption"] = format_robot_type_score_label(grasp_type_id, type_score, grasp_type_names)
        if type_score is not None:
            record["pred_grasp_type_score"] = type_score

    grid_cols = max(1, max(column_counts.values()))
    for record in scene_records:
        record["viser_grid_cols"] = grid_cols
    return scene_records


def select_visualization_samples(output_dir: str, config: DictConfig, scene_path_resolver=None):
    mode = normalize_visualize_mode(get_task_value(config, "visualize_mode", "random_objects"))
    max_grasps = int(get_task_value(config, "max_grasps", 20))
    object_id = get_task_value(config, "object_id", None)
    target_grasp_type_id = get_task_value(config, "target_grasp_type_id", None)
    records = build_visualization_sample_index(output_dir, scene_path_resolver=scene_path_resolver)
    if not records:
        raise RuntimeError(f"No saved grasp files found in {output_dir}")

    return select_visualization_records(
        records,
        mode,
        max_grasps,
        object_id,
        target_grasp_type_id,
        is_our_human_grasp_format=is_our_human_grasp_format_test_data(config),
    )


def select_visualization_records(
    records,
    mode: str,
    max_grasps: int,
    object_id=None,
    target_grasp_type_id=None,
    is_our_human_grasp_format=False,
    batch_index=None,
):
    mode = normalize_visualize_mode(mode)

    if mode == "random_objects":
        grouped = {}
        for record in records:
            group_id = base_object_id_from_sequence(record["object_id"]) if is_our_human_grasp_format else record["object_id"]
            grouped.setdefault(group_id, []).append(record)
        object_ids = randomize_records(grouped.keys(), max_grasps)
        selected = []
        for selected_object_id in object_ids:
            selected.append(random.choice(grouped[selected_object_id]))

    elif mode == "one_scene":
        if object_id is None:
            raise ValueError("task.object_id must be set to a scene id when task.visualize_mode=one_scene.")
        selected = [record for record in records if object_id_matches(record["object_id"], object_id)]
        selected = select_one_scene_grasp_type_batch(selected, max_grasps, 0 if batch_index is None else batch_index)

    elif mode == "one_object":
        if object_id is None:
            raise ValueError("task.object_id must be set when task.visualize_mode=one_object.")
        selected = [record for record in records if object_id_matches(record["object_id"], object_id)]
        selected = sorted(selected, key=lambda record: record["sample_file"])
        selected = select_one_object_variant_batch(selected, max_grasps, 0 if batch_index is None else batch_index)

    elif mode == "one_object_multi_seq":
        if not is_our_human_grasp_format:
            raise RuntimeError(
                "task.visualize_mode=one_object_multi_seq is only valid when test_data.object_path points to "
                "OurHumanGraspFormat."
            )
        if object_id is None:
            raise ValueError("task.object_id must be set when task.visualize_mode=one_object_multi_seq.")
        selected = [record for record in records if sequence_object_id_matches(record["object_id"], object_id)]
        selected = select_random_across_sequences(selected, max_grasps)

    elif mode == "grasp_type":
        if target_grasp_type_id is None:
            raise ValueError("task.target_grasp_type_id must be set when task.visualize_mode=grasp_type.")
        target_grasp_type_id = int(target_grasp_type_id)
        populate_grasp_type_ids(records)
        selected = [record for record in records if record["grasp_type_id"] == target_grasp_type_id]
        selected = randomize_records(selected, max_grasps)

    else:
        raise ValueError(
            f"Unsupported visualize_mode={mode}. Expected one of {list(VISUALIZE_MODE_OPTIONS)}."
        )

    if not selected:
        example_object_ids = list_available_object_ids(records)[:10]
        example_text = ", ".join(example_object_ids) if example_object_ids else "none"
        raise RuntimeError(
            f"No samples matched visualize_mode={mode}, object_id={object_id}, "
            f"target_grasp_type_id={target_grasp_type_id}. Available object examples: {example_text}"
        )

    selected = annotate_scene_labels(selected, mode)
    print(f"[visualize] Selected {len(selected)} sample(s) with visualize_mode={mode}.")
    return selected


def compact_caption(caption: str):
    return caption.split(" | ", 1)[0]


def split_caption_after_viser_label(record):
    """Split a full caption into the compact viser label and content parts.

    Args:
        record: Scene record containing ``caption`` and optional
            ``viser_all_label`` metadata.

    Returns:
        A tuple ``(label_parts, content_parts)``. Label parts come from
        ``viser_all_label`` when present, so labels like ``0 | obj_1`` are not
        confused with regular caption separators.
    """
    caption = str(record.get("caption", ""))
    label = str(record.get("viser_all_label", ""))
    if label and caption.startswith(f"{label} | "):
        return label.split(" | "), caption[len(label) + 3 :].split(" | ")
    if label and caption == label:
        return label.split(" | "), []
    return [], [part.strip() for part in caption.split(" | ") if part.strip()]


def caption_aspects_for_record(record):
    """Extract named caption aspects from one scene record.

    Args:
        record: Scene record with a full caption.

    Returns:
        A dictionary mapping caption aspect ids to display strings.
    """
    label_parts, content_parts = split_caption_after_viser_label(record)
    aspects = {}
    if label_parts:
        aspects["scene_id"] = label_parts[0]
    if len(label_parts) >= 2:
        aspects["object_id"] = " | ".join(label_parts[1:])

    other_parts = []
    for part in content_parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith("Given:") or part.startswith("grasp_type_id="):
            aspects["given_grasp_type"] = part
        elif part.startswith("Pred:") or part.startswith("pred_grasp_type_id="):
            aspects["pred_grasp_type"] = part
        elif part.startswith("pred_grasp_type_prob="):
            aspects["pred_grasp_type_prob"] = part
        elif part.startswith("Pos:"):
            aspects["position_source"] = part
        elif part.startswith("PC:"):
            aspects["point_cloud"] = part
        elif part.startswith("err="):
            aspects["error"] = part
        elif "IK:" in part:
            aspects["ik_status"] = part
        elif "file" not in aspects:
            aspects["file"] = part
        else:
            other_parts.append(part)

    if other_parts:
        aspects["other"] = " | ".join(other_parts)
    return aspects


def build_caption_from_aspects(record, selected_aspects):
    """Build a scene caption from selected aspect ids.

    Args:
        record: Scene record with a full caption.
        selected_aspects: Iterable of aspect ids enabled in the web UI.

    Returns:
        A compact caption containing only the selected aspects. Returns an empty
        string when no selected aspect is available for the record.
    """
    aspects = caption_aspects_for_record(record)
    parts = [aspects[aspect_id] for aspect_id, _ in CAPTION_ASPECTS if aspect_id in selected_aspects and aspect_id in aspects]
    return " | ".join(parts)


def set_viser_dropdown_options(dropdown, options, initial_value=None):
    if hasattr(dropdown, "options"):
        try:
            dropdown.options = options
        except Exception as exc:
            print(f"[visualize] Could not update viser dropdown options: {exc}")
    if initial_value is not None:
        try:
            dropdown.value = initial_value
        except Exception as exc:
            print(f"[visualize] Could not update viser dropdown value: {exc}")


def list_available_object_ids(sample_records):
    return tuple(sorted({str(record["object_id"]) for record in sample_records if str(record["object_id"])}, key=natural_sort_key))


def list_available_base_object_ids(sample_records):
    return tuple(
        sorted(
            {base_object_id_from_sequence(record["object_id"]) for record in sample_records if str(record["object_id"])},
            key=natural_sort_key,
        )
    )


def list_available_grasp_type_ids(sample_records):
    return tuple(sorted({int(record["grasp_type_id"]) for record in sample_records if record["grasp_type_id"] is not None}))


def build_grasp_type_options(sample_records, grasp_type_names=None):
    """Build grasp type dropdown options without forcing sample payload reads.

    Args:
        sample_records: Visualization sample records, which may only contain
            path-derived metadata during fast startup.
        grasp_type_names: Optional ordered grasp type names from the dataset.

    Returns:
        A tuple of dropdown option strings. When dataset grasp names are known,
        all concrete grasp types are exposed immediately so startup does not
        need to read every saved sample.
    """
    if grasp_type_names is not None:
        return tuple(format_grasp_type_option(idx, grasp_type_names) for idx in range(1, len(grasp_type_names)))
    grasp_type_ids = list_available_grasp_type_ids(sample_records)
    return tuple(format_grasp_type_option(idx, grasp_type_names) for idx in grasp_type_ids) or ("",)


def populate_grasp_type_ids(sample_records):
    """Populate grasp type metadata for records indexed without full payloads.

    Args:
        sample_records: Visualization sample records created by
            ``build_visualization_sample_index``.

    Returns:
        The number of records whose grasp type id is available after the pass.
    """
    available_count = 0
    record_iter = progress_iter(sample_records, desc="Reading grasp type ids", total=len(sample_records))
    for record in record_iter:
        if record.get("grasp_type_id") is not None:
            available_count += 1
            continue
        data = record.get("data")
        if data is None:
            try:
                data = np.load(record["sample_file"], allow_pickle=True).item()
            except Exception as exc:
                print(f"[visualize] Could not read grasp type from {record['sample_file']}: {exc}")
                continue
        record["grasp_type_id"] = extract_grasp_type_id(data)
        if record["grasp_type_id"] is not None:
            available_count += 1
    return available_count


def build_selection_record_index(sample_records, is_our_human_grasp_format=False):
    """Build reusable record groups for responsive web UI selection.

    Args:
        sample_records: Visualization records indexed from saved sample files.
        is_our_human_grasp_format: Whether sequence-base object grouping is
            valid for ``one_object_multi_seq``.

    Returns:
        A dictionary containing sorted records grouped by exact object id, base
        object id, and random-object group id. Grasp-type groups are populated
        lazily because they require reading saved sample payloads.
    """
    records_by_object = {}
    records_by_base_object = {}
    records_by_random_group = {}
    records_by_scene = {}
    record_iter = progress_iter(sample_records, desc="Building visualization selection index", total=len(sample_records))
    for record in record_iter:
        object_id = canonical_object_id(record["object_id"])
        base_object_id = base_object_id_from_sequence(object_id)
        random_group_id = base_object_id if is_our_human_grasp_format else object_id
        records_by_object.setdefault(object_id, []).append(record)
        records_by_base_object.setdefault(base_object_id, []).append(record)
        records_by_random_group.setdefault(random_group_id, []).append(record)
        records_by_scene.setdefault(object_id, []).append(record)

    for grouped_records in (
        list(records_by_object.values())
        + list(records_by_base_object.values())
        + list(records_by_random_group.values())
        + list(records_by_scene.values())
    ):
        grouped_records.sort(key=lambda record: record["sample_file"])

    return {
        "records": sample_records,
        "records_by_object": records_by_object,
        "records_by_base_object": records_by_base_object,
        "records_by_random_group": records_by_random_group,
        "records_by_scene": records_by_scene,
        "random_group_ids": tuple(sorted(records_by_random_group.keys(), key=natural_sort_key)),
        "object_ids": tuple(sorted(records_by_object.keys(), key=natural_sort_key)),
        "base_object_ids": tuple(sorted(records_by_base_object.keys(), key=natural_sort_key)),
        "scene_ids": tuple(sorted(records_by_scene.keys(), key=natural_sort_key)),
        "grasp_type_records": {},
        "grasp_type_scan_complete": False,
    }


def get_indexed_scene_records(selection_index, scene_id):
    """Return records for one saved robot scene id.

    Args:
        selection_index: Selection index from ``build_selection_record_index``.
        scene_id: Scene id selected in the web UI or provided by config.

    Returns:
        A sorted list of records saved under the selected scene directory.
    """
    scene_id = canonical_object_id(scene_id)
    scene_records = selection_index["records_by_scene"].get(scene_id)
    if scene_records is not None:
        return scene_records
    return get_indexed_object_records(selection_index, scene_id)


def select_one_scene_grasp_type_batch(records, max_grasps_per_type: int, batch_index: int):
    """Select one page of robot one-scene records with per-type limits.

    Args:
        records: Saved sample records belonging to one scene.
        max_grasps_per_type: Maximum number of samples to show for each real
            grasp type on one page. Non-positive values keep all samples.
        batch_index: Zero-based page index. Pages wrap when ``batch_index``
            exceeds the number of available per-type pages.

    Returns:
        A list of selected records. Grasp type ids are loaded only for records
        in the selected scene, so startup indexing can remain payload-free.
    """
    records = sorted(list(records), key=lambda record: record["sample_file"])
    if max_grasps_per_type <= 0:
        return records

    populate_grasp_type_ids(records)
    grouped = {}
    for record in records:
        grasp_type_id = record.get("grasp_type_id")
        group_key = int(grasp_type_id) if grasp_type_id is not None else 0
        grouped.setdefault(group_key, []).append(record)
    if not grouped:
        return []

    page_count = max(int(np.ceil(len(group_records) / float(max_grasps_per_type))) for group_records in grouped.values())
    page_count = max(1, page_count)
    page_index = max(0, int(batch_index)) % page_count

    selected = []
    for grasp_type_id in sorted(grouped.keys()):
        group_records = grouped[grasp_type_id]
        start = page_index * max_grasps_per_type
        end = start + max_grasps_per_type
        selected.extend(group_records[start:end])

    for record in selected:
        record["one_scene_batch_index"] = page_index
        record["one_scene_batch_count"] = page_count
        record["one_scene_max_grasps_per_type"] = int(max_grasps_per_type)
    return selected


def get_indexed_object_records(selection_index, object_id):
    """Return records for an object id using exact lookup with scan fallback.

    Args:
        selection_index: Selection index from ``build_selection_record_index``.
        object_id: Object id selected in the web UI or provided by config.

    Returns:
        A sorted list of matching records.
    """
    object_id = canonical_object_id(object_id)
    exact_records = selection_index["records_by_object"].get(object_id)
    if exact_records is not None:
        return exact_records
    base_records = selection_index["records_by_base_object"].get(base_object_id_from_sequence(object_id))
    if base_records is not None:
        return base_records
    return sorted(
        [
            record
            for record in selection_index["records"]
            if object_id_matches(record["object_id"], object_id)
        ],
        key=lambda record: record["sample_file"],
    )


def select_grasp_type_records_lazy(selection_index, target_grasp_type_id: int, max_grasps: int, batch_index: int):
    """Randomly select grasp-type records while reading few payloads.

    Args:
        selection_index: Selection index from ``build_selection_record_index``.
        target_grasp_type_id: Integer grasp type id to match.
        max_grasps: Maximum records to return. Non-positive values require a
            complete scan and return all matches.
        batch_index: Zero-based batch index. This is accepted for compatibility
            with Selection panel state; selection is randomized.

    Returns:
        A list of records matching the requested grasp type.
    """
    del batch_index
    target_grasp_type_id = int(target_grasp_type_id)
    target_records = selection_index["grasp_type_records"].setdefault(target_grasp_type_id, [])
    if max_grasps <= 0:
        populate_grasp_type_ids(selection_index["records"])
        target_records[:] = [record for record in selection_index["records"] if record["grasp_type_id"] == target_grasp_type_id]
        selection_index["grasp_type_scan_complete"] = True
        return randomize_records(target_records, max_grasps)

    cached_paths = {record["sample_file"] for record in target_records}
    selected = randomize_records(target_records, max_grasps)
    selected_paths = {record["sample_file"] for record in selected}
    if len(selected) < max_grasps and not selection_index["grasp_type_scan_complete"]:
        shuffled_records = list(selection_index["records"])
        random.shuffle(shuffled_records)
        record_iter = progress_iter(
            shuffled_records,
            desc=f"Scanning grasp type {target_grasp_type_id}",
            total=len(shuffled_records),
        )
        for record in record_iter:
            if record.get("grasp_type_id") is None:
                data = record.get("data")
                if data is None:
                    try:
                        data = np.load(record["sample_file"], allow_pickle=True).item()
                    except Exception as exc:
                        print(f"[visualize] Could not read grasp type from {record['sample_file']}: {exc}")
                        continue
                record["grasp_type_id"] = extract_grasp_type_id(data)
            if record["grasp_type_id"] != target_grasp_type_id:
                continue
            if record["sample_file"] not in cached_paths:
                target_records.append(record)
                cached_paths.add(record["sample_file"])
            if record["sample_file"] not in selected_paths:
                selected.append(record)
                selected_paths.add(record["sample_file"])
            if len(selected) >= max_grasps:
                break
        else:
            selection_index["grasp_type_scan_complete"] = True

    if len(selected) < max_grasps and target_records:
        selected_paths = {record["sample_file"] for record in selected}
        remaining = [record for record in target_records if record["sample_file"] not in selected_paths]
        selected.extend(randomize_records(remaining, max_grasps - len(selected)))
    return selected


def select_visualization_records_from_index(
    selection_index,
    mode: str,
    max_grasps: int,
    object_id=None,
    target_grasp_type_id=None,
    is_our_human_grasp_format=False,
    batch_index=0,
):
    """Select visualization records from precomputed web UI indexes.

    Args:
        selection_index: Selection index from ``build_selection_record_index``.
        mode: Visualization mode.
        max_grasps: Maximum number of records in one batch.
        object_id: Optional selected object id.
        target_grasp_type_id: Optional selected grasp type id.
        is_our_human_grasp_format: Whether ``one_object_multi_seq`` is valid.
        batch_index: Zero-based batch index.

    Returns:
        Selected and annotated sample records.
    """
    mode = normalize_visualize_mode(mode)
    batch_index = max(0, int(batch_index))
    if mode == "random_objects":
        group_ids = randomize_records(selection_index["random_group_ids"], max_grasps)
        selected = []
        for group_id in group_ids:
            group_records = selection_index["records_by_random_group"][group_id]
            selected.append(random.choice(group_records))
    elif mode == "one_scene":
        if object_id is None:
            raise ValueError("task.object_id must be set to a scene id when task.visualize_mode=one_scene.")
        selected = select_one_scene_grasp_type_batch(
            get_indexed_scene_records(selection_index, object_id),
            max_grasps_per_type=max(0, int(max_grasps)),
            batch_index=batch_index,
        )
    elif mode == "one_object":
        if object_id is None:
            raise ValueError("task.object_id must be set when task.visualize_mode=one_object.")
        selected = select_one_object_variant_batch(
            get_indexed_object_records(selection_index, object_id),
            max_grasps,
            batch_index,
        )
    elif mode == "one_object_multi_seq":
        if not is_our_human_grasp_format:
            raise RuntimeError(
                "task.visualize_mode=one_object_multi_seq is only valid when test_data.object_path points to "
                "OurHumanGraspFormat."
            )
        if object_id is None:
            raise ValueError("task.object_id must be set when task.visualize_mode=one_object_multi_seq.")
        base_object_id = base_object_id_from_sequence(object_id)
        selected = select_random_across_sequences(selection_index["records_by_base_object"].get(base_object_id, []), max_grasps)
    elif mode == "grasp_type":
        if target_grasp_type_id is None:
            raise ValueError("task.target_grasp_type_id must be set when task.visualize_mode=grasp_type.")
        selected = select_grasp_type_records_lazy(selection_index, int(target_grasp_type_id), max_grasps, batch_index)
    else:
        raise ValueError(f"Unsupported visualize_mode={mode}. Expected one of {list(VISUALIZE_MODE_OPTIONS)}.")

    if not selected:
        example_object_ids = selection_index["object_ids"][:10]
        example_text = ", ".join(example_object_ids) if example_object_ids else "none"
        raise RuntimeError(
            f"No samples matched visualize_mode={mode}, object_id={object_id}, "
            f"target_grasp_type_id={target_grasp_type_id}. Available object examples: {example_text}"
        )

    selected = annotate_scene_labels(selected, mode)
    print(f"[visualize] Selected {len(selected)} sample(s) with visualize_mode={mode}.")
    return selected


def format_grasp_type_option(grasp_type_id: int, grasp_type_names=None):
    if grasp_type_names is not None and 0 <= grasp_type_id < len(grasp_type_names):
        return f"{grasp_type_id}: {grasp_type_names[grasp_type_id]}"
    return str(grasp_type_id)


def parse_grasp_type_option(option: str):
    return int(str(option).split(":", 1)[0])


def pick_initial_option(options, preferred):
    if not options:
        return None
    if preferred is not None:
        preferred = str(preferred)
        for option in options:
            if str(option) == preferred or str(option).startswith(f"{preferred}:"):
                return option
    return options[0]


def pick_initial_object_option(options, preferred):
    if not options:
        return None
    if preferred is not None:
        for option in options:
            if object_id_matches(option, preferred) or sequence_object_id_matches(option, preferred):
                return option
    return options[0]


def format_gui_wrappable_value(value):
    """Format a long GUI value with safe wrap points.

    Args:
        value: Value to display in a markdown GUI panel.

    Returns:
        HTML-escaped text with zero-width break points inserted after common
        object-id separators so long DGN object names can wrap inside the side
        panel while preserving the visible object id.
    """
    text = str(value)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for separator in ("/", "_", "-"):
        text = text.replace(separator, f"{separator}\u200b")
    text = re.sub(r"([A-Za-z0-9]{12})", lambda match: f"{match.group(1)}\u200b", text)
    return text


def build_viser_selection_controls(
    sample_records,
    config,
    build_scene_records,
    grasp_type_names=None,
    is_our_human_grasp_format=False,
):
    selection_index = build_selection_record_index(sample_records, is_our_human_grasp_format=is_our_human_grasp_format)
    mode_options = VISUALIZE_MODE_OPTIONS if is_our_human_grasp_format else ROBOT_VISER_MODE_OPTIONS
    object_options = selection_index["object_ids"] or ("",)
    base_object_options = selection_index["base_object_ids"] or object_options
    scene_options = selection_index["scene_ids"] or object_options
    grasp_type_options = build_grasp_type_options(sample_records, grasp_type_names)
    initial_mode = normalize_visualize_mode(get_task_value(config, "visualize_mode", "random_objects"))
    if not is_our_human_grasp_format and initial_mode == "one_object":
        initial_mode = "one_scene"
    if initial_mode not in mode_options:
        initial_mode = "random_objects"
    initial_object_options = scene_options if initial_mode == "one_scene" else object_options
    if initial_mode in {"one_object", "one_object_multi_seq"}:
        initial_object_options = base_object_options
    initial_object = pick_initial_object_option(initial_object_options, get_task_value(config, "object_id", None))
    initial_grasp_type = pick_initial_option(grasp_type_options, get_task_value(config, "target_grasp_type_id", None))
    max_grasps = int(get_task_value(config, "max_grasps", 20))
    one_scene_max_grasps_per_type = int(get_task_value(config, "one_scene_max_grasps_per_type", 5))
    batch_state = {"key": None, "index": 0}

    def random_scene_id(scene_id):
        """Return a random scene id different from the current scene when possible.

        Args:
            scene_id: Current scene id shown in robot ``one_scene`` mode.

        Returns:
            Scene id selected from the indexed saved scenes. When only one
            scene exists, the current scene is returned.
        """
        scene_ids = selection_index["scene_ids"]
        if not scene_ids:
            return scene_id
        scene_id = canonical_object_id(scene_id)
        candidates = [item for item in scene_ids if canonical_object_id(item) != scene_id]
        return random.choice(candidates or scene_ids)

    def object_options_for_mode(mode):
        """Return the relevant Object/Scene dropdown options for one mode.

        Args:
            mode: Visualization mode selected in the Viser UI.

        Returns:
            Tuple of dropdown options. Robot ``one_scene`` exposes scene ids;
            legacy object modes expose base object ids.
        """
        mode = normalize_visualize_mode(mode)
        if mode == "one_scene":
            return scene_options
        if mode in {"one_object", "one_object_multi_seq"}:
            return base_object_options
        return object_options

    def load_scene_records(mode, object_id, grasp_type_option, advance_batch=False):
        mode = normalize_visualize_mode(mode)
        target_grasp_type_id = parse_grasp_type_option(grasp_type_option) if grasp_type_option else None
        selection_key = (mode, str(object_id), target_grasp_type_id)
        if selection_key != batch_state["key"]:
            batch_state["key"] = selection_key
            batch_state["index"] = 0
        elif advance_batch:
            batch_state["index"] += 1
        elif not advance_batch:
            batch_state["index"] = 0
        selected_records = select_visualization_records_from_index(
            selection_index,
            mode,
            one_scene_max_grasps_per_type if mode == "one_scene" else max_grasps,
            object_id=object_id,
            target_grasp_type_id=target_grasp_type_id,
            is_our_human_grasp_format=is_our_human_grasp_format,
            batch_index=batch_state["index"],
        )
        if mode == "one_scene":
            batch_state["current_object"] = str(object_id)
        scene_records = build_scene_records(selected_records)
        if mode == "one_scene":
            score_vector = find_robot_availability_score_vector(scene_records)
            batch_state["selection_info"] = format_robot_availability_markdown(score_vector, grasp_type_names)
        else:
            batch_state["selection_info"] = ""
        return scene_records

    def load_action_scene_records(mode, object_id, grasp_type_option, action_name: str):
        """Load scene records for custom robot Selection-panel actions.

        Args:
            mode: Current visualization mode.
            object_id: Current object or scene id selected in the UI.
            grasp_type_option: Current grasp-type dropdown option.
            action_name: Action requested by the UI, currently ``Next Scene``.

        Returns:
            Scene records produced by the selected action.
        """
        mode = normalize_visualize_mode(mode)
        if mode != "one_scene":
            return load_scene_records(mode, object_id, grasp_type_option, advance_batch=True)
        action_name = str(action_name).strip().lower()
        if action_name != "next scene":
            raise ValueError(f"Unsupported selection action: {action_name}")
        next_scene_id = random_scene_id(object_id)
        batch_state["key"] = None
        batch_state["index"] = 0
        batch_state["current_object"] = str(next_scene_id)
        return load_scene_records(mode, next_scene_id, grasp_type_option, advance_batch=False)

    return {
        "mode_options": mode_options,
        "initial_mode": initial_mode,
        "object_options": object_options,
        "base_object_options": base_object_options,
        "scene_options": scene_options,
        "initial_object": initial_object,
        "grasp_type_options": grasp_type_options,
        "initial_grasp_type": initial_grasp_type,
        "load_scene_records": load_scene_records,
        "load_action_scene_records": load_action_scene_records,
        "batch_state": batch_state,
        "object_label": "Scene ID",
        "next_button_label": "Next Batch",
        "extra_action_button_label": "Next Scene",
        "object_options_for_mode": object_options_for_mode,
        "disable_object_for_mode": lambda mode: normalize_visualize_mode(mode) not in {"one_scene", "one_object", "one_object_multi_seq"},
        "disable_grasp_type_for_mode": lambda mode: normalize_visualize_mode(mode) == "one_scene",
        "disable_next_batch_for_mode": lambda mode: normalize_visualize_mode(mode) != "one_scene",
        "disable_extra_action_for_mode": lambda mode: normalize_visualize_mode(mode) != "one_scene",
    }


def build_scene_grid_offsets(num_scenes: int, spacing: float):
    if num_scenes <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    cols = int(np.ceil(np.sqrt(num_scenes)))
    rows = int(np.ceil(num_scenes / cols))
    offsets = []
    for idx in range(num_scenes):
        row = idx // cols
        col = idx % cols
        offsets.append(
            [
                (col - (cols - 1) / 2.0) * spacing,
                ((rows - 1) / 2.0 - row) * spacing,
                0.0,
            ]
        )
    return np.asarray(offsets, dtype=np.float32)


def build_grouped_scene_offsets(scene_records, spacing: float):
    if scene_records and all(
        "viser_grid_row" in record and "viser_grid_col" in record
        for record in scene_records
    ):
        total_rows = max(int(record.get("viser_grid_rows", 0)) for record in scene_records)
        total_cols = max(int(record.get("viser_grid_cols", 0)) for record in scene_records)
        total_rows = max(total_rows, max(int(record["viser_grid_row"]) for record in scene_records) + 1)
        total_cols = max(total_cols, max(int(record["viser_grid_col"]) for record in scene_records) + 1)
        offsets = []
        for record in scene_records:
            row = int(record["viser_grid_row"])
            col = int(record["viser_grid_col"])
            offsets.append(
                [
                    (col - (total_cols - 1) / 2.0) * spacing,
                    ((total_rows - 1) / 2.0 - row) * spacing,
                    0.0,
                ]
            )
        return np.asarray(offsets, dtype=np.float32)

    group_to_indices = {}
    for idx, record in enumerate(scene_records):
        group_id = record.get("viser_spatial_group")
        if group_id:
            group_to_indices.setdefault(str(group_id), []).append(idx)

    if len(group_to_indices) <= 1:
        return build_scene_grid_offsets(len(scene_records), spacing)

    offsets = np.zeros((len(scene_records), 3), dtype=np.float32)
    group_ids = sorted(group_to_indices.keys(), key=natural_sort_key)
    cursor_x = 0.0

    for group_idx, group_id in enumerate(group_ids):
        indices = group_to_indices[group_id]
        local_offsets = build_scene_grid_offsets(len(indices), spacing)
        local_min_x = float(local_offsets[:, 0].min())
        local_max_x = float(local_offsets[:, 0].max())
        if group_idx == 0:
            group_origin_x = -local_min_x
        else:
            group_origin_x = cursor_x - local_min_x + 1.5 * spacing
        for local_idx, record_idx in enumerate(indices):
            offsets[record_idx] = local_offsets[local_idx] + np.asarray([group_origin_x, 0.0, 0.0], dtype=np.float32)
        cursor_x = group_origin_x + local_max_x

    return offsets - offsets.mean(axis=0, keepdims=True)


def transform_points(points, offset):
    return np.asarray(points, dtype=np.float32) + np.asarray(offset, dtype=np.float32).reshape(1, 3)


def transform_trimesh_geometry(geometry, offset):
    geometry = geometry.copy()
    geometry.apply_translation(np.asarray(offset, dtype=np.float32))
    return geometry


def add_scene_elements_to_viser(server, scene_elements, scene_name: str, offset):
    handles = []
    for idx, geometry in enumerate(scene_elements):
        name = f"{scene_name}/geometry_{idx}"
        if isinstance(geometry, trimesh.points.PointCloud):
            points = transform_points(geometry.vertices, offset)
            colors = getattr(geometry.visual, "vertex_colors", None)
            if colors is None or len(colors) == 0:
                colors = np.tile(np.asarray([[255, 0, 0]], dtype=np.uint8), (points.shape[0], 1))
            else:
                colors = np.asarray(colors)[:, :3].astype(np.uint8)
            handles.append(server.scene.add_point_cloud(name, points=points, colors=colors, point_size=0.003))
            continue

        if isinstance(geometry, trimesh.Trimesh):
            mesh = transform_trimesh_geometry(geometry, offset)
            handles.append(server.scene.add_mesh_trimesh(name, mesh=mesh))
            continue

        if isinstance(geometry, trimesh.Scene):
            for sub_idx, sub_geometry in enumerate(geometry.geometry.values()):
                sub_name = f"{name}/sub_geometry_{sub_idx}"
                if isinstance(sub_geometry, trimesh.Trimesh):
                    mesh = transform_trimesh_geometry(sub_geometry, offset)
                    handles.append(server.scene.add_mesh_trimesh(sub_name, mesh=mesh))
    return handles


def add_scene_label_to_viser(
    server,
    scene_name: str,
    caption: str,
    offset,
    log_prefix: str,
    label_font_size_mode: str = "screen",
    label_font_screen_scale: float = 0.8,
    label_height_offset: float = 0.35,
):
    """Add one scene label to Viser with optional font controls.

    Args:
        server: Active Viser server.
        scene_name: Unique scene node name prefix.
        caption: Label text to show.
        offset: Scene translation offset applied in the grid view.
        log_prefix: Prefix used by fallback console logging.
        label_font_size_mode: Viser label font sizing mode.
        label_font_screen_scale: Screen-space font scale passed to Viser.
        label_height_offset: World-space z offset added above each scene.

    Returns:
        The created Viser handle when supported, otherwise ``None``.
    """
    label_position = np.asarray(offset, dtype=np.float32) + np.asarray(
        [0.0, 0.0, float(label_height_offset)],
        dtype=np.float32,
    )
    if hasattr(server.scene, "add_label"):
        try:
            return server.scene.add_label(
                f"{scene_name}/label",
                text=caption,
                position=label_position,
                font_size_mode=str(label_font_size_mode),
                font_screen_scale=float(label_font_screen_scale),
            )
        except TypeError:
            try:
                return server.scene.add_label(f"{scene_name}/label", text=caption, position=label_position)
            except TypeError:
                print(f"[{log_prefix}] {scene_name}: {caption}")
    else:
        print(f"[{log_prefix}] {scene_name}: {caption}")
    return None


def remove_viser_handles(handles):
    for handle in handles:
        if handle is not None and hasattr(handle, "remove"):
            handle.remove()


def show_scenes_with_trimesh(scene_records):
    for record in scene_records:
        scene = trimesh.Scene(record["elements"])
        camera_distance = record.get("camera_distance")
        if camera_distance is not None:
            scene.set_camera(
                angles=(np.deg2rad(60.0), 0.0, np.deg2rad(45.0)),
                distance=camera_distance,
                center=scene.centroid,
            )
        scene.show(caption=record["caption"], smooth=bool(record.get("smooth", True)))


def add_caption_gui(
    server,
    scene_records,
    log_prefix: str,
    on_show_full_caption=None,
    on_hide_full_caption=None,
    on_toggle_caption_aspect=None,
    caption_aspects=None,
):
    if not hasattr(server, "gui") or not scene_records:
        return None

    def build_caption_state(records):
        captions = [record["caption"] for record in records]
        sample_names = [
            record.get("viser_all_label", compact_caption(caption))
            for record, caption in zip(records, captions)
        ]
        options = tuple(f"{idx:04d}: {name}" for idx, name in enumerate(sample_names))
        return captions, options

    captions, options = build_caption_state(scene_records)
    caption_state = {"captions": captions, "options": options, "visible": False}

    with server.gui.add_folder("Captions"):
        caption_dropdown = server.gui.add_dropdown("Sample", options=options, initial_value=options[0])
        show_button = server.gui.add_button("Show Full Caption")
        aspect_buttons = {}
        for aspect_id, aspect_name in CAPTION_ASPECTS:
            if caption_aspects is not None and aspect_id not in caption_aspects:
                continue
            aspect_buttons[aspect_id] = server.gui.add_button(aspect_name)
        if hasattr(server.gui, "add_markdown"):
            caption_handle = server.gui.add_markdown("Caption hidden.")
            caption_mode = "markdown"
        elif hasattr(server.gui, "add_text"):
            caption_handle = server.gui.add_text("Full caption", initial_value="Caption hidden.")
            caption_mode = "text"
        else:
            caption_handle = None
            caption_mode = "print"

        def show_caption(index: int):
            text = caption_state["captions"][index]
            if caption_mode == "markdown" and hasattr(caption_handle, "content"):
                caption_handle.content = f"```text\n{text}\n```"
                caption_state["visible"] = True
            elif caption_mode == "text" and hasattr(caption_handle, "value"):
                caption_handle.value = text
                caption_state["visible"] = True
            else:
                print(f"[{log_prefix}] {caption_state['options'][index]}: {text}")
                caption_state["visible"] = True

        def hide_caption():
            if caption_mode == "markdown" and hasattr(caption_handle, "content"):
                caption_handle.content = "Caption hidden."
            elif caption_mode == "text" and hasattr(caption_handle, "value"):
                caption_handle.value = "Caption hidden."
            caption_state["visible"] = False

        @show_button.on_click
        def _on_show_caption(event):
            del event
            selected_idx = caption_state["options"].index(caption_dropdown.value)
            if caption_state["visible"]:
                hide_caption()
                if on_hide_full_caption is not None:
                    on_hide_full_caption()
                return
            show_caption(selected_idx)
            if on_show_full_caption is not None:
                on_show_full_caption(selected_idx)

        for aspect_id, aspect_button in aspect_buttons.items():
            @aspect_button.on_click
            def _on_toggle_aspect(event, aspect_id=aspect_id):
                del event
                if on_toggle_caption_aspect is not None:
                    on_toggle_caption_aspect(aspect_id)

        @caption_dropdown.on_update
        def _on_caption_select(event):
            del event
            selected_idx = caption_state["options"].index(caption_dropdown.value)
            if caption_state["visible"]:
                show_caption(selected_idx)

        def update_records(records, selected_idx: int):
            captions, options = build_caption_state(records)
            caption_state["captions"] = captions
            caption_state["options"] = options
            selected_idx = int(np.clip(selected_idx, 0, len(options) - 1))
            set_viser_dropdown_options(caption_dropdown, options, options[selected_idx])
            hide_caption()

    return {
        "dropdown": caption_dropdown,
        "show_caption": show_caption,
        "hide_caption": hide_caption,
        "update_records": update_records,
        "aspect_buttons": aspect_buttons,
    }


def show_scenes_with_viser(
    scene_records,
    port: int,
    scene_spacing: float,
    display_mode: str = "all",
    scene_id: int = 0,
    log_prefix: str = "visualize",
    selection_controls=None,
    next_batch_loader=None,
    label_font_size_mode: str = "screen",
    label_font_screen_scale: float = 0.8,
    label_height_offset: float = 0.35,
    caption_aspects=None,
    gui_description: str = "",
    frame_scenes: bool = False,
    human_scores_dir=None,
):
    if not VISER_AVAILABLE:
        raise ImportError(
            "viser is not installed. Install it in the active environment before using task.visualizer=viser."
        )
    if not scene_records:
        raise RuntimeError("No scene records to visualize.")

    display_mode = display_mode if display_mode in {"all", "single"} else "all"
    scene_id = int(np.clip(scene_id, 0, len(scene_records) - 1))

    server = viser.ViserServer(port=port)
    if hasattr(server.gui, "configure_theme"):
        server.gui.configure_theme(control_layout="floating", control_width="large")
    server.scene.set_up_direction("+z")
    if gui_description and hasattr(server.gui, "add_markdown"):
        server.gui.add_markdown(gui_description)
    scene_handles = {"value": []}

    records_state = {"records": scene_records}
    update_human_scores = (
        add_human_scores_gui(server, scene_records, human_scores_dir)
        if human_scores_dir is not None else None
    )

    def frame_current_scenes(all_records=False):
        records = records_state["records"]
        offsets = build_grouped_scene_offsets(records, scene_spacing)
        if state["display_mode"] == "single" and not all_records:
            records = [records[state["scene_id"]]]
            offsets = [np.zeros(3, dtype=np.float32)]
        bounds = []
        for record, offset in zip(records, offsets):
            for geometry in record["elements"]:
                if not hasattr(geometry, "bounds") or geometry.bounds is None:
                    continue
                bounds.append(np.asarray(geometry.bounds, dtype=float) + offset)
        if not bounds:
            return
        bounds = np.asarray(bounds)
        lower, upper = bounds[:, 0, :].min(axis=0), bounds[:, 1, :].max(axis=0)
        center = (lower + upper) / 2
        radius = max(0.5, float(np.linalg.norm(upper - lower)) * 0.85)
        position = center + np.array([radius, -radius, radius * 0.65])
        server.initial_camera.look_at = center
        server.initial_camera.position = position
        for client in server.get_clients().values():
            client.camera.look_at = center
            client.camera.position = position

    if frame_scenes:
        with server.gui.add_folder("Camera"):
            frame_button = server.gui.add_button("Frame All")

            @frame_button.on_click
            def _on_frame_all(event):
                del event
                frame_current_scenes(all_records=True)

    def build_sample_options(records):
        return tuple(
            f"{idx:04d}: {record.get('viser_all_label', compact_caption(record['caption']))}"
            for idx, record in enumerate(records)
        )

    sample_options_state = {"options": build_sample_options(records_state["records"])}
    state = {
        "display_mode": display_mode,
        "scene_id": scene_id,
        "show_full_scene_captions": False,
        "caption_aspects": set(),
    }

    with server.gui.add_folder("Display"):
        display_dropdown = server.gui.add_dropdown(
            "View",
            options=("all", "single"),
            initial_value=state["display_mode"],
        )
        scene_dropdown = server.gui.add_dropdown(
            "Scene",
            options=sample_options_state["options"],
            initial_value=sample_options_state["options"][state["scene_id"]],
        )

    def show_full_scene_captions(selected_idx: int):
        state["scene_id"] = int(np.clip(selected_idx, 0, len(records_state["records"]) - 1))
        state["show_full_scene_captions"] = True
        render_current_view()

    def hide_full_scene_captions():
        state["show_full_scene_captions"] = False
        render_current_view()

    def toggle_caption_aspect(aspect_id: str):
        if aspect_id in state["caption_aspects"]:
            state["caption_aspects"].remove(aspect_id)
        else:
            state["caption_aspects"].add(aspect_id)
        state["show_full_scene_captions"] = False
        render_current_view()

    caption_gui = add_caption_gui(
        server,
        scene_records,
        log_prefix,
        on_show_full_caption=show_full_scene_captions,
        on_hide_full_caption=hide_full_scene_captions,
        on_toggle_caption_aspect=toggle_caption_aspect,
        caption_aspects=caption_aspects,
    )

    def render_current_view():
        current_records = records_state["records"]
        remove_viser_handles(scene_handles["value"])
        scene_handles["value"] = []
        if state["display_mode"] == "single":
            records_and_offsets = [(state["scene_id"], current_records[state["scene_id"]], np.zeros(3, dtype=np.float32))]
        else:
            offsets = build_grouped_scene_offsets(current_records, scene_spacing)
            records_and_offsets = list(zip(range(len(current_records)), current_records, offsets))

        for idx, record, offset in records_and_offsets:
            scene_name = f"/sample_{idx:04d}"
            scene_handles["value"].extend(add_scene_elements_to_viser(server, record["elements"], scene_name, offset))
            if state["show_full_scene_captions"]:
                label_text = record["caption"]
            elif state["caption_aspects"]:
                label_text = build_caption_from_aspects(record, state["caption_aspects"])
            else:
                label_text = str(record.get("label_caption", ""))
            label = (
                add_scene_label_to_viser(
                    server,
                    scene_name,
                    label_text,
                    offset,
                    log_prefix,
                    label_font_size_mode=label_font_size_mode,
                    label_font_screen_scale=label_font_screen_scale,
                    label_height_offset=label_height_offset,
                )
                if label_text
                else None
            )
            if label is not None:
                scene_handles["value"].append(label)

        if hasattr(scene_dropdown, "disabled"):
            scene_dropdown.disabled = state["display_mode"] == "all"
        if caption_gui is not None:
            caption_gui["dropdown"].value = sample_options_state["options"][state["scene_id"]]
            if state["display_mode"] == "single":
                caption_gui["show_caption"](state["scene_id"])
            elif not state["show_full_scene_captions"]:
                caption_gui["hide_caption"]()

        if state["display_mode"] == "single":
            print(f"[{log_prefix}] Showing scene {state['scene_id']}: {compact_caption(current_records[state['scene_id']]['caption'])}")
        else:
            print(f"[{log_prefix}] Showing all {len(current_records)} scenes.")

    def update_scene_records(new_scene_records):
        if not new_scene_records:
            print(f"[{log_prefix}] Selection produced no scenes; keeping the current view.")
            return
        records_state["records"] = new_scene_records
        if update_human_scores is not None:
            update_human_scores(new_scene_records)
        state["scene_id"] = int(np.clip(state["scene_id"], 0, len(new_scene_records) - 1))
        state["show_full_scene_captions"] = False
        state["caption_aspects"].clear()
        sample_options_state["options"] = build_sample_options(new_scene_records)
        set_viser_dropdown_options(
            scene_dropdown,
            sample_options_state["options"],
            sample_options_state["options"][state["scene_id"]],
        )
        if caption_gui is not None:
            caption_gui["update_records"](new_scene_records, state["scene_id"])
        render_current_view()
        if frame_scenes:
            frame_current_scenes()

    if selection_controls is not None and hasattr(server, "gui"):
        split_dropdown = None

        def current_split_name():
            """Return the selected dataset split when split filtering is enabled.

            Args:
                None.

            Returns:
                The current split dropdown value, or ``None`` when this
                visualization mode does not expose split filtering.
            """
            return str(split_dropdown.value) if split_dropdown is not None else None

        def object_options_for_mode(mode):
            mode = normalize_visualize_mode(mode)
            if mode in {"one_object", "one_object_multi_seq"}:
                return selection_controls["base_object_options"]
            return selection_controls["object_options"]

        def selection_object_options(mode):
            """Read object dropdown options for the current selection context.

            Args:
                mode: Current visualize mode.

            Returns:
                Object id options supplied by ``selection_controls``. Human
                visualization may additionally scope these options by split.
            """
            if "object_options_for_mode" not in selection_controls:
                return object_options_for_mode(mode)
            if split_dropdown is not None:
                return selection_controls["object_options_for_mode"](mode, current_split_name())
            return selection_controls["object_options_for_mode"](mode)

        def load_selection_scene_records(advance_batch=False):
            """Load scene records from the current GUI selection state.

            Args:
                advance_batch: Whether to advance the selection batch.

            Returns:
                Scene records produced by the configured selection loader.
            """
            kwargs = {"advance_batch": advance_batch}
            if split_dropdown is not None:
                kwargs["split_name"] = current_split_name()
            return selection_controls["load_scene_records"](
                str(mode_dropdown.value),
                str(object_dropdown.value),
                str(grasp_type_dropdown.value),
                **kwargs,
            )

        def load_selection_action_records(action_name: str):
            """Load scene records for a custom Selection-panel action.

            Args:
                action_name: Name of the action requested by the caller.

            Returns:
                Scene records produced by the configured action loader.
            """
            kwargs = {"action_name": str(action_name)}
            if split_dropdown is not None:
                kwargs["split_name"] = current_split_name()
            return selection_controls["load_action_scene_records"](
                str(mode_dropdown.value),
                str(object_dropdown.value),
                str(grasp_type_dropdown.value),
                **kwargs,
            )

        with server.gui.add_folder("Selection"):
            object_label = str(selection_controls.get("object_label", "Object"))
            next_button_label = str(selection_controls.get("next_button_label", "Next Batch"))
            if "split_options" in selection_controls:
                split_dropdown = server.gui.add_dropdown(
                    "Split",
                    options=selection_controls["split_options"],
                    initial_value=selection_controls["initial_split"],
                )
            mode_dropdown = server.gui.add_dropdown(
                "Mode",
                options=selection_controls["mode_options"],
                initial_value=selection_controls["initial_mode"],
            )
            initial_object_options = selection_object_options(selection_controls["initial_mode"])
            object_dropdown = server.gui.add_dropdown(
                object_label,
                options=initial_object_options,
                initial_value=selection_controls["initial_object"],
            )
            object_info_handle = (
                server.gui.add_markdown("")
                if hasattr(server.gui, "add_markdown")
                else None
            )
            grasp_type_dropdown = server.gui.add_dropdown(
                "Grasp Type",
                options=selection_controls["grasp_type_options"],
                initial_value=selection_controls["initial_grasp_type"],
            )
            apply_button = server.gui.add_button("Apply Selection")
            next_batch_button = server.gui.add_button(next_button_label)
            extra_action_button = None
            extra_action_label = str(selection_controls.get("extra_action_button_label", ""))
            if extra_action_label and "load_action_scene_records" in selection_controls:
                extra_action_button = server.gui.add_button(extra_action_label)
            initial_status = selection_controls.get("batch_state", {}).get("selection_info", "Selection loaded.")
            status_handle = server.gui.add_markdown(initial_status) if hasattr(server.gui, "add_markdown") else None

            def update_object_info():
                if object_info_handle is None or not hasattr(object_info_handle, "content"):
                    return
                batch_state = selection_controls.get("batch_state", {})
                selection_info = str(batch_state.get("selection_info", "") or "")
                content = (
                    f"Selected {object_label}: "
                    f"{format_gui_wrappable_value(object_dropdown.value)}"
                )
                if selection_info:
                    content = f"{content}\n\n{selection_info}"
                object_info_handle.content = content

            def sync_current_object_selection():
                """Reflect a loader-updated object/scene id in the Selection UI.

                Args:
                    None.

                Returns:
                    None. The object dropdown and markdown panel are updated
                    when the selection loader publishes ``current_object``.
                """
                batch_state = selection_controls.get("batch_state", {})
                current_object = batch_state.get("current_object")
                if current_object is None:
                    update_object_info()
                    return
                current_object = str(current_object)
                options = selection_object_options(str(mode_dropdown.value))
                if current_object not in [str(option) for option in options]:
                    options = tuple(list(options) + [current_object])
                    set_viser_dropdown_options(object_dropdown, options)
                try:
                    object_dropdown.value = current_object
                except Exception as exc:
                    print(f"[{log_prefix}] Could not update selected {object_label}: {exc}")
                update_object_info()

            def refresh_selection_widgets():
                if split_dropdown is not None and "mode_options_for_split" in selection_controls:
                    options = selection_controls["mode_options_for_split"](current_split_name())
                    current_mode = str(mode_dropdown.value)
                    set_viser_dropdown_options(mode_dropdown, options, pick_initial_option(options, current_mode))
                if split_dropdown is not None and "grasp_type_options_for_split" in selection_controls:
                    options = selection_controls["grasp_type_options_for_split"](current_split_name())
                    current_type = str(grasp_type_dropdown.value)
                    set_viser_dropdown_options(grasp_type_dropdown, options, pick_initial_option(options, current_type))
                mode = normalize_visualize_mode(str(mode_dropdown.value))
                object_options = selection_object_options(mode)
                preferred_object = object_dropdown.value if hasattr(object_dropdown, "value") else None
                initial_object = pick_initial_object_option(object_options, preferred_object)
                set_viser_dropdown_options(object_dropdown, object_options, initial_object)

                if hasattr(object_dropdown, "disabled"):
                    object_dropdown.disabled = bool(
                        selection_controls.get("disable_object_for_mode", lambda current_mode: False)(mode)
                    )
                if hasattr(grasp_type_dropdown, "disabled"):
                    grasp_type_dropdown.disabled = bool(
                        selection_controls.get("disable_grasp_type_for_mode", lambda current_mode: False)(mode)
                    )
                if hasattr(next_batch_button, "disabled"):
                    next_batch_button.disabled = bool(
                        selection_controls.get("disable_next_batch_for_mode", lambda current_mode: False)(mode)
                    )
                if extra_action_button is not None and hasattr(extra_action_button, "disabled"):
                    extra_action_button.disabled = bool(
                        selection_controls.get("disable_extra_action_for_mode", lambda current_mode: False)(mode)
                    )
                update_object_info()

            def mark_pending_selection():
                if "mode_options_for_split" not in selection_controls:
                    return
                batch_state = selection_controls["batch_state"]
                mode = str(mode_dropdown.value)
                key = (
                    current_split_name(), mode,
                    parse_grasp_type_option(grasp_type_dropdown.value) if mode == "random_objects"
                    else str(object_dropdown.value),
                )
                pending = key != batch_state.get("key")
                next_batch_button.disabled = pending or batch_state.get("page_count", 1) <= 1
                if status_handle is not None:
                    status_handle.content = (
                        "Selection changed. Click Apply Selection; the previous view is still displayed."
                        if pending else batch_state.get("selection_info", "Selection loaded.")
                    )

            refresh_selection_widgets()

            @apply_button.on_click
            def _on_apply_selection(event):
                del event
                try:
                    new_scene_records = load_selection_scene_records()
                except Exception as exc:
                    if status_handle is not None and hasattr(status_handle, "content"):
                        status_handle.content = f"Selection failed: `{exc}`"
                    print(f"[{log_prefix}] Selection failed: {exc}")
                    return
                if status_handle is not None and hasattr(status_handle, "content"):
                    status_handle.content = selection_controls.get("batch_state", {}).get(
                        "selection_info", f"Loaded {len(new_scene_records)} scene(s)."
                    )
                sync_current_object_selection()
                update_scene_records(new_scene_records)
                refresh_selection_widgets()

            @next_batch_button.on_click
            def _on_next_selection_batch(event):
                del event
                try:
                    new_scene_records = load_selection_scene_records(advance_batch=True)
                except Exception as exc:
                    if status_handle is not None and hasattr(status_handle, "content"):
                        status_handle.content = f"Next batch failed: `{exc}`"
                    print(f"[{log_prefix}] Next batch failed: {exc}")
                    return
                batch_index = selection_controls.get("batch_state", {}).get("index", 0)
                if status_handle is not None and hasattr(status_handle, "content"):
                    status_handle.content = selection_controls.get("batch_state", {}).get(
                        "selection_info",
                        f"Loaded {next_button_label.lower()} {batch_index + 1} with {len(new_scene_records)} scene(s).",
                    )
                sync_current_object_selection()
                update_scene_records(new_scene_records)
                refresh_selection_widgets()

            if extra_action_button is not None:
                @extra_action_button.on_click
                def _on_extra_selection_action(event):
                    del event
                    try:
                        new_scene_records = load_selection_action_records(extra_action_label)
                    except Exception as exc:
                        if status_handle is not None and hasattr(status_handle, "content"):
                            status_handle.content = f"{extra_action_label} failed: `{exc}`"
                        print(f"[{log_prefix}] {extra_action_label} failed: {exc}")
                        return
                    if status_handle is not None and hasattr(status_handle, "content"):
                        status_handle.content = (
                            f"Loaded {extra_action_label.lower()} with {len(new_scene_records)} scene(s)."
                        )
                    sync_current_object_selection()
                    update_scene_records(new_scene_records)

            @mode_dropdown.on_update
            def _on_selection_mode_update(event):
                del event
                refresh_selection_widgets()
                mark_pending_selection()

            if split_dropdown is not None:
                @split_dropdown.on_update
                def _on_selection_split_update(event):
                    del event
                    refresh_selection_widgets()
                    mark_pending_selection()

            @object_dropdown.on_update
            def _on_selection_object_update(event):
                del event
                update_object_info()
                mark_pending_selection()

            @grasp_type_dropdown.on_update
            def _on_selection_type_update(event):
                del event
                mark_pending_selection()

    if next_batch_loader is not None and hasattr(server, "gui"):
        with server.gui.add_folder("Dataloader"):
            next_batch_button = server.gui.add_button("Next Batch")
            batch_status = (
                server.gui.add_markdown("Current batch loaded.")
                if hasattr(server.gui, "add_markdown")
                else None
            )

            @next_batch_button.on_click
            def _on_next_batch(event):
                del event
                try:
                    new_scene_records = next_batch_loader()
                except Exception as exc:
                    if batch_status is not None and hasattr(batch_status, "content"):
                        batch_status.content = f"Next batch failed: `{exc}`"
                    print(f"[{log_prefix}] Next batch failed: {exc}")
                    return
                update_scene_records(new_scene_records)
                if batch_status is not None and hasattr(batch_status, "content"):
                    batch_status.content = f"Loaded next batch with {len(new_scene_records)} scene(s)."

    @display_dropdown.on_update
    def _on_display_update(event):
        del event
        state["display_mode"] = str(display_dropdown.value)
        render_current_view()
        if frame_scenes:
            frame_current_scenes()

    @scene_dropdown.on_update
    def _on_scene_update(event):
        del event
        state["scene_id"] = sample_options_state["options"].index(scene_dropdown.value)
        if state["display_mode"] == "single":
            render_current_view()
            if frame_scenes:
                frame_current_scenes()

    render_current_view()
    if frame_scenes:
        frame_current_scenes()

    print(f"[{log_prefix}] Viser server running at http://localhost:{port}")
    print(f"[{log_prefix}] Rendering {len(records_state['records'])} scene(s). Press Ctrl+C to quit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print(f"\n[{log_prefix}] Shutting down.")
    finally:
        remove_viser_handles(scene_handles["value"])


def infer_visualize_mode(config: DictConfig):
    configured = OmegaConf.select(config, "task.mode")
    if configured not in (None, "auto"):
        return str(configured)
    if bool(OmegaConf.select(config, "data.human")):
        return "human"
    if bool(OmegaConf.select(config, "test_data.human")):
        return "human"
    if OmegaConf.select(config, "test_data.robot_urdf_path") is not None:
        return "robot"
    if OmegaConf.select(config, "test_data.robot.urdf_path") is not None:
        return "robot"
    if OmegaConf.select(config, "data.robot_urdf_path") is not None:
        return "robot"
    if OmegaConf.select(config, "data.robot.urdf_path") is not None:
        return "robot"
    return "human"


def get_output_dir(config: DictConfig) -> str:
    ckpt_path = str(config.ckpt)
    ckpt_name = os.path.basename(ckpt_path)
    step_name = os.path.splitext(ckpt_name)[0]

    if step_name.startswith("step_"):
        test_dir = os.path.join(str(config.output_folder), str(config.wandb.id), "tests", step_name)
    else:
        test_dir = ckpt_path.replace("ckpts", "tests").replace(".pth", "")
    return os.path.join(test_dir, config.test_data.name)


def resolve_human_dataset_path(path):
    """Resolve dataset-relative records using the configured dataset root."""
    from dexlearn.utils.resources import resolve_dataset_path
    return resolve_dataset_path(path)


def resolve_visualization_path(config: DictConfig, path):
    """Resolve a visualization input relative to Hydra's launch directory."""
    if not is_configured_path(path) or os.path.isabs(str(path)):
        return str(path)
    launch_dir = OmegaConf.select(config, "hydra.runtime.cwd") or os.getcwd()
    return os.path.abspath(os.path.join(str(launch_dir), str(path)))


def resolve_human_scores_dir(config: DictConfig):
    """Resolve the Human type-branch sample root used by the pose viewer.

    An explicit ``task.human_scores_dir`` takes precedence. Otherwise the
    directory is derived from the type branch experiment name and checkpoint.
    """
    explicit_dir = get_task_value(config, "human_scores_dir", "")
    if is_configured_path(explicit_dir):
        return resolve_visualization_path(config, explicit_dir)

    score_exp_name = get_task_value(config, "human_scores_exp_name", "")
    score_ckpt = get_task_value(config, "human_scores_ckpt", "")
    has_exp_name = is_configured_path(score_exp_name)
    has_ckpt = is_configured_path(score_ckpt)
    if has_exp_name != has_ckpt:
        raise ValueError("Set both task.human_scores_exp_name and task.human_scores_ckpt, or neither.")
    if not has_exp_name:
        return ""

    ckpt_name = os.path.basename(str(score_ckpt))
    step_text = os.path.splitext(ckpt_name)[0] if ckpt_name.endswith(".pth") else ckpt_name
    step_text = step_text.removeprefix("step_")
    if not step_text.isdigit():
        raise ValueError(f"task.human_scores_ckpt must be a checkpoint step, got {score_ckpt!r}")
    step_name = f"step_{int(step_text):06d}"
    output_folder = resolve_visualization_path(config, str(config.output_folder))
    run_id = f"{config.data_name}_{config.algo_name}_{score_exp_name}"
    return os.path.join(output_folder, run_id, "tests", step_name, str(config.test_data.name))


def resolve_robot_dataset_path(path: str, object_root: str) -> str:
    """Resolve a dataset-relative path without inferring historical prefixes."""
    if os.path.isabs(path):
        return path
    resolved = resolve_human_dataset_path(path)
    if os.path.exists(resolved):
        return resolved
    normalized_root = os.path.normpath(object_root)
    marker = f"{os.sep}object{os.sep}"
    if marker in normalized_root and path.startswith("object/"):
        return os.path.join(normalized_root.split(marker, 1)[0], path)
    return resolved


def visualize_with_trimesh(verts, faces, joints=None, color=[200, 200, 250, 255]):
    hand_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    hand_mesh.visual.face_colors = color
    scene_contents = [hand_mesh]
    if joints is not None:
        for joint in joints:
            sphere = trimesh.creation.uv_sphere(radius=0.005)
            sphere.visual.face_colors = [255, 0, 0, 255]
            sphere.apply_translation(joint)
            scene_contents.append(sphere)
    return scene_contents


def infer_pc_source_from_sample_file(sample_file):
    base_name = os.path.basename(sample_file)
    if base_name.startswith("complete_point_cloud"):
        return "complete"
    if base_name.startswith("partial_pc"):
        return "partial"

    try:
        sample_data = np.load(sample_file, allow_pickle=True).item()
    except Exception:
        return "unknown"
    pc_path = str(sample_data.get("pc_path", ""))
    pc_base = os.path.basename(pc_path)
    if "processed_data" in pc_path or pc_base == "complete_point_cloud.npy":
        return "complete"
    if "vision_data" in pc_path or pc_base.startswith("partial_pc"):
        return "partial"
    return "unknown"


def normalize_object_scale(obj_scale, scene_path):
    scale = np.asarray(obj_scale, dtype=np.float32).reshape(-1)
    if scale.size == 1:
        return np.repeat(scale, 3)
    if scale.size == 3:
        return scale
    raise ValueError(f"Unsupported object scale shape {scale.shape} in scene config: {scene_path}")


def object_pose_to_rt(obj_pose, scene_path):
    from dexlearn.utils.rot import numpy_quaternion_to_matrix

    pose = np.asarray(obj_pose, dtype=np.float32)
    if pose.shape == (4, 4):
        return pose[:3, :3], pose[:3, 3]

    pose = pose.reshape(-1)
    if pose.size == 7:
        rot = numpy_quaternion_to_matrix(pose[3:7].reshape(1, 4))[0].astype(np.float32)
        return rot, pose[:3]

    raise ValueError(f"Unsupported object pose shape {pose.shape} in scene config: {scene_path}")


def extract_object_meta(scene_cfg, scene_path):
    """Extract object name, scale, and pose transform from scene config."""
    if "object" in scene_cfg:
        obj_data = scene_cfg["object"]
        obj_name = obj_data.get("name")
        obj_scale = obj_data.get("rel_scale", obj_data.get("scale"))
        obj_pose = obj_data.get("pose")
    else:
        scene = scene_cfg.get("scene")
        if scene is None:
            raise KeyError(f"Could not find 'scene' in scene config: {scene_path}")

        obj_name = scene_cfg.get("task", {}).get("obj_name")
        if obj_name is None:
            candidates = [
                name
                for name, entry in scene.items()
                if isinstance(entry, dict) and "scale" in entry and "pose" in entry and name != "table"
            ]
            if not candidates:
                raise KeyError(f"Could not infer object name from scene config: {scene_path}")
            obj_name = candidates[0]

        if obj_name not in scene:
            raise KeyError(f"Object '{obj_name}' not found in scene config: {scene_path}")
        obj_data = scene[obj_name]
        obj_scale = obj_data.get("scale")
        obj_pose = obj_data.get("pose")

    if obj_name is None:
        raise KeyError(f"Could not find object name in scene config: {scene_path}")
    if obj_scale is None:
        raise KeyError(f"Could not find object scale for '{obj_name}' in scene config: {scene_path}")
    if obj_pose is None:
        raise KeyError(f"Could not find object pose for '{obj_name}' in scene config: {scene_path}")

    obj_scale_xyz = normalize_object_scale(obj_scale, scene_path)
    obj_rot, obj_trans = object_pose_to_rt(obj_pose, scene_path)
    return obj_name, obj_scale_xyz, obj_rot, obj_trans


def transform_complete_pc(pc, obj_scale_xyz, obj_rot, obj_trans):
    # complete_point_cloud.npy is normalized and must be transformed to world coordinates.
    scaled_pc = pc * obj_scale_xyz[None, :]
    return np.matmul(scaled_pc, obj_rot.T) + obj_trans[None, :]


def build_human_visualization_index(sample_records, grasp_type_names):
    """Build human-specific sample indexes for score-only and pose visualization.

    Args:
        sample_records: Saved sample records from ``build_visualization_sample_index``.
        grasp_type_names: Ordered grasp type names such as ``GRASP_TYPES``.

    Returns:
        Dictionary containing canonical-object-level score and pose indexes.
        Score summaries come only from dedicated ``0_any`` samples.
    """
    object_ids = set()
    score_records_by_object = {}
    pose_records_by_object_and_type = {}

    for record in sample_records:
        base_object_id = base_object_id_from_sequence(record["object_id"])
        object_ids.add(base_object_id)
        sample_group = str(record.get("sample_group", ""))
        grasp_type_id = sample_group_to_grasp_type_id(sample_group)
        if grasp_type_id is None:
            continue

        if grasp_type_id == 0:
            score_records_by_object.setdefault(base_object_id, []).append(record)
            continue

        pose_records_by_object_and_type.setdefault(base_object_id, {}).setdefault(grasp_type_id, []).append(record)

    return {
        "records": sample_records,
        "object_ids": tuple(sorted(object_ids, key=natural_sort_key)),
        "score_records_by_object": score_records_by_object,
        "pose_records_by_object_and_type": pose_records_by_object_and_type,
        "grasp_type_names": tuple(grasp_type_names),
        "score_summary_cache": {},
    }


def human_available_modes(human_index):
    modes = []
    if human_index["score_records_by_object"] or human_index["pose_records_by_object_and_type"]:
        modes.append("random_objects")
    if human_index["pose_records_by_object_and_type"]:
        modes.append("one_object")
    return tuple(modes)


def human_available_grasp_type_ids(human_index):
    type_ids = {0} if human_index["score_records_by_object"] else set()
    for records_by_type in human_index["pose_records_by_object_and_type"].values():
        type_ids.update(type_id for type_id, records in records_by_type.items() if records)
    return tuple(sorted(type_ids))


def human_page_count(item_count: int, page_size: int) -> int:
    return max(1, (item_count + page_size - 1) // page_size)


def load_human_split_object_ids(config: DictConfig, split_names=HUMAN_VISER_SPLIT_OPTIONS):
    """Load canonical human object ids for train/test split filtering.

    Args:
        config: Full Hydra config after data configs have been flattened.
        split_names: Split names whose ``.json`` files should be read.

    Returns:
        A dictionary mapping split names to canonical object id sets. Missing
        files produce empty sets so visualization can still fall back to the
        splits that are present in the current saved sample directory.
    """
    object_path = str(getattr(config.test_data, "object_path", ""))
    split_path = str(getattr(config.test_data, "split_path", "valid_split"))
    split_root = os.path.join(object_path, split_path)
    split_object_ids = {}
    for split_name in split_names:
        if split_name == "all":
            continue
        path = os.path.join(split_root, f"{split_name}.json")
        object_ids = set()
        try:
            with open(path, "r", encoding="utf-8") as f:
                object_ids = {base_object_id_from_sequence(item) for item in json.load(f)}
        except FileNotFoundError:
            print(f"[visualize] Human split file not found: {path}")
        except Exception as exc:
            print(f"[visualize] Could not load human split file {path}: {exc}")
        split_object_ids[str(split_name)] = object_ids
    return split_object_ids


def build_human_visualization_indices_by_split(sample_records, grasp_type_names, split_object_ids):
    """Build one human visualization index per dataset split.

    Args:
        sample_records: Saved sample records from ``build_visualization_sample_index``.
        grasp_type_names: Ordered grasp type names such as ``GRASP_TYPES``.
        split_object_ids: Mapping from split name to canonical object id set.

    Returns:
        A dictionary mapping split name to a human visualization index.
        ``all`` contains each saved record once; train/test use the split files.
    """
    records_by_split = {"all": list(sample_records)}
    records_by_split.update({split: [] for split in split_object_ids if split != "all"})
    for record in sample_records:
        base_object_id = base_object_id_from_sequence(record["object_id"])
        for split_name, object_ids in split_object_ids.items():
            if split_name == "all":
                continue
            if base_object_id in object_ids:
                records_by_split.setdefault(split_name, []).append(record)
                break

    indices_by_split = {}
    for split_name, split_records in records_by_split.items():
        if not split_records:
            continue
        indices_by_split[split_name] = build_human_visualization_index(split_records, grasp_type_names)
        print(
            f"[visualize] Human split {split_name}: "
            f"{len(split_records)} sample record(s), "
            f"{len(indices_by_split[split_name]['object_ids'])} object(s)."
        )
    return indices_by_split


def human_object_score_summary(human_index, object_id: str, scene_path_resolver):
    """Aggregate human CE type-prior score outputs for one object.

    Args:
        human_index: Human visualization index from
            ``build_human_visualization_index``.
        object_id: Canonical object id.
        scene_path_resolver: Resolver used to hydrate saved human scene paths.

    Returns:
        Dictionary containing averaged score vector and representative records,
        or ``None`` when no compatible score sample exists for the object.
    """
    object_id = base_object_id_from_sequence(object_id)
    cache = human_index["score_summary_cache"]
    if object_id in cache:
        return cache[object_id]

    dedicated_score_records = human_index["score_records_by_object"].get(object_id, [])
    if not dedicated_score_records:
        cache[object_id] = None
        return None

    score_vectors = []
    scored_records = []
    representative_record = None
    representative_data = None
    records_to_scan = dedicated_score_records[:HUMAN_SCORE_SUMMARY_MAX_RECORDS]
    for record in records_to_scan:
        try:
            load_visualization_record_payload(
                record,
                scene_path_resolver=scene_path_resolver,
                load_scene_cfg=False,
            )
            data = record.get("data") or {}
            scores = get_human_score_vector_from_data(data, human_index["grasp_type_names"])
        except Exception as exc:
            print(f"[visualize] Skipping unreadable score sample {record['sample_file']}: {exc}")
            continue
        if scores is None or not np.all(np.isfinite(scores)):
            continue
        if representative_record is None:
            representative_record = record
            representative_data = data
        score_vectors.append(scores)
        scored_records.append(record)

    if not score_vectors:
        cache[object_id] = None
        return None

    mean_scores = np.mean(np.stack(score_vectors, axis=0), axis=0)
    summary = {
        "object_id": object_id,
        "scores": mean_scores,
        "sample_count": len(scored_records),
        "score_records": scored_records,
        "representative_record": representative_record,
        "representative_data": representative_data,
    }
    human_index["score_summary_cache"][object_id] = summary
    return summary


def load_human_scores_index(scores_dir):
    """Index an optional type-branch sample root containing ``0_any/``."""
    if not is_configured_path(scores_dir):
        return None
    scores_dir = os.path.abspath(os.path.expanduser(str(scores_dir)))
    if not os.path.isdir(os.path.join(scores_dir, "0_any")):
        raise ValueError(f"Human score sample directory must contain a 0_any directory: {scores_dir}")
    records = build_visualization_sample_records(
        scores_dir, list_sample_files(os.path.join(scores_dir, "0_any")), load_payload=False,
    )
    if not records:
        raise ValueError(f"No 0_any score samples found in Human score sample directory: {scores_dir}")
    return build_human_visualization_index(records, GRASP_TYPES)


def format_human_scores_panel(info, scores_dir):
    """Format scores for the loaded object, including explicit missing-data states."""
    text = f"**Object:** {format_gui_wrappable_value(info['object_id'])}\n\n"
    if not is_configured_path(scores_dir):
        return text + (
            "Scores unavailable. Set `task.human_scores_exp_name` and "
            "`task.human_scores_ckpt`, or use `task.human_scores_dir`."
        )
    summary = info["summary"]
    if summary is None:
        return text + "Scores unavailable for this object."
    rows = ["| Grasp type | Score |", "| :--- | ---: |"]
    rows.extend(f"| {name} | {score:.2f} |" for name, score in zip(GRASP_TYPES[1:], summary["scores"]))
    return text + "\n".join(rows) + f"\n\nType-branch object mean ({summary['sample_count']} samples)"


def add_human_scores_gui(server, scene_records, scores_dir):
    """Return a panel updater driven only by successfully loaded scene records."""
    with server.gui.add_folder("Grasp Type Scores", expand_by_default=True) as folder:
        content = server.gui.add_markdown("")
        with server.gui.add_folder("Source", expand_by_default=False):
            server.gui.add_markdown(
                format_gui_wrappable_value(scores_dir) if is_configured_path(scores_dir) else "Not configured."
            )

    def update(records):
        info = records[0].get("human_score_info") if records else None
        folder.visible = info is not None
        content.content = format_human_scores_panel(info, scores_dir) if info is not None else ""

    update(scene_records)
    return update


def format_score_vector_text(scores: np.ndarray, grasp_type_names) -> str:
    """Format one score vector into compact per-type text.

    Args:
        scores: Real-type score vector aligned with type ids ``1..5``.
        grasp_type_names: Ordered grasp type names including ``0_any``.

    Returns:
        Compact text such as ``1:0.91 2:0.02 3:1.00 4:0.00 5:0.14``.
    """
    parts = []
    for idx, score in enumerate(np.asarray(scores).reshape(-1), start=1):
        short_name = str(grasp_type_names[idx]).split("_", 1)[0]
        parts.append(f"{short_name}:{float(score):.2f}")
    return " ".join(parts)


def format_score_array_text(scores: np.ndarray) -> str:
    """Format ordered CE type-prior scores as a compact array string.

    Args:
        scores: Real-type score vector aligned with type ids ``1..5``.

    Returns:
        Ordered score text such as ``[1.00, 0.00, 1.00, 0.00, 0.00]``.
    """
    score_values = np.asarray(scores, dtype=np.float64).reshape(-1)
    return "[" + ", ".join(f"{score:.2f}" for score in score_values) + "]"


def top_scoring_type_text(scores: np.ndarray, grasp_type_names, top_k: int = 3) -> str:
    """Format the top-scoring human grasp types for one object.

    Args:
        scores: Real-type score vector aligned with type ids ``1..5``.
        grasp_type_names: Ordered grasp type names including ``0_any``.
        top_k: Maximum number of ranked types to format.

    Returns:
        Compact ranked text such as ``3_right_full=1.00, 5_both_full=0.98``.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    ranked = np.argsort(-scores)[: max(0, int(top_k))]
    parts = []
    for rank_idx in ranked:
        type_id = rank_idx + 1
        parts.append(f"{grasp_type_names[type_id]}={scores[rank_idx]:.2f}")
    return ", ".join(parts)


def compact_human_label_caption(
    mode: str,
    object_id: str,
    grasp_type_id: int,
    score_summary,
    sample_rank: int = None,
    grasp_error: float = None,
):
    """Build concise default 3D label text for one human visualization scene.

    Args:
        object_id: Canonical object id displayed in this scene.
        grasp_type_id: Grasp type id shown in this scene. ``0`` means score-only.
        score_summary: Optional aggregated score summary for the object.
        sample_rank: Optional 1-based sample rank inside the current object/type block.
        grasp_error: Optional grasp error value from the saved sample.

    Returns:
        Short multi-field caption string suitable for default 3D labels.
    """
    del object_id, sample_rank, grasp_error
    mode = normalize_visualize_mode(mode)
    if int(grasp_type_id) == 0 and score_summary is not None:
        return format_score_array_text(score_summary["scores"])
    return GRASP_TYPES[int(grasp_type_id)] if int(grasp_type_id) > 0 else ""


def sample_human_random_object_records(
    human_index,
    grasp_type_id: int,
    max_objects: int,
    fixed_object_ids=None,
):
    """Select object-level human records for ``random_objects`` mode.

    Args:
        human_index: Human visualization index.
        grasp_type_id: Target grasp type id. ``0`` selects score-only records.
        max_objects: Maximum number of objects/scenes to show.
        fixed_object_ids: Optional ordered object ids to keep the same batch
            while switching grasp types.

    Returns:
        List of dictionaries describing selected objects and representative
        score/pose records.
    """
    candidates = []
    object_ids = tuple(fixed_object_ids) if fixed_object_ids is not None else human_index["object_ids"]
    for object_id in object_ids:
        score_summary = (
            human_object_score_summary(human_index, object_id, resolve_human_dataset_path)
            if grasp_type_id == 0 else None
        )
        if grasp_type_id == 0:
            if score_summary is None or score_summary["representative_record"] is None:
                continue
            candidates.append(
                {
                    "object_id": object_id,
                    "score_summary": score_summary,
                    "record": random.choice(score_summary["score_records"]),
                    "grasp_type_id": 0,
                }
            )
            continue

        pose_records = human_index["pose_records_by_object_and_type"].get(object_id, {}).get(int(grasp_type_id), [])
        if not pose_records:
            continue
        candidates.append(
            {
                "object_id": object_id,
                "score_summary": score_summary,
                "record": random.choice(pose_records),
                "grasp_type_id": int(grasp_type_id),
            }
        )

    if not candidates:
        return []
    if fixed_object_ids is None:
        selected = randomize_records(candidates, max_objects)
    else:
        selected = candidates[: max_objects if max_objects > 0 else None]
    for idx, item in enumerate(selected):
        item["scene_label"] = f"{idx:02d} | {item['object_id']}"
    return selected


def sample_human_one_object_records(
    human_index,
    object_id: str,
    per_type_grasps: int = 5,
    batch_index: int = 0,
):
    """Select one object's 5-type visualization bundle.

    Args:
        human_index: Human visualization index.
        object_id: Target canonical object id.
        per_type_grasps: Number of pose samples to show for each real grasp type.
        batch_index: Zero-based batch index used to page within each grasp type.

    Returns:
        List of dictionaries describing selected pose records ordered by grasp type.
    """
    object_id = base_object_id_from_sequence(object_id)
    selected = []
    for grasp_type_id in range(1, len(human_index["grasp_type_names"])):
        pose_records = sorted(
            human_index["pose_records_by_object_and_type"].get(object_id, {}).get(grasp_type_id, []),
            key=lambda record: record["sample_file"],
        )
        if not pose_records:
            continue
        page_size = max(1, int(per_type_grasps))
        page_start = batch_index * page_size
        for sample_rank, record in enumerate(pose_records[page_start:page_start + page_size], start=page_start + 1):
            selected.append(
                {
                    "object_id": object_id,
                    "score_summary": None,
                    "record": record,
                    "grasp_type_id": grasp_type_id,
                    "sample_rank": sample_rank,
                    "scene_label": f"{GRASP_TYPES[grasp_type_id]} | {sample_rank}",
                }
            )
    return selected


def build_human_selection_controls(
    config, indices_by_split, initial_split, build_scene_records,
    random_object_count=25, per_type_grasps=5,
    human_scores_index=None,
):
    """Build Human selection state independently of MANO and the Viser server."""
    random_object_count = max(1, int(random_object_count))
    per_type_grasps = max(1, int(per_type_grasps))
    split_options = tuple(split for split in HUMAN_VISER_SPLIT_OPTIONS if split in indices_by_split)
    initial_index = indices_by_split[initial_split]
    modes = human_available_modes(initial_index)
    if not modes:
        raise ValueError("No supported Human score or pose sample groups found.")
    requested_mode = normalize_visualize_mode(get_task_value(config, "visualize_mode", "random_objects"))
    initial_mode = pick_initial_option(modes, requested_mode)
    batch_state = {"key": None, "index": 0, "selection_info": ""}
    pools = {}

    def object_options(mode, split_name=None):
        index = indices_by_split[split_name or initial_split]
        if mode == "one_object":
            return tuple(obj for obj in index["object_ids"] if index["pose_records_by_object_and_type"].get(obj))
        return index["object_ids"]

    def type_options(split_name):
        return tuple(
            format_grasp_type_option(i, GRASP_TYPES)
            for i in human_available_grasp_type_ids(indices_by_split[split_name])
        )

    def load_scene_records(mode, object_id, grasp_type_option, advance_batch=False, split_name=None):
        started = time.perf_counter()
        split_name = split_name or initial_split
        index = indices_by_split[split_name]
        if mode not in human_available_modes(index):
            raise ValueError(f"No pose samples available for {mode} in {split_name}.")
        type_id = parse_grasp_type_option(grasp_type_option)
        key = (split_name, mode, type_id if mode == "random_objects" else object_id)
        next_index = batch_state["index"] + 1 if advance_batch and key == batch_state["key"] else 0
        if mode == "random_objects":
            if type_id not in human_available_grasp_type_ids(index):
                raise ValueError(f"No samples for grasp type {type_id} in {split_name}.")
            pool_key = (split_name, type_id)
            if pool_key not in pools:
                pool = [obj for obj in index["object_ids"] if (
                    index["score_records_by_object"].get(obj) if type_id == 0 else
                    index["pose_records_by_object_and_type"].get(obj, {}).get(type_id)
                )]
                random.shuffle(pool)
                pools[pool_key] = pool
            pool = pools[pool_key]
            page_count = human_page_count(len(pool), random_object_count)
            page = next_index % page_count
            entries = sample_human_random_object_records(
                index, type_id, random_object_count,
                fixed_object_ids=pool[page * random_object_count:(page + 1) * random_object_count],
            )
        else:
            records_by_type = index["pose_records_by_object_and_type"].get(object_id, {})
            if not records_by_type:
                raise ValueError(f"No pose samples for object {object_id} in {split_name}.")
            page_count = max(human_page_count(len(records), per_type_grasps) for records in records_by_type.values())
            page = next_index % page_count
            entries = sample_human_one_object_records(index, object_id, per_type_grasps, page)
        if not entries:
            raise ValueError("No readable samples for this selection. The previous view is unchanged.")
        scenes = build_scene_records(entries, mode)
        if mode == "one_object" and scenes:
            loaded_object_id = entries[0]["object_id"]
            scenes[0]["human_score_info"] = {
                "object_id": loaded_object_id,
                "summary": human_object_score_summary(
                    human_scores_index, loaded_object_id, resolve_human_dataset_path,
                ) if human_scores_index is not None else None,
            }
        wrapped = next_index >= page_count
        info = f"Loaded {split_name} / {mode}: Page {page + 1}/{page_count}."
        if wrapped:
            info += " Returned to first page."
        if mode == "one_object":
            info += " Rows with no remaining poses are empty."
        batch_state.update(key=key, index=page, page_count=page_count, selection_info=info)
        print(f"[visualize] {info} Built {len(scenes)} scene(s) in {time.perf_counter() - started:.2f}s.")
        return scenes

    return {
        "mode_options": modes,
        "mode_options_for_split": lambda split: human_available_modes(indices_by_split[split]),
        "grasp_type_options_for_split": type_options,
        "split_options": split_options,
        "initial_split": initial_split,
        "initial_mode": initial_mode,
        "object_options": initial_index["object_ids"],
        "base_object_options": initial_index["object_ids"],
        "initial_object": pick_initial_object_option(
            object_options(initial_mode), get_task_value(config, "object_id", None)
        ),
        "grasp_type_options": type_options(initial_split),
        "initial_grasp_type": pick_initial_option(
            type_options(initial_split), get_task_value(config, "target_grasp_type_id", 0)
        ),
        "load_scene_records": load_scene_records,
        "batch_state": batch_state,
        "object_options_for_mode": object_options,
        "disable_object_for_mode": lambda mode: mode == "random_objects",
        "disable_grasp_type_for_mode": lambda mode: mode == "one_object",
        "disable_next_batch_for_mode": lambda mode: batch_state.get("page_count", 1) <= 1,
    }


def task_visualize_human(config: DictConfig) -> None:
    import torch
    from pytorch3d.transforms import matrix_to_axis_angle
    from manopth.manolayer import ManoLayer
    from mr_utils.utils_calc import posQuat2Isometry3d, quatWXYZ2XYZW

    from dexlearn.utils.human_hand import get_wrist_translation_from_target, normalize_hand_pos_source
    from dexlearn.utils.logger import Logger
    from dexlearn.utils.util import set_seed

    set_seed(config.seed)
    flatten_multidex_data_config(config.data)
    flatten_multidex_data_config(config.test_data)
    Logger(config)
    visualizer = str(get_task_value(config, "visualizer", "trimesh")).lower()
    if visualizer not in {"trimesh", "viser"}:
        raise ValueError(f"Unsupported visualizer={visualizer}. Expected one of ['trimesh', 'viser'].")
    viser_port = int(get_task_value(config, "viser_port", 8080))
    viser_scene_spacing = float(get_task_value(config, "viser_scene_spacing", 0.8))
    viser_display_mode = str(get_task_value(config, "viser_display_mode", "all"))
    viser_scene_id = int(get_task_value(config, "viser_scene_id", 0))
    mano_layers = {}
    for side in ["left", "right"]:
        mano_layers[side] = ManoLayer(
            center_idx=0,
            mano_root=os.environ.get("MANO_ROOT", "assets/mano"),
            side=side,
            use_pca=False,
            flat_hand_mean=True,
            ncomps=45,
            root_rot_mode="axisang",
            joint_rot_mode="axisang",
        ).to(config.device)

    output_dir = get_output_dir(config)
    scores_dir = resolve_human_scores_dir(config)
    human_scores_index = load_human_scores_index(scores_dir) if visualizer == "viser" else None
    scores_dir = os.path.abspath(os.path.expanduser(str(scores_dir))) if is_configured_path(scores_dir) else ""
    pc_source = str(getattr(config.test_data, "pc_source", "partial")).lower()
    if pc_source not in {"partial", "complete"}:
        raise ValueError(f"Unsupported pc_source={pc_source}. Expected one of ['partial', 'complete'].")

    all_sample_records = build_visualization_sample_index(
        output_dir,
        scene_path_resolver=resolve_human_dataset_path,
        load_payload=False,
    )
    if not all_sample_records:
        raise RuntimeError(f"No saved grasp files found in {output_dir}")

    split_object_ids = load_human_split_object_ids(config)
    human_indices_by_split = build_human_visualization_indices_by_split(
        all_sample_records,
        GRASP_TYPES,
        split_object_ids,
    )
    if not human_indices_by_split:
        fallback_split = str(OmegaConf.select(config, "test_data.test_split") or "test")
        print(
            "[visualize] Could not match saved human samples to train/test split files; "
            f"falling back to one split named {fallback_split}."
        )
        human_indices_by_split = {fallback_split: build_human_visualization_index(all_sample_records, GRASP_TYPES)}
    split_options = tuple(split for split in HUMAN_VISER_SPLIT_OPTIONS if split in human_indices_by_split)
    split_options = split_options or tuple(sorted(human_indices_by_split.keys(), key=natural_sort_key))
    preferred_split = str(
        get_task_value(config, "split", OmegaConf.select(config, "test_data.test_split") or split_options[0])
    )
    initial_split = preferred_split if preferred_split in split_options else split_options[0]
    random_object_scene_count = 25
    one_object_per_type = 5

    def get_human_index_for_split(split_name: str):
        """Return the human visualization index for one split.

        Args:
            split_name: Dataset split selected in the UI.

        Returns:
            Human visualization index for the requested split.
        """
        split_name = str(split_name)
        if split_name not in human_indices_by_split:
            raise ValueError(f"Unknown human split {split_name}. Available splits: {list(human_indices_by_split)}")
        return human_indices_by_split[split_name]

    def load_human_visualization_point_cloud(data, scene_cfg, scene_path, sample_file):
        """Load one visualization point cloud in the world frame when needed.

        Args:
            data: Saved sample payload.
            scene_cfg: Loaded scene config.
            scene_path: Scene config path.
            sample_file: Saved sample file path.

        Returns:
            Subsampled point cloud used for visualization.
        """
        if bool(OmegaConf.select(config, "test_data.human")):
            _, obj_scale_xyz, obj_rot, obj_trans = extract_object_meta(scene_cfg, scene_path)
            pc_path = resolve_human_dataset_path(data["pc_path"])
            raw_pc = np.load(pc_path, allow_pickle=True)
            idx = np.random.choice(raw_pc.shape[0], config.data.num_points, replace=True)
            return transform_complete_pc(raw_pc[idx], obj_scale_xyz, obj_rot, obj_trans)

        pc_path = resolve_human_dataset_path(data["pc_path"])
        raw_pc = np.load(pc_path, allow_pickle=True)
        idx = np.random.choice(raw_pc.shape[0], config.data.num_points, replace=True)
        pc = raw_pc[idx]
        if pc_source == "complete":
            _, obj_scale_xyz, obj_rot, obj_trans = extract_object_meta(scene_cfg, scene_path)
            pc = transform_complete_pc(pc, obj_scale_xyz, obj_rot, obj_trans)
        return pc

    def build_human_scene_records(entry_records, mode: str):
        """Build human visualization scene records for the selected entries.

        Args:
            entry_records: Human visualization entries from one selection mode.
            mode: Human visualization mode.

        Returns:
            Scene records consumed by ``show_scenes_with_viser`` or ``trimesh``.
        """
        scene_records = []
        record_iter = progress_iter(entry_records, desc="Building human visualization scenes", total=len(entry_records))
        for entry in record_iter:
            sample_record = entry["record"]
            grasp_type_id = int(entry["grasp_type_id"])
            object_id = str(entry["object_id"])
            score_summary = entry.get("score_summary")

            load_visualization_record_payload(sample_record, scene_path_resolver=resolve_human_dataset_path)
            sample_file = sample_record["sample_file"]
            data = sample_record["data"]
            scene_path = sample_record["scene_path"]
            scene_cfg = sample_record["scene_cfg"]
            if not scene_cfg:
                raise RuntimeError(f"Scene config is unavailable for sample: {sample_file}")

            grasp_pos_source = normalize_hand_pos_source(data.get("grasp_pos_source", "wrist"))
            pc = load_human_visualization_point_cloud(data, scene_cfg, scene_path, sample_file)

            scene_elements = []
            if grasp_type_id > 0 and "grasp_pose" in data:
                grasp_pose = data["grasp_pose"]
                poses = np.split(grasp_pose, len(grasp_pose) // 7)
                side_names = ["right", "left"]
                colors = [
                    [180, 200, 255, 220],
                    [210, 190, 250, 220],
                ]

                for i, p in enumerate(poses):
                    if np.allclose(p, 0, atol=1e-3):
                        continue

                    hand_pose_np = posQuat2Isometry3d(p[:3], quatWXYZ2XYZW(p[3:7]))
                    side = side_names[i]
                    m_layer = mano_layers[side]
                    hand_target_pos = torch.from_numpy(p[:3]).to(config.device).float()
                    hand_rot_mat = torch.from_numpy(hand_pose_np[:3, :3]).to(config.device).unsqueeze(0).float()

                    mano_params = torch.cat(
                        [matrix_to_axis_angle(hand_rot_mat), torch.zeros((1, 45), device=config.device)], dim=-1
                    )
                    verts, joints = m_layer(mano_params, th_betas=torch.zeros((1, 10), device=config.device))
                    wrist_trans = get_wrist_translation_from_target(hand_target_pos, joints[0], grasp_pos_source)

                    hand_pose_np[:3, 3] = wrist_trans.cpu().numpy()
                    scene_elements.append(trimesh.creation.axis(transform=hand_pose_np, origin_size=0.01))

                    v_np = ((verts[0] / 1000.0) + wrist_trans).cpu().numpy()
                    f_np = m_layer.th_faces.cpu().numpy()
                    scene_elements.extend(visualize_with_trimesh(v_np, f_np, None, color=colors[i]))
                    if grasp_pos_source == "index_mcp":
                        mcp_pose = np.eye(4)
                        mcp_pose[:3, :3] = hand_rot_mat[0].cpu().numpy()
                        mcp_pose[:3, 3] = hand_target_pos.cpu().numpy()
                        scene_elements.append(
                            trimesh.creation.axis(
                                transform=mcp_pose,
                                origin_size=0.008,
                                axis_radius=0.003,
                                axis_length=0.08,
                            )
                        )

            scene_elements.append(trimesh.points.PointCloud(pc, colors=[255, 0, 0, 255]))
            scene_elements.append(trimesh.creation.axis(origin_size=0.01, axis_radius=0.001, axis_length=0.3))
            if grasp_type_id == 0:
                full_caption = (
                    f"{entry['scene_label']} | file={os.path.basename(sample_file)} | "
                    f"type branch object mean (n={score_summary['sample_count']})="
                    f"{format_score_array_text(score_summary['scores'])} | "
                    f"representative point cloud | "
                    f"PC: {infer_pc_source_from_sample_file(sample_file)}"
                )
                label_caption = compact_human_label_caption(
                    mode,
                    object_id,
                    0,
                    score_summary,
                )
            else:
                grasp_error = float(data["grasp_error"]) if "grasp_error" in data else None
                full_caption = (
                    f"{entry['scene_label']} | file={os.path.basename(sample_file)} | "
                    f"type={GRASP_TYPES[grasp_type_id]} | "
                )
                if grasp_error is not None:
                    full_caption += f"err={grasp_error:.4f} | "
                full_caption += f"Pos: {grasp_pos_source} | PC: {infer_pc_source_from_sample_file(sample_file)}"
                label_caption = compact_human_label_caption(
                    mode,
                    object_id,
                    grasp_type_id,
                    score_summary,
                    sample_rank=entry.get("sample_rank"),
                    grasp_error=grasp_error,
                )
            scene_record = {
                "elements": scene_elements,
                "caption": full_caption,
                "label_caption": label_caption,
                "viser_all_label": entry["scene_label"],
            }
            if entry.get("spatial_group"):
                scene_record["viser_spatial_group"] = entry["spatial_group"]
            elif mode == "one_object" and grasp_type_id > 0:
                scene_record["viser_grid_row"] = grasp_type_id - 1
                scene_record["viser_grid_col"] = max(0, int(entry.get("sample_rank", 1)) - 1) % one_object_per_type
                scene_record["viser_grid_rows"] = len(GRASP_TYPES) - 1
                scene_record["viser_grid_cols"] = one_object_per_type
            scene_records.append(scene_record)
        return scene_records

    if visualizer == "viser":
        selection_controls = build_human_selection_controls(
            config, human_indices_by_split, initial_split, build_human_scene_records,
            random_object_scene_count, one_object_per_type, human_scores_index,
        )
        scene_records = selection_controls["load_scene_records"](
            selection_controls["initial_mode"],
            selection_controls["initial_object"],
            selection_controls["initial_grasp_type"],
        )
        sample_groups = {sample_group_to_grasp_type_id(record["sample_group"]) for record in all_sample_records}
        branch = (
            "type scores"
            if sample_groups == {0}
            else "pose samples"
            if 0 not in sample_groups
            else "mixed samples"
        )
        caption_aspects = (
            ("object_id", "file", "point_cloud", "other")
            if sample_groups == {0}
            else ("scene_id", "file", "position_source", "point_cloud", "error", "other")
        )
        experiment = str(OmegaConf.select(config, "exp_name") or OmegaConf.select(config, "wandb.id") or "unknown")
        checkpoint = os.path.basename(str(config.ckpt))
        show_scenes_with_viser(
            scene_records,
            port=viser_port,
            scene_spacing=viser_scene_spacing,
            display_mode=viser_display_mode,
            scene_id=viser_scene_id,
            log_prefix="visualize_human",
            selection_controls=selection_controls,
            gui_description=f"**Human Prior:** {branch} | **Experiment:** {experiment} | **Checkpoint:** {checkpoint}"
                            "\n\nType scores come from dedicated `0_any` samples.",
            frame_scenes=True,
            label_font_screen_scale=0.6,
            caption_aspects=caption_aspects,
            human_scores_dir=scores_dir if 0 not in sample_groups else None,
        )
    else:
        initial_mode = normalize_visualize_mode(get_task_value(config, "visualize_mode", "random_objects"))
        initial_grasp_type_id = int(get_task_value(config, "target_grasp_type_id", 0))
        initial_human_index = get_human_index_for_split(initial_split)
        if initial_mode == "one_object":
            entry_records = sample_human_one_object_records(
                initial_human_index,
                object_id=get_task_value(config, "object_id", None),
                per_type_grasps=one_object_per_type,
            )
        else:
            entry_records = sample_human_random_object_records(
                initial_human_index,
                grasp_type_id=initial_grasp_type_id,
                max_objects=random_object_scene_count,
            )
        show_scenes_with_trimesh(build_human_scene_records(entry_records, initial_mode))


def build_robot_components(config: DictConfig):
    import pytorch_kinematics as pk

    from mr_utils.robot.pk_helper import PytorchKinematicsHelper
    from mr_utils.robot.pk_visualizer import Visualizer

    urdf_path = (
        OmegaConf.select(config, "test_data.robot_urdf_path")
        or OmegaConf.select(config, "data.robot_urdf_path")
        or get_task_value(config, "robot_urdf_path", None)
    )
    mesh_dir_path = (
        OmegaConf.select(config, "test_data.robot_mesh_dir_path")
        or OmegaConf.select(config, "data.robot_mesh_dir_path")
        or get_task_value(config, "robot_mesh_dir_path", None)
    )
    metadata_group = (
        OmegaConf.select(config, "test_data.metadata_group")
        or OmegaConf.select(config, "data.metadata_group")
        or get_task_value(config, "metadata_group", None)
    )
    if urdf_path is None:
        raise ValueError("Missing robot URDF path. Set test_data.robot_urdf_path or task.robot_urdf_path.")
    if mesh_dir_path is None:
        raise ValueError("Missing robot mesh dir path. Set test_data.robot_mesh_dir_path or task.robot_mesh_dir_path.")
    if metadata_group is None:
        raise ValueError("Missing robot metadata group. Set test_data.metadata_group or task.metadata_group.")

    chain = pk.build_chain_from_urdf(open(urdf_path).read()).to(device=config.device)
    robot_helper = PytorchKinematicsHelper(
        chain,
        base_pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        device=config.device,
    )
    robot_visualizer = Visualizer(urdf_path, mesh_dir_path, device=config.device)

    metadata_path = os.path.join(config.test_data.grasp_path, metadata_group, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Joint metadata file not found: {metadata_path}")

    with open(metadata_path, "r") as f:
        joint_metadata = json.load(f)

    joint_names = joint_metadata["joint_names"]
    wrist_link_names = joint_metadata["wrist_body_names"]
    right_arm_indices = [idx for idx, name in enumerate(joint_names) if name.startswith("ra_")]
    left_arm_indices = [idx for idx, name in enumerate(joint_names) if name.startswith("la_")]
    right_hand_indices = [idx for idx, name in enumerate(joint_names) if name.startswith("rh_")]
    left_hand_indices = [idx for idx, name in enumerate(joint_names) if name.startswith("lh_")]

    ik_solver_kwargs = dict(
        pos_tolerance=1e-3,
        rot_tolerance=1e-2,
        max_iterations=50,
        retry_configs=None,
        num_retries=20,
        early_stopping_any_converged=True,
        early_stopping_no_improvement="any",
        debug=False,
        lr=0.2,
        regularlization=1e-3,
    )
    for wrist_link_name in wrist_link_names:
        robot_helper.create_serial_chain(wrist_link_name)
        robot_helper.create_ik_solver(wrist_link_name, **ik_solver_kwargs)

    return {
        "joint_names": joint_names,
        "wrist_link_names": wrist_link_names,
        "right_arm_indices": right_arm_indices,
        "left_arm_indices": left_arm_indices,
        "right_hand_indices": right_hand_indices,
        "left_hand_indices": left_hand_indices,
        "robot_helper": robot_helper,
        "robot_visualizer": robot_visualizer,
    }


def recover_pc_path(sample_file: str, scene_cfg: dict, config: DictConfig) -> str:
    original_name = re.sub(r"_\d+\.npy$", ".npy", os.path.basename(sample_file))
    return os.path.join(config.test_data.object_path, config.test_data.pc_path, scene_cfg["scene_id"], original_name)


def load_robot_visualization_pc(sample_file: str, scene_cfg: dict, config: DictConfig):
    pc_path = recover_pc_path(sample_file, scene_cfg, config)
    raw_pc = np.load(pc_path, allow_pickle=True)
    idx = np.random.choice(raw_pc.shape[0], config.test_data.num_points, replace=True)
    return raw_pc[idx]


def get_robot_target_pc_file(config: DictConfig):
    """Read the optional robot visualization target point-cloud file.

    Args:
        config: Full Hydra config for ``task=visualize``.

    Returns:
        The configured target path string, or ``None`` when no target file was
        requested. ``pc_file`` and ``sample_file`` are accepted as aliases for
        quick command-line overrides.
    """
    for key in ("target_pc_file", "pc_file", "sample_file"):
        value = OmegaConf.select(config, f"task.{key}")
        if is_configured_path(value):
            return str(value).strip()
    return None


def is_path_inside(path: str, root: str) -> bool:
    """Check whether one path is inside a root directory.

    Args:
        path: Candidate filesystem path.
        root: Root directory used for containment checking.

    Returns:
        ``True`` when ``path`` is equal to or contained by ``root``.
    """
    path_abs = os.path.abspath(path)
    root_abs = os.path.abspath(root)
    try:
        return os.path.commonpath([path_abs, root_abs]) == root_abs
    except ValueError:
        return False


def resolve_robot_target_pc_scene(target_pc_file: str, config: DictConfig):
    """Infer scene id and point-cloud basename from a configured target file.

    Args:
        target_pc_file: Absolute path, dataset-relative path, or
            ``scene_id/point_cloud.npy`` value from the task config.
        config: Full Hydra config with ``test_data.object_path`` and
            ``test_data.pc_path``.

    Returns:
        A tuple ``(scene_id, pc_basename)`` used to locate saved robot samples
        for this source point-cloud file.
    """
    target_pc_file = str(target_pc_file).strip()
    pc_basename = os.path.basename(target_pc_file)
    if not pc_basename:
        raise ValueError("task.target_pc_file must include a point-cloud file name.")

    object_path = str(config.test_data.object_path)
    pc_path = str(config.test_data.pc_path)
    object_pc_root = os.path.abspath(os.path.join(object_path, pc_path))
    object_root = os.path.abspath(object_path)

    candidate_paths = []
    if os.path.isabs(target_pc_file):
        candidate_paths.append(target_pc_file)
    else:
        candidate_paths.extend(
            [
                os.path.join(object_pc_root, target_pc_file),
                os.path.join(object_root, target_pc_file),
                os.path.abspath(target_pc_file),
            ]
        )

    existing_path = next((path for path in candidate_paths if os.path.exists(path)), None)
    if existing_path is not None and is_path_inside(existing_path, object_pc_root):
        rel_path = os.path.relpath(existing_path, object_pc_root)
        scene_id = os.path.dirname(rel_path)
        return canonical_object_id(scene_id), pc_basename
    if existing_path is not None and is_path_inside(existing_path, object_root):
        rel_path = os.path.relpath(existing_path, object_root).replace("\\", "/")
    else:
        rel_path = target_pc_file.replace("\\", "/").lstrip("/")

    pc_prefix = pc_path.replace("\\", "/").strip("/")
    if pc_prefix and rel_path.startswith(f"{pc_prefix}/"):
        rel_path = rel_path[len(pc_prefix) + 1 :]
    elif pc_prefix and f"/{pc_prefix}/" in rel_path:
        rel_path = rel_path.split(f"/{pc_prefix}/", 1)[1]

    scene_id = os.path.dirname(rel_path)
    scene_id = canonical_object_id(scene_id)
    if not scene_id:
        raise ValueError(
            "task.target_pc_file must include its scene directory, for example "
            "core_bottle_x/tabletop_ur10e/scale002_pose000_0/partial_pc_01_0.npy."
        )
    return scene_id, pc_basename


def build_robot_target_file_sample_index(
    output_dir: str,
    target_pc_file: str,
    config: DictConfig,
    scene_path_resolver=None,
    load_payload: bool = False,
):
    """Index saved robot samples generated from one source point-cloud file.

    Args:
        output_dir: Saved sample root for the current checkpoint and test data.
        target_pc_file: Source point-cloud file configured by the user.
        config: Full Hydra config for resolving dataset-relative paths.
        scene_path_resolver: Optional callable used to resolve saved scene
            paths when payloads are loaded.
        load_payload: Whether to load selected saved samples immediately.

    Returns:
        Visualization sample records for saved grasps matching the target
        point-cloud file.
    """
    direct_candidates = []
    if os.path.isabs(target_pc_file):
        direct_candidates.append(target_pc_file)
    else:
        direct_candidates.append(os.path.join(output_dir, target_pc_file))
        direct_candidates.append(os.path.abspath(target_pc_file))
    direct_sample_file = next(
        (
            path
            for path in direct_candidates
            if os.path.isfile(path) and is_path_inside(path, output_dir)
        ),
        None,
    )
    if direct_sample_file is not None:
        sample_files = [direct_sample_file]
        print(f"[visualize] Using one configured saved sample file: {direct_sample_file}")
        return build_visualization_sample_records(
            output_dir,
            sample_files,
            scene_path_resolver=scene_path_resolver,
            load_payload=load_payload,
            load_scene_cfg=True,
            log_prefix="Indexing configured robot sample file",
        )

    scene_id, pc_basename = resolve_robot_target_pc_scene(target_pc_file, config)
    pc_stem = os.path.splitext(pc_basename)[0]
    scene_glob = glob_escape(scene_id)
    stem_glob = glob_escape(pc_stem)
    patterns = [
        os.path.join(glob_escape(output_dir), "*", scene_glob, f"{stem_glob}.npy"),
        os.path.join(glob_escape(output_dir), "*", scene_glob, f"{stem_glob}_*.npy"),
    ]
    sample_files = sorted(
        {sample_file for pattern in patterns for sample_file in glob(pattern)},
        key=natural_sort_key,
    )
    print(
        f"[visualize] Found {len(sample_files)} saved sample file(s) for target_pc_file={target_pc_file} "
        f"(scene_id={scene_id}, pc={pc_basename}). load_payload={load_payload}, load_scene_cfg=True"
    )
    return build_visualization_sample_records(
        output_dir,
        sample_files,
        scene_path_resolver=scene_path_resolver,
        load_payload=load_payload,
        load_scene_cfg=True,
        log_prefix="Indexing configured robot target file samples",
    )


def solve_stage_pose(stage_qpos: np.ndarray, robot_assets: dict, device: str):
    """Solve one robot stage pose from a saved robot qpos vector.

    Args:
        stage_qpos: Saved robot stage qpos with right-hand and left-hand
            halves concatenated.
        robot_assets: Dictionary returned by ``build_robot_components``.
        device: Torch device used by the IK helper.

    Returns:
        Tuple ``(robot_pose, right_ok, left_ok)`` for a single sample.
    """
    robot_pose, right_ok, left_ok = solve_stage_pose_batch(
        np.asarray(stage_qpos, dtype=np.float32).reshape(1, -1),
        robot_assets,
        device,
    )
    return robot_pose, bool(right_ok[0]), bool(left_ok[0])


def solve_stage_pose_batch(stage_qpos_batch: np.ndarray, robot_assets: dict, device: str):
    """Solve one robot stage pose for a batch of saved robot qpos vectors.

    Args:
        stage_qpos_batch: Array shaped ``(B, D)`` where each row stores
            right-hand and left-hand qpos halves.
        robot_assets: Dictionary returned by ``build_robot_components``.
        device: Torch device used by the IK helper.

    Returns:
        Tuple ``(robot_pose, right_success, left_success)``. ``robot_pose`` is
        shaped ``(B, 7 + joint_num)`` and success arrays are boolean numpy
        arrays shaped ``(B,)``.
    """
    import torch
    from pytorch3d import transforms as pttf

    stage_qpos_batch = np.asarray(stage_qpos_batch, dtype=np.float32)
    if stage_qpos_batch.ndim != 2:
        raise ValueError(f"stage_qpos_batch must have shape (B, D), got {stage_qpos_batch.shape}")
    half_dim = stage_qpos_batch.shape[1] // 2
    right_qpos = torch.from_numpy(stage_qpos_batch[:, :half_dim]).to(device=device, dtype=torch.float32)
    left_qpos = torch.from_numpy(stage_qpos_batch[:, half_dim:]).to(device=device, dtype=torch.float32)

    right_trans, right_quat, right_joint = right_qpos[:, :3], right_qpos[:, 3:7], right_qpos[:, 7:]
    left_trans, left_quat, left_joint = left_qpos[:, :3], left_qpos[:, 3:7], left_qpos[:, 7:]

    batch_size = int(stage_qpos_batch.shape[0])
    right_matrix = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
    right_matrix[:, :3, :3] = pttf.quaternion_to_matrix(right_quat)
    right_matrix[:, :3, 3] = right_trans

    left_matrix = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
    left_matrix[:, :3, :3] = pttf.quaternion_to_matrix(left_quat)
    left_matrix[:, :3, 3] = left_trans

    right_sol = robot_assets["robot_helper"].solve_ik_batch(robot_assets["wrist_link_names"][0], right_matrix)
    left_sol = robot_assets["robot_helper"].solve_ik_batch(robot_assets["wrist_link_names"][1], left_matrix)

    combined_q = torch.zeros((batch_size, len(robot_assets["joint_names"])), device=device, dtype=torch.float32)
    right_q = right_sol["q"].to(device=device, dtype=combined_q.dtype)
    left_q = left_sol["q"].to(device=device, dtype=combined_q.dtype)
    combined_q[:, robot_assets["right_arm_indices"]] = right_q[:, robot_assets["right_arm_indices"]]
    combined_q[:, robot_assets["left_arm_indices"]] = left_q[:, robot_assets["left_arm_indices"]]
    combined_q[:, robot_assets["right_hand_indices"]] = right_joint
    combined_q[:, robot_assets["left_hand_indices"]] = left_joint

    robot_pose = torch.zeros((batch_size, 7 + combined_q.shape[-1]), device=device, dtype=torch.float32)
    robot_pose[:, 3] = 1.0
    robot_pose[:, 7:] = combined_q
    right_success = right_sol["success"].detach().cpu().numpy().astype(bool).reshape(-1)
    left_success = left_sol["success"].detach().cpu().numpy().astype(bool).reshape(-1)
    return robot_pose, right_success, left_success


def task_visualize_robot(config: DictConfig) -> None:
    from dexlearn.dataset import GRASP_TYPES
    from dexlearn.utils.logger import Logger
    from dexlearn.utils.util import set_seed

    set_seed(config.seed)
    flatten_multidex_data_config(config.data)
    flatten_multidex_data_config(config.test_data)
    Logger(config)
    visualizer = str(get_task_value(config, "visualizer", "trimesh")).lower()
    if visualizer not in {"trimesh", "viser"}:
        raise ValueError(f"Unsupported visualizer={visualizer}. Expected one of ['trimesh', 'viser'].")
    viser_port = int(get_task_value(config, "viser_port", 8080))
    viser_scene_spacing = float(get_task_value(config, "viser_scene_spacing", 1.0))
    viser_display_mode = str(get_task_value(config, "viser_display_mode", "all"))
    viser_scene_id = int(get_task_value(config, "viser_scene_id", 0))
    is_our_human_grasp_format = is_our_human_grasp_format_test_data(config)

    output_dir = get_output_dir(config)
    robot_scene_path_resolver = lambda path: resolve_robot_dataset_path(path, config.test_data.object_path)
    target_pc_file = get_robot_target_pc_file(config)
    if target_pc_file is not None:
        all_sample_records = build_robot_target_file_sample_index(
            output_dir,
            target_pc_file,
            config,
            scene_path_resolver=robot_scene_path_resolver,
            load_payload=False,
        )
    else:
        all_sample_records = build_visualization_sample_index(
            output_dir,
            scene_path_resolver=robot_scene_path_resolver,
            load_payload=False,
        )
    if not all_sample_records:
        target_hint = f" for target_pc_file={target_pc_file}" if target_pc_file is not None else ""
        raise RuntimeError(f"No saved grasp files found in {output_dir}{target_hint}")

    robot_assets = build_robot_components(config)
    stage_names = ["grasp_qpos"]
    stage_colors = [(255, 120, 120, 180)]
    robot_axis_mesh = trimesh.creation.axis(origin_size=0.01, axis_radius=0.001, axis_length=0.3)
    robot_pc_cache = {}
    hide_inactive_single_hand_mesh = bool(get_task_value(config, "hide_inactive_single_hand_mesh", True))
    robot_mesh_visual_scope = str(get_task_value(config, "robot_mesh_visual_scope", "full")).lower()
    robot_fingertip_link_names = get_robot_fingertip_link_names(config)

    def load_cached_robot_visualization_pc(sample_file: str, scene_cfg: dict):
        """Load one robot visualization point cloud with per-session caching.

        Args:
            sample_file: Saved sample path used to recover the source partial
                point-cloud path.
            scene_cfg: Scene configuration payload referenced by the sample.

        Returns:
            Subsampled point cloud array used by the robot visualization. The
            same source pc path is subsampled once and then reused across
            candidates from the same scene.
        """
        pc_path = recover_pc_path(sample_file, scene_cfg, config)
        if pc_path not in robot_pc_cache:
            raw_pc = np.load(pc_path, allow_pickle=True)
            idx = np.random.choice(raw_pc.shape[0], config.test_data.num_points, replace=True)
            robot_pc_cache[pc_path] = raw_pc[idx]
        return robot_pc_cache[pc_path]

    def build_robot_scene_records(sample_records):
        scene_records = []
        one_scene_mode = bool(sample_records) and all(
            normalize_visualize_mode(record.get("visualize_mode", "")) == "one_scene" for record in sample_records
        )
        pending_entries = []
        record_iter = progress_iter(sample_records, desc="Building robot visualization scenes", total=len(sample_records))
        for sample_record in record_iter:
            load_visualization_record_payload(sample_record, scene_path_resolver=robot_scene_path_resolver)
            sample_file = sample_record["sample_file"]

            data = sample_record["data"]
            scene_path = sample_record["scene_path"]
            if not os.path.exists(scene_path):
                raise FileNotFoundError(f"Scene config not found: {scene_path}")
            scene_cfg = sample_record["scene_cfg"]
            if not scene_cfg:
                raise RuntimeError(f"Scene config is unavailable for sample: {sample_file}")

            pc = load_cached_robot_visualization_pc(sample_file, scene_cfg)
            scene_elements = [
                trimesh.points.PointCloud(pc, colors=[255, 165, 0, 255]),
                robot_axis_mesh,
            ]
            pending_entries.append(
                {
                    "sample_record": sample_record,
                    "sample_file": sample_file,
                    "data": data,
                    "scene_elements": scene_elements,
                    "ik_status": [],
                    "mesh_filter": robot_mesh_filter_for_grasp_type(
                        extract_grasp_type_id(data),
                        hide_inactive_single_hand_mesh=hide_inactive_single_hand_mesh,
                        visual_scope=robot_mesh_visual_scope,
                        fingertip_link_names=robot_fingertip_link_names,
                    ),
                }
            )

        if pending_entries:
            for stage_name, color in zip(stage_names, stage_colors):
                stage_qpos_batch = np.stack(
                    [np.asarray(entry["data"][stage_name], dtype=np.float32).reshape(-1) for entry in pending_entries],
                    axis=0,
                )
                robot_pose, right_ok, left_ok = solve_stage_pose_batch(stage_qpos_batch, robot_assets, config.device)
                robot_assets["robot_visualizer"].set_robot_parameters(robot_pose, joint_names=robot_assets["joint_names"])
                for entry_idx, entry in enumerate(pending_entries):
                    entry["scene_elements"].append(
                        robot_assets["robot_visualizer"].get_robot_trimesh_data(
                            i=entry_idx,
                            color=color,
                            **entry["mesh_filter"],
                        )
                    )
                    entry["ik_status"].append(
                        f"{stage_name} IK: R={int(right_ok[entry_idx])} L={int(left_ok[entry_idx])}"
                    )

        for entry in pending_entries:
            sample_record = entry["sample_record"]
            sample_file = entry["sample_file"]
            data = entry["data"]

            caption = f"{os.path.basename(sample_file)} | err={float(data['grasp_error']):.4f}"
            grasp_type_id = None
            pred_grasp_type_id = None
            if "grasp_type_id" in data:
                grasp_type_id = int(data["grasp_type_id"])
                caption = f"{caption} | grasp_type_id={GRASP_TYPES[grasp_type_id]}"
            if "pred_grasp_type_id" in data:
                pred_grasp_type_id = int(data["pred_grasp_type_id"])
                caption = f"{caption} | pred_grasp_type_id={GRASP_TYPES[pred_grasp_type_id]}"
            pred_grasp_type_prob = None
            if "pred_grasp_type_prob" in data:
                pred_grasp_type_prob = np.asarray(data["pred_grasp_type_prob"]).reshape(-1)
                prob_str = ", ".join([f"{p:.3f}" for p in pred_grasp_type_prob.tolist()])
                caption = f"{caption} | pred_grasp_type_prob=[{prob_str}]"
                if pred_grasp_type_id is not None and 0 < pred_grasp_type_id <= pred_grasp_type_prob.shape[0]:
                    type_score = float(pred_grasp_type_prob[pred_grasp_type_id - 1])
                    caption = f"{caption} | pred_grasp_type_score={type_score:.3f}"
            if bool(get_task_value(config, "show_ik_status", True)):
                caption = f"{caption} | {' ; '.join(entry['ik_status'])}"
            caption = prefix_caption_with_viser_label(sample_record, caption)

            scene_record = {
                "elements": entry["scene_elements"],
                "caption": caption,
                "camera_distance": float(get_task_value(config, "camera_distance", 1.0)),
                "smooth": False,
            }
            if grasp_type_id is not None:
                scene_record["grasp_type_id"] = grasp_type_id
            if pred_grasp_type_id is not None:
                scene_record["pred_grasp_type_id"] = pred_grasp_type_id
            if pred_grasp_type_prob is not None:
                scene_record["pred_grasp_type_prob"] = pred_grasp_type_prob
            copy_viser_record_metadata(sample_record, scene_record)
            if visualizer == "trimesh":
                show_scenes_with_trimesh([scene_record])
            else:
                scene_records.append(scene_record)
        if one_scene_mode:
            apply_one_scene_grasp_type_grid(
                scene_records,
                real_type_num=len(GRASP_TYPES) - 1,
                grasp_type_names=GRASP_TYPES,
            )
        return scene_records

    if visualizer == "viser":
        selection_controls = build_viser_selection_controls(
            all_sample_records,
            config,
            build_robot_scene_records,
            grasp_type_names=GRASP_TYPES,
            is_our_human_grasp_format=is_our_human_grasp_format,
        )
        scene_records = selection_controls["load_scene_records"](
            selection_controls["initial_mode"],
            selection_controls["initial_object"],
            selection_controls["initial_grasp_type"],
        )
        show_scenes_with_viser(
            scene_records,
            port=viser_port,
            scene_spacing=viser_scene_spacing,
            display_mode=viser_display_mode,
            scene_id=viser_scene_id,
            log_prefix="visualize_robot",
            selection_controls=selection_controls,
        )
    else:
        initial_mode = normalize_visualize_mode(get_task_value(config, "visualize_mode", "random_objects"))
        initial_object = get_task_value(config, "object_id", None)
        if target_pc_file is not None and initial_mode == "one_object":
            initial_mode = "one_scene"
            initial_object = pick_initial_object_option(list_available_object_ids(all_sample_records), initial_object)
        initial_sample_records = select_visualization_records(
            all_sample_records,
            initial_mode,
            int(get_task_value(config, "max_grasps", 20)),
            initial_object,
            get_task_value(config, "target_grasp_type_id", None),
            is_our_human_grasp_format=is_our_human_grasp_format,
        )
        build_robot_scene_records(initial_sample_records)


def task_visualize(config: DictConfig) -> None:
    flatten_multidex_data_config(config.data)
    flatten_multidex_data_config(config.test_data)
    mode = infer_visualize_mode(config)
    if mode == "human":
        task_visualize_human(config)
        return
    if mode == "robot":
        task_visualize_robot(config)
        return
    raise ValueError(f"Unsupported visualization mode={mode}. Expected 'auto', 'human', or 'robot'.")


@hydra.main(config_path="../config", config_name="base", version_base=None)
def main(config: DictConfig) -> None:
    task_visualize(config)


if __name__ == "__main__":
    main()
