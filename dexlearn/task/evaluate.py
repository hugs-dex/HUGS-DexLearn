import csv
import json
import math
import os
import re
from copy import deepcopy
from glob import glob
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation as SciR

try:
    import hydra
    from omegaconf import ListConfig

    from dexlearn.dataset.grasp_types import GRASP_TYPES
    from dexlearn.utils.config import cfg_get, flatten_multidex_data_config, resolve_type_supervision_config
except ModuleNotFoundError:
    hydra = None
    ListConfig = list
    GRASP_TYPES = [
        "0_any",
        "1_right_two",
        "2_right_three",
        "3_right_full",
        "4_both_three",
        "5_both_full",
    ]

    def cfg_get(config, *keys, default=None):
        """Read a nested config value without requiring OmegaConf.

        Args:
            config: Dictionary-like or object-like config.
            *keys: Candidate dotted keys.
            default: Fallback value when no key exists.

        Returns:
            First matched value, otherwise ``default``.
        """
        for key in keys:
            current = config
            found = True
            for part in str(key).split("."):
                if isinstance(current, dict) and part in current:
                    current = current[part]
                elif hasattr(current, part):
                    current = getattr(current, part)
                else:
                    found = False
                    break
            if found:
                return current
        return default

    def flatten_multidex_data_config(config):
        """Fallback no-op for local smoke tests.

        Args:
            config: Data config.

        Returns:
            The original config.
        """
        return config

    def resolve_type_supervision_config(config):
        """Fallback no-op for local smoke tests.

        Args:
            config: Full config.

        Returns:
            The original config.
        """
        return config


EPS = 1e-12
CE_EPS_VALUES = (1e-12, 1e-9, 1e-6, 1e-3)
REAL_TYPE_IDS = tuple(range(1, len(GRASP_TYPES)))
BOTH_THREE_TYPE_ID = 4
WITHOUT_BOTH_THREE_SUFFIX = "without_both_three"
SCALE_FEATURES = ("xy_long", "xy_short", "z_height")
GRASP_TYPE_LABELS = [name.split("_", 1)[1] for name in GRASP_TYPES[1:]]


def _as_list(value: Any) -> list:
    """Convert scalar or list-like config values to a Python list.

    Args:
        value: Scalar, list, tuple, ``ListConfig``, or ``None``.

    Returns:
        Plain list. ``None`` returns an empty list.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple, ListConfig)):
        return list(value)
    return [value]


def _natural_sort_key(value: Any) -> list:
    """Build a natural sort key for ids containing numbers.

    Args:
        value: Object id, scene id, path, or other sortable value.

    Returns:
        List of text and integer chunks.
    """
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value))]


def _original_cwd() -> str:
    """Return Hydra's original cwd when available.

    Args:
        None.

    Returns:
        Filesystem directory used to resolve relative paths.
    """
    if hydra is not None:
        try:
            return hydra.utils.get_original_cwd()
        except Exception:
            pass
    return os.getcwd()


def _abs_path(path: Any) -> str:
    """Resolve one filesystem path.

    Args:
        path: Absolute or relative path-like value.

    Returns:
        Absolute path. Empty values return an empty string.
    """
    if path is None:
        return ""
    text = str(path)
    if not text:
        return ""
    if os.path.isabs(text):
        return text
    return os.path.abspath(os.path.join(_original_cwd(), text))


def _load_json(path: str) -> Any:
    """Load one JSON file.

    Args:
        path: JSON file path.

    Returns:
        Parsed JSON content.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_csv(rows: list[dict], path: str) -> None:
    """Write dictionaries to CSV.

    Args:
        rows: Row dictionaries.
        path: Output CSV path.

    Returns:
        None.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        open(path, "w", encoding="utf-8").close()
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value: Any) -> Any:
    """Convert NumPy objects to JSON-compatible values.

    Args:
        value: Value passed by the JSON encoder.

    Returns:
        JSON-serializable value.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(data: Any, path: str) -> None:
    """Write compact JSON for machine-readable summaries.

    Args:
        data: JSON-serializable content.
        path: Output JSON path.

    Returns:
        None.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)


def suffixed_output_path(output_dir: str, filename: str, suffix: str = "") -> str:
    """Build an output path with an optional stem suffix.

    Args:
        output_dir: Directory where evaluation files are written.
        filename: Base filename, including extension.
        suffix: Optional suffix appended before the extension.

    Returns:
        Output path inside ``output_dir``.
    """
    stem, ext = os.path.splitext(filename)
    if suffix:
        filename = f"{stem}_{suffix}{ext}"
    return os.path.join(output_dir, filename)


def canonical_object_id(object_name: Any) -> str:
    """Normalize an object id by dropping common sequence suffixes.

    Args:
        object_name: Raw object name or path-like id.

    Returns:
        Canonical object id.
    """
    text = str(object_name).strip().replace("\\", "/").strip("/")
    parts = [part for part in text.split("/") if part]
    base = parts[-2] if len(parts) >= 2 and re.match(r"^seq[_-]?\d+.*$", parts[-1]) else os.path.basename(text)
    return re.sub(r"([_-]seq[_-]?\d+.*)$", "", base)


def normalize_sequence_id(sequence_id: Any) -> str:
    """Normalize a sequence id into a stable path-like string.

    Args:
        sequence_id: Raw sequence id.

    Returns:
        Normalized sequence id, or an empty string.
    """
    if sequence_id is None:
        return ""
    return str(sequence_id).strip().replace("\\", "/").strip("/")


def scene_key(object_id: Any, sequence_or_scene_id: Any) -> str:
    """Build a stable scene key used for score-label joins.

    Args:
        object_id: Canonical object id.
        sequence_or_scene_id: Sequence id or scene id.

    Returns:
        ``object_id/sequence`` when a sequence exists, otherwise ``object_id``.
    """
    object_id = canonical_object_id(object_id)
    scene_id = normalize_sequence_id(sequence_or_scene_id)
    if not scene_id:
        return object_id
    if scene_id == object_id or scene_id.startswith(f"{object_id}/"):
        return scene_id
    return f"{object_id}/{os.path.basename(scene_id)}"


def sequence_id_from_grasp(grasp_data: dict, grasp_path: str = "") -> str:
    """Read sequence id from one formatted human grasp.

    Args:
        grasp_data: Loaded grasp dictionary.
        grasp_path: Optional fallback path.

    Returns:
        Sequence id such as ``seq_3``.
    """
    object_data = grasp_data.get("object", {})
    sequence_id = object_data.get("sequence_id")
    if sequence_id is not None:
        return normalize_sequence_id(sequence_id)
    source_scene = object_data.get("source_scene")
    if source_scene is not None:
        match = re.search(r"(seq[_-]?\d+.*)$", str(source_scene))
        if match:
            return normalize_sequence_id(match.group(1))
    if grasp_path:
        return os.path.splitext(os.path.basename(grasp_path))[0]
    return ""


def human_grasp_type_id(grasp_data: dict) -> int:
    """Infer the five-way human grasp type id for one record.

    Args:
        grasp_data: Loaded human grasp dictionary.

    Returns:
        Grasp type id in ``1..5``.
    """
    hand_data = grasp_data.get("hand", {})
    left = hand_data.get("left")
    right = hand_data.get("right")
    left_contacts = left.get("contacts", [False] * 5) if left else [False] * 5
    right_contacts = right.get("contacts", [False] * 5) if right else [False] * 5
    has_left = any(left_contacts)
    has_right = any(right_contacts)
    left_count = int(sum(left_contacts))
    right_count = int(sum(right_contacts))
    if not (has_left or has_right):
        raise ValueError("Human grasp record has no active contacts.")
    if has_left and has_right:
        return 5 if (left_count > 3 or right_count > 3) else 4
    active_count = left_count if has_left else right_count
    if active_count <= 2:
        return 1
    if active_count == 3:
        return 2
    return 3


def object_pose_matrix(grasp_data: dict) -> np.ndarray:
    """Read object pose matrix from a human grasp record.

    Args:
        grasp_data: Loaded human grasp dictionary.

    Returns:
        Homogeneous object pose. Missing poses fall back to identity.
    """
    pose = grasp_data.get("object", {}).get("pose")
    if pose is None:
        return np.eye(4, dtype=np.float64)
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"Unsupported object pose shape: {pose.shape}")
    return pose


def scale_points(points: np.ndarray, scale: Any) -> np.ndarray:
    """Apply scalar or XYZ object scale to point coordinates.

    Args:
        points: Point cloud shaped ``(N, 3)``.
        scale: Scalar or three-dimensional scale.

    Returns:
        Scaled point cloud.
    """
    scale_array = np.asarray(scale, dtype=np.float64).reshape(-1)
    if scale_array.size == 0:
        return points
    if scale_array.size == 1:
        return points * float(scale_array[0])
    if scale_array.size == 3:
        return points * scale_array.reshape(1, 3)
    raise ValueError(f"Unsupported object scale shape: {scale_array.shape}")


def transform_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Transform object-frame points into world frame.

    Args:
        points: Point cloud shaped ``(N, 3)``.
        pose: Homogeneous transform shaped ``(4, 4)``.

    Returns:
        Transformed point cloud.
    """
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4):
        return points
    return points @ pose[:3, :3].T + pose[:3, 3].reshape(1, 3)


def line_like_xy_extents(xy: np.ndarray) -> tuple[float, float]:
    """Compute XY long/short extents for degenerate point sets.

    Args:
        xy: Centered XY points shaped ``(N, 2)``.

    Returns:
        Tuple ``(xy_long, xy_short)``.
    """
    if xy.shape[0] < 2:
        return EPS, EPS
    _, singular_values, vh = np.linalg.svd(xy, full_matrices=False)
    if singular_values.size == 0 or float(singular_values[0]) < EPS:
        return EPS, EPS
    primary = vh[0]
    secondary = np.asarray([-primary[1], primary[0]], dtype=np.float64)
    extents = np.maximum(np.ptp(xy @ np.stack([primary, secondary], axis=1), axis=0), EPS)
    long_value, short_value = np.sort(extents)[::-1]
    return float(long_value), float(short_value)


def yaw_free_xy_extents(points: np.ndarray) -> tuple[float, float]:
    """Compute yaw-invariant XY bounding-box long and short sides.

    Args:
        points: Point cloud shaped ``(N, 3)`` or XY array shaped ``(N, 2)``.

    Returns:
        Tuple ``(xy_long, xy_short)``.
    """
    xy = np.asarray(points, dtype=np.float64)
    if xy.ndim != 2:
        raise ValueError(f"Expected points with two dimensions, got {xy.shape}")
    if xy.shape[1] == 3:
        xy = xy[:, :2]
    if xy.shape[1] != 2:
        raise ValueError(f"Expected XY coordinates, got {xy.shape}")
    xy = xy - xy.mean(axis=0, keepdims=True)
    if xy.shape[0] < 3 or np.linalg.matrix_rank(xy, tol=EPS) < 2:
        return line_like_xy_extents(xy)
    try:
        hull_points = xy[ConvexHull(xy).vertices]
    except QhullError:
        return line_like_xy_extents(xy)
    edges = np.roll(hull_points, shift=-1, axis=0) - hull_points
    edge_lengths = np.linalg.norm(edges, axis=1)
    edges = edges[edge_lengths > EPS]
    if edges.size == 0:
        return line_like_xy_extents(xy)
    angles = np.unique(np.round(np.mod(np.arctan2(edges[:, 1], edges[:, 0]), math.pi / 2.0), decimals=12))
    best_area = float("inf")
    best_extents = None
    for angle in angles:
        c, s = math.cos(float(angle)), math.sin(float(angle))
        axes = np.asarray([[c, -s], [s, c]], dtype=np.float64)
        extents = np.maximum(np.ptp(hull_points @ axes, axis=0), EPS)
        area = float(extents[0] * extents[1])
        if area < best_area:
            best_area = area
            best_extents = extents
    if best_extents is None:
        return line_like_xy_extents(xy)
    long_value, short_value = np.sort(best_extents)[::-1]
    return float(long_value), float(short_value)


def scale_descriptor(points: np.ndarray) -> np.ndarray:
    """Compute the three-dimensional scale descriptor.

    Args:
        points: Canonical or posed point cloud shaped ``(N, 3)``.

    Returns:
        Array ``[xy_long, xy_short, z_height]``.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected point cloud shape (N, 3), got {points.shape}")
    xy_long, xy_short = yaw_free_xy_extents(points)
    z_height = float(max(np.ptp(points[:, 2]), EPS))
    return np.asarray([xy_long, xy_short, z_height], dtype=np.float64)


def object_scale_half_diagonal(points: np.ndarray) -> float:
    """Compute object scale as half of the axis-aligned bounding-box diagonal.

    Args:
        points: Scaled object-frame point cloud shaped ``(N, 3)``.

    Returns:
        Half of the AABB diagonal length.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected point cloud shape (N, 3), got {points.shape}")
    extents = np.maximum(np.ptp(points, axis=0), EPS)
    return float(0.5 * np.linalg.norm(extents))


def load_point_cloud_descriptor(pc_path: str, max_points: int, cache: dict) -> np.ndarray:
    """Load a point cloud and compute its canonical scale descriptor.

    Args:
        pc_path: Point-cloud ``.npy`` path.
        max_points: Deterministic cap for descriptor computation.
        cache: Mutable path-to-descriptor cache.

    Returns:
        Three-dimensional descriptor ``[xy_long, xy_short, z_height]``.
    """
    pc_path = _abs_path(pc_path)
    if pc_path in cache:
        return cache[pc_path]
    points = np.asarray(np.load(pc_path, allow_pickle=True), dtype=np.float64).reshape(-1, 3)
    if max_points > 0 and points.shape[0] > max_points:
        indices = np.linspace(0, points.shape[0] - 1, num=max_points, dtype=np.int64)
        points = points[indices]
    descriptor = scale_descriptor(points)
    cache[pc_path] = descriptor
    return descriptor


def normalize_score_vector(scores: Any) -> np.ndarray:
    """Convert a five-dimensional score vector into a probability distribution.

    Args:
        scores: Real-type scores or probabilities for types ``1..5``.

    Returns:
        Probability vector summing to one.
    """
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == len(REAL_TYPE_IDS) + 1:
        values = values[1:]
    if values.size != len(REAL_TYPE_IDS):
        raise ValueError(f"Expected {len(REAL_TYPE_IDS)} real-type scores, got {values.shape}")
    if np.all(np.isfinite(values)) and float(values.min()) >= 0.0 and float(values.sum()) > EPS:
        return values / float(values.sum())
    shifted = values - float(np.max(values))
    exp_values = np.exp(shifted)
    return exp_values / max(float(exp_values.sum()), EPS)


def rankdata(values: np.ndarray) -> np.ndarray:
    """Compute average ranks for a small numeric vector.

    Args:
        values: Numeric vector.

    Returns:
        Average ranks starting from one.
    """
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    """Compute Pearson correlation with constant-input handling.

    Args:
        x: First vector.
        y: Second vector.

    Returns:
        Correlation coefficient or ``None``.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2 or float(np.std(x)) < EPS or float(np.std(y)) < EPS:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    """Compute Spearman rank correlation.

    Args:
        x: First vector.
        y: Second vector.

    Returns:
        Spearman coefficient or ``None``.
    """
    return pearson_corr(rankdata(np.asarray(x, dtype=np.float64)), rankdata(np.asarray(y, dtype=np.float64)))


def kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float | None:
    """Compute Kendall tau-b for five-type rank agreement.

    Args:
        x: First ranking score vector.
        y: Second ranking score vector.

    Returns:
        Kendall tau-b or ``None`` when all pairs are tied.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    concordant = discordant = tie_x = tie_y = 0
    for i in range(x.size):
        for j in range(i + 1, x.size):
            dx = np.sign(x[i] - x[j])
            dy = np.sign(y[i] - y[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tie_x += 1
            elif dy == 0:
                tie_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + tie_x) * (concordant + discordant + tie_y))
    if denom <= EPS:
        return None
    return float((concordant - discordant) / denom)


def top_k_indices(values: np.ndarray, k: int, type_ids: tuple[int, ...] = REAL_TYPE_IDS) -> set[int]:
    """Return the ids of the top-k real types.

    Args:
        values: Score vector aligned with ``type_ids``.
        k: Number of entries to select.
        type_ids: Active real grasp type ids.

    Returns:
        Set of selected type ids.
    """
    order = np.argsort(-np.asarray(values, dtype=np.float64), kind="mergesort")
    return {int(type_ids[idx]) for idx in order[: min(k, len(order))]}


def positive_set_recall_at_positive_num(
    q: np.ndarray,
    p: np.ndarray,
    type_ids: tuple[int, ...] = REAL_TYPE_IDS,
) -> float | None:
    """Measure how many GT-positive types appear in the top-|P| predictions.

    Args:
        q: Ground-truth type distribution aligned with ``type_ids``.
        p: Predicted type distribution aligned with ``type_ids``.
        type_ids: Active real grasp type ids.

    Returns:
        Recall value in ``[0, 1]`` using ``P={t | q_t>0}`` as GT-positive
        types and the top-``|P|`` predicted types as the retrieved set.
        Returns ``None`` only when the GT-positive set is empty.
    """
    q = np.asarray(q, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    positive_type_ids = {int(type_ids[idx]) for idx, value in enumerate(q) if float(value) > 0.0}
    if not positive_type_ids:
        return None
    predicted_type_ids = top_k_indices(p, len(positive_type_ids), type_ids)
    return float(len(positive_type_ids & predicted_type_ids) / len(positive_type_ids))


def positive_set_soft_recall_at_positive_num(
    q: np.ndarray,
    p: np.ndarray,
    type_ids: tuple[int, ...] = REAL_TYPE_IDS,
) -> float | None:
    """Measure how much GT positive probability mass is covered by top-|P| types.

    Args:
        q: Ground-truth type distribution aligned with ``type_ids``.
        p: Predicted type distribution aligned with ``type_ids``.
        type_ids: Active real grasp type ids.

    Returns:
        Soft recall value in ``[0, 1]`` using ``P={t | q_t>0}`` to define the
        GT positive set and summing the GT mass covered by the predicted
        top-``|P|`` types. Returns ``None`` only when the GT-positive set is
        empty.
    """
    q = np.asarray(q, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    positive_type_num = int(np.sum(q > 0.0))
    if positive_type_num <= 0:
        return None
    predicted_type_ids = top_k_indices(p, positive_type_num, type_ids)
    return float(
        sum(
            float(q[idx])
            for idx, type_id in enumerate(type_ids)
            if int(type_id) in predicted_type_ids
        )
    )


def js_divergence(q: np.ndarray, p: np.ndarray) -> float:
    """Compute Jensen-Shannon divergence using natural logarithms.

    Args:
        q: Reference probability distribution.
        p: Predicted probability distribution.

    Returns:
        JS divergence.
    """
    q = np.asarray(q, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    m = 0.5 * (q + p)
    return float(0.5 * np.sum(q * (np.log(np.clip(q, EPS, None)) - np.log(np.clip(m, EPS, None)))) + 0.5 * np.sum(p * (np.log(np.clip(p, EPS, None)) - np.log(np.clip(m, EPS, None)))))


def ce_eps_key(eps: float) -> str:
    """Build a stable CSV key for a CE clipping epsilon.

    Args:
        eps: Probability floor used before the logarithm.

    Returns:
        Metric key for CE computed with this epsilon.
    """
    exponent = int(round(-math.log10(float(eps))))
    return f"soft_label_ce_eps_1e_{exponent:02d}"


def cross_entropy_with_eps(q: np.ndarray, p: np.ndarray, eps: float) -> float:
    """Compute soft-label cross entropy with a configurable probability floor.

    Args:
        q: Reference probability distribution.
        p: Predicted probability distribution.
        eps: Probability floor applied before ``log``.

    Returns:
        Cross entropy value.
    """
    q = np.asarray(q, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(-np.sum(q * np.log(np.clip(p, float(eps), None))))


def positive_set_cross_entropy(q: np.ndarray, p: np.ndarray, eps: float = EPS) -> float | None:
    """Compute CE inside the human-observed positive type set.

    The metric normalizes ``p`` only over types with ``q_t > 0``. It should be
    read together with soft precision because it evaluates the within-positive
    allocation but intentionally ignores mass assigned outside the positive set.

    Args:
        q: Reference probability distribution.
        p: Predicted probability distribution.
        eps: Probability floor applied before positive-set normalization.

    Returns:
        Positive-set cross entropy, or ``None`` when the positive set is empty.
    """
    q = np.asarray(q, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    positive_mask = q > 0.0
    if not np.any(positive_mask):
        return None
    q_pos = q[positive_mask]
    q_pos = q_pos / max(float(q_pos.sum()), float(eps))
    p_pos = np.clip(p[positive_mask], float(eps), None)
    p_pos = p_pos / max(float(p_pos.sum()), float(eps))
    return cross_entropy_with_eps(q_pos, p_pos, eps)


def metric_mean(rows: list[dict], key: str) -> float | None:
    """Average a metric key while ignoring missing values.

    Args:
        rows: Metric rows.
        key: Metric key.

    Returns:
        Mean value or ``None``.
    """
    values = [float(row[key]) for row in rows if row.get(key) is not None and row.get(key) != ""]
    if not values:
        return None
    return float(np.mean(values))


def format_float(value: Any, digits: int = 4) -> str:
    """Format optional floats for markdown.

    Args:
        value: Numeric value or ``None``.
        digits: Decimal digits.

    Returns:
        Formatted value or ``NA``.
    """
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


class MarkdownReport:
    """Incremental Chinese markdown report writer."""

    def __init__(self, path: str, title: str):
        """Create the report writer.

        Args:
            path: Markdown output path.
            title: Report title.

        Returns:
            None.
        """
        self.path = path
        self.title = title
        self.sections: list[tuple[str, list[str]]] = []

    def add_section(self, title: str, lines: list[str]) -> None:
        """Append and flush one report section.

        Args:
            title: Section title.
            lines: Section lines.

        Returns:
            None.
        """
        self.sections.append((title, lines))
        self.write()

    def write(self) -> None:
        """Write the full markdown report.

        Args:
            None.

        Returns:
            None.
        """
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(f"# {self.title}\n\n")
            for title, lines in self.sections:
                f.write(f"## {title}\n\n")
                for line in lines:
                    f.write(f"{line}\n")
                f.write("\n")


def checkpoint_step_name(config) -> str | None:
    """Resolve the step directory name used by saved outputs.

    Args:
        config: Full Hydra config.

    Returns:
        Step directory name such as ``step_010000``, or ``None``.
    """
    ckpt = getattr(config, "ckpt", None)
    if ckpt is None:
        return None
    text = os.path.basename(str(ckpt))
    match = re.match(r"step_(\d+)(?:\.pth)?$", text)
    if match:
        return f"step_{match.group(1)}"
    if text.isdigit():
        return f"step_{text.zfill(6)}"
    return os.path.splitext(text)[0]


def default_tests_step_dir(config) -> str:
    """Find the default ``tests/step_*`` directory for this run.

    Args:
        config: Full Hydra config.

    Returns:
        Step directory under ``output/<wandb.id>/tests`` when resolvable,
        otherwise an empty string.
    """
    base_dir = _abs_path(os.path.join(str(config.output_folder), str(config.wandb.id), "tests"))
    step_name = checkpoint_step_name(config)
    if step_name:
        candidate = os.path.join(base_dir, step_name)
        if os.path.isdir(candidate):
            return candidate
    candidates = sorted(glob(os.path.join(base_dir, "step_*")), key=_natural_sort_key)
    candidates = [path for path in candidates if os.path.isdir(path)]
    return candidates[-1] if candidates else ""


def default_test_result_dir(config, dataset_name: str) -> str:
    """Find the default saved sample directory for one test dataset.

    Args:
        config: Full Hydra config.
        dataset_name: Dataset folder name under ``tests/step_*``.

    Returns:
        Directory path such as ``tests/step_010000/DGNMulti``.
    """
    step_dir = default_tests_step_dir(config)
    if not step_dir or not dataset_name:
        return ""
    return os.path.join(step_dir, str(dataset_name))


def resolve_test_result_dir(config, field_name: str, dataset_name: str) -> str:
    """Resolve a configured or default saved sample directory.

    Args:
        config: Full Hydra config.
        field_name: Task config field name such as ``human_results_dir``.
        dataset_name: Dataset folder name used by the default path.

    Returns:
        Absolute result directory path, or an empty string.
    """
    value = getattr(config.task, field_name, "")
    if value:
        return _abs_path(value)
    return default_test_result_dir(config, dataset_name)


def default_score_jsonl(config) -> str:
    """Find the default obj_human_prior score JSONL for this run.

    Args:
        config: Full Hydra config.

    Returns:
        Path to ``scene_budget_scores.jsonl`` when resolvable, otherwise empty.
    """
    base_dir = _abs_path(os.path.join(str(config.output_folder), str(config.wandb.id), "obj_human_prior"))
    step_name = checkpoint_step_name(config)
    if step_name:
        candidate = os.path.join(base_dir, step_name, "scene_budget_scores.jsonl")
        if os.path.isfile(candidate):
            return candidate
    candidates = sorted(glob(os.path.join(base_dir, "step_*", "scene_budget_scores.jsonl")), key=_natural_sort_key)
    return candidates[-1] if candidates else ""


def resolve_score_jsonl(config, field_name: str, fallback: str = "") -> str:
    """Resolve a configured score JSONL path.

    Args:
        config: Full Hydra config.
        field_name: Task config field name.
        fallback: Fallback path.

    Returns:
        Existing or intended score JSONL path, or an empty string.
    """
    value = getattr(config.task, field_name, "")
    if value:
        return _abs_path(value)
    if fallback:
        return fallback
    return ""


def _path_after_marker(path: str, marker: str) -> str:
    """Return the path segment after a named directory marker.

    Args:
        path: Path-like string.
        marker: Directory marker such as ``scene_cfg``.

    Returns:
        Relative path after the marker, without a file extension when present.
    """
    parts = [part for part in str(path).replace("\\", "/").split("/") if part]
    if marker not in parts:
        return ""
    index = parts.index(marker)
    rel = "/".join(parts[index + 1 :])
    return os.path.splitext(rel)[0]


def scene_parts_from_result(sample: dict, sample_path: str, results_dir: str) -> tuple[str, str]:
    """Infer canonical object id and scene id from one saved test result.

    Args:
        sample: Loaded saved-sample dictionary.
        sample_path: Saved ``.npy`` result path.
        results_dir: Dataset result directory, e.g. ``tests/step_010000/DGNMulti``.

    Returns:
        Tuple ``(object_id, scene_id)`` suitable for ``scene_key``.
    """
    scene_path = str(sample.get("scene_path", ""))
    rel_scene = _path_after_marker(scene_path, "scene_cfg")
    if rel_scene:
        parts = [part for part in rel_scene.split("/") if part]
        if parts:
            return canonical_object_id(parts[0]), "/".join(parts)

    rel = os.path.relpath(sample_path, results_dir).replace("\\", "/")
    parts = [part for part in rel.split("/") if part]
    if parts and parts[0] in GRASP_TYPES:
        parts = parts[1:]
    if len(parts) >= 2:
        return canonical_object_id(parts[0]), "/".join(parts[:-1])
    if parts:
        return canonical_object_id(parts[0]), os.path.splitext(parts[0])[0]
    return "", ""


def human_split_lookup(config) -> dict[str, str]:
    """Build an object-id to split lookup for human result rows.

    Args:
        config: Full Hydra config.

    Returns:
        Mapping from canonical object id to split name when known.
    """
    lookup: dict[str, str] = {}
    try:
        data_config = deepcopy(config.data)
        flatten_multidex_data_config(data_config)
        split_path = str(cfg_get(data_config, "split_path", "paths.split_path", default="valid_split"))
        splits = [str(split) for split in _as_list(getattr(config.task, "human_splits", ["train", "test"]))]
        for _, object_root in iter_human_roots(data_config):
            for split in splits:
                split_json = os.path.join(object_root, split_path, f"{split}.json")
                if not os.path.isfile(split_json):
                    continue
                for object_id in _load_json(split_json):
                    lookup.setdefault(canonical_object_id(object_id), split)
    except Exception as exc:
        print(f"[evaluate] Could not build human split lookup: {type(exc).__name__}: {exc}")
    return lookup


def score_result_root(results_dir: str, score_grasp_type: str) -> str:
    """Select the subdirectory containing score-bearing saved samples.

    Args:
        results_dir: Dataset result directory.
        score_grasp_type: Grasp-type folder to read, usually ``0_any``.

    Returns:
        Directory to scan recursively.
    """
    if not results_dir:
        return ""
    base_name = os.path.basename(os.path.normpath(results_dir))
    if base_name in GRASP_TYPES:
        return results_dir
    candidate = os.path.join(results_dir, str(score_grasp_type))
    return candidate if os.path.isdir(candidate) else results_dir


def load_test_result_scores(
    results_dir: str,
    max_rows: int = 0,
    split_lookup: dict[str, str] | None = None,
    score_grasp_type: str = "0_any",
) -> list[dict]:
    """Load Human Prior scores from saved ``task=sample`` test results.

    Args:
        results_dir: Dataset result directory under ``tests/step_*``.
        max_rows: Optional cap on aggregated scene rows for quick evaluation.
        split_lookup: Optional human object-id to split mapping.
        score_grasp_type: Grasp-type folder used as score source.

    Returns:
        One averaged score row per scene.
    """
    if not results_dir or not os.path.isdir(results_dir):
        return []
    root = score_result_root(results_dir, score_grasp_type)
    if not root or not os.path.isdir(root):
        return []
    sample_paths = sorted(glob(os.path.join(root, "**", "*.npy"), recursive=True), key=_natural_sort_key)
    groups: dict[tuple[str, str], list[dict]] = {}
    first_rows: dict[tuple[str, str], dict] = {}
    split_lookup = split_lookup or {}

    for sample_path in sample_paths:
        try:
            sample = np.load(sample_path, allow_pickle=True).item()
            if "pred_grasp_type_prob" not in sample:
                continue
            object_id, scene_id = scene_parts_from_result(sample, sample_path, results_dir)
            if not object_id:
                continue
            split = split_lookup.get(object_id, "")
            key_text = scene_key(object_id, scene_id)
            key = (split, key_text)
            score = normalize_score_vector(sample["pred_grasp_type_prob"])
        except Exception as exc:
            print(f"[evaluate] Skip unreadable result {sample_path}: {type(exc).__name__}: {exc}")
            continue

        groups.setdefault(key, []).append({"scores": score})
        if key not in first_rows:
            first_rows[key] = {
                "scene_id": key_text,
                "scene_key": key_text,
                "object_id": object_id,
                "split": split,
                "scene_path": str(sample.get("scene_path", "")),
                "pc_path": str(sample.get("pc_path", "")),
                "score_semantics": "saved_test_pred_grasp_type_prob",
                "source_results_dir": results_dir,
                "score_grasp_type": score_grasp_type,
            }
        if max_rows > 0 and len(groups) >= max_rows:
            break

    rows = []
    for key in sorted(groups, key=lambda item: (item[0], _natural_sort_key(item[1]))):
        score_stack = np.stack([item["scores"] for item in groups[key]], axis=0)
        rows.append(
            {
                **first_rows[key],
                "scores": score_stack.mean(axis=0),
                "score_row_count": int(score_stack.shape[0]),
            }
        )
    return rows


def load_score_jsonl(path: str, max_rows: int = 0) -> list[dict]:
    """Load obj_human_prior score rows.

    Args:
        path: ``scene_budget_scores.jsonl`` path.
        max_rows: Optional cap for quick evaluation.

    Returns:
        List of normalized score rows.
    """
    if not path or not os.path.isfile(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if max_rows > 0 and len(rows) >= max_rows:
                break
            if not line.strip():
                continue
            raw = json.loads(line)
            object_id = canonical_object_id(raw.get("object_id", ""))
            key = scene_key(object_id, raw.get("scene_id", ""))
            rows.append(
                {
                    "scene_id": str(raw.get("scene_id", "")),
                    "scene_key": key,
                    "object_id": object_id,
                    "split": str(raw.get("split", "")),
                    "scene_path": str(raw.get("scene_path", "")),
                    "pc_path": str(raw.get("pc_path", "")),
                    "scores": normalize_score_vector(raw.get("budget_scores", raw.get("scores", []))),
                    "score_semantics": str(raw.get("score_semantics", "")),
                }
            )
    return rows


def aggregate_score_rows_by_scene(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """Average repeated score rows by split and scene key.

    Args:
        rows: Score rows from ``load_score_jsonl``.

    Returns:
        Mapping ``(split, scene_key) -> score record``.
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((str(row.get("split", "")), str(row["scene_key"])), []).append(row)
    aggregated = {}
    for key, group in groups.items():
        scores = np.stack([row["scores"] for row in group], axis=0)
        aggregated[key] = {
            **group[0],
            "scores": scores.mean(axis=0),
            "score_row_count": int(len(group)),
        }
    return aggregated


def find_score_for_scene(score_by_scene: dict, split: str, scene_key_value: str) -> dict | None:
    """Find a score row for one split/scene key.

    Args:
        score_by_scene: Aggregated score mapping.
        split: Requested split.
        scene_key_value: Scene key.

    Returns:
        Score row or ``None``.
    """
    return score_by_scene.get((split, scene_key_value)) or score_by_scene.get(("", scene_key_value))


def pointcloud_matches_sequence(pc_path: str, sequence_id: str) -> bool:
    """Check whether a point-cloud path appears to match a sequence id.

    Args:
        pc_path: Candidate point-cloud path.
        sequence_id: Sequence id.

    Returns:
        True when the filename or a path component contains the sequence id.
    """
    sequence_id = normalize_sequence_id(sequence_id)
    if not sequence_id:
        return False
    base_name = os.path.splitext(os.path.basename(pc_path))[0]
    return base_name == sequence_id or base_name.startswith(f"{sequence_id}_") or sequence_id in pc_path.split(os.sep)


def select_human_pointcloud(object_root: str, pc_rel_path: str, object_id: str, sequence_id: str) -> str:
    """Select a deterministic canonical point cloud for one human scene.

    Args:
        object_root: Human object root.
        pc_rel_path: Point-cloud subdirectory relative to object root.
        object_id: Object id.
        sequence_id: Sequence id.

    Returns:
        Point-cloud path.
    """
    pc_root = os.path.join(object_root, pc_rel_path, object_id)
    candidates = sorted(glob(os.path.join(pc_root, "**", "*.npy"), recursive=True), key=_natural_sort_key)
    if not candidates:
        raise FileNotFoundError(f"No point clouds found under {pc_root}")
    matched = [path for path in candidates if pointcloud_matches_sequence(path, sequence_id)]
    return matched[0] if matched else candidates[0]


def iter_human_roots(data_config) -> list[tuple[str, str]]:
    """Resolve human grasp and object roots from the data config.

    Args:
        data_config: Hydra data config for ``HumanMultiDexDataset``.

    Returns:
        List of ``(grasp_root, object_root)`` pairs.
    """
    data_config = deepcopy(data_config)
    flatten_multidex_data_config(data_config)
    grasp_paths = [_abs_path(path) for path in _as_list(cfg_get(data_config, "grasp_path", "paths.grasp_path"))]
    object_paths = [_abs_path(path) for path in _as_list(cfg_get(data_config, "object_path", "paths.object_path"))]
    if len(grasp_paths) == 1 and len(object_paths) > 1:
        grasp_paths = grasp_paths * len(object_paths)
    if len(object_paths) == 1 and len(grasp_paths) > 1:
        object_paths = object_paths * len(grasp_paths)
    if len(grasp_paths) != len(object_paths):
        raise ValueError("grasp_path and object_path must have matching lengths")
    return list(zip(grasp_paths, object_paths))


def yaw_rotation_matrix(yaw_rad: float) -> np.ndarray:
    """Build a global-Z yaw rotation matrix.

    Args:
        yaw_rad: Yaw angle in radians.

    Returns:
        Rotation matrix shaped ``(3, 3)``.
    """
    c, s = math.cos(float(yaw_rad)), math.sin(float(yaw_rad))
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def wrap_angle_rad(angle: float) -> float:
    """Wrap an angle to ``[-pi, pi)``.

    Args:
        angle: Raw angle in radians.

    Returns:
        Wrapped angle.
    """
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


def optimal_left_yaw(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    """Find global-Z yaw that best aligns ``yaw @ rot_b`` to ``rot_a``.

    Args:
        rot_a: Target rotation matrix.
        rot_b: Source rotation matrix.

    Returns:
        Yaw angle in radians.
    """
    matrix = np.asarray(rot_b, dtype=np.float64) @ np.asarray(rot_a, dtype=np.float64).T
    return wrap_angle_rad(math.atan2(matrix[0, 1] - matrix[1, 0], matrix[0, 0] + matrix[1, 1]))


def rotation_distance_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    """Compute geodesic rotation distance in degrees.

    Args:
        rot_a: First rotation matrix.
        rot_b: Second rotation matrix.

    Returns:
        Angular distance in degrees.
    """
    relative = np.asarray(rot_a, dtype=np.float64).T @ np.asarray(rot_b, dtype=np.float64)
    return float(np.rad2deg(SciR.from_matrix(relative).magnitude()))


def pose_class_rotation_residual(scene: dict, representative: dict) -> tuple[float, float, float]:
    """Compare object poses modulo global yaw and local-Z 180-degree symmetry.

    Args:
        scene: Candidate scene dictionary.
        representative: Representative scene dictionary.

    Returns:
        Tuple ``(residual_deg, yaw_deg, local_z_symmetry_deg)``.
    """
    scene_rot = np.asarray(scene["object_pose"], dtype=np.float64)[:3, :3]
    rep_rot = np.asarray(representative["object_pose"], dtype=np.float64)[:3, :3]
    best = (float("inf"), 0.0, 0.0)
    for local_z in (0.0, math.pi):
        candidate = scene_rot @ yaw_rotation_matrix(local_z)
        yaw = optimal_left_yaw(rep_rot, candidate)
        aligned = yaw_rotation_matrix(yaw) @ candidate
        residual = rotation_distance_deg(rep_rot, aligned)
        if residual < best[0]:
            best = (float(residual), float(np.rad2deg(yaw)), float(np.rad2deg(local_z)))
    return best


def bbox_proportion_distance(scene: dict, representative: dict) -> float:
    """Compare yaw-free bbox proportions between two posed scenes.

    Args:
        scene: Candidate scene dictionary.
        representative: Representative scene dictionary.

    Returns:
        Maximum relative difference over normalized descriptor axes.
    """
    a = np.maximum(np.asarray(scene["posed_descriptor"], dtype=np.float64), EPS)
    b = np.maximum(np.asarray(representative["posed_descriptor"], dtype=np.float64), EPS)
    a = a / max(float(np.linalg.norm(a)), EPS)
    b = b / max(float(np.linalg.norm(b)), EPS)
    denominator = np.maximum(np.maximum(a, b), EPS)
    return float(np.max(np.abs(a - b) / denominator))


def assign_pose_classes(scenes: list[dict], rotation_threshold_deg: float, bbox_threshold: float) -> dict[tuple, dict]:
    """Assign human object scenes to pose classes.

    Args:
        scenes: Human scene dictionaries.
        rotation_threshold_deg: Rotation residual threshold.
        bbox_threshold: Bbox proportion fallback threshold.

    Returns:
        Mapping from ``(component_idx, split, scene_key)`` to pose-class metadata.
    """
    representatives_by_object: dict[tuple[int, str, str], list[dict]] = {}
    assignments = {}
    for scene in scenes:
        object_key = (int(scene["component_idx"]), str(scene["split"]), str(scene["object_id"]))
        representatives = representatives_by_object.setdefault(object_key, [])
        best_rotation = None
        best_bbox = None
        for pose_class_id, representative in enumerate(representatives):
            residual, yaw_deg, local_z_deg = pose_class_rotation_residual(scene, representative)
            bbox_distance = bbox_proportion_distance(scene, representative)
            candidate = {
                "pose_class_id": int(pose_class_id),
                "pose_class_rotation_residual_deg": float(residual),
                "pose_class_yaw_to_representative_deg": float(yaw_deg),
                "pose_class_local_z_symmetry_deg": float(local_z_deg),
                "pose_class_bbox_proportion_distance": float(bbox_distance),
            }
            if residual <= rotation_threshold_deg:
                if best_rotation is None or residual < best_rotation["pose_class_rotation_residual_deg"]:
                    best_rotation = candidate
            elif bbox_distance <= bbox_threshold:
                if best_bbox is None or bbox_distance < best_bbox["pose_class_bbox_proportion_distance"]:
                    best_bbox = candidate

        if best_rotation is not None:
            selected = best_rotation
            selected["pose_class_match_method"] = "rotation"
        elif best_bbox is not None:
            selected = best_bbox
            selected["pose_class_match_method"] = "bbox_proportion"
        else:
            selected = {
                "pose_class_id": int(len(representatives)),
                "pose_class_rotation_residual_deg": 0.0,
                "pose_class_yaw_to_representative_deg": 0.0,
                "pose_class_local_z_symmetry_deg": 0.0,
                "pose_class_bbox_proportion_distance": 0.0,
                "pose_class_match_method": "new",
            }
            representatives.append(scene)

        representative = representatives[int(selected["pose_class_id"])]
        assignments[(int(scene["component_idx"]), str(scene["split"]), str(scene["scene_key"]))] = {
            **selected,
            "pose_class_key": f"{scene['object_id']}/pose_{int(selected['pose_class_id']):03d}",
            "pose_class_representative_scene_key": str(representative["scene_key"]),
        }
    return assignments


def build_human_scene_table(config) -> list[dict]:
    """Build human scene records with grasp type counts and geometry.

    Args:
        config: Full Hydra config.

    Returns:
        Scene dictionaries before pose-class aggregation.
    """
    data_config = deepcopy(config.data)
    flatten_multidex_data_config(data_config)
    splits = [str(split) for split in _as_list(getattr(config.task, "human_splits", ["train", "test"]))]
    split_path = str(cfg_get(data_config, "split_path", "paths.split_path", default="valid_split"))
    pc_rel_path = str(cfg_get(data_config, "pc_path", "paths.pc_path", default="vision_data/complete_pc"))
    max_points = int(getattr(config.task, "scale_max_points", 8192))
    scene_map: dict[tuple[int, str, str], dict] = {}
    descriptor_cache: dict[tuple[str, str, str], tuple[str, np.ndarray, np.ndarray, float]] = {}

    for component_idx, (grasp_root, object_root) in enumerate(iter_human_roots(data_config)):
        for split in splits:
            split_json = os.path.join(object_root, split_path, f"{split}.json")
            if not os.path.isfile(split_json):
                print(f"[evaluate] Skip missing split file: {split_json}")
                continue
            for object_id_raw in sorted(_load_json(split_json), key=_natural_sort_key):
                object_id = canonical_object_id(object_id_raw)
                grasp_paths = sorted(glob(os.path.join(grasp_root, str(object_id_raw), "**", "*.npy"), recursive=True), key=_natural_sort_key)
                for grasp_path in grasp_paths:
                    grasp_data = np.load(grasp_path, allow_pickle=True).item()
                    sequence_id = sequence_id_from_grasp(grasp_data, grasp_path)
                    key = (component_idx, split, scene_key(object_id, sequence_id))
                    scene = scene_map.get(key)
                    if scene is None:
                        pc_path = select_human_pointcloud(object_root, pc_rel_path, str(object_id_raw), sequence_id)
                        descriptor_key = (pc_path, json.dumps(np.asarray(grasp_data.get("object", {}).get("rel_scale", 1.0)).reshape(-1).tolist()), json.dumps(object_pose_matrix(grasp_data).reshape(-1).tolist()))
                        if descriptor_key in descriptor_cache:
                            pc_path, canonical_descriptor, posed_descriptor, object_scale = descriptor_cache[descriptor_key]
                        else:
                            points = np.asarray(np.load(pc_path, allow_pickle=True), dtype=np.float64).reshape(-1, 3)
                            if max_points > 0 and points.shape[0] > max_points:
                                indices = np.linspace(0, points.shape[0] - 1, num=max_points, dtype=np.int64)
                                points = points[indices]
                            scaled = scale_points(points, grasp_data.get("object", {}).get("rel_scale", 1.0))
                            canonical_descriptor = scale_descriptor(scaled)
                            object_scale = object_scale_half_diagonal(scaled)
                            posed_descriptor = scale_descriptor(transform_points(scaled, object_pose_matrix(grasp_data)))
                            descriptor_cache[descriptor_key] = (pc_path, canonical_descriptor, posed_descriptor, object_scale)
                        scene = {
                            "component_idx": component_idx,
                            "split": split,
                            "object_id": object_id,
                            "sequence_id": sequence_id,
                            "scene_key": key[2],
                            "pc_path": pc_path,
                            "object_pose": object_pose_matrix(grasp_data),
                            "scale_descriptor": canonical_descriptor,
                            "object_scale": float(object_scale),
                            "posed_descriptor": posed_descriptor,
                            "type_counts": {type_id: 0 for type_id in REAL_TYPE_IDS},
                            "grasp_paths": [],
                        }
                        scene_map[key] = scene
                    type_id = human_grasp_type_id(grasp_data)
                    if type_id in scene["type_counts"]:
                        scene["type_counts"][type_id] += 1
                    scene["grasp_paths"].append(os.path.abspath(grasp_path))

    scenes = list(scene_map.values())
    scenes.sort(key=lambda item: (item["component_idx"], item["split"], _natural_sort_key(item["object_id"]), _natural_sort_key(item["scene_key"])))
    return scenes


def build_pose_class_labels(config, score_rows: list[dict]) -> tuple[list[dict], dict]:
    """Recompute pose-class grasp-type count labels independently.

    Args:
        config: Full Hydra config.
        score_rows: Human score rows used for joining predictions.

    Returns:
        Tuple ``(label_rows, summary)``.
    """
    scenes = build_human_scene_table(config)
    rotation_threshold = float(getattr(config.task, "pose_class_rotation_threshold_deg", 45.0))
    bbox_threshold = float(getattr(config.task, "pose_class_bbox_proportion_threshold", 0.2))
    assignments = assign_pose_classes(scenes, rotation_threshold, bbox_threshold)
    score_by_scene = aggregate_score_rows_by_scene(score_rows)

    grouped: dict[tuple[int, str, str, int], list[dict]] = {}
    for scene in scenes:
        assignment = assignments[(int(scene["component_idx"]), str(scene["split"]), str(scene["scene_key"]))]
        key = (int(scene["component_idx"]), str(scene["split"]), str(scene["object_id"]), int(assignment["pose_class_id"]))
        scene = dict(scene)
        scene.update(assignment)
        grouped.setdefault(key, []).append(scene)

    rows = []
    for key, member_scenes in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], _natural_sort_key(item[0][2]), item[0][3])):
        component_idx, split, object_id, pose_class_id = key
        counts = np.zeros(len(REAL_TYPE_IDS), dtype=np.float64)
        member_scene_keys = []
        score_vectors = []
        score_scene_keys = []
        descriptors = []
        object_scales = []
        for scene in member_scenes:
            member_scene_keys.append(scene["scene_key"])
            descriptors.append(np.asarray(scene["scale_descriptor"], dtype=np.float64))
            object_scales.append(float(scene["object_scale"]))
            for idx, type_id in enumerate(REAL_TYPE_IDS):
                counts[idx] += int(scene["type_counts"].get(type_id, 0))
            score_row = find_score_for_scene(score_by_scene, split, scene["scene_key"])
            if score_row is not None:
                score_vectors.append(np.asarray(score_row["scores"], dtype=np.float64))
                score_scene_keys.append(scene["scene_key"])
        if float(counts.sum()) <= 0.0:
            continue
        q = counts / float(counts.sum())
        p = np.mean(np.stack(score_vectors, axis=0), axis=0) if score_vectors else None
        descriptor = np.mean(np.stack(descriptors, axis=0), axis=0)
        object_scale = float(np.mean(object_scales))
        first = member_scenes[0]
        row = {
            "component_idx": int(component_idx),
            "split": split,
            "canonical_object_id": object_id,
            "pose_class_id": int(pose_class_id),
            "pose_class_key": str(first["pose_class_key"]),
            "representative_scene_key": str(first["pose_class_representative_scene_key"]),
            "member_scene_keys": ";".join(sorted(member_scene_keys, key=_natural_sort_key)),
            "scored_scene_keys": ";".join(sorted(score_scene_keys, key=_natural_sort_key)),
            "member_scene_num": int(len(member_scene_keys)),
            "scored_scene_num": int(len(score_scene_keys)),
            "record_count": int(counts.sum()),
            "object_scale": object_scale,
            "xy_long": float(descriptor[0]),
            "xy_short": float(descriptor[1]),
            "z_height": float(descriptor[2]),
            "has_score": p is not None,
        }
        for idx, type_id in enumerate(REAL_TYPE_IDS):
            row[f"type_{type_id}_count"] = int(counts[idx])
            row[f"type_{type_id}_q"] = float(q[idx])
            row[f"type_{type_id}_p"] = float(p[idx]) if p is not None else ""
        rows.append(row)
    summary = {
        "scene_num": len(scenes),
        "pose_class_num": len(rows),
        "pose_class_rotation_threshold_deg": rotation_threshold,
        "pose_class_bbox_proportion_threshold": bbox_threshold,
    }
    return rows, summary


def normalize_probability_values(values: list[float]) -> list[float] | None:
    """Normalize non-negative probability-like values.

    Args:
        values: Non-negative values.

    Returns:
        Normalized values, or ``None`` when the sum is zero.
    """
    arr = np.asarray(values, dtype=np.float64)
    total = float(arr.sum())
    if total <= EPS:
        return None
    return (arr / total).tolist()


def label_rows_without_grasp_type(label_rows: list[dict], removed_type_id: int) -> tuple[list[dict], dict]:
    """Remove one grasp type from label rows and renormalize q_t and p_t.

    Args:
        label_rows: Original five-type pose-class label rows.
        removed_type_id: Real grasp type id to remove.

    Returns:
        Tuple of transformed label rows and a summary dictionary.
    """
    active_type_ids = tuple(type_id for type_id in REAL_TYPE_IDS if int(type_id) != int(removed_type_id))
    transformed_rows = []
    removed_record_count = 0
    original_record_count = 0
    for row in label_rows:
        original_record_count += int(row.get("record_count", 0))
        removed_count = int(row.get(f"type_{removed_type_id}_count", 0))
        removed_record_count += removed_count
        counts = [int(row.get(f"type_{type_id}_count", 0)) for type_id in active_type_ids]
        q_values = normalize_probability_values(counts)
        if q_values is None:
            continue

        transformed = deepcopy(row)
        transformed["record_count_raw"] = int(row.get("record_count", 0))
        transformed[f"removed_type_{removed_type_id}_count"] = removed_count
        transformed[f"removed_type_{removed_type_id}_q_raw"] = row.get(f"type_{removed_type_id}_q", "")
        transformed[f"removed_type_{removed_type_id}_p_raw"] = row.get(f"type_{removed_type_id}_p", "")
        transformed.pop(f"type_{removed_type_id}_count", None)
        transformed.pop(f"type_{removed_type_id}_q", None)
        transformed.pop(f"type_{removed_type_id}_p", None)
        transformed["record_count"] = int(sum(counts))
        for index, type_id in enumerate(active_type_ids):
            transformed[f"type_{type_id}_count"] = int(counts[index])
            transformed[f"type_{type_id}_q"] = float(q_values[index])

        if transformed.get("has_score", False):
            p_values = normalize_probability_values([float(row.get(f"type_{type_id}_p", 0.0) or 0.0) for type_id in active_type_ids])
            if p_values is None:
                transformed["has_score"] = False
                for type_id in active_type_ids:
                    transformed[f"type_{type_id}_p"] = ""
            else:
                for index, type_id in enumerate(active_type_ids):
                    transformed[f"type_{type_id}_p"] = float(p_values[index])
        transformed_rows.append(transformed)

    return transformed_rows, {
        "removed_type_id": int(removed_type_id),
        "removed_type_name": GRASP_TYPES[int(removed_type_id)],
        "active_type_ids": list(active_type_ids),
        "active_type_names": [GRASP_TYPES[type_id] for type_id in active_type_ids],
        "original_pose_class_num": len(label_rows),
        "kept_pose_class_num": len(transformed_rows),
        "dropped_pose_class_num": len(label_rows) - len(transformed_rows),
        "original_record_count": int(original_record_count),
        "kept_record_count": int(sum(int(row.get("record_count", 0)) for row in transformed_rows)),
        "removed_record_count": int(removed_record_count),
    }


def metric_row_for_pose_class(row: dict, active_type_ids: tuple[int, ...] = REAL_TYPE_IDS) -> dict | None:
    """Compute one human labeled metric row.

    Args:
        row: Pose-class label row with q and p values.
        active_type_ids: Real grasp type ids included in this metric row.

    Returns:
        Metric row, or ``None`` when prediction is missing.
    """
    if not row.get("has_score", False):
        return None
    q = np.asarray([float(row[f"type_{type_id}_q"]) for type_id in active_type_ids], dtype=np.float64)
    p = np.asarray([float(row[f"type_{type_id}_p"]) for type_id in active_type_ids], dtype=np.float64)
    positive_mask = q > 0.0
    ce = cross_entropy_with_eps(q, p, EPS)
    entropy = float(-np.sum(q * np.log(np.clip(q, EPS, None))))
    l1 = float(np.sum(np.abs(p - q)))
    soft_precision = float(p[positive_mask].sum())
    hard_recall = positive_set_recall_at_positive_num(q, p, active_type_ids)
    soft_recall = positive_set_soft_recall_at_positive_num(q, p, active_type_ids)
    p_top1 = top_k_indices(p, 1, active_type_ids)
    q_top1 = top_k_indices(q, 1, active_type_ids)
    p_top2 = top_k_indices(p, 2, active_type_ids)
    q_top2 = top_k_indices(q, 2, active_type_ids)
    out = {
        "split": row["split"],
        "canonical_object_id": row["canonical_object_id"],
        "pose_class_key": row["pose_class_key"],
        "record_count": int(row["record_count"]),
        "positive_type_num": int(positive_mask.sum()),
        "object_scale": float(row.get("object_scale", 0.5 * np.linalg.norm([float(row["xy_long"]), float(row["xy_short"]), float(row["z_height"])]))),
        "xy_long": float(row["xy_long"]),
        "xy_short": float(row["xy_short"]),
        "z_height": float(row["z_height"]),
        "soft_label_ce": ce,
        "kl_q_p": float(ce - entropy),
        "positive_set_ce": positive_set_cross_entropy(q, p, EPS),
        "distribution_l1": l1,
        "tvd": float(0.5 * l1),
        "js_divergence": js_divergence(q, p),
        "soft_precision": soft_precision,
        "soft_recall_at_positive_num": soft_recall,
        # Keep legacy aliases in CSV/report output so older notes and scripts
        # keep working while the main report now highlights SoftPrecision and
        # SoftRecall@card(P) as the primary metrics.
        "positive_probability_mass": soft_precision,
        "positive_set_recall_at_positive_num": hard_recall,
        "spearman": spearman_corr(q, p),
        "kendall": kendall_tau_b(q, p),
        "top1_match": float(bool(p_top1 & q_top1)),
        "top2_overlap": float(len(p_top2 & q_top2) / 2.0),
    }
    for eps in CE_EPS_VALUES:
        out[ce_eps_key(eps)] = cross_entropy_with_eps(q, p, eps)
    for idx, type_id in enumerate(active_type_ids):
        out[f"type_{type_id}_q"] = float(q[idx])
        out[f"type_{type_id}_p"] = float(p[idx])
        out[f"type_{type_id}_bias"] = float(p[idx] - q[idx])
        out[f"type_{type_id}_under"] = float(max(q[idx] - p[idx], 0.0))
        out[f"type_{type_id}_over"] = float(max(p[idx] - q[idx], 0.0))
    return out


def quantile_edges(values: list[float], bucket_count: int) -> np.ndarray:
    """Compute unique quantile bucket edges.

    Args:
        values: Numeric values.
        bucket_count: Desired number of buckets.

    Returns:
        Edge array including min and max.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.asarray([], dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, max(2, int(bucket_count) + 1))
    edges = np.unique(np.quantile(arr, quantiles))
    if edges.size == 1:
        edges = np.asarray([edges[0], edges[0]], dtype=np.float64)
    return edges


def bucket_label(value: float, edges: np.ndarray) -> str:
    """Assign a value to a quantile bucket.

    Args:
        value: Numeric value.
        edges: Quantile edge array.

    Returns:
        Bucket label.
    """
    if edges.size < 2 or not np.isfinite(value):
        return "all"
    idx = int(np.searchsorted(edges[1:-1], float(value), side="right"))
    lo = float(edges[idx])
    hi = float(edges[idx + 1])
    return f"q{idx + 1}[{lo:.4g},{hi:.4g}]"


def add_scale_buckets(rows: list[dict], bucket_count: int) -> dict[str, np.ndarray]:
    """Add per-axis quantile scale bucket labels to rows.

    Args:
        rows: Rows containing scale descriptor fields.
        bucket_count: Number of quantile buckets.

    Returns:
        Mapping from feature name to bucket edges.
    """
    edges_by_feature = {
        feature: quantile_edges([float(row[feature]) for row in rows], bucket_count)
        for feature in SCALE_FEATURES
    }
    for row in rows:
        for feature, edges in edges_by_feature.items():
            row[f"{feature}_bucket"] = bucket_label(float(row[feature]), edges)
    return edges_by_feature


def summarize_metric_rows(rows: list[dict], group_name: str, group_value: str, active_type_ids: tuple[int, ...] = REAL_TYPE_IDS) -> dict:
    """Summarize human metric rows for one group.

    Args:
        rows: Metric rows.
        group_name: Grouping field name.
        group_value: Grouping value.
        active_type_ids: Real grasp type ids included in this summary.

    Returns:
        Summary row.
    """
    summary = {
        "group_name": group_name,
        "group_value": group_value,
        "pose_class_num": int(len(rows)),
        "object_num": int(len({row["canonical_object_id"] for row in rows})),
    }
    for metric in [
        "soft_label_ce",
        "kl_q_p",
        "positive_set_ce",
        *[ce_eps_key(eps) for eps in CE_EPS_VALUES],
        "distribution_l1",
        "tvd",
        "js_divergence",
        "soft_precision",
        "soft_recall_at_positive_num",
        "positive_probability_mass",
        "positive_set_recall_at_positive_num",
        "spearman",
        "kendall",
        "top1_match",
        "top2_overlap",
    ]:
        summary[metric] = metric_mean(rows, metric)
    for type_id in active_type_ids:
        summary[f"type_{type_id}_signed_bias"] = metric_mean(rows, f"type_{type_id}_bias")
        summary[f"type_{type_id}_mean_under"] = metric_mean(rows, f"type_{type_id}_under")
        summary[f"type_{type_id}_mean_over"] = metric_mean(rows, f"type_{type_id}_over")
    return summary


def object_macro_summary(rows: list[dict], split: str, active_type_ids: tuple[int, ...] = REAL_TYPE_IDS) -> dict:
    """Compute canonical-object macro averages for one split.

    Args:
        rows: Metric rows.
        split: Split label.
        active_type_ids: Real grasp type ids included in this summary.

    Returns:
        Summary row.
    """
    object_rows = []
    for object_id in sorted({row["canonical_object_id"] for row in rows}, key=_natural_sort_key):
        object_metric_rows = [row for row in rows if row["canonical_object_id"] == object_id]
        object_rows.append(summarize_metric_rows(object_metric_rows, "object", object_id, active_type_ids))
    summary = {"group_name": "object_macro", "group_value": split, "pose_class_num": len(rows), "object_num": len(object_rows)}
    for key in object_rows[0].keys() if object_rows else []:
        if key in {"group_name", "group_value", "pose_class_num", "object_num"}:
            continue
        values = [row[key] for row in object_rows if row.get(key) is not None]
        summary[key] = float(np.mean(values)) if values else None
    return summary


def build_human_metric_summaries(metric_rows: list[dict], active_type_ids: tuple[int, ...] = REAL_TYPE_IDS) -> list[dict]:
    """Build required human train/test aggregate summaries.

    Args:
        metric_rows: Per-pose-class metric rows.
        active_type_ids: Real grasp type ids included in this summary.

    Returns:
        Summary rows.
    """
    summaries = []
    for split in sorted({row["split"] for row in metric_rows}):
        split_rows = [row for row in metric_rows if row["split"] == split]
        summaries.append(summarize_metric_rows(split_rows, "scene_micro", split, active_type_ids))
        if split_rows:
            summaries.append(object_macro_summary(split_rows, split, active_type_ids))
        for positive_num in sorted({row["positive_type_num"] for row in split_rows}):
            rows = [row for row in split_rows if row["positive_type_num"] == positive_num]
            summaries.append(summarize_metric_rows(rows, "|P(scene)|", str(positive_num), active_type_ids))
        for feature in SCALE_FEATURES:
            bucket_key = f"{feature}_bucket"
            for bucket in sorted({row[bucket_key] for row in split_rows}):
                rows = [row for row in split_rows if row[bucket_key] == bucket]
                summary = summarize_metric_rows(rows, bucket_key, f"{split}:{bucket}", active_type_ids)
                summaries.append(summary)
    return summaries


def train_test_gap_rows(summaries: list[dict]) -> list[dict]:
    """Build train-test gap rows for comparable summary groups.

    Args:
        summaries: Human summary rows.

    Returns:
        Gap rows where value equals ``test - train``.
    """
    by_key = {(row["group_name"], row["group_value"]): row for row in summaries}
    gaps = []
    for row in summaries:
        group_name = row["group_name"]
        value = str(row["group_value"])
        if not value.startswith("test") and value != "test":
            continue
        train_value = value.replace("test", "train", 1)
        train_row = by_key.get((group_name, train_value))
        if train_row is None:
            continue
        gap = {"group_name": f"{group_name}_gap", "group_value": value, "pose_class_num": row["pose_class_num"], "object_num": row["object_num"]}
        for key, value_item in row.items():
            if key in gap or key in {"group_name", "group_value", "pose_class_num", "object_num"}:
                continue
            if value_item is None or train_row.get(key) is None:
                gap[key] = None
            else:
                gap[key] = float(value_item) - float(train_row[key])
        gaps.append(gap)
    return gaps


def human_report_lines(
    label_rows: list[dict],
    metric_rows: list[dict],
    summary_rows: list[dict],
    output_dir: str,
    active_type_ids: tuple[int, ...] = REAL_TYPE_IDS,
    output_suffix: str = "",
    label_summary: dict | None = None,
) -> list[str]:
    """Format the human 1A markdown section.

    Args:
        label_rows: Pose-class label rows.
        metric_rows: Per-pose-class metric rows.
        summary_rows: Aggregate metric rows.
        output_dir: Directory containing CSV outputs.
        active_type_ids: Real grasp type ids included in this report.
        output_suffix: Optional suffix appended to referenced output files.
        label_summary: Optional label transformation summary.

    Returns:
        Markdown lines.
    """
    active_type_names = ", ".join(GRASP_TYPES[type_id] for type_id in active_type_ids)
    lines = [
        "- 本节实现 `Evaluation Metrics 1 / 1A`。label 由本 evaluator 从 human train/test grasp records 重新计算；没有读取或复用 `scene_budget.py` 的 `hierarchy_count` 输出。",
        f"- Active grasp types: `{active_type_names}`。",
        "- `q_t=count_t/sum_t count_t` 是 pose-class 内 human observed grasp type 分布，只用于 evaluation；`p_t` 是 Human Prior score 归一化后的 active-type 分布。",
        f"- Pose-class label rows: {len(label_rows)}；有 score 可评估的 rows: {len(metric_rows)}。",
        f"- 明细 CSV: `{suffixed_output_path(output_dir, 'evaluation_human_pose_class_labels.csv', output_suffix)}`",
        f"- per-scene metric CSV: `{suffixed_output_path(output_dir, 'evaluation_human_pose_class_metrics.csv', output_suffix)}`",
        f"- aggregate CSV: `{suffixed_output_path(output_dir, 'evaluation_human_metric_summary.csv', output_suffix)}`",
        "",
        "### 指标解释",
        "- `KL(q||p)`：`CE-H(q)`，去掉 label 自身 entropy 后的分布偏差；越低越好。",
        "- `Soft Precision`：`sum_{t in P(scene)} p_t`，其中 `P(scene)={t | q_t>0}`；衡量预测概率有多少落在 GT positive types 上，越高越好。它等价于此前 report 里的 `PosMass`。",
        "- `Soft Recall@card(P)`：令 `P={t | q_t>0}`，取 `p_t` 最高的前 `|P|` 个 type，并把这些 type 覆盖到的 GT 概率质量相加，即 `sum_{t in Top-|P|(p)} q_t`；越高越好。",
        "- `Soft-label CE / NLL`：`-sum_t q_t log(p_t)`，衡量模型给 human observed 分布的 likelihood；越低越好。",
        "- `Positive-set CE`：只在 `q_t>0` 的 human observed positive types 内重新归一化 `p_t` 后计算 CE，衡量 positive set 内部的概率分配；越低越好，需和 SoftPrecision 一起解读。",
        "- `CE@eps`：用不同 `eps` 对 `p_t` 做 log 前下限裁剪后计算 CE，用于诊断平均 CE 对 `p_t≈0` 的敏感性；越低越好。",
        "- `Distribution L1 / TVD`：`sum_t |p_t-q_t|` 和其一半，直观表示概率质量错配量；越低越好。",
        "- `JS divergence`：对称、有界的分布距离，对 `p_t` 很小的情况比 KL 更稳定；越低越好。",
        "- `Positive probability mass`：`sum_{t in P(scene)} p_t`，衡量模型把多少概率放到 human observed positive types 上；越高越好。",
        "- `Positive-set Recall@|P|`：令 `P={t | q_t>0}`，取 `p_t` 最高的前 `|P|` 个 type，统计其中覆盖了多少 GT positive types；越高越好，更接近“数据里出现过的 type 是否都被提出来”。",
        "- `Per-type signed bias / under / over`：分别统计 `p_t-q_t`、`max(q_t-p_t,0)`、`max(p_t-q_t,0)`，用于发现某一类是否系统性低估或高估。",
        "- `Rank agreement`：Spearman、Kendall、top-1 match、top-2 overlap，衡量 active types 排序是否接近 `q_t`，对 top-heavy allocator 更敏感。",
        "",
        "### Train/Test 主汇总",
        "| split | avg | N | KL | SoftPrec | SoftRec@card(P) | CE | PosCE | CE@1e-6 | CE@1e-3 | L1 | TVD | JS | PosMass | PosR@card(P) | Spearman | Kendall | Top1 | Top2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if label_summary and label_summary.get("removed_type_name"):
        lines[2:2] = [
            f"- 移除 grasp type: `{label_summary['removed_type_name']}`；剩余 `q_t` 和 `p_t` 已重新归一化。",
            f"- 原 pose-class rows: {label_summary['original_pose_class_num']}；保留 rows: {label_summary['kept_pose_class_num']}；丢弃 rows: {label_summary['dropped_pose_class_num']}。",
            f"- 原 human grasp record_count: {label_summary['original_record_count']}；移除后 record_count: {label_summary['kept_record_count']}；移除 count: {label_summary['removed_record_count']}。",
        ]
    wanted = [row for row in summary_rows if row["group_name"] in {"scene_micro", "object_macro"} and row["group_value"] in {"train", "test"}]
    for row in wanted:
        lines.append(
            f"| {row['group_value']} | {row['group_name']} | {row['pose_class_num']} | "
            f"{format_float(row.get('kl_q_p'))} | {format_float(row.get('soft_precision'))} | "
            f"{format_float(row.get('soft_recall_at_positive_num'))} | "
            f"{format_float(row.get('soft_label_ce'))} | "
            f"{format_float(row.get('positive_set_ce'))} | {format_float(row.get(ce_eps_key(1e-6)))} | "
            f"{format_float(row.get(ce_eps_key(1e-3)))} | "
            f"{format_float(row.get('distribution_l1'))} | {format_float(row.get('tvd'))} | "
            f"{format_float(row.get('js_divergence'))} | {format_float(row.get('positive_probability_mass'))} | "
            f"{format_float(row.get('positive_set_recall_at_positive_num'))} | "
            f"{format_float(row.get('spearman'))} | {format_float(row.get('kendall'))} | "
            f"{format_float(row.get('top1_match'))} | {format_float(row.get('top2_overlap'))} |"
        )
    lines.extend(["", "### Per-type Signed Bias（scene-micro）", "| split | type | mean(p-q) | mean under | mean over |", "|---|---:|---:|---:|---:|"])
    for row in [item for item in summary_rows if item["group_name"] == "scene_micro" and item["group_value"] in {"train", "test"}]:
        for type_id in active_type_ids:
            lines.append(
                f"| {row['group_value']} | {GRASP_TYPES[type_id]} | "
                f"{format_float(row.get(f'type_{type_id}_signed_bias'))} | "
                f"{format_float(row.get(f'type_{type_id}_mean_under'))} | "
                f"{format_float(row.get(f'type_{type_id}_mean_over'))} |"
            )
    lines.extend(["", "### 分组说明", "- aggregate CSV 额外包含 `|P(scene)|` 分组，以及 `xy_long`、`xy_short`、`z_height` 三个 canonical point-cloud scale descriptor 的分位数 bucket 分组。"])
    return lines


def write_human_1a_outputs(
    config,
    label_rows: list[dict],
    label_summary: dict,
    output_dir: str,
    active_type_ids: tuple[int, ...] = REAL_TYPE_IDS,
    output_suffix: str = "",
) -> tuple[list[str], list[dict], list[dict]]:
    """Write 1A label, metric, summary, and report lines for active types.

    Args:
        config: Full Hydra config.
        label_rows: Pose-class label rows aligned with ``active_type_ids``.
        label_summary: Label construction or transformation summary.
        output_dir: Directory for CSV/JSON outputs.
        active_type_ids: Real grasp type ids included in this evaluation view.
        output_suffix: Optional suffix appended to output filenames.

    Returns:
        Tuple of markdown lines, metric rows, and label rows.
    """
    metric_rows = [row for row in (metric_row_for_pose_class(label_row, active_type_ids) for label_row in label_rows) if row is not None]
    add_scale_buckets(metric_rows, int(getattr(config.task, "scale_bucket_count", 4)))
    summary_rows = build_human_metric_summaries(metric_rows, active_type_ids)
    summary_rows.extend(train_test_gap_rows(summary_rows))
    _write_csv(label_rows, suffixed_output_path(output_dir, "evaluation_human_pose_class_labels.csv", output_suffix))
    _write_csv(metric_rows, suffixed_output_path(output_dir, "evaluation_human_pose_class_metrics.csv", output_suffix))
    _write_csv(summary_rows, suffixed_output_path(output_dir, "evaluation_human_metric_summary.csv", output_suffix))
    _write_json(label_summary, suffixed_output_path(output_dir, "evaluation_human_label_summary.json", output_suffix))
    return human_report_lines(label_rows, metric_rows, summary_rows, output_dir, active_type_ids, output_suffix, label_summary), metric_rows, label_rows


def run_human_1a(config, human_score_rows: list[dict], output_dir: str) -> tuple[list[str], list[dict], list[dict], dict | None]:
    """Run 1A human train/test labeled metrics.

    Args:
        config: Full Hydra config.
        human_score_rows: Human score rows from saved test results or JSONL.
        output_dir: Directory for CSV outputs.

    Returns:
        Tuple of markdown lines, scored human metric rows, label rows, and an
        optional without-both-three result bundle.
    """
    label_rows, label_summary = build_pose_class_labels(config, human_score_rows)
    human_lines, metric_rows, label_rows = write_human_1a_outputs(
        config=config,
        label_rows=label_rows,
        label_summary=label_summary,
        output_dir=output_dir,
    )
    excluded_result = None
    if bool(getattr(config.task, "exclude_both_three", False)):
        excluded_rows, excluded_summary = label_rows_without_grasp_type(label_rows, BOTH_THREE_TYPE_ID)
        excluded_lines, excluded_metric_rows, excluded_rows = write_human_1a_outputs(
            config=config,
            label_rows=excluded_rows,
            label_summary=excluded_summary,
            output_dir=output_dir,
            active_type_ids=tuple(type_id for type_id in REAL_TYPE_IDS if type_id != BOTH_THREE_TYPE_ID),
            output_suffix=WITHOUT_BOTH_THREE_SUFFIX,
        )
        excluded_result = {
            "lines": excluded_lines,
            "metric_rows": excluded_metric_rows,
            "label_rows": excluded_rows,
            "active_type_ids": tuple(type_id for type_id in REAL_TYPE_IDS if type_id != BOTH_THREE_TYPE_ID),
            "output_suffix": WITHOUT_BOTH_THREE_SUFFIX,
        }
    return human_lines, metric_rows, label_rows, excluded_result


def load_scale_anchor_distributions(path: str, scope: str = "train") -> list[dict]:
    """Load scale-anchor grasp-type distributions from HumanGraspData.

    Args:
        path: JSON path generated by HumanGraspData object-scale statistics.
        scope: Distribution scope to load. ``train`` avoids using test labels in
            the scale-anchor baseline. ``all`` is only for explicit diagnostics.

    Returns:
        Anchor rows with normalized five-type distributions.
    """
    payload = _load_json(path)
    scope = str(scope or "train")
    scoped_payload = payload.get("scopes", {}).get(scope)
    if scoped_payload is not None:
        entries = scoped_payload.get("entries", [])
    elif scope == "all":
        entries = payload.get("entries", [])
    else:
        raise ValueError(
            f"Scale-anchor distribution JSON {path} does not contain scope={scope!r}. "
            "Regenerate it with HumanGraspData task=stat so baseline statistics come from train only."
        )
    anchors = []
    for entry in entries:
        distribution = entry.get("distribution", {})
        probs = np.asarray([float(distribution.get(label, 0.0)) for label in GRASP_TYPE_LABELS], dtype=np.float64)
        total = float(probs.sum())
        if total <= EPS:
            continue
        probs = probs / total
        anchors.append(
            {
                "anchor_index": int(entry.get("index", len(anchors))),
                "anchor_scale": float(entry["scale"]),
                "anchor_interval": str(entry.get("interval", "")),
                "anchor_total_count": int(entry.get("total_count", 0)),
                "anchor_scope": scope,
                "distribution": probs,
            }
        )
    if not anchors:
        raise ValueError(f"No valid scale-anchor distributions found in {path} for scope={scope!r}")
    anchors.sort(key=lambda item: (float(item["anchor_scale"]), int(item["anchor_index"])))
    return anchors


def nearest_scale_anchor(object_scale: float, anchors: list[dict]) -> dict:
    """Find the nearest scale anchor for one object scale.

    Args:
        object_scale: Object scale defined as half of AABB diagonal.
        anchors: Anchor rows from ``load_scale_anchor_distributions``.

    Returns:
        The nearest anchor row.
    """
    if not anchors:
        raise ValueError("Scale-anchor baseline requires at least one anchor")
    return min(anchors, key=lambda item: (abs(float(item["anchor_scale"]) - float(object_scale)), int(item["anchor_index"])))


def build_scale_anchor_baseline_predictions(
    label_rows: list[dict],
    anchors: list[dict],
    split: str = "test",
    active_type_ids: tuple[int, ...] = REAL_TYPE_IDS,
) -> list[dict]:
    """Build baseline prediction rows by nearest object-scale anchor.

    Args:
        label_rows: Human pose-class label rows produced by ``build_pose_class_labels``.
        anchors: Scale-anchor distribution rows.
        split: Dataset split to evaluate, normally ``test``.
        active_type_ids: Real grasp type ids included in this baseline view.

    Returns:
        Label-like rows whose ``type_*_p`` values come from the nearest anchor.
    """
    out = []
    for row in label_rows:
        if str(row.get("split", "")) != str(split):
            continue
        object_scale = float(row.get("object_scale", 0.5 * np.linalg.norm([float(row["xy_long"]), float(row["xy_short"]), float(row["z_height"])])))
        anchor = nearest_scale_anchor(object_scale, anchors)
        baseline = dict(row)
        baseline["has_score"] = True
        baseline["baseline_name"] = "nearest_object_scale_anchor"
        baseline["anchor_scope"] = str(anchor.get("anchor_scope", ""))
        baseline["object_scale"] = object_scale
        baseline["anchor_index"] = int(anchor["anchor_index"])
        baseline["anchor_scale"] = float(anchor["anchor_scale"])
        baseline["anchor_interval"] = str(anchor["anchor_interval"])
        baseline["anchor_total_count"] = int(anchor["anchor_total_count"])
        baseline["anchor_abs_scale_delta"] = abs(float(anchor["anchor_scale"]) - object_scale)
        active_distribution = normalize_probability_values(
            [float(anchor["distribution"][int(type_id) - 1]) for type_id in active_type_ids]
        )
        if active_distribution is None:
            continue
        for idx, type_id in enumerate(active_type_ids):
            baseline[f"type_{type_id}_p"] = float(active_distribution[idx])
        out.append(baseline)
    return out


def scale_anchor_baseline_report_lines(
    anchor_path: str,
    anchor_scope: str,
    anchors: list[dict],
    prediction_rows: list[dict],
    metric_rows: list[dict],
    summary_rows: list[dict],
    model_metric_rows: list[dict],
    output_dir: str,
    active_type_ids: tuple[int, ...] = REAL_TYPE_IDS,
    output_suffix: str = "",
) -> list[str]:
    """Format the scale-anchor baseline report section.

    Args:
        anchor_path: Source scale-anchor JSON path.
        anchor_scope: Source distribution scope, normally ``train``.
        anchors: Loaded scale anchors.
        prediction_rows: Baseline label/prediction rows.
        metric_rows: Per-pose-class baseline metric rows.
        summary_rows: Baseline aggregate rows.
        model_metric_rows: Existing model metric rows for comparison.
        output_dir: Directory containing CSV outputs.
        active_type_ids: Real grasp type ids included in this baseline view.
        output_suffix: Optional suffix appended to referenced output files.

    Returns:
        Markdown lines.
    """
    active_type_names = ", ".join(GRASP_TYPES[type_id] for type_id in active_type_ids)
    lines = [
        f"- Baseline 来源: `{anchor_path}`",
        f"- Baseline 统计 scope: `{anchor_scope}`。该 baseline 只使用这个 scope 内的 grasp distribution，默认使用 train 以避免 test label 泄漏。",
        f"- Active grasp types: `{active_type_names}`。",
        "- 对每个 test pose-class row，使用该 object 的 AABB 半对角 scale 找最近 anchor，并把该 anchor 的 active-type 分布作为 `p_t`。",
        f"- anchor 数量: {len(anchors)}；test baseline rows: {len(metric_rows)}。",
        f"- baseline prediction CSV: `{suffixed_output_path(output_dir, 'evaluation_human_scale_anchor_baseline_predictions.csv', output_suffix)}`",
        f"- baseline metric CSV: `{suffixed_output_path(output_dir, 'evaluation_human_scale_anchor_baseline_metrics.csv', output_suffix)}`",
        f"- baseline summary CSV: `{suffixed_output_path(output_dir, 'evaluation_human_scale_anchor_baseline_summary.csv', output_suffix)}`",
        "",
        "### Baseline vs Model（test）",
        "| method | avg | N | KL | SoftPrec | SoftRec@card(P) | CE | PosCE | CE@1e-6 | CE@1e-3 | L1 | TVD | JS | PosMass | PosR@card(P) | Spearman | Kendall | Top1 | Top2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    model_test_rows = [row for row in model_metric_rows if str(row.get("split", "")) == "test"]
    compare_rows = []
    if model_test_rows:
        compare_rows.append(("model", summarize_metric_rows(model_test_rows, "scene_micro", "test", active_type_ids)))
        compare_rows.append(("model", object_macro_summary(model_test_rows, "test", active_type_ids)))
    for row in summary_rows:
        if row["group_name"] in {"scene_micro", "object_macro"} and row["group_value"] == "test":
            compare_rows.append(("scale_anchor", row))
    for method, row in compare_rows:
        lines.append(
            f"| {method} | {row['group_name']} | {row['pose_class_num']} | "
            f"{format_float(row.get('kl_q_p'))} | {format_float(row.get('soft_precision'))} | "
            f"{format_float(row.get('soft_recall_at_positive_num'))} | "
            f"{format_float(row.get('soft_label_ce'))} | "
            f"{format_float(row.get('positive_set_ce'))} | {format_float(row.get(ce_eps_key(1e-6)))} | "
            f"{format_float(row.get(ce_eps_key(1e-3)))} | "
            f"{format_float(row.get('distribution_l1'))} | {format_float(row.get('tvd'))} | "
            f"{format_float(row.get('js_divergence'))} | {format_float(row.get('positive_probability_mass'))} | "
            f"{format_float(row.get('positive_set_recall_at_positive_num'))} | "
            f"{format_float(row.get('spearman'))} | {format_float(row.get('kendall'))} | "
            f"{format_float(row.get('top1_match'))} | {format_float(row.get('top2_overlap'))} |"
        )
    if metric_rows:
        mean_delta = float(np.mean([row["anchor_abs_scale_delta"] for row in metric_rows]))
        max_delta = float(np.max([row["anchor_abs_scale_delta"] for row in metric_rows]))
        lines.extend(["", f"- nearest-anchor scale delta mean={mean_delta:.6f}, max={max_delta:.6f}。"])
    return lines


def run_human_scale_anchor_baseline(
    config,
    label_rows: list[dict],
    model_metric_rows: list[dict],
    output_dir: str,
    active_type_ids: tuple[int, ...] = REAL_TYPE_IDS,
    output_suffix: str = "",
) -> list[str]:
    """Run the nearest-object-scale-anchor human baseline.

    Args:
        config: Full Hydra config.
        label_rows: Human pose-class label rows from 1A evaluation.
        model_metric_rows: Model metric rows from 1A evaluation.
        output_dir: Directory for CSV outputs.
        active_type_ids: Real grasp type ids included in this baseline view.
        output_suffix: Optional suffix appended to output filenames.

    Returns:
        Markdown report lines.
    """
    anchor_path = _abs_path(getattr(config.task, "human_scale_anchor_distribution_json", ""))
    if not anchor_path or not os.path.isfile(anchor_path):
        return [f"- 跳过 scale-anchor baseline：找不到 anchor JSON `{anchor_path}`。"]
    split = str(getattr(config.task, "human_scale_anchor_baseline_split", "test") or "test")
    anchor_scope = str(getattr(config.task, "human_scale_anchor_distribution_scope", "train") or "train")
    anchors = load_scale_anchor_distributions(anchor_path, scope=anchor_scope)
    prediction_rows = build_scale_anchor_baseline_predictions(label_rows, anchors, split=split, active_type_ids=active_type_ids)
    metric_rows = []
    for prediction in prediction_rows:
        metric = metric_row_for_pose_class(prediction, active_type_ids)
        if metric is None:
            continue
        for key in ["baseline_name", "anchor_scope", "anchor_index", "anchor_scale", "anchor_interval", "anchor_total_count", "anchor_abs_scale_delta"]:
            metric[key] = prediction[key]
        metric_rows.append(metric)
    if metric_rows:
        add_scale_buckets(metric_rows, int(getattr(config.task, "scale_bucket_count", 4)))
    summary_rows = build_human_metric_summaries(metric_rows, active_type_ids)
    summary_rows.extend(train_test_gap_rows(summary_rows))
    _write_csv(prediction_rows, suffixed_output_path(output_dir, "evaluation_human_scale_anchor_baseline_predictions.csv", output_suffix))
    _write_csv(metric_rows, suffixed_output_path(output_dir, "evaluation_human_scale_anchor_baseline_metrics.csv", output_suffix))
    _write_csv(summary_rows, suffixed_output_path(output_dir, "evaluation_human_scale_anchor_baseline_summary.csv", output_suffix))
    return scale_anchor_baseline_report_lines(
        anchor_path,
        anchor_scope,
        anchors,
        prediction_rows,
        metric_rows,
        summary_rows,
        model_metric_rows,
        output_dir,
        active_type_ids,
        output_suffix,
    )


def score_rows_with_scale(score_rows: list[dict], task_cfg, output_dir: str, prefix: str) -> list[dict]:
    """Attach canonical point-cloud scale descriptors to score rows.

    Args:
        score_rows: Score records.
        task_cfg: Evaluation task config.
        output_dir: Output directory.
        prefix: File prefix for skipped rows.

    Returns:
        Rows with score and scale fields.
    """
    max_points = int(getattr(task_cfg, "scale_max_points", 8192))
    cache = {}
    rows = []
    skipped = []
    for idx, row in enumerate(score_rows):
        try:
            descriptor = load_point_cloud_descriptor(row["pc_path"], max_points=max_points, cache=cache)
        except Exception as exc:
            skipped.append({"scene_id": row.get("scene_id", ""), "pc_path": row.get("pc_path", ""), "error": str(exc)})
            continue
        out = {
            "scene_id": row["scene_id"],
            "scene_key": row["scene_key"],
            "split": row.get("split", ""),
            "canonical_object_id": row["object_id"],
            "xy_long": float(descriptor[0]),
            "xy_short": float(descriptor[1]),
            "z_height": float(descriptor[2]),
            "aspect_xy": float(descriptor[0] / max(descriptor[1], EPS)),
        }
        for type_id, score in zip(REAL_TYPE_IDS, row["scores"]):
            out[f"type_{type_id}_score"] = float(score)
        out["top1_type"] = int(np.argmax(row["scores"]) + 1)
        rows.append(out)
        if idx > 0 and idx % 50000 == 0:
            print(f"[evaluate] computed scale descriptors for {idx} score rows")
    if skipped:
        _write_csv(skipped, os.path.join(output_dir, f"{prefix}_skipped_scale_rows.csv"))
    return rows


def percentile_summary(values: np.ndarray) -> dict:
    """Compute compact descriptive statistics.

    Args:
        values: One-dimensional numeric values.

    Returns:
        Summary dictionary.
    """
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "ci95_half_width": float(1.96 * values.std(ddof=1) / math.sqrt(values.size)) if values.size > 1 else 0.0,
    }


def summarize_score_distribution(rows: list[dict], group_name: str, group_value: str) -> dict:
    """Summarize score distribution for one DGN scale group.

    Args:
        rows: DGN rows with score fields.
        group_name: Group key name.
        group_value: Group key value.

    Returns:
        Summary row.
    """
    summary = {
        "group_name": group_name,
        "group_value": group_value,
        "scene_num": int(len(rows)),
        "object_num": int(len({row["canonical_object_id"] for row in rows})),
    }
    for type_id in REAL_TYPE_IDS:
        stats = percentile_summary(np.asarray([row[f"type_{type_id}_score"] for row in rows], dtype=np.float64))
        for key, value in stats.items():
            summary[f"type_{type_id}_{key}"] = value
    return summary


def dgn_distribution_summaries(rows: list[dict], bucket_count: int, min_bucket_count: int) -> list[dict]:
    """Build DGN score-vs-scale distribution summaries.

    Args:
        rows: DGN score rows with scale descriptors.
        bucket_count: Number of quantile buckets.
        min_bucket_count: Minimum rows for joint-bucket summaries.

    Returns:
        Summary rows.
    """
    add_scale_buckets(rows, bucket_count)
    aspect_edges = quantile_edges([float(row["aspect_xy"]) for row in rows], bucket_count)
    for row in rows:
        row["aspect_xy_bucket"] = bucket_label(float(row["aspect_xy"]), aspect_edges)
        row["joint_scale_bucket"] = "|".join([row[f"{feature}_bucket"] for feature in SCALE_FEATURES])
        row["xy_aspect_height_bucket"] = f"{row['xy_long_bucket']}|{row['xy_short_bucket']}|{row['z_height_bucket']}|{row['aspect_xy_bucket']}"
    summaries = [summarize_score_distribution(rows, "all", "all")] if rows else []
    for feature in SCALE_FEATURES:
        bucket_key = f"{feature}_bucket"
        for bucket in sorted({row[bucket_key] for row in rows}):
            bucket_rows = [row for row in rows if row[bucket_key] == bucket]
            summaries.append(summarize_score_distribution(bucket_rows, bucket_key, bucket))
    for bucket_key in ["joint_scale_bucket", "xy_aspect_height_bucket"]:
        for bucket in sorted({row[bucket_key] for row in rows}):
            bucket_rows = [row for row in rows if row[bucket_key] == bucket]
            if len(bucket_rows) >= min_bucket_count:
                summaries.append(summarize_score_distribution(bucket_rows, bucket_key, bucket))
    return summaries


def dgn_correlations(rows: list[dict]) -> list[dict]:
    """Compute score-scale Pearson and Spearman correlations.

    Args:
        rows: DGN score rows with scale descriptors.

    Returns:
        Correlation rows.
    """
    out = []
    for feature in SCALE_FEATURES:
        x = np.asarray([row[feature] for row in rows], dtype=np.float64)
        for type_id in REAL_TYPE_IDS:
            y = np.asarray([row[f"type_{type_id}_score"] for row in rows], dtype=np.float64)
            out.append(
                {
                    "scale_feature": feature,
                    "type_id": type_id,
                    "pearson": pearson_corr(x, y),
                    "spearman": spearman_corr(x, y),
                    "n": int(len(rows)),
                }
            )
    return out


def dgn_stability_rows(rows: list[dict]) -> list[dict]:
    """Compute same-object score stability over DGN scenes.

    Args:
        rows: DGN score rows with scale descriptors.

    Returns:
        Per-object stability rows.
    """
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["canonical_object_id"], []).append(row)
    out = []
    for object_id, group in sorted(grouped.items(), key=lambda item: _natural_sort_key(item[0])):
        if len(group) < 2:
            continue
        scores = np.asarray([[row[f"type_{type_id}_score"] for type_id in REAL_TYPE_IDS] for row in group], dtype=np.float64)
        top1 = np.asarray([row["top1_type"] for row in group], dtype=np.int64)
        values, counts = np.unique(top1, return_counts=True)
        majority_rate = float(counts.max() / counts.sum())
        descriptor = np.asarray([[row[feature] for feature in SCALE_FEATURES] for row in group], dtype=np.float64).mean(axis=0)
        out.append(
            {
                "canonical_object_id": object_id,
                "scene_num": int(len(group)),
                "xy_long": float(descriptor[0]),
                "xy_short": float(descriptor[1]),
                "z_height": float(descriptor[2]),
                "mean_type_score_std": float(scores.std(axis=0).mean()),
                "max_type_score_std": float(scores.std(axis=0).max()),
                "top1_flip_rate": float(1.0 - majority_rate),
                "top1_types": " ".join(str(value) for value in values.tolist()),
            }
        )
    return out


def similar_shape_consistency(rows: list[dict], min_bucket_count: int) -> list[dict]:
    """Estimate consistency across objects with similar shape and size.

    Args:
        rows: DGN score rows with bucket labels.
        min_bucket_count: Minimum objects per bucket.

    Returns:
        Bucket consistency rows.
    """
    object_groups: dict[str, list[dict]] = {}
    for row in rows:
        object_groups.setdefault(row["canonical_object_id"], []).append(row)
    object_rows = []
    for object_id, group in object_groups.items():
        score = np.asarray([[row[f"type_{type_id}_score"] for type_id in REAL_TYPE_IDS] for row in group], dtype=np.float64).mean(axis=0)
        base = group[0]
        object_row = {
            "canonical_object_id": object_id,
            "joint_scale_bucket": base.get("joint_scale_bucket", "all"),
            "aspect_xy_bucket": base.get("aspect_xy_bucket", "all"),
            "shape_size_bucket": f"{base.get('joint_scale_bucket', 'all')}|{base.get('aspect_xy_bucket', 'all')}",
        }
        for type_id, value in zip(REAL_TYPE_IDS, score):
            object_row[f"type_{type_id}_score"] = float(value)
        object_rows.append(object_row)
    grouped: dict[str, list[dict]] = {}
    for row in object_rows:
        grouped.setdefault(row["shape_size_bucket"], []).append(row)
    out = []
    for bucket, group in sorted(grouped.items()):
        if len(group) < min_bucket_count:
            continue
        scores = np.asarray([[row[f"type_{type_id}_score"] for type_id in REAL_TYPE_IDS] for row in group], dtype=np.float64)
        out.append(
            {
                "shape_size_bucket": bucket,
                "object_num": int(len(group)),
                "mean_cross_object_type_std": float(scores.std(axis=0).mean()),
                "max_cross_object_type_std": float(scores.std(axis=0).max()),
            }
        )
    return out


def dgn_ordinal_sanity(rows: list[dict]) -> list[dict]:
    """Compute coarse ordinal/rule sanity diagnostics.

    Args:
        rows: DGN rows with score and scale fields.

    Returns:
        Rule diagnostic rows.
    """
    if not rows:
        return []
    values = {feature: np.asarray([row[feature] for row in rows], dtype=np.float64) for feature in SCALE_FEATURES}
    q20 = {feature: float(np.quantile(values[feature], 0.2)) for feature in SCALE_FEATURES}
    q80 = {feature: float(np.quantile(values[feature], 0.8)) for feature in SCALE_FEATURES}
    rules = {
        "small_xy": [row for row in rows if row["xy_long"] <= q20["xy_long"] and row["xy_short"] <= q20["xy_short"]],
        "large_flat": [row for row in rows if row["xy_long"] >= q80["xy_long"] and row["z_height"] <= q20["z_height"]],
        "tall": [row for row in rows if row["z_height"] >= q80["z_height"]],
    }
    out = []
    for name, group in rules.items():
        if not group:
            continue
        summary = {"rule": name, "scene_num": int(len(group))}
        for type_id in REAL_TYPE_IDS:
            summary[f"type_{type_id}_mean_score"] = float(np.mean([row[f"type_{type_id}_score"] for row in group]))
        out.append(summary)
    return out


def repeatability_rows(score_rows: list[dict]) -> list[dict]:
    """Measure repeated score variance for duplicated scene ids.

    Args:
        score_rows: Original score rows before scene aggregation.

    Returns:
        Per-duplicated-scene repeatability rows.
    """
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in score_rows:
        grouped.setdefault((row.get("split", ""), row["scene_key"]), []).append(row)
    out = []
    for (split, key), group in grouped.items():
        if len(group) < 2:
            continue
        scores = np.stack([row["scores"] for row in group], axis=0)
        out.append(
            {
                "split": split,
                "scene_key": key,
                "repeat_count": int(len(group)),
                "mean_type_score_std": float(scores.std(axis=0).mean()),
                "max_type_score_std": float(scores.std(axis=0).max()),
            }
        )
    return out


def human_scale_gap_rows(dgn_rows: list[dict], human_metric_rows: list[dict], min_bucket_count: int, bucket_count: int) -> list[dict]:
    """Compare DGN and human score distributions in matched scale buckets.

    Args:
        dgn_rows: DGN rows with bucket labels and scores.
        human_metric_rows: Human metric rows with bucket labels and p scores.
        min_bucket_count: Minimum rows per domain in a bucket.
        bucket_count: Number of shared quantile buckets.

    Returns:
        DGN-minus-human score gap rows.
    """
    combined_edges = {
        feature: quantile_edges(
            [float(row[feature]) for row in dgn_rows] + [float(row[feature]) for row in human_metric_rows],
            bucket_count,
        )
        for feature in SCALE_FEATURES
    }

    def common_joint_bucket(row: dict) -> str:
        """Assign a row to a joint scale bucket using shared edges.

        Args:
            row: DGN or human row with scale descriptor fields.

        Returns:
            Joint bucket label built from shared quantile edges.
        """
        return "|".join(bucket_label(float(row[feature]), combined_edges[feature]) for feature in SCALE_FEATURES)

    dgn_by_bucket: dict[str, list[dict]] = {}
    for row in dgn_rows:
        dgn_by_bucket.setdefault(common_joint_bucket(row), []).append(row)
    human_by_domain_bucket: dict[tuple[str, str], list[dict]] = {}
    for row in human_metric_rows:
        bucket = common_joint_bucket(row)
        human_by_domain_bucket.setdefault((f"human_{row['split']}", bucket), []).append(row)
    out = []
    for (domain, bucket), human_group in sorted(human_by_domain_bucket.items()):
        dgn_group = dgn_by_bucket.get(bucket, [])
        if len(human_group) < min_bucket_count or len(dgn_group) < min_bucket_count:
            continue
        gap = {
            "human_domain": domain,
            "joint_scale_bucket": bucket,
            "human_count": int(len(human_group)),
            "dgn_count": int(len(dgn_group)),
        }
        for type_id in REAL_TYPE_IDS:
            dgn_mean = float(np.mean([row[f"type_{type_id}_score"] for row in dgn_group]))
            human_mean = float(np.mean([row.get(f"type_{type_id}_p", row.get(f"type_{type_id}_score", 0.0)) for row in human_group]))
            gap[f"type_{type_id}_dgn_minus_human"] = dgn_mean - human_mean
        out.append(gap)
    return out


def dgn_report_lines(
    dgn_rows: list[dict],
    distribution_rows: list[dict],
    correlation_rows: list[dict],
    stability: list[dict],
    consistency: list[dict],
    ordinal_rows: list[dict],
    repeat_rows: list[dict],
    gap_rows: list[dict],
    output_dir: str,
) -> list[str]:
    """Format the DGN 1B markdown section.

    Args:
        dgn_rows: DGN rows with scale descriptors.
        distribution_rows: Score-vs-scale summaries.
        correlation_rows: Score-scale correlations.
        stability: Per-object stability rows.
        consistency: Similar-shape consistency rows.
        ordinal_rows: Rule sanity rows.
        repeat_rows: Stochastic repeatability rows.
        gap_rows: DGN-vs-human gap rows.
        output_dir: Directory containing CSV outputs.

    Returns:
        Markdown lines.
    """
    lines = [
        "- 本节实现 `Evaluation Metrics 1 / 1B`。DGN 没有 human label，因此只做 score 与 canonical point-cloud 三维尺寸 `(xy_long, xy_short, z_height)` 的关系诊断。",
        f"- DGN score rows with scale: {len(dgn_rows)}。",
        f"- 明细 CSV: `{os.path.join(output_dir, 'evaluation_dgn_score_scale_rows.csv')}`",
        f"- distribution CSV: `{os.path.join(output_dir, 'evaluation_dgn_score_scale_distribution.csv')}`",
        f"- correlation CSV: `{os.path.join(output_dir, 'evaluation_dgn_score_scale_correlation.csv')}`",
        "",
        "### 指标解释",
        "- `Score-vs-3D-scale distribution`：按 `xy_long`、`xy_short`、`z_height` 的分位数 bucket 统计每个 type score 的均值、方差、分位数和 95% CI，用来观察 score 是否随尺寸出现系统性偏移。",
        "- `Score-scale correlation`：每个 type 分别计算 score 与三个连续尺寸变量的 Pearson / Spearman correlation，用来发现单调尺度偏差。",
        "- `Per-scale score stability`：同一 canonical object 的多个 DGN scenes 上统计 score 方差和 top-1 flip rate，用来定位 pose / point sampling 导致的不稳定。",
        "- `Similar-shape-size consistency`：在相近 `(xy_long, xy_short, z_height)` 和 aspect ratio bucket 内比较不同 object 的平均 score 方差，用来发现几何尺寸相近但输出不连续的问题。",
        "- `Ordinal / rule sanity`：只做粗规则诊断，例如 small-xy、large-flat、tall bucket 上各 type 平均 score 是否明显异常；它不是硬标签。",
        "- `Score repeatability`：如果同一 scene 有重复 score row，则统计重复预测方差；没有重复 row 时该项不可用。",
        "- `DGN-vs-Human 3D-scale-conditioned gap`：在相同 3D scale bucket 下比较 DGN 与 human train/test 的 score 均值差，用来观察 human-to-DGN OOD calibration gap。",
        "",
        "### Score-scale Correlation",
        "| scale | type | Pearson | Spearman | N |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in correlation_rows:
        lines.append(
            f"| {row['scale_feature']} | {row['type_id']} | {format_float(row['pearson'])} | "
            f"{format_float(row['spearman'])} | {row['n']} |"
        )
    all_row = next((row for row in distribution_rows if row["group_name"] == "all"), None)
    if all_row is not None:
        lines.extend(["", "### Overall Score Percentiles（仅作 scale 分组诊断的入口）", "| type | mean | std | p05 | p50 | p95 | ci95 |", "|---:|---:|---:|---:|---:|---:|---:|"])
        for type_id in REAL_TYPE_IDS:
            lines.append(
                f"| {type_id} | {format_float(all_row.get(f'type_{type_id}_mean'))} | "
                f"{format_float(all_row.get(f'type_{type_id}_std'))} | {format_float(all_row.get(f'type_{type_id}_p05'))} | "
                f"{format_float(all_row.get(f'type_{type_id}_p50'))} | {format_float(all_row.get(f'type_{type_id}_p95'))} | "
                f"{format_float(all_row.get(f'type_{type_id}_ci95_half_width'))} |"
            )
    lines.extend(["", "### Stability / Consistency", f"- Same-object stability rows: {len(stability)}；CSV: `{os.path.join(output_dir, 'evaluation_dgn_object_stability.csv')}`"])
    if stability:
        lines.append(f"- mean top1 flip rate: {np.mean([row['top1_flip_rate'] for row in stability]):.4f}；mean type-score std: {np.mean([row['mean_type_score_std'] for row in stability]):.4f}")
    lines.append(f"- Similar-shape-size consistency rows: {len(consistency)}；CSV: `{os.path.join(output_dir, 'evaluation_dgn_similar_shape_consistency.csv')}`")
    if consistency:
        lines.append(f"- mean cross-object type std in similar buckets: {np.mean([row['mean_cross_object_type_std'] for row in consistency]):.4f}")
    lines.extend(["", "### Ordinal / Rule Sanity", "| rule | N | type1 | type2 | type3 | type4 | type5 |", "|---|---:|---:|---:|---:|---:|---:|"])
    for row in ordinal_rows:
        lines.append(
            f"| {row['rule']} | {row['scene_num']} | " + " | ".join(format_float(row.get(f"type_{type_id}_mean_score")) for type_id in REAL_TYPE_IDS) + " |"
        )
    lines.extend(["", "### Repeatability / Domain Gap", f"- Repeated-scene score rows: {len(repeat_rows)}；CSV: `{os.path.join(output_dir, 'evaluation_dgn_score_repeatability.csv')}`"])
    if repeat_rows:
        lines.append(f"- repeat max type-score std mean: {np.mean([row['max_type_score_std'] for row in repeat_rows]):.4f}")
    lines.append(f"- DGN-vs-Human scale-conditioned gap rows: {len(gap_rows)}；CSV: `{os.path.join(output_dir, 'evaluation_dgn_vs_human_scale_gap.csv')}`")
    return lines


def run_dgn_1b(config, dgn_score_rows: list[dict], human_metric_rows: list[dict], output_dir: str) -> list[str]:
    """Run 1B DGN score-scale diagnostics.

    Args:
        config: Full Hydra config.
        dgn_score_rows: DGN score rows.
        human_metric_rows: Human metric rows for scale-conditioned comparison.
        output_dir: Directory for CSV outputs.

    Returns:
        Markdown lines.
    """
    if not dgn_score_rows:
        return ["- 未找到 DGN test result score，因此跳过 1B。"]
    bucket_count = int(getattr(config.task, "scale_bucket_count", 4))
    min_bucket_count = int(getattr(config.task, "min_bucket_count", 20))
    dgn_rows = score_rows_with_scale(dgn_score_rows, config.task, output_dir, "dgn")
    distribution_rows = dgn_distribution_summaries(dgn_rows, bucket_count=bucket_count, min_bucket_count=min_bucket_count)
    correlation_rows = dgn_correlations(dgn_rows)
    stability = dgn_stability_rows(dgn_rows)
    if stability:
        add_scale_buckets(stability, bucket_count)
    consistency = similar_shape_consistency(dgn_rows, min_bucket_count=max(2, min_bucket_count // 10))
    ordinal_rows = dgn_ordinal_sanity(dgn_rows)
    repeat_rows = repeatability_rows(dgn_score_rows)
    gap_rows = human_scale_gap_rows(dgn_rows, human_metric_rows, min_bucket_count=min_bucket_count, bucket_count=bucket_count) if human_metric_rows else []

    _write_csv(dgn_rows, os.path.join(output_dir, "evaluation_dgn_score_scale_rows.csv"))
    _write_csv(distribution_rows, os.path.join(output_dir, "evaluation_dgn_score_scale_distribution.csv"))
    _write_csv(correlation_rows, os.path.join(output_dir, "evaluation_dgn_score_scale_correlation.csv"))
    _write_csv(stability, os.path.join(output_dir, "evaluation_dgn_object_stability.csv"))
    _write_csv(consistency, os.path.join(output_dir, "evaluation_dgn_similar_shape_consistency.csv"))
    _write_csv(ordinal_rows, os.path.join(output_dir, "evaluation_dgn_ordinal_rule_sanity.csv"))
    _write_csv(repeat_rows, os.path.join(output_dir, "evaluation_dgn_score_repeatability.csv"))
    _write_csv(gap_rows, os.path.join(output_dir, "evaluation_dgn_vs_human_scale_gap.csv"))
    return dgn_report_lines(dgn_rows, distribution_rows, correlation_rows, stability, consistency, ordinal_rows, repeat_rows, gap_rows, output_dir)


def output_dir_from_paths(config, primary_input_path: str) -> str:
    """Resolve evaluation output directory.

    Args:
        config: Full Hydra config.
        primary_input_path: Primary result directory or score JSONL path.

    Returns:
        Output directory path.
    """
    configured = getattr(config.task, "output_dir", "")
    if configured:
        return _abs_path(configured)
    if primary_input_path:
        if os.path.isdir(primary_input_path):
            return os.path.join(os.path.dirname(primary_input_path), "evaluation")
        return os.path.join(os.path.dirname(primary_input_path), "evaluation")
    return _abs_path(os.path.join(str(config.output_folder), str(config.wandb.id), "evaluation"))


def task_evaluate(config) -> None:
    """Hydra entry point for Human Prior intrinsic evaluation 1A/1B.

    Args:
        config: Full Hydra config. The task reads saved ``task=sample`` results
            under ``tests/step_*/{humanMulti,DGNMulti}`` by default and does not
            run model sampling or downstream BODex/Bench.

    Returns:
        None.
    """
    resolve_type_supervision_config(config)
    flatten_multidex_data_config(config.data)
    if hasattr(config, "test_data"):
        flatten_multidex_data_config(config.test_data)

    max_score_rows = int(getattr(config.task, "max_score_rows", 0))
    score_grasp_type = str(getattr(config.task, "score_grasp_type", "0_any") or "0_any")
    exclude_both_three = bool(getattr(config.task, "exclude_both_three", False))
    human_dataset_name = str(getattr(config.task, "human_results_dataset", "humanMulti") or "humanMulti")
    dgn_dataset_name = str(getattr(config.task, "dgn_results_dataset", "DGNMulti") or "DGNMulti")

    human_score_jsonl = resolve_score_jsonl(config, "human_score_jsonl", fallback="")
    dgn_score_jsonl = resolve_score_jsonl(config, "dgn_score_jsonl", fallback="")
    human_results_dir = resolve_test_result_dir(config, "human_results_dir", human_dataset_name)
    dgn_results_dir = resolve_test_result_dir(config, "dgn_results_dir", dgn_dataset_name)

    split_lookup = human_split_lookup(config)
    if human_score_jsonl:
        human_score_rows = load_score_jsonl(human_score_jsonl, max_rows=max_score_rows)
        human_source = human_score_jsonl
    else:
        human_score_rows = load_test_result_scores(
            human_results_dir,
            max_rows=max_score_rows,
            split_lookup=split_lookup,
            score_grasp_type=score_grasp_type,
        )
        human_source = human_results_dir

    if dgn_score_jsonl:
        dgn_score_rows = load_score_jsonl(dgn_score_jsonl, max_rows=max_score_rows)
        dgn_source = dgn_score_jsonl
    else:
        dgn_score_rows = load_test_result_scores(
            dgn_results_dir,
            max_rows=max_score_rows,
            split_lookup=None,
            score_grasp_type=score_grasp_type,
        )
        dgn_source = dgn_results_dir

    output_dir = output_dir_from_paths(config, dgn_source or human_source)
    os.makedirs(output_dir, exist_ok=True)
    report_path_value = getattr(config.task, "report_md", "")
    report_path = _abs_path(report_path_value) if report_path_value else os.path.join(output_dir, "evaluation_report.md")
    report = MarkdownReport(report_path, "Human Prior Intrinsic Evaluation")

    report.add_section(
        "输入",
        [
            f"- human_score_jsonl: `{human_score_jsonl}`",
            f"- human_results_dir: `{human_results_dir}`",
            f"- human score rows: {len(human_score_rows)}",
            f"- dgn_score_jsonl: `{dgn_score_jsonl}`",
            f"- dgn_results_dir: `{dgn_results_dir}`",
            f"- dgn score rows: {len(dgn_score_rows)}",
            f"- score_grasp_type: `{score_grasp_type}`",
            f"- exclude_both_three: `{exclude_both_three}`",
            f"- output_dir: `{output_dir}`",
            "- 本任务只读已保存的 `tests/step_*` sample 结果，不运行 sampling、BimanBODex 或 Bench。",
        ],
    )

    human_metric_rows = []
    human_label_rows = []
    excluded_both_three_result = None
    if bool(getattr(config.task, "run_human_1a", True)):
        try:
            human_lines, human_metric_rows, human_label_rows, excluded_both_three_result = run_human_1a(config, human_score_rows, output_dir)
            report.add_section("1A Human Train/Test 有 Label", human_lines)
        except Exception as exc:
            report.add_section("1A Human Train/Test 有 Label", [f"- 1A 运行失败：`{type(exc).__name__}: {exc}`"])
            raise

    excluded_baseline_lines = []
    if bool(getattr(config.task, "run_human_scale_anchor_baseline", True)):
        try:
            baseline_lines = run_human_scale_anchor_baseline(config, human_label_rows, human_metric_rows, output_dir)
            report.add_section("1A-Baseline Human Scale Anchor", baseline_lines)
            if excluded_both_three_result is not None:
                excluded_baseline_lines = run_human_scale_anchor_baseline(
                    config,
                    excluded_both_three_result["label_rows"],
                    excluded_both_three_result["metric_rows"],
                    output_dir,
                    excluded_both_three_result["active_type_ids"],
                    excluded_both_three_result["output_suffix"],
                )
        except Exception as exc:
            report.add_section("1A-Baseline Human Scale Anchor", [f"- baseline 运行失败：`{type(exc).__name__}: {exc}`"])
            raise

    if bool(getattr(config.task, "run_dgn_1b", True)):
        dgn_lines = run_dgn_1b(config, dgn_score_rows, human_metric_rows, output_dir)
        report.add_section("1B DGN Testset 无 Label", dgn_lines)

    report.add_section(
        "结论使用边界",
        [
            "- 1A 的 `q_t` 来自 human observed count distribution，不是下游最优 utility。",
            "- 1B 没有 label，只能诊断 score 与三维 canonical point-cloud 尺寸的关系、稳定性和 human-to-DGN 分布差异。",
            "- 最终 allocator 是否有效仍需要固定总 budget 下的 BimanBODex + Bench paired downstream evaluation 验证。",
        ],
    )
    if excluded_both_three_result is not None:
        excluded_report_path = suffixed_output_path(output_dir, "evaluation_report.md", WITHOUT_BOTH_THREE_SUFFIX)
        excluded_report = MarkdownReport(excluded_report_path, "Human Prior Intrinsic Evaluation（without both_three）")
        excluded_report.add_section(
            "输入",
            [
                f"- source evaluation output_dir: `{output_dir}`",
                f"- score_grasp_type: `{score_grasp_type}`",
                f"- exclude_both_three: `{exclude_both_three}`",
                f"- output_suffix: `{WITHOUT_BOTH_THREE_SUFFIX}`",
                "- 本报告与主 evaluation 同次生成；只在 1A/baseline 统计口径中移除 `4_both_three`，并对剩余四类的 `q_t` / `p_t` 重新归一化。",
            ],
        )
        excluded_report.add_section("1A Human Train/Test 有 Label", excluded_both_three_result["lines"])
        if excluded_baseline_lines:
            excluded_report.add_section("1A-Baseline Human Scale Anchor", excluded_baseline_lines)
        excluded_report.add_section(
            "结论使用边界",
            [
                "- 本报告回答的是排除 `4_both_three` 后，Human Prior 在剩余四类上的分布匹配表现。",
                "- 它不重新训练模型，也不重新运行 sampling、BimanBODex 或 Bench。",
                "- 移除 type 后的 `p_t` 会重新归一化，因此数值不能和原五类固定总预算口径直接混用。",
            ],
        )
        print(f"[evaluate] Wrote without-both-three report to {excluded_report_path}")
    print(f"[evaluate] Wrote report to {report_path}")
