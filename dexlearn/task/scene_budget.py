import csv
import json
import math
import os
import re
from copy import deepcopy
from glob import glob
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation as SciR
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm.auto import tqdm

from dexlearn.dataset import create_test_dataloader, get_sparse_tensor
from dexlearn.dataset.grasp_types import GRASP_TYPES
from dexlearn.network.models import GeometryBudgetHead, PointCloudBudgetHead
from dexlearn.utils.config import cfg_get, flatten_multidex_data_config
from dexlearn.utils.util import set_seed


EPS = 1e-8
MIRROR = np.diag([-1.0, 1.0, 1.0]).astype(np.float64)
GEOMETRY_FEATURE_NAMES = [
    "bbox_xy_major",
    "bbox_xy_minor",
    "bbox_z",
]


def _natural_sort_key(value: Any) -> list:
    """Build a natural-sort key for ids that contain numbers.

    Args:
        value: Value to sort, such as ``obj_10`` or ``seq_2``.

    Returns:
        List of lowercase text chunks and integer chunks.
    """
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value))]


def _as_list(value: Any) -> list:
    """Convert an OmegaConf or Python value into a regular list.

    Args:
        value: Scalar, list, tuple, or ListConfig value.

    Returns:
        A Python list. Scalars become a one-element list.
    """
    if isinstance(value, ListConfig):
        return list(value)
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _absolute_path(path: str) -> str:
    """Resolve a path relative to the original Hydra working directory.

    Args:
        path: Absolute or relative path.

    Returns:
        Absolute filesystem path.
    """
    path = str(path)
    if os.path.isabs(path):
        return path
    try:
        root_dir = hydra.utils.get_original_cwd()
    except ValueError:
        root_dir = os.getcwd()
    return os.path.abspath(os.path.join(root_dir, path))


def _json_default(value: Any) -> Any:
    """Convert NumPy values into JSON-serializable Python values.

    Args:
        value: Value passed by ``json.dump``.

    Returns:
        JSON-serializable representation.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _load_json(path: str) -> Any:
    """Load one JSON file.

    Args:
        path: JSON file path.

    Returns:
        Parsed JSON content.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(data: Any, path: str) -> None:
    """Save JSON data with a stable indentation.

    Args:
        data: JSON-serializable object.
        path: Destination path.

    Returns:
        None.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)


def _normalize_sequence_id(sequence_id: Any) -> str:
    """Normalize a sequence id into a path-safe string.

    Args:
        sequence_id: Raw sequence id from grasp metadata.

    Returns:
        Normalized sequence id, or an empty string if unavailable.
    """
    if sequence_id is None:
        return ""
    return str(sequence_id).strip().replace("\\", "/").strip("/")


def _get_sequence_id(grasp_data: dict) -> str:
    """Read the sequence id from one formatted human grasp record.

    Args:
        grasp_data: Loaded grasp dictionary.

    Returns:
        Sequence id, or an empty string if the record has no sequence id.
    """
    object_data = grasp_data.get("object", {})
    sequence_id = object_data.get("sequence_id")
    if sequence_id is not None:
        return _normalize_sequence_id(sequence_id)

    source_scene = object_data.get("source_scene")
    if source_scene is not None and "_seq_" in str(source_scene):
        return f"seq_{str(source_scene).rsplit('_seq_', 1)[1]}"
    return ""


def _scene_key(object_id: str, sequence_id: str) -> str:
    """Build the canonical scene key used by the scene budget task.

    Args:
        object_id: Canonical object id.
        sequence_id: Sequence id, if available.

    Returns:
        ``object_id/sequence_id`` when a sequence id exists, otherwise
        ``object_id``.
    """
    sequence_id = _normalize_sequence_id(sequence_id)
    if not sequence_id:
        return str(object_id)
    return f"{object_id}/{sequence_id}"


def _pointcloud_matches_sequence(pc_path: str, sequence_id: str) -> bool:
    """Check whether a point-cloud path matches a sequence id.

    Args:
        pc_path: Candidate point-cloud path.
        sequence_id: Required sequence id.

    Returns:
        True if the path name or a parent component contains the sequence id.
    """
    sequence_id = _normalize_sequence_id(sequence_id)
    if not sequence_id:
        return False
    base_name = os.path.splitext(os.path.basename(pc_path))[0]
    path_parts = pc_path.split(os.sep)
    return base_name == sequence_id or base_name.startswith(f"{sequence_id}_") or sequence_id in path_parts


def _select_pointcloud_path(object_path: str, pc_rel_path: str, object_id: str, sequence_id: str) -> str:
    """Select a deterministic point cloud for a human object scene.

    Args:
        object_path: Formatted human object root.
        pc_rel_path: Relative complete point-cloud folder under ``object_path``.
        object_id: Canonical object id.
        sequence_id: Sequence id used to prefer same-sequence geometry.

    Returns:
        Path to the selected point-cloud ``.npy`` file.
    """
    pc_root = os.path.join(object_path, pc_rel_path, object_id)
    candidates = sorted(glob(os.path.join(pc_root, "**", "*.npy"), recursive=True))
    if not candidates:
        raise FileNotFoundError(f"No point clouds found under {pc_root}")

    matched = [pc_path for pc_path in candidates if _pointcloud_matches_sequence(pc_path, sequence_id)]
    if matched:
        return matched[0]
    return candidates[0]


def _scale_points(points: np.ndarray, scale: Any) -> np.ndarray:
    """Apply object scale to point coordinates.

    Args:
        points: Point cloud with shape ``(N, 3)``.
        scale: Scalar or per-axis object scale.

    Returns:
        Scaled point cloud with shape ``(N, 3)``.
    """
    scale_array = np.asarray(scale, dtype=np.float32)
    if scale_array.size == 1:
        return points * float(scale_array.reshape(-1)[0])
    if scale_array.size == 3:
        return points * scale_array.reshape(1, 3)
    raise ValueError(f"Unsupported object scale shape {scale_array.shape}")


def _transform_points(points: np.ndarray, pose: Any) -> np.ndarray:
    """Transform object-frame points into the scene/world frame.

    Args:
        points: Scaled object-frame points with shape ``(N, 3)``.
        pose: Homogeneous object pose. If unavailable, identity is used.

    Returns:
        Transformed point cloud with shape ``(N, 3)``.
    """
    if pose is None:
        return points
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4):
        return points
    return points @ pose[:3, :3].T + pose[:3, 3].reshape(1, 3)


def _object_pose_matrix(grasp_data: dict) -> np.ndarray:
    """Read the object pose matrix from one grasp record.

    Args:
        grasp_data: Loaded human grasp dictionary.

    Returns:
        Homogeneous object pose with shape ``(4, 4)``. Missing poses fall back
        to identity because older synthetic records may not store scene pose.
    """
    pose = grasp_data.get("object", {}).get("pose")
    if pose is None:
        return np.eye(4, dtype=np.float64)
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"Unsupported object pose shape {pose.shape}")
    return pose


def _load_scene_points(pc_path: str, grasp_data: dict, max_points: int) -> np.ndarray:
    """Load scaled scene geometry points for one grasp record.

    Args:
        pc_path: Point-cloud path selected for the scene.
        grasp_data: Loaded grasp dictionary that provides object scale.
        max_points: Maximum number of deterministic points used for features.

    Returns:
        Scaled point cloud with shape ``(M, 3)``.
    """
    raw_points = np.load(pc_path, allow_pickle=True).astype(np.float32)
    if raw_points.ndim != 2 or raw_points.shape[1] != 3:
        raise ValueError(f"Expected point cloud with shape (N, 3), got {raw_points.shape} from {pc_path}")
    if max_points > 0 and raw_points.shape[0] > max_points:
        # Deterministic subsampling keeps the feature export reproducible.
        indices = np.linspace(0, raw_points.shape[0] - 1, num=max_points, dtype=np.int64)
        raw_points = raw_points[indices]
    object_data = grasp_data.get("object", {})
    return _scale_points(raw_points, object_data.get("rel_scale", 1.0))


def _determine_grasp_type_id_and_mirror(grasp_data: dict) -> tuple[int, bool]:
    """Determine grasp type id and whether to mirror a left-only grasp.

    Args:
        grasp_data: Loaded human grasp dictionary.

    Returns:
        Tuple ``(grasp_type_id, mirrored)`` where left-only single-hand grasps
        are converted to the right-hand frame for type ids ``1..3``.
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
        raise ValueError("Grasp record has no active hand contacts")
    if has_left and has_right:
        return (5 if (left_count > 3 or right_count > 3) else 4), False

    active_count = left_count if has_left else right_count
    if active_count <= 2:
        grasp_type_id = 1
    elif active_count == 3:
        grasp_type_id = 2
    else:
        grasp_type_id = 3
    return grasp_type_id, has_left and not has_right


def _hand_rotation_matrix(hand_data: dict, mirrored: bool = False) -> np.ndarray:
    """Read a wrist rotation matrix, optionally mirrored to the right hand.

    Args:
        hand_data: Hand metadata containing an axis-angle ``rot`` field.
        mirrored: Whether to mirror a left-hand rotation across the YZ plane.

    Returns:
        Rotation matrix with shape ``(3, 3)``.
    """
    raw_rot = np.asarray(hand_data["rot"], dtype=np.float64)
    if raw_rot.shape[-2:] == (3, 3):
        rot = raw_rot.reshape(-1, 3, 3)[0]
    else:
        rotvec = raw_rot.reshape(-1)
        if rotvec.size != 3:
            raise ValueError(f"Unsupported hand rotation shape {raw_rot.shape}")
        rot = SciR.from_rotvec(rotvec).as_matrix()
    if mirrored:
        rot = MIRROR @ rot @ MIRROR
    return rot.astype(np.float64)


def _unit_direction(vector: np.ndarray) -> np.ndarray:
    """Normalize a vector while handling degenerate zero-length cases.

    Args:
        vector: Raw direction vector.

    Returns:
        Unit vector. Degenerate input returns a zero vector.
    """
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < EPS:
        return np.zeros(3, dtype=np.float64)
    return vector / norm


def _extract_grasp_descriptor(grasp_data: dict, scaled_points: np.ndarray) -> dict:
    """Extract type, wrist orientation, and wrist direction descriptors.

    Args:
        grasp_data: Loaded human grasp dictionary.
        scaled_points: Scaled object-frame point cloud used to estimate the
            object center.

    Returns:
        Descriptor dictionary used for same-type diverse grasp clustering.
    """
    grasp_type_id, mirrored = _determine_grasp_type_id_and_mirror(grasp_data)
    object_pose = _object_pose_matrix(grasp_data)
    world_points = _transform_points(scaled_points, object_pose)
    object_center = world_points.mean(axis=0)
    hand_data = grasp_data.get("hand", {})
    descriptor = {
        "grasp_type_id": int(grasp_type_id),
        "hands": {},
        "object_rotation": object_pose[:3, :3].astype(np.float64),
    }

    if mirrored:
        left = hand_data["left"]
        center = MIRROR @ object_center
        trans = MIRROR @ np.asarray(left["trans"], dtype=np.float64).reshape(-1)[:3]
        descriptor["object_rotation"] = MIRROR @ descriptor["object_rotation"] @ MIRROR
        descriptor["hands"]["right"] = {
            "direction": _unit_direction(trans - center),
            "rotation": _hand_rotation_matrix(left, mirrored=True),
        }
        return descriptor

    for side in ["right", "left"]:
        hand = hand_data.get(side)
        if not hand or not any(hand.get("contacts", [])):
            continue
        trans = np.asarray(hand["trans"], dtype=np.float64).reshape(-1)[:3]
        descriptor["hands"][side] = {
            "direction": _unit_direction(trans - object_center),
            "rotation": _hand_rotation_matrix(hand, mirrored=False),
        }
    return descriptor


def _safe_covariance(points: np.ndarray, dim: int) -> np.ndarray:
    """Compute a numerically stable covariance matrix.

    Args:
        points: Centered points with shape ``(N, dim)``.
        dim: Expected coordinate dimension.

    Returns:
        Positive-definite covariance matrix with shape ``(dim, dim)``.
    """
    if points.shape[0] < 2:
        return np.eye(dim, dtype=np.float64) * EPS
    cov = np.cov(points, rowvar=False)
    cov = np.asarray(cov, dtype=np.float64).reshape(dim, dim)
    return cov + np.eye(dim, dtype=np.float64) * EPS


def _sorted_eigh(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return eigenvalues and eigenvectors sorted in descending order.

    Args:
        cov: Symmetric covariance matrix.

    Returns:
        Tuple ``(eigenvalues, eigenvectors)`` sorted by eigenvalue descending.
    """
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], EPS)
    eigvecs = eigvecs[:, order]
    return eigvals, eigvecs


def _projected_extents(points: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """Compute sorted extents after projecting points onto axes.

    Args:
        points: Centered point cloud with shape ``(N, D)``.
        axes: Projection axes with shape ``(D, D)``.

    Returns:
        Descending extents along the axes.
    """
    projected = points @ axes
    extents = np.ptp(projected, axis=0)
    return np.maximum(np.sort(extents)[::-1], EPS)


def _line_like_xy_bbox_extents(xy: np.ndarray) -> tuple[float, float]:
    """Estimate yaw-aligned XY box extents for degenerate 2D points.

    Args:
        xy: Centered XY points with shape ``(N, 2)``.

    Returns:
        Tuple ``(major, minor)`` with non-negative extents. The minor extent is
        clamped to ``EPS`` when points are line-like or all identical.
    """
    if xy.shape[0] < 2:
        return EPS, EPS
    _, singular_values, vh = np.linalg.svd(xy, full_matrices=False)
    if singular_values.size == 0 or float(singular_values[0]) < EPS:
        return EPS, EPS
    primary_axis = vh[0]
    secondary_axis = np.asarray([-primary_axis[1], primary_axis[0]], dtype=np.float64)
    projected = xy @ np.stack([primary_axis, secondary_axis], axis=1)
    extents = np.maximum(np.ptp(projected, axis=0), EPS)
    major, minor = np.sort(extents)[::-1]
    return float(major), float(minor)


def _yaw_aligned_xy_bbox_extents(xy: np.ndarray) -> tuple[float, float]:
    """Compute the minimum-area XY bounding box extents with free yaw.

    Args:
        xy: Centered XY points with shape ``(N, 2)``.

    Returns:
        Tuple ``(major, minor)`` for a 2D bounding box that may rotate around
        the global Z axis. Extents are sorted so ``major >= minor``.
    """
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"Expected XY points with shape (N, 2), got {xy.shape}")
    if xy.shape[0] < 3 or np.linalg.matrix_rank(xy, tol=EPS) < 2:
        return _line_like_xy_bbox_extents(xy)

    try:
        hull_points = xy[ConvexHull(xy).vertices]
    except QhullError:
        return _line_like_xy_bbox_extents(xy)

    edges = np.roll(hull_points, shift=-1, axis=0) - hull_points
    edge_lengths = np.linalg.norm(edges, axis=1)
    edges = edges[edge_lengths > EPS]
    if edges.size == 0:
        return _line_like_xy_bbox_extents(xy)

    # A minimum-area rectangle has one side parallel to a convex-hull edge.
    angles = np.mod(np.arctan2(edges[:, 1], edges[:, 0]), math.pi / 2.0)
    angles = np.unique(np.round(angles, decimals=12))

    best_extents = None
    best_area = float("inf")
    for angle in angles:
        cos_angle = math.cos(float(angle))
        sin_angle = math.sin(float(angle))
        axes = np.asarray(
            [
                [cos_angle, -sin_angle],
                [sin_angle, cos_angle],
            ],
            dtype=np.float64,
        )
        extents = np.maximum(np.ptp(hull_points @ axes, axis=0), EPS)
        area = float(extents[0] * extents[1])
        if area < best_area:
            best_area = area
            best_extents = extents

    if best_extents is None:
        return _line_like_xy_bbox_extents(xy)
    major, minor = np.sort(best_extents)[::-1]
    return float(major), float(minor)


def extract_yaw_invariant_geometry_feature(points: np.ndarray) -> np.ndarray:
    """Extract yaw-invariant bounding-box dimensions for one posed object.

    Args:
        points: Posed scene/world-frame point cloud with shape ``(N, 3)``.
            Translation and global Z yaw do not affect the returned feature.

    Returns:
        Three-dimensional feature vector ``[xy_major, xy_minor, z_extent]``
        aligned with ``GEOMETRY_FEATURE_NAMES``. The XY extents come from a
        minimum-area bounding box that may rotate around the global Z axis; the
        Z extent is measured along the global vertical axis.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {points.shape}")

    centered = points - points.mean(axis=0, keepdims=True)
    xy_major, xy_minor = _yaw_aligned_xy_bbox_extents(centered[:, :2])
    z_extent = float(max(np.ptp(centered[:, 2]), EPS))
    values = [xy_major, xy_minor, z_extent]
    return np.asarray(values, dtype=np.float32)


def _extract_posed_geometry_feature(grasp_data: dict, scaled_points: np.ndarray) -> np.ndarray:
    """Extract bbox geometry from canonical points transformed by object pose.

    Args:
        grasp_data: Loaded human grasp dictionary that provides object pose.
        scaled_points: Scaled canonical/object-frame point cloud with shape
            ``(N, 3)``.

    Returns:
        Feature vector aligned with ``GEOMETRY_FEATURE_NAMES`` after applying
        the object's scene pose to the canonical point cloud.
    """
    object_pose = _object_pose_matrix(grasp_data)
    world_points = _transform_points(scaled_points, object_pose)
    return extract_yaw_invariant_geometry_feature(world_points)


def _iter_human_data_roots(data_config: DictConfig) -> list[tuple[str, str]]:
    """Resolve one or more human grasp/object roots from the data config.

    Args:
        data_config: Hydra data config for ``HumanMultiDexDataset``.

    Returns:
        List of ``(grasp_path, object_path)`` pairs.
    """
    data_config = deepcopy(data_config)
    flatten_multidex_data_config(data_config)
    grasp_paths = [_absolute_path(path) for path in _as_list(cfg_get(data_config, "grasp_path", "paths.grasp_path"))]
    object_paths = [_absolute_path(path) for path in _as_list(cfg_get(data_config, "object_path", "paths.object_path"))]
    if len(grasp_paths) == 1 and len(object_paths) > 1:
        grasp_paths = grasp_paths * len(object_paths)
    if len(object_paths) == 1 and len(grasp_paths) > 1:
        object_paths = object_paths * len(grasp_paths)
    if len(grasp_paths) != len(object_paths):
        raise ValueError("grasp_path and object_path must have matching lengths")
    return list(zip(grasp_paths, object_paths))


def _build_scene_index(config: DictConfig) -> list[dict]:
    """Build a human-only scene table with geometry and grasp descriptors.

    Args:
        config: Full Hydra config.

    Returns:
        List of scene dictionaries. Each scene has geometry features, grasp
        descriptors, and an aggregated record count.
    """
    task_cfg = config.task
    data_config = deepcopy(config.data)
    flatten_multidex_data_config(data_config)
    splits = [str(split) for split in _as_list(cfg_get(task_cfg, "splits", default=["train", "test"]))]
    split_path = str(cfg_get(data_config, "split_path", "paths.split_path", default="valid_split"))
    pc_rel_path = str(cfg_get(data_config, "pc_path", "paths.pc_path", default="vision_data/complete_pc"))
    max_points = int(cfg_get(task_cfg, "feature.max_points", default=8192))

    scene_map = {}
    for component_idx, (grasp_root, object_root) in enumerate(_iter_human_data_roots(data_config)):
        for split in splits:
            split_json = os.path.join(object_root, split_path, f"{split}.json")
            object_ids = sorted(_load_json(split_json), key=_natural_sort_key)
            for object_id in object_ids:
                grasp_pattern = os.path.join(grasp_root, str(object_id), "**", "*.npy")
                for grasp_path in sorted(glob(grasp_pattern, recursive=True), key=_natural_sort_key):
                    grasp_data = np.load(grasp_path, allow_pickle=True).item()
                    sequence_id = _get_sequence_id(grasp_data)
                    key = (component_idx, split, _scene_key(str(object_id), sequence_id))
                    pc_path = None
                    points = None
                    scene = scene_map.get(key)
                    if scene is None:
                        pc_path = _select_pointcloud_path(object_root, pc_rel_path, str(object_id), sequence_id)
                        points = _load_scene_points(pc_path, grasp_data, max_points=max_points)
                        object_pose = _object_pose_matrix(grasp_data)
                        feature = _extract_posed_geometry_feature(grasp_data, points)
                        posed_points = _transform_points(points, object_pose).astype(np.float32)
                        scene = {
                            "component_idx": component_idx,
                            "split": split,
                            "scene_key": key[2],
                            "object_id": str(object_id),
                            "sequence_id": sequence_id,
                            "pc_path": pc_path,
                            "object_pose": object_pose,
                            "record_count": 0,
                            "feature": feature,
                            "point_cloud": posed_points,
                            "grasp_descriptors": [],
                        }
                        scene_map[key] = scene
                    if points is None:
                        points = _load_scene_points(scene["pc_path"], grasp_data, max_points=max_points)
                    grasp_descriptor = _extract_grasp_descriptor(grasp_data, points)
                    grasp_descriptor["grasp_path"] = os.path.abspath(grasp_path)
                    grasp_descriptor["grasp_index_in_scene"] = int(scene["record_count"])
                    grasp_descriptor["source_scene"] = str(grasp_data.get("object", {}).get("source_scene", ""))
                    scene["grasp_descriptors"].append(grasp_descriptor)
                    scene["record_count"] += 1

    scenes = list(scene_map.values())
    scenes.sort(
        key=lambda item: (
            item["component_idx"],
            item["split"],
            _natural_sort_key(item["object_id"]),
            _natural_sort_key(item["sequence_id"]),
            _natural_sort_key(item["scene_key"]),
        )
    )
    for scene in scenes:
        scene["effective_record_count"] = float(scene["record_count"])
    return scenes


def _feature_matrix(scenes: list[dict]) -> np.ndarray:
    """Stack scene geometry features into a matrix.

    Args:
        scenes: Scene dictionaries returned by ``_build_scene_index``.

    Returns:
        NumPy array with shape ``(N, F)``.
    """
    if not scenes:
        raise RuntimeError("No human scenes were indexed for scene budget generation")
    return np.stack([np.asarray(scene["feature"], dtype=np.float32) for scene in scenes], axis=0)


def _standardize_features(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize feature columns with zero mean and unit variance.

    Args:
        features: Raw feature matrix with shape ``(N, F)``.

    Returns:
        Tuple ``(standardized, mean, std)``.
    """
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return ((features - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def _compute_nearest_neighbor_indices(features: np.ndarray, nearest_scene_num: int) -> np.ndarray:
    """Compute nearest scene indices in standardized feature space.

    Args:
        features: Standardized feature matrix with shape ``(N, F)``.
        nearest_scene_num: Number of nearest scenes in each local class.

    Returns:
        Integer array with shape ``(N, K)`` containing nearest scene indices.
    """
    scene_num = features.shape[0]
    if scene_num == 0:
        raise RuntimeError("Cannot compute nearest neighbors for an empty feature matrix")
    if nearest_scene_num <= 0:
        raise ValueError(f"legacy_nearest_n.nearest_scene_num must be positive, got {nearest_scene_num}")

    k = min(int(nearest_scene_num), scene_num)
    diff = features[:, None, :] - features[None, :, :]
    distance_sq = np.sum(diff * diff, axis=-1)
    return np.argsort(distance_sq, axis=1)[:, :k].astype(np.int64)


def _yaw_rotation_matrix(yaw_rad: float) -> np.ndarray:
    """Build a rotation matrix around the vertical Z axis.

    Args:
        yaw_rad: Yaw angle in radians.

    Returns:
        Rotation matrix with shape ``(3, 3)``.
    """
    cos_yaw = math.cos(float(yaw_rad))
    sin_yaw = math.sin(float(yaw_rad))
    return np.asarray(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _wrap_angle_rad(angle: float) -> float:
    """Wrap an angle into ``[-pi, pi)``.

    Args:
        angle: Raw angle in radians.

    Returns:
        Wrapped angle in radians.
    """
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


def _horizontal_angle(vector: np.ndarray) -> float | None:
    """Read the yaw angle of a vector's XY projection.

    Args:
        vector: 3D direction vector.

    Returns:
        Yaw angle in radians, or ``None`` for near-vertical vectors.
    """
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    if float(np.linalg.norm(vector[:2])) < EPS:
        return None
    return float(math.atan2(vector[1], vector[0]))


def _optimal_rotation_left_yaw(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    """Find the yaw that best aligns ``yaw * rot_b`` to ``rot_a``.

    Args:
        rot_a: Target rotation matrix with shape ``(3, 3)``.
        rot_b: Source rotation matrix with shape ``(3, 3)``.

    Returns:
        Yaw angle in radians maximizing ``trace(rot_a.T @ yaw @ rot_b)``.
    """
    matrix = np.asarray(rot_b, dtype=np.float64) @ np.asarray(rot_a, dtype=np.float64).T
    return _wrap_angle_rad(math.atan2(matrix[0, 1] - matrix[1, 0], matrix[0, 0] + matrix[1, 1]))


def _yaw_alignment_candidates(record_a: dict, record_b: dict) -> list[float]:
    """Generate object-coupled yaw candidates for grasp comparison.

    Args:
        record_a: First grasp descriptor.
        record_b: Second grasp descriptor. The returned yaw is applied to this
            descriptor's wrist direction and rotation before comparison.

    Returns:
        Deduplicated yaw candidates in radians. Query-aligned records from the
        same neighborhood frame use zero additional yaw; otherwise the fallback
        candidate is derived from object pose, so wrist yaw is only ignored
        when the object also rotated by the same global vertical yaw.
    """
    alignment_frame_a = record_a.get("yaw_alignment_frame")
    alignment_frame_b = record_b.get("yaw_alignment_frame")
    if alignment_frame_a is not None and alignment_frame_a == alignment_frame_b:
        return [0.0]

    object_rot_a = record_a.get("object_rotation")
    object_rot_b = record_b.get("object_rotation")
    if object_rot_a is None or object_rot_b is None:
        candidates = [0.0]
    else:
        candidates = [_optimal_rotation_left_yaw(object_rot_a, object_rot_b)]

    deduped = []
    for candidate in candidates:
        candidate = _wrap_angle_rad(candidate)
        if not any(abs(_wrap_angle_rad(candidate - existing)) < 1e-6 for existing in deduped):
            deduped.append(candidate)
    return deduped


def _rotation_distance_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    """Compute geodesic distance between two wrist rotations.

    Args:
        rot_a: First rotation matrix with shape ``(3, 3)``.
        rot_b: Second rotation matrix with shape ``(3, 3)``.

    Returns:
        Angular distance in degrees.
    """
    relative = np.asarray(rot_a, dtype=np.float64).T @ np.asarray(rot_b, dtype=np.float64)
    return float(np.rad2deg(SciR.from_matrix(relative).magnitude()))


def _direction_distance_deg(direction_a: np.ndarray, direction_b: np.ndarray) -> float:
    """Compute angular distance between two wrist direction vectors.

    Args:
        direction_a: First unit direction vector.
        direction_b: Second unit direction vector.

    Returns:
        Angular distance in degrees. Degenerate zero vectors return ``0``.
    """
    norm_a = float(np.linalg.norm(direction_a))
    norm_b = float(np.linalg.norm(direction_b))
    if norm_a < EPS or norm_b < EPS:
        return 0.0
    dot = float(np.clip(np.dot(direction_a, direction_b) / (norm_a * norm_b), -1.0, 1.0))
    return float(np.rad2deg(math.acos(dot)))


def _same_posed_object(
    scene_a: dict,
    scene_b: dict,
    translation_threshold_m: float,
    rotation_threshold_deg: float,
) -> bool:
    """Check whether two scenes are the same posed object up to global Z yaw.

    Args:
        scene_a: First scene dictionary from ``_build_scene_index``.
        scene_b: Second scene dictionary from ``_build_scene_index``.
        translation_threshold_m: Maximum object-origin translation difference.
        rotation_threshold_deg: Maximum object rotation difference after
            optimal global Z-yaw alignment.

    Returns:
        True if the scenes share the same canonical object identity and their
        object poses differ only by a small translation residual and global
        vertical yaw.
    """
    if int(scene_a["component_idx"]) != int(scene_b["component_idx"]):
        return False
    if str(scene_a["object_id"]) != str(scene_b["object_id"]):
        return False

    pose_a = np.asarray(scene_a["object_pose"], dtype=np.float64)
    pose_b = np.asarray(scene_b["object_pose"], dtype=np.float64)
    translation_distance = float(np.linalg.norm(pose_a[:3, 3] - pose_b[:3, 3]))
    if translation_distance > float(translation_threshold_m):
        return False

    yaw = _optimal_rotation_left_yaw(pose_a[:3, :3], pose_b[:3, :3])
    aligned_b = _yaw_rotation_matrix(yaw) @ pose_b[:3, :3]
    rotation_distance = _rotation_distance_deg(pose_a[:3, :3], aligned_b)
    return rotation_distance <= float(rotation_threshold_deg)


def _scene_to_scene_yaw(query_scene: dict, source_scene: dict) -> float:
    """Compute the global Z-yaw that aligns a source scene to a query scene.

    Args:
        query_scene: Scene defining the common yaw-aligned neighborhood frame.
        source_scene: Neighbor scene whose object and wrists will be rotated.

    Returns:
        Yaw angle in radians. Applying this yaw to the source scene's object
        pose best aligns its object rotation to the query scene around the
        global vertical axis.
    """
    query_pose = np.asarray(query_scene["object_pose"], dtype=np.float64)
    source_pose = np.asarray(source_scene["object_pose"], dtype=np.float64)
    return _optimal_rotation_left_yaw(query_pose[:3, :3], source_pose[:3, :3])


def _align_grasp_descriptor_to_query(record: dict, yaw_rad: float, query_scene: dict) -> dict:
    """Rotate one grasp descriptor into a query-centered Z-yaw frame.

    Args:
        record: Raw grasp descriptor from a neighbor scene.
        yaw_rad: Global Z-yaw applied to both object and active wrists.
        query_scene: Scene whose key names the common alignment frame.

    Returns:
        A copied descriptor whose wrist directions and wrist rotations have
        been rotated by ``yaw_rad``. The same rotation is applied to the stored
        object rotation, preserving each wrist pose relative to its object.
    """
    yaw_matrix = _yaw_rotation_matrix(yaw_rad)
    aligned = {
        "grasp_type_id": int(record["grasp_type_id"]),
        "hands": {},
        "object_rotation": yaw_matrix @ np.asarray(record["object_rotation"], dtype=np.float64),
        "yaw_alignment_frame": f"{query_scene['component_idx']}:{query_scene['split']}:{query_scene['scene_key']}",
        "yaw_alignment_rad": float(yaw_rad),
    }
    for side, hand in record.get("hands", {}).items():
        aligned["hands"][side] = {
            "direction": yaw_matrix @ np.asarray(hand["direction"], dtype=np.float64),
            "rotation": yaw_matrix @ np.asarray(hand["rotation"], dtype=np.float64),
        }
    return aligned


def _align_neighborhood_grasp_records_to_query(query_scene: dict, neighbor_scenes: list[dict]) -> list[dict]:
    """Align all neighbor grasp descriptors into the query scene yaw frame.

    Args:
        query_scene: Scene whose object-pose yaw defines the comparison frame.
        neighbor_scenes: Nearest scenes selected in yaw-invariant feature space.

    Returns:
        List of query-aligned grasp descriptors from all neighbor scenes.
    """
    aligned_records = []
    for source_scene in neighbor_scenes:
        yaw = _scene_to_scene_yaw(query_scene, source_scene)
        for record in source_scene.get("grasp_descriptors", []):
            aligned_records.append(_align_grasp_descriptor_to_query(record, yaw, query_scene))
    return aligned_records


def _count_posed_objects(
    scenes: list[dict],
    translation_threshold_m: float,
    rotation_threshold_deg: float,
) -> int:
    """Count posed-object groups in a nearest-scene neighborhood.

    Args:
        scenes: Neighbor scene dictionaries.
        translation_threshold_m: Translation threshold for merging same-object
            scenes.
        rotation_threshold_deg: Rotation threshold after optimal Z-yaw
            alignment for merging same-object scenes.

    Returns:
        Number of posed-object groups. Different canonical objects are always
        counted as different groups.
    """
    if translation_threshold_m < 0.0 or rotation_threshold_deg < 0.0:
        raise ValueError("Posed-object thresholds must be non-negative")

    representatives = []
    for scene in scenes:
        if not any(
            _same_posed_object(
                scene,
                representative,
                translation_threshold_m=translation_threshold_m,
                rotation_threshold_deg=rotation_threshold_deg,
            )
            for representative in representatives
        ):
            representatives.append(scene)
    return int(len(representatives))


def _grasp_type_name(grasp_type_id: int) -> str:
    """Return a stable human-readable grasp type name.

    Args:
        grasp_type_id: Integer grasp type id.

    Returns:
        Name from ``GRASP_TYPES`` when available, otherwise the numeric id as a
        string.
    """
    if 0 <= int(grasp_type_id) < len(GRASP_TYPES):
        return str(GRASP_TYPES[int(grasp_type_id)])
    return str(grasp_type_id)


def _pose_class_rotation_residual_deg(
    scene: dict,
    representative: dict,
) -> tuple[float, float, float]:
    """Measure object-pose residual modulo global yaw and local Z symmetry.

    Args:
        scene: Candidate scene dictionary.
        representative: Existing pose-class representative scene.

    Returns:
        Tuple ``(residual_deg, yaw_to_representative_deg, local_z_symmetry_deg)``.
        The residual ignores pure global vertical Z-yaw differences and treats
        a 180-degree rotation around the object's canonical local Z axis as a
        symmetry candidate.
    """
    scene_pose = np.asarray(scene["object_pose"], dtype=np.float64)
    representative_pose = np.asarray(representative["object_pose"], dtype=np.float64)
    scene_rotation = scene_pose[:3, :3]
    representative_rotation = representative_pose[:3, :3]

    best_residual_deg = float("inf")
    best_yaw_deg = 0.0
    best_local_z_symmetry_deg = 0.0
    for local_z_symmetry_rad in (0.0, math.pi):
        # Right multiplication applies the symmetry in the object's canonical
        # local frame before the global world-Z yaw alignment is estimated.
        candidate_rotation = scene_rotation @ _yaw_rotation_matrix(local_z_symmetry_rad)
        yaw = _optimal_rotation_left_yaw(representative_rotation, candidate_rotation)
        aligned_scene_rotation = _yaw_rotation_matrix(yaw) @ candidate_rotation
        residual_deg = _rotation_distance_deg(representative_rotation, aligned_scene_rotation)
        if residual_deg < best_residual_deg:
            best_residual_deg = float(residual_deg)
            best_yaw_deg = float(np.rad2deg(yaw))
            best_local_z_symmetry_deg = float(np.rad2deg(local_z_symmetry_rad))

    return best_residual_deg, best_yaw_deg, best_local_z_symmetry_deg


def _bbox_proportion_distance(scene: dict, representative: dict) -> float:
    """Compare posed bbox axis proportions for two scenes.

    Args:
        scene: Candidate scene dictionary with a ``feature`` vector.
        representative: Existing pose-class representative scene with a
            ``feature`` vector.

    Returns:
        Maximum per-axis relative difference between L2-normalized posed bbox
        dimensions. The feature uses yaw-free XY major/minor extents plus world
        Z height, so this comparison is invariant to global Z yaw.
    """
    scene_bbox = np.maximum(np.asarray(scene["feature"], dtype=np.float64), EPS)
    representative_bbox = np.maximum(np.asarray(representative["feature"], dtype=np.float64), EPS)
    scene_proportion = scene_bbox / max(float(np.linalg.norm(scene_bbox)), EPS)
    representative_proportion = representative_bbox / max(float(np.linalg.norm(representative_bbox)), EPS)
    denominator = np.maximum(np.maximum(scene_proportion, representative_proportion), EPS)
    return float(np.max(np.abs(scene_proportion - representative_proportion) / denominator))


def _assign_pose_classes(
    scenes: list[dict],
    rotation_threshold_deg: float,
    bbox_proportion_threshold: float,
) -> dict[tuple, dict]:
    """Assign object scenes to pose classes using rotation and bbox fallback.

    Args:
        scenes: Scene dictionaries returned by ``_build_scene_index``.
        rotation_threshold_deg: Maximum non-yaw rotation residual for merging a
            scene into an existing pose class.
        bbox_proportion_threshold: Maximum posed bbox proportion distance for
            the fallback merge rule after the rotation rule fails.

    Returns:
        Mapping from ``(component_idx, split, scene_key)`` to pose-class
        metadata. Pose class ids are local to each canonical object within each
        component and split.
    """
    if rotation_threshold_deg < 0.0:
        raise ValueError("Pose class rotation threshold must be non-negative")
    if bbox_proportion_threshold < 0.0:
        raise ValueError("Pose class bbox proportion threshold must be non-negative")

    representatives_by_object = {}
    assignments = {}
    for scene in scenes:
        object_key = (int(scene["component_idx"]), str(scene["split"]), str(scene["object_id"]))
        representatives = representatives_by_object.setdefault(object_key, [])
        best_rotation_match = None
        best_bbox_match = None
        for pose_class_idx, representative in enumerate(representatives):
            residual_deg, yaw_deg, local_z_symmetry_deg = _pose_class_rotation_residual_deg(
                scene,
                representative,
            )
            bbox_distance = _bbox_proportion_distance(scene, representative)
            candidate = {
                "pose_class_id": int(pose_class_idx),
                "pose_class_rotation_residual_deg": float(residual_deg),
                "pose_class_yaw_to_representative_deg": float(yaw_deg),
                "pose_class_local_z_symmetry_deg": float(local_z_symmetry_deg),
                "pose_class_bbox_proportion_distance": float(bbox_distance),
            }
            if residual_deg <= float(rotation_threshold_deg):
                if (
                    best_rotation_match is None
                    or residual_deg < best_rotation_match["pose_class_rotation_residual_deg"]
                ):
                    best_rotation_match = candidate
            elif bbox_distance <= float(bbox_proportion_threshold):
                if (
                    best_bbox_match is None
                    or bbox_distance < best_bbox_match["pose_class_bbox_proportion_distance"]
                ):
                    best_bbox_match = candidate

        if best_rotation_match is not None:
            selected = best_rotation_match
            selected["pose_class_match_method"] = "rotation"
        elif best_bbox_match is not None:
            selected = best_bbox_match
            selected["pose_class_match_method"] = "bbox_proportion"
        else:
            best_idx = len(representatives)
            representatives.append(scene)
            selected = {
                "pose_class_id": int(best_idx),
                "pose_class_rotation_residual_deg": 0.0,
                "pose_class_yaw_to_representative_deg": 0.0,
                "pose_class_local_z_symmetry_deg": 0.0,
                "pose_class_bbox_proportion_distance": 0.0,
                "pose_class_match_method": "new",
            }

        representative = representatives[int(selected["pose_class_id"])]
        assignment_key = (int(scene["component_idx"]), str(scene["split"]), str(scene["scene_key"]))
        assignments[assignment_key] = {
            "pose_class_id": int(selected["pose_class_id"]),
            "pose_class_key": f"{scene['object_id']}/pose_{int(selected['pose_class_id']):03d}",
            "pose_class_representative_scene_key": str(representative["scene_key"]),
            "pose_class_rotation_residual_deg": float(selected["pose_class_rotation_residual_deg"]),
            "pose_class_yaw_to_representative_deg": float(selected["pose_class_yaw_to_representative_deg"]),
            "pose_class_local_z_symmetry_deg": float(selected["pose_class_local_z_symmetry_deg"]),
            "pose_class_bbox_proportion_distance": float(selected["pose_class_bbox_proportion_distance"]),
            "pose_class_match_method": str(selected["pose_class_match_method"]),
        }
    return assignments


def _build_scene_budget_label_hierarchy_rows(scenes: list[dict], task_cfg: DictConfig) -> tuple[list[dict], dict]:
    """Build a canonical-object / pose-class / type / grasp CSV table.

    Args:
        scenes: Scene dictionaries returned by ``_build_scene_index``.
        task_cfg: Task config containing the pose-class threshold.

    Returns:
        Tuple ``(rows, summary)``. Each row corresponds to one grasp record and
        includes the canonical object id, pose class, grasp type, and source
        grasp path.
    """
    rotation_threshold_deg = float(
        cfg_get(task_cfg, "label_structure.pose_class_rotation_threshold_deg", default=45.0)
    )
    bbox_proportion_threshold = float(
        cfg_get(task_cfg, "label_structure.pose_class_bbox_proportion_threshold", default=0.2)
    )
    assignments = _assign_pose_classes(
        scenes,
        rotation_threshold_deg=rotation_threshold_deg,
        bbox_proportion_threshold=bbox_proportion_threshold,
    )

    rows = []
    grasp_rank_by_group = {}
    pose_classes_by_object = {}
    grasp_types_by_pose = {}
    for scene in scenes:
        assignment_key = (int(scene["component_idx"]), str(scene["split"]), str(scene["scene_key"]))
        pose_assignment = assignments[assignment_key]
        object_pose = np.asarray(scene["object_pose"], dtype=np.float64)
        pose_class_id = int(pose_assignment["pose_class_id"])
        object_group_key = (int(scene["component_idx"]), str(scene["split"]), str(scene["object_id"]))
        pose_classes_by_object.setdefault(object_group_key, set()).add(pose_class_id)

        for descriptor in scene.get("grasp_descriptors", []):
            grasp_type_id = int(descriptor["grasp_type_id"])
            type_group_key = object_group_key + (pose_class_id, grasp_type_id)
            grasp_index_in_type = grasp_rank_by_group.get(type_group_key, 0)
            grasp_rank_by_group[type_group_key] = grasp_index_in_type + 1
            grasp_types_by_pose.setdefault(object_group_key + (pose_class_id,), set()).add(grasp_type_id)

            rows.append(
                {
                    "component_idx": int(scene["component_idx"]),
                    "split": str(scene["split"]),
                    "scene_id": str(scene["scene_key"]),
                    "scene_key": str(scene["scene_key"]),
                    "canonical_object_id": str(scene["object_id"]),
                    "sequence_id": str(scene["sequence_id"]),
                    "pose_class_id": pose_class_id,
                    "pose_class_key": pose_assignment["pose_class_key"],
                    "pose_class_representative_scene_id": pose_assignment["pose_class_representative_scene_key"],
                    "pose_class_representative_scene_key": pose_assignment["pose_class_representative_scene_key"],
                    "pose_class_rotation_threshold_deg": rotation_threshold_deg,
                    "pose_class_rotation_residual_deg": pose_assignment["pose_class_rotation_residual_deg"],
                    "pose_class_yaw_to_representative_deg": pose_assignment["pose_class_yaw_to_representative_deg"],
                    "pose_class_local_z_symmetry_deg": pose_assignment["pose_class_local_z_symmetry_deg"],
                    "pose_class_bbox_proportion_threshold": bbox_proportion_threshold,
                    "pose_class_bbox_proportion_distance": pose_assignment["pose_class_bbox_proportion_distance"],
                    "pose_class_match_method": pose_assignment["pose_class_match_method"],
                    "grasp_type_id": grasp_type_id,
                    "grasp_type_name": _grasp_type_name(grasp_type_id),
                    "grasp_index_in_scene": int(descriptor.get("grasp_index_in_scene", 0)),
                    "grasp_index_in_type": int(grasp_index_in_type),
                    "grasp_path": str(descriptor.get("grasp_path", "")),
                    "pc_path": str(scene["pc_path"]),
                    "object_pose_tx": float(object_pose[0, 3]),
                    "object_pose_ty": float(object_pose[1, 3]),
                    "object_pose_tz": float(object_pose[2, 3]),
                    "object_pose_flat": json.dumps(object_pose.reshape(-1).tolist(), separators=(",", ":")),
                    "source_scene": str(descriptor.get("source_scene", "")),
                }
            )

    rows.sort(
        key=lambda row: (
            int(row["component_idx"]),
            str(row["split"]),
            _natural_sort_key(row["canonical_object_id"]),
            int(row["pose_class_id"]),
            int(row["grasp_type_id"]),
            int(row["grasp_index_in_type"]),
            _natural_sort_key(row["scene_key"]),
        )
    )
    summary = {
        "pose_class_rotation_threshold_deg": rotation_threshold_deg,
        "pose_class_global_z_yaw_invariant": True,
        "pose_class_local_z_symmetry_degrees": [0.0, 180.0],
        "pose_class_bbox_fallback_enabled": True,
        "pose_class_bbox_proportion_threshold": bbox_proportion_threshold,
        "canonical_object_num": int(len(pose_classes_by_object)),
        "pose_class_num": int(sum(len(values) for values in pose_classes_by_object.values())),
        "object_pose_class_min": int(min((len(values) for values in pose_classes_by_object.values()), default=0)),
        "object_pose_class_max": int(max((len(values) for values in pose_classes_by_object.values()), default=0)),
        "pose_grasp_type_group_num": int(len(grasp_types_by_pose)),
        "grasp_record_num": int(len(rows)),
    }
    return rows, summary


def _same_type_pose_distance(
    record_a: dict,
    record_b: dict,
    orientation_threshold_deg: float,
    direction_threshold_deg: float,
) -> tuple[float, float]:
    """Measure yaw-invariant same-type orientation and direction differences.

    Args:
        record_a: First grasp descriptor.
        record_b: Second grasp descriptor.
        orientation_threshold_deg: Orientation threshold used to score the
            object-coupled yaw candidate.
        direction_threshold_deg: Direction threshold used to score the
            object-coupled yaw candidate.

    Returns:
        Tuple ``(orientation_deg, direction_deg)`` after applying the selected
        global Z-yaw alignment to ``record_b``. Query-aligned records from the
        same neighborhood frame use no extra yaw. If active hand sets do not
        match, both distances are infinite.
    """
    hands_a = record_a.get("hands", {})
    hands_b = record_b.get("hands", {})
    if set(hands_a.keys()) != set(hands_b.keys()):
        return float("inf"), float("inf")
    if not hands_a:
        return float("inf"), float("inf")

    best_orientation = float("inf")
    best_direction = float("inf")
    best_score = float("inf")
    orientation_norm = max(float(orientation_threshold_deg), EPS)
    direction_norm = max(float(direction_threshold_deg), EPS)
    for yaw in _yaw_alignment_candidates(record_a, record_b):
        yaw_matrix = _yaw_rotation_matrix(yaw)
        orientation_distances = []
        direction_distances = []
        for side in sorted(hands_a.keys()):
            aligned_rotation = yaw_matrix @ hands_b[side]["rotation"]
            aligned_direction = yaw_matrix @ hands_b[side]["direction"]
            orientation_distances.append(_rotation_distance_deg(hands_a[side]["rotation"], aligned_rotation))
            direction_distances.append(_direction_distance_deg(hands_a[side]["direction"], aligned_direction))
        orientation_deg = float(max(orientation_distances))
        direction_deg = float(max(direction_distances))
        score = max(orientation_deg / orientation_norm, direction_deg / direction_norm)
        if score < best_score:
            best_score = score
            best_orientation = orientation_deg
            best_direction = direction_deg
    return best_orientation, best_direction


def _count_diverse_grasp_classes(
    grasp_records: list[dict],
    orientation_threshold_deg: float,
    direction_threshold_deg: float,
) -> int:
    """Count diverse grasp classes within one local scene neighborhood.

    Args:
        grasp_records: Grasp descriptors from the nearest scene neighborhood.
            The proxy path passes query-aligned descriptors, so all records are
            already in one common Z-yaw frame before clustering.
        orientation_threshold_deg: Maximum same-class wrist orientation
            distance in degrees for records with the same grasp type.
        direction_threshold_deg: Maximum same-class wrist direction distance
            in degrees for records with the same grasp type.

    Returns:
        Number of clustered diverse grasp classes. Different grasp types are
        always counted as different classes.
    """
    if orientation_threshold_deg < 0.0 or direction_threshold_deg < 0.0:
        raise ValueError("Diversity thresholds must be non-negative")

    representatives_by_type = {}
    for record in grasp_records:
        grasp_type_id = int(record["grasp_type_id"])
        representatives = representatives_by_type.setdefault(grasp_type_id, [])
        matched = False
        for representative in representatives:
            orientation_deg, direction_deg = _same_type_pose_distance(
                record,
                representative,
                orientation_threshold_deg=orientation_threshold_deg,
                direction_threshold_deg=direction_threshold_deg,
            )
            if orientation_deg <= orientation_threshold_deg and direction_deg <= direction_threshold_deg:
                matched = True
                break
        if not matched:
            representatives.append(record)
    return int(sum(len(representatives) for representatives in representatives_by_type.values()))


def _compute_legacy_nearest_n_labels(scenes: list[dict], task_cfg: DictConfig) -> tuple[list[dict], dict]:
    """Compute legacy nearest-scene diverse-grasp-count labels.

    Args:
        scenes: Scene dictionaries with raw geometry features and counts.
        task_cfg: Task config containing legacy nearest-neighbor settings.

    Returns:
        Tuple of per-scene budget rows and summary statistics.
    """
    raw_features = _feature_matrix(scenes)
    standardized_features, feature_mean, feature_std = _standardize_features(raw_features)
    nearest_scene_num = int(cfg_get(task_cfg, "legacy_nearest_n.nearest_scene_num", default=32))
    orientation_threshold_deg = float(cfg_get(task_cfg, "legacy_nearest_n.orientation_threshold_deg", default=30.0))
    direction_threshold_deg = float(cfg_get(task_cfg, "legacy_nearest_n.direction_threshold_deg", default=30.0))
    posed_object_translation_threshold_m = float(
        cfg_get(task_cfg, "legacy_nearest_n.posed_object_translation_threshold_m", default=0.03)
    )
    posed_object_rotation_threshold_deg = float(
        cfg_get(task_cfg, "legacy_nearest_n.posed_object_rotation_threshold_deg", default=10.0)
    )
    clip_min = float(cfg_get(task_cfg, "legacy_nearest_n.clip_min", default=0.5))
    clip_max = float(cfg_get(task_cfg, "legacy_nearest_n.clip_max", default=3.0))
    if clip_min <= 0.0 or clip_max < clip_min:
        raise ValueError(f"Invalid legacy_nearest_n clip range [{clip_min}, {clip_max}]")

    neighbor_indices = _compute_nearest_neighbor_indices(standardized_features, nearest_scene_num)
    neighbor_record_counts = []
    posed_object_counts = []
    grasp_per_posed_object = []
    raw_proxy = []
    for scene_idx, indices in enumerate(neighbor_indices):
        query_scene = scenes[scene_idx]
        neighbor_scenes = [scenes[int(neighbor_idx)] for neighbor_idx in indices]
        grasp_records = _align_neighborhood_grasp_records_to_query(query_scene, neighbor_scenes)
        total_grasp_count = len(grasp_records)
        posed_object_count = _count_posed_objects(
            neighbor_scenes,
            translation_threshold_m=posed_object_translation_threshold_m,
            rotation_threshold_deg=posed_object_rotation_threshold_deg,
        )
        neighbor_record_counts.append(total_grasp_count)
        posed_object_counts.append(posed_object_count)
        grasp_per_posed_object.append(total_grasp_count / max(posed_object_count, 1))
        raw_proxy.append(
            float(
                _count_diverse_grasp_classes(
                    grasp_records,
                    orientation_threshold_deg=orientation_threshold_deg,
                    direction_threshold_deg=direction_threshold_deg,
                )
            )
        )
    raw_proxy = np.asarray(raw_proxy, dtype=np.float64)
    neighbor_record_counts = np.asarray(neighbor_record_counts, dtype=np.int64)
    posed_object_counts = np.asarray(posed_object_counts, dtype=np.int64)
    grasp_per_posed_object = np.asarray(grasp_per_posed_object, dtype=np.float64)

    mean_raw_proxy = float(raw_proxy.mean()) if raw_proxy.size else 1.0
    if mean_raw_proxy <= 0.0:
        mean_raw_proxy = 1.0
    multipliers = np.clip(raw_proxy / mean_raw_proxy, clip_min, clip_max).astype(np.float32)
    mean_grasp_per_posed_object = float(grasp_per_posed_object.mean()) if grasp_per_posed_object.size else 1.0
    if mean_grasp_per_posed_object <= 0.0:
        mean_grasp_per_posed_object = 1.0
    grasp_per_posed_object_multipliers = np.clip(
        grasp_per_posed_object / mean_grasp_per_posed_object,
        clip_min,
        clip_max,
    ).astype(np.float32)

    rows = []
    for scene_idx, scene in enumerate(scenes):
        indices = neighbor_indices[scene_idx]
        feature_values = raw_features[scene_idx]
        row = {
            "component_idx": int(scene["component_idx"]),
            "split": scene["split"],
            "scene_key": scene["scene_key"],
            "object_id": scene["object_id"],
            "sequence_id": scene["sequence_id"],
            "pc_path": scene["pc_path"],
            "record_count": int(scene["record_count"]),
            "effective_record_count": float(scene["effective_record_count"]),
            "class_scene_num": int(len(indices)),
            "neighbor_record_num": int(neighbor_record_counts[scene_idx]),
            "total_grasp_count": int(neighbor_record_counts[scene_idx]),
            "posed_object_count": int(posed_object_counts[scene_idx]),
            "grasp_per_posed_object": float(grasp_per_posed_object[scene_idx]),
            "grasp_per_posed_object_multiplier": float(grasp_per_posed_object_multipliers[scene_idx]),
            "diverse_grasp_class_count": float(raw_proxy[scene_idx]),
            "budget_multiplier": float(multipliers[scene_idx]),
            "log_budget_multiplier": float(math.log(max(float(multipliers[scene_idx]), EPS))),
            "neighbor_scene_keys": ";".join(scenes[int(idx)]["scene_key"] for idx in indices),
            "_point_cloud": np.asarray(scene["point_cloud"], dtype=np.float32),
        }
        for feature_name, feature_value in zip(GEOMETRY_FEATURE_NAMES, feature_values):
            row[feature_name] = float(feature_value)
        rows.append(row)

    summary = {
        "scene_num": int(len(scenes)),
        "feature_names": GEOMETRY_FEATURE_NAMES,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "budget_label": {
            "neighborhood_mode": "nearest_n",
            "nearest_scene_num": int(min(max(nearest_scene_num, 1), len(scenes))),
            "orientation_threshold_deg": orientation_threshold_deg,
            "direction_threshold_deg": direction_threshold_deg,
            "same_type_yaw_invariant": True,
            "same_type_yaw_alignment_axis": "z",
            "same_type_yaw_source": "query_object_pose",
            "same_type_yaw_alignment_frame": "query_scene",
            "posed_object_translation_threshold_m": posed_object_translation_threshold_m,
            "posed_object_rotation_threshold_deg": posed_object_rotation_threshold_deg,
            "raw_mean": mean_raw_proxy,
            "raw_min": float(raw_proxy.min()) if raw_proxy.size else 0.0,
            "raw_max": float(raw_proxy.max()) if raw_proxy.size else 0.0,
            "class_scene_num_mean": float(np.mean([len(indices) for indices in neighbor_indices])),
            "class_scene_num_min": int(min(len(indices) for indices in neighbor_indices)),
            "class_scene_num_max": int(max(len(indices) for indices in neighbor_indices)),
            "neighbor_record_num_mean": float(neighbor_record_counts.mean()) if neighbor_record_counts.size else 0.0,
            "neighbor_record_num_min": int(neighbor_record_counts.min()) if neighbor_record_counts.size else 0,
            "neighbor_record_num_max": int(neighbor_record_counts.max()) if neighbor_record_counts.size else 0,
            "posed_object_count_mean": float(posed_object_counts.mean()) if posed_object_counts.size else 0.0,
            "posed_object_count_min": int(posed_object_counts.min()) if posed_object_counts.size else 0,
            "posed_object_count_max": int(posed_object_counts.max()) if posed_object_counts.size else 0,
            "grasp_per_posed_object_mean": mean_grasp_per_posed_object,
            "grasp_per_posed_object_min": float(grasp_per_posed_object.min())
            if grasp_per_posed_object.size
            else 0.0,
            "grasp_per_posed_object_max": float(grasp_per_posed_object.max())
            if grasp_per_posed_object.size
            else 0.0,
            "grasp_per_posed_object_multiplier_mean": float(grasp_per_posed_object_multipliers.mean())
            if grasp_per_posed_object_multipliers.size
            else 0.0,
            "grasp_per_posed_object_multiplier_min": float(grasp_per_posed_object_multipliers.min())
            if grasp_per_posed_object_multipliers.size
            else 0.0,
            "grasp_per_posed_object_multiplier_max": float(grasp_per_posed_object_multipliers.max())
            if grasp_per_posed_object_multipliers.size
            else 0.0,
            "diverse_grasp_class_count_mean": float(raw_proxy.mean()) if raw_proxy.size else 0.0,
            "diverse_grasp_class_count_min": float(raw_proxy.min()) if raw_proxy.size else 0.0,
            "diverse_grasp_class_count_max": float(raw_proxy.max()) if raw_proxy.size else 0.0,
            "multiplier_mean": float(multipliers.mean()) if multipliers.size else 0.0,
            "multiplier_min": float(multipliers.min()) if multipliers.size else 0.0,
            "multiplier_max": float(multipliers.max()) if multipliers.size else 0.0,
            "clip_min": clip_min,
            "clip_max": clip_max,
        },
    }
    return rows, summary


def _aggregate_pose_class_feature(member_scenes: list[dict], reducer: str) -> np.ndarray:
    """Reduce member scene bbox features into one pose-class feature.

    Args:
        member_scenes: Original scene dictionaries assigned to one pose class.
        reducer: Feature reducer name. Supported values are ``mean`` and
            ``medoid``.

    Returns:
        Feature vector aligned with ``GEOMETRY_FEATURE_NAMES``.
    """
    if not member_scenes:
        raise ValueError("Cannot aggregate pose-class features without member scenes")
    reducer = str(reducer).lower()
    features = np.stack([np.asarray(scene["feature"], dtype=np.float32) for scene in member_scenes], axis=0)
    if reducer == "mean":
        return features.mean(axis=0).astype(np.float32)
    if reducer == "medoid":
        center = features.mean(axis=0, keepdims=True)
        medoid_idx = int(np.argmin(np.sum((features - center) ** 2, axis=1)))
        return features[medoid_idx].astype(np.float32)
    raise ValueError(f"Unsupported hierarchy_count.feature_reducer={reducer}")


def _mean_for_normalization(group_infos: list[dict], normalization_split: str) -> float:
    """Compute the mean count used for direct-count multiplier labels.

    Args:
        group_infos: Pose-class scene summaries with ``grasp_record_count``.
        normalization_split: Preferred data split for normalization.

    Returns:
        Positive mean grasp-record count. Falls back to all rows if the
        requested split is absent.
    """
    split_counts = [
        float(info["grasp_record_count"])
        for info in group_infos
        if str(info["split"]) == str(normalization_split)
    ]
    counts = split_counts if split_counts else [float(info["grasp_record_count"]) for info in group_infos]
    mean_count = float(np.mean(counts)) if counts else 1.0
    return mean_count if mean_count > 0.0 else 1.0


def _build_hierarchy_count_labels(
    scenes: list[dict],
    label_hierarchy_rows: list[dict],
    task_cfg: DictConfig,
) -> tuple[list[dict], dict]:
    """Aggregate per-grasp hierarchy rows into pose-class scene-budget labels.

    Args:
        scenes: Original scene dictionaries returned by ``_build_scene_index``.
        label_hierarchy_rows: Per-grasp rows returned by
            ``_build_scene_budget_label_hierarchy_rows``.
        task_cfg: Task config containing direct-count settings.

    Returns:
        Tuple of pose-class scene rows and summary statistics. Each row is one
        direct-count scene-budget target.
    """
    if not label_hierarchy_rows:
        raise RuntimeError("Cannot build hierarchy_count labels without scene_budget_label_hierarchy rows")

    feature_reducer = str(cfg_get(task_cfg, "hierarchy_count.feature_reducer", default="mean")).lower()
    normalization_split = str(cfg_get(task_cfg, "hierarchy_count.normalization_split", default="train"))
    clip_min = float(cfg_get(task_cfg, "hierarchy_count.clip_min", "legacy_nearest_n.clip_min", default=0.5))
    clip_max = float(cfg_get(task_cfg, "hierarchy_count.clip_max", "legacy_nearest_n.clip_max", default=3.0))
    if clip_min <= 0.0 or clip_max < clip_min:
        raise ValueError(f"Invalid hierarchy_count clip range [{clip_min}, {clip_max}]")

    scene_by_key = {
        (int(scene["component_idx"]), str(scene["split"]), str(scene["scene_key"])): scene
        for scene in scenes
    }
    rows_by_pose_class = {}
    for row in label_hierarchy_rows:
        group_key = (
            int(row["component_idx"]),
            str(row["split"]),
            str(row["canonical_object_id"]),
            int(row["pose_class_id"]),
        )
        rows_by_pose_class.setdefault(group_key, []).append(row)

    group_infos = []
    for group_key, group_rows in sorted(
        rows_by_pose_class.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            _natural_sort_key(item[0][2]),
            item[0][3],
        ),
    ):
        component_idx, split, object_id, pose_class_id = group_key
        member_scene_keys = sorted({str(row["scene_key"]) for row in group_rows}, key=_natural_sort_key)
        member_scenes = []
        for scene_key in member_scene_keys:
            scene_lookup_key = (component_idx, split, scene_key)
            if scene_lookup_key not in scene_by_key:
                raise KeyError(f"Hierarchy row references missing scene {scene_lookup_key}")
            member_scenes.append(scene_by_key[scene_lookup_key])

        pose_class_key = str(group_rows[0]["pose_class_key"])
        representative_scene_key = str(group_rows[0]["pose_class_representative_scene_key"])
        representative_scene = scene_by_key.get((component_idx, split, representative_scene_key), member_scenes[0])
        feature = _aggregate_pose_class_feature(member_scenes, reducer=feature_reducer)

        type_counts = {}
        for row in group_rows:
            grasp_type_id = int(row["grasp_type_id"])
            type_counts[grasp_type_id] = type_counts.get(grasp_type_id, 0) + 1
        type_counts = dict(sorted(type_counts.items()))
        member_sequence_ids = sorted(
            {str(row["sequence_id"]) for row in group_rows if str(row.get("sequence_id", ""))},
            key=_natural_sort_key,
        )

        group_infos.append(
            {
                "component_idx": component_idx,
                "split": split,
                "object_id": object_id,
                "pose_class_id": pose_class_id,
                "pose_class_key": pose_class_key,
                "pose_class_representative_scene_key": representative_scene_key,
                "representative_scene": representative_scene,
                "feature": feature,
                "grasp_record_count": int(len(group_rows)),
                "member_scene_num": int(len(member_scene_keys)),
                "member_scene_keys": member_scene_keys,
                "member_sequence_ids": member_sequence_ids,
                "grasp_type_counts": type_counts,
            }
        )

    mean_train_count = _mean_for_normalization(group_infos, normalization_split=normalization_split)
    raw_features = np.stack([info["feature"] for info in group_infos], axis=0).astype(np.float32)
    _, feature_mean, feature_std = _standardize_features(raw_features)
    raw_counts = np.asarray([info["grasp_record_count"] for info in group_infos], dtype=np.float64)
    member_scene_counts = np.asarray([info["member_scene_num"] for info in group_infos], dtype=np.float64)

    multipliers = np.clip(raw_counts / mean_train_count, clip_min, clip_max).astype(np.float32)
    log_multipliers = np.log(np.maximum(multipliers, EPS)).astype(np.float32)

    rows = []
    for idx, info in enumerate(group_infos):
        representative_scene = info["representative_scene"]
        type_counts = info["grasp_type_counts"]
        row = {
            "component_idx": int(info["component_idx"]),
            "split": str(info["split"]),
            "scene_key": str(info["pose_class_key"]),
            "pose_class_scene_key": str(info["pose_class_key"]),
            "object_id": str(info["object_id"]),
            "canonical_object_id": str(info["object_id"]),
            "pose_class_id": int(info["pose_class_id"]),
            "pose_class_key": str(info["pose_class_key"]),
            "pose_class_representative_scene_key": str(info["pose_class_representative_scene_key"]),
            "sequence_id": str(representative_scene.get("sequence_id", "")),
            "member_sequence_ids": ";".join(info["member_sequence_ids"]),
            "pc_path": str(representative_scene["pc_path"]),
            "record_count": int(info["grasp_record_count"]),
            "grasp_record_count": int(info["grasp_record_count"]),
            "effective_record_count": float(info["grasp_record_count"]),
            "member_scene_num": int(info["member_scene_num"]),
            "class_scene_num": int(info["member_scene_num"]),
            "member_scene_keys": ";".join(info["member_scene_keys"]),
            "grasp_type_ids": ";".join(str(type_id) for type_id in type_counts),
            "grasp_type_counts_json": json.dumps(type_counts, separators=(",", ":")),
            "label_source": "hierarchy_count",
            "feature_reducer": feature_reducer,
            "_point_cloud": np.asarray(representative_scene["point_cloud"], dtype=np.float32),
            "count_multiplier": float(multipliers[idx]),
            "log_count_multiplier": float(log_multipliers[idx]),
            # Keep compatibility with existing budget head and downstream filenames.
            "budget_multiplier": float(multipliers[idx]),
            "log_budget_multiplier": float(log_multipliers[idx]),
        }
        for grasp_type_id in range(1, 6):
            row[f"grasp_type_{grasp_type_id}_count"] = int(type_counts.get(grasp_type_id, 0))
        for feature_name, feature_value in zip(GEOMETRY_FEATURE_NAMES, raw_features[idx]):
            row[feature_name] = float(feature_value)
        rows.append(row)

    count_sum = int(raw_counts.sum()) if raw_counts.size else 0
    summary = {
        "label_source": "hierarchy_count",
        "scene_num": int(len(rows)),
        "feature_names": GEOMETRY_FEATURE_NAMES,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "budget_label": {
            "label_source": "hierarchy_count",
            "scene_group_key": "pose_class",
            "raw_label": "grasp_record_count",
            "target": "log_count_multiplier",
            "target_alias": "log_budget_multiplier",
            "feature_reducer": feature_reducer,
            "normalization_split": normalization_split,
            "mean_train_grasp_record_count": mean_train_count,
            "clip_min": clip_min,
            "clip_max": clip_max,
            "grasp_record_count_sum": count_sum,
            "grasp_record_count_mean": float(raw_counts.mean()) if raw_counts.size else 0.0,
            "grasp_record_count_min": int(raw_counts.min()) if raw_counts.size else 0,
            "grasp_record_count_max": int(raw_counts.max()) if raw_counts.size else 0,
            "member_scene_num_mean": float(member_scene_counts.mean()) if member_scene_counts.size else 0.0,
            "member_scene_num_min": int(member_scene_counts.min()) if member_scene_counts.size else 0,
            "member_scene_num_max": int(member_scene_counts.max()) if member_scene_counts.size else 0,
            "multiplier_mean": float(multipliers.mean()) if multipliers.size else 0.0,
            "multiplier_min": float(multipliers.min()) if multipliers.size else 0.0,
            "multiplier_max": float(multipliers.max()) if multipliers.size else 0.0,
        },
        "validation": {
            "label_hierarchy_row_num": int(len(label_hierarchy_rows)),
            "scene_count_sum_matches_hierarchy": bool(count_sum == len(label_hierarchy_rows)),
        },
    }
    return rows, summary


def _ordered_fieldnames(rows: list[dict], preferred_fields: list[str]) -> list[str]:
    """Build stable CSV field names from preferred and discovered columns.

    Args:
        rows: Rows that will be written to CSV.
        preferred_fields: Fields that should appear first when present.

    Returns:
        Ordered field names containing all row keys.
    """
    discovered = set()
    for row in rows:
        discovered.update(row.keys())
    ordered = [field for field in preferred_fields if field in discovered]
    extra = sorted((field for field in discovered if field not in ordered), key=_natural_sort_key)
    return ordered + extra


def _write_csv(rows: list[dict], path: str, fieldnames: list[str]) -> None:
    """Write dictionaries to a CSV file.

    Args:
        rows: Rows to write.
        path: Destination CSV path.
        fieldnames: Ordered CSV columns.

    Returns:
        None.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _save_scene_budget_outputs(
    rows: list[dict],
    summary: dict,
    output_dir: str,
    label_hierarchy_rows: list[dict] | None = None,
) -> dict:
    """Save the compact scene-budget label artifacts.

    Args:
        rows: Per-scene budget rows kept in memory for budget-head training.
        summary: Scene-budget summary dictionary.
        output_dir: Directory for generated files.
        label_hierarchy_rows: Optional per-grasp hierarchy rows.

    Returns:
        Dictionary containing the written output file paths. The intermediate
        feature/label CSVs are intentionally not saved; training consumes
        ``rows`` directly, and downstream inspection uses the hierarchy CSV and
        budget-head prediction CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    summary_json = os.path.join(output_dir, "scene_budget_summary.json")
    output_paths = {"summary_json": summary_json}

    if label_hierarchy_rows is not None:
        hierarchy_csv = os.path.join(output_dir, "scene_budget_label_hierarchy.csv")
        _write_csv(
            label_hierarchy_rows,
            hierarchy_csv,
            [
                "component_idx",
                "split",
                "scene_id",
                "scene_key",
                "canonical_object_id",
                "sequence_id",
                "pose_class_id",
                "pose_class_key",
                "pose_class_representative_scene_id",
                "pose_class_representative_scene_key",
                "pose_class_match_method",
                "grasp_type_id",
                "grasp_type_name",
                "grasp_index_in_scene",
                "grasp_index_in_type",
                "grasp_path",
                "source_scene",
            ],
        )
        output_paths["label_hierarchy_csv"] = hierarchy_csv

    _save_json(summary, summary_json)
    return output_paths

def _load_scene_budget_rows(scene_csv: str) -> list[dict]:
    """Load scene-budget rows from a generated CSV file.

    Args:
        scene_csv: Optional path to a legacy scene-budget row CSV.

    Returns:
        List of row dictionaries.
    """
    with open(scene_csv, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _rows_to_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str], list[str], str]:
    """Convert scene-budget CSV rows into feature and target arrays.

    Args:
        rows: Scene-budget rows loaded from a CSV.

    Returns:
        Tuple ``(features, targets, object_ids, scene_keys, target_column)``.
    """
    if not rows:
        raise RuntimeError("No rows are available for budget head training")
    features = np.asarray(
        [[float(row[name]) for name in GEOMETRY_FEATURE_NAMES] for row in rows],
        dtype=np.float32,
    )
    if str(rows[0].get("log_count_multiplier", "")).strip():
        target_column = "log_count_multiplier"
    else:
        target_column = "log_budget_multiplier"
    targets = np.asarray([float(row[target_column]) for row in rows], dtype=np.float32)
    object_ids = [str(row.get("object_id") or row.get("canonical_object_id")) for row in rows]
    scene_keys = [str(row["scene_key"]) for row in rows]
    return features, targets, object_ids, scene_keys, target_column


def _split_budget_rows(
    rows: list[dict],
    train_cfg: DictConfig,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Split budget rows using explicit dataset split labels.

    Args:
        rows: Scene-budget rows containing a ``split`` field.
        train_cfg: Training config section with ``train_split`` and
            ``val_split`` names.

    Returns:
        Tuple ``(train_indices, val_indices, strategy)``. The scene-budget
        task does not create its own validation split because object assets are
        already split into train/test lists.
    """
    train_split = str(cfg_get(train_cfg, "train_split", default="train"))
    val_split = str(cfg_get(train_cfg, "val_split", default="test"))
    train_indices = [idx for idx, row in enumerate(rows) if str(row.get("split", "")) == train_split]
    val_indices = [idx for idx, row in enumerate(rows) if str(row.get("split", "")) == val_split]
    if not train_indices or not val_indices:
        split_counts = {}
        for row in rows:
            split_name = str(row.get("split", ""))
            split_counts[split_name] = split_counts.get(split_name, 0) + 1
        raise ValueError(
            "scene_budget training requires explicit train/test split rows. "
            f"Requested train_split={train_split}, val_split={val_split}, "
            f"available split counts={split_counts}"
        )
    return np.asarray(train_indices, dtype=np.int64), np.asarray(val_indices, dtype=np.int64), "explicit_split"

def _rankdata(values: np.ndarray) -> np.ndarray:
    """Compute average ranks for one-dimensional values.

    Args:
        values: Array to rank.

    Returns:
        Rank array with ties assigned their average rank.
    """
    values = np.asarray(values)
    order = np.argsort(values)
    ranks = np.empty(values.shape[0], dtype=np.float64)
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _regression_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """Compute scalar regression metrics.

    Args:
        pred: Predicted values.
        target: Target values.

    Returns:
        Dictionary with MSE, MAE, and Spearman rank correlation.
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.size == 0:
        return {"mse": None, "mae": None, "spearman": None}
    mse = float(np.mean((pred - target) ** 2))
    mae = float(np.mean(np.abs(pred - target)))
    if pred.size < 2 or np.std(pred) < EPS or np.std(target) < EPS:
        spearman = None
    else:
        pred_rank = _rankdata(pred)
        target_rank = _rankdata(target)
        spearman = float(np.corrcoef(pred_rank, target_rank)[0, 1])
    return {"mse": mse, "mae": mae, "spearman": spearman}


def _evaluate_head(model: GeometryBudgetHead, features: torch.Tensor, targets: torch.Tensor, device: str) -> tuple[dict, np.ndarray]:
    """Evaluate a budget head on one tensor split.

    Args:
        model: Trained budget head.
        features: Normalized feature tensor.
        targets: Log-multiplier target tensor.
        device: Torch device string.

    Returns:
        Tuple of metrics and predictions as a NumPy array.
    """
    if features.shape[0] == 0:
        return _regression_metrics(np.asarray([]), np.asarray([])), np.asarray([], dtype=np.float32)
    model.eval()
    with torch.no_grad():
        pred = model(features.to(device)).detach().cpu().numpy()
    metrics = _regression_metrics(pred, targets.detach().cpu().numpy())
    return metrics, pred.astype(np.float32)


def _save_multiplier_scatter_plot(rows: list[dict], split: str, title: str, path: str) -> str | None:
    """Save a target-versus-prediction scatter plot for one split.

    Args:
        rows: Prediction rows produced by the budget-head trainer.
        split: Row split name to plot, such as ``train`` or ``val``.
        title: Plot title.
        path: Output image path.

    Returns:
        The written image path, or ``None`` when the requested split has no rows.
    """
    selected = [row for row in rows if str(row.get("split", "")) == split]
    if not selected:
        return None

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    target = np.asarray([float(row["target_multiplier"]) for row in selected], dtype=np.float32)
    pred = np.asarray([float(row["pred_multiplier"]) for row in selected], dtype=np.float32)
    metrics = _regression_metrics(pred, target)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=160)
    ax.scatter(target, pred, s=14, alpha=0.7, linewidths=0)
    lower = float(min(target.min(), pred.min()))
    upper = float(max(target.max(), pred.max()))
    pad = max((upper - lower) * 0.05, 1e-3)
    line_min = lower - pad
    line_max = upper + pad
    ax.plot([line_min, line_max], [line_min, line_max], linestyle="--", linewidth=1.0, color="#666666")
    ax.set_xlim(line_min, line_max)
    ax.set_ylim(line_min, line_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Target multiplier")
    ax.set_ylabel("Predicted multiplier")
    subtitle = []
    if metrics["mse"] is not None:
        subtitle.append(f"MSE={metrics['mse']:.4f}")
    if metrics["mae"] is not None:
        subtitle.append(f"MAE={metrics['mae']:.4f}")
    if metrics["spearman"] is not None:
        subtitle.append(f"Spearman={metrics['spearman']:.4f}")
    if subtitle:
        ax.set_title(f"{title}\n" + "  ".join(subtitle))
    else:
        ax.set_title(title)
    ax.grid(True, linewidth=0.4, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _budget_input_type(train_cfg: DictConfig) -> str:
    """Resolve the configured budget-head input type.

    Args:
        train_cfg: Training config section.

    Returns:
        Normalized input type, either ``bbox`` or ``pointcloud``.
    """
    input_type = str(cfg_get(train_cfg, "input_type", default="bbox")).lower()
    aliases = {"geometry": "bbox", "pc": "pointcloud", "point_cloud": "pointcloud"}
    input_type = aliases.get(input_type, input_type)
    if input_type not in {"bbox", "pointcloud"}:
        raise ValueError(f"Unsupported scene_budget train.input_type={input_type}; expected bbox or pointcloud")
    return input_type


def _move_tensor_batch_to_device(batch: dict, device: str) -> dict:
    """Move tensor values in a batch to the requested device.

    Args:
        batch: Batch dictionary containing tensors and metadata.
        device: Torch device string.

    Returns:
        The same dictionary with tensor values moved in-place.
    """
    for key, value in list(batch.items()):
        if type(value).__module__ != "torch":
            continue
        if "Int" not in value.type() and "Long" not in value.type() and "Short" not in value.type():
            value = value.float()
        batch[key] = value.to(device)
    return batch


def _z_yaw_rotation_matrix(angle_rad: float) -> np.ndarray:
    """Build a global Z-yaw rotation matrix.

    Args:
        angle_rad: Rotation angle in radians.

    Returns:
        Rotation matrix with shape ``(3, 3)``.
    """
    cos_angle = math.cos(float(angle_rad))
    sin_angle = math.sin(float(angle_rad))
    return np.asarray(
        [
            [cos_angle, -sin_angle, 0.0],
            [sin_angle, cos_angle, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


class _PointCloudBudgetDataset(Dataset):
    """Dataset over in-memory scene-budget point clouds."""

    def __init__(self, rows: list[dict], targets: np.ndarray, point_num: int, z_yaw_aug: bool = False):
        """Initialize the point-cloud budget dataset.

        Args:
            rows: Scene-budget rows containing ``_point_cloud`` arrays.
            targets: Log-multiplier target array aligned with ``rows``.
            point_num: Number of points sampled per scene.
            z_yaw_aug: Whether to apply random global Z-yaw augmentation.

        Returns:
            None.
        """
        if int(point_num) <= 0:
            raise ValueError(f"point_num must be positive for pointcloud input, got {point_num}")
        self.rows = rows
        self.targets = np.asarray(targets, dtype=np.float32)
        self.point_num = int(point_num)
        self.z_yaw_aug = bool(z_yaw_aug)

    def __len__(self) -> int:
        """Return the number of point-cloud budget rows."""
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        """Load one centered point-cloud sample.

        Args:
            idx: Row index.

        Returns:
            Dictionary containing ``point_clouds`` and ``target`` tensors.
        """
        row = self.rows[int(idx)]
        if "_point_cloud" not in row:
            raise KeyError("Point-cloud input requires in-memory rows with '_point_cloud'; use task.mode=all")
        points = np.asarray(row["_point_cloud"], dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected _point_cloud with shape (N, 3), got {points.shape}")
        replace = points.shape[0] < self.point_num
        if self.z_yaw_aug:
            sample_indices = np.random.choice(points.shape[0], self.point_num, replace=replace)
        else:
            if replace:
                sample_indices = np.arange(self.point_num, dtype=np.int64) % points.shape[0]
            else:
                sample_indices = np.linspace(0, points.shape[0] - 1, num=self.point_num, dtype=np.int64)
        sampled = points[sample_indices].astype(np.float32, copy=True)
        sampled = sampled - sampled.mean(axis=0, keepdims=True)
        if self.z_yaw_aug:
            sampled = sampled @ _z_yaw_rotation_matrix(np.random.uniform(-np.pi, np.pi)).T
        return {
            "point_clouds": torch.from_numpy(sampled.astype(np.float32, copy=False)),
            "target": torch.tensor(float(self.targets[int(idx)]), dtype=torch.float32),
        }


def _collate_pointcloud_budget_samples(samples: list[dict], voxel_size: float) -> dict:
    """Collate point-cloud budget samples into a MinkowskiEngine batch.

    Args:
        samples: Samples returned by ``_PointCloudBudgetDataset``.
        voxel_size: Sparse voxel size used by the point-cloud backbone.

    Returns:
        Batch dictionary accepted by ``PointCloudBudgetHead``.
    """
    point_clouds = torch.stack([sample["point_clouds"] for sample in samples], dim=0)
    targets = torch.stack([sample["target"] for sample in samples], dim=0)
    batch = get_sparse_tensor(point_clouds, float(voxel_size))
    batch["target"] = targets
    return batch


def _pointcloud_eval_predictions(
    model: PointCloudBudgetHead,
    rows: list[dict],
    targets: np.ndarray,
    point_num: int,
    voxel_size: float,
    batch_size: int,
    device: str,
) -> tuple[dict, np.ndarray]:
    """Evaluate a point-cloud budget model.

    Args:
        model: Point-cloud budget head.
        rows: Rows to evaluate.
        targets: Log-multiplier targets aligned with ``rows``.
        point_num: Number of points sampled per scene.
        voxel_size: Sparse voxel size for the backbone.
        batch_size: Evaluation batch size.
        device: Torch device string.

    Returns:
        Tuple of regression metrics and prediction array.
    """
    dataset = _PointCloudBudgetDataset(rows, targets, point_num=point_num, z_yaw_aug=False)
    loader = DataLoader(
        dataset,
        batch_size=max(int(batch_size), 1),
        shuffle=False,
        drop_last=False,
        collate_fn=lambda samples: _collate_pointcloud_budget_samples(samples, voxel_size=voxel_size),
    )
    preds = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = _move_tensor_batch_to_device(batch, device)
            preds.append(model(batch).detach().cpu().numpy().astype(np.float32))
    pred = np.concatenate(preds, axis=0) if preds else np.asarray([], dtype=np.float32)
    return _regression_metrics(pred, targets), pred


def _load_pointcloud_backbone_checkpoint(model: PointCloudBudgetHead, checkpoint_path: str | None) -> dict:
    """Load ``backbone.*`` weights from a main training checkpoint.

    Args:
        model: Point-cloud budget model whose backbone should be initialized.
        checkpoint_path: Optional checkpoint path produced by ``task=train``.

    Returns:
        Dictionary describing the loaded checkpoint. Empty when no checkpoint
        path is configured.
    """
    if not checkpoint_path:
        return {}
    checkpoint_path = _absolute_path(str(checkpoint_path))
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    backbone_state = {
        key[len("backbone.") :]: value
        for key, value in state_dict.items()
        if str(key).startswith("backbone.")
    }
    if not backbone_state:
        raise KeyError(f"No backbone.* weights found in {checkpoint_path}")
    missing_keys, unexpected_keys = model.backbone.load_state_dict(backbone_state, strict=False)
    return {
        "checkpoint": checkpoint_path,
        "loaded_key_num": int(len(backbone_state)),
        "missing_keys": list(missing_keys),
        "unexpected_keys": list(unexpected_keys),
    }


def _train_pointcloud_budget_head(rows: list[dict], summary: dict, config: DictConfig, output_dir: str) -> dict:
    """Train a point-cloud encoder based budget head.

    Args:
        rows: In-memory scene-budget rows containing ``_point_cloud`` arrays.
        summary: Scene-budget summary dictionary.
        config: Full Hydra config.
        output_dir: Directory for checkpoint, CSV, plot, and summary outputs.

    Returns:
        Training summary dictionary.
    """
    task_cfg = config.task
    train_cfg = task_cfg.train
    _, targets, object_ids, scene_keys, target_column = _rows_to_arrays(rows)
    train_indices, val_indices, split_strategy = _split_budget_rows(rows, train_cfg=train_cfg)

    pointcloud_cfg = cfg_get(train_cfg, "pointcloud", default=None)
    backbone_cfg = cfg_get(pointcloud_cfg, "backbone", default=None)
    if backbone_cfg is None:
        backbone_cfg = cfg_get(config.algo.model, "backbone", default=None)
    if backbone_cfg is None:
        raise ValueError("Point-cloud budget input requires task.train.pointcloud.backbone or algo.model.backbone")
    backbone_cfg = OmegaConf.create(OmegaConf.to_container(backbone_cfg, resolve=True))
    voxel_size = float(cfg_get(pointcloud_cfg, "voxel_size", "backbone.voxel_size", default=backbone_cfg.voxel_size))
    point_num = int(cfg_get(pointcloud_cfg, "num_points", default=1024))
    z_yaw_aug = bool(cfg_get(pointcloud_cfg, "z_yaw_aug", default=True))
    freeze_encoder = bool(cfg_get(pointcloud_cfg, "freeze_encoder", "freeze_backbone", default=False))
    encoder_checkpoint = cfg_get(pointcloud_cfg, "encoder_checkpoint", "backbone_checkpoint", default=None)

    device = str(cfg_get(train_cfg, "device", default=config.device))
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    hidden_dims = [
        int(dim)
        for dim in _as_list(cfg_get(pointcloud_cfg, "hidden_dims", "head_hidden_dims", default=[128, 64]))
    ]
    model = PointCloudBudgetHead(
        backbone_cfg=backbone_cfg,
        hidden_dims=hidden_dims,
        dropout=float(cfg_get(train_cfg, "dropout", default=0.1)),
    )
    loaded_encoder = _load_pointcloud_backbone_checkpoint(model, encoder_checkpoint)
    if freeze_encoder:
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)
    model.to(device)

    train_rows = [rows[int(idx)] for idx in train_indices.tolist()]
    train_targets = targets[train_indices]
    train_dataset = _PointCloudBudgetDataset(train_rows, train_targets, point_num=point_num, z_yaw_aug=z_yaw_aug)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg_get(train_cfg, "batch_size", default=16)),
        shuffle=True,
        drop_last=False,
        collate_fn=lambda samples: _collate_pointcloud_budget_samples(samples, voxel_size=voxel_size),
    )

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(cfg_get(train_cfg, "lr", default=1e-3)),
        weight_decay=float(cfg_get(train_cfg, "weight_decay", default=1e-3)),
    )
    loss_name = str(cfg_get(train_cfg, "loss", default="huber")).lower()
    if loss_name == "mse":
        criterion = torch.nn.MSELoss()
    elif loss_name in {"huber", "smooth_l1"}:
        criterion = torch.nn.SmoothL1Loss(beta=float(cfg_get(train_cfg, "huber_beta", default=0.25)))
    else:
        raise ValueError(f"Unsupported budget head loss={loss_name}")

    best_state = None
    best_score = float("inf")
    best_epoch = 0
    stopped_epoch = None
    epochs_without_improvement = 0
    history = []
    max_epochs = int(cfg_get(train_cfg, "max_epochs", default=100))
    log_every = max(int(cfg_get(train_cfg, "log_every", default=10)), 1)
    early_stopping_cfg = cfg_get(train_cfg, "early_stopping", default=None)
    early_stopping_enabled = bool(cfg_get(early_stopping_cfg, "enabled", default=True))
    early_stopping_patience = max(int(cfg_get(early_stopping_cfg, "patience", default=20)), 1)
    early_stopping_min_delta = float(cfg_get(early_stopping_cfg, "min_delta", default=1e-4))
    early_stopping_min_epochs = max(int(cfg_get(early_stopping_cfg, "min_epochs", default=10)), 1)

    all_rows = rows
    epoch_iter = tqdm(
        range(1, max_epochs + 1),
        desc="scene_budget pointcloud train",
        dynamic_ncols=True,
    )
    for epoch in epoch_iter:
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = _move_tensor_batch_to_device(batch, device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, batch["target"])
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        all_metrics, all_pred = _pointcloud_eval_predictions(
            model,
            all_rows,
            targets,
            point_num=point_num,
            voxel_size=voxel_size,
            batch_size=int(cfg_get(train_cfg, "batch_size", default=16)),
            device=device,
        )
        train_metrics = _regression_metrics(all_pred[train_indices], targets[train_indices])
        val_metrics = _regression_metrics(all_pred[val_indices], targets[val_indices])
        score = val_metrics["mse"]
        if score is not None and score < best_score - early_stopping_min_delta:
            best_score = float(score)
            best_epoch = int(epoch)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_iter.set_postfix(
            loss=f"{float(np.mean(train_losses)) if train_losses else 0.0:.4f}",
            val_mse=f"{score:.4f}" if score is not None else "nan",
            best_epoch=best_epoch,
            refresh=False,
        )

        if epoch == 1 or epoch == max_epochs or epoch % log_every == 0:
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(np.mean(train_losses)) if train_losses else 0.0,
                    "train": train_metrics,
                    "val": val_metrics,
                }
            )
        if (
            early_stopping_enabled
            and epoch >= early_stopping_min_epochs
            and epochs_without_improvement >= early_stopping_patience
        ):
            stopped_epoch = int(epoch)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    all_metrics, all_pred = _pointcloud_eval_predictions(
        model,
        rows,
        targets,
        point_num=point_num,
        voxel_size=voxel_size,
        batch_size=int(cfg_get(train_cfg, "batch_size", default=16)),
        device=device,
    )
    train_metrics = _regression_metrics(all_pred[train_indices], targets[train_indices])
    val_metrics = _regression_metrics(all_pred[val_indices], targets[val_indices])

    ckpt_path = os.path.join(output_dir, "geometry_budget_head.pth")
    torch.save(
        {
            "input_type": "pointcloud",
            "model_state_dict": model.state_dict(),
            "backbone_config": OmegaConf.to_container(backbone_cfg, resolve=True),
            "point_num": point_num,
            "voxel_size": voxel_size,
            "hidden_dims": hidden_dims,
            "dropout": float(cfg_get(train_cfg, "dropout", default=0.0)),
            "target": target_column,
            "best_epoch": best_epoch,
            "metrics": {"all": all_metrics, "train": train_metrics, "val": val_metrics},
        },
        ckpt_path,
    )

    pred_rows = []
    train_set = set(int(idx) for idx in train_indices.tolist())
    val_set = set(int(idx) for idx in val_indices.tolist())
    for idx, row in enumerate(rows):
        split = "train" if idx in train_set else "val" if idx in val_set else "unused"
        pred_rows.append(
            {
                "scene_key": scene_keys[idx],
                "object_id": object_ids[idx],
                "data_split": str(row.get("split", "")),
                "pose_class_id": row.get("pose_class_id", ""),
                "split": split,
                "target_column": target_column,
                "target_log_multiplier": float(targets[idx]),
                "pred_log_multiplier": float(all_pred[idx]),
                "target_multiplier": float(math.exp(float(targets[idx]))),
                "pred_multiplier": float(math.exp(float(all_pred[idx]))),
                "target_log_budget_multiplier": float(targets[idx]),
                "pred_log_budget_multiplier": float(all_pred[idx]),
                "target_budget_multiplier": float(math.exp(float(targets[idx]))),
                "pred_budget_multiplier": float(math.exp(float(all_pred[idx]))),
            }
        )
    pred_csv = os.path.join(output_dir, "budget_head_predictions.csv")
    _write_csv(
        pred_rows,
        pred_csv,
        [
            "scene_key",
            "object_id",
            "data_split",
            "pose_class_id",
            "split",
            "target_column",
            "target_log_multiplier",
            "pred_log_multiplier",
            "target_multiplier",
            "pred_multiplier",
            "target_log_budget_multiplier",
            "pred_log_budget_multiplier",
            "target_budget_multiplier",
            "pred_budget_multiplier",
        ],
    )
    plot_paths = {}
    train_plot = _save_multiplier_scatter_plot(
        pred_rows,
        split="train",
        title="Scene Budget Train: target vs predicted multiplier",
        path=os.path.join(output_dir, "budget_head_train_multiplier_scatter.png"),
    )
    if train_plot is not None:
        plot_paths["train_multiplier_scatter_png"] = train_plot
    test_plot = _save_multiplier_scatter_plot(
        pred_rows,
        split="val",
        title="Scene Budget Test: target vs predicted multiplier",
        path=os.path.join(output_dir, "budget_head_test_multiplier_scatter.png"),
    )
    if test_plot is not None:
        plot_paths["test_multiplier_scatter_png"] = test_plot

    train_summary = {
        "checkpoint": ckpt_path,
        "prediction_csv": pred_csv,
        "prediction_plots": plot_paths,
        "input_type": "pointcloud",
        "target": target_column,
        "train_scene_num": int(len(train_indices)),
        "val_scene_num": int(len(val_indices)),
        "split_strategy": split_strategy,
        "best_epoch": int(best_epoch),
        "stopped_epoch": stopped_epoch,
        "pointcloud": {
            "backbone": OmegaConf.to_container(backbone_cfg, resolve=True),
            "point_num": point_num,
            "voxel_size": voxel_size,
            "z_yaw_aug": z_yaw_aug,
            "freeze_encoder": freeze_encoder,
            "loaded_encoder": loaded_encoder,
        },
        "regularization": {
            "hidden_dims": hidden_dims,
            "dropout": float(cfg_get(train_cfg, "dropout", default=0.1)),
            "weight_decay": float(cfg_get(train_cfg, "weight_decay", default=1e-3)),
            "early_stopping": {
                "enabled": early_stopping_enabled,
                "patience": early_stopping_patience,
                "min_delta": early_stopping_min_delta,
                "min_epochs": early_stopping_min_epochs,
            },
        },
        "history": history,
        "metrics": {"all": all_metrics, "train": train_metrics, "val": val_metrics},
    }
    _save_json(train_summary, os.path.join(output_dir, "budget_head_train_summary.json"))
    return train_summary


def _train_budget_head(rows: list[dict], summary: dict, config: DictConfig, output_dir: str) -> dict:
    """Train the independent geometry-only budget head.

    Args:
        rows: Scene-budget rows containing geometry features and log targets.
        summary: Scene-budget summary containing feature normalization statistics.
        config: Full Hydra config.
        output_dir: Directory for checkpoint and prediction outputs.

    Returns:
        Training summary dictionary.
    """
    task_cfg = config.task
    train_cfg = task_cfg.train
    input_type = _budget_input_type(train_cfg)
    if input_type == "pointcloud":
        return _train_pointcloud_budget_head(rows, summary, config, output_dir)

    features, targets, object_ids, scene_keys, target_column = _rows_to_arrays(rows)
    feature_mean = np.asarray(summary["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(summary["feature_std"], dtype=np.float32)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std).astype(np.float32)
    normalized_features = ((features - feature_mean) / feature_std).astype(np.float32)

    train_indices, val_indices, split_strategy = _split_budget_rows(
        rows,
        train_cfg=train_cfg,
    )
    device = str(cfg_get(train_cfg, "device", default=config.device))
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    feature_tensor = torch.from_numpy(normalized_features)
    target_tensor = torch.from_numpy(targets)
    train_dataset = TensorDataset(feature_tensor[train_indices], target_tensor[train_indices])
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg_get(train_cfg, "batch_size", default=64)),
        shuffle=True,
        drop_last=False,
    )

    hidden_dims = [int(dim) for dim in _as_list(cfg_get(train_cfg, "hidden_dims", default=[16, 16]))]
    model = GeometryBudgetHead(
        input_dim=normalized_features.shape[1],
        hidden_dims=hidden_dims,
        dropout=float(cfg_get(train_cfg, "dropout", default=0.1)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg_get(train_cfg, "lr", default=1e-3)),
        weight_decay=float(cfg_get(train_cfg, "weight_decay", default=1e-3)),
    )
    loss_name = str(cfg_get(train_cfg, "loss", default="huber")).lower()
    if loss_name == "mse":
        criterion = torch.nn.MSELoss()
    elif loss_name in {"huber", "smooth_l1"}:
        criterion = torch.nn.SmoothL1Loss(beta=float(cfg_get(train_cfg, "huber_beta", default=0.25)))
    else:
        raise ValueError(f"Unsupported budget head loss={loss_name}")

    best_state = None
    best_score = float("inf")
    best_epoch = 0
    stopped_epoch = None
    epochs_without_improvement = 0
    history = []
    max_epochs = int(cfg_get(train_cfg, "max_epochs", default=100))
    log_every = max(int(cfg_get(train_cfg, "log_every", default=10)), 1)
    early_stopping_cfg = cfg_get(train_cfg, "early_stopping", default=None)
    early_stopping_enabled = bool(cfg_get(early_stopping_cfg, "enabled", default=True))
    early_stopping_patience = max(int(cfg_get(early_stopping_cfg, "patience", default=20)), 1)
    early_stopping_min_delta = float(cfg_get(early_stopping_cfg, "min_delta", default=1e-4))
    early_stopping_min_epochs = max(int(cfg_get(early_stopping_cfg, "min_epochs", default=10)), 1)
    epoch_iter = tqdm(
        range(1, max_epochs + 1),
        desc="scene_budget train",
        dynamic_ncols=True,
    )
    for epoch in epoch_iter:
        model.train()
        train_losses = []
        for batch_feature, batch_target in train_loader:
            batch_feature = batch_feature.to(device)
            batch_target = batch_target.to(device)
            optimizer.zero_grad()
            pred = model(batch_feature)
            loss = criterion(pred, batch_target)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        train_metrics, _ = _evaluate_head(
            model,
            feature_tensor[train_indices],
            target_tensor[train_indices],
            device,
        )
        val_metrics, _ = _evaluate_head(
            model,
            feature_tensor[val_indices],
            target_tensor[val_indices],
            device,
        )
        score = val_metrics["mse"]
        if score is not None and score < best_score - early_stopping_min_delta:
            best_score = float(score)
            best_epoch = int(epoch)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_iter.set_postfix(
            loss=f"{float(np.mean(train_losses)) if train_losses else 0.0:.4f}",
            val_mse=f"{score:.4f}" if score is not None else "nan",
            best_epoch=best_epoch,
            refresh=False,
        )

        if epoch == 1 or epoch == max_epochs or epoch % log_every == 0:
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(np.mean(train_losses)) if train_losses else 0.0,
                    "train": train_metrics,
                    "val": val_metrics,
                }
            )

        if (
            early_stopping_enabled
            and epoch >= early_stopping_min_epochs
            and epochs_without_improvement >= early_stopping_patience
        ):
            stopped_epoch = int(epoch)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    all_metrics, all_pred = _evaluate_head(model, feature_tensor, target_tensor, device)
    train_metrics, _ = _evaluate_head(model, feature_tensor[train_indices], target_tensor[train_indices], device)
    val_metrics, _ = _evaluate_head(model, feature_tensor[val_indices], target_tensor[val_indices], device)

    ckpt_path = os.path.join(output_dir, "geometry_budget_head.pth")
    torch.save(
        {
            "input_type": "bbox",
            "model_state_dict": model.state_dict(),
            "feature_names": GEOMETRY_FEATURE_NAMES,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "hidden_dims": hidden_dims,
            "dropout": float(cfg_get(train_cfg, "dropout", default=0.0)),
            "target": target_column,
            "best_epoch": best_epoch,
            "metrics": {"all": all_metrics, "train": train_metrics, "val": val_metrics},
        },
        ckpt_path,
    )

    pred_rows = []
    train_set = set(int(idx) for idx in train_indices.tolist())
    val_set = set(int(idx) for idx in val_indices.tolist())
    for idx, row in enumerate(rows):
        split = "train" if idx in train_set else "val" if idx in val_set else "unused"
        pred_rows.append(
            {
                "scene_key": scene_keys[idx],
                "object_id": object_ids[idx],
                "data_split": str(row.get("split", "")),
                "pose_class_id": row.get("pose_class_id", ""),
                "split": split,
                "target_column": target_column,
                "target_log_multiplier": float(targets[idx]),
                "pred_log_multiplier": float(all_pred[idx]),
                "target_multiplier": float(math.exp(float(targets[idx]))),
                "pred_multiplier": float(math.exp(float(all_pred[idx]))),
                "target_log_budget_multiplier": float(targets[idx]),
                "pred_log_budget_multiplier": float(all_pred[idx]),
                "target_budget_multiplier": float(math.exp(float(targets[idx]))),
                "pred_budget_multiplier": float(math.exp(float(all_pred[idx]))),
            }
        )
    pred_csv = os.path.join(output_dir, "budget_head_predictions.csv")
    _write_csv(
        pred_rows,
        pred_csv,
        [
            "scene_key",
            "object_id",
            "data_split",
            "pose_class_id",
            "split",
            "target_column",
            "target_log_multiplier",
            "pred_log_multiplier",
            "target_multiplier",
            "pred_multiplier",
            "target_log_budget_multiplier",
            "pred_log_budget_multiplier",
            "target_budget_multiplier",
            "pred_budget_multiplier",
        ],
    )
    plot_paths = {}
    train_plot = _save_multiplier_scatter_plot(
        pred_rows,
        split="train",
        title="Scene Budget Train: target vs predicted multiplier",
        path=os.path.join(output_dir, "budget_head_train_multiplier_scatter.png"),
    )
    if train_plot is not None:
        plot_paths["train_multiplier_scatter_png"] = train_plot
    test_plot = _save_multiplier_scatter_plot(
        pred_rows,
        split="val",
        title="Scene Budget Test: target vs predicted multiplier",
        path=os.path.join(output_dir, "budget_head_test_multiplier_scatter.png"),
    )
    if test_plot is not None:
        plot_paths["test_multiplier_scatter_png"] = test_plot

    train_summary = {
        "checkpoint": ckpt_path,
        "prediction_csv": pred_csv,
        "prediction_plots": plot_paths,
        "input_type": "bbox",
        "target": target_column,
        "train_scene_num": int(len(train_indices)),
        "val_scene_num": int(len(val_indices)),
        "split_strategy": split_strategy,
        "best_epoch": int(best_epoch),
        "stopped_epoch": stopped_epoch,
        "regularization": {
            "hidden_dims": hidden_dims,
            "dropout": float(cfg_get(train_cfg, "dropout", default=0.1)),
            "weight_decay": float(cfg_get(train_cfg, "weight_decay", default=1e-3)),
            "early_stopping": {
                "enabled": early_stopping_enabled,
                "patience": early_stopping_patience,
                "min_delta": early_stopping_min_delta,
                "min_epochs": early_stopping_min_epochs,
            },
        },
        "history": history,
        "metrics": {"all": all_metrics, "train": train_metrics, "val": val_metrics},
    }
    _save_json(train_summary, os.path.join(output_dir, "budget_head_train_summary.json"))
    return train_summary


def _load_budget_head_checkpoint(checkpoint_path: str, device: str) -> tuple[torch.nn.Module, dict]:
    """Load a trained geometry budget head checkpoint.

    Args:
        checkpoint_path: Path to ``geometry_budget_head.pth``.
        device: Torch device string for inference.

    Returns:
        Tuple ``(model, metadata)`` ready for prediction.
    """
    checkpoint_path = _absolute_path(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Budget head checkpoint does not exist: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    input_type = str(ckpt.get("input_type", "bbox")).lower()
    if input_type == "pointcloud":
        backbone_cfg = OmegaConf.create(ckpt["backbone_config"])
        model = PointCloudBudgetHead(
            backbone_cfg=backbone_cfg,
            hidden_dims=ckpt.get("hidden_dims", [128, 64]),
            dropout=float(ckpt.get("dropout", 0.0)),
        ).to(device)
        state_dict = ckpt.get("model_state_dict", ckpt.get("model"))
        if state_dict is None:
            raise KeyError(f"Checkpoint {checkpoint_path} does not contain model_state_dict")
        model.load_state_dict(state_dict)
        model.eval()
        return model, {
            "input_type": "pointcloud",
            "point_num": int(ckpt.get("point_num", 1024)),
            "voxel_size": float(ckpt.get("voxel_size", backbone_cfg.voxel_size)),
        }

    feature_names = list(ckpt.get("feature_names", GEOMETRY_FEATURE_NAMES))
    if feature_names != GEOMETRY_FEATURE_NAMES:
        raise ValueError(f"Unsupported budget-head feature names {feature_names}; expected {GEOMETRY_FEATURE_NAMES}")

    model = GeometryBudgetHead(
        input_dim=len(feature_names),
        hidden_dims=ckpt.get("hidden_dims", [64, 64]),
        dropout=float(ckpt.get("dropout", 0.0)),
    ).to(device)
    state_dict = ckpt.get("model_state_dict", ckpt.get("model"))
    if state_dict is None:
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain model_state_dict")
    model.load_state_dict(state_dict)
    model.eval()

    feature_mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(ckpt["feature_std"], dtype=np.float32)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std).astype(np.float32)
    return model, {
        "input_type": "bbox",
        "feature_mean": feature_mean,
        "feature_std": feature_std,
    }


def _as_batch_list(value: Any, batch_size: int, default: str = "") -> list[str]:
    """Convert one collated dataloader field into a string list.

    Args:
        value: Collated value returned by ``minkowski_collate_fn``.
        batch_size: Expected batch size.
        default: Value used when the field is absent.

    Returns:
        List of strings with length ``batch_size``.
    """
    if value is None:
        return [default for _ in range(batch_size)]
    if isinstance(value, torch.Tensor):
        flat = value.detach().cpu().reshape(-1).tolist()
        return [str(item) for item in flat]
    if isinstance(value, np.ndarray):
        flat = value.reshape(-1).tolist()
        return [str(item) for item in flat]
    if isinstance(value, (list, tuple)):
        if len(value) == batch_size:
            return [str(item) for item in value]
        if len(value) == 1:
            return [str(value[0]) for _ in range(batch_size)]
    return [str(value) for _ in range(batch_size)]


def _scene_metadata_from_scene_path(scene_path: str) -> tuple[str, str]:
    """Read scene id and object id from a sample-style scene config path.

    Args:
        scene_path: Scene config path from the test dataloader.

    Returns:
        Tuple ``(scene_id, object_id)``. Missing metadata falls back to the file
        stem and an empty object id.
    """
    if not scene_path:
        return "", ""
    fallback_scene_id = os.path.splitext(os.path.basename(scene_path))[0]
    try:
        scene_cfg = np.load(_absolute_path(scene_path), allow_pickle=True).item()
    except Exception:
        return fallback_scene_id, ""

    scene_id = scene_cfg.get("scene_id")
    object_id = ""
    if "object" in scene_cfg:
        object_id = str(scene_cfg.get("object", {}).get("name", ""))
    if scene_id is None and "scene" in scene_cfg:
        scene = scene_cfg["scene"]
        scene_id = scene.get("id", scene.get("scene_id")) if isinstance(scene, dict) else None
        task_obj = scene_cfg.get("task", {}).get("obj_name") if isinstance(scene_cfg.get("task"), dict) else None
        if task_obj is not None:
            object_id = str(task_obj)
    if scene_id is None:
        scene_id = fallback_scene_id
    return str(scene_id), object_id


def _prediction_rows_from_test_loader(
    config: DictConfig,
    model: torch.nn.Module,
    checkpoint_meta: dict,
    device: str,
) -> list[dict]:
    """Run budget-head prediction on the configured sample-style test loader.

    Args:
        config: Full Hydra config. ``config.test_data`` selects human or DGN
            inference inputs in the same way as ``task=sample``.
        model: Loaded budget head.
        checkpoint_meta: Metadata returned by ``_load_budget_head_checkpoint``.
        device: Torch device string.

    Returns:
        Prediction rows for ``budget_head_predictions.csv``.
    """
    test_loader = create_test_dataloader(config)
    deduplicate_scenes = bool(cfg_get(config.task, "inference.deduplicate_scenes", default=True))
    seen_scene_ids = set()
    rows = []

    with torch.no_grad():
        for batch in test_loader:
            point_clouds = batch["point_clouds"].detach().cpu().numpy()
            batch_size = int(point_clouds.shape[0])
            scene_paths = _as_batch_list(batch.get("scene_path"), batch_size)
            pc_paths = _as_batch_list(batch.get("pc_path"), batch_size)
            save_paths = _as_batch_list(batch.get("save_path"), batch_size)
            grasp_type_ids = _as_batch_list(batch.get("grasp_type_id"), batch_size)

            pred_log_batch = None
            if str(checkpoint_meta.get("input_type", "bbox")) == "pointcloud":
                pred_log_batch = (
                    model(_move_tensor_batch_to_device(batch, device)).detach().cpu().numpy().astype(np.float32)
                )

            features = []
            metadata = []
            for idx in range(batch_size):
                scene_id, object_id = _scene_metadata_from_scene_path(scene_paths[idx])
                dedupe_key = scene_id or scene_paths[idx] or pc_paths[idx]
                if deduplicate_scenes and dedupe_key in seen_scene_ids:
                    continue
                seen_scene_ids.add(dedupe_key)

                feature = extract_yaw_invariant_geometry_feature(point_clouds[idx])
                features.append(feature)
                metadata.append(
                    {
                        "scene_id": scene_id,
                        "scene_path": scene_paths[idx],
                        "pc_path": pc_paths[idx],
                        "save_path": save_paths[idx],
                        "object_id": object_id,
                        "grasp_type_id": grasp_type_ids[idx],
                        "batch_idx": idx,
                    }
                )

            if not features:
                continue
            if str(checkpoint_meta.get("input_type", "bbox")) == "pointcloud":
                feature_array = np.stack(features, axis=0).astype(np.float32)
                pred_log = np.asarray([pred_log_batch[int(meta["batch_idx"])] for meta in metadata], dtype=np.float32)
            else:
                feature_array = np.stack(features, axis=0).astype(np.float32)
                feature_mean = np.asarray(checkpoint_meta["feature_mean"], dtype=np.float32)
                feature_std = np.asarray(checkpoint_meta["feature_std"], dtype=np.float32)
                normalized = ((feature_array - feature_mean) / feature_std).astype(np.float32)
                pred_log = model(torch.from_numpy(normalized).to(device)).detach().cpu().numpy().astype(np.float32)

            for idx, meta in enumerate(metadata):
                row = {
                    "scene_id": meta["scene_id"],
                    "object_id": meta["object_id"],
                    "test_data": str(config.test_data_name),
                    "scene_path": meta["scene_path"],
                    "pc_path": meta["pc_path"],
                    "save_path": meta["save_path"],
                    "grasp_type_id": meta["grasp_type_id"],
                    "pred_log_budget_multiplier": float(pred_log[idx]),
                    "pred_budget_multiplier": float(math.exp(float(pred_log[idx]))),
                }
                for feature_name, feature_value in zip(GEOMETRY_FEATURE_NAMES, feature_array[idx]):
                    row[feature_name] = float(feature_value)
                rows.append(row)
    return rows


def _predict_budget_head(config: DictConfig, output_dir: str) -> dict:
    """Predict scene-budget multipliers for the configured test data.

    Args:
        config: Full Hydra config.
        output_dir: Directory where prediction artifacts are written.

    Returns:
        Summary dictionary with the prediction CSV path and row count.
    """
    checkpoint = cfg_get(config.task, "inference.checkpoint", "checkpoint", default=None)
    if checkpoint is None:
        checkpoint = os.path.join(output_dir, "geometry_budget_head.pth")
    checkpoint = _absolute_path(str(checkpoint))

    device = str(cfg_get(config.task, "inference.device", "train.device", default=config.device))
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    model, checkpoint_meta = _load_budget_head_checkpoint(checkpoint, device=device)

    output_csv = cfg_get(config.task, "inference.output_csv", default=None)
    if output_csv is None:
        output_csv = os.path.join(output_dir, "budget_head_predictions.csv")
    output_csv = _absolute_path(str(output_csv))
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    rows = _prediction_rows_from_test_loader(
        config,
        model=model,
        checkpoint_meta=checkpoint_meta,
        device=device,
    )
    _write_csv(
        rows,
        output_csv,
        [
            "scene_id",
            "object_id",
            "test_data",
            "scene_path",
            "pc_path",
            "save_path",
            "grasp_type_id",
            "bbox_xy_major",
            "bbox_xy_minor",
            "bbox_z",
            "pred_log_budget_multiplier",
            "pred_budget_multiplier",
        ],
    )
    summary = {
        "checkpoint": checkpoint,
        "prediction_csv": output_csv,
        "prediction_row_num": int(len(rows)),
        "test_data": str(config.test_data_name),
        "input_type": str(checkpoint_meta.get("input_type", "bbox")),
        "deduplicate_scenes": bool(cfg_get(config.task, "inference.deduplicate_scenes", default=True)),
    }
    _save_json(summary, os.path.join(output_dir, "budget_head_inference_summary.json"))
    return summary


def _output_dir(config: DictConfig) -> str:
    """Resolve the scene budget output directory.

    Args:
        config: Full Hydra config.

    Returns:
        Absolute output directory path.
    """
    configured = cfg_get(config.task, "output_dir", default=None)
    if configured:
        return _absolute_path(str(configured))
    return _absolute_path(os.path.join(str(config.output_folder), str(config.wandb.id), "scene_budget"))


def task_scene_budget(config: DictConfig) -> None:
    """Run human-only geometry scene budget generation and optional budget head training.

    Args:
        config: Full Hydra config.

    Returns:
        None.
    """
    set_seed(int(config.seed))
    output_dir = _output_dir(config)
    os.makedirs(output_dir, exist_ok=True)

    mode = str(cfg_get(config.task, "mode", default="all")).lower()
    if mode == "inference":
        mode = "predict"
    if mode not in {"all", "build_labels", "train_head", "predict"}:
        raise ValueError("task.mode must be one of all, build_labels, train_head, predict")

    print(f"[scene_budget] start mode={mode}", flush=True)
    if mode == "predict":
        print("[scene_budget] loading checkpoint and running prediction", flush=True)
        prediction_summary = _predict_budget_head(config, output_dir)
        run_summary = {
            "mode": mode,
            "output_dir": output_dir,
            "prediction": prediction_summary,
            "config": OmegaConf.to_container(config.task, resolve=True),
        }
        _save_json(run_summary, os.path.join(output_dir, "scene_budget_run_summary.json"))
        print(f"Saved scene budget predictions to {prediction_summary['prediction_csv']}")
        return

    if mode in {"all", "build_labels"}:
        print("[scene_budget] building scene index", flush=True)
        label_source = str(cfg_get(config.task, "label_source", default="hierarchy_count")).lower()
        legacy_sources = {"legacy_nearest_n", "nearest_n", "diverse_grasp_class"}
        if label_source not in {"hierarchy_count"} | legacy_sources:
            raise ValueError("task.label_source must be hierarchy_count or legacy_nearest_n")

        scenes = _build_scene_index(config)
        print(f"[scene_budget] built {len(scenes)} scenes", flush=True)
        print("[scene_budget] building label hierarchy", flush=True)
        label_hierarchy_rows, label_hierarchy_summary = _build_scene_budget_label_hierarchy_rows(scenes, config.task)
        print("[scene_budget] aggregating budget rows", flush=True)
        if label_source == "hierarchy_count":
            rows, summary = _build_hierarchy_count_labels(scenes, label_hierarchy_rows, config.task)
        else:
            rows, summary = _compute_legacy_nearest_n_labels(scenes, config.task)
            summary["label_source"] = "legacy_nearest_n"
            summary.setdefault("budget_label", {})["label_source"] = "legacy_nearest_n"
        summary["label_hierarchy"] = label_hierarchy_summary
        print("[scene_budget] saving label artifacts", flush=True)
        output_paths = _save_scene_budget_outputs(rows, summary, output_dir, label_hierarchy_rows=label_hierarchy_rows)
    else:
        scene_csv = cfg_get(config.task, "scene_csv", default=None)
        summary_json = cfg_get(config.task, "summary_json", default=None)
        if scene_csv is None or summary_json is None:
            raise ValueError("task.mode=train_head requires task.scene_csv and task.summary_json")
        print(f"[scene_budget] loading training rows from {scene_csv}", flush=True)
        output_paths = {
            "scene_csv": _absolute_path(str(scene_csv)),
            "summary_json": _absolute_path(str(summary_json)),
        }
        rows = _load_scene_budget_rows(output_paths["scene_csv"])
        summary = _load_json(output_paths["summary_json"])

    if mode in {"all", "train_head"} and bool(cfg_get(config.task, "train.enabled", default=True)):
        print("[scene_budget] training budget head", flush=True)
        train_summary = _train_budget_head(rows, summary, config, output_dir)
    else:
        train_summary = None

    print("[scene_budget] writing run summary", flush=True)
    run_summary = {
        "mode": mode,
        "output_dir": output_dir,
        "scene_budget_outputs": output_paths,
        "train": train_summary,
        "config": OmegaConf.to_container(config.task, resolve=True),
    }
    _save_json(run_summary, os.path.join(output_dir, "scene_budget_run_summary.json"))
    print(f"Saved scene budget outputs to {output_dir}")
