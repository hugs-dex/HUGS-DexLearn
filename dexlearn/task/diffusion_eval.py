import math
import os
from time import perf_counter
from copy import deepcopy
from glob import glob
from typing import Any

import numpy as np
from omegaconf import DictConfig
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as SciR
from tqdm import tqdm

from dexlearn.dataset.grasp_types import GRASP_TYPES
from dexlearn.task.evaluate import (
    EPS,
    MarkdownReport,
    _abs_path,
    _as_list,
    _natural_sort_key,
    _load_json,
    _write_csv,
    _write_json,
    canonical_object_id,
    cfg_get,
    default_tests_step_dir,
    flatten_multidex_data_config,
    human_split_lookup,
    iter_human_roots,
    resolve_test_result_dir,
    scene_key,
    scene_parts_from_result,
    sequence_id_from_grasp,
)


REAL_TYPE_IDS = tuple(range(1, len(GRASP_TYPES)))
REAL_TYPE_NAMES = tuple(GRASP_TYPES[type_id] for type_id in REAL_TYPE_IDS)
MIRROR_X = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)
FIXED_LEFT_QUAT_WXYZ = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
DEFAULT_MATCH_THRESHOLDS_BY_GRASP_TYPE = {
    1: (0.03, 30.0),
    2: (0.03, 30.0),
    3: (0.03, 30.0),
    4: (0.03, 30.0),
    5: (0.05, 45.0),
}


def tqdm_progress(iterable: Any, desc: str, total: int | None = None, unit: str = "it") -> Any:
    """Wrap an iterable with a consistent tqdm progress bar.

    Args:
        iterable: Iterable object to wrap.
        desc: Short progress-bar label shown in the terminal.
        total: Optional item count. When omitted, tqdm infers it if possible.
        unit: Unit label printed by tqdm.

    Returns:
        Iterable that yields the same values while displaying progress.
    """
    return tqdm(iterable, desc=desc, total=total, unit=unit, dynamic_ncols=True)


def normalize_quaternions(quat: np.ndarray) -> np.ndarray:
    """Normalize quaternions with a small zero-norm guard.

    Args:
        quat: Quaternion array with final dimension four.

    Returns:
        Normalized float32 quaternion array.
    """
    quat = np.asarray(quat, dtype=np.float32)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-8)
    return (quat / norm).astype(np.float32, copy=False)


def quat_wxyz_from_rotmat(rotmat: np.ndarray) -> np.ndarray:
    """Convert one rotation matrix to a WXYZ quaternion.

    Args:
        rotmat: Rotation matrix shaped ``(3, 3)``.

    Returns:
        Normalized quaternion in ``[w, x, y, z]`` order.
    """
    rotmat = np.asarray(rotmat, dtype=np.float64)
    if rotmat.shape == (1, 3, 3):
        rotmat = rotmat[0]
    if rotmat.shape != (3, 3):
        raise ValueError(f"Expected rotation matrix shape (3, 3), got {rotmat.shape}")
    quat_xyzw = SciR.from_matrix(rotmat).as_quat()
    quat_wxyz = np.asarray(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float32,
    )
    return normalize_quaternions(quat_wxyz)


def rotvec_to_rotmat(rotvec: np.ndarray) -> np.ndarray:
    """Convert one rotation-vector field to a single rotation matrix.

    Args:
        rotvec: Rotation-vector array with shape ``(3,)`` or ``(1, 3)``.

    Returns:
        Rotation matrix shaped ``(3, 3)``.
    """
    rotvec = np.asarray(rotvec, dtype=np.float64)
    if rotvec.shape == (1, 3):
        rotvec = rotvec[0]
    rotvec = rotvec.reshape(3)
    return SciR.from_rotvec(rotvec).as_matrix().astype(np.float32)


def active_hand_mask(grasp_type_id: int) -> np.ndarray:
    """Return the active-hand mask for one grasp type.

    Args:
        grasp_type_id: Human grasp type id in ``[1, 5]``.

    Returns:
        Boolean array shaped ``(2,)`` for right and left hands.
    """
    mask = np.zeros(2, dtype=bool)
    mask[0] = True
    if int(grasp_type_id) >= 4:
        mask[1] = True
    return mask


def normalize_grasp_type_filter(values: Any) -> list[int]:
    """Normalize configured grasp-type filters to integer ids.

    Args:
        values: Configured grasp-type names, ids, or ``None``.

    Returns:
        Sorted list of real grasp-type ids.
    """
    if values is None:
        return list(REAL_TYPE_IDS)
    normalized = []
    for value in _as_list(values):
        if isinstance(value, (int, np.integer)):
            type_id = int(value)
        else:
            text = str(value).strip()
            if text.isdigit():
                type_id = int(text)
            elif text in GRASP_TYPES:
                type_id = GRASP_TYPES.index(text)
            else:
                raise ValueError(f"Unsupported grasp type filter: {value}")
        if type_id not in REAL_TYPE_IDS:
            raise ValueError(f"Expected real grasp type id in {REAL_TYPE_IDS}, got {type_id}")
        normalized.append(type_id)
    return sorted(set(normalized))


def grasp_type_id_from_value(value: Any) -> int:
    """Normalize one grasp type identifier to its integer id.

    Args:
        value: Grasp type id, full name such as ``1_right_two``, or short name
            such as ``right_two``.

    Returns:
        Integer real grasp type id.
    """
    if isinstance(value, (int, np.integer)):
        type_id = int(value)
    else:
        text = str(value).strip()
        if text.isdigit():
            type_id = int(text)
        elif text in GRASP_TYPES:
            type_id = GRASP_TYPES.index(text)
        else:
            matching_ids = [type_id for type_id in REAL_TYPE_IDS if GRASP_TYPES[type_id].split("_", 1)[1] == text]
            if not matching_ids:
                raise ValueError(f"Unsupported grasp type identifier: {value}")
            type_id = int(matching_ids[0])
    if type_id not in REAL_TYPE_IDS:
        raise ValueError(f"Expected real grasp type id in {REAL_TYPE_IDS}, got {type_id}")
    return type_id


def resolve_match_thresholds_by_grasp_type(task_cfg: Any) -> dict[int, dict[str, float]]:
    """Resolve record-recall thresholds for each real grasp type.

    Args:
        task_cfg: Task-level Hydra config containing optional scalar legacy
            thresholds and optional ``match_thresholds_by_grasp_type`` overrides.

    Returns:
        Mapping from grasp type id to translation and rotation thresholds.
    """
    thresholds = {
        type_id: {"translation_m": float(trans), "rotation_deg": float(rot)}
        for type_id, (trans, rot) in DEFAULT_MATCH_THRESHOLDS_BY_GRASP_TYPE.items()
    }
    configured = getattr(task_cfg, "match_thresholds_by_grasp_type", None)
    if configured is None:
        legacy_trans = getattr(task_cfg, "match_translation_threshold_m", None)
        legacy_rot = getattr(task_cfg, "match_rotation_threshold_deg", None)
        if legacy_trans is not None or legacy_rot is not None:
            for type_id in REAL_TYPE_IDS:
                if legacy_trans is not None:
                    thresholds[type_id]["translation_m"] = float(legacy_trans)
                if legacy_rot is not None:
                    thresholds[type_id]["rotation_deg"] = float(legacy_rot)
        return thresholds

    for key, value in configured.items():
        type_id = grasp_type_id_from_value(key)
        if not hasattr(value, "get"):
            raise ValueError(f"Threshold override for {key!r} must be a mapping.")
        trans = value.get("translation_m", value.get("match_translation_threshold_m", value.get("translation_threshold_m", None)))
        rot = value.get("rotation_deg", value.get("match_rotation_threshold_deg", value.get("rotation_threshold_deg", None)))
        if trans is not None:
            thresholds[type_id]["translation_m"] = float(trans)
        if rot is not None:
            thresholds[type_id]["rotation_deg"] = float(rot)
    return thresholds


def format_match_thresholds_by_grasp_type(thresholds: dict[int, dict[str, float]]) -> str:
    """Format per-grasp-type thresholds for human-readable reports.

    Args:
        thresholds: Mapping returned by ``resolve_match_thresholds_by_grasp_type``.

    Returns:
        Compact string that lists translation and rotation thresholds by type.
    """
    parts = []
    for type_id in REAL_TYPE_IDS:
        type_thresholds = thresholds[type_id]
        parts.append(
            f"{GRASP_TYPES[type_id]}<={type_thresholds['translation_m']:.3f}m/"
            f"{type_thresholds['rotation_deg']:.1f}deg"
        )
    return ", ".join(parts)


def infer_human_grasp_type_and_mirror(grasp_data: dict) -> tuple[int, bool]:
    """Infer grasp type id and whether the sample is mirrored to right-only form.

    Args:
        grasp_data: Raw formatted human grasp dictionary.

    Returns:
        Tuple ``(grasp_type_id, mirrored)``.
    """
    left = grasp_data.get("hand", {}).get("left")
    right = grasp_data.get("hand", {}).get("right")
    left_contacts = left.get("contacts", [False] * 5) if left else [False] * 5
    right_contacts = right.get("contacts", [False] * 5) if right else [False] * 5
    has_left = any(left_contacts)
    has_right = any(right_contacts)
    left_count = int(sum(left_contacts))
    right_count = int(sum(right_contacts))
    if not (has_left or has_right):
        raise ValueError("Human grasp record has no active contacts.")
    if has_left and has_right:
        return (5 if (left_count > 3 or right_count > 3) else 4), False
    active_count = left_count if has_left else right_count
    if active_count <= 2:
        grasp_type_id = 1
    elif active_count == 3:
        grasp_type_id = 2
    else:
        grasp_type_id = 3
    return grasp_type_id, bool(has_left and not has_right)


def extract_gt_pose_fields(grasp_data: dict, mirrored: bool) -> tuple[np.ndarray, np.ndarray]:
    """Extract GT index-MCP positions and wrist quaternions in world coordinates.

    Args:
        grasp_data: Raw formatted human grasp dictionary.
        mirrored: Whether a left-only grasp should be mirrored to right-only.

    Returns:
        Tuple ``(index_mcp_pos, wrist_quat)`` with shapes ``(2, 3)`` and ``(2, 4)``.
    """
    index_mcp_pos = np.zeros((2, 3), dtype=np.float32)
    wrist_quat = np.zeros((2, 4), dtype=np.float32)
    wrist_quat[1] = FIXED_LEFT_QUAT_WXYZ

    if not mirrored:
        for hand_idx, side in enumerate(("right", "left")):
            hand_data = grasp_data.get("hand", {}).get(side)
            if hand_data:
                if "index_mcp_pos" not in hand_data:
                    raise KeyError(f"GT hand '{side}' is missing index_mcp_pos.")
                index_mcp_pos[hand_idx] = np.asarray(hand_data["index_mcp_pos"], dtype=np.float32)
                rotmat = rotvec_to_rotmat(hand_data["rot"])
                wrist_quat[hand_idx] = quat_wxyz_from_rotmat(rotmat)
            elif side == "right":
                wrist_quat[hand_idx] = FIXED_LEFT_QUAT_WXYZ
        return index_mcp_pos, normalize_quaternions(wrist_quat)

    left_hand = grasp_data.get("hand", {}).get("left")
    if left_hand is None:
        raise ValueError("Mirrored grasp is missing left hand data.")
    if "index_mcp_pos" not in left_hand:
        raise KeyError("Mirrored GT left hand is missing index_mcp_pos.")
    left_pos = np.asarray(left_hand["index_mcp_pos"], dtype=np.float32)
    left_rot = rotvec_to_rotmat(left_hand["rot"])
    index_mcp_pos[0] = MIRROR_X @ left_pos
    wrist_quat[0] = quat_wxyz_from_rotmat(MIRROR_X @ left_rot @ MIRROR_X)
    return index_mcp_pos, normalize_quaternions(wrist_quat)


def split_saved_grasp_pose(grasp_pose: np.ndarray) -> np.ndarray:
    """Split a flat saved human pose into per-hand position/quaternion fields.

    Args:
        grasp_pose: Flat saved grasp pose shaped ``(7 * H,)``.

    Returns:
        Per-hand pose array shaped ``(H, 7)``.
    """
    pose = np.asarray(grasp_pose, dtype=np.float32).reshape(-1)
    if pose.size == 0 or pose.size % 7 != 0:
        raise ValueError(f"Expected grasp_pose to be non-empty and divisible by 7, got shape {np.shape(grasp_pose)}")
    hand_num = pose.size // 7
    if hand_num > 2:
        raise ValueError(f"Unsupported hand count {hand_num} in grasp_pose with shape {np.shape(grasp_pose)}")
    return pose.reshape(hand_num, 7)


def saved_sample_to_pose_fields(
    grasp_pose: np.ndarray,
    grasp_pos_source: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Read index-MCP positions and wrist quaternions from one saved sample pose.

    Args:
        grasp_pose: Flat saved grasp pose shaped ``(7 * H,)``.
        grasp_pos_source: Position target convention used during sampling.

    Returns:
        Tuple ``(index_mcp_pos, wrist_quat)`` with shapes ``(H, 3)`` and
        ``(H, 4)``.
    """
    hand_poses = split_saved_grasp_pose(np.asarray(grasp_pose, dtype=np.float32))
    grasp_pos_source = str(grasp_pos_source).strip().lower()
    if grasp_pos_source != "index_mcp":
        raise ValueError(
            "diffusion_eval no longer runs MANO recovery; saved grasp_pose "
            f"must use grasp_pos_source='index_mcp', got {grasp_pos_source!r}."
        )
    index_mcp_pos = hand_poses[:, :3]
    wrist_quat = normalize_quaternions(hand_poses[:, 3:7])
    return index_mcp_pos.astype(np.float32, copy=False), wrist_quat


def translation_distance_matrix(a_pos: np.ndarray, b_pos: np.ndarray, hand_mask: np.ndarray) -> np.ndarray:
    """Compute pairwise mean index-MCP translation distances between two pose sets.

    Args:
        a_pos: Index-MCP positions shaped ``(A, 2, 3)``.
        b_pos: Index-MCP positions shaped ``(B, 2, 3)``.
        hand_mask: Boolean active-hand mask shaped ``(2,)``.

    Returns:
        Pairwise distance matrix shaped ``(A, B)`` in meters.
    """
    active = np.asarray(hand_mask, dtype=bool)
    if not np.any(active):
        raise ValueError("At least one hand must be active for translation distances.")
    diff = a_pos[:, None, active, :] - b_pos[None, :, active, :]
    return np.linalg.norm(diff, axis=-1).mean(axis=-1)


def rotation_distance_matrix_deg(a_quat: np.ndarray, b_quat: np.ndarray, hand_mask: np.ndarray) -> np.ndarray:
    """Compute pairwise mean wrist rotation geodesic distances in degrees.

    Args:
        a_quat: Quaternions shaped ``(A, 2, 4)`` in WXYZ order.
        b_quat: Quaternions shaped ``(B, 2, 4)`` in WXYZ order.
        hand_mask: Boolean active-hand mask shaped ``(2,)``.

    Returns:
        Pairwise rotation distance matrix shaped ``(A, B)`` in degrees.
    """
    active = np.asarray(hand_mask, dtype=bool)
    if not np.any(active):
        raise ValueError("At least one hand must be active for rotation distances.")
    a = normalize_quaternions(np.asarray(a_quat, dtype=np.float32))[:, None, active, :]
    b = normalize_quaternions(np.asarray(b_quat, dtype=np.float32))[None, :, active, :]
    dots = np.abs(np.sum(a * b, axis=-1))
    dots = np.clip(dots, -1.0, 1.0)
    angles = 2.0 * np.arccos(dots)
    return np.rad2deg(angles).mean(axis=-1)


def pose_group_key(split: str, scene_key_value: str, grasp_type_id: int) -> tuple[str, str, int]:
    """Build the canonical grouping key for one scene/type set.

    Args:
        split: Dataset split name.
        scene_key_value: Canonical scene key.
        grasp_type_id: Real human grasp type id.

    Returns:
        Hashable grouping tuple.
    """
    return str(split), str(scene_key_value), int(grasp_type_id)


def pointcloud_source_from_path(pc_path: str) -> str:
    """Infer whether the saved point cloud is canonical complete or scene partial.

    Args:
        pc_path: Saved point-cloud path from the sample payload.

    Returns:
        One of ``complete``, ``partial``, or ``unknown``.
    """
    pc_path = str(pc_path or "")
    pc_base = os.path.basename(pc_path)
    if "processed_data" in pc_path or pc_base == "complete_point_cloud.npy":
        return "complete"
    if "vision_data" in pc_path or pc_base.startswith("partial_pc"):
        return "partial"
    return "unknown"


def normalize_object_scale(obj_scale: Any, scene_path: str) -> np.ndarray:
    """Normalize scene object scale to XYZ form.

    Args:
        obj_scale: Scale value from scene metadata.
        scene_path: Scene path used only for error context.

    Returns:
        Float32 XYZ scale array shaped ``(3,)``.
    """
    scale = np.asarray(obj_scale, dtype=np.float32).reshape(-1)
    if scale.size == 1:
        return np.repeat(scale, 3)
    if scale.size == 3:
        return scale
    raise ValueError(f"Unsupported object scale shape {scale.shape} in scene config: {scene_path}")


def object_pose_to_rt(obj_pose: Any, scene_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Convert scene object pose metadata into rotation and translation arrays.

    Args:
        obj_pose: Object pose entry from the scene config.
        scene_path: Scene path used only for error context.

    Returns:
        Tuple ``(rotation, translation)`` with shapes ``(3, 3)`` and ``(3,)``.
    """
    pose = np.asarray(obj_pose, dtype=np.float32)
    if pose.shape == (4, 4):
        return pose[:3, :3], pose[:3, 3]
    pose = pose.reshape(-1)
    if pose.size == 7:
        quat_xyzw = pose[3:7]
        rot = SciR.from_quat(quat_xyzw).as_matrix().astype(np.float32)
        return rot, pose[:3].astype(np.float32)
    raise ValueError(f"Unsupported object pose shape {pose.shape} in scene config: {scene_path}")


def extract_object_transform(scene_cfg: dict, scene_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract the scene object scale and pose transform.

    Args:
        scene_cfg: Loaded scene-config dictionary.
        scene_path: Absolute scene-config path.

    Returns:
        Tuple ``(obj_points_rot, obj_points_transformed_scale)`` represented as
        ``(rotation, translation)`` plus normalized XYZ scale.
    """
    if "object" in scene_cfg:
        object_data = scene_cfg["object"]
        obj_scale = object_data.get("rel_scale", object_data.get("scale"))
        obj_pose = object_data.get("pose")
    else:
        scene = scene_cfg.get("scene")
        if scene is None:
            raise KeyError(f"Could not find object entry in scene config: {scene_path}")
        object_name = scene_cfg.get("task", {}).get("obj_name")
        if object_name is None:
            candidates = [
                name
                for name, entry in scene.items()
                if isinstance(entry, dict) and "scale" in entry and "pose" in entry and name != "table"
            ]
            if not candidates:
                raise KeyError(f"Could not infer object entry in scene config: {scene_path}")
            object_name = candidates[0]
        object_data = scene.get(object_name)
        if object_data is None:
            raise KeyError(f"Could not find object '{object_name}' in scene config: {scene_path}")
        obj_scale = object_data.get("scale")
        obj_pose = object_data.get("pose")
    if obj_scale is None or obj_pose is None:
        raise KeyError(f"Scene config is missing object scale or pose: {scene_path}")
    scale_xyz = normalize_object_scale(obj_scale, scene_path)
    rot, trans = object_pose_to_rt(obj_pose, scene_path)
    return scale_xyz, rot, trans


def load_scene_object_points(
    scene_path: str,
    pc_path: str,
    max_points: int,
    cache: dict[tuple[str, str, int], np.ndarray],
) -> np.ndarray:
    """Load the object point cloud in the same world frame as saved poses.

    Args:
        scene_path: Saved scene-config path from the generated sample.
        pc_path: Saved point-cloud path from the generated sample.
        max_points: Optional deterministic point cap. Non-positive keeps all points.
        cache: In-memory point-cloud cache shared across groups.

    Returns:
        Float32 point cloud shaped ``(N, 3)`` in world coordinates.
    """
    scene_path = _abs_path(scene_path)
    pc_path = _abs_path(pc_path)
    cache_key = (scene_path, pc_path, int(max_points))
    if cache_key in cache:
        return cache[cache_key]

    raw_points = np.asarray(np.load(pc_path, allow_pickle=True), dtype=np.float32).reshape(-1, 3)
    if max_points > 0 and raw_points.shape[0] > max_points:
        indices = np.linspace(0, raw_points.shape[0] - 1, num=max_points, dtype=np.int64)
        raw_points = raw_points[indices]

    if pointcloud_source_from_path(pc_path) == "complete":
        scene_cfg = np.load(scene_path, allow_pickle=True).item()
        scale_xyz, obj_rot, obj_trans = extract_object_transform(scene_cfg, scene_path)
        points = np.matmul(raw_points * scale_xyz[None, :], obj_rot.T) + obj_trans[None, :]
    else:
        points = raw_points

    cache[cache_key] = points.astype(np.float32, copy=False)
    return cache[cache_key]


def compute_pose_to_surface_distances(
    index_mcp_pos: np.ndarray,
    hand_mask: np.ndarray,
    object_points: np.ndarray,
) -> np.ndarray:
    """Compute mean active-hand index-MCP-to-surface distances for each pose.

    Args:
        index_mcp_pos: Generated index-MCP positions shaped ``(N, 2, 3)``.
        hand_mask: Active-hand mask shaped ``(2,)``.
        object_points: Object point cloud shaped ``(P, 3)`` in world coordinates.

    Returns:
        Float array shaped ``(N,)`` with one mean nearest-surface distance per pose.
    """
    active = np.asarray(hand_mask, dtype=bool)
    if not np.any(active):
        raise ValueError("At least one active hand is required for surface-distance evaluation.")
    if object_points.size == 0:
        raise ValueError("Object point cloud must be non-empty for surface-distance evaluation.")

    query_points = np.asarray(index_mcp_pos, dtype=np.float32)[:, active, :]
    if not np.isfinite(query_points).all():
        return np.full((query_points.shape[0],), np.nan, dtype=np.float32)

    tree = cKDTree(np.asarray(object_points, dtype=np.float64))
    nearest = tree.query(query_points.reshape(-1, 3), k=1)[0]
    nearest = np.asarray(nearest, dtype=np.float32).reshape(query_points.shape[0], -1)
    return nearest.mean(axis=1)


def load_generated_pose_groups(config: DictConfig, pose_key_filter: set[tuple[str, str, int]] | None = None) -> tuple[dict, dict]:
    """Load saved diffusion samples grouped by split, scene, and grasp type.

    Args:
        config: Full Hydra config.
        pose_key_filter: Optional group-key set that needs full wrist/index-MCP
            pose fields. Groups outside the set are kept as metadata-only
            entries so missing-GT reporting remains unchanged.

    Returns:
        Tuple ``(groups, metadata)`` where ``groups`` maps group keys to pose
        arrays and ``metadata`` stores loader-level provenance.
    """
    split_lookup = human_split_lookup(config)
    results_dir = resolve_test_result_dir(config, "human_results_dir", str(config.task.human_results_dataset))
    if not results_dir or not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Human diffusion result directory not found: {results_dir}")

    grasp_type_filter = set(normalize_grasp_type_filter(getattr(config.task, "grasp_types", None)))
    sample_paths = sorted(glob(os.path.join(results_dir, "**", "*.npy"), recursive=True), key=_natural_sort_key)
    groups: dict[tuple[str, str, int], dict] = {}
    skipped_pose_file_num = 0

    print(f"[diffusion_eval] Found {len(sample_paths)} generated sample files under {results_dir}")
    for sample_path in tqdm_progress(sample_paths, desc="load generated samples", unit="file"):
        try:
            sample = np.load(sample_path, allow_pickle=True).item()
            if "grasp_pose" not in sample and ("index_mcp_pos" not in sample or "wrist_quat" not in sample):
                continue
            grasp_type_raw = np.asarray(sample.get("grasp_type_id", -1)).reshape(-1)
            if grasp_type_raw.size == 0:
                continue
            grasp_type_id = int(grasp_type_raw[0])
            if grasp_type_id not in grasp_type_filter:
                continue
            object_id, scene_id = scene_parts_from_result(sample, sample_path, results_dir)
            if not object_id:
                continue
            split = split_lookup.get(object_id, "")
            key = pose_group_key(split, scene_key(object_id, scene_id), grasp_type_id)
            group = groups.setdefault(
                key,
                {
                    "split": split,
                    "scene_key": key[1],
                    "object_id": object_id,
                    "grasp_type_id": grasp_type_id,
                    "grasp_type_name": GRASP_TYPES[grasp_type_id],
                    "scene_path": str(sample.get("scene_path", "")),
                    "pc_path": str(sample.get("pc_path", "")),
                    "sample_paths": [],
                    "wrist_quat": [],
                    "index_mcp_pos": [],
                    "grasp_error": [],
                },
            )
            group["sample_paths"].append(sample_path)
            if "grasp_error" in sample:
                group["grasp_error"].append(float(np.asarray(sample["grasp_error"]).reshape(-1)[0]))

            # Most generated groups do not have GT records in the requested
            # split. Keep their metadata for missing-GT diagnostics, but avoid
            # reading pose arrays because those poses are never scored.
            if pose_key_filter is not None and key not in pose_key_filter:
                continue

            grasp_pos_source = str(sample.get("grasp_pos_source", getattr(config.data, "hand_pos_source", "wrist")))
            if "index_mcp_pos" in sample and "wrist_quat" in sample:
                index_mcp_pos = np.asarray(sample["index_mcp_pos"], dtype=np.float32)
                wrist_quat = normalize_quaternions(np.asarray(sample["wrist_quat"], dtype=np.float32))
            elif "grasp_pose" in sample:
                index_mcp_pos, wrist_quat = saved_sample_to_pose_fields(
                    np.asarray(sample["grasp_pose"], dtype=np.float32),
                    grasp_pos_source,
                )
            else:
                raise KeyError("Generated sample must contain index_mcp_pos+wrist_quat or grasp_pose.")
        except Exception as exc:
            print(f"[diffusion_eval] Skip unreadable generated sample {sample_path}: {type(exc).__name__}: {exc}")
            skipped_pose_file_num += 1
            continue

        group["wrist_quat"].append(wrist_quat.astype(np.float32, copy=False))
        group["index_mcp_pos"].append(index_mcp_pos.astype(np.float32, copy=False))

    max_groups = int(getattr(config.task, "max_scene_type_groups", 0))
    ordered_items = sorted(groups.items(), key=lambda item: (item[0][0], _natural_sort_key(item[0][1]), item[0][2]))
    if max_groups > 0:
        ordered_items = ordered_items[:max_groups]

    normalized_groups = {}
    for key, group in tqdm_progress(ordered_items, desc="normalize generated groups", unit="group"):
        has_pose = bool(group["index_mcp_pos"])
        normalized_groups[key] = {
            **group,
            "wrist_quat": (
                normalize_quaternions(np.stack(group["wrist_quat"], axis=0))
                if has_pose
                else np.empty((0, 2, 4), dtype=np.float32)
            ),
            "index_mcp_pos": (
                np.stack(group["index_mcp_pos"], axis=0).astype(np.float32)
                if has_pose
                else np.empty((0, 2, 3), dtype=np.float32)
            ),
            "grasp_error_mean": float(np.mean(group["grasp_error"])) if group["grasp_error"] else None,
            "pose_recovered": int(has_pose),
        }

    metadata = {
        "results_dir": results_dir,
        "grasp_type_filter": [GRASP_TYPES[type_id] for type_id in sorted(grasp_type_filter)],
        "raw_sample_file_num": int(len(sample_paths)),
        "skipped_pose_file_num": int(skipped_pose_file_num),
        "scene_type_group_num": int(len(normalized_groups)),
    }
    return normalized_groups, metadata


def load_gt_pose_groups(config: DictConfig) -> tuple[dict, dict]:
    """Load GT human index-MCP positions and wrist rotations by scene/type.

    Args:
        config: Full Hydra config.

    Returns:
        Tuple ``(groups, metadata)`` where ``groups`` maps group keys to GT pose sets.
    """
    data_config = deepcopy(config.data)
    flatten_multidex_data_config(data_config)
    split_path = str(cfg_get(data_config, "split_path", "paths.split_path", default="valid_split"))
    grasp_type_filter = set(normalize_grasp_type_filter(getattr(config.task, "grasp_types", None)))
    requested_splits = [str(split) for split in _as_list(getattr(config.task, "human_splits", ["test"]))]
    groups: dict[tuple[str, str, int], dict] = {}

    for root_index, (grasp_root, object_root) in enumerate(iter_human_roots(data_config)):
        for split in requested_splits:
            split_json = os.path.join(object_root, split_path, f"{split}.json")
            if not os.path.isfile(split_json):
                continue
            object_ids = sorted(_load_json(split_json), key=_natural_sort_key)
            desc = f"load GT grasps root{root_index}:{split}"
            for object_id_raw in tqdm_progress(object_ids, desc=desc, unit="object"):
                object_id = canonical_object_id(object_id_raw)
                grasp_paths = sorted(
                    glob(os.path.join(grasp_root, str(object_id_raw), "**", "*.npy"), recursive=True),
                    key=_natural_sort_key,
                )
                for grasp_path in grasp_paths:
                    try:
                        grasp_data = np.load(grasp_path, allow_pickle=True).item()
                        grasp_type_id, mirrored = infer_human_grasp_type_and_mirror(grasp_data)
                        if grasp_type_id not in grasp_type_filter:
                            continue
                        scene_key_value = scene_key(object_id, sequence_id_from_grasp(grasp_data, grasp_path))
                        index_mcp_pos, wrist_quat = extract_gt_pose_fields(grasp_data, mirrored)
                    except Exception as exc:
                        print(f"[diffusion_eval] Skip unreadable GT grasp {grasp_path}: {type(exc).__name__}: {exc}")
                        continue

                    key = pose_group_key(split, scene_key_value, grasp_type_id)
                    group = groups.setdefault(
                        key,
                        {
                            "split": split,
                            "scene_key": scene_key_value,
                            "object_id": object_id,
                            "grasp_type_id": grasp_type_id,
                            "grasp_type_name": GRASP_TYPES[grasp_type_id],
                            "grasp_paths": [],
                            "index_mcp_pos": [],
                            "wrist_quat": [],
                        },
                    )
                    group["grasp_paths"].append(os.path.abspath(grasp_path))
                    group["index_mcp_pos"].append(index_mcp_pos.astype(np.float32, copy=False))
                    group["wrist_quat"].append(wrist_quat.astype(np.float32, copy=False))

    normalized_groups = {}
    for key, group in sorted(groups.items(), key=lambda item: (item[0][0], _natural_sort_key(item[0][1]), item[0][2])):
        normalized_groups[key] = {
            **group,
            "index_mcp_pos": np.stack(group["index_mcp_pos"], axis=0).astype(np.float32),
            "wrist_quat": normalize_quaternions(np.stack(group["wrist_quat"], axis=0)),
        }

    metadata = {
        "human_splits": requested_splits,
        "grasp_type_filter": [GRASP_TYPES[type_id] for type_id in sorted(grasp_type_filter)],
        "scene_type_group_num": int(len(normalized_groups)),
    }
    return normalized_groups, metadata


def build_group_surface_metrics(
    generated_group: dict,
    object_points_cache: dict[tuple[str, str, int], np.ndarray],
    max_surface_points: int,
) -> dict:
    """Compute generated index-MCP-to-surface sanity metrics for one scene/type.

    Args:
        generated_group: Generated pose group dictionary.
        object_points_cache: Shared posed point-cloud cache.
        max_surface_points: Optional deterministic point cap for surface lookup.

    Returns:
        Flat metric dictionary for one generated scene/type group.
    """
    grasp_type_id = int(generated_group["grasp_type_id"])
    hand_mask = active_hand_mask(grasp_type_id)
    scene_path = str(generated_group.get("scene_path", ""))
    pc_path = str(generated_group.get("pc_path", ""))
    if not scene_path or not pc_path:
        return {
            "index_mcp_surface_min_m": None,
            "index_mcp_surface_mean_m": None,
            "index_mcp_surface_max_m": None,
            "surface_distance_available": 0,
        }
    try:
        object_points = load_scene_object_points(
            scene_path=scene_path,
            pc_path=pc_path,
            max_points=max_surface_points,
            cache=object_points_cache,
        )
        pose_surface_distances = compute_pose_to_surface_distances(
            np.asarray(generated_group["index_mcp_pos"], dtype=np.float32),
            hand_mask,
            object_points,
        )
    except Exception as exc:
        print(
            "[diffusion_eval] Skip surface-distance evaluation for "
            f"{generated_group['scene_key']} / {generated_group['grasp_type_name']}: "
            f"{type(exc).__name__}: {exc}"
        )
        return {
            "index_mcp_surface_min_m": None,
            "index_mcp_surface_mean_m": None,
            "index_mcp_surface_max_m": None,
            "surface_distance_available": 0,
        }
    valid_mask = np.isfinite(pose_surface_distances)
    if not np.any(valid_mask):
        return {
            "index_mcp_surface_min_m": None,
            "index_mcp_surface_mean_m": None,
            "index_mcp_surface_max_m": None,
            "surface_distance_available": 0,
        }
    return {
        "index_mcp_surface_min_m": float(np.min(pose_surface_distances[valid_mask])),
        "index_mcp_surface_mean_m": float(np.mean(pose_surface_distances[valid_mask])),
        "index_mcp_surface_max_m": float(np.max(pose_surface_distances[valid_mask])),
        "surface_distance_available": 1,
        "pose_surface_distances": pose_surface_distances,
    }


def metric_mean(rows: list[dict], key: str) -> float | None:
    """Average one metric key while ignoring missing values.

    Args:
        rows: Metric row dictionaries.
        key: Metric field name.

    Returns:
        Mean value or ``None``.
    """
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    if not values:
        return None
    return float(np.mean(values))


def build_record_match_rows(
    generated_group: dict,
    gt_group: dict,
    match_translation_threshold_m: float,
    match_rotation_threshold_deg: float,
    surface_metrics: dict,
) -> list[dict]:
    """Evaluate whether each GT record is covered by the generated pose set.

    Args:
        generated_group: Generated pose group dictionary.
        gt_group: GT pose group dictionary.
        match_translation_threshold_m: Type-specific index-MCP translation
            threshold for record recall.
        match_rotation_threshold_deg: Type-specific rotation threshold for
            record recall.
        surface_metrics: Optional generated-group surface-distance metrics.

    Returns:
        One metric row per GT record in the group.
    """
    grasp_type_id = int(generated_group["grasp_type_id"])
    hand_mask = active_hand_mask(grasp_type_id)
    gen_pos = np.asarray(generated_group["index_mcp_pos"], dtype=np.float32)
    gen_quat = np.asarray(generated_group["wrist_quat"], dtype=np.float32)
    gt_pos = np.asarray(gt_group["index_mcp_pos"], dtype=np.float32)
    gt_quat = np.asarray(gt_group["wrist_quat"], dtype=np.float32)
    if gen_pos.shape[0] == 0:
        raise ValueError(f"Generated group {generated_group['scene_key']} / {generated_group['grasp_type_name']} has no pose fields.")

    trans = translation_distance_matrix(gen_pos, gt_pos, hand_mask)
    rot = rotation_distance_matrix_deg(gen_quat, gt_quat, hand_mask)
    combined = (
        trans / max(float(match_translation_threshold_m), EPS)
        + rot / max(float(match_rotation_threshold_deg), EPS)
    )
    match_mask = np.logical_and(trans <= match_translation_threshold_m, rot <= match_rotation_threshold_deg)
    pose_surface_distances = surface_metrics.get("pose_surface_distances")
    rows = []
    for gt_index in range(gt_pos.shape[0]):
        best_gen_index = int(np.argmin(combined[:, gt_index]))
        matched_indices = np.flatnonzero(match_mask[:, gt_index])
        rows.append(
            {
                "gt_record_index": int(gt_index),
                "gt_grasp_path": str(gt_group["grasp_paths"][gt_index]) if gt_index < len(gt_group["grasp_paths"]) else "",
                "generated_sample_num": int(gen_pos.shape[0]),
                "matched_sample_num": int(matched_indices.size),
                "record_recall": float(matched_indices.size > 0),
                "match_translation_threshold_m": float(match_translation_threshold_m),
                "match_rotation_threshold_deg": float(match_rotation_threshold_deg),
                "nn_generated_index": best_gen_index,
                "nn_trans_m": float(trans[best_gen_index, gt_index]),
                "nn_rot_deg": float(rot[best_gen_index, gt_index]),
                "nn_within_threshold": int(match_mask[best_gen_index, gt_index]),
                "nn_index_mcp_surface_m": (
                    float(pose_surface_distances[best_gen_index])
                    if pose_surface_distances is not None and np.isfinite(pose_surface_distances[best_gen_index])
                    else None
                ),
                "index_mcp_surface_min_m": surface_metrics.get("index_mcp_surface_min_m"),
                "index_mcp_surface_mean_m": surface_metrics.get("index_mcp_surface_mean_m"),
                "index_mcp_surface_max_m": surface_metrics.get("index_mcp_surface_max_m"),
                "surface_distance_available": int(surface_metrics.get("surface_distance_available", 0)),
            }
        )
    return rows


def aggregate_metric_rows(rows: list[dict]) -> list[dict]:
    """Aggregate per-record metric rows by split and grasp type.

    Args:
        rows: Per-record diffusion metric rows.

    Returns:
        Summary rows.
    """
    metric_keys = [
        "record_recall",
        "matched_sample_num",
        "match_translation_threshold_m",
        "match_rotation_threshold_deg",
        "nn_trans_m",
        "nn_rot_deg",
        "nn_index_mcp_surface_m",
        "index_mcp_surface_min_m",
        "index_mcp_surface_mean_m",
        "index_mcp_surface_max_m",
        "grasp_error_mean",
    ]
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        split = str(row.get("split", ""))
        groups.setdefault((split, "all"), []).append(row)
        groups.setdefault((split, str(row["grasp_type_name"])), []).append(row)

    summary_rows = []
    for (split, grasp_type_name), group_rows in sorted(groups.items(), key=lambda item: (item[0][0], _natural_sort_key(item[0][1]))):
        summary = {
            "split": split,
            "grasp_type_name": grasp_type_name,
            "record_num": int(len(group_rows)),
        }
        for key in metric_keys:
            summary[key] = metric_mean(group_rows, key)
        if grasp_type_name == "all":
            summary["match_translation_threshold_m"] = None
            summary["match_rotation_threshold_deg"] = None
        summary_rows.append(summary)
    return summary_rows


def default_output_dir(config: DictConfig) -> str:
    """Resolve the default output directory for diffusion evaluation artifacts.

    Args:
        config: Full Hydra config.

    Returns:
        Absolute output directory path.
    """
    configured = getattr(config.task, "output_dir", "")
    if configured:
        return _abs_path(configured)
    step_dir = default_tests_step_dir(config)
    if not step_dir:
        return _abs_path(os.path.join(str(config.output_folder), str(config.wandb.id), "diffusion_eval"))
    return os.path.join(step_dir, "diffusion_eval")


def format_float(value: Any, digits: int = 4) -> str:
    """Format optional floats for markdown output.

    Args:
        value: Numeric value or ``None``.
        digits: Decimal digit count.

    Returns:
        Formatted numeric string or ``NA``.
    """
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def build_report_lines(config: DictConfig, metadata: dict, metric_rows: list[dict], summary_rows: list[dict]) -> list[tuple[str, list[str]]]:
    """Build markdown report sections for diffusion evaluation.

    Args:
        config: Full Hydra config.
        metadata: Loader and task metadata.
        metric_rows: Per-scene/type metric rows.
        summary_rows: Aggregated summary rows.

    Returns:
        Ordered report sections.
    """
    sections = []
    sections.append(
        (
            "输入",
            [
                f"- human_results_dir: `{metadata['generated']['results_dir']}`",
                f"- generated scene/type groups: {metadata['generated']['scene_type_group_num']}",
                f"- gt scene/type groups: {metadata['gt']['scene_type_group_num']}",
                f"- evaluated GT records: {len(metric_rows)}",
                f"- grasp_types: {', '.join(metadata['generated']['grasp_type_filter'])}",
                f"- match thresholds by grasp type: {metadata['match_thresholds_by_grasp_type_text']}",
                f"- surface_max_points: {int(getattr(config.task, 'surface_max_points', 8192))}",
            ],
        )
    )
    profile = metadata.get("profile", {})
    if profile:
        sections.append(
            (
                "耗时",
                [
                    f"- {name}: {float(seconds):.2f}s"
                    for name, seconds in profile.items()
                    if isinstance(seconds, (int, float))
                ],
            )
        )
    sections.append(
        (
            "指标定义",
            [
                "- `record_recall`: 对每一个 GT record，检查该 scene/type 下生成的 index-MCP / wrist-rotation pose 集合中，是否至少有一个样本同时满足 translation 与 rotation 阈值。",
                "- recall 阈值按 grasp type 单独定义；`1_right_two` 到 `4_both_three` 使用 3cm / 30deg，`5_both_full` 使用 5cm / 45deg，除非 config 显式覆盖。",
                "- `matched_sample_num`: 对同一个 GT record，落入阈值窗口的生成样本个数；它不是 precision，只表示命中的候选数量。",
                "- `nn_trans_m`: 按 grasp type 使用对应 active hand 与对应阈值归一化后选最近邻，再统计 index MCP translation 距离。",
                "- `nn_rot_deg`: 按 grasp type 使用对应 active hand 与对应阈值归一化后选最近邻，再统计 wrist quaternion rotation geodesic 距离。",
                "- `nn_index_mcp_surface_m`: 上述最近邻生成样本的 index MCP 到物体表面的最近距离，用于排除明显离物体太远的错误姿态。",
                "- `index_mcp_surface_*`: 同一个 scene/type 的全部生成样本在 index MCP 到物体表面距离上的 min / mean / max，用于整体 sanity check。",
            ],
        )
    )
    lines = [
        "| split | grasp_type | N | T_thr(m) | R_thr(deg) | Recall | MatchedK | NN_t(m) | NN_r(deg) | NN_idx2surf(m) | SetMinSurf(m) | SetMeanSurf(m) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['split']} | {row['grasp_type_name']} | {row['record_num']} | "
            f"{format_float(row['match_translation_threshold_m'])} | {format_float(row['match_rotation_threshold_deg'])} | "
            f"{format_float(row['record_recall'])} | {format_float(row['matched_sample_num'])} | "
            f"{format_float(row['nn_trans_m'])} | {format_float(row['nn_rot_deg'])} | "
            f"{format_float(row['nn_index_mcp_surface_m'])} | {format_float(row['index_mcp_surface_min_m'])} | "
            f"{format_float(row['index_mcp_surface_mean_m'])} |"
        )
    sections.append(("汇总", lines))
    return sections


def task_diffusion_eval(config: DictConfig) -> None:
    """Run intrinsic diffusion index-MCP pose evaluation on saved human samples.

    Args:
        config: Full Hydra config.

    Returns:
        None.
    """
    total_start = perf_counter()
    profile: dict[str, float] = {}

    stage_start = perf_counter()
    gt_groups, gt_meta = load_gt_pose_groups(config)
    profile["load_gt_s"] = perf_counter() - stage_start

    stage_start = perf_counter()
    generated_groups, generated_meta = load_generated_pose_groups(config, pose_key_filter=set(gt_groups.keys()))
    profile["load_generated_s"] = perf_counter() - stage_start

    stage_start = perf_counter()
    output_dir = default_output_dir(config)
    os.makedirs(output_dir, exist_ok=True)
    profile["init_output_s"] = perf_counter() - stage_start

    match_thresholds_by_grasp_type = resolve_match_thresholds_by_grasp_type(config.task)
    max_surface_points = int(getattr(config.task, "surface_max_points", 8192))

    metric_rows = []
    missing_gt_rows = []
    object_points_cache: dict[tuple[str, str, int], np.ndarray] = {}
    surface_total_s = 0.0
    match_total_s = 0.0
    eval_start = perf_counter()
    generated_items = list(generated_groups.items())
    for key, generated_group in tqdm_progress(
        generated_items,
        desc="evaluate scene/type groups",
        total=len(generated_items),
        unit="group",
    ):
        gt_group = gt_groups.get(key)
        if gt_group is None:
            missing_gt_rows.append(
                {
                    "split": generated_group["split"],
                    "scene_key": generated_group["scene_key"],
                    "object_id": generated_group["object_id"],
                    "grasp_type_id": int(generated_group["grasp_type_id"]),
                    "grasp_type_name": generated_group["grasp_type_name"],
                }
            )
            continue

        surface_start = perf_counter()
        surface_metrics = build_group_surface_metrics(
            generated_group,
            object_points_cache=object_points_cache,
            max_surface_points=max_surface_points,
        )
        surface_total_s += perf_counter() - surface_start

        match_start = perf_counter()
        grasp_type_id = int(generated_group["grasp_type_id"])
        match_thresholds = match_thresholds_by_grasp_type[grasp_type_id]
        group_rows = build_record_match_rows(
            generated_group,
            gt_group,
            match_translation_threshold_m=float(match_thresholds["translation_m"]),
            match_rotation_threshold_deg=float(match_thresholds["rotation_deg"]),
            surface_metrics=surface_metrics,
        )
        match_total_s += perf_counter() - match_start
        for row in group_rows:
            metric_rows.append(
                {
                "split": generated_group["split"],
                "scene_key": generated_group["scene_key"],
                "object_id": generated_group["object_id"],
                "grasp_type_id": int(generated_group["grasp_type_id"]),
                "grasp_type_name": generated_group["grasp_type_name"],
                "gt_sample_num": int(np.asarray(gt_group["index_mcp_pos"]).shape[0]),
                "scene_path": generated_group["scene_path"],
                "pc_path": generated_group["pc_path"],
                "grasp_error_mean": generated_group["grasp_error_mean"],
                **row,
                }
            )
    profile["evaluate_groups_s"] = perf_counter() - eval_start
    profile["surface_distance_s"] = surface_total_s
    profile["record_match_s"] = match_total_s

    stage_start = perf_counter()
    metric_rows.sort(
        key=lambda row: (row["split"], _natural_sort_key(row["scene_key"]), row["grasp_type_id"], row["gt_record_index"])
    )
    summary_rows = aggregate_metric_rows(metric_rows)
    profile["aggregate_s"] = perf_counter() - stage_start

    metric_csv = os.path.join(output_dir, "diffusion_eval_record_metrics.csv")
    summary_csv = os.path.join(output_dir, "diffusion_eval_summary.csv")
    missing_gt_csv = os.path.join(output_dir, "diffusion_eval_missing_gt.csv")
    metadata_json = os.path.join(output_dir, "diffusion_eval_metadata.json")
    report_md = _abs_path(getattr(config.task, "report_md", "")) or os.path.join(output_dir, "diffusion_eval_report.md")

    stage_start = perf_counter()
    _write_csv(metric_rows, metric_csv)
    _write_csv(summary_rows, summary_csv)
    _write_csv(missing_gt_rows, missing_gt_csv)
    profile["write_csv_s"] = perf_counter() - stage_start
    profile["total_s"] = perf_counter() - total_start
    metadata = {
        "generated": generated_meta,
        "gt": gt_meta,
        "output_dir": output_dir,
        "metric_csv": metric_csv,
        "summary_csv": summary_csv,
        "missing_gt_csv": missing_gt_csv,
        "report_md": report_md,
        "profile": profile,
        "match_thresholds_by_grasp_type": {
            GRASP_TYPES[type_id]: match_thresholds_by_grasp_type[type_id]
            for type_id in REAL_TYPE_IDS
        },
        "match_thresholds_by_grasp_type_text": format_match_thresholds_by_grasp_type(match_thresholds_by_grasp_type),
    }
    _write_json(metadata, metadata_json)

    stage_start = perf_counter()
    report = MarkdownReport(report_md, "DexLearn Diffusion Wrist Pose Evaluation")
    for title, lines in build_report_lines(config, metadata, metric_rows, summary_rows):
        report.add_section(title, lines)
    profile["write_report_s"] = perf_counter() - stage_start
    profile["total_s"] = perf_counter() - total_start
    metadata["profile"] = profile
    _write_json(metadata, metadata_json)

    print("[diffusion_eval] Timing summary:")
    for name, seconds in sorted(profile.items(), key=lambda item: item[1], reverse=True):
        print(f"[diffusion_eval]   {name}: {seconds:.2f}s")

    print(f"[diffusion_eval] Wrote record metrics to {metric_csv}")
    print(f"[diffusion_eval] Wrote summary to {summary_csv}")
    if missing_gt_rows:
        print(f"[diffusion_eval] Wrote {len(missing_gt_rows)} missing-GT rows to {missing_gt_csv}")
    print(f"[diffusion_eval] Wrote report to {report_md}")
