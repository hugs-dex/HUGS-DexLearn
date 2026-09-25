import os
import random
from glob import glob

import numpy as np
import torch
import trimesh
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from pytorch3d.transforms import matrix_to_axis_angle, quaternion_to_matrix

from dexlearn.dataset.grasp_types import GRASP_TYPES
from dexlearn.task.visualize import (
    VISER_AVAILABLE,
    compact_caption,
    extract_object_meta,
    format_grasp_type_option,
    normalize_visualize_mode,
    parse_grasp_type_option,
    pick_initial_object_option,
    pick_initial_option,
    progress_iter,
    resolve_human_dataset_path,
    show_scenes_with_trimesh,
    show_scenes_with_viser,
    transform_complete_pc,
    visualize_with_trimesh,
)
from dexlearn.utils.human_hand import get_wrist_translation_from_target, normalize_hand_pos_source
from dexlearn.utils.util import set_seed


HUMAN_PRIOR_VISER_MODE_OPTIONS = ("random_objects", "one_scene")
REAL_GRASP_TYPE_IDS = tuple(range(1, len(GRASP_TYPES)))
HUMAN_BOTH_THREE_TYPE_ID = 4
FINGERTIP_JOINT_INDICES = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}
GRASP_TYPE_FINGERS = {
    1: {"right": ["thumb", "index"], "left": []},
    2: {"right": ["thumb", "index", "middle"], "left": []},
    3: {"right": ["thumb", "index", "middle", "ring", "pinky"], "left": []},
    4: {"right": ["thumb", "index", "middle"], "left": ["thumb", "index", "middle"]},
    5: {
        "right": ["thumb", "index", "middle", "ring", "pinky"],
        "left": ["thumb", "index", "middle", "ring", "pinky"],
    },
}


def get_task_value(config: DictConfig, key: str, default):
    """Read a value from ``config.task`` with a Python default.

    Args:
        config: Full Hydra config.
        key: Field name under ``task``.
        default: Value returned when the field is missing or null.

    Returns:
        Configured value or ``default``.
    """
    value = OmegaConf.select(config, f"task.{key}")
    return default if value is None else value


def natural_sort_key(value) -> list:
    """Build a human-friendly sort key for paths and object ids.

    Args:
        value: Value to sort.

    Returns:
        List containing text and integer tokens.
    """
    import re

    return [int(token) if token.isdigit() else token for token in re.split(r"(\d+)", str(value))]


def normalize_prior_visualize_mode(mode: str) -> str:
    """Normalize visualize_human_prior mode names and legacy aliases.

    Args:
        mode: User-provided mode name.

    Returns:
        Canonical mode name used by this task.
    """
    mode = normalize_visualize_mode(mode)
    aliases = {
        "one_object": "one_scene",
        "single_object": "one_scene",
        "object": "one_scene",
        "scene": "one_scene",
        "single_scene": "one_scene",
        "random_scene": "one_scene",
    }
    return aliases.get(str(mode), str(mode))


def resolve_prior_export_dir(config: DictConfig) -> str:
    """Resolve the robot-specific object human-prior export directory.

    Args:
        config: Full Hydra config containing ``task.prior_dir``.

    Returns:
        Absolute path whose direct children are object export directories.
    """
    prior_dir = to_absolute_path(str(get_task_value(config, "prior_dir", "")))
    if not prior_dir:
        raise ValueError("task.prior_dir must be set for task=visualize_human_prior.")

    if not os.path.isdir(prior_dir):
        raise FileNotFoundError(f"Object human-prior export directory not found: {prior_dir}")
    return prior_dir


def list_subdirs(path: str) -> list[str]:
    """List immediate subdirectories in stable order.

    Args:
        path: Directory to scan.

    Returns:
        Sorted absolute child directory paths.
    """
    try:
        children = [entry.path for entry in os.scandir(path) if entry.is_dir()]
    except FileNotFoundError:
        return []
    return sorted(children, key=natural_sort_key)


def discover_object_roots(prior_dir: str) -> dict[str, str]:
    """Discover object directories without scanning every scene export.

    Args:
        prior_dir: Robot-specific export directory produced by
            ``obj_human_prior_export``. This should be shaped like
            ``obj_human_prior/<step>/<asset>/<robot_name>``.

    Returns:
        Mapping from object id to object root directory.
    """
    object_roots: dict[str, str] = {}
    for object_root in list_subdirs(prior_dir):
        object_roots.setdefault(os.path.basename(object_root), object_root)
    return dict(sorted(object_roots.items(), key=lambda item: natural_sort_key(item[0])))


def load_prior_scene(scene_file: str) -> dict:
    """Load one per-scene object human-prior export.

    Args:
        scene_file: Path to a saved per-scene ``.npy`` file.

    Returns:
        Scene export dictionary.
    """
    scene_data = np.load(scene_file, allow_pickle=True).item()
    required_keys = {"scene_id", "object_id", "budget_scores", "wrist_quat", "active_hand_mask", "pc_path"}
    missing_keys = sorted(required_keys.difference(scene_data.keys()))
    if missing_keys:
        raise KeyError(f"{scene_file} is missing required object human-prior fields: {missing_keys}")
    return scene_data


def scene_record_from_file(scene_file: str, object_id: str, object_root: str) -> dict:
    """Create path-derived scene metadata without loading the payload.

    Args:
        scene_file: Per-scene export file path.
        object_id: Object id associated with ``object_root``.
        object_root: Root directory for this object's scene exports.

    Returns:
        Lightweight scene record used by lazy visualization selection.
    """
    rel_path = os.path.relpath(scene_file, object_root)
    scene_suffix = os.path.splitext(rel_path)[0].replace(os.sep, "/")
    return {
        "scene_file": scene_file,
        "scene_id": f"{object_id}/{scene_suffix}",
        "object_id": object_id,
        "split": "",
        "score_semantics": "",
        "budget_scores": None,
    }


def load_object_scene_records(prior_index: dict, object_id: str) -> list[dict]:
    """Load or build the lazy scene-file list for one object.

    Args:
        prior_index: Index returned by ``build_prior_index``.
        object_id: Object id selected by random or one-object mode.

    Returns:
        Sorted lightweight scene records for the object.
    """
    object_id = str(object_id)
    if object_id not in prior_index["object_roots"]:
        raise KeyError(f"Unknown object_id={object_id}.")
    cached_records = prior_index["object_scene_cache"].get(object_id)
    if cached_records is not None:
        return cached_records

    object_root = prior_index["object_roots"][object_id]
    scene_files = [
        scene_file
        for scene_file in glob(os.path.join(object_root, "**", "*.npy"), recursive=True)
    ]
    scene_files = sorted(scene_files, key=natural_sort_key)
    if not scene_files:
        raise RuntimeError(f"No per-scene object human-prior files found under {object_root}")
    records = [scene_record_from_file(scene_file, object_id, object_root) for scene_file in scene_files]
    prior_index["object_scene_cache"][object_id] = records
    return records


def scene_record_from_scene_id(prior_index: dict, scene_id: str) -> dict:
    """Create a scene record directly from a scene id.

    Args:
        prior_index: Index returned by ``build_prior_index``.
        scene_id: Scene id shaped like ``object_id/tabletop_ur10e/scale...``.

    Returns:
        Lightweight scene record for the requested scene.
    """
    scene_id = str(scene_id).strip().strip("/")
    if "/" not in scene_id:
        raise ValueError(f"scene_id must include an object id and scene suffix: {scene_id}")
    object_id, scene_suffix = scene_id.split("/", 1)
    if object_id not in prior_index["object_roots"]:
        raise KeyError(f"Unknown object_id={object_id} in scene_id={scene_id}.")
    scene_file = os.path.join(prior_index["object_roots"][object_id], f"{scene_suffix}.npy")
    if not os.path.isfile(scene_file):
        raise FileNotFoundError(f"Could not resolve scene_id={scene_id} to export file: {scene_file}")
    record = scene_record_from_file(scene_file, object_id, prior_index["object_roots"][object_id])
    prior_index["scene_record_by_id"][record["scene_id"]] = record
    return record


def scene_options(prior_index: dict, config: DictConfig) -> tuple[str, ...]:
    """Return a bounded random scene-id pool for one-scene mode.

    Args:
        prior_index: Index returned by ``build_prior_index``.
        config: Full Hydra config containing ``task.scene_option_count``.

    Returns:
        Tuple of selectable scene ids. This is intentionally bounded so the
        web UI does not need to receive hundreds of thousands of options.
    """
    preferred_scene_id = get_task_value(config, "scene_id", None)
    option_count = int(get_task_value(config, "scene_option_count", 256))
    options: list[str] = []
    seen: set[str] = set()

    if preferred_scene_id:
        preferred_record = scene_record_from_scene_id(prior_index, str(preferred_scene_id))
        options.append(preferred_record["scene_id"])
        seen.add(preferred_record["scene_id"])

    max_attempts = max(option_count * 4, 32)
    for _ in range(max_attempts):
        if option_count > 0 and len(options) >= option_count:
            break
        record = random_scene_record(prior_index)
        if record["scene_id"] in seen:
            continue
        options.append(record["scene_id"])
        seen.add(record["scene_id"])

    if not options:
        record = random_scene_record(prior_index)
        options.append(record["scene_id"])
    return tuple(options)


def find_scene_record(prior_index: dict, scene_id: str) -> dict:
    """Find a scene record by id.

    Args:
        prior_index: Index returned by ``build_prior_index``.
        scene_id: Scene id selected in one-scene mode.

    Returns:
        Lightweight scene record for the requested scene.
    """
    record = prior_index["scene_record_by_id"].get(str(scene_id))
    return record if record is not None else scene_record_from_scene_id(prior_index, str(scene_id))


def random_scene_record(prior_index: dict) -> dict:
    """Select one random scaled/posed scene without global payload loading.

    Args:
        prior_index: Index returned by ``build_prior_index``.

    Returns:
        Lightweight scene record for a random scene.
    """
    object_id = random.choice(prior_index["object_options"])
    record = random.choice(load_object_scene_records(prior_index, object_id))
    prior_index["scene_record_by_id"][record["scene_id"]] = record
    return record


def build_prior_index(prior_dir: str) -> dict:
    """Build a lightweight index over per-scene object human-prior exports.

    Args:
        prior_dir: Robot-specific export directory produced by
            ``obj_human_prior_export``.

    Returns:
        Dictionary with sorted scene records and object-id lookup tables.
    """
    object_roots = discover_object_roots(prior_dir)
    if not object_roots:
        raise RuntimeError(f"No object human-prior exports found in task.prior_dir={prior_dir}")

    return {
        "prior_dir": prior_dir,
        "object_roots": object_roots,
        "object_scene_cache": {},
        "scene_record_by_id": {},
        "object_options": tuple(sorted(object_roots.keys(), key=natural_sort_key)),
    }


def enabled_grasp_type_ids(config: DictConfig) -> tuple[int, ...]:
    """Return enabled real grasp types for visualize_human_prior.

    Args:
        config: Full Hydra config.

    Returns:
        Ordered tuple of enabled real grasp type ids.
    """
    exclude_both_three = bool(get_task_value(config, "exclude_both_three", False))
    return tuple(
        grasp_type_id
        for grasp_type_id in REAL_GRASP_TYPE_IDS
        if not (exclude_both_three and grasp_type_id == HUMAN_BOTH_THREE_TYPE_ID)
    )


def scene_enabled_type_pairs(scene_data: dict, enabled_type_ids: tuple[int, ...]) -> list[tuple[int, int]]:
    """Return exported grasp-type rows that are enabled for display.

    Args:
        scene_data: Loaded per-scene export dictionary.
        enabled_type_ids: Enabled real grasp type ids.

    Returns:
        List of ``(export_row_index, grasp_type_id)`` pairs in export order.
    """
    grasp_type_ids = np.asarray(scene_data.get("grasp_type_ids", REAL_GRASP_TYPE_IDS), dtype=np.int64).reshape(-1)
    enabled_type_id_set = {int(grasp_type_id) for grasp_type_id in enabled_type_ids}
    return [
        (row_index, int(grasp_type_id))
        for row_index, grasp_type_id in enumerate(grasp_type_ids.tolist())
        if int(grasp_type_id) in enabled_type_id_set
    ]


def point_cloud_color_rgba(config: DictConfig) -> list[int]:
    """Resolve the configured point-cloud color to one RGBA list.

    Args:
        config: Full Hydra config.

    Returns:
        RGBA color list with integer values in ``0..255``.
    """
    color_value = get_task_value(config, "point_cloud_color", "red")
    if isinstance(color_value, str):
        presets = {
            "red": [255, 0, 0, 255],
            "gray": [160, 160, 160, 255],
        }
        color_key = str(color_value).strip().lower()
        if color_key in presets:
            return presets[color_key]
        raise ValueError(f"Unsupported task.point_cloud_color={color_value}. Expected one of {sorted(presets)}.")

    color_array = np.asarray(color_value, dtype=np.int64).reshape(-1)
    if color_array.size == 3:
        color_array = np.concatenate([color_array, np.asarray([255], dtype=np.int64)])
    if color_array.size != 4:
        raise ValueError("task.point_cloud_color must be a preset string or an RGB/RGBA array.")
    color_array = np.clip(color_array, 0, 255)
    return [int(channel) for channel in color_array.tolist()]


def one_scene_score_markdown(scene_data: dict, enabled_type_ids: tuple[int, ...]) -> str:
    """Format per-grasp-type scores for the one-scene side panel.

    Args:
        scene_data: Loaded per-scene export dictionary.
        enabled_type_ids: Enabled real grasp type ids.

    Returns:
        Markdown text listing one score per enabled grasp type.
    """
    grasp_type_ids = np.asarray(scene_data.get("grasp_type_ids", REAL_GRASP_TYPE_IDS), dtype=np.int64).reshape(-1)
    scores = np.asarray(scene_data["budget_scores"], dtype=np.float32).reshape(-1)
    score_by_type = {
        int(grasp_type_id): float(scores[idx])
        for idx, grasp_type_id in enumerate(grasp_type_ids.tolist())
        if idx < scores.shape[0]
    }
    lines = ["Grasp Type Scores:"]
    for grasp_type_id in enabled_type_ids:
        if int(grasp_type_id) not in score_by_type:
            continue
        lines.append(f"- `{GRASP_TYPES[int(grasp_type_id)]}`: {score_by_type[int(grasp_type_id)]:.4f}")
    return "\n".join(lines)


def target_fingertip_indices(grasp_type_id: int, side: str) -> list[int]:
    """Return MANO fingertip joints selected by the grasp type.

    Args:
        grasp_type_id: Grasp type id in ``[1, 5]``.
        side: Hand side, either ``right`` or ``left``.

    Returns:
        MANO joint indices for fingertips that participate in the grasp type.
    """
    finger_names = GRASP_TYPE_FINGERS.get(int(grasp_type_id), {}).get(str(side), [])
    return [FINGERTIP_JOINT_INDICES[name] for name in finger_names]


def resolve_existing_path(path: str) -> str:
    """Resolve a saved dataset path against the current machine.

    Args:
        path: Saved absolute or workspace-relative path.

    Returns:
        Existing path when resolvable, otherwise the original path.
    """
    path = str(path)
    if os.path.exists(path):
        return path
    resolved = to_absolute_path(path)
    if os.path.exists(resolved):
        return resolved
    try:
        resolved = resolve_human_dataset_path(path)
    except Exception:
        resolved = path
    return resolved


def score_text(scores: np.ndarray, grasp_type_ids=None, enabled_type_ids=None) -> str:
    """Format the five grasp-type scores for compact labels.

    Args:
        scores: Score vector aligned with grasp type ids 1..5.
        grasp_type_ids: Optional grasp type ids aligned with ``scores``.
        enabled_type_ids: Optional enabled grasp type ids.

    Returns:
        Compact score string.
    """
    score_values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if grasp_type_ids is not None and enabled_type_ids is not None:
        grasp_type_ids = np.asarray(grasp_type_ids, dtype=np.int64).reshape(-1)
        enabled_type_id_set = {int(grasp_type_id) for grasp_type_id in enabled_type_ids}
        keep_indices = [
            idx
            for idx, grasp_type_id in enumerate(grasp_type_ids.tolist())
            if int(grasp_type_id) in enabled_type_id_set
        ]
        score_values = score_values[keep_indices]
    return "[" + ", ".join(f"{float(score):.2f}" for score in score_values) + "]"


def type_score(scene_data: dict, grasp_type_id: int) -> float:
    """Read one grasp-type score from a scene export.

    Args:
        scene_data: Loaded per-scene export dictionary.
        grasp_type_id: Real grasp type id in ``1..5``.

    Returns:
        Score value for that grasp type.
    """
    grasp_type_ids = np.asarray(scene_data.get("grasp_type_ids", REAL_GRASP_TYPE_IDS), dtype=np.int64).reshape(-1)
    matches = np.where(grasp_type_ids == int(grasp_type_id))[0]
    if len(matches) == 0:
        raise KeyError(f"grasp_type_id={grasp_type_id} not found in scene export.")
    return float(np.asarray(scene_data["budget_scores"], dtype=np.float32).reshape(-1)[int(matches[0])])


def load_scene_point_cloud(scene_data: dict, scene_path: str, num_points: int) -> np.ndarray:
    """Load and transform the exported scene point cloud for visualization.

    Args:
        scene_data: Loaded per-scene export dictionary.
        scene_path: Resolved scene config path.
        num_points: Number of points to sample for display.

    Returns:
        Point cloud in the same world frame as the exported hand poses.
    """
    pc_path = resolve_existing_path(str(scene_data["pc_path"]))
    pc = np.load(pc_path, allow_pickle=True)
    if int(num_points) > 0 and pc.shape[0] > int(num_points):
        idx = np.random.choice(pc.shape[0], int(num_points), replace=False)
        pc = pc[idx]
    elif int(num_points) > 0 and pc.shape[0] < int(num_points):
        idx = np.random.choice(pc.shape[0], int(num_points), replace=True)
        pc = pc[idx]

    if "complete_point_cloud.npy" in os.path.basename(pc_path) or "processed_data" in pc_path:
        scene_cfg = np.load(scene_path, allow_pickle=True).item()
        _, obj_scale_xyz, obj_rot, obj_trans = extract_object_meta(scene_cfg, scene_path)
        return transform_complete_pc(pc, obj_scale_xyz, obj_rot, obj_trans)
    return np.asarray(pc, dtype=np.float32)


def create_mano_layers(device: str) -> dict:
    """Create MANO layers for right and left human hands.

    Args:
        device: Torch device used for MANO forward passes.

    Returns:
        Mapping from side name to MANO layer.
    """
    from dexlearn.utils.dependencies import require_module
    ManoLayer = require_module("manopth.manolayer", "human prior visualization").ManoLayer
    return {
        side: ManoLayer(
            center_idx=0,
            mano_root=os.environ.get("MANO_ROOT", "assets/mano"),
            side=side,
            use_pca=False,
            flat_hand_mean=True,
            ncomps=45,
            root_rot_mode="axisang",
            joint_rot_mode="axisang",
        ).to(device)
        for side in ("right", "left")
    }


def add_hand_pose_elements(
    scene_elements: list,
    mano_layers: dict,
    device: str,
    position: np.ndarray,
    wrist_quat: np.ndarray,
    active_hand_mask: np.ndarray,
    grasp_pos_source: str,
    grasp_type_id: int,
    show_mano_wrist_frame: bool = True,
    show_index_mcp_frame: bool = True,
) -> None:
    """Append MANO meshes and pose axes for active hands.

    Args:
        scene_elements: Mutable list of trimesh geometries.
        mano_layers: Mapping returned by ``create_mano_layers``.
        device: Torch device used for tensor conversion.
        position: Target position array shaped ``(2, 3)``.
        wrist_quat: Wrist quaternion array shaped ``(2, 4)`` in WXYZ order.
        active_hand_mask: Boolean active-hand mask shaped ``(2,)``.
        grasp_pos_source: Whether ``position`` stores wrist or index-MCP points.
        grasp_type_id: Grasp type id used to choose highlighted fingertips.
        show_mano_wrist_frame: Whether to show the MANO wrist frame axis.
        show_index_mcp_frame: Whether to show the target index-MCP frame axis.

    Returns:
        None. ``scene_elements`` is modified in place.
    """
    side_names = ("right", "left")
    colors = ([180, 200, 255, 220], [210, 190, 250, 220])
    for hand_idx, side in enumerate(side_names):
        if hand_idx >= len(active_hand_mask) or not bool(active_hand_mask[hand_idx]):
            continue

        hand_target_pos = torch.from_numpy(np.asarray(position[hand_idx], dtype=np.float32)).to(device)
        quat = torch.from_numpy(np.asarray(wrist_quat[hand_idx], dtype=np.float32)).to(device)
        quat = quat / torch.clamp(torch.linalg.norm(quat), min=1e-8)
        hand_rot_mat = quaternion_to_matrix(quat.unsqueeze(0)).float()
        mano_params = torch.cat(
            [matrix_to_axis_angle(hand_rot_mat), torch.zeros((1, 45), device=device)], dim=-1
        )

        mano_layer = mano_layers[side]
        verts, joints = mano_layer(mano_params, th_betas=torch.zeros((1, 10), device=device))
        wrist_trans = get_wrist_translation_from_target(hand_target_pos.float(), joints[0], grasp_pos_source)

        hand_pose_np = np.eye(4, dtype=np.float32)
        hand_pose_np[:3, :3] = hand_rot_mat[0].detach().cpu().numpy()
        hand_pose_np[:3, 3] = wrist_trans.detach().cpu().numpy()
        if bool(show_mano_wrist_frame):
            scene_elements.append(trimesh.creation.axis(transform=hand_pose_np, origin_size=0.01))

        v_np = ((verts[0] / 1000.0) + wrist_trans).detach().cpu().numpy()
        f_np = mano_layer.th_faces.detach().cpu().numpy()
        joints_np = ((joints[0] / 1000.0) + wrist_trans).detach().cpu().numpy()
        scene_elements.extend(visualize_with_trimesh(v_np, f_np, None, color=colors[hand_idx]))
        for fingertip_index in target_fingertip_indices(int(grasp_type_id), side):
            sphere = trimesh.creation.uv_sphere(radius=0.005)
            sphere.visual.face_colors = [255, 0, 0, 255]
            sphere.apply_translation(joints_np[fingertip_index])
            scene_elements.append(sphere)

        if grasp_pos_source == "index_mcp" and bool(show_index_mcp_frame):
            target_pose_np = np.eye(4, dtype=np.float32)
            target_pose_np[:3, :3] = hand_rot_mat[0].detach().cpu().numpy()
            target_pose_np[:3, 3] = hand_target_pos.detach().cpu().numpy()
            scene_elements.append(
                trimesh.creation.axis(
                    transform=target_pose_np,
                    origin_size=0.008,
                    axis_radius=0.003,
                    axis_length=0.08,
                )
            )


def build_score_scene_record(record: dict, scene_data: dict, pc: np.ndarray, config: DictConfig) -> dict:
    """Build one score-only scene record for random-object mode.

    Args:
        record: Indexed scene metadata.
        scene_data: Loaded per-scene export dictionary.
        pc: Visualization point cloud in world coordinates.
        config: Full Hydra config.

    Returns:
        Scene record consumed by the shared viser/trimesh renderer.
    """
    scores = np.asarray(scene_data["budget_scores"], dtype=np.float32)
    grasp_type_ids = np.asarray(scene_data.get("grasp_type_ids", REAL_GRASP_TYPE_IDS), dtype=np.int64).reshape(-1)
    type_ids = enabled_grasp_type_ids(config)
    scene_elements = [
        trimesh.points.PointCloud(pc, colors=point_cloud_color_rgba(config)),
        trimesh.creation.axis(
            origin_size=float(get_task_value(config, "scene_origin_axis_origin_size", 0.01)),
            axis_radius=float(get_task_value(config, "scene_origin_axis_radius", 0.001)),
            axis_length=float(get_task_value(config, "scene_origin_axis_length", 0.3)),
        ),
    ]
    label = score_text(scores, grasp_type_ids=grasp_type_ids, enabled_type_ids=type_ids)
    caption = (
        f"object={record['object_id']} | scene={record['scene_id']} | "
        f"scores={label} | semantics={scene_data.get('score_semantics', '')}"
    )
    return {
        "elements": scene_elements,
        "caption": caption,
        "label_caption": label,
        "viser_all_label": record["object_id"],
        "source_scene_id": record["scene_id"],
        "source_object_id": record["object_id"],
    }


def grasp_type_index(scene_data: dict, grasp_type_id: int) -> int:
    """Find the row index for an exported grasp type id.

    Args:
        scene_data: Loaded per-scene export dictionary.
        grasp_type_id: Real grasp type id in ``1..5``.

    Returns:
        Zero-based index into pose arrays for the requested grasp type.
    """
    grasp_type_ids = np.asarray(scene_data.get("grasp_type_ids", REAL_GRASP_TYPE_IDS), dtype=np.int64).reshape(-1)
    matches = np.where(grasp_type_ids == int(grasp_type_id))[0]
    if len(matches) == 0:
        raise KeyError(f"grasp_type_id={grasp_type_id} not found in scene export.")
    return int(matches[0])


def build_pose_scene_record(
    record: dict,
    scene_data: dict,
    pc: np.ndarray,
    mano_layers: dict,
    device: str,
    grasp_type_index: int,
    sample_index: int,
    visible_samples_per_type: int,
    config: DictConfig,
    show_caption: bool = True,
) -> dict:
    """Build one wrist-pose scene record for one-scene mode.

    Args:
        record: Indexed scene metadata.
        scene_data: Loaded per-scene export dictionary.
        pc: Visualization point cloud in world coordinates.
        mano_layers: Mapping returned by ``create_mano_layers``.
        device: Torch device used for MANO forward passes.
        grasp_type_index: Zero-based index into exported grasp type arrays.
        sample_index: Zero-based sample index within that type.

    Returns:
        Scene record consumed by the shared viser/trimesh renderer.
    """
    grasp_type_ids = np.asarray(scene_data.get("grasp_type_ids", REAL_GRASP_TYPE_IDS), dtype=np.int64)
    grasp_type_id = int(grasp_type_ids[grasp_type_index])
    position_key = str(scene_data.get("export_position_key", "index_mcp_pos"))
    if position_key not in scene_data:
        position_key = "index_mcp_pos" if "index_mcp_pos" in scene_data else "wrist_pos"
    grasp_pos_source = normalize_hand_pos_source(scene_data.get("grasp_pos_source", "wrist"))
    show_mano_wrist_frame = bool(get_task_value(config, "show_mano_wrist_frame", True))
    show_index_mcp_frame = bool(get_task_value(config, "show_index_mcp_frame", True))

    position = np.asarray(scene_data[position_key], dtype=np.float32)[grasp_type_index, sample_index]
    wrist_quat = np.asarray(scene_data["wrist_quat"], dtype=np.float32)[grasp_type_index, sample_index]
    active_hand_mask = np.asarray(scene_data["active_hand_mask"], dtype=bool)[grasp_type_index, sample_index]

    scene_elements = [trimesh.points.PointCloud(pc, colors=point_cloud_color_rgba(config))]
    add_hand_pose_elements(
        scene_elements,
        mano_layers,
        device,
        position,
        wrist_quat,
        active_hand_mask,
        grasp_pos_source,
        grasp_type_id=grasp_type_id,
        show_mano_wrist_frame=show_mano_wrist_frame,
        show_index_mcp_frame=show_index_mcp_frame,
    )
    scene_elements.append(
        trimesh.creation.axis(
            origin_size=float(get_task_value(config, "scene_origin_axis_origin_size", 0.01)),
            axis_radius=float(get_task_value(config, "scene_origin_axis_radius", 0.001)),
            axis_length=float(get_task_value(config, "scene_origin_axis_length", 0.3)),
        )
    )

    score = type_score(scene_data, grasp_type_id)
    label = f"{GRASP_TYPES[grasp_type_id]} | {score:.2f} | {sample_index + 1}"
    caption = (
        f"object={record['object_id']} | scene={record['scene_id']} | "
        f"type={GRASP_TYPES[grasp_type_id]} | type_score={score:.4f} | "
        f"sample={sample_index + 1} | position={position_key}"
    )
    return {
        "elements": scene_elements,
        "caption": caption if show_caption else "",
        "label_caption": label if show_caption else "",
        "viser_all_label": f"{GRASP_TYPES[grasp_type_id]} | {sample_index + 1}",
        "source_scene_id": record["scene_id"],
        "source_object_id": record["object_id"],
        "viser_grid_row": grasp_type_index,
        "viser_grid_col": sample_index,
        "viser_grid_rows": len(grasp_type_ids),
        "viser_grid_cols": int(visible_samples_per_type),
    }


def load_record_scene_payload(record: dict, num_points: int) -> tuple[dict, np.ndarray]:
    """Load scene export and point cloud for one indexed record.

    Args:
        record: Indexed scene metadata.
        num_points: Number of points to sample for display.

    Returns:
        Tuple ``(scene_data, point_cloud)``.
    """
    scene_data = load_prior_scene(record["scene_file"])
    scene_path = resolve_existing_path(str(scene_data.get("scene_path", "")))
    if not scene_path or not os.path.exists(scene_path):
        raise FileNotFoundError(f"scene_path is not resolvable for {record['scene_file']}: {scene_path}")
    pc = load_scene_point_cloud(scene_data, scene_path, num_points)
    return scene_data, pc


def random_object_records(prior_index: dict, max_objects: int, batch_index: int) -> list[dict]:
    """Select one representative scene for random object score display.

    Args:
        prior_index: Index returned by ``build_prior_index``.
        max_objects: Maximum number of objects to show.
        batch_index: Zero-based batch index for paging through shuffled objects.

    Returns:
        List of indexed scene records.
    """
    object_ids = list(prior_index["object_options"])
    random.shuffle(object_ids)
    if int(max_objects) > 0 and object_ids:
        start = (max(0, int(batch_index)) * int(max_objects)) % len(object_ids)
        object_ids = object_ids[start:] + object_ids[:start]
        object_ids = object_ids[: int(max_objects)]
    selected = []
    for object_id in object_ids:
        selected.append(random.choice(load_object_scene_records(prior_index, object_id)))
    return selected


def one_scene_record(prior_index: dict, scene_id: str | None = None, random_next: bool = False) -> dict:
    """Select one scaled/posed scene for pose display.

    Args:
        prior_index: Index returned by ``build_prior_index``.
        scene_id: Scene id selected in the UI.
        random_next: Whether to ignore ``scene_id`` and choose a random scene.

    Returns:
        Indexed scene record.
    """
    if random_next or not scene_id:
        return random_scene_record(prior_index)
    return find_scene_record(prior_index, str(scene_id))


def one_scene_grid_row(grasp_type_index_value: int, target_grasp_type_id: int) -> int:
    """Resolve the displayed row for one-scene pose records.

    Args:
        grasp_type_index_value: Zero-based row index in the exported pose array.
        target_grasp_type_id: Selected grasp type id. ``0`` means all real
            grasp types are visible.

    Returns:
        Grid row used by the shared viser renderer. A concrete selected type is
        displayed as a single-row grid, while ``0_any`` preserves export rows.
    """
    return 0 if int(target_grasp_type_id) != 0 else int(grasp_type_index_value)


def build_scene_records(
    prior_index: dict,
    mode: str,
    object_id: str,
    batch_index: int,
    config: DictConfig,
    mano_layers: dict | None = None,
    grasp_type_id: int = 0,
) -> list[dict]:
    """Build renderer scene records for the selected visualization mode.

    Args:
        prior_index: Index returned by ``build_prior_index``.
        mode: Visualization mode, either ``random_objects`` or ``one_scene``.
        object_id: Selected object id or scene id, depending on mode.
        batch_index: Zero-based page index.
        config: Full Hydra config.
        mano_layers: Optional MANO layers required for pose rendering.
        grasp_type_id: Selected grasp type. ``0`` means score-only in
            ``random_objects`` and all real types in ``one_scene``.

    Returns:
        Scene records consumed by shared visualization renderers.
    """
    mode = normalize_prior_visualize_mode(mode)
    num_points = int(get_task_value(config, "num_points", 1024))
    type_ids = enabled_grasp_type_ids(config)
    scene_records = []
    if mode == "random_objects":
        selected = random_object_records(prior_index, int(get_task_value(config, "random_object_count", 25)), batch_index)
        for record in progress_iter(selected, desc="Building score scenes", total=len(selected)):
            scene_data, pc = load_record_scene_payload(record, num_points)
            if int(grasp_type_id) == 0:
                scene_records.append(build_score_scene_record(record, scene_data, pc, config))
                continue
            if mano_layers is None:
                raise ValueError("MANO layers are required for random_objects pose visualization.")
            type_index = grasp_type_index(scene_data, int(grasp_type_id))
            position_key = str(scene_data.get("export_position_key", "index_mcp_pos"))
            if position_key not in scene_data:
                position_key = "index_mcp_pos" if "index_mcp_pos" in scene_data else "wrist_pos"
            sample_count = np.asarray(scene_data[position_key]).shape[1]
            sample_index = random.randrange(int(sample_count))
            scene_records.append(
                build_pose_scene_record(
                    record,
                    scene_data,
                    pc,
                    mano_layers,
                    str(config.device),
                    type_index,
                    sample_index,
                    1,
                    config,
                    show_caption=True,
                )
            )
            scene_records[-1]["viser_grid_row"] = 0
            scene_records[-1]["viser_grid_col"] = 0
            scene_records[-1]["viser_grid_rows"] = 1
            scene_records[-1]["viser_grid_cols"] = 1
        return scene_records

    if mode != "one_scene":
        raise ValueError(f"Unsupported visualize_human_prior mode={mode}.")
    if mano_layers is None:
        raise ValueError("MANO layers are required for one_scene pose visualization.")

    record = one_scene_record(prior_index, scene_id=object_id, random_next=False)
    prior_index["scene_record_by_id"][record["scene_id"]] = record
    scene_data, pc = load_record_scene_payload(record, num_points)
    position_key = str(scene_data.get("export_position_key", "index_mcp_pos"))
    if position_key not in scene_data:
        position_key = "index_mcp_pos" if "index_mcp_pos" in scene_data else "wrist_pos"
    pose_array = np.asarray(scene_data[position_key])
    visible_samples_per_type = min(
        int(get_task_value(config, "one_scene_samples_per_type", get_task_value(config, "one_object_samples_per_type", 5))),
        pose_array.shape[1],
    )
    if visible_samples_per_type <= 0:
        raise ValueError("task.one_scene_samples_per_type must be positive.")
    show_one_scene_caption = bool(get_task_value(config, "show_one_scene_caption", False))
    batch_start = (max(0, int(batch_index)) * visible_samples_per_type) % pose_array.shape[1]
    sample_indices = [
        (batch_start + offset) % pose_array.shape[1]
        for offset in range(min(visible_samples_per_type, pose_array.shape[1]))
    ]
    enabled_pairs = scene_enabled_type_pairs(scene_data, type_ids)
    if int(grasp_type_id) == 0:
        type_pairs = enabled_pairs
    else:
        type_pairs = [(grasp_type_index(scene_data, int(grasp_type_id)), int(grasp_type_id))]
    for display_row, (grasp_type_index_value, _) in enumerate(type_pairs):
        for grid_col, sample_index in enumerate(sample_indices):
            scene_records.append(
                build_pose_scene_record(
                    record,
                    scene_data,
                    pc,
                    mano_layers,
                    str(config.device),
                    int(grasp_type_index_value),
                    sample_index,
                    len(sample_indices),
                    config,
                    show_caption=show_one_scene_caption,
                )
            )
            scene_records[-1]["viser_grid_col"] = grid_col
            scene_records[-1]["viser_grid_row"] = 0 if int(grasp_type_id) != 0 else display_row
            scene_records[-1]["viser_grid_rows"] = 1 if int(grasp_type_id) != 0 else len(type_pairs)
            scene_records[-1]["viser_grid_cols"] = len(sample_indices)
            if scene_records[-1]["caption"]:
                scene_records[-1]["caption"] = f"{scene_records[-1]['caption']} | batch={batch_index + 1}"
    return scene_records


def build_prior_selection_controls(prior_index: dict, config: DictConfig, mano_layers_state: dict) -> dict:
    """Build shared viser selection controls for object human-prior exports.

    Args:
        prior_index: Index returned by ``build_prior_index``.
        config: Full Hydra config.
        mano_layers_state: Mutable holder that creates MANO layers only when
            one-scene pose rendering is first requested.

    Returns:
        Selection-control dictionary consumed by ``show_scenes_with_viser``.
    """
    initial_mode = normalize_prior_visualize_mode(get_task_value(config, "visualize_mode", "random_objects"))
    if initial_mode not in HUMAN_PRIOR_VISER_MODE_OPTIONS:
        initial_mode = "random_objects"
    object_options = prior_index["object_options"] or ("",)
    if initial_mode == "one_scene":
        initial_scene_options = scene_options(prior_index, config) or ("",)
        initial_object = pick_initial_option(
            initial_scene_options,
            get_task_value(config, "scene_id", get_task_value(config, "object_id", None)),
        )
    else:
        initial_object = pick_initial_object_option(object_options, get_task_value(config, "object_id", None))
    type_ids = enabled_grasp_type_ids(config)
    grasp_type_options = tuple(format_grasp_type_option(idx, GRASP_TYPES) for idx in (0, *type_ids))
    preferred_grasp_type_id = get_task_value(config, "target_grasp_type_id", 0)
    if int(preferred_grasp_type_id) == HUMAN_BOTH_THREE_TYPE_ID and HUMAN_BOTH_THREE_TYPE_ID not in type_ids:
        preferred_grasp_type_id = 0
    initial_grasp_type = pick_initial_option(grasp_type_options, preferred_grasp_type_id)
    batch_state = {"key": None, "index": 0}

    def update_one_scene_selection_info(scene_id: str) -> None:
        """Refresh right-panel per-type score text for one selected scene.

        Args:
            scene_id: Scene id currently selected in one-scene mode.

        Returns:
            None. Updates ``batch_state`` in place.
        """
        record = find_scene_record(prior_index, str(scene_id))
        scene_data = load_prior_scene(record["scene_file"])
        batch_state["selection_info"] = one_scene_score_markdown(scene_data, type_ids)

    def load_scene_records(mode, object_id, grasp_type_option, advance_batch=False, split_name=None):
        """Load scene records from the current viser selection.

        Args:
            mode: Selected UI visualization mode.
            object_id: Selected object id.
            grasp_type_option: Selected grasp type. ``0_any`` shows score-only
                records in random-object mode.
            advance_batch: Whether to advance to the next object page or
                one-scene pose sample page.
            split_name: Unused split name kept for the shared control contract.

        Returns:
            Scene records for the requested selection.
        """
        del split_name
        target_grasp_type_id = parse_grasp_type_option(grasp_type_option) if grasp_type_option else 0
        if int(target_grasp_type_id) == HUMAN_BOTH_THREE_TYPE_ID and HUMAN_BOTH_THREE_TYPE_ID not in type_ids:
            target_grasp_type_id = 0
        mode = normalize_prior_visualize_mode(mode)
        selection_key = (mode, str(object_id), int(target_grasp_type_id))
        if selection_key != batch_state["key"]:
            batch_state["key"] = selection_key
            batch_state["index"] = 0
        elif advance_batch:
            batch_state["index"] += 1
        mano_layers = None
        if mode == "one_scene" or int(target_grasp_type_id) != 0:
            if mano_layers_state.get("value") is None:
                print("[visualize_human_prior] Initializing MANO layers for pose view.")
                mano_layers_state["value"] = create_mano_layers(str(config.device))
            mano_layers = mano_layers_state["value"]
        scene_records = build_scene_records(
            prior_index,
            mode,
            str(object_id),
            batch_state["index"],
            config,
            mano_layers=mano_layers,
            grasp_type_id=int(target_grasp_type_id),
        )
        if mode == "one_scene" and scene_records:
            batch_state["current_object"] = str(scene_records[0].get("source_scene_id", object_id))
            update_one_scene_selection_info(batch_state["current_object"])
        elif mode == "random_objects":
            batch_state["current_object"] = str(object_id)
            batch_state["selection_info"] = ""
        return scene_records

    def load_action_scene_records(mode, object_id, grasp_type_option, action_name: str, split_name=None):
        """Load scene records for human-prior-specific Selection actions.

        Args:
            mode: Selected UI visualization mode.
            object_id: Selected object id or scene id.
            grasp_type_option: Selected grasp type option.
            action_name: Button action label. ``Next Scene`` selects another
                random scene while keeping the current grasp type.
            split_name: Unused split name kept for the shared control contract.

        Returns:
            Scene records for the requested action.
        """
        del split_name
        mode = normalize_prior_visualize_mode(mode)
        if mode != "one_scene" or str(action_name) != "Next Scene":
            return load_scene_records(mode, object_id, grasp_type_option, advance_batch=True)

        target_grasp_type_id = parse_grasp_type_option(grasp_type_option) if grasp_type_option else 0
        if int(target_grasp_type_id) == HUMAN_BOTH_THREE_TYPE_ID and HUMAN_BOTH_THREE_TYPE_ID not in type_ids:
            target_grasp_type_id = 0
        random_record = one_scene_record(prior_index, random_next=True)
        batch_state["key"] = (mode, random_record["scene_id"], int(target_grasp_type_id))
        batch_state["index"] = 0
        if mano_layers_state.get("value") is None:
            print("[visualize_human_prior] Initializing MANO layers for pose view.")
            mano_layers_state["value"] = create_mano_layers(str(config.device))
        scene_records = build_scene_records(
            prior_index,
            mode,
            random_record["scene_id"],
            batch_state["index"],
            config,
            mano_layers=mano_layers_state["value"],
            grasp_type_id=int(target_grasp_type_id),
        )
        if scene_records:
            batch_state["current_object"] = str(scene_records[0].get("source_scene_id", random_record["scene_id"]))
            update_one_scene_selection_info(batch_state["current_object"])
        return scene_records

    return {
        "mode_options": HUMAN_PRIOR_VISER_MODE_OPTIONS,
        "initial_mode": initial_mode,
        "object_options": object_options,
        "base_object_options": object_options,
        "initial_object": initial_object,
        "grasp_type_options": grasp_type_options,
        "initial_grasp_type": initial_grasp_type,
        "load_scene_records": load_scene_records,
        "batch_state": batch_state,
        "object_options_for_mode": lambda mode: (
            scene_options(prior_index, config) if normalize_prior_visualize_mode(mode) == "one_scene" else object_options
        ),
        "object_label": "Scene",
        "next_button_label": "Next Batch",
        "extra_action_button_label": "Next Scene",
        "disable_object_for_mode": lambda mode: normalize_prior_visualize_mode(mode) == "random_objects",
        "disable_grasp_type_for_mode": lambda mode: False,
        "disable_next_batch_for_mode": lambda mode: False,
        "disable_extra_action_for_mode": lambda mode: normalize_prior_visualize_mode(mode) != "one_scene",
        "load_action_scene_records": load_action_scene_records,
    }


def task_visualize_human_prior(config: DictConfig) -> None:
    """Visualize exported object human-prior scores and wrist pose samples.

    Args:
        config: Full Hydra config for ``task=visualize_human_prior``.

    Returns:
        None. Starts an interactive viser server or opens a trimesh viewer.
    """
    set_seed(config.seed)
    visualizer = str(get_task_value(config, "visualizer", "viser")).lower()
    if visualizer not in {"viser", "trimesh"}:
        raise ValueError(f"Unsupported visualizer={visualizer}. Expected one of ['viser', 'trimesh'].")
    if visualizer == "viser" and not VISER_AVAILABLE:
        raise ImportError("viser is not installed. Install it before using task=visualize_human_prior.")

    prior_dir = resolve_prior_export_dir(config)
    print(f"[visualize_human_prior] Visualizing object human-prior export: {prior_dir}")
    prior_index = build_prior_index(prior_dir)
    print(f"[visualize_human_prior] Indexed {len(prior_index['object_options'])} object(s) lazily.")
    mano_layers_state = {"value": None}

    viser_port = int(get_task_value(config, "viser_port", 8080))
    viser_scene_spacing = float(get_task_value(config, "viser_scene_spacing", 0.5))
    viser_display_mode = str(get_task_value(config, "viser_display_mode", "all"))
    viser_scene_id = int(get_task_value(config, "viser_scene_id", 0))
    viser_label_font_size_mode = str(get_task_value(config, "viser_label_font_size_mode", "screen"))
    viser_label_font_screen_scale = float(get_task_value(config, "viser_label_font_screen_scale", 0.8))
    viser_label_height_offset = float(get_task_value(config, "viser_label_height_offset", 0.35))
    initial_mode = normalize_prior_visualize_mode(get_task_value(config, "visualize_mode", "random_objects"))
    initial_object = pick_initial_object_option(prior_index["object_options"], get_task_value(config, "object_id", None))

    if visualizer == "viser":
        selection_controls = build_prior_selection_controls(prior_index, config, mano_layers_state)
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
            log_prefix="visualize_human_prior",
            selection_controls=selection_controls,
            label_font_size_mode=viser_label_font_size_mode,
            label_font_screen_scale=viser_label_font_screen_scale,
            label_height_offset=viser_label_height_offset,
        )
        return

    mano_layers = None
    if initial_mode == "one_scene":
        mano_layers = create_mano_layers(str(config.device))
        initial_object = pick_initial_option(
            scene_options(prior_index, config),
            get_task_value(config, "scene_id", get_task_value(config, "object_id", None)),
        )
    scene_records = build_scene_records(
        prior_index,
        initial_mode,
        initial_object,
        0,
        config,
        mano_layers=mano_layers,
    )
    print(f"[visualize_human_prior] Showing {len(scene_records)} scene(s) with trimesh.")
    for idx, record in enumerate(scene_records):
        print(f"[visualize_human_prior] {idx:04d}: {compact_caption(record['caption'])}")
    show_scenes_with_trimesh(scene_records)
