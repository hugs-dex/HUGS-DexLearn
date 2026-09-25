import os
import sys
from glob import glob

import numpy as np
import torch
from omegaconf import DictConfig
from pytorch3d.transforms import matrix_to_axis_angle, quaternion_to_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dexlearn.utils.logger import Logger
from dexlearn.utils.human_hand import (
    MANO_INDEX_MCP_IDX,
    HAND_SIDES,
    get_wrist_translation_from_target,
    normalize_hand_pos_source,
)
from dexlearn.utils.util import set_seed

def get_input_dir(config: DictConfig) -> str:
    return os.path.join(
        config.ckpt.replace("ckpts", "tests").replace(".pth", ""),
        config.test_data.name,
    )


def get_output_dir(input_dir: str, output_suffix: str) -> str:
    if not output_suffix:
        return input_dir
    return f"{input_dir}_{output_suffix}"


def get_group_dirs(root_dir: str, include_groups):
    if include_groups is not None:
        return include_groups
    return sorted([name for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))])


def get_sample_files(root_dir: str, include_groups):
    sample_files = []
    for group in include_groups:
        group_path = os.path.join(root_dir, group)
        if not os.path.isdir(group_path):
            continue
        sample_files.extend(sorted(glob(os.path.join(group_path, "**/*.npy"), recursive=True)))
    return sample_files


def build_mano_layers(config: DictConfig):
    from dexlearn.utils.dependencies import require_module
    ManoLayer = require_module("manopth.manolayer", "human prior formatting").ManoLayer
    mano_layers = {}
    for side in HAND_SIDES:
        mano_layers[side] = ManoLayer(
            center_idx=0,
            mano_root=config.task.mano_root,
            side=side,
            use_pca=False,
            flat_hand_mean=True,
            ncomps=45,
            root_rot_mode="axisang",
            joint_rot_mode="axisang",
        ).to(config.device)
    return mano_layers


def split_grasp_pose(grasp_pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(grasp_pose, dtype=np.float32).reshape(-1)
    if pose.size == 0 or pose.size % 7 != 0:
        raise ValueError(f"Expected grasp_pose to be non-empty and divisible by 7, got shape {grasp_pose.shape}")

    hand_num = pose.size // 7
    if hand_num > len(HAND_SIDES):
        raise ValueError(f"Unsupported hand count {hand_num} in grasp_pose with shape {grasp_pose.shape}")
    return pose.reshape(hand_num, 7)


def infer_index_mcp_positions(hand_poses: np.ndarray, mano_layers, device: str) -> np.ndarray:
    return infer_positions_from_pose(hand_poses, mano_layers, device, "wrist")[1]


def infer_positions_from_pose(
    hand_poses: np.ndarray, mano_layers, device: str, grasp_pos_source: str
) -> tuple[np.ndarray, np.ndarray]:
    wrist_pos = np.zeros((hand_poses.shape[0], 3), dtype=np.float32)
    index_mcp_pos = np.zeros((hand_poses.shape[0], 3), dtype=np.float32)
    grasp_pos_source = normalize_hand_pos_source(grasp_pos_source)

    with torch.no_grad():
        for hand_idx, side in enumerate(HAND_SIDES[: hand_poses.shape[0]]):
            hand_pose = hand_poses[hand_idx]
            if np.allclose(hand_pose, 0.0, atol=1e-8):
                continue

            target_pos = torch.from_numpy(hand_pose[:3]).to(device=device, dtype=torch.float32)
            wrist_quat = torch.from_numpy(hand_pose[3:7]).to(device=device, dtype=torch.float32)
            wrist_quat_norm = torch.linalg.norm(wrist_quat)
            if wrist_quat_norm.item() < 1e-8:
                raise ValueError(f"Encountered near-zero wrist quaternion for {side} hand pose: {hand_pose.tolist()}")
            wrist_quat = (wrist_quat / wrist_quat_norm).unsqueeze(0)

            hand_rot_mat = quaternion_to_matrix(wrist_quat)
            mano_params = torch.cat(
                [matrix_to_axis_angle(hand_rot_mat), torch.zeros((1, 45), device=device, dtype=torch.float32)],
                dim=-1,
            )
            _, joints = mano_layers[side](mano_params, th_betas=torch.zeros((1, 10), device=device, dtype=torch.float32))
            wrist_trans = get_wrist_translation_from_target(target_pos, joints[0], grasp_pos_source)
            wrist_pos[hand_idx] = wrist_trans.cpu().numpy()
            index_mcp_pos[hand_idx] = (joints[0, MANO_INDEX_MCP_IDX] / 1000.0 + wrist_trans).cpu().numpy()

    return wrist_pos, index_mcp_pos


def task_human_prior_format(config: DictConfig) -> None:
    set_seed(config.seed)
    config.wandb.mode = "disabled"
    logger = Logger(config)

    input_dir = get_input_dir(config)
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input sample directory not found: {input_dir}")

    include_groups = get_group_dirs(input_dir, config.task.include_groups)
    sample_files = get_sample_files(input_dir, include_groups)
    if len(sample_files) == 0:
        raise RuntimeError(f"No human sample files found in {input_dir}")

    output_dir = get_output_dir(input_dir, config.task.output_suffix)
    mano_layers = build_mano_layers(config)

    print(f"Formatting {len(sample_files)} human sample files from {input_dir}")
    print(f"Saving formatted files to {output_dir}")

    for sample_file in sample_files:
        data = np.load(sample_file, allow_pickle=True).item()
        if "grasp_pose" not in data:
            raise KeyError(f"Expected 'grasp_pose' in sample file: {sample_file}")

        hand_poses = split_grasp_pose(data["grasp_pose"])
        grasp_pos_source = normalize_hand_pos_source(data.get("grasp_pos_source", "wrist"))
        wrist_pos, index_mcp_pos = infer_positions_from_pose(hand_poses, mano_layers, config.device, grasp_pos_source)
        formatted_data = dict(data)
        formatted_data["wrist_pos"] = wrist_pos
        formatted_data["wrist_quat"] = hand_poses[:, 3:7].astype(np.float32)
        formatted_data["index_mcp_pos"] = index_mcp_pos

        relative_path = os.path.relpath(sample_file, input_dir)
        save_path = os.path.join(output_dir, relative_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, formatted_data)

    print(f"Finished formatting {len(sample_files)} files.")
