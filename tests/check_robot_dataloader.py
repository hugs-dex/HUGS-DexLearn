import json
import os
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
import trimesh
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import pytorch_kinematics as pk

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dexlearn.dataset import InfLoader, create_dataset, minkowski_collate_fn
from dexlearn.dataset.grasp_types import GRASP_TYPES
from dexlearn.network.models import *  # noqa: F401,F403
from dexlearn.task.visualize import show_scenes_with_viser
from dexlearn.utils.config import flatten_multidex_data_config
from dexlearn.utils.logger import Logger
from dexlearn.utils.util import set_seed
from mr_utils.robot.pk_helper import PytorchKinematicsHelper
from mr_utils.robot.pk_visualizer import Visualizer


def cfg_select(config: DictConfig, key: str, default):
    value = OmegaConf.select(config, key)
    return default if value is None else value


def store_sample(batch_data, sample_idx):
    sample = {}
    for key, value in batch_data.items():
        if torch.is_tensor(value):
            sample[key] = value[sample_idx : sample_idx + 1].clone()
        elif isinstance(value, list):
            sample[key] = [value[sample_idx]]
        else:
            sample[key] = value
    return sample


def merge_samples(samples):
    merged = {}
    for key in samples[0]:
        first_value = samples[0][key]
        if torch.is_tensor(first_value):
            merged[key] = torch.cat([sample[key] for sample in samples], dim=0)
        elif isinstance(first_value, list):
            merged[key] = [sample[key][0] for sample in samples]
        else:
            merged[key] = [sample[key] for sample in samples]
    return merged


def collect_target_grasp_batch(train_loader, target_grasp_type_ids, max_iter):
    """Collect one visualization batch with the requested grasp types.

    Args:
        train_loader: Infinite dataloader wrapper that returns device batches.
        target_grasp_type_ids: Ordered grasp type ids to include in the batch.
        max_iter: Maximum number of dataloader batches to scan.

    Returns:
        Merged batch containing one sample for each requested grasp type.
    """
    pending_samples = {}
    target_grasp_type_ids = list(target_grasp_type_ids)

    for _ in range(max_iter):
        data = train_loader.get()
        batch_size = data["grasp_type_id"].shape[0]

        for i in range(batch_size):
            grasp_type_id = int(data["grasp_type_id"][i])
            if grasp_type_id in target_grasp_type_ids and grasp_type_id not in pending_samples:
                pending_samples[grasp_type_id] = store_sample(data, i)

        if all(grasp_type_id in pending_samples for grasp_type_id in target_grasp_type_ids):
            ordered_samples = [pending_samples[grasp_type_id] for grasp_type_id in target_grasp_type_ids]
            return merge_samples(ordered_samples)

    raise RuntimeError(
        f"Could not collect all target grasp types {target_grasp_type_ids} within {max_iter} iterations."
    )


def build_robot_batch_scene_records(
    data,
    config,
    robot_helper,
    robot_visualizer,
    joint_names,
    wrist_link_names,
    right_arm_indices,
    left_arm_indices,
    right_hand_indices,
    left_hand_indices,
):
    """Build viser scene records from one robot dataloader visualization batch.

    Args:
        data: Merged robot dataloader batch.
        config: Hydra runtime config containing the torch device.
        robot_helper: Kinematics helper with IK solvers for both wrists.
        robot_visualizer: Robot mesh visualizer.
        joint_names: Full robot joint name list.
        wrist_link_names: Right and left wrist link names used for IK.
        right_arm_indices: Joint indices for the right arm.
        left_arm_indices: Joint indices for the left arm.
        right_hand_indices: Joint indices for the right hand.
        left_hand_indices: Joint indices for the left hand.

    Returns:
        List of scene records accepted by ``show_scenes_with_viser``.
    """
    batch_size = int(data["grasp_type_id"].shape[0])
    right_trans = data["right_hand_trans"][:, 0, 1, :]
    right_rot = data["right_hand_rot"][:, 0, 1, ...]
    left_trans = data["left_hand_trans"][:, 0, 1, :]
    left_rot = data["left_hand_rot"][:, 0, 1, ...]
    right_joint = data["right_hand_joint"][:, 0, 1, ...]
    left_joint = data["left_hand_joint"][:, 0, 1, ...]

    if right_joint.shape[-1] != len(right_hand_indices):
        raise ValueError(
            f"Right hand joint dim mismatch: expected {len(right_hand_indices)}, got {right_joint.shape[-1]}"
        )
    if left_joint.shape[-1] != len(left_hand_indices):
        raise ValueError(
            f"Left hand joint dim mismatch: expected {len(left_hand_indices)}, got {left_joint.shape[-1]}"
        )

    right_matrix = torch.eye(4, device=config.device, dtype=right_rot.dtype).unsqueeze(0).repeat(batch_size, 1, 1)
    right_matrix[:, :3, :3] = right_rot
    right_matrix[:, :3, 3] = right_trans

    left_matrix = torch.eye(4, device=config.device, dtype=left_rot.dtype).unsqueeze(0).repeat(batch_size, 1, 1)
    left_matrix[:, :3, :3] = left_rot
    left_matrix[:, :3, 3] = left_trans

    right_sol = robot_helper.solve_ik_batch(wrist_link_names[0], right_matrix)
    left_sol = robot_helper.solve_ik_batch(wrist_link_names[1], left_matrix)

    combined_q = torch.zeros((batch_size, len(joint_names)), device=config.device, dtype=right_sol["q"].dtype)
    combined_q[:, right_arm_indices] = right_sol["q"][:, right_arm_indices]
    combined_q[:, left_arm_indices] = left_sol["q"][:, left_arm_indices]
    combined_q[:, right_hand_indices] = right_joint.to(device=config.device, dtype=combined_q.dtype)
    combined_q[:, left_hand_indices] = left_joint.to(device=config.device, dtype=combined_q.dtype)

    robot_pose = torch.zeros((batch_size, 7 + combined_q.shape[-1]), device=config.device, dtype=torch.float32)
    robot_pose[:, 3] = 1.0
    robot_pose[:, 7:] = combined_q.to(dtype=torch.float32)
    robot_visualizer.set_robot_parameters(robot_pose, joint_names=joint_names)

    scene_records = []
    for i in range(batch_size):
        grasp_type_id = int(data["grasp_type_id"][i])
        grasp_type_name = GRASP_TYPES[grasp_type_id]
        caption = (
            f"{i} | path: {data['path'][i]} | grasp_type: {grasp_type_name} | "
            f"Right IK: {int(right_sol['success'][i])} | Left IK: {int(left_sol['success'][i])}"
        )
        print(caption)

        scene_elements = [
            robot_visualizer.get_robot_trimesh_data(i=i, color=(180, 180, 220, 255)),
            trimesh.points.PointCloud(data["point_clouds"][i, ...].cpu().numpy(), colors=[255, 165, 0, 255]),
            trimesh.creation.axis(origin_size=0.01, axis_radius=0.001, axis_length=0.3),
        ]
        scene_records.append({"elements": scene_elements, "caption": caption, "smooth": False})

    return scene_records


@hydra.main(config_path="../dexlearn/config", config_name="base", version_base=None)
def main(config: DictConfig) -> None:
    set_seed(config.seed)
    Logger(config)
    flatten_multidex_data_config(config.data)

    target_grasp_type_ids = [1, 2, 3, 4, 5]
    train_dataset = create_dataset(config, mode="train")

    config.algo.batch_size = int(cfg_select(config, "check_batch_size", 4))
    train_loader = InfLoader(
        DataLoader(
            train_dataset,
            batch_size=config.algo.batch_size,
            drop_last=True,
            num_workers=0,
            shuffle=False,
            collate_fn=minkowski_collate_fn,
        ),
        config.device,
    )

    urdf_path = config.data.robot_urdf_path
    mesh_dir_path = config.data.robot_mesh_dir_path
    chain = pk.build_chain_from_urdf(open(urdf_path).read()).to(device=config.device)
    robot_helper = PytorchKinematicsHelper(
        chain,
        base_pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        device=config.device,
    )
    robot_visualizer = Visualizer(urdf_path, mesh_dir_path, device=config.device)

    metadata = os.path.join(config.data.grasp_path, f"{config.data.metadata_group}/metadata.json")
    if not os.path.exists(metadata):
        raise FileNotFoundError(f"Joint metadata file not found: {metadata}")
    with open(metadata, "r") as f:
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

    max_iter = int(cfg_select(config, "check_max_iter", config.algo.max_iter))

    def load_next_batch_scene_records():
        """Load and convert the next robot visualization batch.

        Args:
            None.

        Returns:
            List of scene records for the next representative robot batch.
        """
        data = collect_target_grasp_batch(train_loader, target_grasp_type_ids, max_iter)
        return build_robot_batch_scene_records(
            data=data,
            config=config,
            robot_helper=robot_helper,
            robot_visualizer=robot_visualizer,
            joint_names=joint_names,
            wrist_link_names=wrist_link_names,
            right_arm_indices=right_arm_indices,
            left_arm_indices=left_arm_indices,
            right_hand_indices=right_hand_indices,
            left_hand_indices=left_hand_indices,
        )

    scene_records = load_next_batch_scene_records()
    show_scenes_with_viser(
        scene_records,
        port=int(cfg_select(config, "viser_port", 8080)),
        scene_spacing=float(cfg_select(config, "viser_scene_spacing", 1.0)),
        display_mode=str(cfg_select(config, "viser_display_mode", "all")),
        scene_id=int(cfg_select(config, "viser_scene_id", 0)),
        log_prefix="check_robot_dataloader",
        next_batch_loader=load_next_batch_scene_records,
    )


if __name__ == "__main__":
    main()
