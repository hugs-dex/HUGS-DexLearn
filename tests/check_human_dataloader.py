import os
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
import trimesh
from omegaconf import DictConfig, OmegaConf
from pytorch3d.transforms import matrix_to_axis_angle
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dexlearn.dataset import InfLoader, create_dataset, minkowski_collate_fn
from dexlearn.dataset.grasp_types import GRASP_TYPES
from dexlearn.network.models import *  # noqa: F401,F403
from dexlearn.task.visualize import show_scenes_with_viser
from dexlearn.utils.config import flatten_multidex_data_config
from dexlearn.utils.human_hand import (
    ManoConfig,
    get_wrist_translation_from_target,
    infer_dataset_name_from_grasp_path,
    normalize_hand_pos_source,
)
from dexlearn.utils.logger import Logger
from dexlearn.utils.util import set_seed
from manopth.manolayer import ManoLayer


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


def cfg_select(config: DictConfig, key: str, default):
    value = OmegaConf.select(config, key)
    return default if value is None else value


def build_hand_mesh_elements(verts, faces, joints=None, color=(200, 200, 250, 255)):
    hand_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    hand_mesh.visual.face_colors = color
    scene_elements = [hand_mesh]

    if joints is not None:
        for joint in joints:
            sphere = trimesh.creation.uv_sphere(radius=0.005)
            sphere.visual.face_colors = [255, 0, 0, 255]
            sphere.apply_translation(joint)
            scene_elements.append(sphere)

    return scene_elements


def target_fingertip_indices(grasp_type_id, side):
    """Return fingertip joint indices that should be highlighted.

    Args:
        grasp_type_id: Integer grasp type id in ``[1, 5]``.
        side: Hand side, either ``right`` or ``left``.

    Returns:
        List of MANO joint indices for fingertips involved by the grasp-type
        label.
    """
    finger_names = GRASP_TYPE_FINGERS.get(int(grasp_type_id), {}).get(side, [])
    return [FINGERTIP_JOINT_INDICES[name] for name in finger_names]


def build_human_batch_scene_records(data, mano_layers, mano_cfg, hand_pos_source, hand_colors, device):
    """Build viser scene records from one human dataloader batch.

    Args:
        data: Collated human dataloader batch moved to the target device.
        mano_layers: Mapping from hand side to MANO layer.
        mano_cfg: MANO configuration inferred from the dataset format.
        hand_pos_source: Name of the hand target position convention.
        hand_colors: Mapping from hand side to RGBA mesh color.
        device: Torch device used for temporary MANO tensors.

    Returns:
        List of scene records accepted by ``show_scenes_with_viser``.
    """
    batch_size = int(data["grasp_type_id"].shape[0])
    scene_records = []

    for i in range(batch_size):
        grasp_type_id = int(data["grasp_type_id"][i])
        grasp_type_name = GRASP_TYPES[grasp_type_id]
        caption = f"{i} | path: {data['path'][i]} | grasp_type: {grasp_type_name}"
        scene_elements = []

        for side in ["right", "left"]:
            hand_target_pos = data[f"{side}_hand_trans"][i, 0, 0, :]
            hand_rot_mat = data[f"{side}_hand_rot"][i, 0, 0, ...]

            if side == "left" and "left_hand_fixed" in data and bool(data["left_hand_fixed"][i]):
                continue
            if not hand_rot_mat.any():
                continue

            hand_rot_aa = matrix_to_axis_angle(hand_rot_mat.unsqueeze(0))
            if f"{side}_mano_pose" in data:
                mano_pose = data[f"{side}_mano_pose"][i].unsqueeze(0).to(device)
                mano_betas = data[f"{side}_mano_betas"][i].unsqueeze(0).to(device)
            else:
                mano_pose = torch.zeros((1, mano_cfg.ncomps), device=device)
                mano_betas = torch.zeros((1, 10), device=device)
            mano_params = torch.cat([hand_rot_aa, mano_pose], dim=-1)

            verts, joints = mano_layers[side](mano_params, th_betas=mano_betas)
            wrist_trans = get_wrist_translation_from_target(hand_target_pos, joints[0], hand_pos_source)
            verts = (verts / 1000.0) + wrist_trans.unsqueeze(0)
            joints = (joints / 1000.0) + wrist_trans.unsqueeze(0)

            hand_pose = torch.eye(4, device=hand_rot_mat.device, dtype=hand_rot_mat.dtype)
            hand_pose[:3, :3] = hand_rot_mat
            hand_pose[:3, 3] = wrist_trans
            scene_elements.append(
                trimesh.creation.axis(
                    transform=hand_pose.cpu().numpy(),
                    origin_size=0.01,
                    axis_radius=0.005,
                    axis_length=0.1,
                )
            )

            v_np = verts[0].cpu().detach().numpy()
            j_np = joints[0].cpu().detach().numpy()
            f_np = mano_layers[side].th_faces.cpu().numpy()
            fingertip_indices = target_fingertip_indices(grasp_type_id, side)
            fingertip_joints = j_np[fingertip_indices] if fingertip_indices else None
            scene_elements.extend(build_hand_mesh_elements(v_np, f_np, fingertip_joints, color=hand_colors[side]))

            if hand_pos_source == "index_mcp":
                mcp_pose = torch.eye(4, device=hand_rot_mat.device, dtype=hand_rot_mat.dtype)
                mcp_pose[:3, :3] = hand_rot_mat
                mcp_pose[:3, 3] = hand_target_pos
                scene_elements.append(
                    trimesh.creation.axis(
                        transform=mcp_pose.cpu().numpy(),
                        origin_size=0.008,
                        axis_radius=0.003,
                        axis_length=0.08,
                    )
                )

        pc_np = data["point_clouds"][i, ...].cpu().numpy()
        scene_elements.append(trimesh.points.PointCloud(pc_np, colors=[255, 165, 0, 255]))
        scene_elements.append(trimesh.creation.axis(origin_size=0.01, axis_radius=0.001, axis_length=0.3))
        scene_records.append({"elements": scene_elements, "caption": caption})

    return scene_records


@hydra.main(config_path="../dexlearn/config", config_name="base", version_base=None)
def main(config: DictConfig) -> None:
    set_seed(config.seed)
    Logger(config)
    flatten_multidex_data_config(config.data)

    grasp_path = os.path.normpath(str(config.data.grasp_path))
    dataset_name = infer_dataset_name_from_grasp_path(grasp_path)
    mano_cfg = ManoConfig(dataset_name)
    hand_pos_source = normalize_hand_pos_source(getattr(config.data, "hand_pos_source", "wrist"))

    mano_layers = {
        side: ManoLayer(
            center_idx=0,
            mano_root=os.environ.get("MANO_ROOT", "assets/mano"),
            side=side,
            use_pca=mano_cfg.use_pca,
            flat_hand_mean=mano_cfg.flat_hand_mean,
            ncomps=mano_cfg.ncomps,
            root_rot_mode="axisang",
            joint_rot_mode="axisang",
        ).to(config.device)
        for side in ["right", "left"]
    }

    hand_colors = {
        "right": (200, 200, 250, 255),
        "left": (250, 200, 200, 255),
    }

    train_dataset = create_dataset(config, mode="train")
    train_dataset.config.load_mano_params = True

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

    def load_next_batch_scene_records():
        """Load and convert the next human dataloader batch.

        Args:
            None.

        Returns:
            List of scene records for the next batch.
        """
        data = train_loader.get()
        return build_human_batch_scene_records(
            data=data,
            mano_layers=mano_layers,
            mano_cfg=mano_cfg,
            hand_pos_source=hand_pos_source,
            hand_colors=hand_colors,
            device=config.device,
        )

    scene_records = load_next_batch_scene_records()

    show_scenes_with_viser(
        scene_records,
        port=int(cfg_select(config, "viser_port", 8080)),
        scene_spacing=float(cfg_select(config, "viser_scene_spacing", 0.8)),
        display_mode=str(cfg_select(config, "viser_display_mode", "all")),
        scene_id=int(cfg_select(config, "viser_scene_id", 0)),
        log_prefix="check_human_dataloader",
        next_batch_loader=load_next_batch_scene_records,
    )


if __name__ == "__main__":
    main()
