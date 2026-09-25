from __future__ import annotations

import os
import sys
from glob import glob

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dexlearn.utils.util import set_seed
from dexlearn.utils.human_hand import HAND_SIDES, ManoConfig, infer_dataset_name_from_grasp_path
from dexlearn.utils.config import flatten_multidex_data_config

def build_mano_layers(config: DictConfig):
    from dexlearn.utils.dependencies import require_module
    ManoLayer = require_module("manopth.manolayer", "human preprocessing").ManoLayer
    dataset_name = infer_dataset_name_from_grasp_path(config.data.grasp_path)
    mano_cfg = ManoConfig(dataset_name)
    mano_layers = {}
    for side in HAND_SIDES:
        mano_layers[side] = ManoLayer(
            center_idx=0,
            mano_root=config.task.mano_root,
            side=side,
            use_pca=mano_cfg.use_pca,
            flat_hand_mean=mano_cfg.flat_hand_mean,
            ncomps=mano_cfg.ncomps,
            root_rot_mode="axisang",
            joint_rot_mode="axisang",
        ).to(config.device)
    return mano_layers


def get_grasp_files(grasp_root: str, include_patterns):
    grasp_files = []
    for pattern in include_patterns:
        grasp_files.extend(sorted(glob(os.path.join(grasp_root, pattern), recursive=True)))
    return sorted(set(grasp_files))


def compute_index_mcp_pos(hand_data: dict, mano_layer: ManoLayer, device: str):
    hand_rot = torch.as_tensor(hand_data["rot"], device=device, dtype=torch.float32).reshape(1, 3)
    mano_pose = torch.as_tensor(hand_data["mano_pose"], device=device, dtype=torch.float32).reshape(1, -1)
    mano_betas = torch.as_tensor(hand_data["mano_betas"], device=device, dtype=torch.float32).reshape(1, -1)
    hand_trans = torch.as_tensor(hand_data["trans"], device=device, dtype=torch.float32).reshape(1, 3)

    mano_params = torch.cat([hand_rot, mano_pose], dim=-1)
    with torch.no_grad():
        _, joints = mano_layer(mano_params, th_betas=mano_betas)
    index_mcp_pos = joints[0, 5] / 1000.0 + hand_trans[0]
    return index_mcp_pos.cpu().numpy().astype(np.float32)


def task_human_preprocess(config: DictConfig) -> None:
    set_seed(config.seed)
    flatten_multidex_data_config(config.data)

    grasp_root = os.path.normpath(str(config.data.grasp_path))
    if not os.path.isdir(grasp_root):
        raise FileNotFoundError(f"Human grasp root not found: {grasp_root}")

    include_patterns = list(getattr(config.task, "include_patterns", ["**/*.npy"]))
    grasp_files = get_grasp_files(grasp_root, include_patterns)
    if len(grasp_files) == 0:
        raise RuntimeError(f"No grasp files found under {grasp_root} with patterns {include_patterns}")

    if getattr(config.task, "max_files", None):
        grasp_files = grasp_files[: int(config.task.max_files)]

    overwrite = bool(getattr(config.task, "overwrite", False))
    dry_run = bool(getattr(config.task, "dry_run", False))
    mano_layers = build_mano_layers(config)

    updated_num = 0
    skipped_num = 0
    for grasp_path in tqdm(grasp_files, desc="Preprocessing human grasps"):
        grasp_data = np.load(grasp_path, allow_pickle=True).item()
        changed = False

        for side in HAND_SIDES:
            hand_data = grasp_data.get("hand", {}).get(side)
            if not hand_data:
                continue
            if "index_mcp_pos" in hand_data and not overwrite:
                continue
            hand_data["index_mcp_pos"] = compute_index_mcp_pos(hand_data, mano_layers[side], config.device)
            changed = True

        if not changed:
            skipped_num += 1
            continue

        updated_num += 1
        if not dry_run:
            np.save(grasp_path, grasp_data)

    action = "Would update" if dry_run else "Updated"
    print(f"{action} {updated_num} grasp files. Skipped {skipped_num} unchanged files.")


if __name__ == "__main__":
    main = hydra.main(config_path="../config", config_name="base", version_base=None)(task_human_preprocess)
    main()
