import os

import torch


MANO_INDEX_MCP_IDX = 5
HAND_SIDES = ("right", "left")
VALID_HAND_POS_SOURCES = {"wrist", "index_mcp"}


class ManoConfig:
    def __init__(self, dataset_name):
        if dataset_name == "ContactPose":
            use_pca = True
            flat_hand_mean = False
            ncomps = 15
        elif dataset_name == "HOGraspNet":
            use_pca = False
            flat_hand_mean = True
            ncomps = 45
        elif dataset_name == "GRAB":
            use_pca = True
            flat_hand_mean = True
            ncomps = 24
        elif dataset_name == "OurHumanGraspFormat":
            use_pca = False
            flat_hand_mean = True
            ncomps = 45
        else:
            raise NotImplementedError(f"Unsupported human dataset for MANO config: {dataset_name}")

        self.use_pca = use_pca
        self.flat_hand_mean = flat_hand_mean
        self.ncomps = ncomps


def infer_dataset_name_from_grasp_path(grasp_path: str) -> str:
    grasp_path = os.path.normpath(str(grasp_path))
    path_parts = grasp_path.split(os.sep)
    if "grasp" not in path_parts:
        raise ValueError(f"Invalid grasp_path (missing 'grasp' folder): {grasp_path}")
    grasp_idx = path_parts.index("grasp")
    if grasp_idx == 0:
        raise ValueError(f"Invalid grasp_path (cannot infer dataset name): {grasp_path}")
    return path_parts[grasp_idx - 1]


def normalize_hand_pos_source(value, default: str = "wrist") -> str:
    hand_pos_source = str(value if value is not None else default).lower()
    if hand_pos_source not in VALID_HAND_POS_SOURCES:
        raise ValueError(
            f"Unsupported hand_pos_source={hand_pos_source}. "
            f"Expected one of {sorted(VALID_HAND_POS_SOURCES)}."
        )
    return hand_pos_source


def get_wrist_translation_from_target(target_pos: torch.Tensor, joints_local: torch.Tensor, hand_pos_source: str):
    hand_pos_source = normalize_hand_pos_source(hand_pos_source)
    if hand_pos_source == "wrist":
        return target_pos
    return target_pos - joints_local[MANO_INDEX_MCP_IDX] / 1000.0
