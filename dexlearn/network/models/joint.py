"""Joint contact-mode and wrist-pose Human Prior model."""

from collections.abc import Iterable

import torch

from ..backbones import *  # noqa: F401,F403
from ..final_layers import *  # noqa: F401,F403


class JointHybridDiffusionModel(torch.nn.Module):
    """Object-conditioned coupled categorical/continuous diffusion model."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.backbone = eval(cfg.backbone.name)(cfg.backbone)
        cfg.head.in_feat_dim = int(cfg.backbone.out_feat_dim)
        self.output_head = eval(cfg.head.name)(cfg.head)
        if not hasattr(self.output_head, "sample_joint"):
            raise TypeError("JointHybridDiffusionModel requires a head with sample_joint")

    def encode_object(self, data: dict) -> torch.Tensor:
        """Encode the object observation without reading grasp-type placeholders."""
        global_feature, _ = self.backbone(data)
        return global_feature

    def forward(self, data: dict) -> dict:
        """Compute coupled pose-v and categorical clean-label losses."""
        return self.output_head(data, self.encode_object(data))

    def sample_joint(
        self,
        data: dict,
        pool_size: int,
        seed: int | Iterable[int],
        return_trajectory: bool = False,
    ) -> dict:
        """Generate one raw joint pool after a single object-backbone pass."""
        object_feature = self.encode_object(data)
        return self.output_head.sample_joint(
            object_feature,
            pool_size,
            seed,
            return_trajectory=return_trajectory,
        )

    def sample(self, data: dict, sample_num: int = 1):
        """Expose a compact legacy-style tuple for bounded diagnostics."""
        result = self.sample_joint(data, sample_num, seed=0)
        return result["robot_pose"], result["type_ids"], result["joint_path_score"]
