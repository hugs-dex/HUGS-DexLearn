"""Reverse T-to-C and reusable marginal wrist-pose Human Prior models."""

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from einops import repeat

from ..backbones import *  # noqa: F401,F403
from ..final_layers import *  # noqa: F401,F403
from ..final_layers.diffusion import (
    bimanual_t24_from_data,
    canonicalize_bimanual_t24,
)
from dexlearn.dataset.grasp_types import GRASP_TYPES
from dexlearn.utils.RMS import Normalization


REAL_GRASP_TYPE_NUM = len(GRASP_TYPES) - 1


def _mlp(dimensions: Iterable[int]) -> torch.nn.Sequential:
    """Build a ReLU MLP from an explicit dimension sequence.

    Args:
        dimensions: Input, hidden, and output dimensions.

    Returns:
        Sequential MLP with ReLU between all non-final linear layers.
    """
    dims = [int(value) for value in dimensions]
    if len(dims) < 2:
        raise ValueError("MLP dimensions must contain at least input and output sizes")
    layers = []
    for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(torch.nn.Linear(in_dim, out_dim))
        if index < len(dims) - 2:
            layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)


class MarginalPoseDiffusionModel(torch.nn.Module):
    """Object-conditioned marginal wrist-pose diffusion ``p(T|o)``."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.backbone = eval(cfg.backbone.name)(cfg.backbone)
        cfg.head.in_feat_dim = cfg.backbone.out_feat_dim
        self.output_head = eval(cfg.head.name)(cfg.head)
        if not hasattr(self.output_head, "sample_with_t24"):
            raise TypeError("MarginalPoseDiffusionModel requires an output head with sample_with_t24")

    def encode_object(self, data: dict) -> torch.Tensor:
        """Encode only the object observation.

        Args:
            data: Batch containing sparse point-cloud inputs. Grasp type fields are
                intentionally ignored.

        Returns:
            Global object feature tensor.
        """
        global_feature, _ = self.backbone(data)
        return global_feature

    def forward(self, data: dict) -> dict:
        global_feature = self.encode_object(data)
        sample_num = int(data["right_hand_trans"].shape[1])
        global_feature = repeat(global_feature, "b c -> (b s) c", s=sample_num)
        return self.output_head.forward(data, global_feature)

    def _initial_noise(
        self,
        global_feature: torch.Tensor,
        sample_num: int,
        seed: int | Iterable[int],
    ) -> torch.Tensor:
        """Build per-scene diffusion noise without using the global RNG state."""
        if isinstance(seed, int):
            scene_seeds = [int(seed)] * global_feature.shape[0]
        else:
            scene_seeds = [int(value) for value in seed]
            if len(scene_seeds) != global_feature.shape[0]:
                raise ValueError(
                    f"Expected {global_feature.shape[0]} pose seeds, got {len(scene_seeds)}"
                )
        pose_dim = int(self.output_head.policy.channels)
        rows = []
        for scene_seed in scene_seeds:
            generator = torch.Generator(device=global_feature.device)
            generator.manual_seed(scene_seed)
            rows.append(
                torch.randn(
                    (sample_num, pose_dim),
                    generator=generator,
                    device=global_feature.device,
                    dtype=global_feature.dtype,
                )
            )
        return torch.stack(rows, dim=0).reshape(-1, pose_dim)

    def sample_with_t24(
        self,
        data: dict,
        sample_num: int,
        seed: int | Iterable[int] | None = None,
    ):
        """Sample T before any contact-mode prediction.

        Args:
            data: Object observation batch.
            sample_num: Number of marginal poses per object.
            seed: Optional scalar or one explicit diffusion seed per scene.

        Returns:
            Tuple ``(canonical_t24, robot_pose, log_prob)``.
        """
        global_feature = self.encode_object(data)
        initial_noise = None if seed is None else self._initial_noise(global_feature, sample_num, seed)
        if initial_noise is None:
            return self.output_head.sample_with_t24(global_feature, sample_num)
        return self.output_head.sample_with_t24(global_feature, sample_num, initial_noise=initial_noise)

    def sample(
        self,
        data: dict,
        sample_num: int = 1,
        seed: int | Iterable[int] | None = None,
    ):
        _, robot_pose, log_prob = self.sample_with_t24(data, sample_num, seed=seed)
        return robot_pose, log_prob


class PoseConditionedTypeModel(torch.nn.Module):
    """Independent pose-afterwards posterior ``q(c|T,o)``."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.backbone = eval(cfg.backbone.name)(cfg.backbone)

        pose_cfg = cfg.pose_encoder
        pose_hidden_dims = list(getattr(pose_cfg, "hidden_dims", [64, 128]))
        if not pose_hidden_dims:
            raise ValueError("pose_encoder.hidden_dims must not be empty")
        pose_feature_dim = int(pose_hidden_dims[-1])
        normalization_cfg = getattr(cfg, "pose_normalization", None)
        max_update = int(getattr(normalization_cfg, "max_update", 2000))
        self.pose_normalization = Normalization(24, max_update=max_update)
        self.pose_encoder = _mlp([24, *pose_hidden_dims])

        object_feature_dim = int(getattr(cfg.object_projection, "out_feat_dim", pose_feature_dim))
        self.object_projection = _mlp([cfg.backbone.out_feat_dim, object_feature_dim])
        fusion_hidden_dim = int(getattr(cfg.fusion_classifier, "hidden_dim", 128))
        self.type_classifier = _mlp(
            [pose_feature_dim + object_feature_dim, fusion_hidden_dim, REAL_GRASP_TYPE_NUM]
        )

    def encode_object(self, data: dict) -> torch.Tensor:
        """Encode the object with the posterior-owned backbone."""
        global_feature, _ = self.backbone(data)
        return global_feature

    def canonical_t24_from_data(self, data: dict) -> torch.Tensor:
        """Build the GT pose input using the same canonicalization as inference."""
        batch_size, sample_num = data["right_hand_trans"].shape[:2]
        canonical_t24 = canonicalize_bimanual_t24(bimanual_t24_from_data(data))
        return canonical_t24.reshape(batch_size, sample_num, 24)

    def logits_from_t24(
        self,
        data: dict,
        canonical_t24: torch.Tensor,
        global_feature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict mode logits after a canonical pose is available.

        Args:
            data: Object observation batch.
            canonical_t24: Pose tensor shaped ``(B, 24)`` or ``(B, S, 24)``.
            global_feature: Optional posterior-backbone feature for reuse.

        Returns:
            Logits shaped ``(B, 5)`` or ``(B, S, 5)``.
        """
        if canonical_t24.ndim not in {2, 3}:
            raise ValueError(f"Expected T24 with shape (B,24) or (B,S,24), got {tuple(canonical_t24.shape)}")
        if canonical_t24.shape[-1] != 24:
            raise ValueError(f"Expected canonical T24 with final dimension 24, got {tuple(canonical_t24.shape)}")
        if global_feature is None:
            global_feature = self.encode_object(data)
        if global_feature.shape[0] != canonical_t24.shape[0]:
            raise ValueError("Object feature and T24 batch dimensions must match")

        original_shape = canonical_t24.shape[:-1]
        normalized_pose = self.pose_normalization(canonical_t24.reshape(-1, 24))
        pose_feature = self.pose_encoder(normalized_pose)
        object_feature = self.object_projection(global_feature)
        if canonical_t24.ndim == 3:
            object_feature = object_feature[:, None, :].expand(-1, canonical_t24.shape[1], -1).reshape(
                -1, object_feature.shape[-1]
            )
        fused = torch.cat([object_feature, pose_feature], dim=-1)
        return self.type_classifier(fused).reshape(*original_shape, REAL_GRASP_TYPE_NUM)

    def posterior_probabilities(
        self,
        data: dict,
        canonical_t24: torch.Tensor,
        global_feature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the categorical posterior after T has been generated."""
        return F.softmax(self.logits_from_t24(data, canonical_t24, global_feature), dim=-1)

    def forward(self, data: dict) -> dict:
        canonical_t24 = self.canonical_t24_from_data(data)
        logits = self.logits_from_t24(data, canonical_t24)
        gt_type = data["grasp_type_id"].long()
        if ((gt_type < 1) | (gt_type > REAL_GRASP_TYPE_NUM)).any():
            raise ValueError("PoseConditionedTypeModel requires grasp_type_id in [1, 5]")
        sample_num = canonical_t24.shape[1]
        gt_type = repeat(gt_type, "b -> (b s)", s=sample_num)
        loss_type = F.cross_entropy(logits.reshape(-1, REAL_GRASP_TYPE_NUM), gt_type - 1)
        return {"loss_type": loss_type}
