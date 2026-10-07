"""Hierarchical robot grasp model."""

import torch
import torch.nn.functional as F
from einops import repeat, rearrange

from ..backbones import *
from ..type_emb import *
from ..final_layers import *
from dexlearn.dataset.grasp_types import GRASP_TYPES
from .hierarchical import _build_type_head, REAL_GRASP_TYPE_NUM


class RobotHierarchicalModel(torch.nn.Module):
    """Robot pose diffusion with per-type availability prediction."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.type_objective = str(getattr(cfg, "type_objective", "availability")).lower()
        if self.type_objective != "availability":
            raise ValueError("RobotHierarchicalModel only supports type_objective=availability")
        availability_cfg = getattr(cfg, "type_availability", None)
        self.availability_score_threshold = (
            float(getattr(availability_cfg, "score_threshold", 0.5)) if availability_cfg is not None else 0.5
        )
        self.availability_min_available_types = (
            int(getattr(availability_cfg, "min_available_types", 1)) if availability_cfg is not None else 1
        )
        self.availability_use_score_prior = (
            bool(getattr(availability_cfg, "use_score_prior", True)) if availability_cfg is not None else True
        )
        if not 0.0 <= self.availability_score_threshold <= 1.0:
            raise ValueError("type_availability.score_threshold must be in [0, 1]")
        if self.availability_min_available_types < 1:
            raise ValueError("type_availability.min_available_types must be >= 1")
        self.backbone = eval(cfg.backbone.name)(cfg.backbone)

        # Keep the unused 0_any logit to preserve existing robot checkpoints.
        self.type_classifier = _build_type_head(cfg.backbone.out_feat_dim, len(GRASP_TYPES))

        # Grasp type embedding for diffusion conditioning
        self.grasp_type_emb = eval(cfg.grasp_type_emb.name)(cfg.grasp_type_emb)

        # Diffusion head
        cfg.head.in_feat_dim = cfg.backbone.out_feat_dim + cfg.grasp_type_emb.out_feat_dim
        self.output_head = eval(cfg.head.name)(cfg.head)

    def _availability_scores(self, type_logits: torch.Tensor) -> torch.Tensor:
        """Convert real-type logits into independent availability scores.

        Args:
            type_logits: Type head logits with shape ``(B, len(GRASP_TYPES))``.

        Returns:
            Sigmoid scores with shape ``(B, 5)`` for real grasp types ``1..5``.
        """
        return torch.sigmoid(type_logits[:, 1:])

    def _compute_type_loss(self, type_logits: torch.Tensor, data: dict) -> torch.Tensor:
        """Compute binary availability loss for the five real grasp types.

        Args:
            type_logits: Raw type-head logits from the point-cloud feature.
            data: Training batch containing scene-level binary availability targets.

        Returns:
            Scalar type loss used by ``task=train``.
        """
        if "target_type_availability" not in data:
            raise KeyError("Availability training requires target_type_availability in the batch")
        target = data["target_type_availability"].to(device=type_logits.device, dtype=type_logits.dtype)
        real_type_logits = type_logits[:, 1:]
        if target.shape != real_type_logits.shape:
            raise ValueError(
                "target_type_availability must have shape "
                f"{tuple(real_type_logits.shape)}, got {tuple(target.shape)}"
            )
        return F.binary_cross_entropy_with_logits(real_type_logits, target)

    def _availability_mask(self, type_scores: torch.Tensor, input_type: torch.Tensor) -> torch.Tensor:
        """Build the real-type mask used for availability-guided sampling.

        Args:
            type_scores: Availability scores with shape ``(B, 5)``.
            input_type: Requested type ids with shape ``(B,)``. ``0`` means the
                model should use all available real types.

        Returns:
            Boolean mask with shape ``(B, 5)`` aligned with real type ids ``1..5``.
        """
        available = type_scores >= self.availability_score_threshold
        topk = min(self.availability_min_available_types, type_scores.shape[-1])
        fallback_ids = torch.topk(type_scores, k=topk, dim=-1).indices
        fallback_mask = torch.zeros_like(available)
        fallback_mask.scatter_(1, fallback_ids, True)
        available = available | fallback_mask

        explicit_mask = input_type > 0
        if explicit_mask.any():
            requested = torch.clamp(input_type[explicit_mask] - 1, min=0, max=type_scores.shape[-1] - 1)
            available[explicit_mask] = False
            available[explicit_mask, requested] = True
        return available

    def _sample_with_availability(self, data: dict, global_feature: torch.Tensor, type_logits: torch.Tensor, sample_num: int):
        """Sample pose candidates for every available grasp type.

        Args:
            data: Inference batch. ``grasp_type_id=0`` requests availability
                driven all-type sampling; explicit ids request one fixed type.
            global_feature: Backbone feature tensor with shape ``(B, C)``.
            type_logits: Raw type-head logits with shape ``(B, len(GRASP_TYPES))``.
            sample_num: Number of pose candidates to draw per real grasp type.

        Returns:
            Tuple ``(robot_pose, grasp_type_ids, type_scores, log_prob)`` where
            candidates for unavailable types are assigned very low log-probability
            before top-k selection.
        """
        batch_size = global_feature.shape[0]
        type_scores = self._availability_scores(type_logits)
        input_type = data["grasp_type_id"]
        available = self._availability_mask(type_scores, input_type)

        real_type_ids = torch.arange(1, len(GRASP_TYPES), device=global_feature.device, dtype=input_type.dtype)
        global_feature_by_type = repeat(global_feature, "b c -> (b t) c", t=REAL_GRASP_TYPE_NUM)
        grasp_type_by_type = repeat(real_type_ids, "t -> (b t)", b=batch_size)
        cond_feat = torch.cat([global_feature_by_type, self.grasp_type_emb(grasp_type_by_type)], dim=-1)
        robot_pose, log_prob = self.output_head.sample(cond_feat, grasp_type_by_type, sample_num)

        robot_pose = rearrange(robot_pose, "(b t) s ... -> b (t s) ...", b=batch_size, t=REAL_GRASP_TYPE_NUM)
        log_prob = rearrange(log_prob, "(b t) s -> b (t s)", b=batch_size, t=REAL_GRASP_TYPE_NUM)
        grasp_type_ids = repeat(real_type_ids, "t -> b (t s)", b=batch_size, s=sample_num)
        available = repeat(available, "b t -> b (t s)", s=sample_num)

        if self.availability_use_score_prior:
            type_log_prior = torch.log(type_scores.clamp_min(1e-6))
            log_prob = log_prob + repeat(type_log_prior, "b t -> b (t s)", s=sample_num)
        log_prob = log_prob.masked_fill(~available, torch.finfo(log_prob.dtype).min)

        type_scores_per_candidate = repeat(type_scores, "b c -> b k c", k=log_prob.shape[1])
        return robot_pose, grasp_type_ids, type_scores_per_candidate, log_prob

    def forward(self, data: dict):
        result_dict = {}

        # Encode object pointcloud
        global_feature, local_feature = self.backbone(data)

        # Predict scene-level grasp-type availability.
        type_logits = self.type_classifier(global_feature)
        result_dict["loss_type"] = self._compute_type_loss(type_logits, data)
        batch_num, sample_num = data["grasp_type_id"].shape[0], data["right_hand_trans"].shape[1]
        assert sample_num == 1

        # Generate wrist poses conditioned on object feature and GT grasp type.
        global_feature_expanded = repeat(global_feature, "b c -> (b t) c", t=sample_num)
        grasp_type_id = repeat(data["grasp_type_id"], "b -> (b t)", t=sample_num)
        cond_feat = torch.cat([global_feature_expanded, self.grasp_type_emb(grasp_type_id)], dim=-1)
        diffusion_dict = self.output_head.forward(data, cond_feat)
        result_dict.update(diffusion_dict)

        return result_dict

    def sample(self, data: dict, sample_num: int = 1):
        # Encode object pointcloud
        global_feature, local_feature = self.backbone(data)

        type_logits = self.type_classifier(global_feature)
        return self._sample_with_availability(data, global_feature, type_logits, sample_num)
