import torch
import torch.nn.functional as F
from einops import repeat, rearrange
from typing import Tuple

from ..backbones import *
from ..type_emb import *
from ..final_layers import *
from dexlearn.dataset.grasp_types import GRASP_TYPES

REAL_GRASP_TYPE_IDS = tuple(range(1, len(GRASP_TYPES)))
REAL_GRASP_TYPE_NUM = len(REAL_GRASP_TYPE_IDS)


def _build_type_head(input_dim: int, output_dim: int, hidden_dim: int = 256) -> torch.nn.Module:
    """Build the shared MLP used by grasp-type heads.

    Args:
        input_dim: Input feature dimension from the point-cloud backbone.
        output_dim: Number of logits produced by the head.
        hidden_dim: Hidden feature size of the intermediate MLP layer.

    Returns:
        Torch module that maps backbone features to type logits.
    """
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, output_dim),
    )


def _normalize_real_type_loss_weights(type_loss_weights):
    """Normalize configured CE class weights for the real grasp types.

    Args:
        type_loss_weights: Optional iterable of class weights. The legacy
            categorical path may provide weights for all ``GRASP_TYPES``,
            including ``0_any``.

    Returns:
        Tensor-ready Python list aligned with the 5 real grasp types, or
        ``None`` when no weights are configured.
    """
    if type_loss_weights is None:
        return None
    weight_list = list(type_loss_weights)
    if len(weight_list) == len(GRASP_TYPES):
        return weight_list[1:]
    if len(weight_list) == REAL_GRASP_TYPE_NUM:
        return weight_list
    raise ValueError(
        f"type_loss_weights must have length {len(GRASP_TYPES)} or {REAL_GRASP_TYPE_NUM}, "
        f"got {len(weight_list)}"
    )


class HierarchicalTypeObjectiveModel(torch.nn.Module):
    """Hierarchical human model whose type predictor is trained with CE only."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.train_type_only = bool(getattr(cfg, "train_type_only", False))
        self.type_objective = str(getattr(cfg, "type_objective", "ce")).lower()
        if self.type_objective != "ce":
            raise ValueError("HierarchicalTypeObjectiveModel only supports type_objective=ce")

        self.backbone = eval(cfg.backbone.name)(cfg.backbone)
        type_head_cfg = getattr(cfg, "type_head", None)
        hidden_dim = int(getattr(type_head_cfg, "hidden_dim", 256))
        self.type_classifier = _build_type_head(
            cfg.backbone.out_feat_dim,
            REAL_GRASP_TYPE_NUM,
            hidden_dim=hidden_dim,
        )

        self.grasp_type_emb = eval(cfg.grasp_type_emb.name)(cfg.grasp_type_emb)

        type_loss_weights = getattr(cfg, "type_loss_weights", None)
        if type_loss_weights is not None:
            type_loss_weights = _normalize_real_type_loss_weights(type_loss_weights)
        if type_loss_weights is not None:
            self.register_buffer(
                "type_loss_weights",
                torch.tensor(list(type_loss_weights), dtype=torch.float32),
                persistent=False,
            )
        else:
            self.type_loss_weights = None

        cfg.head.in_feat_dim = cfg.backbone.out_feat_dim + cfg.grasp_type_emb.out_feat_dim
        self.output_head = eval(cfg.head.name)(cfg.head)

    def _compute_type_scores(self, global_feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict type logits and normalized real-type scores.

        Args:
            global_feature: Backbone feature tensor with shape ``(B, C)``.

        Returns:
            Tuple of raw logits and five real-type scores for grasp types
            ``1..5``.
        """
        type_logits = self.type_classifier(global_feature)
        type_scores = F.softmax(type_logits, dim=-1)
        return type_logits, type_scores

    def _compute_ce_loss(self, type_logits: torch.Tensor, data: dict) -> torch.Tensor:
        """Compute 5-way softmax cross-entropy loss.

        Args:
            type_logits: Predicted logits with shape ``(B, 5)`` for real
                grasp types ``1..5``.
            data: Batch dictionary containing either ``target_type_distribution``
                as soft labels over real types ``1..5`` or hard
                ``grasp_type_id`` labels in ``[1, 5]``.

        Returns:
            Scalar categorical negative-log-likelihood over the 5 real types.
        """
        if "target_type_distribution" in data:
            target_distribution = data["target_type_distribution"].to(
                dtype=type_logits.dtype,
                device=type_logits.device,
            )
            if target_distribution.shape != type_logits.shape:
                raise ValueError(
                    "target_type_distribution must match type_logits shape, got "
                    f"{tuple(target_distribution.shape)} vs {tuple(type_logits.shape)}"
                )
            target_sum = target_distribution.sum(dim=-1)
            if not torch.allclose(target_sum, torch.ones_like(target_sum), atol=1e-4):
                raise ValueError("target_type_distribution rows must sum to 1")
            log_probs = F.log_softmax(type_logits, dim=-1)
            return -(target_distribution * log_probs).sum(dim=-1).mean()

        gt_type = data["grasp_type_id"].long()
        if (gt_type == 0).any():
            raise ValueError("Training data contains grasp_type_id = 0, which should not be used for training")
        gt_index = gt_type - 1
        if ((gt_index < 0) | (gt_index >= REAL_GRASP_TYPE_NUM)).any():
            raise ValueError("CE training requires grasp_type_id in [1, 5]")
        return F.cross_entropy(type_logits, gt_index, weight=self.type_loss_weights)

    def forward(self, data: dict):
        result_dict = {}
        global_feature, local_feature = self.backbone(data)
        type_logits, type_scores = self._compute_type_scores(global_feature)
        result_dict["loss_type"] = self._compute_ce_loss(type_logits, data)

        if self.train_type_only:
            with torch.no_grad():
                result_dict["metric_type_score_mean"] = type_scores.mean()
            return result_dict

        batch_num, sample_num = data["grasp_type_id"].shape[0], data["right_hand_trans"].shape[1]
        assert sample_num == 1
        grasp_type_id = repeat(data["grasp_type_id"], "b -> (b t)", t=sample_num)
        if ((grasp_type_id <= 0) | (grasp_type_id >= len(GRASP_TYPES))).any():
            raise ValueError("Training data for HierarchicalTypeObjectiveModel must use explicit types in [1, 5]")

        global_feature_expanded = repeat(global_feature, "b c -> (b t) c", t=sample_num)
        cond_feat = torch.cat([global_feature_expanded, self.grasp_type_emb(grasp_type_id)], dim=-1)
        diffusion_dict = self.output_head.forward(data, cond_feat)
        result_dict.update(diffusion_dict)

        with torch.no_grad():
            result_dict["metric_type_score_mean"] = type_scores.mean()
        return result_dict

    def sample(self, data: dict, sample_num: int = 1):
        global_feature, local_feature = self.backbone(data)
        type_logits, type_scores = self._compute_type_scores(global_feature)
        input_type = data["grasp_type_id"]
        if input_type.ndim != 1:
            raise ValueError(f"Expected grasp_type_id to have shape (B,), got {tuple(input_type.shape)}")

        # Type-only checkpoints do not train the diffusion head, so inference
        # must export Human Prior scores without producing pose samples.
        if self.train_type_only:
            if ((input_type < 0) | (input_type >= len(GRASP_TYPES))).any():
                raise ValueError("Type-only sampling expects grasp_type_id in [0, 5]")
            pred_type_id = torch.argmax(type_scores, dim=-1) + 1
            return {
                "pred_grasp_type_prob": type_scores.unsqueeze(1),
                "pred_grasp_type_id": pred_type_id.unsqueeze(1),
            }

        if torch.any(input_type == 0):
            if not torch.all(input_type == 0):
                raise ValueError("Mixed 0_any and explicit grasp types are not supported in one batch")
            pred_type_id = torch.argmax(type_scores, dim=-1) + 1
            return {
                "pred_grasp_type_prob": type_scores.unsqueeze(1),
                "pred_grasp_type_id": pred_type_id.unsqueeze(1),
            }

        if ((input_type <= 0) | (input_type >= len(GRASP_TYPES))).any():
            raise ValueError("Explicit sampling requires grasp_type_id in [1, 5]")

        batch_size = global_feature.shape[0]
        grasp_type_ids = input_type.unsqueeze(1).expand(-1, sample_num)
        global_feature_expanded = repeat(global_feature, "b c -> (b s) c", s=sample_num)
        grasp_type_ids_flat = rearrange(grasp_type_ids, "b s -> (b s)")
        cond_feat = torch.cat([global_feature_expanded, self.grasp_type_emb(grasp_type_ids_flat)], dim=-1)
        robot_pose, log_prob = self.output_head.sample(cond_feat, grasp_type_ids_flat, 1)

        robot_pose = rearrange(robot_pose, "(b s) t ... -> b (s t) ...", b=batch_size, s=sample_num)
        log_prob = rearrange(log_prob, "(b s) t -> b (s t)", b=batch_size, s=sample_num)
        type_scores = repeat(type_scores, "b c -> b s c", s=sample_num)
        return robot_pose, grasp_type_ids, type_scores, log_prob
