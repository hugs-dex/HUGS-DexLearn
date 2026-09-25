"""Independent contact-mode and wrist-pose Human Prior models."""

from collections.abc import Iterable

import torch
import torch.nn.functional as F

from ..backbones import *  # noqa: F401,F403
from dexlearn.dataset.grasp_types import GRASP_TYPES
from .hierarchical import _build_type_head, _normalize_real_type_loss_weights


REAL_GRASP_TYPE_NUM = len(GRASP_TYPES) - 1


def _scene_seed_list(seed: int | Iterable[int], batch_size: int) -> list[int]:
    """Normalize one scalar or per-scene seed sequence.

    Args:
        seed: Scalar seed shared by the batch or one seed per scene.
        batch_size: Number of object observations in the batch.

    Returns:
        List containing exactly one integer seed per scene.
    """
    if isinstance(seed, int):
        return [int(seed)] * batch_size
    seeds = [int(value) for value in seed]
    if len(seeds) != batch_size:
        raise ValueError(f"Expected {batch_size} scene seeds, got {len(seeds)}")
    return seeds


class ObjectModeMarginalModel(torch.nn.Module):
    """Object-only categorical marginal ``p(c|o)`` over real modes 1..5."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.type_objective = str(getattr(cfg, "type_objective", "ce")).lower()
        if self.type_objective != "ce":
            raise ValueError("ObjectModeMarginalModel only supports type_objective=ce")

        self.backbone = eval(cfg.backbone.name)(cfg.backbone)
        type_head_cfg = getattr(cfg, "type_head", None)
        hidden_dim = int(getattr(type_head_cfg, "hidden_dim", 64))
        self.type_classifier = _build_type_head(
            int(cfg.backbone.out_feat_dim),
            REAL_GRASP_TYPE_NUM,
            hidden_dim=hidden_dim,
        )

        type_loss_weights = getattr(cfg, "type_loss_weights", None)
        if type_loss_weights is not None:
            type_loss_weights = _normalize_real_type_loss_weights(type_loss_weights)
            self.register_buffer(
                "type_loss_weights",
                torch.tensor(list(type_loss_weights), dtype=torch.float32),
                persistent=False,
            )
        else:
            self.type_loss_weights = None

    def encode_object(self, data: dict) -> torch.Tensor:
        """Encode only the object observation."""
        global_feature, _ = self.backbone(data)
        return global_feature

    def mode_probabilities(self, data: dict) -> torch.Tensor:
        """Return five normalized real-mode probabilities for each object."""
        return F.softmax(self.type_classifier(self.encode_object(data)), dim=-1)

    def _compute_ce_loss(self, probabilities: torch.Tensor, data: dict) -> torch.Tensor:
        """Compute hard- or soft-label five-way marginal cross entropy."""
        log_probabilities = torch.log(probabilities.clamp_min(1e-8))
        if "target_type_distribution" in data:
            target = data["target_type_distribution"].to(
                dtype=probabilities.dtype,
                device=probabilities.device,
            )
            if target.shape != probabilities.shape:
                raise ValueError(
                    "target_type_distribution must match mode probabilities, got "
                    f"{tuple(target.shape)} vs {tuple(probabilities.shape)}"
                )
            if not torch.allclose(target.sum(dim=-1), torch.ones_like(target[:, 0]), atol=1e-4):
                raise ValueError("target_type_distribution rows must sum to 1")
            return -(target * log_probabilities).sum(dim=-1).mean()

        type_ids = data["grasp_type_id"].long()
        if ((type_ids < 1) | (type_ids > REAL_GRASP_TYPE_NUM)).any():
            raise ValueError("Mode marginal training requires grasp_type_id in [1, 5]")
        return F.nll_loss(log_probabilities, type_ids - 1, weight=self.type_loss_weights)

    def forward(self, data: dict) -> dict:
        """Compute the mode-marginal training loss without reading pose fields."""
        probabilities = self.mode_probabilities(data)
        return {
            "loss_type": self._compute_ce_loss(probabilities, data),
            "metric_type_score_mean": probabilities.mean().detach(),
        }

    def sample_modes(
        self,
        data: dict,
        sample_num: int,
        seed: int | Iterable[int],
    ) -> dict:
        """Sample categorical modes with one independently controlled RNG per scene."""
        if sample_num <= 0:
            raise ValueError(f"sample_num must be positive, got {sample_num}")
        probabilities = self.mode_probabilities(data)
        scene_seeds = _scene_seed_list(seed, probabilities.shape[0])
        sampled_rows = []
        for batch_index, scene_seed in enumerate(scene_seeds):
            generator = torch.Generator(device=probabilities.device)
            generator.manual_seed(scene_seed)
            sampled = torch.multinomial(
                probabilities[batch_index],
                num_samples=sample_num,
                replacement=True,
                generator=generator,
            )
            sampled_rows.append(sampled + 1)
        return {
            "mode_probabilities": probabilities,
            "sampled_type_ids": torch.stack(sampled_rows, dim=0),
            "sampling_seeds": torch.as_tensor(scene_seeds, dtype=torch.int64),
        }

    def sample(
        self,
        data: dict,
        sample_num: int = 1,
        seed: int | Iterable[int] = 0,
    ):
        """Expose the existing score-only sample schema for generic diagnostics."""
        result = self.sample_modes(data, sample_num, seed=seed)
        return {
            "pred_grasp_type_prob": result["mode_probabilities"].unsqueeze(1),
            "pred_grasp_type_id": result["sampled_type_ids"],
        }
