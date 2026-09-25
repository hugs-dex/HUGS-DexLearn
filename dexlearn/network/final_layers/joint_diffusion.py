"""Coupled categorical/continuous diffusion for the Joint Human Prior."""

import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler
from einops import rearrange, repeat

from dexlearn.network.final_layers.diffusion import (
    bimanual_t24_from_data,
    bimanual_t24_to_pose,
    canonicalize_bimanual_t24,
)
from dexlearn.utils.RMS import Normalization


REAL_GRASP_TYPE_NUM = 5
T24_DIM = 24


class SinusoidalTimeEmbedding(torch.nn.Module):
    """Map normalized scalar timesteps to sinusoidal features."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        if dim < 4 or dim % 2 != 0:
            raise ValueError(f"Time embedding dimension must be even and >= 4, got {dim}")
        self.dim = int(dim)
        self.theta = float(theta)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        """Embed normalized timesteps shaped ``(N,)``."""
        half_dim = self.dim // 2
        exponent = -math.log(self.theta) * torch.arange(
            half_dim,
            device=timestep.device,
            dtype=timestep.dtype,
        ) / float(half_dim - 1)
        frequency = torch.exp(exponent)
        angle = timestep[:, None] * frequency[None, :] * self.theta
        return torch.cat([angle.sin(), angle.cos()], dim=-1)


class ResidualMLPBlock(torch.nn.Module):
    """Pre-normalized residual MLP block used by the shared denoiser trunk."""

    def __init__(self, width: int):
        super().__init__()
        self.norm = torch.nn.LayerNorm(width)
        self.fc1 = torch.nn.Linear(width, width)
        self.fc2 = torch.nn.Linear(width, width)
        self.activation = torch.nn.Mish()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(value)
        hidden = self.activation(self.fc1(hidden))
        return value + self.fc2(hidden)


class CoupledJointDenoiser(torch.nn.Module):
    """Shared-state denoiser with categorical and continuous output heads."""

    def __init__(self, cfg):
        super().__init__()
        object_dim = int(cfg.object_feat_dim)
        object_projection_dim = int(getattr(cfg, "object_projection_dim", 256))
        category_embedding_dim = int(getattr(cfg, "category_embedding_dim", 64))
        pose_projection_dim = int(getattr(cfg, "pose_projection_dim", 128))
        time_embedding_dim = int(getattr(cfg, "time_embedding_dim", 128))
        trunk_width = int(getattr(cfg, "trunk_width", 512))
        trunk_blocks = int(getattr(cfg, "trunk_blocks", 4))
        if trunk_blocks <= 0:
            raise ValueError("denoiser.trunk_blocks must be positive")

        self.object_projection = torch.nn.Linear(object_dim, object_projection_dim)
        self.category_embedding = torch.nn.Embedding(REAL_GRASP_TYPE_NUM, category_embedding_dim)
        self.pose_projection = torch.nn.Linear(T24_DIM, pose_projection_dim)
        self.time_embedding = SinusoidalTimeEmbedding(time_embedding_dim)
        fusion_dim = object_projection_dim + category_embedding_dim + pose_projection_dim + time_embedding_dim
        self.input_projection = torch.nn.Linear(fusion_dim, trunk_width)
        self.blocks = torch.nn.ModuleList([ResidualMLPBlock(trunk_width) for _ in range(trunk_blocks)])
        self.final_norm = torch.nn.LayerNorm(trunk_width)
        self.category_head = torch.nn.Linear(trunk_width, REAL_GRASP_TYPE_NUM)
        self.pose_head = torch.nn.Linear(trunk_width, T24_DIM)
        self.activation = torch.nn.Mish()

    def forward(
        self,
        category_t: torch.Tensor,
        pose_t: torch.Tensor,
        object_feature: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict both heads from the same immutable joint state.

        Args:
            category_t: Internal categorical states in ``[0, 4]`` shaped ``(N,)``.
            pose_t: Noisy normalized T24 tensor shaped ``(N, 24)``.
            object_feature: Object features shaped ``(N, C)``.
            timestep: Normalized timesteps shaped ``(N,)``.

        Returns:
            Tuple of clean-category logits and pose v-prediction.
        """
        if pose_t.ndim != 2 or pose_t.shape[-1] != T24_DIM:
            raise ValueError(f"Expected pose_t shape (N,24), got {tuple(pose_t.shape)}")
        if category_t.ndim != 1 or category_t.shape[0] != pose_t.shape[0]:
            raise ValueError("category_t must have shape (N,) matching pose_t")
        if object_feature.ndim != 2 or object_feature.shape[0] != pose_t.shape[0]:
            raise ValueError("object_feature must have shape (N,C) matching pose_t")
        if timestep.ndim != 1 or timestep.shape[0] != pose_t.shape[0]:
            raise ValueError("timestep must have shape (N,) matching pose_t")

        fused = torch.cat(
            [
                self.object_projection(object_feature),
                self.category_embedding(category_t),
                self.pose_projection(pose_t),
                self.time_embedding(timestep),
            ],
            dim=-1,
        )
        hidden = self.activation(self.input_projection(fused))
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)
        return self.category_head(hidden), self.pose_head(hidden)


class JointCategoricalPoseDiffusion(torch.nn.Module):
    """Mixed categorical/Gaussian diffusion over real mode and canonical T24."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        diffusion_cfg = cfg.diffusion
        scheduler_type = str(diffusion_cfg.scheduler_type)
        if scheduler_type != "DDIMScheduler":
            raise ValueError("JointCategoricalPoseDiffusion currently requires DDIMScheduler")
        self.scheduler = DDIMScheduler(**diffusion_cfg.scheduler)
        self.timesteps = int(diffusion_cfg.scheduler.num_train_timesteps)
        self.inference_timesteps = int(diffusion_cfg.num_inference_timesteps)
        self.prediction_type = str(diffusion_cfg.scheduler.prediction_type)
        if self.prediction_type != "v_prediction":
            raise ValueError("JointCategoricalPoseDiffusion requires v_prediction")
        self.pose_loss_type = str(getattr(diffusion_cfg, "loss_type", "l1")).lower()
        if self.pose_loss_type not in {"l1", "l2"}:
            raise ValueError("joint pose loss_type must be l1 or l2")
        self.continuous_path_score_weight = float(
            getattr(diffusion_cfg, "continuous_path_score_weight", 1.0)
        )
        if self.continuous_path_score_weight < 0.0:
            raise ValueError("continuous_path_score_weight must be non-negative")

        cfg.denoiser.object_feat_dim = int(cfg.in_feat_dim)
        self.denoiser = CoupledJointDenoiser(cfg.denoiser)
        normalization_cfg = getattr(cfg, "pose_normalization", None)
        max_update = int(getattr(normalization_cfg, "max_update", 2000))
        self.pose_normalization = Normalization(T24_DIM, max_update=max_update)

    @property
    def alpha_cumprod(self) -> torch.Tensor:
        """Return the shared cumulative alpha schedule used by both processes."""
        return self.scheduler.alphas_cumprod

    def _alpha_bar(self, timestep: torch.Tensor) -> torch.Tensor:
        return self.alpha_cumprod.to(timestep.device, dtype=torch.float32)[timestep.long()]

    def categorical_forward_probabilities(
        self,
        clean_category: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ``q(c_t | c_0)`` for uniform categorical corruption."""
        clean_one_hot = F.one_hot(clean_category.long(), num_classes=REAL_GRASP_TYPE_NUM).float()
        alpha_bar = self._alpha_bar(timestep).to(clean_one_hot.dtype)[:, None]
        return alpha_bar * clean_one_hot + (1.0 - alpha_bar) / REAL_GRASP_TYPE_NUM

    def categorical_forward_sample(
        self,
        clean_category: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Sample categorical forward noise at arbitrary training timesteps."""
        probabilities = self.categorical_forward_probabilities(clean_category, timestep)
        return torch.multinomial(probabilities, num_samples=1).squeeze(-1)

    def categorical_reverse_probabilities(
        self,
        category_t: torch.Tensor,
        clean_logits: torch.Tensor,
        timestep: int,
        previous_timestep: int,
    ) -> torch.Tensor:
        """Compute a skipped-step D3PM posterior mixed by predicted clean mode."""
        clean_prob = torch.softmax(clean_logits, dim=-1)
        if previous_timestep < 0:
            return clean_prob

        device = clean_logits.device
        dtype = clean_logits.dtype
        alpha_bar_t = self.alpha_cumprod.to(device=device, dtype=dtype)[int(timestep)]
        alpha_bar_s = self.alpha_cumprod.to(device=device, dtype=dtype)[int(previous_timestep)]
        transition_alpha = (alpha_bar_t / alpha_bar_s.clamp_min(1e-12)).clamp(0.0, 1.0)

        identity = torch.eye(REAL_GRASP_TYPE_NUM, device=device, dtype=dtype)
        uniform = torch.full_like(identity, 1.0 / REAL_GRASP_TYPE_NUM)
        qbar_s = alpha_bar_s * identity + (1.0 - alpha_bar_s) * uniform
        qbar_t = alpha_bar_t * identity + (1.0 - alpha_bar_t) * uniform
        q_s_to_t = transition_alpha * identity + (1.0 - transition_alpha) * uniform

        likelihood = q_s_to_t[:, category_t.long()].transpose(0, 1)
        denominator = qbar_t[:, category_t.long()].transpose(0, 1).clamp_min(1e-12)
        weighted_clean = clean_prob / denominator
        prior_mixture = weighted_clean @ qbar_s
        posterior = likelihood * prior_mixture
        return posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def forward(self, data: dict, object_feature: torch.Tensor) -> dict:
        """Train both heads from one shared noisy ``(c_t, T_t)`` state."""
        clean_t24 = bimanual_t24_from_data(data)
        normalized_t24 = self.pose_normalization(clean_t24)
        batch_size = int(data["right_hand_trans"].shape[0])
        sample_num = int(data["right_hand_trans"].shape[1])
        expected_rows = batch_size * sample_num
        if object_feature.shape[0] == batch_size:
            object_feature = repeat(object_feature, "b c -> (b s) c", s=sample_num)
        if object_feature.shape[0] != expected_rows:
            raise ValueError("Object feature rows must match flattened T24 training targets")

        external_type = repeat(data["grasp_type_id"].long(), "b -> (b s)", s=sample_num)
        if ((external_type < 1) | (external_type > REAL_GRASP_TYPE_NUM)).any():
            raise ValueError("Joint training requires grasp_type_id in [1, 5]")
        clean_category = external_type - 1

        timestep = torch.randint(
            0,
            self.timesteps,
            (expected_rows,),
            device=normalized_t24.device,
            dtype=torch.long,
        )
        pose_noise = torch.randn_like(normalized_t24)
        pose_t = self.scheduler.add_noise(normalized_t24, pose_noise, timestep)
        category_t = self.categorical_forward_sample(clean_category, timestep)
        normalized_timestep = timestep.to(normalized_t24.dtype) / float(self.timesteps)
        category_logits, pose_prediction = self.denoiser(
            category_t,
            pose_t,
            object_feature,
            normalized_timestep,
        )
        pose_target = self.scheduler.get_velocity(normalized_t24, pose_noise, timestep)
        if self.pose_loss_type == "l1":
            loss_pose = F.smooth_l1_loss(pose_prediction, pose_target)
        else:
            loss_pose = F.mse_loss(pose_prediction, pose_target)
        loss_categorical = F.cross_entropy(category_logits, clean_category)
        return {
            "loss_pose_v": loss_pose,
            "loss_categorical": loss_categorical,
        }

    @staticmethod
    def _normalize_seeds(seed: int | Iterable[int], batch_size: int) -> list[int]:
        if isinstance(seed, Iterable) and not isinstance(seed, (str, bytes)):
            seeds = [int(value) for value in seed]
            if len(seeds) != batch_size:
                raise ValueError(f"Expected {batch_size} per-scene seeds, got {len(seeds)}")
            return seeds
        base_seed = int(seed)
        return [base_seed + batch_index for batch_index in range(batch_size)]

    @staticmethod
    def _scene_generators(device: torch.device, seeds: list[int]) -> list[torch.Generator]:
        generators = []
        for seed in seeds:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
            generators.append(generator)
        return generators

    @staticmethod
    def _sample_rows(
        probabilities: torch.Tensor,
        batch_size: int,
        sample_num: int,
        generators: list[torch.Generator],
    ) -> torch.Tensor:
        reshaped = probabilities.reshape(batch_size, sample_num, -1)
        sampled = [
            torch.multinomial(reshaped[row], num_samples=1, generator=generators[row]).squeeze(-1)
            for row in range(batch_size)
        ]
        return torch.stack(sampled, dim=0).reshape(batch_size * sample_num)

    @staticmethod
    def _randn_rows(
        batch_size: int,
        sample_num: int,
        feature_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        generators: list[torch.Generator],
    ) -> torch.Tensor:
        rows = [
            torch.randn((sample_num, feature_dim), device=device, dtype=dtype, generator=generators[row])
            for row in range(batch_size)
        ]
        return torch.stack(rows, dim=0).reshape(batch_size * sample_num, feature_dim)

    @staticmethod
    def _randint_rows(
        batch_size: int,
        sample_num: int,
        device: torch.device,
        generators: list[torch.Generator],
    ) -> torch.Tensor:
        rows = [
            torch.randint(
                0,
                REAL_GRASP_TYPE_NUM,
                (sample_num,),
                device=device,
                generator=generators[row],
            )
            for row in range(batch_size)
        ]
        return torch.stack(rows, dim=0).reshape(batch_size * sample_num)

    def sample_joint(
        self,
        object_feature: torch.Tensor,
        sample_num: int,
        seed: int | Iterable[int],
        return_trajectory: bool = False,
    ) -> dict:
        """Sample a raw coupled joint pool without mode conditioning.

        Args:
            object_feature: Object feature tensor shaped ``(B, C)``.
            sample_num: Raw joint sample count per object.
            seed: Scalar base seed or one deterministic seed per scene.
            return_trajectory: Whether to retain all intermediate categorical and
                normalized pose states for architecture/reproducibility tests.

        Returns:
            Joint sample dictionary using external type ids ``1..5``.
        """
        if sample_num <= 0:
            raise ValueError("sample_num must be positive")
        batch_size = int(object_feature.shape[0])
        device = object_feature.device
        dtype = object_feature.dtype
        seeds = self._normalize_seeds(seed, batch_size)
        generators = self._scene_generators(device, seeds)
        repeated_object = repeat(object_feature, "b c -> (b s) c", s=sample_num)

        pose_t = self._randn_rows(batch_size, sample_num, T24_DIM, device, dtype, generators)
        category_t = self._randint_rows(batch_size, sample_num, device, generators)
        categorical_path_score = torch.zeros(batch_size * sample_num, device=device, dtype=dtype)
        continuous_path_score = torch.zeros_like(categorical_path_score)
        category_trajectory = [category_t.reshape(batch_size, sample_num).clone()] if return_trajectory else None
        pose_trajectory = [pose_t.reshape(batch_size, sample_num, T24_DIM).clone()] if return_trajectory else None

        self.scheduler.set_timesteps(self.inference_timesteps, device=device)
        schedule = [int(value) for value in self.scheduler.timesteps.detach().cpu().tolist()]
        final_clean_prob = None
        for step_index, timestep in enumerate(schedule):
            previous_timestep = schedule[step_index + 1] if step_index + 1 < len(schedule) else -1
            timestep_batch = torch.full(
                (batch_size * sample_num,),
                timestep,
                device=device,
                dtype=torch.long,
            )
            normalized_timestep = timestep_batch.to(dtype) / float(self.timesteps)

            # Both predictions are computed before either state is updated.
            category_logits, pose_prediction = self.denoiser(
                category_t,
                pose_t,
                repeated_object,
                normalized_timestep,
            )
            category_prob = self.categorical_reverse_probabilities(
                category_t,
                category_logits,
                timestep,
                previous_timestep,
            )
            category_prev = self._sample_rows(category_prob, batch_size, sample_num, generators)
            selected_category_prob = category_prob.gather(1, category_prev[:, None]).squeeze(1)
            categorical_path_score = categorical_path_score + selected_category_prob.clamp_min(1e-12).log()

            alpha_bar = self.alpha_cumprod.to(device=device, dtype=dtype)[timestep]
            predicted_noise = pose_prediction * alpha_bar.sqrt() + pose_t * (1.0 - alpha_bar).sqrt()
            continuous_path_score = continuous_path_score - 0.5 * predicted_noise.square().mean(dim=-1)
            pose_prev = self.scheduler.step(
                pose_prediction,
                timestep,
                pose_t,
                eta=0.0,
            ).prev_sample

            category_t, pose_t = category_prev, pose_prev
            final_clean_prob = torch.softmax(category_logits, dim=-1)
            if return_trajectory:
                category_trajectory.append(category_t.reshape(batch_size, sample_num).clone())
                pose_trajectory.append(pose_t.reshape(batch_size, sample_num, T24_DIM).clone())

        preprojection_t24 = self.pose_normalization.inv(pose_t)
        canonical_t24 = canonicalize_bimanual_t24(preprojection_t24)
        canonical_t24 = rearrange(canonical_t24, "(b s) d -> b s d", b=batch_size, s=sample_num)
        preprojection_t24 = rearrange(
            preprojection_t24,
            "(b s) d -> b s d",
            b=batch_size,
            s=sample_num,
        )
        robot_pose = bimanual_t24_to_pose(canonical_t24)
        external_type_ids = rearrange(category_t + 1, "(b s) -> b s", b=batch_size, s=sample_num)
        categorical_path_score = rearrange(
            categorical_path_score,
            "(b s) -> b s",
            b=batch_size,
            s=sample_num,
        )
        continuous_path_score = rearrange(
            continuous_path_score,
            "(b s) -> b s",
            b=batch_size,
            s=sample_num,
        )
        joint_path_score = categorical_path_score + self.continuous_path_score_weight * continuous_path_score
        final_type_probability = final_clean_prob.gather(1, category_t[:, None]).squeeze(1)
        final_type_probability = rearrange(
            final_type_probability,
            "(b s) -> b s",
            b=batch_size,
            s=sample_num,
        )
        result = {
            "canonical_t24": canonical_t24,
            "type_ids": external_type_ids,
            "robot_pose": robot_pose,
            "joint_path_score": joint_path_score,
            "diagnostics": {
                "preprojection_t24": preprojection_t24,
                "categorical_path_score": categorical_path_score,
                "continuous_path_score": continuous_path_score,
                "final_type_probability": final_type_probability,
                "scene_seeds": torch.as_tensor(seeds, device=device, dtype=torch.long),
                "sampling_timesteps": torch.as_tensor(schedule, device=device, dtype=torch.long),
            },
        }
        if return_trajectory:
            result["diagnostics"].update(
                {
                    "category_trajectory": torch.stack(category_trajectory, dim=2),
                    "normalized_pose_trajectory": torch.stack(pose_trajectory, dim=2),
                }
            )
        return result
