import tempfile
import unittest
from unittest import mock

import torch
from omegaconf import OmegaConf

from dexlearn.network.final_layers.diffusion import (
    DiffusionBiRT_v2,
    bimanual_t24_to_pose,
    canonicalize_bimanual_t24,
)
from dexlearn.network.models import reverse as reverse_module


def _identity_t24(batch_size, sample_num=None):
    shape = (batch_size, 24) if sample_num is None else (batch_size, sample_num, 24)
    t24 = torch.zeros(shape, dtype=torch.float32)
    identity = torch.eye(3, dtype=torch.float32).reshape(9)
    t24[..., 0:9] = identity
    t24[..., 12:21] = identity
    return t24


class FakeBackbone(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.projection = torch.nn.Linear(int(cfg.input_dim), int(cfg.out_feat_dim), bias=False)

    def forward(self, data):
        global_feature = self.projection(data["object_feature"])
        return global_feature, global_feature[:, None, :]


class FakeMarginalHead(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.last_forward_condition = None

    def forward(self, data, cond_feat):
        self.last_forward_condition = cond_feat.detach().clone()
        return {"loss_diffusion": cond_feat.square().mean() * self.scale}

    def sample_with_t24(self, cond_feat, sample_num):
        batch_size = cond_feat.shape[0]
        canonical_t24 = _identity_t24(batch_size, sample_num)
        canonical_t24[..., 9:12] = cond_feat[:, None, :3]
        canonical_t24[..., 21:24] = -cond_feat[:, None, :3]
        robot_pose = bimanual_t24_to_pose(canonical_t24)
        log_prob = cond_feat.mean(dim=-1, keepdim=True).expand(-1, sample_num)
        return canonical_t24, robot_pose, log_prob


class FixedDiffusion(torch.nn.Module):
    def __init__(self, samples, log_prob):
        super().__init__()
        self.register_buffer("samples", samples)
        self.register_buffer("fixed_log_prob", log_prob)

    def sample(self, cond):
        if cond.shape[0] != self.samples.shape[0]:
            raise ValueError("Condition rows must match fixed diffusion samples")
        return self.samples.clone(), self.fixed_log_prob.clone()


class IdentityNormalization(torch.nn.Module):
    def inv(self, value):
        return value


def _marginal_config():
    return OmegaConf.create(
        {
            "backbone": {"name": "FakeBackbone", "input_dim": 4, "out_feat_dim": 4},
            "head": {"name": "FakeMarginalHead", "in_feat_dim": None},
        }
    )


def _posterior_config(max_update=2):
    return OmegaConf.create(
        {
            "backbone": {"name": "FakeBackbone", "input_dim": 4, "out_feat_dim": 4},
            "pose_normalization": {"max_update": max_update},
            "pose_encoder": {"hidden_dims": [8, 6]},
            "object_projection": {"out_feat_dim": 5},
            "fusion_classifier": {"hidden_dim": 7},
        }
    )


class ReverseModelTest(unittest.TestCase):
    def setUp(self):
        self.class_patches = (
            mock.patch.object(reverse_module, "FakeBackbone", FakeBackbone, create=True),
            mock.patch.object(reverse_module, "FakeMarginalHead", FakeMarginalHead, create=True),
        )
        for class_patch in self.class_patches:
            class_patch.start()

    def tearDown(self):
        for class_patch in reversed(self.class_patches):
            class_patch.stop()

    def test_canonical_t24_projects_rotations_and_preserves_translations(self):
        raw_t24 = _identity_t24(2)
        raw_t24[:, 0:9] += torch.tensor(
            [[0.0, 0.02, 0.0, -0.01, 0.0, 0.03, 0.0, -0.02, 0.0]], dtype=torch.float32
        )
        raw_t24[:, 12:21] += torch.tensor(
            [[0.0, -0.03, 0.01, 0.02, 0.0, 0.0, -0.01, 0.0, 0.0]], dtype=torch.float32
        )
        raw_t24[:, 9:12] = torch.tensor([[0.1, 0.2, 0.3], [-0.1, 0.4, 0.2]])
        raw_t24[:, 21:24] = torch.tensor([[0.7, -0.2, 0.1], [0.0, 0.5, -0.4]])

        canonical = canonicalize_bimanual_t24(raw_t24)

        self.assertTrue(torch.equal(canonical[:, 9:12], raw_t24[:, 9:12]))
        self.assertTrue(torch.equal(canonical[:, 21:24], raw_t24[:, 21:24]))
        for rotation_slice in (slice(0, 9), slice(12, 21)):
            rotations = canonical[:, rotation_slice].reshape(-1, 3, 3)
            expected_identity = torch.eye(3).expand(rotations.shape[0], -1, -1)
            self.assertTrue(torch.allclose(rotations @ rotations.transpose(-1, -2), expected_identity, atol=1e-5))
            self.assertTrue(torch.allclose(torch.det(rotations), torch.ones(rotations.shape[0]), atol=1e-5))

        pose = bimanual_t24_to_pose(canonical)
        self.assertEqual(tuple(pose.shape), (2, 1, 14))
        self.assertTrue(torch.allclose(pose[:, 0, 0:3], raw_t24[:, 9:12]))
        self.assertTrue(torch.allclose(pose[:, 0, 7:10], raw_t24[:, 21:24]))
        self.assertTrue(torch.allclose(torch.linalg.vector_norm(pose[:, 0, 3:7], dim=-1), torch.ones(2)))
        self.assertTrue(torch.allclose(torch.linalg.vector_norm(pose[:, 0, 10:14], dim=-1), torch.ones(2)))

    def test_diffusion_sample_api_remains_backward_compatible(self):
        batch_size, sample_num = 2, 3
        raw_t24 = _identity_t24(batch_size * sample_num)
        raw_t24[:, 1] = torch.linspace(-0.1, 0.1, batch_size * sample_num)
        raw_t24[:, 9:12] = torch.randn(batch_size * sample_num, 3)
        raw_t24[:, 21:24] = torch.randn(batch_size * sample_num, 3)
        fixed_log_prob = torch.linspace(-1.0, 1.0, batch_size * sample_num)
        head = DiffusionBiRT_v2.__new__(DiffusionBiRT_v2)
        torch.nn.Module.__init__(head)
        head.diffusion = FixedDiffusion(raw_t24, fixed_log_prob)
        head.RMS = IdentityNormalization()
        condition = torch.randn(batch_size, 4)

        canonical_t24, new_pose, new_log_prob = head.sample_with_t24(condition, sample_num)
        legacy_pose, legacy_log_prob = head.sample(condition, torch.tensor([1, 5]), sample_num)

        self.assertEqual(tuple(canonical_t24.shape), (batch_size, sample_num, 24))
        self.assertTrue(torch.equal(new_pose, legacy_pose))
        self.assertTrue(torch.equal(new_log_prob, legacy_log_prob))
        self.assertTrue(torch.equal(new_pose, bimanual_t24_to_pose(canonical_t24)))

    def test_marginal_generator_ignores_mode_and_expands_only_object_condition(self):
        torch.manual_seed(3)
        model = reverse_module.MarginalPoseDiffusionModel(_marginal_config())
        object_feature = torch.randn(2, 4)
        data_a = {
            "object_feature": object_feature,
            "grasp_type_id": torch.tensor([1, 5]),
            "right_hand_trans": torch.zeros(2, 3, 1, 3),
        }
        data_b = dict(data_a)
        data_b["grasp_type_id"] = torch.tensor([4, 2])

        t24_a, pose_a, score_a = model.sample_with_t24(data_a, 4)
        t24_b, pose_b, score_b = model.sample_with_t24(data_b, 4)

        self.assertTrue(torch.equal(t24_a, t24_b))
        self.assertTrue(torch.equal(pose_a, pose_b))
        self.assertTrue(torch.equal(score_a, score_b))
        model(data_a)
        self.assertEqual(tuple(model.output_head.last_forward_condition.shape), (6, 4))

    def test_posterior_is_categorical_pose_conditioned_and_checkpointed(self):
        torch.manual_seed(5)
        posterior = reverse_module.PoseConditionedTypeModel(_posterior_config())
        data = {"object_feature": torch.randn(2, 4)}
        canonical_t24 = _identity_t24(2, 3)
        canonical_t24[:, :, 9] = torch.tensor([[0.0, 0.2, 0.4], [0.1, 0.3, 0.5]])

        posterior.train()
        probabilities = posterior.posterior_probabilities(data, canonical_t24)

        self.assertEqual(tuple(probabilities.shape), (2, 3, 5))
        self.assertTrue(torch.all(probabilities >= 0.0))
        self.assertTrue(torch.allclose(probabilities.sum(dim=-1), torch.ones(2, 3), atol=1e-6))
        self.assertFalse(torch.equal(probabilities[:, 0], probabilities[:, 2]))
        state = posterior.state_dict()
        self.assertIn("pose_normalization.running_ms.mean", state)
        self.assertIn("pose_normalization.running_ms.std", state)
        self.assertIn("pose_normalization.max_update", state)

        with tempfile.NamedTemporaryFile(suffix=".pth") as checkpoint_file:
            torch.save(state, checkpoint_file.name)
            restored = reverse_module.PoseConditionedTypeModel(_posterior_config())
            restored.load_state_dict(torch.load(checkpoint_file.name, map_location="cpu"))
        self.assertTrue(
            torch.equal(
                restored.pose_normalization.running_ms.mean,
                posterior.pose_normalization.running_ms.mean,
            )
        )
        self.assertIsNot(restored.pose_normalization, posterior.pose_normalization)

    def test_posterior_forward_uses_record_hard_labels_for_all_pose_samples(self):
        torch.manual_seed(7)
        posterior = reverse_module.PoseConditionedTypeModel(_posterior_config())
        batch_size, sample_num = 2, 3
        identity = torch.eye(3).reshape(1, 1, 1, 3, 3).expand(batch_size, sample_num, 1, -1, -1)
        data = {
            "object_feature": torch.randn(batch_size, 4),
            "right_hand_trans": torch.randn(batch_size, sample_num, 1, 3),
            "right_hand_rot": identity.clone(),
            "left_hand_trans": torch.randn(batch_size, sample_num, 1, 3),
            "left_hand_rot": identity.clone(),
            "grasp_type_id": torch.tensor([1, 5]),
        }

        result = posterior(data)
        result["loss_type"].backward()

        self.assertEqual(result["loss_type"].ndim, 0)
        self.assertTrue(torch.isfinite(result["loss_type"]))
        self.assertIsNotNone(posterior.type_classifier[-1].weight.grad)

    def test_marginal_and_posterior_share_no_parameter_identity(self):
        marginal = reverse_module.MarginalPoseDiffusionModel(_marginal_config())
        posterior = reverse_module.PoseConditionedTypeModel(_posterior_config(max_update=0))

        marginal_parameter_ids = {id(parameter) for parameter in marginal.parameters()}
        posterior_parameter_ids = {id(parameter) for parameter in posterior.parameters()}

        self.assertTrue(marginal_parameter_ids)
        self.assertTrue(posterior_parameter_ids)
        self.assertTrue(marginal_parameter_ids.isdisjoint(posterior_parameter_ids))
        self.assertIsNot(marginal.backbone, posterior.backbone)


if __name__ == "__main__":
    unittest.main()
