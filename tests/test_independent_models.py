import unittest
from unittest import mock

import torch
from omegaconf import OmegaConf

from dexlearn.network.final_layers.diffusion import (
    bimanual_t24_to_pose,
    canonicalize_bimanual_t24,
)
from dexlearn.network.models import independent as independent_module
from dexlearn.network.models import reverse as reverse_module


class FakeBackbone(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.projection = torch.nn.Linear(int(cfg.input_dim), int(cfg.out_feat_dim), bias=False)

    def forward(self, data):
        global_feature = self.projection(data["object_feature"])
        return global_feature, global_feature[:, None]


class FakeSeededPoseHead(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.policy = type("Policy", (), {"channels": 24})()

    def forward(self, data, cond_feat):
        return {"loss_diffusion": cond_feat.square().mean() * self.scale}

    def sample_with_t24(self, cond_feat, sample_num, initial_noise=None):
        if initial_noise is None:
            initial_noise = torch.zeros(cond_feat.shape[0] * sample_num, 24, device=cond_feat.device)
        canonical_t24 = canonicalize_bimanual_t24(initial_noise)
        canonical_t24 = canonical_t24.reshape(cond_feat.shape[0], sample_num, 24)
        return (
            canonical_t24,
            bimanual_t24_to_pose(canonical_t24),
            initial_noise.reshape(cond_feat.shape[0], sample_num, 24).mean(dim=-1),
        )


def _mode_config():
    return OmegaConf.create(
        {
            "type_objective": "ce",
            "backbone": {"name": "FakeBackbone", "input_dim": 4, "out_feat_dim": 8},
            "type_head": {"hidden_dim": 6},
        }
    )


def _pose_config():
    return OmegaConf.create(
        {
            "backbone": {"name": "FakeBackbone", "input_dim": 4, "out_feat_dim": 8},
            "head": {"name": "FakeSeededPoseHead", "in_feat_dim": None},
        }
    )


class IndependentModelTest(unittest.TestCase):
    def setUp(self):
        self.patches = (
            mock.patch.object(independent_module, "FakeBackbone", FakeBackbone, create=True),
            mock.patch.object(reverse_module, "FakeBackbone", FakeBackbone, create=True),
            mock.patch.object(reverse_module, "FakeSeededPoseHead", FakeSeededPoseHead, create=True),
        )
        for patch in self.patches:
            patch.start()

    def tearDown(self):
        for patch in reversed(self.patches):
            patch.stop()

    def test_mode_marginal_uses_only_object_input_and_soft_labels(self):
        torch.manual_seed(3)
        model = independent_module.ObjectModeMarginalModel(_mode_config())
        data = {
            "object_feature": torch.randn(2, 4),
            "target_type_distribution": torch.tensor(
                [[0.1, 0.2, 0.3, 0.2, 0.2], [0.4, 0.1, 0.1, 0.2, 0.2]],
                dtype=torch.float32,
            ),
        }

        result = model(data)
        result["loss_type"].backward()
        first = model.sample_modes(data, sample_num=32, seed=[11, 22])
        data["right_hand_trans"] = torch.randn(2, 7, 1, 3)
        data["left_hand_rot"] = torch.randn(2, 7, 1, 3, 3)
        second = model.sample_modes(data, sample_num=32, seed=[11, 22])

        self.assertTrue(torch.isfinite(result["loss_type"]))
        self.assertEqual(tuple(first["mode_probabilities"].shape), (2, 5))
        self.assertTrue(torch.equal(first["sampled_type_ids"], second["sampled_type_ids"]))
        self.assertTrue(torch.equal(first["mode_probabilities"], second["mode_probabilities"]))
        self.assertTrue(((first["sampled_type_ids"] >= 1) & (first["sampled_type_ids"] <= 5)).all())
        self.assertIsNotNone(model.type_classifier[-1].weight.grad)
        self.assertFalse(hasattr(model, "grasp_type_emb"))
        self.assertFalse(hasattr(model, "output_head"))

    def test_pose_marginal_seed_is_independent_of_mode_placeholder(self):
        torch.manual_seed(5)
        model = reverse_module.MarginalPoseDiffusionModel(_pose_config())
        object_feature = torch.randn(2, 4)
        data_a = {"object_feature": object_feature, "grasp_type_id": torch.tensor([1, 5])}
        data_b = {"object_feature": object_feature, "grasp_type_id": torch.tensor([4, 2])}

        first = model.sample_with_t24(data_a, sample_num=6, seed=[101, 202])
        second = model.sample_with_t24(data_b, sample_num=6, seed=[101, 202])
        changed_seed = model.sample_with_t24(data_a, sample_num=6, seed=[303, 404])

        for first_value, second_value in zip(first, second):
            self.assertTrue(torch.equal(first_value, second_value))
        self.assertFalse(torch.equal(first[0], changed_seed[0]))
        self.assertEqual(tuple(first[0].shape), (2, 6, 24))
        self.assertEqual(tuple(first[1].shape), (2, 6, 1, 14))

    def test_mode_and_pose_marginals_share_no_parameters(self):
        mode_model = independent_module.ObjectModeMarginalModel(_mode_config())
        pose_model = reverse_module.MarginalPoseDiffusionModel(_pose_config())

        mode_parameter_ids = {id(parameter) for parameter in mode_model.parameters()}
        pose_parameter_ids = {id(parameter) for parameter in pose_model.parameters()}

        self.assertTrue(mode_parameter_ids)
        self.assertTrue(pose_parameter_ids)
        self.assertTrue(mode_parameter_ids.isdisjoint(pose_parameter_ids))
        self.assertIsNot(mode_model.backbone, pose_model.backbone)

    def test_branch_randomness_and_gradients_are_independently_controlled(self):
        torch.manual_seed(13)
        mode_model = independent_module.ObjectModeMarginalModel(_mode_config())
        pose_model = reverse_module.MarginalPoseDiffusionModel(_pose_config())
        object_feature = torch.randn(2, 4)
        mode_data = {
            "object_feature": object_feature,
            "target_type_distribution": torch.full((2, 5), 0.2),
        }
        pose_data = {
            "object_feature": object_feature,
            "grasp_type_id": torch.tensor([1, 5]),
            "right_hand_trans": torch.zeros(2, 1, 1, 3),
        }

        base_modes = mode_model.sample_modes(mode_data, sample_num=128, seed=[17, 19])
        base_pose = pose_model.sample_with_t24(pose_data, sample_num=8, seed=[23, 29])
        changed_modes = mode_model.sample_modes(mode_data, sample_num=128, seed=[31, 37])
        unchanged_pose = pose_model.sample_with_t24(pose_data, sample_num=8, seed=[23, 29])
        unchanged_modes = mode_model.sample_modes(mode_data, sample_num=128, seed=[17, 19])
        changed_pose = pose_model.sample_with_t24(pose_data, sample_num=8, seed=[41, 43])

        self.assertFalse(torch.equal(base_modes["sampled_type_ids"], changed_modes["sampled_type_ids"]))
        self.assertTrue(torch.equal(base_pose[0], unchanged_pose[0]))
        self.assertTrue(torch.equal(base_modes["sampled_type_ids"], unchanged_modes["sampled_type_ids"]))
        self.assertFalse(torch.equal(base_pose[0], changed_pose[0]))

        mode_model.zero_grad(set_to_none=True)
        pose_model.zero_grad(set_to_none=True)
        mode_model(mode_data)["loss_type"].backward()
        self.assertTrue(any(parameter.grad is not None for parameter in mode_model.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in pose_model.parameters()))

        mode_model.zero_grad(set_to_none=True)
        pose_model.zero_grad(set_to_none=True)
        pose_model(pose_data)["loss_diffusion"].backward()
        self.assertTrue(all(parameter.grad is None for parameter in mode_model.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in pose_model.parameters()))


if __name__ == "__main__":
    unittest.main()
