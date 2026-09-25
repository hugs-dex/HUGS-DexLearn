import unittest
from unittest import mock

import torch
from omegaconf import OmegaConf

from dexlearn.network.final_layers.joint_diffusion import (
    CoupledJointDenoiser,
    JointCategoricalPoseDiffusion,
)
from dexlearn.network.models import joint as joint_module


def _head_config(in_feat_dim=8, inference_steps=4):
    return OmegaConf.create(
        {
            "in_feat_dim": in_feat_dim,
            "pose_normalization": {"max_update": 0},
            "denoiser": {
                "object_feat_dim": None,
                "object_projection_dim": 7,
                "category_embedding_dim": 5,
                "pose_projection_dim": 6,
                "time_embedding_dim": 8,
                "trunk_width": 16,
                "trunk_blocks": 2,
            },
            "diffusion": {
                "scheduler_type": "DDIMScheduler",
                "scheduler": {
                    "beta_schedule": "squaredcos_cap_v2",
                    "prediction_type": "v_prediction",
                    "num_train_timesteps": 16,
                    "clip_sample": False,
                },
                "num_inference_timesteps": inference_steps,
                "loss_type": "l1",
                "continuous_path_score_weight": 1.0,
            },
        }
    )


def _identity_training_data(batch_size=2, sample_num=3):
    identity = torch.eye(3).reshape(1, 1, 1, 3, 3).expand(batch_size, sample_num, 1, -1, -1)
    return {
        "right_hand_trans": torch.randn(batch_size, sample_num, 1, 3),
        "right_hand_rot": identity.clone(),
        "left_hand_trans": torch.randn(batch_size, sample_num, 1, 3),
        "left_hand_rot": identity.clone(),
        "grasp_type_id": torch.tensor([1, 5]),
    }


class FakeBackbone(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.projection = torch.nn.Linear(int(cfg.input_dim), int(cfg.out_feat_dim), bias=False)
        self.forward_count = 0

    def forward(self, data):
        self.forward_count += 1
        global_feature = self.projection(data["object_feature"])
        return global_feature, global_feature[:, None]


class JointModelTest(unittest.TestCase):
    def test_both_heads_depend_on_the_other_noisy_variable(self):
        torch.manual_seed(3)
        cfg = OmegaConf.create(
            {
                "object_feat_dim": 8,
                "object_projection_dim": 7,
                "category_embedding_dim": 5,
                "pose_projection_dim": 6,
                "time_embedding_dim": 8,
                "trunk_width": 16,
                "trunk_blocks": 2,
            }
        )
        denoiser = CoupledJointDenoiser(cfg)
        category = torch.tensor([0, 1, 2, 3])
        pose = torch.randn(4, 24)
        object_feature = torch.randn(4, 8)
        timestep = torch.full((4,), 0.5)

        category_logits, pose_prediction = denoiser(category, pose, object_feature, timestep)
        _, changed_pose_prediction = denoiser((category + 1) % 5, pose, object_feature, timestep)
        changed_category_logits, _ = denoiser(category, pose + 0.25, object_feature, timestep)

        self.assertFalse(torch.equal(pose_prediction, changed_pose_prediction))
        self.assertFalse(torch.equal(category_logits, changed_category_logits))

    def test_training_uses_real_classes_and_returns_separate_losses(self):
        torch.manual_seed(5)
        head = JointCategoricalPoseDiffusion(_head_config())
        head.train()
        data = _identity_training_data()
        result = head(data, torch.randn(2, 8))
        total = result["loss_pose_v"] + result["loss_categorical"]
        total.backward()

        self.assertEqual(set(result), {"loss_pose_v", "loss_categorical"})
        self.assertTrue(torch.isfinite(total))
        self.assertIsNotNone(head.denoiser.category_embedding.weight.grad)
        self.assertIsNotNone(head.denoiser.pose_projection.weight.grad)

    def test_categorical_forward_and_reverse_probabilities_are_normalized(self):
        head = JointCategoricalPoseDiffusion(_head_config())
        clean = torch.tensor([0, 1, 2, 3, 4])
        timestep = torch.tensor([0, 3, 7, 12, 15])
        forward_prob = head.categorical_forward_probabilities(clean, timestep)
        reverse_prob = head.categorical_reverse_probabilities(
            category_t=torch.tensor([4, 3, 2, 1, 0]),
            clean_logits=torch.randn(5, 5),
            timestep=12,
            previous_timestep=8,
        )

        self.assertTrue(torch.allclose(forward_prob.sum(dim=-1), torch.ones(5), atol=1e-6))
        self.assertTrue(torch.allclose(reverse_prob.sum(dim=-1), torch.ones(5), atol=1e-6))
        self.assertTrue((reverse_prob >= 0.0).all())

    def test_sampling_is_coupled_deterministic_and_ignores_placeholder_type(self):
        model_cfg = OmegaConf.create(
            {
                "backbone": {"name": "FakeBackbone", "input_dim": 4, "out_feat_dim": 8},
                "head": OmegaConf.to_container(_head_config(in_feat_dim=8), resolve=True),
            }
        )
        model_cfg.head.name = "JointCategoricalPoseDiffusion"
        with mock.patch.object(joint_module, "FakeBackbone", FakeBackbone, create=True):
            torch.manual_seed(7)
            model = joint_module.JointHybridDiffusionModel(model_cfg)
        model.eval()
        object_feature = torch.randn(2, 4)
        data_a = {"object_feature": object_feature, "grasp_type_id": torch.tensor([0, 0])}
        data_b = {"object_feature": object_feature, "grasp_type_id": torch.tensor([4, 2])}

        first = model.sample_joint(data_a, pool_size=6, seed=[101, 202], return_trajectory=True)
        second = model.sample_joint(data_b, pool_size=6, seed=[101, 202], return_trajectory=True)

        self.assertEqual(model.backbone.forward_count, 2)
        for key in ("canonical_t24", "type_ids", "robot_pose", "joint_path_score"):
            self.assertTrue(torch.equal(first[key], second[key]))
        self.assertTrue(
            torch.equal(
                first["diagnostics"]["category_trajectory"],
                second["diagnostics"]["category_trajectory"],
            )
        )
        self.assertTrue(
            torch.equal(
                first["diagnostics"]["normalized_pose_trajectory"],
                second["diagnostics"]["normalized_pose_trajectory"],
            )
        )
        self.assertEqual(tuple(first["canonical_t24"].shape), (2, 6, 24))
        self.assertEqual(tuple(first["robot_pose"].shape), (2, 6, 1, 14))
        self.assertTrue(((first["type_ids"] >= 1) & (first["type_ids"] <= 5)).all())
        self.assertTrue(torch.isfinite(first["canonical_t24"]).all())
        self.assertTrue(torch.isfinite(first["joint_path_score"]).all())
        self.assertEqual(first["diagnostics"]["category_trajectory"].shape[2], 5)


if __name__ == "__main__":
    unittest.main()
