import unittest

from omegaconf import OmegaConf

from dexlearn.task.train import _build_reverse_branch_config, _reverse_training_branches


def _training_config():
    return OmegaConf.create(
        {
            "exp_name": "reverse_debug",
            "data_name": "humanMulti",
            "algo_name": "humanMultiReverse",
            "ckpt": "old.pth",
            "resume": True,
            "wandb": {"id": "humanMulti_humanMultiReverse_reverse_debug", "resume": True},
            "data": {
                "sampling": {
                    "train_unit": "object_uniform",
                    "pose_group_soft_labels": True,
                }
            },
            "algo": {
                "training": {
                    "run": "both",
                    "pose_marginal": {
                        "exp_suffix": "pose_marginal",
                        "max_iter": 10000,
                        "save_every": 2500,
                        "val_every": 2500,
                        "lr": 1e-3,
                        "lr_min": 1e-4,
                    },
                    "type_posterior": {
                        "exp_suffix": "type_posterior",
                        "max_iter": 300,
                        "save_every": 50,
                        "val_every": 50,
                        "lr": 2e-3,
                        "lr_min": 2e-4,
                    },
                },
                "models": {
                    "pose_marginal": {"name": "MarginalPoseDiffusionModel"},
                    "type_posterior": {"name": "PoseConditionedTypeModel", "type_objective": "ce"},
                },
                "model": {"name": "placeholder"},
                "max_iter": 1,
                "save_every": 1,
                "val_every": 1,
                "lr": 1.0,
                "lr_min": 1.0,
                "supervision": {
                    "balancing": {
                        "enabled": True,
                        "sampler": {"enabled": True},
                        "loss_weight": {"enabled": True},
                    }
                },
                "loss_weight": {"loss_diffusion": 1.0, "loss_type": 1.0},
            },
            "model_registry": {"key_features": None},
        }
    )


class ReverseTrainingConfigTest(unittest.TestCase):
    def test_both_branch_order_is_pose_then_posterior(self):
        self.assertEqual(
            _reverse_training_branches(_training_config()),
            ["pose_marginal", "type_posterior"],
        )

    def test_pose_branch_is_fresh_unbalanced_diffusion_only(self):
        base = _training_config()
        branch = _build_reverse_branch_config(base, "pose_marginal", "reverse_debug_pose_marginal")

        self.assertEqual(branch.exp_name, "reverse_debug_pose_marginal")
        self.assertIsNone(branch.ckpt)
        self.assertFalse(branch.resume)
        self.assertFalse(branch.wandb.resume)
        self.assertEqual(branch.algo.model.name, "MarginalPoseDiffusionModel")
        self.assertEqual(branch.algo.max_iter, 10000)
        self.assertEqual(branch.algo.loss_weight.loss_diffusion, 1.0)
        self.assertEqual(branch.algo.loss_weight.loss_type, 0.0)
        self.assertEqual(branch.data.sampling.train_unit, "record_uniform")
        self.assertFalse(branch.data.sampling.pose_group_soft_labels)
        self.assertFalse(branch.algo.supervision.balancing.enabled)
        self.assertEqual(base.algo.model.name, "placeholder")
        self.assertEqual(base.ckpt, "old.pth")

    def test_posterior_branch_uses_hard_label_ce_budget(self):
        branch = _build_reverse_branch_config(
            _training_config(),
            "type_posterior",
            "reverse_debug_type_posterior",
        )

        self.assertEqual(branch.algo.model.name, "PoseConditionedTypeModel")
        self.assertEqual(branch.algo.model.type_objective, "ce")
        self.assertEqual(branch.algo.max_iter, 300)
        self.assertEqual(branch.algo.save_every, 50)
        self.assertEqual(branch.algo.val_every, 50)
        self.assertEqual(branch.algo.loss_weight.loss_diffusion, 0.0)
        self.assertEqual(branch.algo.loss_weight.loss_type, 1.0)
        self.assertFalse(branch.data.sampling.pose_group_soft_labels)
        self.assertFalse(branch.algo.supervision.balancing.sampler.enabled)
        self.assertFalse(branch.algo.supervision.balancing.loss_weight.enabled)


if __name__ == "__main__":
    unittest.main()
