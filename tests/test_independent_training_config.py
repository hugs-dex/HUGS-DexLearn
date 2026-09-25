import unittest
from pathlib import Path

from omegaconf import OmegaConf

from dexlearn.task.train import (
    _build_independent_marginal_branch_config,
    _independent_marginal_training_branches,
    _training_mode,
)


def _training_config():
    config_path = Path(__file__).parents[1] / "dexlearn" / "config" / "algo" / "humanMultiIndependent.yaml"
    algo = OmegaConf.load(config_path)
    return OmegaConf.create(
        {
            "exp_name": "independent_debug",
            "data_name": "humanMulti",
            "algo_name": "humanMultiIndependent",
            "ckpt": "old.pth",
            "resume": True,
            "wandb": {"id": "humanMulti_humanMultiIndependent_independent_debug", "resume": True},
            "data": {
                "dataset_type": "HumanMultiDexDataset",
                "sampling": {"train_unit": "object_uniform", "pose_group_soft_labels": False},
            },
            "algo": OmegaConf.to_container(algo, resolve=False),
            "model_registry": {"key_features": None},
        }
    )


class IndependentTrainingConfigTest(unittest.TestCase):
    def test_config_identity_and_branch_order(self):
        config = _training_config()

        self.assertEqual(_training_mode(config), "independent_marginals_from_scratch")
        self.assertEqual(config.algo.factorization, "independent_C_T")
        self.assertEqual(
            _independent_marginal_training_branches(config),
            ["mode_marginal", "pose_marginal"],
        )

    def test_mode_branch_is_fresh_object_only_soft_label_ce(self):
        base = _training_config()
        branch = _build_independent_marginal_branch_config(
            base,
            "mode_marginal",
            "independent_debug_mode_marginal",
        )

        self.assertEqual(branch.algo.model.name, "ObjectModeMarginalModel")
        self.assertEqual(branch.algo.max_iter, 300)
        self.assertEqual(branch.algo.loss_weight.loss_type, 1.0)
        self.assertEqual(branch.algo.loss_weight.loss_diffusion, 0.0)
        self.assertEqual(branch.data.sampling.train_unit, "record_uniform")
        self.assertTrue(branch.data.sampling.pose_group_soft_labels)
        self.assertFalse(branch.algo.supervision.balancing.enabled)
        self.assertIsNone(branch.ckpt)
        self.assertFalse(branch.resume)
        self.assertEqual(base.algo.model.name, "MarginalPoseDiffusionModel")
        self.assertEqual(base.ckpt, "old.pth")

    def test_pose_branch_is_fresh_object_only_diffusion(self):
        branch = _build_independent_marginal_branch_config(
            _training_config(),
            "pose_marginal",
            "independent_debug_pose_marginal",
        )

        self.assertEqual(branch.algo.model.name, "MarginalPoseDiffusionModel")
        self.assertEqual(branch.algo.max_iter, 10000)
        self.assertEqual(branch.algo.loss_weight.loss_type, 0.0)
        self.assertEqual(branch.algo.loss_weight.loss_diffusion, 1.0)
        self.assertFalse(branch.data.sampling.pose_group_soft_labels)
        self.assertFalse(branch.algo.supervision.balancing.sampler.enabled)
        self.assertFalse(branch.algo.supervision.balancing.loss_weight.enabled)


if __name__ == "__main__":
    unittest.main()
