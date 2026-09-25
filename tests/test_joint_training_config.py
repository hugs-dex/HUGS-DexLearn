import unittest
from pathlib import Path

from omegaconf import OmegaConf

from dexlearn.task.train import _build_joint_coupled_training_config, _training_mode
from dexlearn.utils.config import resolve_type_supervision_config


class JointTrainingConfigTest(unittest.TestCase):
    def test_joint_config_is_single_checkpoint_coupled_training(self):
        config_path = Path(__file__).parents[1] / "dexlearn" / "config" / "algo" / "humanMultiJoint.yaml"
        algo = OmegaConf.load(config_path)
        full_config = OmegaConf.create(
            {
                "algo": OmegaConf.to_container(algo, resolve=False),
                "data": {
                    "dataset_type": "HumanMultiDexDataset",
                    "sampling": {"train_unit": "posed_object_uniform", "pose_group_soft_labels": True},
                },
                "model_registry": {"key_features": None},
            }
        )
        resolve_type_supervision_config(full_config)
        training_config = _build_joint_coupled_training_config(full_config)

        self.assertEqual(_training_mode(full_config), "joint_coupled_diffusion")
        self.assertEqual(full_config.algo.factorization, "joint_C_T")
        self.assertEqual(full_config.algo.model.name, "JointHybridDiffusionModel")
        self.assertEqual(full_config.algo.model.head.name, "JointCategoricalPoseDiffusion")
        self.assertEqual(full_config.algo.loss_weight.loss_pose_v, 1.0)
        self.assertEqual(full_config.algo.loss_weight.loss_categorical, 1.0)
        self.assertEqual(full_config.algo.joint_sampling.pool_size, 500)
        self.assertEqual(full_config.algo.test_grasp_num, 100)
        self.assertEqual(full_config.algo.test_topk, 20)
        self.assertEqual(full_config.algo.model.type_objective, "ce")
        self.assertFalse(full_config.algo.supervision.balancing.enabled)
        self.assertEqual(training_config.data.sampling.train_unit, "record_uniform")
        self.assertFalse(training_config.data.sampling.pose_group_soft_labels)
        self.assertIn("joint_C_T", training_config.model_registry.key_features)
        self.assertEqual(full_config.data.sampling.train_unit, "posed_object_uniform")


if __name__ == "__main__":
    unittest.main()
