import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from omegaconf import OmegaConf

from dexlearn.network.final_layers.diffusion import bimanual_t24_to_pose
from dexlearn.task import obj_human_prior_export as export_module
from dexlearn.task.obj_human_prior_export import (
    JOINT_FACTORIZATION,
    build_joint_hard_mode_adapter,
    export_scene_dir,
    joint_mode_budget_scores,
    sample_joint_scene_scores_and_poses,
    validate_scene_export_completeness,
)


RAW_COUNTS = [0, 3, 20, 100, 377]


def _joint_config(object_path="objects"):
    return OmegaConf.create(
        {
            "seed": 17,
            "algo": {
                "human": True,
                "factorization": JOINT_FACTORIZATION,
                "test_grasp_num": 100,
                "test_topk": 20,
                "sample_selection": {
                    "enabled": True,
                    "scope": "global",
                    "mode": "prob_pose",
                    "intermediate_topk": 50,
                    "translation_scale_m": 0.05,
                    "rotation_weight": 1.0,
                },
                "joint_sampling": {
                    "pool_size": 500,
                    "candidate_cap": 100,
                    "sampling_seed": 31,
                    "include_raw_pool": True,
                    "raw_artifact_format": "npz",
                    "zero_support_policy": "finite_placeholder",
                    "low_support_policy": "deterministic_replacement",
                },
                "model": {"name": "JointHybridDiffusionModel", "type_objective": "ce"},
            },
            "data": {"hand_pos_source": "index_mcp"},
            "test_data": {"test_split": "all", "object_path": object_path},
            "task": {
                "samples_per_type": 20,
                "robot_name": "leap",
                "robot_size": 1.0,
                "quat_norm_tol": 1e-3,
                "include_log_prob": False,
                "include_grasp_pose": False,
                "object_splits": ["all"],
                "score_grasp_types": ["0_any"],
                "pose_grasp_types": [],
            },
        }
    )


def _raw_type_ids():
    values = []
    for type_id, count in enumerate(RAW_COUNTS, start=1):
        values.extend([type_id] * count)
    return torch.tensor(values, dtype=torch.long)[None]


def _canonical_t24(pool_size=500):
    t24 = torch.zeros(1, pool_size, 24, dtype=torch.float32)
    identity = torch.eye(3, dtype=torch.float32).reshape(9)
    t24[..., 0:9] = identity
    t24[..., 12:21] = identity
    t24[..., 9] = torch.arange(pool_size, dtype=torch.float32) / 1000.0
    t24[..., 21] = -torch.arange(pool_size, dtype=torch.float32) / 1000.0
    return t24


class FakeJointModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.placeholder_ids = []

    def sample_joint(self, data, pool_size, seed):
        self.placeholder_ids.append(data["grasp_type_id"].detach().cpu().clone())
        canonical_t24 = _canonical_t24(pool_size)
        type_ids = _raw_type_ids()
        robot_pose = bimanual_t24_to_pose(canonical_t24)
        joint_path_score = torch.linspace(-2.0, 2.0, pool_size)[None]
        return {
            "canonical_t24": canonical_t24,
            "type_ids": type_ids,
            "robot_pose": robot_pose,
            "joint_path_score": joint_path_score,
            "diagnostics": {
                "preprojection_t24": canonical_t24.clone(),
                "categorical_path_score": torch.zeros_like(joint_path_score),
                "continuous_path_score": joint_path_score.clone(),
                "final_type_probability": torch.full_like(joint_path_score, 0.5),
                "sampling_timesteps": torch.tensor([12, 8, 4, 0]),
            },
        }


class JointExportTest(unittest.TestCase):
    def test_raw_frequency_uses_external_id_mapping(self):
        scores, counts = joint_mode_budget_scores(_raw_type_ids())

        self.assertEqual(counts.tolist(), [RAW_COUNTS])
        self.assertTrue(
            torch.allclose(
                scores,
                torch.tensor([[value / 500.0 for value in RAW_COUNTS]]),
            )
        )

    def test_hard_group_adapter_handles_cap_replacement_and_zero_support(self):
        config = _joint_config()
        type_ids = _raw_type_ids()
        t24 = _canonical_t24()
        robot_pose = bimanual_t24_to_pose(t24)
        path_score = torch.linspace(-2.0, 2.0, 500)[None]
        valid_mask = torch.ones(1, 500, dtype=torch.bool)

        first = build_joint_hard_mode_adapter(
            robot_pose,
            path_score,
            type_ids,
            valid_mask,
            ["obj/scene"],
            config,
        )
        second = build_joint_hard_mode_adapter(
            robot_pose,
            path_score,
            type_ids,
            valid_mask,
            ["obj/scene"],
            config,
        )

        self.assertEqual(first["raw_type_counts"].tolist(), [RAW_COUNTS])
        self.assertEqual(first["candidate_count"].tolist(), [[0, 3, 20, 100, 100]])
        self.assertEqual(first["unique_sample_count"].tolist(), [[0, 3, 20, 20, 20]])
        self.assertEqual(first["zero_support_mask"].tolist(), [[True, False, False, False, False]])
        self.assertTrue(first["export_replacement_mask"][0, 0].all())
        self.assertFalse(first["export_replacement_mask"][0, 1, :3].any())
        self.assertTrue(first["export_replacement_mask"][0, 1, 3:].all())
        self.assertFalse(first["export_replacement_mask"][0, 2:].any())
        self.assertTrue(torch.equal(first["selected_raw_indices"], second["selected_raw_indices"]))
        self.assertTrue(torch.equal(first["budget_scores"], second["budget_scores"]))
        for type_index, type_id in enumerate(range(1, 6)):
            indices = first["selected_raw_indices"][0, type_index]
            supported = indices >= 0
            if supported.any():
                self.assertTrue((type_ids[0, indices[supported]] == type_id).all())

    def test_budget_frequency_is_not_renormalized_after_validity_filtering(self):
        config = _joint_config()
        type_ids = _raw_type_ids()
        robot_pose = bimanual_t24_to_pose(_canonical_t24())
        path_score = torch.linspace(-2.0, 2.0, 500)[None]
        valid_mask = torch.ones(1, 500, dtype=torch.bool)
        valid_mask[0, -50:] = False

        adapter = build_joint_hard_mode_adapter(
            robot_pose,
            path_score,
            type_ids,
            valid_mask,
            ["obj/scene"],
            config,
        )

        self.assertEqual(adapter["raw_type_counts"].tolist(), [RAW_COUNTS])
        self.assertEqual(adapter["valid_sample_count"].tolist(), [[0, 3, 20, 100, 327]])
        self.assertTrue(
            torch.allclose(
                adapter["budget_scores"],
                torch.tensor([[value / 500.0 for value in RAW_COUNTS]]),
            )
        )

    def test_single_scene_export_writes_raw_artifact_and_consumer_schema(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            scene_source = tmp_path / "source_scene.npy"
            pc_source = tmp_path / "source_pc.npy"
            np.save(scene_source, {"scene_id": "obj/scene", "object": {"name": "obj"}})
            pc_source.touch()
            output_dir = str(tmp_path / "export")
            config = _joint_config(object_path=str(tmp_path / "objects"))
            config.task.output_dir = output_dir
            data = {
                "scene_path": [str(scene_source)],
                "pc_path": [str(pc_source)],
                "pc_centroid": torch.tensor([[[0.5, -0.25, 0.1]]], dtype=torch.float32),
                "grasp_type_id": torch.tensor([0]),
            }
            checkpoint_meta = {
                "joint_checkpoint_sha256": "a" * 64,
                "checkpoint_path": "joint.pth",
                "checkpoint_iter": 10000,
                "uses_independent_models": False,
                "score_checkpoint_path": "joint.pth",
                "score_checkpoint_iter": 10000,
                "pose_checkpoint_path": "joint.pth",
                "pose_checkpoint_iter": 10000,
            }
            model = FakeJointModel()
            scene_dir = export_scene_dir(output_dir, config)

            with mock.patch.object(export_module, "create_test_dataloader", return_value=[data]):
                score_lines, scene_index = sample_joint_scene_scores_and_poses(
                    config,
                    model,
                    {"obj": "all"},
                    output_dir,
                    scene_dir,
                    checkpoint_meta,
                )

            self.assertEqual(len(score_lines), 1)
            self.assertEqual(len(scene_index), 1)
            saved = np.load(scene_index[0]["scene_file"], allow_pickle=True).item()
            self.assertEqual(saved["factorization"], JOINT_FACTORIZATION)
            self.assertEqual(saved["index_mcp_pos"].shape, (5, 20, 2, 3))
            self.assertEqual(saved["wrist_quat"].shape, (5, 20, 2, 4))
            self.assertEqual(saved["active_hand_mask"].shape, (5, 20, 2))
            self.assertTrue(np.allclose(saved["budget_scores"], np.asarray(RAW_COUNTS) / 500.0))
            self.assertTrue(saved["joint_zero_support_mask"][0])
            raw_path = Path(output_dir) / saved["joint_raw_artifact_relative_path"]
            self.assertTrue(raw_path.is_file())
            with np.load(raw_path, allow_pickle=False) as raw:
                self.assertEqual(raw["type_ids"].shape, (500,))
                self.assertEqual(raw["canonical_t24"].shape, (500, 24))
                self.assertTrue(((raw["type_ids"] >= 1) & (raw["type_ids"] <= 5)).all())
            validate_scene_export_completeness(saved, config, checkpoint_meta)
            self.assertEqual(model.placeholder_ids[0].tolist(), [0])


if __name__ == "__main__":
    unittest.main()
