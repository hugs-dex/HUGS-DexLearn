import json
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
    REAL_GRASP_TYPE_IDS,
    _stable_reverse_seed,
    build_manifest,
    build_reverse_conditional_adapter,
    build_scene_export_record,
    collect_complete_scene_exports,
    export_scene_dir,
    reverse_raw_joint_type_ids,
    reverse_weighted_resample_indices,
    sample_reverse_scene_scores_and_poses,
    scene_file_path,
    scene_index_from_summary,
    scene_summary_from_data,
    validate_reverse_shared_pool,
    validate_scene_export_completeness,
    write_obj_human_prior_export,
)


def _reverse_config(object_path="objects"):
    return OmegaConf.create(
        {
            "seed": 17,
            "algo": {
                "human": True,
                "factorization": "reverse_T_to_C",
                "test_grasp_num": 4,
                "test_topk": 2,
                "sample_selection": {
                    "enabled": True,
                    "scope": "global",
                    "mode": "prob_pose",
                    "intermediate_topk": 3,
                    "translation_scale_m": 0.05,
                    "rotation_weight": 1.0,
                },
                "reverse_sampling": {
                    "marginal_pool_size": 8,
                    "conditional_candidate_num": 4,
                    "resampling_policy": "weighted_without_replacement",
                    "resampling_seed": 29,
                    "include_raw_pool": True,
                    "ess_warning_threshold": 0.0,
                },
                "posterior_contract": {
                    "training_pose_source": "gt_clean_canonical_t24",
                    "inference_pose_source": "generated_marginal_canonical_t24",
                    "train_inference_gap": "gt_pose_to_generated_pose",
                },
                "model": {"name": "MarginalPoseDiffusionModel"},
            },
            "data": {"hand_pos_source": "index_mcp"},
            "test_data": {
                "test_split": "all",
                "object_path": object_path,
            },
            "task": {
                "samples_per_type": 2,
                "robot_name": "leap",
                "robot_size": 1.0,
                "quat_norm_tol": 1e-3,
                "include_log_prob": False,
                "include_grasp_pose": False,
                "object_splits": ["all"],
                "score_grasp_types": ["0_any"],
                "pose_grasp_types": [
                    "1_right_two",
                    "2_right_three",
                    "3_right_full",
                    "4_both_three",
                    "5_both_full",
                ],
            },
        }
    )


def _candidate_pose(batch_size=2, pool_size=8):
    pose = torch.zeros(batch_size, pool_size, 1, 14, dtype=torch.float32)
    pose[:, :, 0, 0] = torch.arange(pool_size, dtype=torch.float32)
    pose[:, :, 0, 3] = 1.0
    pose[:, :, 0, 10] = 1.0
    return pose


def _canonical_t24_pool(batch_size=1, pool_size=8):
    t24 = torch.zeros(batch_size, pool_size, 24, dtype=torch.float32)
    identity = torch.eye(3, dtype=torch.float32).reshape(9)
    t24[..., 0:9] = identity
    t24[..., 12:21] = identity
    t24[..., 9] = torch.arange(pool_size, dtype=torch.float32) / 100.0
    t24[..., 21] = -torch.arange(pool_size, dtype=torch.float32) / 100.0
    return t24


class FakeMarginalModel(torch.nn.Module):
    def sample_with_t24(self, data, sample_num):
        batch_size = len(data["scene_path"])
        canonical_t24 = _canonical_t24_pool(batch_size, sample_num)
        robot_pose = bimanual_t24_to_pose(canonical_t24)
        log_prob = torch.linspace(-1.0, 1.0, sample_num).expand(batch_size, -1)
        return canonical_t24, robot_pose, log_prob


class FakePosteriorModel(torch.nn.Module):
    def posterior_probabilities(self, data, canonical_t24):
        logits = torch.stack(
            [
                canonical_t24[..., 9],
                -canonical_t24[..., 9],
                canonical_t24[..., 21],
                -canonical_t24[..., 21],
                torch.zeros_like(canonical_t24[..., 9]),
            ],
            dim=-1,
        )
        return torch.softmax(logits, dim=-1)


def _pose_records(samples_per_type=2):
    records = {}
    for type_id in REAL_GRASP_TYPE_IDS:
        index_mcp_pos = np.zeros((samples_per_type, 2, 3), dtype=np.float32)
        wrist_quat = np.zeros((samples_per_type, 2, 4), dtype=np.float32)
        wrist_quat[..., 0] = 1.0
        active_hand_mask = np.zeros((samples_per_type, 2), dtype=bool)
        active_hand_mask[:, 0] = True
        if type_id >= 4:
            active_hand_mask[:, 1] = True
        records[type_id] = {
            "index_mcp_pos": index_mcp_pos,
            "wrist_quat": wrist_quat,
            "active_hand_mask": active_hand_mask,
        }
    return records


def _attach_reverse_fields(scene_data):
    pool_size = 8
    type_num = 5
    candidate_num = 4
    selected_num = 2
    posterior = np.full((pool_size, type_num), 0.2, dtype=np.float32)
    scene_data.update(
        {
            "reverse_sampling_order": np.asarray(
                ["marginal_T", "posterior_C", "budget_integral", "resample", "select", "export"]
            ),
            "reverse_marginal_pool_size": np.int64(pool_size),
            "reverse_conditional_candidate_num": np.int64(candidate_num),
            "reverse_resampling_policy": "weighted_without_replacement",
            "reverse_resampling_base_seed": np.int64(29),
            "reverse_resampling_seeds": np.asarray(
                [_stable_reverse_seed(29, "obj/scene", type_id) for type_id in REAL_GRASP_TYPE_IDS],
                dtype=np.int64,
            ),
            "reverse_importance_ess": np.full((type_num,), pool_size, dtype=np.float32),
            "reverse_resampled_pool_indices": np.tile(np.arange(candidate_num), (type_num, 1)),
            "reverse_selected_resample_indices": np.tile(np.arange(selected_num), (type_num, 1)),
            "reverse_selected_pool_indices": np.tile(np.arange(selected_num), (type_num, 1)),
            "reverse_selected_centered_t24": np.zeros((type_num, selected_num, 24), dtype=np.float32),
            "reverse_selected_marginal_log_prob": np.zeros((type_num, selected_num), dtype=np.float32),
            "reverse_selected_posterior_probability": np.full(
                (type_num, selected_num), 0.2, dtype=np.float32
            ),
            "reverse_selected_conditional_score": np.full(
                (type_num, selected_num), np.log(0.2), dtype=np.float32
            ),
            "reverse_raw_pool_included": np.bool_(True),
            "reverse_raw_centered_t24": np.zeros((pool_size, 24), dtype=np.float32),
            "reverse_raw_marginal_log_prob": np.zeros((pool_size,), dtype=np.float32),
            "reverse_raw_posterior_probability": posterior,
            "reverse_raw_sampled_type_ids": np.ones((pool_size,), dtype=np.int64),
            "reverse_score_checkpoint_sha256": "a" * 64,
            "reverse_pose_checkpoint_sha256": "b" * 64,
        }
    )
    return scene_data


class ReverseExportTest(unittest.TestCase):
    def test_weighted_resampling_is_deterministic_unique_and_handles_underflow(self):
        posterior = torch.zeros(1, 8, 5)
        posterior[:, 0, :] = 1.0

        first_indices, first_seeds = reverse_weighted_resample_indices(
            posterior,
            ["obj/scene"],
            candidate_num=4,
            base_seed=29,
        )
        second_indices, second_seeds = reverse_weighted_resample_indices(
            posterior,
            ["obj/scene"],
            candidate_num=4,
            base_seed=29,
        )

        self.assertTrue(torch.equal(first_indices, second_indices))
        self.assertTrue(np.array_equal(first_seeds, second_seeds))
        for type_indices in first_indices[0]:
            self.assertEqual(len(torch.unique(type_indices)), 4)
            self.assertIn(0, type_indices.tolist())

    def test_raw_joint_mode_sampling_is_scene_stable_and_in_range(self):
        posterior = torch.softmax(torch.arange(80, dtype=torch.float32).reshape(2, 8, 5), dim=-1)

        first = reverse_raw_joint_type_ids(posterior, ["obj/a", "obj/b"], base_seed=29)
        second = reverse_raw_joint_type_ids(posterior, ["obj/a", "obj/b"], base_seed=29)

        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first.shape, (2, 8))
        self.assertTrue(((first >= 1) & (first <= 5)).all())

    def test_adapter_preserves_shared_pool_indices_scores_and_ess(self):
        torch.manual_seed(11)
        config = _reverse_config()
        posterior = torch.softmax(torch.randn(2, 8, 5), dim=-1)
        marginal_log_prob = torch.linspace(-2.0, 1.0, 8).repeat(2, 1)
        robot_pose = _candidate_pose()

        adapter = build_reverse_conditional_adapter(
            robot_pose,
            marginal_log_prob,
            posterior,
            ["obj/a", "obj/b"],
            config,
        )
        repeated = build_reverse_conditional_adapter(
            robot_pose,
            marginal_log_prob,
            posterior,
            ["obj/a", "obj/b"],
            config,
        )

        self.assertEqual(tuple(adapter["robot_pose"].shape), (2, 5, 2, 1, 14))
        self.assertEqual(tuple(adapter["resampled_pool_indices"].shape), (2, 5, 4))
        self.assertEqual(tuple(adapter["selected_pool_indices"].shape), (2, 5, 2))
        self.assertTrue(torch.equal(adapter["selected_pool_indices"], repeated["selected_pool_indices"]))
        self.assertTrue(
            torch.equal(
                adapter["robot_pose"][..., 0, 0].long(),
                adapter["selected_pool_indices"],
            )
        )

        posterior_by_type = posterior.permute(0, 2, 1)
        selected_posterior = torch.gather(posterior_by_type, 2, adapter["selected_pool_indices"])
        selected_marginal = torch.gather(
            marginal_log_prob[:, None, :].expand(-1, 5, -1),
            2,
            adapter["selected_pool_indices"],
        )
        expected_score = selected_marginal + torch.log(selected_posterior.clamp_min(1e-12))
        self.assertTrue(torch.allclose(adapter["conditional_score"], expected_score))

        expected_ess = posterior.sum(dim=1).square() / posterior.square().sum(dim=1)
        self.assertTrue(torch.allclose(adapter["importance_ess"], expected_ess))

    def test_shared_pool_contract_rejects_non_categorical_posterior(self):
        canonical_t24 = torch.zeros(1, 8, 24)
        robot_pose = _candidate_pose(batch_size=1)
        marginal_log_prob = torch.zeros(1, 8)
        posterior = torch.full((1, 8, 5), 0.2)

        validate_reverse_shared_pool(canonical_t24, robot_pose, marginal_log_prob, posterior, 8)

        invalid_posterior = posterior.clone()
        invalid_posterior[0, 0] = 0.3
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            validate_reverse_shared_pool(
                canonical_t24,
                robot_pose,
                marginal_log_prob,
                invalid_posterior,
                8,
            )

    def test_reverse_completeness_checks_factorization_and_checkpoint_hashes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene_path = Path(tmp_dir) / "scene.npy"
            pc_path = Path(tmp_dir) / "pc.npy"
            scene_path.touch()
            pc_path.touch()
            config = _reverse_config(object_path=str(Path(tmp_dir) / "objects"))
            score_record = {
                "scene_id": "obj/scene",
                "object_id": "obj",
                "split": "all",
                "scene_path": str(scene_path),
                "pc_path": str(pc_path),
                "budget_scores": np.full((5,), 0.2, dtype=np.float32),
            }
            scene_data = _attach_reverse_fields(build_scene_export_record(score_record, _pose_records(), config))
            checkpoint_meta = {
                "score_checkpoint_sha256": "a" * 64,
                "pose_checkpoint_sha256": "b" * 64,
            }

            validate_scene_export_completeness(scene_data, config, checkpoint_meta)

            wrong_checkpoint = dict(checkpoint_meta)
            wrong_checkpoint["pose_checkpoint_sha256"] = "c" * 64
            with self.assertRaisesRegex(ValueError, "different marginal checkpoint"):
                validate_scene_export_completeness(scene_data, config, wrong_checkpoint)

            wrong_factorization = dict(scene_data)
            wrong_factorization["factorization"] = "proposed_C_to_T"
            with self.assertRaisesRegex(ValueError, "factorization"):
                validate_scene_export_completeness(wrong_factorization, config, checkpoint_meta)

    def test_manifest_records_reverse_contract(self):
        config = _reverse_config()
        checkpoint_meta = {
            "checkpoint_path": "pose.pth",
            "checkpoint_iter": 10000,
            "uses_independent_models": True,
            "score_checkpoint_path": "posterior.pth",
            "score_checkpoint_iter": 300,
            "score_checkpoint_sha256": "a" * 64,
            "score_ckpt": "000300",
            "pose_checkpoint_path": "pose.pth",
            "pose_checkpoint_iter": 10000,
            "pose_checkpoint_sha256": "b" * 64,
            "pose_ckpt": "010000",
            "model_roles": {
                "score": "pose_conditioned_type_posterior",
                "pose": "marginal_pose_generator",
            },
            "score_model_config": {"name": "PoseConditionedTypeModel"},
            "pose_model_config": {"name": "MarginalPoseDiffusionModel"},
        }

        manifest = build_manifest(config, checkpoint_meta)

        self.assertEqual(manifest["factorization"], "reverse_T_to_C")
        self.assertEqual(manifest["reverse_sampling"]["marginal_pool_size"], 8)
        self.assertEqual(manifest["pose_candidate_num"], 4)
        self.assertEqual(manifest["posterior_contract"]["training_pose_source"], "gt_clean_canonical_t24")
        self.assertEqual(manifest["score_checkpoint_sha256"], "a" * 64)
        self.assertEqual(manifest["pose_checkpoint_sha256"], "b" * 64)

    def test_complete_reverse_scene_can_be_reused_without_losing_summary_provenance(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            object_path = tmp_path / "objects"
            object_path.mkdir()
            output_dir = tmp_path / "export"
            config = _reverse_config(object_path=str(object_path))
            scene_path = tmp_path / "scene.npy"
            pc_path = tmp_path / "pc.npy"
            scene_path.touch()
            pc_path.touch()
            score_record = {
                "scene_id": "obj/scene",
                "object_id": "obj",
                "split": "all",
                "scene_path": str(scene_path),
                "pc_path": str(pc_path),
                "budget_scores": np.full((5,), 0.2, dtype=np.float32),
            }
            scene_data = _attach_reverse_fields(build_scene_export_record(score_record, _pose_records(), config))
            checkpoint_meta = {
                "score_checkpoint_sha256": "a" * 64,
                "pose_checkpoint_sha256": "b" * 64,
            }
            saved_scene_file = scene_file_path(export_scene_dir(str(output_dir), config), "obj/scene")
            Path(saved_scene_file).parent.mkdir(parents=True)
            np.save(saved_scene_file, scene_data)

            complete = collect_complete_scene_exports(str(output_dir), config, checkpoint_meta)

            self.assertEqual(set(complete), {"obj/scene"})
            reused_data, reused_path = complete["obj/scene"]
            summary = scene_summary_from_data(reused_data, reused_path)
            index = scene_index_from_summary(summary)
            paths = write_obj_human_prior_export(
                [summary],
                [index],
                str(output_dir),
                {"factorization": "reverse_T_to_C"},
                config,
            )
            self.assertEqual(paths["scene_count"], 1)
            with open(paths["score_jsonl"], "r", encoding="utf-8") as score_file:
                score_line = json.loads(score_file.readline())
            self.assertEqual(score_line["factorization"], "reverse_T_to_C")
            self.assertEqual(score_line["scene_id"], "obj/scene")

    def test_single_scene_reverse_export_path_writes_consumer_schema(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            scene_source = tmp_path / "source_scene.npy"
            pc_source = tmp_path / "source_pc.npy"
            np.save(scene_source, {"scene_id": "obj/scene", "object": {"name": "obj"}})
            pc_source.touch()
            config = _reverse_config(object_path=str(tmp_path / "objects"))
            data = {
                "scene_path": [str(scene_source)],
                "pc_path": [str(pc_source)],
                "pc_centroid": torch.tensor([[[0.5, -0.25, 0.1]]], dtype=torch.float32),
            }
            checkpoint_meta = {
                "score_checkpoint_sha256": "a" * 64,
                "pose_checkpoint_sha256": "b" * 64,
            }
            scene_dir = str(tmp_path / "export" / "objects" / "leap")

            with mock.patch.object(export_module, "create_test_dataloader", return_value=[data]):
                score_lines, scene_index = sample_reverse_scene_scores_and_poses(
                    config,
                    FakePosteriorModel(),
                    FakeMarginalModel(),
                    {"obj": "all"},
                    scene_dir,
                    checkpoint_meta,
                )

            self.assertEqual(len(score_lines), 1)
            self.assertEqual(len(scene_index), 1)
            saved = np.load(scene_index[0]["scene_file"], allow_pickle=True).item()
            self.assertEqual(saved["factorization"], "reverse_T_to_C")
            self.assertEqual(saved["index_mcp_pos"].shape, (5, 2, 2, 3))
            self.assertEqual(saved["wrist_quat"].shape, (5, 2, 2, 4))
            self.assertEqual(saved["active_hand_mask"].shape, (5, 2, 2))
            self.assertTrue(saved["active_hand_mask"][:3, :, 0].all())
            self.assertFalse(saved["active_hand_mask"][:3, :, 1].any())
            self.assertTrue(saved["active_hand_mask"][3:, :, :].all())
            self.assertTrue(
                np.allclose(
                    saved["budget_scores"],
                    saved["reverse_raw_posterior_probability"].mean(axis=0),
                )
            )


if __name__ == "__main__":
    unittest.main()
