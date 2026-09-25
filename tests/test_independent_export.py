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
    INDEPENDENT_FACTORIZATION,
    _stable_independent_seed,
    build_independent_mode_blind_adapter,
    build_manifest,
    export_scene_dir,
    independent_pairing_indices,
    independent_sampling_config,
    load_independent_export_models,
    sample_independent_scene_scores_and_poses,
    validate_independent_scene_export,
    validate_scene_export_completeness,
)


def _independent_config(object_path="objects"):
    return OmegaConf.create(
        {
            "seed": 17,
            "exp_name": "independent_debug",
            "data_name": "humanMulti",
            "algo_name": "humanMultiIndependent",
            "output_folder": "output",
            "wandb": {"id": "humanMulti_humanMultiIndependent_independent_debug"},
            "algo": {
                "human": True,
                "factorization": INDEPENDENT_FACTORIZATION,
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
                "independent_sampling": {
                    "pool_size": 20,
                    "candidate_num": 4,
                    "base_seed": 31,
                    "mode_seed": 41,
                    "pose_seed": 43,
                    "pairing_seed": 47,
                    "adapter_seed": 53,
                    "include_raw_pool": True,
                    "raw_artifact_format": "npz",
                    "adapter_policy": "mode_blind_stratified",
                    "inactive_left_translation": [0.0, 0.0, -0.5],
                    "inactive_left_translation_tolerance_m": 0.05,
                    "inactive_left_rotation_tolerance_deg": 15.0,
                },
                "models": {
                    "mode_marginal": {"name": "ObjectModeMarginalModel"},
                    "pose_marginal": {"name": "MarginalPoseDiffusionModel"},
                },
                "model": {
                    "name": "MarginalPoseDiffusionModel",
                    "backbone": {"name": "FakeBackbone", "voxel_size": 0.005},
                },
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
                "pose_grasp_types": [],
                "output_dir": None,
            },
        }
    )


def _canonical_t24(batch_size=1, pool_size=20):
    t24 = torch.zeros(batch_size, pool_size, 24, dtype=torch.float32)
    identity = torch.eye(3, dtype=torch.float32).reshape(9)
    t24[..., 0:9] = identity
    t24[..., 12:21] = identity
    t24[..., 9] = torch.arange(pool_size, dtype=torch.float32) / 100.0
    t24[..., 10] = torch.arange(batch_size, dtype=torch.float32)[:, None] / 10.0
    t24[..., 21:24] = torch.tensor([0.0, 0.0, -0.5])
    t24[:, 1::2, 21] = torch.arange(1, pool_size, 2, dtype=torch.float32) / 50.0
    t24[:, 1::2, 23] = 0.1
    return t24


class FakeModeMarginal(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seeds = []

    def sample_modes(self, data, sample_num, seed):
        self.seeds.append(list(seed))
        probabilities = torch.tensor(
            [[0.05, 0.15, 0.2, 0.25, 0.35]],
            dtype=torch.float32,
        ).expand(len(data["scene_path"]), -1)
        rows = []
        for batch_index, scene_seed in enumerate(seed):
            generator = torch.Generator(device=probabilities.device)
            generator.manual_seed(int(scene_seed))
            rows.append(
                torch.multinomial(
                    probabilities[batch_index],
                    sample_num,
                    replacement=True,
                    generator=generator,
                )
                + 1
            )
        return {
            "mode_probabilities": probabilities,
            "sampled_type_ids": torch.stack(rows),
            "sampling_seeds": torch.as_tensor(seed, dtype=torch.int64),
        }


class FakePoseMarginal(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seeds = []

    def sample_with_t24(self, data, sample_num, seed=None):
        self.seeds.append(list(seed))
        canonical_t24 = _canonical_t24(len(data["scene_path"]), sample_num)
        robot_pose = bimanual_t24_to_pose(canonical_t24)
        log_prob = torch.linspace(-2.0, 2.0, sample_num).expand(len(data["scene_path"]), -1)
        return canonical_t24, robot_pose, log_prob


class IndependentExportTest(unittest.TestCase):
    def test_checkpoint_loader_uses_heterogeneous_nested_models(self):
        config = _independent_config()
        config.task.score_ckpt = "000300"
        config.task.pose_ckpt = "010000"
        mode_model = FakeModeMarginal()
        pose_model = FakePoseMarginal()

        with mock.patch.object(
            export_module,
            "load_export_model",
            side_effect=[
                (mode_model, "mode.pth", 300),
                (pose_model, "pose.pth", 10000),
            ],
        ) as load_model, mock.patch.object(
            export_module,
            "checkpoint_sha256",
            side_effect=lambda path: "a" * 64 if path == "mode.pth" else "b" * 64,
        ):
            loaded_mode, loaded_pose, metadata = load_independent_export_models(config)

        self.assertIs(loaded_mode, mode_model)
        self.assertIs(loaded_pose, pose_model)
        mode_config = load_model.call_args_list[0].args[0]
        pose_config = load_model.call_args_list[1].args[0]
        self.assertEqual(mode_config.algo.model.name, "ObjectModeMarginalModel")
        self.assertEqual(pose_config.algo.model.name, "MarginalPoseDiffusionModel")
        self.assertEqual(mode_config.exp_name, "independent_debug_mode_marginal")
        self.assertEqual(pose_config.exp_name, "independent_debug_pose_marginal")
        self.assertEqual(metadata["score_checkpoint_sha256"], "a" * 64)
        self.assertEqual(metadata["pose_checkpoint_sha256"], "b" * 64)
        self.assertEqual(metadata["model_roles"]["score"], "object_only_mode_marginal")

    def test_sampling_config_requires_exact_mode_blind_blocks(self):
        config = _independent_config()
        sampling = independent_sampling_config(config)

        self.assertEqual(sampling["pool_size"], 20)
        self.assertEqual(sampling["candidate_num"], 4)
        self.assertEqual(sampling["adapter_policy"], "mode_blind_stratified")

        config.algo.independent_sampling.pool_size = 19
        with self.assertRaisesRegex(ValueError, "stratification"):
            independent_sampling_config(config)

    def test_pairing_stream_changes_only_pose_source_indices(self):
        first_mode, first_pose, first_seeds = independent_pairing_indices(
            ["obj/a", "obj/b"],
            pool_size=20,
            base_seed=47,
        )
        second_mode, second_pose, second_seeds = independent_pairing_indices(
            ["obj/a", "obj/b"],
            pool_size=20,
            base_seed=59,
        )

        self.assertTrue(torch.equal(first_mode, second_mode))
        self.assertFalse(torch.equal(first_pose, second_pose))
        self.assertFalse(np.array_equal(first_seeds, second_seeds))
        for row in first_pose:
            self.assertEqual(sorted(row.tolist()), list(range(20)))

    def test_mode_blind_adapter_ignores_mode_stream_and_uses_adapter_stream(self):
        config = _independent_config()
        t24 = _canonical_t24()
        pose = bimanual_t24_to_pose(t24)
        log_prob = torch.linspace(-2.0, 2.0, 20)[None]

        first = build_independent_mode_blind_adapter(pose, log_prob, ["obj/scene"], config)
        config.algo.independent_sampling.mode_seed = 999
        second = build_independent_mode_blind_adapter(pose, log_prob, ["obj/scene"], config)
        config.algo.independent_sampling.adapter_seed = 997
        changed_adapter = build_independent_mode_blind_adapter(pose, log_prob, ["obj/scene"], config)

        self.assertTrue(torch.equal(first["block_raw_pose_indices"], second["block_raw_pose_indices"]))
        self.assertTrue(torch.equal(first["selected_raw_pose_indices"], second["selected_raw_pose_indices"]))
        self.assertFalse(
            torch.equal(first["block_raw_pose_indices"], changed_adapter["block_raw_pose_indices"])
        )
        self.assertEqual(tuple(first["selected_robot_pose"].shape), (1, 5, 2, 1, 14))
        self.assertEqual(
            sorted(first["block_raw_pose_indices"].reshape(-1).tolist()),
            list(range(20)),
        )

    def test_single_scene_export_preserves_raw_pairs_and_consumer_schema(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            scene_source = tmp_path / "source_scene.npy"
            pc_source = tmp_path / "source_pc.npy"
            np.save(scene_source, {"scene_id": "obj/scene", "object": {"name": "obj"}})
            pc_source.touch()
            output_dir = str(tmp_path / "export")
            config = _independent_config(object_path=str(tmp_path / "objects"))
            config.task.output_dir = output_dir
            data = {
                "scene_path": [str(scene_source)],
                "pc_path": [str(pc_source)],
                "pc_centroid": torch.tensor([[[0.5, -0.25, 0.1]]], dtype=torch.float32),
                "grasp_type_id": torch.tensor([0]),
            }
            checkpoint_meta = {
                "score_checkpoint_sha256": "a" * 64,
                "pose_checkpoint_sha256": "b" * 64,
                "checkpoint_path": "pose.pth",
                "checkpoint_iter": 10000,
                "uses_independent_models": True,
                "score_checkpoint_path": "mode.pth",
                "score_checkpoint_iter": 300,
                "score_ckpt": "000300",
                "pose_checkpoint_path": "pose.pth",
                "pose_checkpoint_iter": 10000,
                "pose_ckpt": "010000",
            }
            mode_model = FakeModeMarginal()
            pose_model = FakePoseMarginal()
            scene_dir = export_scene_dir(output_dir, config)

            with mock.patch.object(export_module, "create_test_dataloader", return_value=[data]):
                score_lines, scene_index = sample_independent_scene_scores_and_poses(
                    config,
                    mode_model,
                    pose_model,
                    {"obj": "all"},
                    output_dir,
                    scene_dir,
                    checkpoint_meta,
                )

            self.assertEqual(len(score_lines), 1)
            self.assertEqual(len(scene_index), 1)
            saved = np.load(scene_index[0]["scene_file"], allow_pickle=True).item()
            self.assertEqual(saved["factorization"], INDEPENDENT_FACTORIZATION)
            self.assertEqual(saved["index_mcp_pos"].shape, (5, 2, 2, 3))
            self.assertEqual(saved["wrist_quat"].shape, (5, 2, 2, 4))
            self.assertEqual(saved["active_hand_mask"].shape, (5, 2, 2))
            self.assertTrue(
                np.allclose(saved["budget_scores"], np.asarray([0.05, 0.15, 0.2, 0.25, 0.35]))
            )
            self.assertFalse(bool(saved["independent_pair_compatibility_used_for_selection"]))
            self.assertFalse(bool(saved["independent_mode_conditioned_pose_generation"]))
            raw_path = Path(output_dir) / saved["independent_raw_artifact_relative_path"]
            self.assertTrue(raw_path.is_file())
            with np.load(raw_path, allow_pickle=False) as raw:
                permutation = raw["independent_pairing_permutation"]
                marginal_t24 = raw["independent_pose_marginal_centered_t24"]
                self.assertEqual(raw["independent_raw_type_ids"].shape, (20,))
                self.assertEqual(raw["independent_raw_centered_t24"].shape, (20, 24))
                self.assertTrue(
                    np.allclose(raw["independent_raw_centered_t24"], marginal_t24[permutation])
                )
                self.assertFalse(bool(raw["compatibility_used_for_selection"]))
            validate_independent_scene_export(saved, config, checkpoint_meta, output_dir=output_dir)
            validate_scene_export_completeness(saved, config, checkpoint_meta)
            self.assertEqual(
                mode_model.seeds[0],
                [_stable_independent_seed(41, "obj/scene", "mode")],
            )
            self.assertEqual(
                pose_model.seeds[0],
                [_stable_independent_seed(43, "obj/scene", "pose")],
            )

    def test_manifest_records_independent_contract(self):
        config = _independent_config()
        checkpoint_meta = {
            "checkpoint_path": "pose.pth",
            "checkpoint_iter": 10000,
            "checkpoint_sha256": "b" * 64,
            "uses_independent_models": True,
            "score_checkpoint_path": "mode.pth",
            "score_checkpoint_iter": 300,
            "score_checkpoint_sha256": "a" * 64,
            "score_ckpt": "000300",
            "pose_checkpoint_path": "pose.pth",
            "pose_checkpoint_iter": 10000,
            "pose_checkpoint_sha256": "b" * 64,
            "pose_ckpt": "010000",
            "model_roles": {
                "score": "object_only_mode_marginal",
                "pose": "object_only_pose_marginal",
            },
            "score_model_config": {"name": "ObjectModeMarginalModel"},
            "pose_model_config": {"name": "MarginalPoseDiffusionModel"},
        }

        manifest = build_manifest(config, checkpoint_meta)

        self.assertEqual(manifest["factorization"], INDEPENDENT_FACTORIZATION)
        self.assertEqual(manifest["independent_sampling"]["pool_size"], 20)
        self.assertEqual(manifest["pose_candidate_num"], 4)
        self.assertEqual(manifest["score_checkpoint_sha256"], "a" * 64)
        self.assertEqual(manifest["pose_checkpoint_sha256"], "b" * 64)
        self.assertEqual(manifest["consumer_contract"]["adapter"], "mode_blind_stratified")
        self.assertFalse(manifest["consumer_contract"]["compatibility_correction"])


if __name__ == "__main__":
    unittest.main()
