import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from dexlearn.task.obj_human_prior_export import (
    REAL_GRASP_TYPE_IDS,
    build_manifest,
    build_scene_export_record,
    export_sample_selection_metadata,
    validate_scene_export_completeness,
)


def _build_config(*, enabled=True, scope="global", mode="prob_pose"):
    return OmegaConf.create(
        {
            "seed": 7,
            "algo": {
                "human": True,
                "test_grasp_num": 100,
                "test_topk": 20,
                "sample_selection": {
                    "enabled": enabled,
                    "scope": scope,
                    "mode": mode,
                    "intermediate_topk": 50,
                    "translation_scale_m": 0.05,
                    "rotation_weight": 1.0,
                },
                "model": {
                    "name": "HierarchicalTypeObjectiveModel",
                    "type_objective": "ce",
                },
            },
            "data": {"hand_pos_source": "index_mcp"},
            "test_data": {
                "test_split": "all",
                "object_path": "objects",
            },
            "task": {
                "samples_per_type": 20,
                "robot_name": "shadow_hand",
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


def _pose_records(samples_per_type):
    records = {}
    for grasp_type_id in REAL_GRASP_TYPE_IDS:
        index_mcp_pos = np.zeros((samples_per_type, 2, 3), dtype=np.float32)
        wrist_quat = np.zeros((samples_per_type, 2, 4), dtype=np.float32)
        wrist_quat[..., 0] = 1.0
        active_hand_mask = np.zeros((samples_per_type, 2), dtype=bool)
        active_hand_mask[:, 0] = True
        if grasp_type_id >= 4:
            active_hand_mask[:, 1] = True
        records[grasp_type_id] = {
            "index_mcp_pos": index_mcp_pos,
            "wrist_quat": wrist_quat,
            "active_hand_mask": active_hand_mask,
        }
    return records


class ExportSampleSelectionMetadataTest(unittest.TestCase):
    def test_six_field_selection_config_is_exported(self):
        metadata = export_sample_selection_metadata(_build_config())

        self.assertTrue(metadata["sample_selection_enabled"])
        self.assertEqual(metadata["sample_selection_scope"], "global")
        self.assertEqual(metadata["sample_selection_mode"], "prob_pose")
        self.assertEqual(metadata["sample_selection_intermediate_topk"], 50)
        self.assertEqual(metadata["sample_selection_translation_scale_m"], 0.05)
        self.assertEqual(metadata["sample_selection_rotation_weight"], 1.0)

    def test_scene_and_manifest_record_enabled_and_scope(self):
        config = _build_config(enabled=False, scope="per_type", mode="random")
        checkpoint_meta = {
            "checkpoint_path": "pose.pth",
            "checkpoint_iter": 100,
            "uses_independent_models": True,
            "score_checkpoint_path": "score.pth",
            "score_checkpoint_iter": 10,
            "score_ckpt": "000010",
            "pose_checkpoint_path": "pose.pth",
            "pose_checkpoint_iter": 100,
            "pose_ckpt": "000100",
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            scene_path = Path(tmp_dir) / "scene.npy"
            pc_path = Path(tmp_dir) / "pc.npy"
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
            scene_data = build_scene_export_record(score_record, _pose_records(20), config)

            self.assertFalse(bool(scene_data["sample_selection_enabled"]))
            self.assertEqual(scene_data["sample_selection_scope"], "per_type")
            manifest = build_manifest(config, checkpoint_meta)
            self.assertFalse(manifest["sample_selection"]["enabled"])
            self.assertEqual(manifest["sample_selection"]["scope"], "per_type")

    def test_manifest_records_explicit_scene_list_hash(self):
        config = _build_config()
        checkpoint_meta = {
            "checkpoint_path": "pose.pth",
            "checkpoint_iter": 100,
            "uses_independent_models": True,
            "score_checkpoint_path": "score.pth",
            "score_checkpoint_iter": 10,
            "score_ckpt": "000010",
            "pose_checkpoint_path": "pose.pth",
            "pose_checkpoint_iter": 100,
            "pose_ckpt": "000100",
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene_paths = ["obj_a/tabletop/scene.npy", "obj_b/tabletop/scene.npy"]
            digest = hashlib.sha256(
                "".join(f"{entry}\n" for entry in scene_paths).encode("utf-8")
            ).hexdigest()
            scene_list_path = Path(tmp_dir) / "scene_list.json"
            scene_list_path.write_text(
                json.dumps(
                    {
                        "scene_count": len(scene_paths),
                        "scene_list_sha256": digest,
                        "scene_paths": scene_paths,
                    }
                ),
                encoding="utf-8",
            )
            config.test_data.test_scene_list_path = str(scene_list_path)

            manifest = build_manifest(config, checkpoint_meta)

            self.assertEqual(manifest["scene_list"]["scene_count"], 2)
            self.assertEqual(manifest["scene_list"]["scene_list_sha256"], digest)
            self.assertIn("commit", manifest["repository"])
            self.assertIn("python", manifest["runtime"])

    def test_completeness_rejects_enabled_or_scope_mismatch(self):
        config = _build_config(enabled=True, scope="global", mode="random")
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene_path = Path(tmp_dir) / "scene.npy"
            pc_path = Path(tmp_dir) / "pc.npy"
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
            scene_data = build_scene_export_record(score_record, _pose_records(20), config)
            validate_scene_export_completeness(scene_data, config)

            legacy_scene_data = dict(scene_data)
            legacy_scene_data.pop("sample_selection_enabled")
            with self.assertRaisesRegex(KeyError, "sample_selection_enabled"):
                validate_scene_export_completeness(legacy_scene_data, config)

            disabled_config = _build_config(enabled=False, scope="global", mode="random")
            with self.assertRaisesRegex(ValueError, "sample_selection_enabled"):
                validate_scene_export_completeness(scene_data, disabled_config)

            per_type_config = _build_config(enabled=True, scope="per_type", mode="random")
            with self.assertRaisesRegex(ValueError, "sample_selection_scope"):
                validate_scene_export_completeness(scene_data, per_type_config)


if __name__ == "__main__":
    unittest.main()
