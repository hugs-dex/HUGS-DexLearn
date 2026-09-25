import json
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from dexlearn.dataset.robot_multidex import RobotMultiDexDataset


def _config(object_path, scene_list_path, *, test_scene_num=0):
    return OmegaConf.create(
        {
            "object_path": str(object_path),
            "split_path": "valid_split",
            "test_split": "all",
            "test_scene_cfg": "tabletop_ur10e/**.npy",
            "test_scene_list_path": str(scene_list_path),
            "mini_test": False,
            "test_object_num": 0,
            "test_scene_num": test_scene_num,
            "test_subset_seed": 0,
            "grasp_type_lst": ["1_right_two", "5_both_full"],
            "pc_path": "vision_data/azure_kinect_dk",
            "pc_source": "complete",
            "preload_point_clouds": False,
        }
    )


class RobotMultiDexSceneListTest(unittest.TestCase):
    def _create_assets(self, root):
        object_path = root / "DGN_2k"
        (object_path / "valid_split").mkdir(parents=True)
        (object_path / "valid_split" / "all.json").write_text(
            json.dumps(["obj_a", "obj_b"]), encoding="utf-8"
        )
        relative_paths = [
            Path("obj_a/tabletop_ur10e/scale002_pose000_0.npy"),
            Path("obj_b/tabletop_ur10e/scale003_pose001_0.npy"),
        ]
        for relative_path in relative_paths:
            scene_path = object_path / "scene_cfg" / relative_path
            scene_path.parent.mkdir(parents=True, exist_ok=True)
            scene_path.touch()
        return object_path, relative_paths

    def test_exact_scene_list_accepts_relative_ids_and_absolute_paths(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            object_path, relative_paths = self._create_assets(root)
            scene_list_path = root / "scenes.json"
            scene_list_path.write_text(
                json.dumps(
                    {
                        "scene_count": 2,
                        "scene_paths": [
                            str(relative_paths[0].with_suffix("")),
                            str(object_path / "scene_cfg" / relative_paths[1]),
                        ],
                    }
                ),
                encoding="utf-8",
            )

            dataset = RobotMultiDexDataset(_config(object_path, scene_list_path), mode="test")

            expected_paths = sorted(str(object_path / "scene_cfg" / path) for path in relative_paths)
            self.assertEqual(dataset.test_cfg_lst, expected_paths)
            self.assertEqual(dataset.data_num, 4)

    def test_scene_list_rejects_random_scene_subsampling(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            object_path, relative_paths = self._create_assets(root)
            scene_list_path = root / "scenes.json"
            scene_list_path.write_text(json.dumps([str(relative_paths[0])]), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                RobotMultiDexDataset(
                    _config(object_path, scene_list_path, test_scene_num=1),
                    mode="test",
                )

    def test_scene_list_rejects_duplicates_and_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            object_path, relative_paths = self._create_assets(root)
            scene_list_path = root / "scenes.json"
            duplicate_entry = str(relative_paths[0])
            scene_list_path.write_text(
                json.dumps({"scene_count": 2, "scene_paths": [duplicate_entry, duplicate_entry]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate explicit scene config"):
                RobotMultiDexDataset(_config(object_path, scene_list_path), mode="test")

            scene_list_path.write_text(
                json.dumps({"scene_count": 3, "scene_paths": [duplicate_entry]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Declared scene_count"):
                RobotMultiDexDataset(_config(object_path, scene_list_path), mode="test")


if __name__ == "__main__":
    unittest.main()
