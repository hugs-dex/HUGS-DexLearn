import os
from os.path import join as pjoin
from glob import glob
import random

import numpy as np
from torch.utils.data import Dataset

from dexlearn.utils.rot import numpy_quaternion_to_matrix
from dexlearn.utils.util import load_json, load_scene_cfg

import pdb


class DexDataset(Dataset):
    def __init__(self, config: dict, mode: str, sc_voxel_size: float = None):
        self.config = config
        self.sc_voxel_size = sc_voxel_size
        self.mode = mode

        if self.config.grasp_type_lst is not None:
            self.grasp_type_lst = self.config.grasp_type_lst
        else:
            self.grasp_type_lst = os.listdir(self.config.grasp_path)
        self.grasp_type_num = len(self.grasp_type_lst)
        self.object_pc_folder = pjoin(self.config.object_path, self.config.pc_path)

        if mode == "train" or mode == "eval":
            self.init_train_eval(mode)
        elif mode == "test":
            self.init_test()
        return

    def init_train_eval(self, mode):
        split_name = "test" if mode == "eval" else "train"
        self.obj_id_lst = load_json(pjoin(self.config.object_path, self.config.split_path, f"{split_name}.json"))

        self.grasp_obj_dict = {}
        self.data_num = 0
        for grasp_type in self.grasp_type_lst:
            self.grasp_obj_dict[grasp_type] = []
            for obj_id in self.obj_id_lst:
                obj_grasp_data = len(
                    glob(
                        pjoin(self.config.grasp_path, grasp_type, obj_id, "**/**.npy"),
                        recursive=True,
                    )
                )
                if obj_grasp_data == 0:
                    continue
                self.data_num += obj_grasp_data
                self.grasp_obj_dict[grasp_type].append(obj_id)
            if len(self.grasp_obj_dict[grasp_type]) == 0:
                self.grasp_obj_dict.pop(grasp_type)
        print(f"mode: {mode}, grasp type number: {self.grasp_type_num}, grasp data num: {self.data_num}")
        return

    def init_test(self):
        split_name = self.config.test_split
        self.obj_id_lst = []
        self.test_cfg_lst = []
        self.obj_id_lst = load_json(pjoin(self.config.object_path, self.config.split_path, f"{split_name}.json"))
        if self.config.mini_test:
            self.obj_id_lst = self.obj_id_lst[:100]
        for o in self.obj_id_lst:
            self.test_cfg_lst.extend(
                glob(
                    pjoin(
                        self.config.object_path,
                        "scene_cfg",
                        o,
                        self.config.test_scene_cfg,
                    )
                )
            )
        self.data_num = self.grasp_type_num * len(self.test_cfg_lst)
        print(
            f"Test split: {split_name}, grasp type number: {self.grasp_type_num}, object cfg num: {len(self.test_cfg_lst)}"
        )
        return

    def __len__(self):
        return self.data_num

    def __getitem__(self, id: int):
        ret_dict = {}

        if self.mode == "train" or self.mode == "eval":
            # random select grasp data
            rand_grasp_type = random.choice(self.grasp_type_lst)
            grasp_obj_lst = self.grasp_obj_dict[rand_grasp_type]
            rand_obj_id = random.choice(grasp_obj_lst)
            grasp_npy_lst = glob(
                pjoin(self.config.grasp_path, rand_grasp_type, rand_obj_id, "**/**.npy"),
                recursive=True,
            )
            grasp_path = random.choice(sorted(grasp_npy_lst))
            grasp_data = np.load(grasp_path, allow_pickle=True).item()

            robot_pose = np.stack(
                [
                    grasp_data["pregrasp_qpos"],
                    grasp_data["grasp_qpos"],
                    grasp_data["squeeze_qpos"],
                ],
                axis=-2,
            )
            if len(robot_pose.shape) == 3:
                rand_pose_id = np.random.randint(robot_pose.shape[0])
                robot_pose = robot_pose[rand_pose_id : rand_pose_id + 1]  # 1, 3, J
            else:
                raise NotImplementedError

            scene_cfg = load_scene_cfg(grasp_data["scene_path"])

            # read point cloud
            pc_path_lst = glob(pjoin(self.object_pc_folder, scene_cfg["scene_id"], "partial_pc**.npy"))
            pc_path = random.choice(sorted(pc_path_lst))
            raw_pc = np.load(pc_path, allow_pickle=True)
            idx = np.random.choice(raw_pc.shape[0], self.config.num_points, replace=True)
            pc = raw_pc[idx]
            if "scene_scale" in grasp_data:
                pc *= grasp_data["scene_scale"][rand_pose_id]

            ret_dict["hand_trans"] = robot_pose[:, :, :3]  # (K, n, 3)
            ret_dict["hand_rot"] = numpy_quaternion_to_matrix(robot_pose[:, :, 3:7])  # (K, n, 3, 3)
            ret_dict["hand_joint"] = robot_pose[:, :, 7:]  # (K, n, Q)

        elif self.mode == "test":
            rand_grasp_type = self.grasp_type_lst[id // len(self.test_cfg_lst)]
            scene_path = self.test_cfg_lst[id % len(self.test_cfg_lst)]
            scene_cfg = load_scene_cfg(scene_path)

            # read point cloud
            pc_path_lst = glob(
                pjoin(self.object_pc_folder, scene_cfg["scene_id"], "partial_pc**.npy"),
            )
            pc_path = random.choice(sorted(pc_path_lst))
            raw_pc = np.load(pc_path, allow_pickle=True)
            idx = np.random.choice(raw_pc.shape[0], self.config.num_points, replace=True)
            pc = raw_pc[idx]

            ret_dict["save_path"] = pjoin(
                self.config.name, rand_grasp_type, scene_cfg["scene_id"], os.path.basename(pc_path)
            )
            ret_dict["scene_path"] = scene_path
            ret_dict["pc_path"] = pc_path

        # Move the pointcloud centroid to the origin. Move the robot pose accordingly.
        if self.config.pc_centering:
            pc_centroid = np.mean(pc, axis=-2, keepdims=True)
            pc = pc - pc_centroid  # normalization
            ret_dict["pc_centroid"] = pc_centroid.astype(np.float32)
            if self.mode != "test":
                ret_dict["hand_trans"] = ret_dict["hand_trans"] - pc_centroid[None, :, :]

        if self.mode == "test":
            pc = self._apply_test_pc_runtime_scale(pc)

        ret_dict["point_clouds"] = pc  # (N, 3)
        ret_dict["grasp_type_id"] = int(rand_grasp_type.split("_")[0]) if self.config.grasp_type_cond else 0
        if self.sc_voxel_size is not None:
            ret_dict["coors"] = pc / self.sc_voxel_size  # (N, 3)
            ret_dict["feats"] = pc  # (N, 3)
        return ret_dict

    def _apply_test_pc_runtime_scale(self, pc):
        """Scale test point clouds for runtime-only inference conditioning.

        Args:
            pc: Centered test point cloud in physical scene units.

        Returns:
            Point cloud multiplied by ``config.pc_runtime_scale``. Training and
            eval samples never call this helper, so their distribution is
            unchanged.
        """
        runtime_scale = float(getattr(self.config, "pc_runtime_scale", 1.0))
        if runtime_scale <= 0.0:
            raise ValueError(f"pc_runtime_scale must be positive, got {runtime_scale}")
        if runtime_scale == 1.0:
            return pc.astype(np.float32, copy=False)
        return (pc * np.float32(runtime_scale)).astype(np.float32, copy=False)
