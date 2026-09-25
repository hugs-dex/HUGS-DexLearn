import os
from os.path import join as pjoin
from glob import glob
import random
import time
import numpy as np
from torch.utils.data import Dataset

from dexlearn.utils.util import load_json
from scipy.spatial.transform import Rotation as sciR


class HumanBiDexDataset(Dataset):
    def __init__(self, config: dict, mode: str, sc_voxel_size: float = None):
        self.config = config
        self.sc_voxel_size = sc_voxel_size
        self.mode = mode

        # Initialize cache dict
        self.grasp_path_dict = {}  # {grasp_type: {obj_id: [path1, path2, ...]}}
        self.pc_path_dict = {}  # {obj_name: [pc_path1, ...]}
        # self.pc_data_cache = {}   # 内存缓存点云数据 (可选)

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
        self.data_num = 0

        print(f"Pre-indexing {mode} data paths...")
        self.grasp_path_dict = {}

        for obj_id in self.obj_id_lst:
            # 只在这里执行一次 glob
            found_paths = glob(
                pjoin(self.config.grasp_path, obj_id, "**/**.npy"),
                recursive=True,
            )

            if len(found_paths) == 0:
                continue

            self.data_num += len(found_paths)
            self.grasp_path_dict[obj_id] = sorted(found_paths)

        print(f"mode: {mode}, grasp type number: {self.grasp_type_num}, grasp data num: {self.data_num}")

    def init_test(self):
        split_name = self.config.test_split
        self.obj_id_lst = load_json(pjoin(self.config.object_path, self.config.split_path, f"{split_name}.json"))

        if self.config.mini_test:
            self.obj_id_lst = self.obj_id_lst[:100]

        self.test_cfg_lst = []

        # --- 修改部分：统一将 config 转化为列表 ---
        scene_patterns = self.config.test_scene_cfg
        if isinstance(scene_patterns, str):
            scene_patterns = [scene_patterns]

        for obj_id in self.obj_id_lst:
            base_dir = pjoin(self.config.object_path, "scene_cfg", obj_id)

            # 针对每一个 pattern 进行搜索
            for pattern in scene_patterns:
                full_pattern = pjoin(base_dir, pattern)

                # 使用 recursive=True 以支持 ** 语法
                # 使用 set 去重，防止同一个文件被多个 pattern 重复匹配
                found_files = glob(full_pattern, recursive=True)
                self.test_cfg_lst.extend(found_files)

        # 最终去重并排序（保证实验可复现）
        self.test_cfg_lst = sorted(list(set(self.test_cfg_lst)))

        self.data_num = self.grasp_type_num * len(self.test_cfg_lst)
        print(
            f"Test split: {split_name}, grasp type number: {self.grasp_type_num}, "
            f"object cfg num: {len(self.test_cfg_lst)}"
        )
        return

    def __len__(self):
        return self.data_num

    def __getitem__(self, id: int):
        # print(f"data id: {id}") # DEBUG

        t_start = time.perf_counter()
        metrics = {}

        ret_dict = {}

        if self.mode == "train" or self.mode == "eval":
            t0 = time.perf_counter()

            # random select grasp data
            grasp_obj_lst = self.obj_id_lst
            rand_obj_id = random.choice(grasp_obj_lst)
            grasp_npy_lst = self.grasp_path_dict[rand_obj_id]
            grasp_path = random.choice(sorted(grasp_npy_lst))
            metrics["glob_time"] = time.perf_counter() - t0

            t1 = time.perf_counter()
            grasp_data = np.load(grasp_path, allow_pickle=True).item()
            metrics["grasp_load_time"] = time.perf_counter() - t1

            ret_dict["path"] = grasp_path

            # grasp type
            l_contacts = grasp_data["hand"]["left"]["contacts"] if grasp_data["hand"]["left"] else [False] * 5
            r_contacts = grasp_data["hand"]["right"]["contacts"] if grasp_data["hand"]["right"] else [False] * 5
            has_l = any(l_contacts)
            has_r = any(r_contacts)
            if has_l and has_r:
                grasp_type = "2_both"  # Both
            elif has_l:
                grasp_type = "1_left"  # Left Only
            elif has_r:
                grasp_type = "0_right"  # Right Only
            else:
                raise ValueError(f"Grasp data has no contact: {grasp_path}")
            assert grasp_type in self.grasp_type_lst, f"Grasp type {grasp_type} not in grasp_type_lst"
            rand_grasp_type = grasp_type

            # wrist pose
            for side, is_active in grasp_data["hand"].items():
                if is_active:
                    ret_dict[f"{side}_hand_trans"] = np.asarray(grasp_data["hand"][side]["trans"]).reshape(1, 1, 3)
                    ret_dict[f"{side}_hand_rot"] = (
                        sciR.from_rotvec(grasp_data["hand"][side]["rot"]).as_matrix().reshape(1, 1, 3, 3)
                    )
                else:
                    ret_dict[f"{side}_hand_trans"] = np.zeros((1, 1, 3))  # assign zero
                    ret_dict[f"{side}_hand_rot"] = np.zeros((1, 1, 3, 3))

            # object info
            obj_name = grasp_data["object"]["name"]
            obj_scale = grasp_data["object"]["rel_scale"]
            obj_pose = grasp_data["object"]["pose"]

            t2 = time.perf_counter()

            # read object point cloud
            if obj_name not in self.pc_path_dict:
                # store the pointcloud file path
                self.pc_path_dict[obj_name] = sorted(glob(pjoin(self.object_pc_folder, obj_name, "**.npy")))
            pc_path = random.choice(self.pc_path_dict[obj_name])

            raw_pc = np.load(pc_path, allow_pickle=True)
            idx = np.random.choice(raw_pc.shape[0], self.config.num_points, replace=True)
            scaled_pc = raw_pc[idx] * obj_scale  # re-scale the raw mesh (pointcloud) to the actual scale
            R = obj_pose[:3, :3]  # (3, 3)
            t = obj_pose[:3, 3]  # (3,)
            transformed_pc = np.matmul(scaled_pc, R.T) + t  # transform the object pointcloud based on the object pose
            pc = transformed_pc

            metrics["pc_process_time"] = time.perf_counter() - t2

        elif self.mode == "test":
            rand_grasp_type = self.grasp_type_lst[id // len(self.test_cfg_lst)]
            scene_path = self.test_cfg_lst[id % len(self.test_cfg_lst)]

            scene_cfg = np.load(scene_path, allow_pickle=True).item()

            obj_name = scene_cfg["object"]["name"]
            obj_scale = scene_cfg["object"]["rel_scale"]
            obj_pose = scene_cfg["object"]["pose"]

            # read point cloud
            if obj_name not in self.pc_path_dict:
                # store the pointcloud file path
                self.pc_path_dict[obj_name] = sorted(glob(pjoin(self.object_pc_folder, obj_name, "**.npy")))
            pc_path = random.choice(self.pc_path_dict[obj_name])

            raw_pc = np.load(pc_path, allow_pickle=True)
            idx = np.random.choice(raw_pc.shape[0], self.config.num_points, replace=True)
            scaled_pc = raw_pc[idx] * obj_scale  # re-scale the raw mesh (pointcloud) to the actual scale
            R = obj_pose[:3, :3]  # (3, 3)
            t = obj_pose[:3, 3]  # (3,)
            transformed_pc = np.matmul(scaled_pc, R.T) + t  # transform the object pointcloud based on the object pose
            pc = transformed_pc

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
                for side, is_active in grasp_data["hand"].items():
                    if is_active:
                        ret_dict[f"{side}_hand_trans"] -= pc_centroid[None, :, :]

        if self.mode == "test":
            pc = self._apply_test_pc_runtime_scale(pc)

        ret_dict["point_clouds"] = pc  # (N, 3)
        ret_dict["grasp_type_id"] = int(rand_grasp_type.split("_")[0]) if self.config.grasp_type_cond else 0
        if self.sc_voxel_size is not None:
            ret_dict["coors"] = pc / self.sc_voxel_size  # (N, 3)
            ret_dict["feats"] = pc  # (N, 3)

        # total_time = time.perf_counter() - t_start
        # print(f"\n--- Data ID {id} Profiling ---")
        # for k, v in metrics.items():
        #     print(f"{k}: {v:.4f}s ({v/total_time:.1%})")
        # print(f"Total Time: {total_time:.4f}s")

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
