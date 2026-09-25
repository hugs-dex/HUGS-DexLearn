import os
import csv
from os.path import join as pjoin
from glob import glob
from collections import Counter, defaultdict
import random
import numpy as np
from torch.utils.data import Dataset

from dexlearn.utils.util import load_json
from dexlearn.utils.human_hand import normalize_hand_pos_source
from dexlearn.utils.config import flatten_multidex_data_config
from .grasp_types import GRASP_TYPES
from scipy.spatial.transform import Rotation as sciR

# Fixed left hand pose for right-only grasps
FIXED_LEFT_HAND_TRANS = np.array([0.0, 0.0, -0.5])
FIXED_LEFT_HAND_ROT = np.eye(3)
REAL_GRASP_TYPE_IDS = tuple(range(1, len(GRASP_TYPES)))
TRAIN_SAMPLING_UNIT_ALIASES = {
    "record": "record_uniform",
    "record_uniform": "record_uniform",
    "object": "object_uniform",
    "object_uniform": "object_uniform",
    "posed_object": "posed_object_uniform",
    "pose_group": "posed_object_uniform",
    "posed_object_uniform": "posed_object_uniform",
}


class HumanMultiDexDataset(Dataset):
    """Dataset for human bimanual grasps with multiple grasp types."""

    _MIRROR = np.diag([-1.0, 1.0, 1.0])  # YZ-plane reflection matrix
    _MIRRORED_ROTVEC_SIGN = np.array([1.0, -1.0, -1.0], dtype=np.float32)

    def __init__(self, config: dict, mode: str, sc_voxel_size: float = None):
        flatten_multidex_data_config(config)
        self.config = config
        self.sc_voxel_size = sc_voxel_size
        self.mode = mode
        self.grasp_path_dict = {}
        self.record_grasp_path_lst = []
        self.pc_path_dict = {}
        self.type_grasp_path_dict = defaultdict(list)
        self.type_object_grasp_path_dict = defaultdict(lambda: defaultdict(list))
        self.object_sequence_type_counter = defaultdict(lambda: defaultdict(Counter))
        self.object_pose_group_dict = defaultdict(list)
        self.object_pose_group_by_key = {}
        self.grasp_path_pose_group_key = {}
        self.object_type_counter = defaultdict(Counter)
        self.type_counter = Counter()
        self.object_pc_folder = pjoin(config.object_path, config.pc_path)
        self.hand_pos_source = normalize_hand_pos_source(getattr(config, "hand_pos_source", "wrist"))
        self.random_pc_across_sequences = bool(getattr(config, "random_pc_across_sequences", True))
        self.type_objective = str(getattr(config, "type_objective", "ce")).lower()
        if self.type_objective != "ce":
            raise ValueError("HumanMultiDexDataset only supports type_objective=ce")
        self.supervision_scope = str(getattr(config, "supervision_scope", "record")).lower()
        self.negative_policy = str(getattr(config, "negative_policy", "softmax")).lower()
        self.train_sampling_unit = self._normalize_train_sampling_unit(
            getattr(config, "train_sampling_unit", "record_uniform")
        )
        self.pose_group_soft_labels = bool(
            getattr(config, "pose_group_soft_labels", self.train_sampling_unit == "posed_object_uniform")
        )
        self.type_balancing_enabled = (
            self.mode == "train"
            and bool(getattr(config, "type_balancing_enabled", False))
        )
        self.type_sampler_enabled = (
            self.mode == "train"
            and bool(getattr(config, "type_sampler_enabled", False))
        )
        self.type_sampler_alpha = float(getattr(config, "type_sampler_alpha", 1.0))
        self.type_sampler_object_uniform = bool(getattr(config, "type_sampler_object_uniform", True))

        if mode == "test":
            self.grasp_type_lst = (
                config.grasp_type_lst
                if hasattr(config, "grasp_type_lst") and config.grasp_type_lst
                else [gt for gt in GRASP_TYPES if gt != "0_any"]
            )
            self.grasp_type_num = len(self.grasp_type_lst)

        if mode in ["train", "eval"]:
            self._init_train_eval(mode)
        elif mode == "test":
            self._init_test()

    def _normalize_train_sampling_unit(self, sampling_unit):
        """Normalize the configured train sampling unit.

        Args:
            sampling_unit: Config value selecting record- or object-uniform
                train/eval sampling.

        Returns:
            Canonical sampling unit string, either ``record_uniform`` or
            ``object_uniform``.
        """
        normalized = TRAIN_SAMPLING_UNIT_ALIASES.get(str(sampling_unit).strip().lower())
        if normalized is None:
            raise ValueError(
                f"Unsupported train_sampling_unit={sampling_unit}. "
                f"Expected one of {sorted(TRAIN_SAMPLING_UNIT_ALIASES)}."
            )
        return normalized

    def _init_train_eval(self, mode):

        split_name = self._resolve_train_eval_split_name(mode)
        self.split_name = split_name
        self.mano_pose_dim = 24 if "GRAB" in self.config.grasp_path else 45
        self.obj_id_lst = load_json(pjoin(self.config.object_path, self.config.split_path, f"{split_name}.json"))

        print(f"Pre-indexing {mode} data paths...")
        indexed_obj_ids = []
        for obj_id in self.obj_id_lst:
            found_paths = glob(pjoin(self.config.grasp_path, obj_id, "**/**.npy"), recursive=True)
            if found_paths:
                self.grasp_path_dict[obj_id] = sorted(found_paths)
                indexed_obj_ids.append(obj_id)
        self.obj_id_lst = indexed_obj_ids

        self.record_grasp_path_lst = [
            grasp_path
            for obj_id in self.obj_id_lst
            for grasp_path in self.grasp_path_dict[obj_id]
        ]
        self.record_data_num = len(self.record_grasp_path_lst)
        self._index_grasp_type_distribution()
        if self.train_sampling_unit == "posed_object_uniform":
            self._validate_pose_group_sampling_index()
        self.data_num = self.record_data_num
        print(
            f"mode: {mode}, split: {split_name}, grasp data num: {self.data_num}, "
            f"sampling unit: {self.train_sampling_unit}"
        )

    def _resolve_train_eval_split_name(self, mode):
        """Resolve the object split file used by train/eval indexing.

        Args:
            mode: Dataset mode passed by the dataloader builder. ``train``
                reads the configured train split, and ``eval`` keeps the
                validation/test split.

        Returns:
            Split name without the ``.json`` suffix.
        """
        if mode == "eval":
            return "test"
        split_name = str(getattr(self.config, "train_split", "train")).strip()
        if split_name not in {"train", "all"}:
            raise ValueError(f"Unsupported human train_split={split_name}. Expected 'train' or 'all'.")
        return split_name

    def _index_grasp_type_distribution(self):
        """Index train/eval grasp paths by grasp type and object.

        Args:
            None.

        Returns:
            None. The per-type path dictionaries and counters are stored on
            this dataset instance.
        """
        self.type_grasp_path_dict = defaultdict(list)
        self.type_object_grasp_path_dict = defaultdict(lambda: defaultdict(list))
        self.object_sequence_type_counter = defaultdict(lambda: defaultdict(Counter))
        self.object_pose_group_dict = defaultdict(list)
        self.object_pose_group_by_key = {}
        self.grasp_path_pose_group_key = {}
        self.object_type_counter = defaultdict(Counter)
        self.type_counter = Counter()
        metadata_pose_index = self._load_metadata_pose_index() if self._pose_group_metadata_required() else None
        pose_group_records = defaultdict(list)

        for obj_id, grasp_paths in self.grasp_path_dict.items():
            for grasp_path in grasp_paths:
                try:
                    grasp_data = np.load(grasp_path, allow_pickle=True).item()
                    grasp_type, _ = self._determine_grasp_type(grasp_data)
                except Exception as exc:
                    raise RuntimeError(f"Failed to index grasp type for {grasp_path}") from exc

                grasp_type_id = int(grasp_type.split("_", 1)[0])
                self.type_grasp_path_dict[grasp_type_id].append(grasp_path)
                self.type_object_grasp_path_dict[grasp_type_id][obj_id].append(grasp_path)
                self.object_type_counter[obj_id][grasp_type_id] += 1
                self.type_counter[grasp_type_id] += 1
                sequence_id = self._get_grasp_sequence_id(grasp_data)
                if sequence_id is not None:
                    self.object_sequence_type_counter[obj_id][sequence_id][grasp_type_id] += 1
                if metadata_pose_index is not None:
                    pose_index = self._lookup_pose_index(
                        metadata_pose_index,
                        obj_id=obj_id,
                        sequence_id=sequence_id,
                        grasp_data=grasp_data,
                        grasp_path=grasp_path,
                    )
                    pose_group_key = (obj_id, pose_index)
                    self.grasp_path_pose_group_key[grasp_path] = pose_group_key
                    pose_group_records[pose_group_key].append(
                        {
                            "grasp_path": grasp_path,
                            "sequence_id": sequence_id or "",
                            "grasp_type_id": grasp_type_id,
                            "grasp_data": grasp_data,
                        }
                    )

        if metadata_pose_index is not None:
            self._finalize_pose_group_index(pose_group_records)

    def _metadata_csv_path(self):
        """Resolve the formatted dataset metadata CSV path.

        Args:
            None.

        Returns:
            Path to ``metadata.csv`` for the formatted human grasp dataset.
        """
        configured = getattr(self.config, "metadata_path", None)
        if configured:
            return str(configured)
        dataset_root = os.path.dirname(str(self.config.object_path).rstrip(os.sep))
        return pjoin(dataset_root, "metadata.csv")

    def _pose_group_metadata_required(self):
        """Return whether train/eval indexing needs metadata pose groups.

        Args:
            None.

        Returns:
            ``True`` when posed-object sampling or soft-label supervision is
            enabled.
        """
        return self.train_sampling_unit == "posed_object_uniform" or self.pose_group_soft_labels

    def _load_metadata_pose_index(self):
        """Load ``scene -> pose_index`` annotations from metadata.csv.

        Args:
            None.

        Returns:
            Dictionary mapping scene ids such as ``obj_31_seq_0`` to the
            annotated pose index string.
        """
        metadata_csv = self._metadata_csv_path()
        if not os.path.exists(metadata_csv):
            if not self._pose_group_metadata_required():
                print(f"Warning: metadata.csv not found; posed-object soft labels are disabled: {metadata_csv}")
                return None
            raise FileNotFoundError(
                f"metadata.csv is required for posed-object soft labels but was not found: {metadata_csv}"
            )

        pose_index_by_scene = {}
        with open(metadata_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if "scene" not in (reader.fieldnames or []) or "pose_index" not in (reader.fieldnames or []):
                raise ValueError(f"metadata.csv must contain 'scene' and 'pose_index' columns: {metadata_csv}")
            for row in reader:
                scene = str(row.get("scene", "")).strip()
                pose_index = str(row.get("pose_index", "")).strip()
                if scene and pose_index:
                    pose_index_by_scene[scene] = pose_index
        return pose_index_by_scene

    def _scene_id_from_obj_sequence(self, obj_id, sequence_id):
        """Build a metadata scene id from object and sequence ids.

        Args:
            obj_id: Canonical object id, for example ``obj_31``.
            sequence_id: Sequence id, for example ``seq_0``.

        Returns:
            Scene id such as ``obj_31_seq_0``, or ``None`` when the sequence is
            unavailable.
        """
        if sequence_id is None or sequence_id == "":
            return None
        return f"{obj_id}_{sequence_id}"

    def _lookup_pose_index(self, metadata_pose_index, obj_id, sequence_id, grasp_data, grasp_path):
        """Find the annotated pose index for one grasp record.

        Args:
            metadata_pose_index: Mapping loaded from ``metadata.csv``.
            obj_id: Canonical object id inferred from the grasp folder.
            sequence_id: Sequence id read from the grasp record.
            grasp_data: Loaded grasp data dictionary.
            grasp_path: Path to the current grasp record, used in errors.

        Returns:
            Pose-index string from the metadata table.
        """
        scene_candidates = []
        scene_from_sequence = self._scene_id_from_obj_sequence(obj_id, sequence_id)
        if scene_from_sequence is not None:
            scene_candidates.append(scene_from_sequence)

        object_data = grasp_data.get("object", {})
        source_scene = object_data.get("source_scene")
        if source_scene is not None:
            scene_candidates.append(str(source_scene))

        for scene in scene_candidates:
            if scene in metadata_pose_index:
                return metadata_pose_index[scene]

        raise KeyError(
            "Could not find pose_index in metadata.csv for grasp record "
            f"{grasp_path}; tried scenes={scene_candidates}"
        )

    def _representative_pose_group_record(self, records):
        """Choose a deterministic representative record for cache metadata.

        Args:
            records: Pose-group member records collected during indexing.

        Returns:
            The first record after sorting by grasp path.
        """
        return sorted(records, key=lambda item: item["grasp_path"])[0]

    def _finalize_pose_group_index(self, pose_group_records):
        """Build posed-object group targets from indexed grasp records.

        Args:
            pose_group_records: Mapping from ``(object_id, pose_index)`` to
                member grasp records.

        Returns:
            None. Pose-group dictionaries are stored on this dataset instance.
        """
        self.object_pose_group_dict = defaultdict(list)
        self.object_pose_group_by_key = {}

        for pose_group_key, records in sorted(pose_group_records.items()):
            obj_id, pose_index = pose_group_key
            type_counts = np.zeros(len(REAL_GRASP_TYPE_IDS), dtype=np.float32)
            for record in records:
                grasp_type_id = int(record["grasp_type_id"])
                if grasp_type_id not in REAL_GRASP_TYPE_IDS:
                    raise ValueError(f"Invalid grasp_type_id={grasp_type_id} in pose group {pose_group_key}")
                type_counts[grasp_type_id - 1] += 1.0

            record_count = float(type_counts.sum())
            if record_count <= 0.0:
                continue

            representative = self._representative_pose_group_record(records)
            representative_data = representative["grasp_data"]
            obj_name = representative_data.get("object", {}).get("name", obj_id)
            sequence_id = representative.get("sequence_id", "")
            target_distribution = type_counts / record_count
            group = {
                "object_id": obj_id,
                "pose_index": pose_index,
                "pose_group_key": pose_group_key,
                "member_sequences": sorted({str(record.get("sequence_id", "")) for record in records}),
                "member_grasp_paths": sorted(record["grasp_path"] for record in records),
                "type_counts": type_counts,
                "target_type_distribution": target_distribution.astype(np.float32),
                "record_count": int(record_count),
                "representative_pc_path": self._representative_pointcloud_path(obj_name, sequence_id),
                "representative_object_pose": np.asarray(
                    representative_data.get("object", {}).get("pose", np.eye(4)),
                    dtype=np.float32,
                ),
            }
            self.object_pose_group_by_key[pose_group_key] = group
            self.object_pose_group_dict[obj_id].append(group)
        self._write_pose_group_cache()

    def _pose_group_cache_path(self):
        """Resolve the optional posed-object group cache CSV path.

        Args:
            None.

        Returns:
            Cache CSV path. A configured empty value disables cache writing.
        """
        configured = getattr(self.config, "pose_group_cache_path", None)
        if configured is not None:
            configured = str(configured)
            return configured if configured else None
        dataset_root = os.path.dirname(str(self.config.object_path).rstrip(os.sep))
        split_name = getattr(self, "split_name", "test" if self.mode == "eval" else self.mode)
        return pjoin(dataset_root, "human_prior_pose_groups", f"{split_name}.csv")

    def _write_pose_group_cache(self):
        """Write a readable posed-object group cache for inspection.

        Args:
            None.

        Returns:
            None. The function writes a CSV cache when the target path is
            configured and writable.
        """
        cache_path = self._pose_group_cache_path()
        if cache_path is None:
            return
        fieldnames = [
            "split",
            "object_id",
            "pose_index",
            "member_sequences",
            "member_grasp_paths",
            "count_1",
            "count_2",
            "count_3",
            "count_4",
            "count_5",
            "record_count",
            "representative_pc_path",
            "representative_object_pose",
        ]
        try:
            cache_dir = os.path.dirname(cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for obj_id in sorted(self.object_pose_group_dict.keys()):
                    for group in sorted(self.object_pose_group_dict[obj_id], key=lambda item: str(item["pose_index"])):
                        counts = group["type_counts"].astype(int).tolist()
                        writer.writerow(
                            {
                                "split": getattr(self, "split_name", "test" if self.mode == "eval" else self.mode),
                                "object_id": group["object_id"],
                                "pose_index": group["pose_index"],
                                "member_sequences": "|".join(group["member_sequences"]),
                                "member_grasp_paths": "|".join(group["member_grasp_paths"]),
                                "count_1": counts[0],
                                "count_2": counts[1],
                                "count_3": counts[2],
                                "count_4": counts[3],
                                "count_5": counts[4],
                                "record_count": group["record_count"],
                                "representative_pc_path": group["representative_pc_path"],
                                "representative_object_pose": np.asarray(
                                    group["representative_object_pose"], dtype=np.float32
                                ).reshape(-1).tolist(),
                            }
                        )
        except OSError as exc:
            print(f"Warning: failed to write posed-object group cache to {cache_path}: {exc}")

    def _representative_pointcloud_path(self, obj_name, sequence_id):
        """Return a representative point-cloud path without random sampling.

        Args:
            obj_name: Canonical object id used under ``object_pc_folder``.
            sequence_id: Sequence id preferred for the representative path.

        Returns:
            Point-cloud path string, or an empty string when no point cloud is
            indexed yet.
        """
        if obj_name not in self.pc_path_dict:
            self.pc_path_dict[obj_name] = sorted(glob(pjoin(self.object_pc_folder, obj_name, "**.npy")))
        pc_candidates = self.pc_path_dict.get(obj_name, [])
        if sequence_id:
            matching = [pc_path for pc_path in pc_candidates if self._pointcloud_matches_sequence(pc_path, sequence_id)]
            if matching:
                return matching[0]
        return pc_candidates[0] if pc_candidates else ""

    def _validate_pose_group_sampling_index(self):
        """Validate that every indexed train object has posed-object groups.

        Args:
            None.

        Returns:
            None.
        """
        missing = [obj_id for obj_id in self.obj_id_lst if not self.object_pose_group_dict.get(obj_id)]
        if missing:
            raise RuntimeError(
                "posed_object_uniform sampling requires at least one metadata pose group per object; "
                f"missing examples={missing[:5]}"
            )

    def get_distribution_analysis(self):
        """Return the indexed grasp-type distribution for this dataset.

        Args:
            None.

        Returns:
            Dictionary containing global type counts, per-object type counts,
            and the dataset sampling configuration.
        """
        type_counts = {str(type_id): int(self.type_counter.get(type_id, 0)) for type_id in range(len(GRASP_TYPES))}
        object_counts = {
            str(obj_id): {str(type_id): int(count) for type_id, count in sorted(counter.items())}
            for obj_id, counter in sorted(self.object_type_counter.items())
        }
        return {
            "mode": self.mode,
            "split_name": str(getattr(self, "split_name", "")),
            "data_num": int(self.data_num),
            "record_data_num": int(getattr(self, "record_data_num", self.data_num)),
            "object_num": int(len(self.obj_id_lst)) if hasattr(self, "obj_id_lst") else 0,
            "pose_group_num": int(sum(len(groups) for groups in self.object_pose_group_dict.values())),
            "type_counts": type_counts,
            "object_type_counts": object_counts,
            "type_objective": self.type_objective,
            "sampling_unit": self.train_sampling_unit,
            "supervision": {
                "scope": self.supervision_scope,
                "negative_policy": self.negative_policy,
            },
            "sampler": {
                "type_balancing_enabled": bool(self.type_balancing_enabled),
                "type_sampler_enabled": bool(self.type_sampler_enabled),
                "type_sampler_alpha": float(self.type_sampler_alpha),
                "type_sampler_object_uniform": bool(self.type_sampler_object_uniform),
            },
        }

    def _sample_balanced_grasp_path(self):
        """Sample one grasp path with tempered type balancing.

        Args:
            None.

        Returns:
            Path to one selected grasp file.
        """
        if self.type_sampler_alpha < 0.0:
            raise ValueError(f"type_sampler_alpha must be non-negative, got {self.type_sampler_alpha}")

        active_type_ids = sorted(type_id for type_id, paths in self.type_grasp_path_dict.items() if paths)
        if not active_type_ids:
            rand_obj_id = random.choice(self.obj_id_lst)
            return random.choice(self.grasp_path_dict[rand_obj_id])

        type_weights = [len(self.type_grasp_path_dict[type_id]) ** self.type_sampler_alpha for type_id in active_type_ids]
        sampled_type_id = random.choices(active_type_ids, weights=type_weights, k=1)[0]

        if self.type_sampler_object_uniform:
            object_path_dict = self.type_object_grasp_path_dict[sampled_type_id]
            obj_id = random.choice(sorted(object_path_dict.keys()))
            return random.choice(object_path_dict[obj_id])
        return random.choice(self.type_grasp_path_dict[sampled_type_id])

    def _sample_unbiased_grasp_path(self, sample_index):
        """Select one grasp path without explicit grasp-type reweighting.

        Args:
            sample_index: Dataset index received by ``__getitem__``. Train
                shuffling is handled by the PyTorch DataLoader.

        Returns:
            Path to one selected grasp record.
        """
        if not self.record_grasp_path_lst:
            raise RuntimeError("HumanMultiDexDataset has no indexed grasp records")

        sample_index = int(sample_index)
        if self.train_sampling_unit == "record_uniform":
            return self.record_grasp_path_lst[sample_index % len(self.record_grasp_path_lst)]

        if self.train_sampling_unit == "object_uniform":
            obj_id = self.obj_id_lst[sample_index % len(self.obj_id_lst)]
            grasp_paths = self.grasp_path_dict[obj_id]
            object_sample_index = sample_index // len(self.obj_id_lst)
            return grasp_paths[object_sample_index % len(grasp_paths)]

        if self.train_sampling_unit == "posed_object_uniform":
            return self._sample_pose_group_grasp_path(sample_index)

        raise ValueError(f"Unsupported train_sampling_unit={self.train_sampling_unit}")

    def _sample_pose_group_grasp_path(self, sample_index):
        """Sample one record through object -> posed-object -> record.

        Args:
            sample_index: Dataset index received by ``__getitem__``.

        Returns:
            Path to one grasp record from the sampled posed-object group.
        """
        obj_id = self.obj_id_lst[int(sample_index) % len(self.obj_id_lst)]
        pose_groups = self.object_pose_group_dict.get(obj_id, [])
        if not pose_groups:
            raise RuntimeError(f"Object '{obj_id}' has no posed-object groups")
        if self.mode == "train":
            pose_group = random.choice(pose_groups)
            return random.choice(pose_group["member_grasp_paths"])

        pose_group_index = (int(sample_index) // len(self.obj_id_lst)) % len(pose_groups)
        pose_group = pose_groups[pose_group_index]
        record_index = (int(sample_index) // (len(self.obj_id_lst) * len(pose_groups))) % len(
            pose_group["member_grasp_paths"]
        )
        return pose_group["member_grasp_paths"][record_index]

    def _attach_pose_group_target(self, grasp_path, ret_dict):
        """Attach the posed-object soft type target for one grasp record.

        Args:
            grasp_path: Path to the sampled grasp record.
            ret_dict: Sample dictionary updated in-place.

        Returns:
            None.
        """
        if not self.pose_group_soft_labels or not self.object_pose_group_by_key:
            return
        pose_group_key = self.grasp_path_pose_group_key.get(grasp_path)
        if pose_group_key is None:
            raise KeyError(f"No posed-object group was indexed for grasp path: {grasp_path}")
        pose_group = self.object_pose_group_by_key[pose_group_key]
        ret_dict["target_type_distribution"] = pose_group["target_type_distribution"].copy()
        ret_dict["pose_group_record_count"] = np.int64(pose_group["record_count"])

    def _get_object_id_from_grasp_path(self, grasp_path):
        """Read the canonical object id from a grasp record path.

        Args:
            grasp_path: Path under ``config.grasp_path``.

        Returns:
            Canonical object id inferred from the first relative path segment.
        """
        rel_path = os.path.relpath(grasp_path, self.config.grasp_path)
        return rel_path.split(os.sep, 1)[0]

    def _init_test(self):
        split_name = self.config.test_split
        self.obj_id_lst = load_json(pjoin(self.config.object_path, self.config.split_path, f"{split_name}.json"))

        if self.config.mini_test:
            self.obj_id_lst = self.obj_id_lst[:100]
        self.obj_id_lst = self._subsample_test_items(
            self.obj_id_lst,
            max_count=int(getattr(self.config, "test_object_num", 0)),
            seed=int(getattr(self.config, "test_subset_seed", 0)),
            item_name="object",
        )

        scene_patterns = (
            [self.config.test_scene_cfg] if isinstance(self.config.test_scene_cfg, str) else self.config.test_scene_cfg
        )

        test_cfg_set = set()
        for obj_id in self.obj_id_lst:
            base_dir = pjoin(self.config.object_path, "scene_cfg", obj_id)
            for pattern in scene_patterns:
                test_cfg_set.update(glob(pjoin(base_dir, pattern), recursive=True))

        self.test_cfg_lst = self._subsample_test_items(
            sorted(test_cfg_set),
            max_count=int(getattr(self.config, "test_scene_num", 0)),
            seed=int(getattr(self.config, "test_subset_seed", 0)) + 1,
            item_name="scene",
        )
        self.data_num = self.grasp_type_num * len(self.test_cfg_lst)  # TO BE CHECKED
        print(
            f"Test split: {split_name}, grasp type list: {self.grasp_type_lst}, "
            f"object cfg num: {len(self.test_cfg_lst)}"
        )

    def _subsample_test_items(self, items, max_count, seed, item_name):
        """Randomly subsample test objects or scenes for faster evaluation sampling.

        Args:
            items: Ordered object ids or scene config paths.
            max_count: Maximum number of items to keep. Values <= 0 keep all items.
            seed: Deterministic seed for the local sampler.
            item_name: Human-readable item name used in logs.

        Returns:
            Sorted list containing either all items or a deterministic random subset.
        """
        items = list(items)
        if max_count <= 0 or max_count >= len(items):
            return items
        sampler = random.Random(seed)
        selected = sorted(sampler.sample(items, max_count))
        print(f"Randomly selected {len(selected)}/{len(items)} test {item_name}s with seed={seed}.")
        return selected

    def __len__(self):
        return self.data_num

    def __getitem__(self, id: int):
        ret_dict = {}

        if self.mode in ["train", "eval"]:
            rand_grasp_type, pc, mirrored, grasp_data = self._load_train_data(id, ret_dict)
        else:  # test
            rand_grasp_type, pc = self._load_test_data(id, ret_dict)
            mirrored = False
            grasp_data = None

        if self.config.pc_centering:
            pc = self._apply_pc_centering(pc, ret_dict, mirrored, grasp_data)

        if self.mode == "train":
            pc = self._apply_scale_aug(pc, ret_dict)
            pc = self._apply_geometric_aug(pc, ret_dict)

        if self.mode == "train":
            pc = self._apply_point_dropout_aug(pc)
            pc = self._apply_pc_noise_aug(pc)

        if self.mode == "test":
            pc = self._apply_test_pc_runtime_scale(pc)

        ret_dict["point_clouds"] = pc
        ret_dict["grasp_type_id"] = int(rand_grasp_type.split("_")[0])
        if self.sc_voxel_size is not None:
            ret_dict["coors"] = pc / self.sc_voxel_size
            ret_dict["feats"] = pc

        return ret_dict

    def _determine_grasp_type(self, grasp_data):
        """Determine grasp type and whether mirroring is needed."""
        l_contacts = grasp_data["hand"]["left"]["contacts"] if grasp_data["hand"]["left"] else [False] * 5
        r_contacts = grasp_data["hand"]["right"]["contacts"] if grasp_data["hand"]["right"] else [False] * 5
        has_l, has_r = any(l_contacts), any(r_contacts)
        l_count, r_count = sum(l_contacts), sum(r_contacts)

        if not (has_l or has_r):
            raise ValueError("Grasp data has no contact")

        if has_l and has_r:
            grasp_type = GRASP_TYPES[5] if (l_count > 3 or r_count > 3) else GRASP_TYPES[4]
            return grasp_type, False

        # Single hand grasp - determine type by finger count
        count = l_count if has_l else r_count
        if count <= 2:
            grasp_type = GRASP_TYPES[1]
        elif count == 3:
            grasp_type = GRASP_TYPES[2]
        else:
            grasp_type = GRASP_TYPES[3]

        # Mirror left-only grasps to right hand
        mirrored = has_l and not has_r
        return grasp_type, mirrored

    def _extract_hand_poses(self, grasp_data, mirrored, ret_dict):
        """Extract and process hand poses from grasp data."""
        if not mirrored:
            for side, is_active in grasp_data["hand"].items():
                if is_active:
                    ret_dict[f"{side}_hand_trans"] = self._get_hand_translation(grasp_data["hand"][side]).reshape(1, 1, 3)
                    ret_dict[f"{side}_hand_rot"] = (
                        sciR.from_rotvec(grasp_data["hand"][side]["rot"]).as_matrix().reshape(1, 1, 3, 3)
                    )
                else:
                    ret_dict[f"{side}_hand_trans"] = FIXED_LEFT_HAND_TRANS.reshape(1, 1, 3)
                    ret_dict[f"{side}_hand_rot"] = FIXED_LEFT_HAND_ROT.reshape(1, 1, 3, 3)
                    if side == "left":
                        ret_dict["left_hand_fixed"] = True
        else:
            # Mirror left-only grasp to right hand
            l_trans = self._get_hand_translation(grasp_data["hand"]["left"])
            l_rot = sciR.from_rotvec(grasp_data["hand"]["left"]["rot"]).as_matrix()
            ret_dict["right_hand_trans"] = (self._MIRROR @ l_trans).reshape(1, 1, 3)
            ret_dict["right_hand_rot"] = (self._MIRROR @ l_rot @ self._MIRROR).reshape(1, 1, 3, 3)
            ret_dict["left_hand_trans"] = FIXED_LEFT_HAND_TRANS.reshape(1, 1, 3)
            ret_dict["left_hand_rot"] = FIXED_LEFT_HAND_ROT.reshape(1, 1, 3, 3)
            ret_dict["left_hand_fixed"] = True

        ret_dict.setdefault("left_hand_fixed", False)

        if getattr(self.config, "load_mano_params", False):
            self._extract_mano_params(grasp_data, mirrored, ret_dict)

    def _get_hand_translation(self, hand_data):
        if self.hand_pos_source == "wrist":
            return np.asarray(hand_data["trans"], dtype=np.float32)

        if "index_mcp_pos" not in hand_data:
            raise KeyError(
                "Missing hand['index_mcp_pos'] while hand_pos_source='index_mcp'. "
                "Run the human_preprocess task first."
            )
        return np.asarray(hand_data["index_mcp_pos"], dtype=np.float32)

    def _extract_mano_params(self, grasp_data, mirrored, ret_dict):
        """Extract MANO parameters for visualization."""
        if not mirrored:
            for side, is_active in grasp_data["hand"].items():
                if is_active:
                    ret_dict[f"{side}_mano_pose"] = np.asarray(grasp_data["hand"][side]["mano_pose"]).flatten()
                    ret_dict[f"{side}_mano_betas"] = np.asarray(grasp_data["hand"][side]["mano_betas"]).flatten()
                else:
                    ret_dict[f"{side}_mano_pose"] = np.zeros(self.mano_pose_dim, dtype=np.float32)
                    ret_dict[f"{side}_mano_betas"] = np.zeros(10, dtype=np.float32)
        else:
            left_mano_pose = np.asarray(grasp_data["hand"]["left"]["mano_pose"]).flatten().astype(np.float32)
            ret_dict["right_mano_pose"] = self._mirror_mano_pose(left_mano_pose)
            ret_dict["right_mano_betas"] = np.asarray(grasp_data["hand"]["left"]["mano_betas"]).flatten()
            ret_dict["left_mano_pose"] = np.zeros(self.mano_pose_dim, dtype=np.float32)
            ret_dict["left_mano_betas"] = np.zeros(10, dtype=np.float32)

    def _mirror_mano_pose(self, mano_pose):
        """
        Mirror MANO axis-angle joints from left hand to right hand across the YZ plane.
        For reflection S=diag(-1,1,1), mirrored rotvec is det(S)*S*w = [1,-1,-1]*w.
        """
        pose = np.asarray(mano_pose).flatten().astype(np.float32)
        if pose.size == 45:
            return (pose.reshape(-1, 3) * self._MIRRORED_ROTVEC_SIGN[None, :]).reshape(-1)
        return pose

    def _get_grasp_sequence_id(self, grasp_data):
        """Read the sequence id associated with one grasp record.

        Args:
            grasp_data: Loaded human grasp dictionary.

        Returns:
            Sequence id string such as ``seq_0`` when available, otherwise
            ``None``.
        """
        object_data = grasp_data.get("object", {})
        sequence_id = object_data.get("sequence_id")
        if sequence_id is not None:
            return str(sequence_id)

        source_scene = object_data.get("source_scene")
        if source_scene is not None and "_seq_" in str(source_scene):
            return f"seq_{str(source_scene).rsplit('_seq_', 1)[1]}"
        return None

    def _get_scene_sequence_id(self, scene_cfg):
        """Read the sequence id associated with one test scene config.

        Args:
            scene_cfg: Loaded scene config dictionary.

        Returns:
            Sequence id string such as ``seq_0`` when available, otherwise
            ``None``.
        """
        object_data = scene_cfg.get("object", {})
        sequence_id = object_data.get("sequence_id")
        if sequence_id is not None:
            return str(sequence_id)

        scene_id = scene_cfg.get("scene_id")
        if scene_id is not None and "/" in str(scene_id):
            return str(scene_id).split("/", 1)[1]
        return None

    def _pointcloud_matches_sequence(self, pc_path, sequence_id):
        """Check whether a point-cloud path belongs to the requested sequence.

        Args:
            pc_path: Candidate point-cloud path.
            sequence_id: Required sequence id, for example ``seq_0``.

        Returns:
            ``True`` when the path name or parent directories identify the same
            sequence.
        """
        if sequence_id is None:
            return False
        sequence_id = str(sequence_id)
        base_name = os.path.splitext(os.path.basename(pc_path))[0]
        path_parts = pc_path.split(os.sep)
        return (
            base_name == sequence_id
            or base_name.startswith(f"{sequence_id}_")
            or sequence_id in path_parts
        )

    def _pointcloud_sequence_match_rank(self, pc_path, sequence_id):
        """Rank how directly a point-cloud path matches a sequence id.

        Args:
            pc_path: Candidate point-cloud path.
            sequence_id: Required sequence id, for example ``seq_0``.

        Returns:
            Integer rank where smaller means a more direct match.
        """
        base_name = os.path.splitext(os.path.basename(pc_path))[0]
        if base_name == f"{sequence_id}_textured_pc":
            return 0
        if base_name == sequence_id:
            return 1
        if base_name.startswith(f"{sequence_id}_"):
            return 2
        return 3

    def _select_pointcloud_path(self, obj_name, sequence_id=None):
        """Select a point-cloud path for one object and optional sequence.

        Args:
            obj_name: Canonical object id used under ``object_pc_folder``.
            sequence_id: Sequence id that must be matched when cross-sequence
                point-cloud sampling is disabled.

        Returns:
            Path to one selected point-cloud file.
        """
        if obj_name not in self.pc_path_dict:
            self.pc_path_dict[obj_name] = sorted(glob(pjoin(self.object_pc_folder, obj_name, "**.npy")))
        if not self.pc_path_dict[obj_name]:
            raise FileNotFoundError(
                f"No point-cloud files found for object '{obj_name}' under "
                f"{pjoin(self.object_pc_folder, obj_name)}"
            )

        pc_candidates = self.pc_path_dict[obj_name]
        if not self.random_pc_across_sequences:
            pc_candidates = [
                pc_path
                for pc_path in pc_candidates
                if self._pointcloud_matches_sequence(pc_path, sequence_id)
            ]
            if not pc_candidates:
                raise FileNotFoundError(
                    f"No point-cloud file for object '{obj_name}' matches sequence_id='{sequence_id}' "
                    f"under {pjoin(self.object_pc_folder, obj_name)}"
                )
            pc_candidates = sorted(
                pc_candidates,
                key=lambda pc_path: (self._pointcloud_sequence_match_rank(pc_path, sequence_id), pc_path),
            )
            return pc_candidates[0]
        return random.choice(pc_candidates)

    def _load_pointcloud(self, obj_name, obj_scale, obj_pose, mirrored=False, sequence_id=None):
        """Load and transform object point cloud.

        Args:
            obj_name: Canonical object id used under ``object_pc_folder``.
            obj_scale: Scalar or per-axis object scale.
            obj_pose: Object pose matrix from object frame to world frame.
            mirrored: Whether to mirror the point cloud across the YZ plane.
            sequence_id: Sequence id to use when same-sequence point-cloud
                sampling is required.

        Returns:
            Tuple of transformed point cloud and selected point-cloud path.
        """
        pc_path = self._select_pointcloud_path(obj_name, sequence_id=sequence_id)

        raw_pc = np.load(pc_path, allow_pickle=True)
        idx = np.random.choice(raw_pc.shape[0], self.config.num_points, replace=True)
        scaled_pc = raw_pc[idx] * obj_scale

        R, t = obj_pose[:3, :3], obj_pose[:3, 3]
        pc = np.matmul(scaled_pc, R.T) + t

        if mirrored:
            pc = pc.copy()
            pc[:, 0] = -pc[:, 0]

        return pc, pc_path

    def _load_train_data(self, sample_index, ret_dict):
        """Load one record-level training/eval sample.

        Args:
            sample_index: Dataset index received by ``__getitem__``.
            ret_dict: Sample dictionary that will be filled in-place.

        Returns:
            Tuple ``(grasp_type, pc, mirrored, grasp_data)`` for one record.
        """
        if self.type_balancing_enabled and self.type_sampler_enabled:
            grasp_path = self._sample_balanced_grasp_path()
        else:
            grasp_path = self._sample_unbiased_grasp_path(sample_index)
        grasp_data = np.load(grasp_path, allow_pickle=True).item()

        ret_dict["path"] = grasp_path
        self._attach_pose_group_target(grasp_path, ret_dict)

        grasp_type, mirrored = self._determine_grasp_type(grasp_data)
        self._extract_hand_poses(grasp_data, mirrored, ret_dict)

        obj_name = grasp_data["object"]["name"]
        obj_scale = grasp_data["object"]["rel_scale"]
        obj_pose = grasp_data["object"]["pose"]
        sequence_id = self._get_grasp_sequence_id(grasp_data)

        pc, pc_path = self._load_pointcloud(obj_name, obj_scale, obj_pose, mirrored, sequence_id=sequence_id)
        ret_dict["pc_path"] = pc_path

        return grasp_type, pc, mirrored, grasp_data

    def _load_test_data(self, id, ret_dict):
        """Load test data."""
        grasp_type = self.grasp_type_lst[id // len(self.test_cfg_lst)]
        scene_path = self.test_cfg_lst[id % len(self.test_cfg_lst)]

        scene_cfg = np.load(scene_path, allow_pickle=True).item()
        obj_name = scene_cfg["object"]["name"]
        obj_scale = scene_cfg["object"]["rel_scale"]
        obj_pose = scene_cfg["object"]["pose"]
        sequence_id = self._get_scene_sequence_id(scene_cfg)

        pc, pc_path = self._load_pointcloud(obj_name, obj_scale, obj_pose, sequence_id=sequence_id)

        ret_dict["save_path"] = pjoin(self.config.name, grasp_type, scene_cfg["scene_id"], os.path.basename(pc_path))
        ret_dict["scene_path"] = scene_path
        ret_dict["pc_path"] = pc_path

        return grasp_type, pc

    def _apply_pc_centering(self, pc, ret_dict, mirrored, grasp_data):
        """Center point cloud and adjust hand poses accordingly."""
        pc_centroid = np.mean(pc, axis=-2, keepdims=True)
        pc = pc - pc_centroid
        ret_dict["pc_centroid"] = pc_centroid.astype(np.float32)

        if self.mode != "test":
            if "right_hand_trans" in ret_dict:
                ret_dict["right_hand_trans"] -= pc_centroid[None, :, :]
            if "left_hand_trans" in ret_dict and not ret_dict.get("left_hand_fixed", False):
                ret_dict["left_hand_trans"] -= pc_centroid[None, :, :]

        return pc

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

    def _apply_scale_aug(self, pc, ret_dict):
        """Apply train-time uniform scale augmentation to object and active hands.

        Args:
            pc: Centered object point cloud with shape ``(N, 3)``.
            ret_dict: Sample dictionary containing active hand translation targets.

        Returns:
            The scaled point cloud. Hand translations in ``ret_dict`` are updated
            in-place with the same factor.
        """
        scale = self._sample_scale_aug()
        ret_dict["scale_aug_factor"] = np.float32(scale)
        if scale == 1.0:
            return pc.astype(np.float32, copy=False)

        pc = pc * np.float32(scale)
        for side in ["right", "left"]:
            if f"{side}_hand_trans" not in ret_dict:
                continue
            if side == "left" and ret_dict.get("left_hand_fixed", False):
                continue
            ret_dict[f"{side}_hand_trans"] = ret_dict[f"{side}_hand_trans"] * np.float32(scale)
        return pc.astype(np.float32, copy=False)

    def _sample_scale_aug(self):
        """Sample a train-time uniform scale augmentation factor.

        Args:
            None.

        Returns:
            A positive scalar scale factor. ``1.0`` means no scaling.
        """
        if not bool(getattr(self.config, "scale_aug", False)):
            return 1.0
        scale_min = float(getattr(self.config, "scale_min", 1.0))
        scale_max = float(getattr(self.config, "scale_max", 1.0))
        if scale_min <= 0.0 or scale_max <= 0.0:
            raise ValueError("Scale augmentation bounds must be positive")
        if scale_min > scale_max:
            raise ValueError(f"scale_min must be <= scale_max, got {scale_min} > {scale_max}")
        if scale_min == scale_max:
            return scale_min
        return float(np.random.uniform(scale_min, scale_max))

    def _apply_geometric_aug(self, pc, ret_dict):
        """Apply train-time rotation and translation after point-cloud centering."""
        rot = self._sample_rotation_aug()
        trans = self._sample_translation_aug()

        if rot is not None:
            pc = pc @ rot.T
            for side in ["right", "left"]:
                if f"{side}_hand_trans" in ret_dict:
                    if side == "left" and ret_dict.get("left_hand_fixed", False):
                        continue
                    ret_dict[f"{side}_hand_trans"] = ret_dict[f"{side}_hand_trans"] @ rot.T
                    ret_dict[f"{side}_hand_rot"] = rot @ ret_dict[f"{side}_hand_rot"]

        if trans is not None:
            pc = pc + trans.reshape(1, 3)
            for side in ["right", "left"]:
                if f"{side}_hand_trans" in ret_dict:
                    if side == "left" and ret_dict.get("left_hand_fixed", False):
                        continue
                    ret_dict[f"{side}_hand_trans"] = ret_dict[f"{side}_hand_trans"] + trans.reshape(1, 1, 3)

        return pc

    def _sample_rotation_aug(self):
        z_rotation_enabled = bool(
            getattr(self.config, "z_rotation_aug", getattr(self.config, "rotation_aug", False))
        )
        xy_rotation_enabled = bool(getattr(self.config, "xy_rotation_aug", False))
        if not z_rotation_enabled and not xy_rotation_enabled:
            return None

        z_angle = np.random.uniform(-np.pi, np.pi) if z_rotation_enabled else 0.0
        xy_angle_limit = np.deg2rad(float(getattr(self.config, "xy_rotation_max_angle_deg", 0.0)))
        x_angle, y_angle = (
            np.random.uniform(-xy_angle_limit, xy_angle_limit, size=2) if xy_rotation_enabled else (0.0, 0.0)
        )

        # Compose all enabled axis rotations into one matrix so point clouds and hands share one rigid transform.
        return sciR.from_euler("xyz", [x_angle, y_angle, z_angle]).as_matrix().astype(np.float32)

    def _sample_translation_aug(self):
        if not bool(getattr(self.config, "translation_aug", False)):
            return None
        translation_range = float(getattr(self.config, "translation_range", 0.0))
        if translation_range <= 0.0:
            return None
        return np.random.uniform(-translation_range, translation_range, size=3).astype(np.float32)

    def _apply_pc_noise_aug(self, pc):
        """Add clipped zero-mean Gaussian noise to object points during training.

        Args:
            pc: Object point cloud with shape ``(N, 3)``.

        Returns:
            Point cloud with optional clipped Gaussian jitter applied.
        """
        noise_scale = float(getattr(self.config, "pc_noise_scale", 0.0))
        if noise_scale <= 0.0:
            return pc.astype(np.float32, copy=False)
        noise = np.random.normal(loc=0.0, scale=noise_scale, size=pc.shape).astype(np.float32)
        clip_multiplier = float(getattr(self.config, "pc_noise_clip_multiplier", 3.0))
        if clip_multiplier > 0.0:
            clip_value = np.float32(clip_multiplier * noise_scale)
            noise = np.clip(noise, -clip_value, clip_value)
        return pc.astype(np.float32, copy=False) + noise

    def _apply_point_dropout_aug(self, pc):
        """Randomly drop object points and resample to the original count.

        Args:
            pc: Object point cloud with shape ``(N, 3)``.

        Returns:
            Point cloud with the same shape as ``pc`` after optional point
            dropout and replacement sampling. Hand targets are unchanged
            because this augmentation only changes point visibility/density.
        """
        if not bool(getattr(self.config, "point_dropout_aug", False)):
            return pc.astype(np.float32, copy=False)

        dropout_ratio = float(getattr(self.config, "point_dropout_ratio", 0.0))
        if dropout_ratio <= 0.0:
            return pc.astype(np.float32, copy=False)
        if dropout_ratio >= 1.0:
            raise ValueError(f"point_dropout_ratio must be < 1.0, got {dropout_ratio}")

        point_num = int(pc.shape[0])
        keep_num = max(int(round(point_num * (1.0 - dropout_ratio))), 1)
        keep_indices = np.random.choice(point_num, keep_num, replace=False)
        kept_pc = pc[keep_indices]
        resample_indices = np.random.choice(keep_num, point_num, replace=True)
        return kept_pc[resample_indices].astype(np.float32, copy=False)
