import json
import os
import re
import torch.utils
from torch.utils.data import DataLoader
import torch.utils.data
from torch.utils.data._utils.collate import default_collate
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
import torch
from copy import deepcopy

from .grasp_types import GRASP_TYPES

from .base_dex import DexDataset
from .human_dex import HumanDexDataset
from .human_bidex import HumanBiDexDataset
from .human_multidex import HumanMultiDexDataset
from .robot_multidex import RobotMultiDexDataset
from dexlearn.utils.config import (
    apply_type_supervision_to_data_config,
    cfg_get,
    flatten_multidex_data_config,
    resolve_type_supervision_config,
)


def _natural_sort_key(value):
    """Build a natural-sort key for strings that contain numbers.

    Args:
        value: Value to sort, usually an object id such as ``obj_10``.

    Returns:
        List containing lowercase text chunks and integer chunks so ``obj_2``
        sorts before ``obj_10``.
    """
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value))]


def _order_training_distribution_for_json(value, parent_key=None):
    """Order training distribution metadata before JSON serialization.

    Args:
        value: Nested JSON-serializable value from the distribution analysis.
        parent_key: Key that owns ``value`` in the parent dictionary.

    Returns:
        A copy of ``value`` where object-id mappings use natural key order.
    """
    if isinstance(value, dict):
        items = value.items()
        if parent_key == "object_type_counts" or all(str(key).isdigit() for key in value.keys()):
            items = sorted(items, key=lambda item: _natural_sort_key(item[0]))
        return {key: _order_training_distribution_for_json(child, key) for key, child in items}
    if isinstance(value, list):
        return [_order_training_distribution_for_json(child, parent_key) for child in value]
    return value


def _iter_leaf_datasets(dataset):
    """Iterate over leaf datasets, unwrapping ``ConcatDataset`` when needed.

    Args:
        dataset: Dataset or ConcatDataset returned by ``create_dataset``.

    Returns:
        Generator of leaf dataset instances.
    """
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        for child in dataset.datasets:
            yield from _iter_leaf_datasets(child)
    else:
        yield dataset


def _merge_distribution_analyses(dataset):
    """Merge distribution analyses from all train dataset components.

    Args:
        dataset: Train dataset, potentially a ConcatDataset.

    Returns:
        Dictionary with merged grasp-type and per-object counts.
    """
    merged_type_counts = {str(type_id): 0 for type_id in range(len(GRASP_TYPES))}
    merged_object_counts = {}
    data_num = 0
    object_num = 0
    components = []

    for component_idx, component in enumerate(_iter_leaf_datasets(dataset)):
        if not hasattr(component, "get_distribution_analysis"):
            continue
        analysis = component.get_distribution_analysis()
        components.append(analysis)
        data_num += int(analysis.get("data_num", 0))
        object_num += int(analysis.get("object_num", 0))
        for type_id, count in analysis.get("type_counts", {}).items():
            merged_type_counts[str(type_id)] = merged_type_counts.get(str(type_id), 0) + int(count)
        for obj_id, counts in analysis.get("object_type_counts", {}).items():
            merged_key = str(obj_id)
            if merged_key in merged_object_counts:
                merged_key = f"component_{component_idx}:{obj_id}"
            merged_object_counts[merged_key] = counts

    return {
        "data_num": int(data_num),
        "object_num": int(object_num),
        "type_counts": merged_type_counts,
        "object_type_counts": merged_object_counts,
        "components": components,
    }


def _compute_tempered_type_probabilities(type_counts, alpha):
    """Compute q(type) proportional to count(type)^alpha.

    Args:
        type_counts: Mapping from type id string to sample count.
        alpha: Tempering exponent. ``1`` keeps data frequency and ``0`` makes
            active types uniform.

    Returns:
        Mapping from type id string to sampling probability.
    """
    if alpha < 0.0:
        raise ValueError(f"type sampler alpha must be non-negative, got {alpha}")
    active = {str(k): int(v) for k, v in type_counts.items() if int(v) > 0}
    weights = {type_id: count ** alpha for type_id, count in active.items()}
    total = float(sum(weights.values()))
    if total <= 0.0:
        return {str(type_id): 0.0 for type_id in type_counts}
    return {str(type_id): float(weights.get(str(type_id), 0.0) / total) for type_id in type_counts}


def _compute_type_loss_weights(type_counts, reference_probs, beta):
    """Compute mean-normalized class weights for type classification loss.

    Args:
        type_counts: Mapping from type id string to sample count.
        reference_probs: Probability distribution used as the class-weight
            reference.
        beta: Inverse-frequency exponent. ``0`` returns uniform active weights.

    Returns:
        List of class weights aligned with ``GRASP_TYPES``. Inactive classes get
        zero weight.
    """
    if beta < 0.0:
        raise ValueError(f"type loss weight beta must be non-negative, got {beta}")
    weights = [0.0 for _ in GRASP_TYPES]
    active_weights = []
    for type_id in range(len(GRASP_TYPES)):
        count = int(type_counts.get(str(type_id), 0))
        prob = float(reference_probs.get(str(type_id), 0.0))
        if count <= 0 or prob <= 0.0:
            continue
        weight = prob ** (-beta) if beta > 0.0 else 1.0
        weights[type_id] = float(weight)
        active_weights.append(float(weight))

    if active_weights:
        mean_weight = float(sum(active_weights) / len(active_weights))
        if mean_weight > 0.0:
            weights = [float(weight / mean_weight) if weight > 0.0 else 0.0 for weight in weights]
    return weights


def _save_training_distribution_analysis(config, analysis):
    """Save training distribution analysis to the run output folder.

    Args:
        config: Full Hydra config for the training run.
        analysis: Distribution analysis dictionary.

    Returns:
        Path to the JSON analysis file.
    """
    output_dir = os.path.join(str(config.output_folder), str(config.wandb.id))
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "training_data_distribution.json")
    ordered_analysis = _order_training_distribution_for_json(analysis)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ordered_analysis, f, indent=2)
    return json_path


def _prepare_training_distribution_balance(train_dataset, config):
    """Analyze train data and prepare sampler/loss balancing metadata.

    Args:
        train_dataset: Dataset that will be used by the training DataLoader.
        config: Full Hydra config for the training run.

    Returns:
        Distribution analysis dictionary saved to the output folder.
    """
    resolve_type_supervision_config(config)
    analysis = _merge_distribution_analyses(train_dataset)
    type_counts = analysis["type_counts"]
    data_total = float(sum(int(v) for v in type_counts.values()))
    data_probs = {str(type_id): (int(type_counts[str(type_id)]) / data_total if data_total > 0 else 0.0) for type_id in type_counts}

    balancing_cfg = OmegaConf.select(config, "algo.supervision.balancing")
    if balancing_cfg is not None:
        balancing_enabled = bool(cfg_get(balancing_cfg, "enabled", default=False))
        sampler_cfg = cfg_get(balancing_cfg, "sampler", default=None)
        loss_weight_cfg = cfg_get(balancing_cfg, "loss_weight", default=None)
        sampler_enabled = balancing_enabled and bool(cfg_get(sampler_cfg, "enabled", default=True))
        sampler_alpha = float(cfg_get(sampler_cfg, "alpha", default=1.0))
        sampler_object_uniform = bool(cfg_get(sampler_cfg, "object_uniform", default=True))
        loss_weight_enabled = balancing_enabled and bool(cfg_get(loss_weight_cfg, "enabled", default=False))
        loss_weight_beta = float(cfg_get(loss_weight_cfg, "beta", default=0.0))
        reference_name = str(cfg_get(loss_weight_cfg, "reference", default="sampler")).lower()
    else:
        balancing_enabled = bool(getattr(config.data, "type_balancing_enabled", False))
        sampler_enabled = balancing_enabled and bool(getattr(config.data, "type_sampler_enabled", False))
        sampler_alpha = float(getattr(config.data, "type_sampler_alpha", 1.0))
        sampler_object_uniform = bool(getattr(config.data, "type_sampler_object_uniform", True))
        loss_weight_enabled = balancing_enabled and bool(getattr(config.data, "type_loss_weight_enabled", False))
        loss_weight_beta = float(getattr(config.data, "type_loss_weight_beta", 0.0))
        reference_name = str(getattr(config.data, "type_loss_weight_reference", "sampler")).lower()

    sampler_probs = _compute_tempered_type_probabilities(type_counts, sampler_alpha if sampler_enabled else 1.0)
    reference_probs = sampler_probs if reference_name == "sampler" else data_probs
    type_loss_weights = (
        _compute_type_loss_weights(type_counts, reference_probs, loss_weight_beta) if loss_weight_enabled else None
    )

    analysis["type_probabilities"] = {"data": data_probs, "sampler": sampler_probs}
    analysis["type_loss_weights"] = type_loss_weights
    analysis["type_balancing"] = {
        "enabled": balancing_enabled,
        "sampler_enabled": sampler_enabled,
        "sampler_alpha": sampler_alpha,
        "sampler_object_uniform": sampler_object_uniform,
        "loss_weight_enabled": loss_weight_enabled,
        "loss_weight_beta": loss_weight_beta,
        "loss_weight_reference": reference_name,
    }

    if type_loss_weights is not None:
        with open_dict(config.algo.model):
            config.algo.model.type_loss_weights = type_loss_weights

    json_path = _save_training_distribution_analysis(config, analysis)
    print(f"Saved training data distribution analysis to {json_path}")
    if type_loss_weights is not None:
        print(f"Using type loss weights: {type_loss_weights}")
    return analysis


def create_dataset(config, mode):
    resolve_type_supervision_config(config)
    sp_voxel_size = config.algo.model.backbone.voxel_size if "MinkUNet" in config.algo.model.backbone.name else None

    data_config = deepcopy(config.data if mode != "test" else config.test_data)
    flatten_multidex_data_config(data_config)
    if mode == "test" and not hasattr(data_config, "pc_runtime_scale"):
        with open_dict(data_config):
            data_config.pc_runtime_scale = 1.0
    apply_type_supervision_to_data_config(config, data_config, mode)

    object_path = cfg_get(data_config, "object_path", "paths.object_path")
    if isinstance(object_path, ListConfig):
        dataset_lst = []
        for p in object_path:
            new_data_config = deepcopy(data_config)
            new_data_config.object_path = p
            single_dataset = eval(data_config.dataset_type)(new_data_config, mode, sp_voxel_size)
            dataset_lst.append(single_dataset)
        dataset = torch.utils.data.ConcatDataset(dataset_lst)
    else:
        dataset = eval(data_config.dataset_type)(data_config, mode, sp_voxel_size)

    return dataset


def create_train_dataloader(config: DictConfig, train_shuffle=True):
    train_dataset = create_dataset(config, mode="train")
    _prepare_training_distribution_balance(train_dataset, config)
    val_dataset = create_dataset(config, mode="eval")
    batch_size = config.algo.batch_size
    train_drop_last = len(train_dataset) >= batch_size
    val_drop_last = len(val_dataset) >= batch_size
    persistent_workers = bool(getattr(config, "persistent_workers", False)) and config.num_workers > 0
    prefetch_factor = getattr(config, "prefetch_factor", 2)

    train_loader = InfLoader(
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            drop_last=train_drop_last,
            num_workers=config.num_workers,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor if config.num_workers > 0 else None,
            shuffle=train_shuffle,
            collate_fn=minkowski_collate_fn,
        ),
        config.device,
    )
    val_loader = InfLoader(
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            drop_last=val_drop_last,
            num_workers=config.num_workers,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor if config.num_workers > 0 else None,
            shuffle=False,
            collate_fn=minkowski_collate_fn,
        ),
        config.device,
    )
    return train_loader, val_loader


def create_test_dataloader(config: DictConfig, mode="test"):
    test_dataset = create_dataset(config, mode=mode)
    persistent_workers = bool(getattr(config, "persistent_workers", False)) and config.num_workers > 0
    prefetch_factor = getattr(config, "prefetch_factor", 2)
    test_loader = FiniteLoader(
        DataLoader(
            test_dataset,
            batch_size=config.algo.batch_size,
            drop_last=False,
            num_workers=config.num_workers,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor if config.num_workers > 0 else None,
            shuffle=False,
            collate_fn=minkowski_collate_fn,
        ),
        config.device,
    )
    return test_loader


class InfLoader:
    # a simple wrapper for DataLoader which can get data infinitely
    def __init__(self, loader: DataLoader, device: str):
        self.loader = loader
        self.iter_loader = iter(self.loader)
        self.device = device
        self._end = object()

    def get(self):
        data = next(self.iter_loader, self._end)
        if data is self._end:
            self.iter_loader = iter(self.loader)
            data = next(self.iter_loader, self._end)
            if data is self._end:
                raise RuntimeError("DataLoader is empty. Check dataset size and batch_size/drop_last.")

        for k, v in data.items():
            if type(v).__module__ == "torch":
                if "Int" not in v.type() and "Long" not in v.type() and "Short" not in v.type():
                    v = v.float()
                data[k] = v.to(self.device)
        return data


class FiniteLoader:
    # a simple wrapper for DataLoader which can get data infinitely
    def __init__(self, loader: DataLoader, device: str):
        self.loader = loader
        self.iter_loader = iter(self.loader)
        self.device = device

    def __len__(self):
        return len(self.iter_loader)

    def __iter__(self):
        return self

    def __next__(self):
        data = next(self.iter_loader)
        for k, v in data.items():
            if type(v).__module__ == "torch":
                if "Int" not in v.type() and "Long" not in v.type() and "Short" not in v.type():
                    v = v.float()
                data[k] = v.to(self.device)
        return data


# some magic to get MinkowskiEngine sparse tensor
def minkowski_collate_fn(list_data):
    scene_path_data = None
    if "scene_path" in list_data[0].keys():
        scene_path_data = [d.pop("scene_path") for d in list_data]

    coors_data = None
    if "coors" in list_data[0].keys():
        from dexlearn.utils.dependencies import require_module
        ME = require_module("MinkowskiEngine", "sparse point-cloud collation")
        coors_data = [d.pop("coors") for d in list_data]
        feats_data = [d.pop("feats") for d in list_data]
        coordinates_batch, features_batch = ME.utils.sparse_collate(coors_data, feats_data)
        coordinates_batch, features_batch, original2quantize, quantize2original = ME.utils.sparse_quantize(
            coordinates_batch,
            features_batch,
            return_index=True,
            return_inverse=True,
        )

    res = default_collate(list_data)

    if scene_path_data is not None:
        res["scene_path"] = scene_path_data

    if coors_data is not None:
        res["coors"] = coordinates_batch
        res["feats"] = features_batch
        res["original2quantize"] = original2quantize
        res["quantize2original"] = quantize2original
    return res


def get_sparse_tensor(pc: torch.tensor, voxel_size: float):
    """
    pc: (B, N, 3)
    return dict(point_clouds, coors, feats, quantize2original)
    """
    from dexlearn.utils.dependencies import require_module
    ME = require_module("MinkowskiEngine", "sparse point-cloud encoding")
    coors = pc / voxel_size
    feats = pc
    coordinates_batch, features_batch = ME.utils.sparse_collate([coor for coor in coors], [feat for feat in feats])
    coordinates_batch, features_batch, _, quantize2original = ME.utils.sparse_quantize(
        coordinates_batch.float(),
        features_batch,
        return_index=True,
        return_inverse=True,
    )
    return dict(
        point_clouds=pc,
        coors=coordinates_batch.to(pc.device),
        feats=features_batch,
        quantize2original=quantize2original.to(pc.device),
    )
