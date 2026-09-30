"""Check external inputs before importing expensive runtimes."""

import os
from pathlib import Path

import numpy as np
from omegaconf import ListConfig, OmegaConf


def data_root():
    """Return the public dataset root."""
    return Path(os.environ.get("HUGS_DATASET_ROOT") or "assets")


def resolve_dataset_path(value, root=None):
    """Resolve a dataset-relative reference without historical path inference."""
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return str(path)
    if ".." in path.parts:
        raise ValueError(f"Dataset reference escapes HUGS_DATASET_ROOT: {value}")
    return str(Path(root) / path if root is not None else data_root() / path)


def portable_dataset_record(value, root=None):
    """Copy nested metadata, making paths under the dataset root relative.

    Use lexical paths first so immutable bundle symlinks retain their public
    layout. Numeric arrays and paths outside the dataset are left unchanged.
    """
    base = Path(os.path.abspath(os.path.expanduser(str(root if root is not None else data_root()))))
    if isinstance(value, dict):
        return {key: portable_dataset_record(item, base) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(portable_dataset_record(item, base) for item in value)
    if isinstance(value, np.ndarray) and value.dtype.kind in {"U", "O"}:
        converted = [portable_dataset_record(item, base) for item in value.flat]
        result = np.empty(value.shape, dtype=object)
        for index, item in enumerate(converted):
            result.flat[index] = item
        return result.astype(str) if value.dtype.kind == "U" else result
    if isinstance(value, (str, Path)) and os.path.isabs(value):
        try:
            return Path(os.path.abspath(value)).relative_to(base).as_posix()
        except ValueError:
            pass
    return value


def require_path(value, label, *, directory=False):
    path = Path(str(value)).expanduser()
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        raise FileNotFoundError(
            f"Missing {label}: {path}. Configure HUGS_DATASET_ROOT to a data bundle, "
            "HUGS_ASSET_ROOT for robot assets, MANO_ROOT for MANO models, or the explicit task path. "
            "This source + contract release does not supply data or checkpoints."
        )
    return path


def _dataset_paths(config, *, training):
    if isinstance(config, ListConfig):
        for item in config:
            yield from _dataset_paths(item, training=training)
        return
    for key in ("object_path", "grasp_path", "metadata_path"):
        if key == "grasp_path" and not training:
            continue
        value = OmegaConf.select(config, f"paths.{key}") or OmegaConf.select(config, key)
        if value:
            values = value if isinstance(value, ListConfig) else [value]
            for item in values:
                yield item, key, key != "metadata_path"


def validate_resources(cfg):
    """Fail clearly on missing inputs without rewriting any data."""
    task = str(cfg.task_name)
    budget_mode = OmegaConf.select(cfg, "task.mode") if task == "scene_budget" else None
    if task in {"train", "human_preprocess"} or (task == "scene_budget" and budget_mode in {"all", "build_labels"}):
        for value, key, directory in _dataset_paths(cfg.data, training=True):
            require_path(value, f"data.{key}", directory=directory)
    elif task in {"sample", "obj_human_prior_export"}:
        for value, key, directory in _dataset_paths(cfg.test_data, training=False):
            require_path(value, f"test_data.{key}", directory=directory)
    if task == "scene_budget" and budget_mode == "train_head":
        for key in ("scene_csv", "summary_json"):
            value = OmegaConf.select(cfg, f"task.{key}")
            if value:
                require_path(value, f"task.{key}")
    if task in {"human_preprocess", "human_prior_format"}:
        mano = OmegaConf.select(cfg, "task.mano_root") or OmegaConf.select(cfg, "mano_root")
        for side in ("RIGHT", "LEFT"):
            require_path(Path(str(mano)) / f"MANO_{side}.pkl", f"MANO_{side}.pkl")
    if task == "visualize_human_prior":
        require_path(cfg.task.prior_dir, "task.prior_dir", directory=True)
    # Numbered selectors remain the responsibility of the Logger/export loader.
    if task in {"sample", "obj_human_prior_export"}:
        keys = ("task.score_ckpt", "task.pose_ckpt") if task == "obj_human_prior_export" else ("ckpt",)
        for key in keys:
            value = OmegaConf.select(cfg, key)
            if value and ("/" in str(value) or str(value).endswith((".pth", ".pt"))):
                require_path(value, key)
