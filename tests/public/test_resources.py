"""Missing inputs must fail before model import or dataset mutation."""

from pathlib import Path
from omegaconf import OmegaConf
import pytest

from dexlearn.task import get_task
from dexlearn.utils.resources import validate_resources


def test_missing_dataset_identifies_configuration(tmp_path):
    cfg = OmegaConf.create({"task_name": "train", "data": {"paths": {"grasp_path": str(tmp_path / "missing")}}})
    with pytest.raises(FileNotFoundError, match="HUGS_DATASET_ROOT"):
        validate_resources(cfg)
    assert list(tmp_path.iterdir()) == []


def test_missing_checkpoint_identifies_parameter(tmp_path):
    cfg = OmegaConf.create({"task_name": "sample", "test_data": {"object_path": str(tmp_path)}, "ckpt": str(tmp_path / "missing.pth")})
    with pytest.raises(FileNotFoundError, match="Missing ckpt"):
        validate_resources(cfg)


def test_missing_mano_is_diagnosed(tmp_path):
    cfg = OmegaConf.create({"task_name": "human_preprocess", "data": {}, "task": {"mano_root": str(tmp_path)}})
    with pytest.raises(FileNotFoundError, match="MANO_RIGHT.pkl"):
        validate_resources(cfg)


def test_unknown_task_is_not_evaluated():
    with pytest.raises(ValueError, match="Unknown task"):
        get_task("__import__('os').system('false')")
