"""Training dispatch and independent-branch isolation after removing staged training."""

import importlib
from pathlib import Path
from unittest.mock import Mock

from omegaconf import OmegaConf
import pytest

train = importlib.import_module("dexlearn.task.train")
CONFIG = Path(__file__).resolve().parents[1] / "dexlearn/config/algo"
SINGLE_STAGE_CONFIGS = [
    path for path in sorted(CONFIG.glob("*.yaml"))
    if OmegaConf.select(OmegaConf.load(path), "training.mode", default="single_stage") == "single_stage"
]


@pytest.mark.parametrize("path", SINGLE_STAGE_CONFIGS, ids=lambda path: path.stem)
def test_algorithms_without_training_mode_use_single_stage(path, monkeypatch):
    cfg = OmegaConf.create({"algo": OmegaConf.load(path)})
    runner = Mock()
    monkeypatch.setattr(train, "_task_train_single", runner)
    assert train._training_mode(cfg) == "single_stage"
    train.task_train(cfg)
    runner.assert_called_once_with(cfg)


@pytest.mark.parametrize("mode", ["legacy_shared_encoder_two_stage", "two_stage_diffusion_then_frozen_type_head"])
def test_removed_training_modes_fail_before_training(mode, monkeypatch):
    runner = Mock()
    monkeypatch.setattr(train, "_task_train_single", runner)
    with pytest.raises(ValueError, match="Unsupported algo.training.mode"):
        train.task_train(OmegaConf.create({"algo": {"training": {"mode": mode}}}))
    runner.assert_not_called()


def test_main_independent_training_still_launches_two_fresh_branches(monkeypatch):
    cfg = OmegaConf.create({
        "algo": OmegaConf.load(CONFIG / "humanMultiHierar.yaml"),
        "exp_name": "main", "data_name": "humanMulti", "algo_name": "humanMultiHierar",
        "ckpt": "previous.pth", "resume": True,
        "wandb": {"id": "original", "resume": True}, "model_registry": {"key_features": None},
    })
    runner = Mock()
    monkeypatch.setattr(train, "_task_train_single", runner)
    monkeypatch.setattr(train.wandb, "finish", Mock())
    train.task_train(cfg)
    type_cfg, pose_cfg = [call.args[0] for call in runner.call_args_list]
    assert (type_cfg.exp_name, pose_cfg.exp_name) == ("main_type", "main_diffusion")
    assert (type_cfg.algo.max_iter, pose_cfg.algo.max_iter) == (300, 10000)
    assert type_cfg.algo.model.train_type_only
    assert not pose_cfg.algo.model.train_type_only
    assert type_cfg.algo.freeze.output_head and type_cfg.algo.freeze.grasp_type_emb
    assert pose_cfg.algo.freeze.type_classifier
    for branch in (type_cfg, pose_cfg):
        assert branch.ckpt is None and not branch.resume and not branch.wandb.resume
        assert not branch.algo.freeze.backbone
    assert cfg.ckpt == "previous.pth" and cfg.resume


def test_joint_single_stage_keeps_main_model_and_trains_all_modules(monkeypatch):
    cfg = OmegaConf.create({
        "algo": OmegaConf.load(CONFIG / "humanMultiHierar.yaml"),
        "model_registry": {"key_features": None},
    })
    cfg.algo.training.mode = "joint_single_stage"
    runner = Mock()
    monkeypatch.setattr(train, "_task_train_single", runner)
    train.task_train(cfg)
    runner.assert_called_once()
    trained = runner.call_args.args[0]
    assert trained.algo.model.name == "HierarchicalTypeObjectiveModel"
    assert not trained.algo.model.train_type_only
    assert not any(trained.algo.freeze.values())
