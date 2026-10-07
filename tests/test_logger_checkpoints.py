"""Ordinary experiment checkpoint recovery and explicit paths remain supported."""

from omegaconf import OmegaConf
import pytest

from dexlearn.utils import logger as logger_module


@pytest.mark.parametrize("override", [None, "000100", "step_000100.pth", "000100.pth", "external"])
def test_resume_resolves_latest_step_or_explicit_path(tmp_path, monkeypatch, override):
    monkeypatch.setattr(logger_module.wandb, "init", lambda **kwargs: None)
    ckpts = tmp_path / "run" / "ckpts"
    ckpts.mkdir(parents=True)
    for step in ("000100", "000300"):
        (ckpts / f"step_{step}.pth").touch()
    external = tmp_path / "external.pth"
    external.touch()
    cfg = OmegaConf.create({
        "output_folder": str(tmp_path),
        "ckpt": str(external) if override == "external" else override,
        "wandb": {"id": "run", "resume": True, "folder": str(tmp_path),
                  "project": "test", "group": "test", "mode": "disabled"},
    })
    logger = logger_module.Logger(cfg)
    expected = external if override == "external" else ckpts / f"step_{'000300' if override is None else '000100'}.pth"
    assert cfg.ckpt == str(expected)
    assert logger.save_ckpt_dir == str(ckpts)
    assert logger.save_test_dir == str(tmp_path / "run" / "tests")
