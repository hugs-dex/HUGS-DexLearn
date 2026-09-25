"""Hydra entry point with task-local runtime dependencies."""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dexlearn.utils.config import resolve_type_supervision_config
from dexlearn.task import run_task
from dexlearn.utils.resources import validate_resources


@hydra.main(config_path="config", config_name="base", version_base=None)
def main(cfg: DictConfig):
    resolve_type_supervision_config(cfg)
    validate_resources(cfg)
    run_task(str(cfg.task_name), cfg)


if __name__ == "__main__":
    main()
