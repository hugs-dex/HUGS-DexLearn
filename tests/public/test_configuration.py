"""Exercise every retained configuration and task without GPU dependencies."""

import os
from pathlib import Path
import subprocess
import sys

from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import pytest

from dexlearn.task import TASK_NAMES

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "dexlearn/config"
ALGO_DATA = {
    "diffusion": "bodex_shadow", "nflow": "bodex_shadow",
    "humanDiffusion": "human", "humanNflow": "human",
    "humanBiDiffusion": "humanbi", "robotMultiHierar": "shadowMulti",
}


def resolve(overrides):
    with initialize_config_dir(config_dir=str(CONFIG), version_base=None):
        cfg = compose(config_name="base", overrides=overrides, return_hydra_config=True)
        HydraConfig.instance().set_config(cfg)
        return {key: OmegaConf.to_container(cfg[key], resolve=True) for key in ("data", "algo", "test_data", "task")}


@pytest.mark.parametrize("task", TASK_NAMES)
def test_task_cli_configuration_without_runtime(task):
    proc = subprocess.run(
        [sys.executable, "-m", "dexlearn.main", f"task={task}", "--cfg", "job", "--resolve"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert f"task_name: {task}" in proc.stdout


@pytest.mark.parametrize("config_file", sorted(CONFIG.glob("*/*.yaml")))
def test_every_retained_config_resolves(config_file):
    group, name = config_file.parent.name, config_file.stem
    overrides = [f"{group}={name}"]
    if group == "algo":
        overrides.append(f"data={ALGO_DATA.get(name, 'humanMulti')}")
    resolve(overrides)


@pytest.mark.parametrize("algo", sorted(p.stem for p in (CONFIG / "algo").glob("human*.yaml")))
def test_human_baselines_have_config_contract(algo):
    cfg = resolve([f"data={ALGO_DATA.get(algo, 'humanMulti')}", f"algo={algo}", "task=train"])
    assert "model" in cfg["algo"]


@pytest.mark.parametrize("hand", ["shadowMulti", "leapspMulti", "leapMulti"])
@pytest.mark.parametrize("task", ["train", "sample", "visualize", "robot_type_eval"])
def test_robot_workflows_resolve(hand, task):
    resolve(["algo=robotMultiHierar", f"data={hand}", f"test_data={hand}", f"task={task}"])


def test_environment_precedence_and_asset_paths(monkeypatch):
    monkeypatch.delenv("HUGS_DATASET_ROOT", raising=False)
    assert resolve([])["data"]["paths"]["grasp_path"].startswith("assets/")
    monkeypatch.setenv("HUGS_DATASET_ROOT", "/public/bundle")
    monkeypatch.setenv("HUGS_ASSET_ROOT", "/public/assets")
    cfg = resolve(["data=shadowMulti", "test_data=shadowMulti"])
    assert cfg["data"]["paths"]["grasp_path"].startswith("/public/bundle/")
    assert cfg["data"]["robot"]["urdf_path"].startswith("/public/assets/robot/")
    assert cfg["test_data"]["robot_urdf_path"].startswith("/public/assets/robot/")


def test_task_registry_does_not_import_runtimes():
    code = (
        "import sys; from dexlearn.task import task_train, TASK_NAMES; "
        "import dexlearn.main; "
        "assert not any(m in sys.modules for m in ['torch', 'MinkowskiEngine', 'manopth', 'viser'])"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_imported_console_entry_finds_packaged_configuration():
    proc = subprocess.run([
        sys.executable, "-c", "from dexlearn.main import main; main()",
        "task=sample", "--cfg", "job", "--resolve",
    ], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "task_name: sample" in proc.stdout
