"""Run real failing subprocesses to verify worker-to-shell status propagation."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "dexlearn/scripts"


def fail_python(tmp_path):
    script = tmp_path / "python-fails"
    script.write_text("#!/bin/sh\necho intentional-child-failure >&2\nexit 17\n")
    script.chmod(0o755)
    return script


def test_sample_failure_reaches_shell(tmp_path):
    proc = subprocess.run([
        sys.executable, str(SCRIPTS / "launch_multi_sample.py"),
        "--python-bin", str(fail_python(tmp_path)), "--log-dir", str(tmp_path / "logs"),
        "--exp-names", "example", "--gpus", "0", "1", "--sample-kinds", "score", "pose",
    ], capture_output=True, text=True, env={k: v for k, v in os.environ.items() if k != "DRY_RUN"})
    assert proc.returncode != 0
    assert "Sample worker failed" in proc.stderr
    assert len(list((tmp_path / "logs").glob("*.log"))) == 2


def test_sample_dry_run_does_not_create_logs_or_launch_children(tmp_path):
    log_dir = tmp_path / "logs"
    env = dict(os.environ, PYTHON_BIN=str(fail_python(tmp_path)), LOG_DIR=str(log_dir), DRY_RUN="1")
    subprocess.run([sys.executable, str(SCRIPTS / "launch_multi_sample.py"), "--exp-names", "example", "--gpus", "0"], env=env, check=True, capture_output=True)
    assert not log_dir.exists()


def test_sample_jobs_are_assigned_once(monkeypatch):
    spec = importlib.util.spec_from_file_location("public_sample_launcher", SCRIPTS / "launch_multi_sample.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    calls = []
    monkeypatch.setattr(module, "run_job", lambda args, job, gpu: calls.append(job))
    jobs = module.build_jobs(["a", "b"], ["humanMulti", "DGNMulti"], ["score", "pose"])
    for i in range(3):
        module.run_worker(None, jobs, i, 3, str(i))
    assert len(calls) == len(jobs)
    assert set(calls) == set(jobs)
