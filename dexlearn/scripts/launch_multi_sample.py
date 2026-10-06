#!/usr/bin/env python3
"""Launch sample jobs needed by ``task=type_eval`` across GPUs.

This script intentionally only generates saved samples consumed by
``dexlearn/main.py task=type_eval``. It launches two evaluation products:
``score`` samples from ``0_any`` and fixed-type ``pose`` samples from real
grasp types. DGN jobs can be restricted to a deterministic random subset so
quick evaluation does not require sampling the full DGN split.

Example:
    python \
      dexlearn/scripts/launch_multi_sample.py \
      --exp-names example_a example_b \
      --gpus 0 1 \
      --dgn-test-subset-seed 7 \
      --dry-run
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


DEFAULT_EXP_NAMES = ["example"]
DEFAULT_GPUS = ["0"]
DEFAULT_TEST_DATASETS = ["humanMulti"]
DEFAULT_SAMPLE_KINDS = ["score", "pose"]

SCORE_GRASP_TYPES = '["0_any"]'
POSE_GRASP_TYPES = '["1_right_two","2_right_three","3_right_full","4_both_three","5_both_full"]'

BASE_OVERRIDES = [
    "task=sample",
    "data=humanMulti",
    "algo=humanMultiHierar",
    "device=cuda:0",
]


@dataclass(frozen=True)
class SampleJob:
    """One Hydra sample command to run.

    Args:
        exp_name: Run id such as ``example``.
        test_data: Hydra test data config name such as ``humanMulti``.
        sample_kind: ``score`` for 0_any scores or ``pose`` for explicit types.

    Returns:
        Dataclass instance describing one sample command.
    """

    exp_name: str
    test_data: str
    sample_kind: str

    @property
    def grasp_types(self) -> str:
        """Return the Hydra grasp type override for this job."""
        if self.sample_kind == "score":
            return SCORE_GRASP_TYPES
        if self.sample_kind == "pose":
            return POSE_GRASP_TYPES
        raise ValueError(f"Unknown sample kind: {self.sample_kind}")


def repo_root() -> Path:
    """Resolve the HUGS-DexLearn repository root.

    Args:
        None.

    Returns:
        Absolute path to the repository root.
    """

    return Path(__file__).resolve().parents[2]


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean flag from the environment.

    Args:
        name: Environment variable name.
        default: Value to use when the variable is unset.

    Returns:
        Boolean value parsed from common truthy strings.
    """

    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_csv_env(name: str, default: list[str]) -> list[str]:
    """Read a comma-separated environment variable.

    Args:
        name: Environment variable name.
        default: Values to use when the variable is unset.

    Returns:
        Parsed list with empty entries removed.
    """

    value = os.environ.get(name)
    if not value:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def split_overrides(overrides: str) -> list[str]:
    """Split a whitespace-separated Hydra override string.

    Args:
        overrides: String such as ``"algo.batch_size=512 test_data.mini_test=true"``.

    Returns:
        List of CLI arguments. Empty input returns an empty list.
    """

    if not overrides.strip():
        return []
    return shlex.split(overrides)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        argv: Optional explicit argv for tests.

    Returns:
        Parsed argument namespace.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default=os.environ.get("CKPT", "010000"))
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", sys.executable))
    parser.add_argument("--log-dir", default=os.environ.get("LOG_DIR", "output/multi_sample_logs"))
    parser.add_argument("--workers-per-gpu", type=int, default=int(os.environ.get("WORKERS_PER_GPU", "1")))
    parser.add_argument("--dry-run", action="store_true", default=env_flag("DRY_RUN", False))
    parser.add_argument("--exp-names", nargs="+", default=parse_csv_env("EXP_NAMES", DEFAULT_EXP_NAMES))
    parser.add_argument("--gpus", nargs="+", default=parse_csv_env("GPUS", DEFAULT_GPUS))
    parser.add_argument("--test-datasets", nargs="+", default=parse_csv_env("TEST_DATASETS", DEFAULT_TEST_DATASETS))
    parser.add_argument("--sample-kinds", nargs="+", default=parse_csv_env("SAMPLE_KINDS", DEFAULT_SAMPLE_KINDS))
    parser.add_argument("--common-extra-overrides", default=os.environ.get("COMMON_EXTRA_OVERRIDES", ""))
    parser.add_argument("--score-extra-overrides", default=os.environ.get("SCORE_EXTRA_OVERRIDES", ""))
    parser.add_argument("--pose-extra-overrides", default=os.environ.get("POSE_EXTRA_OVERRIDES", ""))
    parser.add_argument("--dgn-test-object-num", type=int, default=int(os.environ.get("DGN_TEST_OBJECT_NUM", "0")))
    parser.add_argument("--dgn-test-scene-num", type=int, default=int(os.environ.get("DGN_TEST_SCENE_NUM", "1000")))
    parser.add_argument("--dgn-test-subset-seed", type=int, default=int(os.environ.get("DGN_TEST_SUBSET_SEED", "0")))
    parser.add_argument("--human-test-object-num", type=int, default=int(os.environ.get("HUMAN_TEST_OBJECT_NUM", "0")))
    parser.add_argument("--human-test-scene-num", type=int, default=int(os.environ.get("HUMAN_TEST_SCENE_NUM", "0")))
    parser.add_argument("--human-test-subset-seed", type=int, default=int(os.environ.get("HUMAN_TEST_SUBSET_SEED", "0")))
    return parser.parse_args(argv)


def build_jobs(exp_names: Iterable[str], test_datasets: Iterable[str], sample_kinds: Iterable[str]) -> list[SampleJob]:
    """Build the full sample job matrix.

    Args:
        exp_names: Run ids to sample.
        test_datasets: Test data config names.
        sample_kinds: Sample products, usually ``score`` and ``pose``.

    Returns:
        Ordered list of sample jobs.
    """

    jobs: list[SampleJob] = []
    for exp_name in exp_names:
        for test_data in test_datasets:
            for sample_kind in sample_kinds:
                jobs.append(SampleJob(exp_name=exp_name, test_data=test_data, sample_kind=sample_kind))
    return jobs


def command_for_job(args: argparse.Namespace, job: SampleJob) -> list[str]:
    """Build the subprocess command for one sample job.

    Args:
        args: Parsed script arguments.
        job: Sample job description.

    Returns:
        Command list suitable for ``subprocess.run``.
    """

    kind_extra = args.score_extra_overrides if job.sample_kind == "score" else args.pose_extra_overrides
    command = [
        args.python_bin,
        "dexlearn/main.py",
        *BASE_OVERRIDES,
        f"exp_name={job.exp_name}",
        f"ckpt={args.ckpt}",
        f"test_data={job.test_data}",
        f"test_data.grasp_type_lst={job.grasp_types}",
        *split_overrides(args.common_extra_overrides),
        *split_overrides(kind_extra),
    ]
    command.extend(subset_overrides_for_job(args, job))
    return command


def subset_overrides_for_job(args: argparse.Namespace, job: SampleJob) -> list[str]:
    """Build dataset-specific random subset overrides for evaluation sampling.

    Args:
        args: Parsed script arguments.
        job: Sample job description.

    Returns:
        Hydra overrides that limit test objects/scenes for the job's dataset.
    """

    if job.test_data == "DGNMulti":
        object_num = args.dgn_test_object_num
        scene_num = args.dgn_test_scene_num
        seed = args.dgn_test_subset_seed
    elif job.test_data == "humanMulti":
        object_num = args.human_test_object_num
        scene_num = args.human_test_scene_num
        seed = args.human_test_subset_seed
    else:
        return []

    overrides: list[str] = []
    if object_num > 0:
        overrides.append(f"test_data.test_object_num={object_num}")
    if scene_num > 0:
        overrides.append(f"test_data.test_scene_num={scene_num}")
    if object_num > 0 or scene_num > 0:
        overrides.append(f"test_data.test_subset_seed={seed}")
    return overrides


def run_job(args: argparse.Namespace, job: SampleJob, gpu_id: str) -> None:
    """Run one sample job on one physical GPU.

    Args:
        args: Parsed script arguments.
        job: Sample job description.
        gpu_id: Physical GPU id for ``CUDA_VISIBLE_DEVICES``.

    Returns:
        None. Raises ``CalledProcessError`` if the sample command fails.
    """

    log_dir = Path(args.log_dir)
    log_file = log_dir / f"sample_{job.exp_name}_{job.test_data}_{job.sample_kind}.log"
    command = command_for_job(args, job)
    stamp = datetime.now().strftime("%F %T")
    print(f"[{stamp}] GPU {gpu_id}: {job.exp_name} {job.test_data} {job.sample_kind} -> {log_file}", flush=True)

    if args.dry_run:
        print("  DRY_RUN " + " ".join([f"CUDA_VISIBLE_DEVICES={gpu_id}", *command]), flush=True)
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    with log_file.open("w", encoding="utf-8") as log_handle:
        subprocess.run(command, cwd=repo_root(), env=env, stdout=log_handle, stderr=subprocess.STDOUT, check=True)


def run_worker(args: argparse.Namespace, jobs: list[SampleJob], worker_idx: int, worker_count: int, gpu_id: str) -> None:
    """Run the subset of jobs assigned to one worker slot.

    Args:
        args: Parsed script arguments.
        jobs: Full ordered job list.
        worker_idx: Index of this worker slot.
        worker_count: Total number of worker slots.
        gpu_id: Physical GPU id assigned to this worker.

    Returns:
        None. Raises on the first failed job in this worker.
    """

    for job_idx, job in enumerate(jobs):
        if job_idx % worker_count != worker_idx:
            continue
        run_job(args, job, gpu_id)


def main(argv: list[str] | None = None) -> int:
    """Run sample jobs according to CLI/env configuration.

    Args:
        argv: Optional explicit argv for tests.

    Returns:
        Process exit code.
    """

    args = parse_args(argv)
    if args.workers_per_gpu < 1:
        raise ValueError(f"--workers-per-gpu must be >= 1, got {args.workers_per_gpu}")
    if not args.gpus:
        raise ValueError("At least one GPU id is required")

    os.chdir(repo_root())
    jobs = build_jobs(args.exp_names, args.test_datasets, args.sample_kinds)
    worker_count = len(args.gpus) * args.workers_per_gpu
    print(
        f"Prepared {len(jobs)} sample job(s) across {len(args.gpus)} GPU(s) "
        f"with {args.workers_per_gpu} worker(s)/GPU.",
        flush=True,
    )

    failures: list[BaseException] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = []
        for worker_idx in range(worker_count):
            gpu_id = args.gpus[worker_idx % len(args.gpus)]
            futures.append(executor.submit(run_worker, args, jobs, worker_idx, worker_count, gpu_id))
        for future in as_completed(futures):
            try:
                future.result()
            except BaseException as exc:  # noqa: BLE001 - report all worker failures.
                failures.append(exc)

    if failures:
        for exc in failures:
            print(f"Sample worker failed: {exc}", file=sys.stderr)
        return 1

    print("All sample workers finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
