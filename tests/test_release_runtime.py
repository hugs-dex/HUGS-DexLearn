"""Real CPU module and export contracts; no private data or checkpoints."""

import copy
import importlib
from pathlib import Path
import sys

import numpy as np
from omegaconf import OmegaConf
import pytest

from dexlearn.task import TASK_NAMES, get_task
from dexlearn.task.obj_human_prior_export import (
    build_manifest, build_scene_export_record, checkpoint_sha256,
    validate_scene_export, validate_scene_export_completeness,
)


def export_fixture(tmp_path):
    cfg = OmegaConf.create({
        "seed": 7,
        "algo": {"human": True, "test_grasp_num": 4, "test_topk": 2,
                 "sample_selection": {"enabled": True, "scope": "global", "mode": "random"},
                 "model": {"name": "HierarchicalTypeObjectiveModel", "type_objective": "ce"}},
        "data": {"hand_pos_source": "index_mcp"},
        "test_data": {"object_path": "objects", "test_split": "all"},
        "task": {"samples_per_type": 2, "robot_name": "shadow_hand", "robot_size": 1.0,
                 "include_log_prob": False, "include_grasp_pose": False},
    })
    scene_path, pc_path = tmp_path / "scene.npy", tmp_path / "pc.npy"
    np.save(scene_path, {"scene": {}})
    np.save(pc_path, np.zeros((4, 3), dtype=np.float32))
    score = {"scene_id": "object/tabletop/example", "object_id": "object", "split": "all",
             "scene_path": str(scene_path), "pc_path": str(pc_path), "budget_scores": np.full(5, 0.2)}
    poses = {}
    for type_id in range(1, 6):
        quat = np.zeros((2, 2, 4), dtype=np.float32)
        quat[..., 0] = 1
        mask = np.ones((2, 2), dtype=bool)
        mask[:, 1] = type_id >= 4
        poses[type_id] = {"index_mcp_pos": np.zeros((2, 2, 3), dtype=np.float32),
                          "wrist_quat": quat, "active_hand_mask": mask}
    return cfg, build_scene_export_record(score, poses, cfg)


@pytest.mark.parametrize("name", TASK_NAMES)
def test_every_task_imports_without_optional_assets(name):
    assert callable(get_task(name))


def test_imports_resolve_outside_private_checkouts():
    for name in ("dexlearn", "torch", "diffusers", "nflows", "pytorch3d"):
        module = importlib.import_module(name)
        assert Path(module.__file__).resolve().is_relative_to(
            Path(__file__).resolve().parents[1] if name == "dexlearn" else Path(sys.prefix).resolve()
        )


def test_valid_export_and_checkpoint_manifest(tmp_path):
    cfg, record = export_fixture(tmp_path)
    validate_scene_export_completeness(record, cfg)
    ckpt = tmp_path / "synthetic.pth"
    ckpt.write_bytes(b"synthetic checkpoint hash fixture, not model weights")
    digest = checkpoint_sha256(str(ckpt))
    meta = {"checkpoint_path": str(ckpt), "checkpoint_iter": 1, "checkpoint_sha256": digest,
            "score_checkpoint_path": str(ckpt), "score_checkpoint_iter": 1,
            "score_checkpoint_sha256": digest, "pose_checkpoint_path": str(ckpt),
            "pose_checkpoint_iter": 1, "pose_checkpoint_sha256": digest, "uses_independent_models": False}
    manifest = build_manifest(cfg, meta)
    assert manifest["checkpoint_sha256"] == manifest["pose_checkpoint_sha256"] == digest
    assert len(digest) == 64
    assert manifest["model_config"]["name"] == cfg.algo.model.name
    assert manifest["coordinate_contract"]["quaternion_order"] == "wxyz"
    assert manifest["coordinate_contract"]["hand_order"] == ["right", "left"]
    assert manifest["seed"] == 7


@pytest.mark.parametrize("factorization", ["independent_C_T", "joint_C_T", "reverse_T_to_C"])
def test_other_factorization_exports_are_not_reused_as_main_prior(tmp_path, factorization):
    cfg, record = export_fixture(tmp_path)
    record["factorization"] = factorization
    with pytest.raises(ValueError, match="Existing export uses factorization"):
        validate_scene_export_completeness(record, cfg)


def test_main_prior_without_factorization_tag_remains_readable(tmp_path):
    cfg, record = export_fixture(tmp_path)
    record.pop("factorization")
    validate_scene_export_completeness(record, cfg)


@pytest.mark.parametrize("mutation,match", [
    ("shape", "shape"), ("nan", "Non-finite"), ("quat", "Quaternion norm"),
    ("type", "grasp_type_ids"), ("mask", "boolean"),
])
def test_invalid_export_rejected(tmp_path, mutation, match):
    _, record = export_fixture(tmp_path)
    if mutation == "shape": record["index_mcp_pos"] = record["index_mcp_pos"][..., :2]
    if mutation == "nan": record["budget_scores"][0] = np.nan
    if mutation == "quat": record["wrist_quat"][0, 0, 0, 0] = 2
    if mutation == "type": record["grasp_type_ids"] = np.array([5, 4, 3, 2, 1])
    if mutation == "mask": record["active_hand_mask"] = record["active_hand_mask"].astype(int)
    with pytest.raises(ValueError, match=match):
        validate_scene_export(record, 1e-3)
