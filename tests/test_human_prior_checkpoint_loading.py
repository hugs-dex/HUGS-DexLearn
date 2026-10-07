"""Checkpoint routing for the retained human-prior architectures."""

from pathlib import Path
from unittest.mock import Mock

from omegaconf import OmegaConf
import pytest

from dexlearn.task import obj_human_prior_export as export


CONFIG = Path(__file__).resolve().parents[1] / "dexlearn/config"


def export_config(algo, tmp_path):
    return OmegaConf.create({
        "algo": OmegaConf.load(CONFIG / "algo" / f"{algo}.yaml"),
        "task": OmegaConf.load(CONFIG / "task/obj_human_prior_export.yaml"),
        "data": {"dataset_type": "HumanMultiDexDataset", "hand_pos_source": "index_mcp"},
        "test_data": {"object_path": "objects", "test_split": "all"},
        "seed": 0, "exp_name": "prior", "data_name": "humanMulti", "algo_name": algo,
        "ckpt": "010000", "output_folder": str(tmp_path),
        "wandb": {"id": f"humanMulti_{algo}_prior"},
    })


@pytest.mark.parametrize("algo,separate,model_names,experiments,factorization", [
    ("humanMultiHierar", False, ["HierarchicalTypeObjectiveModel"], ["prior"], "proposed_C_to_T"),
    ("humanMultiHierar", True, ["HierarchicalTypeObjectiveModel"] * 2,
     ["prior_type", "prior_diffusion"], "proposed_C_to_T"),
    ("humanMultiReverse", True, ["PoseConditionedTypeModel", "MarginalPoseDiffusionModel"],
     ["prior_type_posterior", "prior_pose_marginal"], "reverse_T_to_C"),
    ("humanMultiJoint", False, ["JointHybridDiffusionModel"], ["prior"], "joint_C_T"),
])
def test_retained_checkpoint_routes(algo, separate, model_names, experiments, factorization, tmp_path, monkeypatch):
    cfg = export_config(algo, tmp_path)
    if separate:
        cfg.task.score_ckpt = "000300"
        cfg.task.pose_ckpt = "010000"
        if algo == "humanMultiHierar":
            cfg.task.score_exp_name, cfg.task.pose_exp_name = experiments
    loaded_models = []

    def load_model(branch):
        checkpoint = tmp_path / f"{branch.exp_name}_{branch.ckpt}.pth"
        checkpoint.write_bytes(branch.exp_name.encode())
        model = object()
        loaded_models.append(model)
        return model, str(checkpoint), int(branch.ckpt)

    loader = Mock(side_effect=load_model)
    monkeypatch.setattr(export, "load_export_model", loader)
    score_model, pose_model, metadata = export.load_export_models(cfg)
    branches = [call.args[0] for call in loader.call_args_list]
    assert [branch.algo.model.name for branch in branches] == model_names
    assert [branch.exp_name for branch in branches] == experiments
    assert [branch.ckpt for branch in branches] == (["000300", "010000"] if separate else ["010000"])
    for branch in branches:
        assert branch.wandb.id == f"humanMulti_{algo}_{branch.exp_name}"
    assert score_model is loaded_models[0] and pose_model is loaded_models[-1]
    assert metadata["uses_independent_models"] == separate
    assert metadata["score_checkpoint_sha256"] == export.checkpoint_sha256(metadata["score_checkpoint_path"])
    assert metadata["pose_checkpoint_sha256"] == export.checkpoint_sha256(metadata["pose_checkpoint_path"])
    manifest = export.build_manifest(cfg, metadata)
    assert manifest["factorization"] == factorization
    assert manifest["uses_independent_models"] == separate
    assert "independent_sampling" not in manifest
    expected_step = "step_010000_000300" if separate else "step_010000"
    assert Path(export.resolve_output_dir(cfg, metadata)).name == expected_step
    assert cfg.exp_name == "prior" and cfg.ckpt == "010000"


def test_removed_factorization_fails_before_loading_any_checkpoint(tmp_path, monkeypatch):
    cfg = export_config("humanMultiHierar", tmp_path)
    cfg.algo.factorization = "independent_C_T"
    loader = Mock()
    monkeypatch.setattr(export, "load_export_model", loader)
    with pytest.raises(ValueError, match="Unsupported Human Prior factorization"):
        export.load_export_models(cfg)
    loader.assert_not_called()
