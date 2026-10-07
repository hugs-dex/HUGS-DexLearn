"""Robot availability behavior and checkpoint compatibility without CUDA."""

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
import pytest

from dexlearn.network.models import robot_hierarchical as robot_module


class FakeBackbone(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.projection = torch.nn.Linear(4, cfg.out_feat_dim, bias=False)

    def forward(self, data):
        features = self.projection(data["object_feature"])
        return features, features[:, None]


class FakePoseHead(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.projection = torch.nn.Linear(cfg.in_feat_dim, 14)

    def forward(self, data, condition):
        return {"loss_diffusion": self.projection(condition).square().mean()}

    def sample(self, condition, type_ids, sample_num):
        pose = self.projection(condition)[:, None, None].expand(-1, sample_num, 1, -1)
        return pose, torch.zeros(condition.shape[0], sample_num)


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(robot_module, "FakeBackbone", FakeBackbone, raising=False)
    monkeypatch.setattr(robot_module, "FakePoseHead", FakePoseHead, raising=False)
    cfg = OmegaConf.create({
        "type_objective": "availability",
        "type_availability": {"score_threshold": 0.7, "min_available_types": 1, "use_score_prior": True},
        "backbone": {"name": "FakeBackbone", "out_feat_dim": 8},
        "grasp_type_emb": {"name": "LearnableTypeCond", "in_feat_dim": 8, "out_feat_dim": 3, "disabled": False},
        "head": {"name": "FakePoseHead", "in_feat_dim": None},
    })
    return robot_module.RobotHierarchicalModel(cfg)


def batch():
    return {
        "object_feature": torch.randn(2, 4),
        "grasp_type_id": torch.tensor([1, 5]),
        "right_hand_trans": torch.zeros(2, 1, 1, 3),
        "target_type_availability": torch.tensor([[1., 0., 1., 0., 0.], [0., 1., 0., 1., 1.]]),
    }


def test_forward_uses_five_binary_targets_and_trains_both_heads(model):
    data = batch()
    logits = model.type_classifier(model.backbone(data)[0])
    losses = model(data)
    torch.testing.assert_close(
        losses["loss_type"], F.binary_cross_entropy_with_logits(logits[:, 1:], data["target_type_availability"])
    )
    sum(losses.values()).backward()
    for module in (model.backbone, model.type_classifier, model.grasp_type_emb, model.output_head):
        assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in module.parameters())
    assert model.type_classifier[-1].weight.grad[0].count_nonzero() == 0


@pytest.mark.parametrize("threshold,expected_types", [(0.7, [2, 4]), (0.99, [2])])
def test_sampling_threshold_fallback_explicit_type_and_score_prior(model, threshold, expected_types):
    model.availability_score_threshold = threshold
    with torch.no_grad():
        model.type_classifier[-1].weight.zero_()
        # The large placeholder logit must never become a generated mode.
        model.type_classifier[-1].bias.copy_(torch.tensor([100., -2., 2., -1., 1., -3.]))
    data = batch()
    data["grasp_type_id"] = torch.tensor([0, 5])
    poses, types, scores, log_prob = model.sample(data, sample_num=2)
    assert poses.shape == (2, 10, 1, 14)
    assert scores.shape == (2, 10, 5)
    valid = log_prob > torch.finfo(log_prob.dtype).min
    assert types[0, valid[0]].tolist() == [type_id for type_id in expected_types for _ in range(2)]
    assert types[1, valid[1]].tolist() == [5, 5]
    expected_scores = torch.sigmoid(torch.tensor([-2., 2., -1., 1., -3.]))
    torch.testing.assert_close(scores, expected_scores.expand(2, 10, 5))
    for row in range(2):
        torch.testing.assert_close(log_prob[row, valid[row]], expected_scores[types[row, valid[row]] - 1].log())
    model.availability_use_score_prior = False
    without_prior = model.sample(data, sample_num=2)[-1]
    torch.testing.assert_close(without_prior[valid], torch.zeros_like(without_prior[valid]))
    assert torch.equal(without_prior > torch.finfo(without_prior.dtype).min, valid)


def test_existing_robot_state_dict_loads_strictly(model, tmp_path):
    # Original robot parameter names/shapes, including the unused sixth logit.
    shapes = {
        "backbone.projection.weight": (8, 4),
        "type_classifier.0.weight": (256, 8),
        "type_classifier.0.bias": (256,),
        "type_classifier.2.weight": (6, 256),
        "type_classifier.2.bias": (6,),
        "grasp_type_emb.grasp_type_feat.weight": (6, 3),
        "output_head.projection.weight": (14, 11),
        "output_head.projection.bias": (14,),
    }
    state = {name: torch.randn(shape) for name, shape in shapes.items()}
    path = tmp_path / "robot.pth"
    torch.save({"model": state}, path)
    model.load_state_dict(torch.load(path, map_location="cpu")["model"], strict=True)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name])
    assert all(torch.isfinite(loss) for loss in model(batch()).values())


def test_robot_model_rejects_ce_objective(model):
    model.cfg.type_objective = "ce"
    with pytest.raises(ValueError, match="only supports type_objective=availability"):
        robot_module.RobotHierarchicalModel(model.cfg)


def test_availability_targets_are_required_and_shape_checked(model):
    data = batch()
    data.pop("target_type_availability")
    with pytest.raises(KeyError, match="target_type_availability"):
        model(data)
    data["target_type_availability"] = torch.zeros(2, 6)
    with pytest.raises(ValueError, match="must have shape"):
        model(data)
