"""Human sample visualization semantics without MANO or a running Viser server."""

import numpy as np
import pytest
from omegaconf import OmegaConf

pytest.importorskip("trimesh")

from dexlearn.dataset.grasp_types import GRASP_TYPES
from dexlearn.task.visualize import (
    build_human_selection_controls,
    build_human_visualization_index,
    build_human_visualization_indices_by_split,
    compact_human_label_caption,
    human_available_grasp_type_ids,
    human_available_modes,
    human_object_score_summary,
)


def record(object_id, group, rank=0, scores=None):
    data = {"pred_grasp_type_prob": np.asarray(scores)} if scores is not None else {}
    return {
        "object_id": object_id,
        "sample_group": group,
        "sample_file": f"/{group}/{object_id}/{rank}.npy",
        "data": data,
        "scene_cfg": {},
    }


def test_scores_only_use_dedicated_type_samples():
    pose = record("object_a", "1_right_two", scores=[0.9] * 5)
    pose_index = build_human_visualization_index([pose], GRASP_TYPES)
    assert human_object_score_summary(pose_index, "object_a", lambda path: path) is None
    assert human_available_grasp_type_ids(pose_index) == (1,)

    score = record("object_a", "0_any", scores=[0.1, 0.2, 0.3, 0.4, 0.5])
    mixed_index = build_human_visualization_index([pose, score], GRASP_TYPES)
    summary = human_object_score_summary(mixed_index, "object_a", lambda path: path)
    np.testing.assert_allclose(summary["scores"], score["data"]["pred_grasp_type_prob"])
    assert summary["sample_count"] == 1
    assert summary["score_records"] == [score]
    assert compact_human_label_caption("random_objects", "object_a", 0, summary) == "object_a n=1 5:0.50"


def test_all_split_contains_each_saved_record_once():
    rows = [record("object_a", "0_any"), record("object_b", "1_right_two")]
    indices = build_human_visualization_indices_by_split(
        rows, GRASP_TYPES, {"train": {"object_a"}, "test": {"object_b"}}
    )
    assert indices["all"]["records"] == rows
    assert indices["train"]["records"] == rows[:1]
    assert indices["test"]["records"] == rows[1:]
    assert human_available_modes(indices["train"]) == ("random_objects",)
    assert human_available_modes(indices["test"]) == ("random_objects", "one_object")


def test_selection_filters_modes_and_pages_without_repeating_last_page():
    scores = [record(f"object_{i}", "0_any", scores=[i] * 5) for i in range(3)]
    poses = [record("object_pose", "1_right_two", rank=i, scores=[0.99] * 5) for i in range(3)]
    indices = build_human_visualization_indices_by_split(
        scores + poses, GRASP_TYPES,
        {"train": {f"object_{i}" for i in range(3)}, "test": {"object_pose"}},
    )
    config = OmegaConf.create({"task": {"visualize_mode": "one_object", "target_grasp_type_id": 0}})
    controls = build_human_selection_controls(
        config, indices, "train", lambda entries, mode: entries,
        random_object_count=2, per_type_grasps=2,
    )
    assert controls["initial_mode"] == "random_objects"
    assert controls["mode_options_for_split"]("train") == ("random_objects",)
    assert controls["mode_options_for_split"]("test") == ("random_objects", "one_object")
    assert controls["grasp_type_options_for_split"]("test") == ("1: 1_right_two",)
    with pytest.raises(ValueError, match="No pose samples"):
        controls["load_scene_records"]("one_object", "object_0", "0: 0_any", split_name="train")

    first = controls["load_scene_records"]("random_objects", "object_0", "0: 0_any", split_name="train")
    last = controls["load_scene_records"]("random_objects", "object_0", "0: 0_any", True, "train")
    assert len(first) == 2 and len(last) == 1
    assert {entry["object_id"] for entry in first}.isdisjoint({entry["object_id"] for entry in last})
    assert "Page 2/2" in controls["batch_state"]["selection_info"]
    controls["load_scene_records"]("random_objects", "object_0", "0: 0_any", True, "train")
    assert "Returned to first page" in controls["batch_state"]["selection_info"]

    page = controls["load_scene_records"]("one_object", "object_pose", "1: 1_right_two", split_name="test")
    assert len(page) == 2
    assert all(entry["score_summary"] is None for entry in page)
    last_page = controls["load_scene_records"]("one_object", "object_pose", "1: 1_right_two", True, "test")
    assert len(last_page) == 1
    assert last_page[0]["sample_rank"] == 3
