"""Human sample visualization semantics without MANO or a running Viser server."""

from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

pytest.importorskip("trimesh")

from dexlearn.dataset.grasp_types import GRASP_TYPES
from dexlearn.task.visualize import (
    add_human_scores_gui,
    build_human_selection_controls,
    build_human_visualization_index,
    build_human_visualization_indices_by_split,
    compact_human_label_caption,
    format_human_scores_panel,
    human_available_grasp_type_ids,
    human_available_modes,
    human_object_score_summary,
    load_human_scores_index,
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
    assert compact_human_label_caption("random_objects", "object_a", 0, summary) == "[0.10, 0.20, 0.30, 0.40, 0.50]"


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


def test_pose_score_panel_uses_loaded_object_and_survives_paging():
    poses = [record("object_a", "1_right_two", rank=i) for i in range(2)]
    poses.append(record("object_b", "1_right_two"))
    pose_index = build_human_visualization_index(poses, GRASP_TYPES)
    score_index = build_human_visualization_index([
        record("object_a", "0_any", scores=[0.1, 0.2, 0.3, 0.4, 0.5]),
        record("object_a", "0_any", rank=1, scores=[0.3, 0.4, 0.5, 0.6, 0.7]),
    ], GRASP_TYPES)
    config = OmegaConf.create({"task": {"visualize_mode": "one_object"}})
    controls = build_human_selection_controls(
        config, {"all": pose_index}, "all", lambda entries, mode: [{}],
        per_type_grasps=1, human_scores_index=score_index,
    )

    first = controls["load_scene_records"]("one_object", "object_a", "1: 1_right_two")
    info = first[0]["human_score_info"]
    np.testing.assert_allclose(info["summary"]["scores"], [0.2, 0.3, 0.4, 0.5, 0.6])
    assert "| 1_right_two | 0.20 |" in format_human_scores_panel(info, "/scores")
    assert "2 samples" in format_human_scores_panel(info, "/scores")
    next_page = controls["load_scene_records"]("one_object", "object_a", "1: 1_right_two", True)
    assert next_page[0]["human_score_info"]["summary"] is info["summary"]

    other = controls["load_scene_records"]("one_object", "object_b", "1: 1_right_two")
    assert "Scores unavailable for this object" in format_human_scores_panel(
        other[0]["human_score_info"], "/scores"
    )
    random_page = controls["load_scene_records"]("random_objects", "object_b", "1: 1_right_two")
    assert "human_score_info" not in random_page[0]


def test_score_index_requires_0_any_and_skips_bad_samples(tmp_path):
    with pytest.raises(ValueError, match="must contain a 0_any directory"):
        load_human_scores_index(tmp_path)
    sample_dir = tmp_path / "0_any" / "object_a"
    sample_dir.mkdir(parents=True)
    np.save(sample_dir / "bad.npy", {"unexpected": 1})
    np.save(sample_dir / "good.npy", {"pred_grasp_type_prob": np.arange(1, 6) / 10})
    index = load_human_scores_index(tmp_path)
    summary = human_object_score_summary(index, "object_a", lambda path: path)
    np.testing.assert_allclose(summary["scores"], np.arange(1, 6) / 10)
    assert summary["sample_count"] == 1


def test_gui_score_panel_updates_only_from_loaded_records():
    class Folder:
        visible = True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    folders = []
    markdowns = []

    def add_folder(*args, **kwargs):
        folder = Folder()
        folders.append(folder)
        return folder

    def add_markdown(content):
        handle = SimpleNamespace(content=content)
        markdowns.append(handle)
        return handle

    server = SimpleNamespace(gui=SimpleNamespace(add_folder=add_folder, add_markdown=add_markdown))
    info = {"object_id": "object_a", "summary": {"scores": np.arange(1, 6) / 10, "sample_count": 1}}
    update = add_human_scores_gui(server, [{"human_score_info": info}], "/scores")
    assert folders[0].visible
    assert "| 5_both_full | 0.50 |" in markdowns[0].content
    update([{}])
    assert not folders[0].visible
    update([{"human_score_info": {"object_id": "object_b", "summary": None}}])
    assert folders[0].visible
    visible_text = markdowns[0].content.replace("\u200b", "")
    assert "object_b" in visible_text
    assert "object_a" not in visible_text
