"""Export viewer selection and GUI regressions, without MANO or a web server."""
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

pytest.importorskip("pytorch3d")
from dexlearn.task import visualize as viewer
from dexlearn.task import visualize_human_prior as prior


@pytest.fixture
def exported_prior(tmp_path, monkeypatch):
    for i in range(4):
        obj = tmp_path / f"object_{i}"
        obj.mkdir()
        np.save(obj / "scene.npy", {
            "scene_id": f"object_{i}/scene", "object_id": f"object_{i}",
            "budget_scores": np.arange(5) / 10,
            "index_mcp_pos": np.zeros((5, 7, 2, 3)),
            "wrist_quat": np.tile([1., 0., 0., 0.], (5, 7, 2, 1)),
            "active_hand_mask": np.ones((5, 7, 2), dtype=bool),
            "pc_path": "unused", "grasp_pos_source": "index_mcp",
        })
    monkeypatch.setattr(prior, "load_record_scene_payload", lambda record, count: (
        prior.load_prior_scene(record["scene_file"]), np.zeros((3, 3)),
    ))
    monkeypatch.setattr(prior, "add_hand_pose_elements", lambda *args, **kwargs: None)
    config = OmegaConf.create({"device": "cpu", "task": {
        "visualize_mode": "one_scene", "scene_option_count": 2,
        "one_scene_samples_per_type": 5, "random_object_count": 4,
    }})
    index = prior.build_prior_index(str(tmp_path))
    controls = prior.build_prior_selection_controls(index, config, {"value": {}})
    return index, config, controls


def load_initial(controls):
    return controls["load_scene_records"](
        controls["initial_mode"], controls["initial_object"], controls["initial_grasp_type"],
    )


def test_stable_scene_options_paging_and_next_scene(exported_prior, monkeypatch):
    index, config, controls = exported_prior
    selected = controls["initial_object"]
    options = controls["object_options_for_mode"]("one_scene")
    assert selected in options
    assert options == controls["object_options_for_mode"]("one_scene")
    first = load_initial(controls)
    assert len(first) == 25
    assert "Page 1/2; samples 1–5/7" in controls["batch_state"]["selection_info"]
    last = controls["load_scene_records"]("one_scene", selected, "0: 0_any", True)
    assert len(last) == 10
    assert {r["sample_index"] for r in last} == {5, 6}
    assert controls["batch_state"]["current_object"] == selected
    assert controls["object_options_for_mode"]("one_scene") == options
    wrapped = controls["load_scene_records"]("one_scene", selected, "0: 0_any", True)
    assert {r["sample_index"] for r in wrapped} == set(range(5))

    outside = next(f"{obj}/scene" for obj in index["object_options"] if f"{obj}/scene" not in options)
    monkeypatch.setattr(prior, "random_scene_record", lambda _: prior.find_scene_record(index, outside))
    records = controls["load_action_scene_records"]("one_scene", selected, "1: 1_right_two", "Next Scene")
    assert {r["source_scene_id"] for r in records} == {outside}
    assert controls["batch_state"]["key"] == ("one_scene", outside, 1)
    assert outside in controls["object_options_for_mode"]("one_scene")
    assert len(controls["object_options_for_mode"]("one_scene")) == 2


def test_grid_and_hidden_caption_metadata(exported_prior):
    index, config, controls = exported_prior
    records = prior.build_scene_records(index, "random_objects", "", 0, config, {}, 1)
    assert len(np.unique(viewer.build_grouped_scene_offsets(records, 0.5), axis=0)) == 4
    records = load_initial(controls)
    assert len(np.unique(viewer.build_grouped_scene_offsets(records, 0.5), axis=0)) == 25
    assert records[0]["label_caption"] == ""
    assert "scene=" in records[0]["caption"]
    assert "sample=1" in viewer.build_caption_from_aspects(records[0], {"sample"})
    assert "type_score=0.0000" in viewer.build_caption_from_aspects(records[0], {"type_score"})


def test_failed_load_preserves_selection(exported_prior, monkeypatch):
    _, _, controls = exported_prior
    load_initial(controls)
    previous = controls["batch_state"].copy()
    def fail(*args, **kwargs):
        raise ValueError("Unreadable scene")
    monkeypatch.setattr(prior, "build_scene_records", fail)
    with pytest.raises(ValueError, match="Unreadable"):
        controls["load_scene_records"]("one_scene", "other/scene", "1: 1_right_two")
    assert controls["batch_state"] == previous


class GuiHandle:
    """Minimal synchronous GUI handle with Viser's initial-value constraint."""
    def __init__(self, label="", options=None, initial_value=None, **kwargs):
        self.label, self.options = label, options
        if options is not None:
            assert initial_value in options
        self._value = initial_value
        self.disabled = False
        self.content = ""
        self.callbacks = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def on_update(self, callback):
        self.callbacks.append(callback)
        return callback

    on_click = on_update

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if value != self._value:
            self._value = value
            self.click()

    def click(self):
        for callback in self.callbacks:
            callback(None)


class Gui:
    def __init__(self):
        self.handles = {}
        self.markdown = []

    def add_dropdown(self, label, **kwargs):
        handle = GuiHandle(label, **kwargs)
        self.handles[label] = handle
        return handle

    add_button = add_dropdown
    add_folder = add_dropdown

    def add_markdown(self, content):
        handle = GuiHandle()
        handle.content = content
        self.markdown.append(handle)
        return handle


def test_gui_pending_selection_and_navigation(exported_prior, monkeypatch):
    index, _, controls = exported_prior
    records = load_initial(controls)
    gui = Gui()
    server = SimpleNamespace(gui=gui, scene=SimpleNamespace(set_up_direction=lambda _: None))
    monkeypatch.setattr(viewer, "VISER_AVAILABLE", True)
    monkeypatch.setattr(viewer, "viser", SimpleNamespace(ViserServer=lambda **kwargs: server))
    monkeypatch.setattr(viewer, "add_scene_elements_to_viser", lambda *args: [])
    monkeypatch.setattr(viewer, "add_scene_label_to_viser", lambda *args, **kwargs: None)

    def exercise_gui(_):
        # Initial one_scene dropdown must reference the actual rendered scene.
        scene = gui.handles["Scene"]
        assert scene.value == records[0]["source_scene_id"]
        assert "IK Status" not in gui.handles
        assert "Sample" in gui.handles
        status = gui.markdown[-1]
        assert "Page 1/2" in status.content
        assert sum("Grasp Type Scores:" in h.content for h in gui.markdown) == 1
        gui.handles["Grasp Type"].value = "1: 1_right_two"
        assert "Click Apply Selection" in status.content
        assert gui.handles["Next Batch"].disabled
        assert gui.handles["Next Scene"].disabled
        gui.handles["Apply Selection"].click()
        assert not gui.handles["Next Batch"].disabled
        gui.handles["Next Batch"].click()
        assert scene.value == records[0]["source_scene_id"]
        assert "Page 2/2" in status.content
        other = next(f"{obj}/scene" for obj in index["object_options"] if f"{obj}/scene" != scene.value)
        monkeypatch.setattr(prior, "random_scene_record", lambda _: prior.find_scene_record(index, other))
        gui.handles["Next Scene"].click()
        assert scene.value == other
        assert "Page 1/2" in status.content
        assert gui.handles["Grasp Type"].value == "1: 1_right_two"
        gui.handles["Mode"].value = "random_objects"
        assert gui.handles["Next Batch"].disabled
        gui.handles["Apply Selection"].click()
        assert not gui.handles["Next Batch"].disabled
        assert gui.handles["Next Scene"].disabled
        assert scene.disabled
        assert "random objects" in status.content
        gui.handles["Mode"].value = "one_scene"
        gui.handles["Apply Selection"].click()
        assert scene.value == controls["batch_state"]["current_object"]
        raise KeyboardInterrupt

    monkeypatch.setattr(viewer, "time", SimpleNamespace(sleep=exercise_gui))
    viewer.show_scenes_with_viser(
        records, port=0, scene_spacing=0.5, selection_controls=controls,
        caption_aspects=prior.PRIOR_CAPTION_ASPECTS,
    )
