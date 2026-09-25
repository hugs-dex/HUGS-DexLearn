"""Pass a synthetic producer record through a separately installed BODex reader."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig

from dexlearn.task.obj_human_prior_export import build_scene_export_record, validate_scene_export_completeness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bodex-root", type=Path, required=True)
    args = parser.parse_args()
    bodex = args.bodex_root.resolve()
    sys.path.insert(0, str(bodex / "src"))
    from curobo.util import human_prior_seed as consumer
    assert Path(consumer.__file__).resolve().is_relative_to(bodex)
    config_dir = Path(__file__).resolve().parents[1] / "dexlearn/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="base", overrides=["task=obj_human_prior_export", "test_data=DGNMulti", "algo.test_topk=2"], return_hydra_config=True)
        HydraConfig.instance().set_config(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene, pc = root / "scene.npy", root / "pc.npy"
            np.save(scene, {"scene": {}})
            np.save(pc, np.zeros((4, 3), dtype=np.float32))
            score = {"scene_id": "object/tabletop/example", "object_id": "object", "split": "all",
                     "scene_path": str(scene), "pc_path": str(pc), "budget_scores": np.full(5, 0.2)}
            poses = {}
            for type_id in range(1, 6):
                q = np.zeros((2, 2, 4), dtype=np.float32); q[..., 0] = 1
                mask = np.ones((2, 2), dtype=bool); mask[:, 1] = type_id >= 4
                poses[type_id] = {"index_mcp_pos": np.zeros((2, 2, 3), dtype=np.float32), "wrist_quat": q, "active_hand_mask": mask}
            record = build_scene_export_record(score, poses, cfg)
            validate_scene_export_completeness(record, cfg)
            target = root / (record["scene_id"] + ".npy")
            target.parent.mkdir(parents=True)
            np.save(target, record)
            loaded, _ = consumer.load_human_prior_record(str(root), record["scene_id"])
            assert np.array_equal(loaded["index_mcp_pos"], record["index_mcp_pos"])
            assert loaded["wrist_quat"].shape == (5, 2, 2, 4)
            try:
                consumer.load_human_prior_record(str(root), "missing/scene")
            except FileNotFoundError:
                pass
            else:
                raise AssertionError("Consumer accepted a missing scene")
    print(json.dumps({"status": "passed", "input": "synthetic; no model inference", "types": [1, 2, 3, 4, 5],
                      "consumer_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=bodex, text=True).strip(),
                      "consumer_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=bodex, text=True))}, indent=2))


if __name__ == "__main__":
    main()
