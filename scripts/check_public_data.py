"""Read one object and one human dataset item from a specified immutable bundle."""

import argparse
import hashlib
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import numpy as np

from dexlearn.dataset.human_multidex import HumanMultiDexDataset
from dexlearn.dataset.robot_multidex import RobotMultiDexDataset
from dexlearn.utils.util import set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    set_seed(0)
    bundle = args.bundle.resolve()
    config_dir = Path(__file__).resolve().parents[1] / "dexlearn/config"
    report = {"scope": "one test item per dataset; no training, cache writes, MANO or checkpoints", "datasets": {}}
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        for name, cls in [("DGNMulti", RobotMultiDexDataset), ("humanMulti", HumanMultiDexDataset)]:
            cfg = compose(config_name="base", overrides=[f"data_root={bundle}", f"test_data={name}", "test_data.test_object_num=1", "test_data.test_scene_num=1"], return_hydra_config=True)
            HydraConfig.instance().set_config(cfg)
            if name == "DGNMulti": cfg.test_data.preload_point_clouds = False
            dataset = cls(cfg.test_data, mode="test", sc_voxel_size=None)
            item = dataset[0]
            pc = np.asarray(item["point_clouds"])
            assert pc.shape == (1024, 3) and np.isfinite(pc).all()
            paths = {}
            for key in ("scene_path", "pc_path"):
                path = Path(item[key])
                paths[key] = {"path": str(path.relative_to(bundle)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            report["datasets"][name] = {"status": "passed", "point_cloud_shape": list(pc.shape), "type_id": int(item["grasp_type_id"]), "inputs": paths}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
