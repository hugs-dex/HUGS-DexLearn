import os
import sys

import numpy as np
import hydra
from omegaconf import DictConfig
import trimesh

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dexlearn.utils.util import set_seed
from dexlearn.dataset import create_test_dataloader, GRASP_TYPES


@hydra.main(config_path="../config", config_name="base", version_base=None)
def main(config: DictConfig) -> None:
    set_seed(config.seed)
    test_loader = create_test_dataloader(config)

    for data in test_loader:
        batch_size = data["point_clouds"].shape[0]

        for i in range(batch_size):
            pc_np = data["point_clouds"][i].detach().cpu().numpy()

            grasp_type_name = "unknown"
            if "grasp_type_id" in data:
                grasp_type_id = int(data["grasp_type_id"][i])
                grasp_type_name = GRASP_TYPES[grasp_type_id] if 0 <= grasp_type_id < len(GRASP_TYPES) else str(grasp_type_id)

            scene_path = data["scene_path"][i] if "scene_path" in data else "N/A"
            pc_path = data["pc_path"][i] if "pc_path" in data else "N/A"
            caption = f"grasp_type: {grasp_type_name} | scene: {scene_path} | pc: {pc_path}"

            scene_elements = [
                trimesh.points.PointCloud(pc_np, colors=[255, 165, 0, 255]),  # Orange
                trimesh.creation.axis(origin_size=0.01, axis_radius=0.001, axis_length=0.3),
            ]
            scene = trimesh.Scene(scene_elements)
            # Use an oblique Z-up-friendly initial view for tabletop scenes.
            scene.set_camera(
                angles=(np.deg2rad(80.0), 0.0, np.deg2rad(45.0)),
                distance=0.8,
                center=scene.centroid,
            )
            scene.show(caption=caption)

    return


if __name__ == "__main__":
    main()
