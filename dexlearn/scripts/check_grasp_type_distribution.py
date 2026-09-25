import sys
import os
from collections import Counter

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dexlearn.utils.util import set_seed
from dexlearn.dataset import create_dataset, GRASP_TYPES


@hydra.main(config_path="../config", config_name="base", version_base=None)
def main(config: DictConfig) -> None:
    set_seed(config.seed)

    train_dataset = create_dataset(config, mode="train")
    print(f"Total training samples: {len(train_dataset)}")

    grasp_type_ids = []
    for i in tqdm(range(len(train_dataset)), desc="Checking training data"):
        data = train_dataset[i]
        grasp_type_ids.append(int(data["grasp_type_id"]))

    print("\nTraining Data Grasp Type Distribution:")
    counter = Counter(grasp_type_ids)
    for type_id in sorted(counter.keys()):
        count = counter[type_id]
        percentage = count / len(grasp_type_ids) * 100
        print(f"  {GRASP_TYPES[type_id]}: {count} ({percentage:.1f}%)")


if __name__ == "__main__":
    main()
