import wandb
from os.path import join as pjoin
import torch
import os
import glob
import numpy as np
import numbers
from dexlearn.utils.resources import portable_dataset_record


class Logger:
    def __init__(self, cfg):
        self.dataset_root = getattr(cfg, "data_root", None)
        self.config = cfg.wandb
        self.save_ckpt_dir = pjoin(cfg.output_folder, self.config.id, "ckpts")
        os.makedirs(self.save_ckpt_dir, exist_ok=True)
        self.save_test_dir = pjoin(cfg.output_folder, self.config.id, "tests")
        os.makedirs(self.save_test_dir, exist_ok=True)

        wandb_resume = None
        if self.config.resume:
            all_ckpts = self._find_checkpoints()
            if len(all_ckpts) > 0:
                wandb_resume = "allow"
                if cfg.ckpt is None:
                    cfg.ckpt = all_ckpts[-1]
                else:
                    cfg.ckpt = self._resolve_checkpoint_path(cfg.ckpt)

        wandb.init(
            dir=self.config.folder,
            project=self.config.project,
            group=self.config.group,
            id=self.config.id,
            mode=self.config.mode,
            resume=wandb_resume,
        )

    def _find_checkpoints(self) -> list[str]:
        """Find checkpoints in the current experiment directory."""
        return sorted(glob.glob(pjoin(self.save_ckpt_dir, "step_*.pth")))

    def _resolve_checkpoint_path(self, ckpt) -> str:
        """Resolve a checkpoint override to an existing local checkpoint path.

        Args:
            ckpt: Checkpoint override such as ``000300``, ``step_000300.pth``,
                or an explicit path.

        Returns:
            Resolved checkpoint path when found, otherwise the original value.
        """
        ckpt_text = str(ckpt)
        if os.path.exists(ckpt_text):
            return ckpt_text
        if os.path.dirname(ckpt_text):
            return ckpt_text

        filename = ckpt_text if ckpt_text.endswith(".pth") else f"step_{ckpt_text}.pth"
        if not filename.startswith("step_"):
            filename = f"step_{filename}"

        path = pjoin(self.save_ckpt_dir, filename)
        if os.path.exists(path):
            return path
        return ckpt_text

    def log(self, dic: dict, mode: str, step: int):
        """
        log a dictionary, requires all values to be scalar
        mode is used to distinguish train, val, ...
        step is the iteration number
        """
        wandb.log({f"{mode}/{k}": v for k, v in dic.items()}, step=step)

    def save(self, dic: dict, step: int):
        """
        save a dictionary to a file
        """
        torch.save(dic, pjoin(self.save_ckpt_dir, f"step_{str(step).zfill(6)}.pth"))

    def save_samples(self, dic: dict, step: int, save_path: list):
        for i, suffix in enumerate(save_path):
            path = pjoin(self.save_test_dir, f"step_{str(step).zfill(6)}", suffix)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            for k, v in dic.items():
                if k != "candidate_valid_mask" and type(v).__module__ == "torch":
                    torch_key = k
                    break

            candidate_valid_mask = dic.get("candidate_valid_mask", None)
            saved_index = 0
            for j in range(dic[torch_key].shape[1]):
                # ``candidate_valid_mask`` allows each batch row to save a
                # different number of generated grasps while tensors remain
                # padded to a common candidate dimension.
                if candidate_valid_mask is not None and not bool(candidate_valid_mask[i, j].detach().cpu().item()):
                    continue
                save_dict = {}
                for k, v in dic.items():
                    if k == "candidate_valid_mask":
                        continue
                    if k in ["scene_path", "pc_path"]:
                        save_dict[k] = v[i]
                    elif k in ["grasp_type_id"]:
                        save_dict[k] = v[i].detach().cpu().numpy()
                    elif type(v).__module__ == "torch":
                        save_dict[k] = v[i, j].detach().cpu().numpy()
                    elif isinstance(v, (str, bytes, numbers.Number, np.generic)):
                        save_dict[k] = v
                    elif isinstance(v, np.ndarray):
                        save_dict[k] = v
                    else:
                        raise NotImplementedError
                path_j = path.split(".npy")[0] + f"_{saved_index}.npy"
                save_dict = portable_dataset_record(save_dict, self.dataset_root)
                save_dict["path_root"] = "HUGS_DATASET_ROOT"
                np.save(path_j, save_dict)
                saved_index += 1
