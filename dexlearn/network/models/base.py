import torch

from ..backbones import *
from ..type_emb import *
from ..final_layers import *


class BaseModel(torch.nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.backbone = eval(cfg.backbone.name)(cfg.backbone)
        self.grasp_type_emb = eval(cfg.grasp_type_emb.name)(cfg.grasp_type_emb)
        cfg.head.in_feat_dim = cfg.backbone.out_feat_dim + cfg.grasp_type_emb.out_feat_dim
        self.output_head = eval(cfg.head.name)(cfg.head)

    def forward(self, data: dict):
        global_feature, local_feature = self.backbone(data)
        cond_feat = torch.cat([global_feature, self.grasp_type_emb(data, global_feature)], dim=-1)
        ret_dict = self.output_head.forward(data, cond_feat)
        return ret_dict

    def sample(self, data: dict, sample_num: int = 1):
        global_feature, local_feature = self.backbone(data)
        cond_feat = torch.cat([global_feature, self.grasp_type_emb(data, global_feature, True)], dim=-1)

        grasp_type = data["grasp_type_id"]
        result = self.output_head.sample(cond_feat, grasp_type, sample_num)

        if len(result) == 3:
            return result  # robot_pose, pred_grasp_type, log_prob
        else:
            return result  # robot_pose, log_prob
