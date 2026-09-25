import torch
from nflows.nn.nets.resnet import ResidualNet
from dexlearn.dataset.grasp_types import GRASP_TYPES


class LearnableTypeCond(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.in_feat_dim = cfg.in_feat_dim
        # if cfg.use_predictor:
        #     self.predictor = ResidualNet(
        #         in_features=cfg.in_feat_dim,
        #         out_features=40,
        #         hidden_features=512,
        #         num_blocks=5,
        #         activation=torch.nn.ReLU(),
        #     )
        # # using cross entropy loss
        # self.predictor_loss = torch.nn.CrossEntropyLoss()
        self.grasp_type_feat = torch.nn.Embedding(num_embeddings=len(GRASP_TYPES), embedding_dim=cfg.out_feat_dim)
        return

    def forward(self, grasp_type_id, global_feature=None, predicted=False):
        if self.cfg.disabled:
            return self.grasp_type_feat(grasp_type_id * 0)
        else:
            return self.grasp_type_feat(grasp_type_id)
