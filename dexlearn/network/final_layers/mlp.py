import torch
import torch.nn as nn
from nflows.nn.nets.resnet import ResidualNet


class MLPRTJ(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.joint_mlp = BasicMLP(
            cfg.in_feat_dim + 9 + 3, (cfg.joint_num + 12) * cfg.traj_length
        )
        self.joint_loss = torch.nn.SmoothL1Loss(reduction="none")
        return

    def forward(self, data, global_feature):
        return

    def sample(self, data, global_feature, sample_num):
        return


class BasicMLP(nn.Module):
    """
    A mlp with 2 residual blocks
    # A 4 layer NN with ReLU as activation function
    """

    def __init__(self, Ni, No, Nh=64, mask=None):
        super(BasicMLP, self).__init__()
        self.mask = mask
        self.net = ResidualNet(
            Ni,
            No,
            hidden_features=Nh,
            num_blocks=2,
            dropout_probability=0.0,
            use_batch_norm=False,
        )

    def apply_mask(self):
        pass

    def forward(self, x: torch.Tensor):
        return self.net(x)
