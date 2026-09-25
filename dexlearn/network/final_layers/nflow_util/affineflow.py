import torch
from torch import nn
import numpy as np

import pytorch3d.transforms as pttf
from scipy import linalg as la

from ..mlp import BasicMLP


def calculate_16_r(so3_mat, real_mat, rotation, vector, ldjs=0):
    quat = pttf.matrix_to_quaternion(rotation)
    quat = so3_mat @ quat.reshape(-1, 4, 1)
    length = quat.norm(dim=-2, keepdim=True)
    t_vector = torch.einsum("nab,nb->na", real_mat, vector)
    t_rotation = pttf.quaternion_to_matrix((quat / length).reshape(-1, 4))
    return (
        t_rotation,
        t_vector,
        ldjs
        + so3_mat.det().abs().log()
        + real_mat.det().abs().log()
        - 4 * length.reshape(-1).log(),
    )


class ConditionLU(nn.Module):
    def __init__(self, in_channel) -> None:
        super().__init__()
        self.in_channel = in_channel
        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(np.copy(w_s))
        w_u = torch.from_numpy(w_u)

        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", torch.from_numpy(u_mask))
        self.register_buffer("l_mask", torch.from_numpy(l_mask))
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        # self.w_l_net = BasicMLP(feature_dim, in_channel*in_channel)
        # self.w_u_net = BasicMLP(feature_dim, in_channel*in_channel)
        # self.w_s_net = BasicMLP(feature_dim, in_channel)

    def forward(self, feature):
        w_l, w_u, w_s = torch.split(
            feature,
            [
                self.in_channel * self.in_channel,
                self.in_channel * self.in_channel,
                self.in_channel,
            ],
            dim=-1,
        )

        weight = torch.einsum(
            "ab,nbc,ncd->nad",
            self.w_p,
            (
                w_l.reshape(-1, self.in_channel, self.in_channel) * self.l_mask
                + self.l_eye
            ),
            (
                (w_u.reshape(-1, self.in_channel, self.in_channel) * self.u_mask)
                + torch.diag(self.s_sign) * torch.exp(w_s)[:, None, :]
            ),
        )
        return weight


class ActNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super(ActNorm, self).__init__()

        # Learnable parameters
        self.scale = nn.Parameter(torch.zeros(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

        # Running statistics
        self.register_buffer("initialized", torch.tensor(0))
        self.eps = eps

    def forward(self, rotation, vector, ldjs):
        if self.initialized.item() == 0:
            # print('initialize actnorm')
            bias = -vector.mean(dim=0)
            scale = -torch.log(vector.std(dim=0) + self.eps)
            self.bias.data = bias.detach()
            self.scale.data = scale.detach()
            self.initialized.fill_(1)

        # Scale and bias the normalized input
        vector = (vector + self.bias) * self.scale.exp()
        return rotation, vector, ldjs + self.scale.sum()

    def inverse(self, rotation, vector, ldjs):
        assert self.initialized.item() == 1
        vector = vector / self.scale.exp() - self.bias
        return rotation, vector, ldjs - self.scale.sum()


class Condition16TransR(nn.Module):
    def __init__(self, real_dim, feature_dim):
        super().__init__()
        self.net = BasicMLP(
            feature_dim,
            16 + real_dim * (1 + 2 * real_dim),
            mask=slice(16, 16 + real_dim * (1 + 2 * real_dim)),
        )
        self.lu = ConditionLU(real_dim)
        self.actnorm = ActNorm(real_dim)

    def forward(self, rotation, vector, permute=None, feature=None):
        new_feature = self.net(feature)
        so3_mat = new_feature[:, :16].reshape(-1, 4, 4) + torch.eye(
            4, device=rotation.device
        )
        real_mat = self.lu(new_feature[..., 16:])
        result = self.actnorm(*calculate_16_r(so3_mat, real_mat, rotation, vector))
        return result

    def inverse(self, rotation, vector, permute=None, feature=None):
        new_feature = self.net(feature)
        so3_mat = torch.linalg.inv(
            new_feature[:, :16].reshape(-1, 4, 4) + torch.eye(4, device=rotation.device)
        )
        real_mat = torch.linalg.inv(self.lu(new_feature[..., 16:]))
        result = calculate_16_r(
            so3_mat, real_mat, *self.actnorm.inverse(rotation, vector, 0)
        )
        return result

    def mask(self):
        self.net.apply_mask()
