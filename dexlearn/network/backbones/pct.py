import torch
import torch.nn as nn
from flash_attn import flash_attn_func


class PointCloudTransformerEncoder(nn.Module):
    """
    class for Point Cloud Transformer Encoder
    """
    def __init__(
        self,
        in_channels,
        feature_dim,
        use_flash_attn=False,
        use_batch_norm=False,
        dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)

        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(128)
            self.bn2 = nn.BatchNorm1d(128)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()

        self.sa1 = SA_Layer(128, use_batch_norm, use_flash_attn)
        self.sa2 = SA_Layer(128, use_batch_norm, use_flash_attn)
        self.sa3 = SA_Layer(128, use_batch_norm, use_flash_attn)
        self.sa4 = SA_Layer(128, use_batch_norm, use_flash_attn)

        self.conv_fuse = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024) if use_batch_norm else nn.Identity(),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.convs1 = nn.Conv1d(1024 * 3, 512, 1)
        self.dp1 = nn.Dropout(dropout)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, feature_dim, 1)
        if use_batch_norm:
            self.bns1 = nn.BatchNorm1d(512)
            self.bns2 = nn.BatchNorm1d(256)
        else:
            self.bns1 = nn.Identity()
            self.bns2 = nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
        - x: torch.Tensor[B, N, in_channels], input point cloud

        Returns:
        - x: torch.Tensor[B, N, feature_dim], output feature
        """
        x = x.transpose(1, 2)
        batch_size, _, N = x.shape
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv_fuse(x)
        x_max = torch.max(x, 2).values
        x_avg = torch.mean(x, 2)
        x_max_feature = x_max.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg_feature = x_avg.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        x_global_feature = torch.cat([x_max_feature, x_avg_feature], 1) # 1024 + 1024
        x = torch.cat([x, x_global_feature], 1) # 1024 * 3
        x = self.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = self.relu(self.bns2(self.convs2(x)))
        x = self.convs3(x)
        x = x.transpose(1, 2)
        return x


class SA_Layer(nn.Module):

    def __init__(self, channels, use_batch_norm, use_flash_attn):
        super(SA_Layer, self).__init__()
        self.use_flash_attn = use_flash_attn
        self.q_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        if use_batch_norm:
            self.after_norm = nn.BatchNorm1d(channels)
        else:
            self.after_norm = nn.Identity()
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
        - x: torch.Tensor[B, D, N], input feature

        Returns:
        - x: torch.Tensor[B, D, N], output feature
        """
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c
        x_k = self.k_conv(x)# b, c, n
        x_v = self.v_conv(x)

        if self.use_flash_attn:
            x_q = x_q.unsqueeze(2).to(dtype=torch.float16)
            x_k = x_k.permute(0, 2, 1).unsqueeze(2).to(dtype=torch.float16)
            x_v = x_v.permute(0, 2, 1).unsqueeze(2).to(dtype=torch.float16)
            x_r = flash_attn_func(x_q, x_k, x_v)
            x_r = x_r.to(dtype=torch.float32).squeeze(2).permute(0, 2, 1)
        else:
            energy = torch.bmm(x_q, x_k) # b, n, n
            attention = self.softmax(energy)
            attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
            x_r = torch.bmm(x_v, attention) # b, c, n

        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r

        return x

'''
import numpy as np
import torch
import torch.nn as nn
from flash_attn import flash_attn_func
from einops import rearrange


class PointCloudTransformerEncoder(nn.Module):
    """
    class for Point Cloud Transformer Encoder
    """
    def __init__(
        self,
        in_channels,
        feature_dim,
        use_flash_attn=False,
        use_batch_norm=False,
        dropout=0.5,
        n_heads=4):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)

        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(128)
            self.bn2 = nn.BatchNorm1d(128)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()

        self.sa1 = SA_Layer(128, n_heads, False, use_flash_attn)
        self.sa2 = SA_Layer(128, n_heads, use_batch_norm, use_flash_attn)
        self.sa3 = SA_Layer(128, n_heads, use_batch_norm, use_flash_attn)
        self.sa4 = SA_Layer(128, n_heads, use_batch_norm, use_flash_attn)

        self.conv_fuse = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024) if use_batch_norm else nn.Identity(),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.convs1 = nn.Conv1d(1024 * 3, 512, 1)
        self.dp1 = nn.Dropout(dropout)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, feature_dim, 1)
        if use_batch_norm:
            self.bns1 = nn.BatchNorm1d(512)
            self.bns2 = nn.BatchNorm1d(256)
        else:
            self.bns1 = nn.Identity()
            self.bns2 = nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
        - x: torch.Tensor[B, N, in_channels], input point cloud

        Returns:
        - x: torch.Tensor[B, N, feature_dim], output feature
        """
        pc = x.clone()[..., :3]
        x = x.transpose(1, 2)
        batch_size, _, N = x.shape
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x, pc)
        x2 = self.sa2(x1, pc)
        x3 = self.sa3(x2, pc)
        x4 = self.sa4(x3, pc)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv_fuse(x)
        x_max = torch.max(x, 2).values
        x_avg = torch.mean(x, 2)
        x_max_feature = x_max.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg_feature = x_avg.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        x_global_feature = torch.cat([x_max_feature, x_avg_feature], 1) # 1024 + 1024
        x = torch.cat([x, x_global_feature], 1) # 1024 * 3
        x = self.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = self.relu(self.bns2(self.convs2(x)))
        x = self.convs3(x)
        x = x.transpose(1, 2)
        return x


class SA_Layer(nn.Module):

    def __init__(self, channels, n_heads, use_batch_norm, use_flash_attn, theta=200):
        super(SA_Layer, self).__init__()
        self.n_heads = n_heads
        self.use_flash_attn = use_flash_attn
        self.q_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels, 1, bias=False)
        # init k conv with same weight as q conv
        self.k_conv.weight.data = self.q_conv.weight.data.clone()
        self.v_conv = nn.Conv1d(channels, channels, 1)
        # zero init
        # torch.nn.init.zeros_(self.v_conv.weight)
        # torch.nn.init.zeros_(self.v_conv.bias)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv2 = nn.Conv1d(channels, channels, 1)
        torch.nn.init.zeros_(self.trans_conv2.weight)
        torch.nn.init.zeros_(self.trans_conv2.bias)
        if use_batch_norm:
            self.after_norm = nn.BatchNorm1d(channels)
        else:
            self.after_norm = nn.Identity()
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.theta_num = channels // n_heads // 3 // 2
        thetas = torch.exp(torch.arange(self.theta_num, dtype=torch.float32) * np.log(theta) / (self.theta_num - 1))
        self.register_buffer('thetas', thetas)

    def update_embed(self, x, pc):
        """
        rotatary position embedding

        Args:
        - x: torch.Tensor[B, N, K, D], input feature
        - pc: torch.Tensor[B, N, 3], point cloud

        Returns:
        - x: torch.Tensor[B, N, K, D], output feature
        """
        orig_dim = x.shape[-1]
        thetas = self.thetas * pc.unsqueeze(-1) # b, n, 3, x
        cos, sin = thetas.cos(), thetas.sin()
        cos, sin = (rearrange(a, 'b n t x -> b n 1 (t x)') for a in (cos, sin))
        x = rearrange(x[..., :cos.shape[-1] * 2], 'b n k (d t) -> b n k d t', t=2)
        x1, x2 = x[..., 0], x[..., 1]
        new_x1 = x1 * cos - x2 * sin
        new_x2 = x1 * sin + x2 * cos
        x = rearrange(torch.stack((new_x1, new_x2), dim=-1), 'b n k d t -> b n k (d t)')
        return torch.cat((x, torch.zeros_like(x[..., :orig_dim-x.shape[-1]])), dim=-1)

    def forward(self, x, pc):
        """
        Args:
        - x: torch.Tensor[B, D, N], input feature
        - pc: torch.Tensor[B, N, 3], point cloud

        Returns:
        - x: torch.Tensor[B, D, N], output feature
        """
        x_q = self.q_conv(x) # b, c, n
        x_k = self.k_conv(x) # b, c, n
        x_v = self.v_conv(x)

        if self.use_flash_attn:
            x_q, x_k, x_v = (rearrange(x_x, 'b (k d) n -> b n k d', k=self.n_heads) for x_x in (x_q, x_k, x_v))
            x_q_2, x_k_2 = (self.update_embed(x_x, pc) for x_x in (x_q, x_k))
            x_q_2, x_k_2, x_v = (a.to(dtype=torch.float16) for a in (x_q_2, x_k_2, x_v))
            x_r = flash_attn_func(x_q_2, x_k_2, x_v)
            x_r = rearrange(x_r, 'b n k d -> b (k d) n', k=self.n_heads).to(dtype=torch.float32)
        else:
            raise NotImplementedError
            energy = torch.bmm(x_q, x_k) # b, n, n
            attention = self.softmax(energy)
            attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
            x_r = torch.bmm(x_v, attention) # b, c, n

        x = x + x_r
        x_r = self.trans_conv2(self.act(self.after_norm(self.trans_conv(x))))
        x = x + x_r

        return x

'''
