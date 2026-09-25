"""mobius flow"""

import numpy as np
import torch
from torch import nn

from ..mlp import BasicMLP


def _h(z, w, D=3):
    w_norm = torch.norm(w, dim=-1, keepdim=True)  # n x k x 1

    h_z = (1 - w_norm**2) / (
        torch.norm((z.reshape(-1, 1, D) - w), dim=-1, keepdim=True) ** 2
    ) * (z.reshape(-1, 1, D) - w) - w

    return h_z


class MobiusFlow(nn.Module):
    def __init__(self, D, K, real_dim, feature_dim=None):
        super(MobiusFlow, self).__init__()

        self.D = D # what is this?
        self.K = K
        self.real_dim = real_dim
        self.cond_real_dim = real_dim // 2 # TODO: ?
        self.tran_real_dim = real_dim - self.cond_real_dim

        self.feature_dim = feature_dim

        input_dim = D + self.feature_dim + self.cond_real_dim

        self.conditioner = BasicMLP(
            input_dim,
            4 * K + self.tran_real_dim * 2,
            mask=slice(self.K * 4, self.K * 4 + self.tran_real_dim),
        )
        self.register_buffer("_I", torch.eye(D), persistent=False)

    def mask(self):
        self.conditioner.apply_mask()

    def forward(self, rotation, vector, permute=None, feature=None):
        assert permute != None, "The permuting function is needed in this module"
        assert feature != None, "The input feature is needed in this module"
        x = rotation[..., permute[0]]
        y = rotation[..., permute[1]]

        condition_input = torch.cat(
            (y, feature, vector[..., : self.cond_real_dim]), dim=-1
        )

        conditions = self.conditioner(condition_input)
        weights, w, real_s, real_b = torch.split(
            conditions,
            [self.K, 3 * self.K, self.tran_real_dim, self.tran_real_dim],
            dim=1,
        )
        w = w.reshape(-1, self.K, self.D)
        proj = self._I[None, ...] - torch.einsum("ni,nj->nij", y, y)
        w = torch.einsum("nij,nkj->nki", proj, w)
        r = -x
        r = r / r.norm(dim=-1, keepdim=True)
        v = torch.cross(y, r, dim=-1)
        v = v / v.norm(dim=-1, keepdim=True)

        weights = torch.nn.functional.softplus(weights)
        sum_weight = weights.sum(dim=-1, keepdim=True)
        weights = weights / sum_weight
        w = 0.7 / (1 + torch.norm(w, dim=-1, keepdim=True)) * w

        tx, ldj = self._forward(x, r, v, weights, w)
        if (permute[1] - permute[0]) == 1 or (permute[1] - permute[0]) == -2:
            tz = torch.cross(tx, y)
        else:
            tz = torch.cross(y, tx)
        tz = tz / tz.norm(dim=-1, keepdim=True)
        trotation = torch.empty(rotation.size()).to(rotation.device)
        trotation[..., permute[0]] = tx
        trotation[..., permute[1]] = y
        trotation[..., permute[2]] = tz

        tvector = vector.clone()
        real_s = real_s.exp() * 0 + 1
        tvector[..., self.cond_real_dim :] = (
            vector[..., self.cond_real_dim :] * real_s + real_b
        )

        ldj = ldj + real_s.log().sum(dim=-1)

        return trotation, tvector, ldj

    def _h(self, z, w):
        return _h(z, w, self.D)

    def _forward(self, x, r, v, weights, w):
        z = x

        h_z = self._h(z, w)
        radians = torch.atan2(
            torch.einsum("nki,ni->nk", h_z, v), torch.einsum("nki,ni->nk", h_z, r)
        )
        tx = radians
        tx = torch.where(tx >= 0, tx, tx + np.pi * 2)
        tx = torch.sum(weights * tx, dim=1, keepdim=True)

        tx = r * torch.cos(tx) + v * torch.sin(tx)

        z_w = z[:, None, :] - w
        z_w_norm = torch.norm(z_w, dim=-1)
        z_w_unit = z_w / z_w_norm[..., None]

        theta = torch.atan2(
            torch.einsum("ni,ni->n", x, v), torch.einsum("ni,ni->n", x, r)
        ).reshape(-1, 1)
        dz_dtheta = -torch.sin(theta) * r + torch.cos(theta) * v

        # n x k x 2 x 2
        dh_dz = (
            (1 - torch.norm(w, dim=-1) ** 2)[..., None, None]
            * (
                self._I[None, None, ...]
                - 2 * torch.einsum("nki,nkj->nkij", z_w_unit, z_w_unit)
            )
            / (z_w_norm[..., None, None] ** 2)
        )

        dh_dtheta = torch.einsum("nkpq,nq->nkp", dh_dz, dz_dtheta)
        dtx = torch.sum(torch.norm(dh_dtheta, dim=-1) * weights, dim=1)
        return tx, torch.log(dtx)

    def inverse(self, trotation, tvector, permute=None, feature=None):
        assert permute != None, "The permuting function is needed in this module"
        assert feature != None, "feature input is needed in this module"

        tx = trotation[..., permute[0]]
        ty = trotation[..., permute[1]]

        condition_input = torch.cat(
            (ty, feature, tvector[..., : self.cond_real_dim]), dim=-1
        )

        conditions = self.conditioner(condition_input)
        weights, w, real_s, real_b = torch.split(
            conditions,
            [self.K, 3 * self.K, self.tran_real_dim, self.tran_real_dim],
            dim=1,
        )
        w = w.reshape(-1, self.K, self.D)
        proj = self._I[None, ...] - torch.einsum("ni,nj->nij", ty, ty)
        w = torch.einsum("nij,nkj->nki", proj, w)

        r = -tx
        r = r / r.norm(dim=-1, keepdim=True)
        v = torch.cross(ty, r)
        v = v / v.norm(dim=-1, keepdim=True)
        weights = torch.nn.functional.softplus(weights)
        sum_weight = weights.sum(dim=-1, keepdim=True)
        weights = weights / sum_weight
        w = 0.7 / (1 + torch.norm(w, dim=-1, keepdim=True)) * w

        ttheta = torch.atan2(
            torch.einsum("ni,ni->n", tx, v), torch.einsum("ni,ni->n", tx, r)
        ).reshape(-1, 1)
        ttheta = torch.where(ttheta >= 0, ttheta, ttheta + np.pi * 2)

        ttheta = torch.where(
            abs(ttheta - np.pi * 2) < 1e-4,
            torch.zeros(ttheta.size(), dtype=ttheta.dtype, device=ttheta.device),
            ttheta,
        )

        theta = self._bin_find_root(ttheta, r, v, weights, w)
        x = r * torch.cos(theta) + v * torch.sin(theta)
        _, ldj = self._forward(x, r, v, weights, w)

        if (permute[1] - permute[0]) == 1 or (permute[1] - permute[0]) == -2:
            z = torch.cross(x, ty)
        else:
            z = torch.cross(ty, x)
        z = z / z.norm(dim=-1, keepdim=True)
        rotation = torch.empty(trotation.size()).to(trotation.device)
        rotation[..., permute[0]] = x
        rotation[..., permute[1]] = ty
        rotation[..., permute[2]] = z

        vector = tvector.clone()
        real_s = real_s.exp() * 0 + 1
        vector[..., self.cond_real_dim :] = (
            tvector[..., self.cond_real_dim :] - real_b
        ) / real_s

        ldj = ldj + real_s.log().sum(dim=-1)

        return rotation, vector, -ldj

    def _bin_find_root(self, y, r, v, weights, w):
        return BinFind.apply(y, r, v, weights, w)


class BinFind(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, r, v, weights, w):
        ctx.r = r.clone().detach()
        ctx.v = v.clone().detach()
        ctx.weights = weights.clone().detach()
        ctx.w = w.clone().detach()
        a = torch.ones(y.size(), device=y.device, dtype=y.dtype) * np.pi / 2
        b = torch.ones(y.size(), device=y.device, dtype=y.dtype) * 3 / 2 * np.pi
        time = 1
        while abs(torch.max(b - a)) >= 1e-4:
            x0 = (a + b) / 2
            fx0 = BinFind._forward_theta(x0, r, v, weights, w) - y

            if time > 100:
                print("fail")
                break

            bigger = fx0 < 0
            lesser = fx0 >= 0
            a = a + (b - a) / 2 * bigger
            b = b - (b - a) / 2 * lesser

            time += 1
        ctx.x = x0.clone().detach()
        ctx.y = y.clone().detach()
        return x0

    @staticmethod
    def _h(z, w, D=3):
        return _h(z, w, D)

    @staticmethod
    def _forward_theta(x, r, v, weights, w):
        """input: theta, return theta' and partial theta'/ partial theta
        used to compute inverse"""
        z = r * torch.cos(x) + v * torch.sin(x)

        h_z = BinFind._h(z, w)
        radians = torch.atan2(
            torch.einsum("nki,ni->nk", h_z, v), torch.einsum("nki,ni->nk", h_z, r)
        )
        tx = radians
        tx = torch.where(tx >= 0, tx, tx + np.pi * 2)
        tx = torch.sum(weights * tx, dim=1, keepdim=True)

        return tx

    @staticmethod
    def backward(ctx, x_grad):
        x = ctx.x
        y = ctx.y
        r = ctx.r
        v = ctx.v
        weights = ctx.weights
        w = ctx.w
        with torch.set_grad_enabled(True):
            x.requires_grad_(True)
            r.requires_grad_(True)
            v.requires_grad_(True)
            weights.requires_grad_(True)
            w.requires_grad_(True)
            x_grad_2, r_grad, v_grad, weights_grad, w_grad = torch.autograd.grad(
                BinFind._forward_theta(x, r, v, weights, w),
                (x, r, v, weights, w),
                torch.ones_like(x_grad),
            )
            y_grad = (
                torch.where(x_grad_2 != 0, 1 / x_grad_2, torch.zeros_like(x_grad_2))
                * x_grad
            )
            r_grad = (
                torch.where(x_grad_2 != 0, -r_grad / x_grad_2, torch.zeros_like(r_grad))
                * x_grad
            )
            v_grad = (
                torch.where(x_grad_2 != 0, -v_grad / x_grad_2, torch.zeros_like(v_grad))
                * x_grad
            )
            w_grad = torch.where(
                x_grad_2.unsqueeze(-1) != 0,
                -w_grad / x_grad_2.unsqueeze(-1),
                torch.zeros_like(w_grad),
            ) * x_grad.unsqueeze(-1)
            weights_grad = (
                torch.where(
                    x_grad_2 != 0,
                    -weights_grad / x_grad_2,
                    torch.zeros_like(weights_grad),
                )
                * x_grad
            )
        return y_grad, r_grad, v_grad, weights_grad, w_grad
