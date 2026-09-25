import torch
from torch import nn

from einops import rearrange, repeat
from pytorch3d import transforms as pttf

from .mobiusflow import MobiusFlow
from .affineflow import Condition16TransR

_permute_prop = torch.Tensor(
    [[0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 1, 2], [1, 2, 0], [2, 0, 1]]
).type(torch.long)


class Flow(nn.Module):
    def __init__(
        self,
        config: dict,
        feature_dim: int,
        real_dim: int,
    ):
        super(Flow, self).__init__()
        self.layer_num = config["layer"]
        self.condition = 1
        self._permute = _permute_prop  # permute strategy

        self.feature_dim = feature_dim
        self.real_dim = real_dim

        layers = []

        for i in range(self.layer_num):
            layers.append(Condition16TransR(real_dim, feature_dim))
            layers.append(MobiusFlow(3, config["k"], real_dim, feature_dim))

        print("total layers of flow: ", len(layers))
        self.layers = nn.ModuleList(layers)
        self.gaussian = torch.distributions.Normal(0, 1)

    def mask(self):
        """
        doint so to ensure a good initialization
        otherwise it is easy to get nan
        """
        for layer in self.layers:
            layer.mask()

    def forward(
        self, rotation: torch.Tensor, vector: torch.Tensor, feature: torch.Tensor
    ):
        """
        sample -> noise
        Usually no need to call this directly

        rotation: (B, 3, 3)
        vector: (B, N)
        feature: (B, D)

        return
            noise rotation: (B, 3, 3)
            noise vector: (B, N)
            ldjs (log det of jacobians): (B, N)
        """
        permute = self._permute.to(rotation.device)

        ldjs = 0
        exchange_count = 0

        for i in range(len(self.layers)):
            rotation, vector, ldj = self.layers[i](
                rotation, vector, permute[exchange_count % 6], feature
            )
            ldjs += ldj
            if isinstance(self.layers[i], MobiusFlow):
                exchange_count += 1
        mask = vector.isnan().any(dim=-1)
        vector = torch.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
        ldjs += self.gaussian.log_prob(vector).sum(dim=-1)
        ldjs[mask] = 0
        return rotation, vector, ldjs

    def inverse(
        self, rotation: torch.Tensor, vector: torch.Tensor, feature: torch.Tensor
    ):
        """
        noise -> sample
        Usually no need to call this directly

        noise rotation: (B, 3, 3)
        noise vector: (B, N)
        feature: (B, D)

        return
            rotation: (B, 3, 3)
            vector: (B, N)
            ldjs (log det of jacobians): (B, N)
        """
        permute = self._permute.to(rotation.device)

        ldjs = -self.gaussian.log_prob(vector).sum(dim=-1)
        exchange_count = self.layer_num

        if not self.condition:
            feature = None

        for i in range(len(self.layers))[::-1]:
            if isinstance(self.layers[i], MobiusFlow):
                exchange_count -= 1
            rotation, vector, ldj = self.layers[i].inverse(
                rotation, vector, permute[exchange_count % 6], feature
            )
            ldjs += ldj

        return rotation, vector, ldjs

    def log_prob(
        self, rotation: torch.Tensor, vector: torch.Tensor, feature: torch.Tensor
    ):
        """
        calculate the log probability of the given sample

        rotation: (B, 3, 3)
        vector: (B, N)
        feature: (B, D)

        return
            ldjs (log det of jacobians): (B, N)
        """
        return self.forward(rotation, vector, feature)[2]

    def sample_and_log_prob(
        self, sample_num: int, feature: torch.Tensor, allow_fail: bool = False
    ):
        """
        sample and calculate the log probability of the given sample

        sample_num: int
        feature: (B, D)

        return
            rotation: (B, N, 3, 3)
            vector: (B, N, D)
            log_prob: (B, N)

        """
        try:
            feature = repeat(feature, "b d -> (b n) d", n=sample_num)
            total_sample_num = feature.shape[0]
            random_rotations = pttf.random_rotations(
                total_sample_num, device=feature.device
            )
            random_vectors = self.gaussian.sample((total_sample_num, self.real_dim)).to(
                feature.device
            )
            rotations, vectors, ldjs = self.inverse(
                random_rotations, random_vectors, feature
            )
            log_prob = -ldjs
            return (
                rearrange(rotations, "(b n) x y -> b n x y", n=sample_num),
                rearrange(vectors, "(b n) d -> b n d", n=sample_num),
                rearrange(log_prob, "(b n) -> b n", n=sample_num),
            )
        except RuntimeError as e:
            print("Error: ", e)
            if allow_fail:
                return self.sample_and_log_prob(sample_num, torch.randn_like(feature))
            else:
                raise e
