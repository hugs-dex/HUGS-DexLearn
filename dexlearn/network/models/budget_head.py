from collections.abc import Sequence

import torch

from ..backbones import *


class GeometryBudgetHead(torch.nn.Module):
    """Small regression head for geometry-only budget multipliers."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (64, 64), dropout: float = 0.0):
        """Initialize the budget head.

        Args:
            input_dim: Number of normalized geometry features.
            hidden_dims: Hidden layer widths for the MLP.
            dropout: Dropout probability inserted after hidden activations.

        Returns:
            None.
        """
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        layers = []
        prev_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            hidden_dim = int(hidden_dim)
            if hidden_dim <= 0:
                raise ValueError(f"hidden dimensions must be positive, got {hidden_dim}")
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            if dropout > 0.0:
                layers.append(torch.nn.Dropout(float(dropout)))
            prev_dim = hidden_dim
        layers.append(torch.nn.Linear(prev_dim, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, geometry_feature: torch.Tensor) -> torch.Tensor:
        """Predict log budget multiplier from normalized geometry features.

        Args:
            geometry_feature: Tensor with shape ``(B, F)``.

        Returns:
            Tensor with shape ``(B,)`` containing predicted log multipliers.
        """
        if geometry_feature.ndim != 2:
            raise ValueError(f"Expected geometry_feature with shape (B, F), got {tuple(geometry_feature.shape)}")
        return self.net(geometry_feature).squeeze(-1)


class PointCloudBudgetHead(torch.nn.Module):
    """Point-cloud encoder plus regression head for scene-budget multipliers."""

    def __init__(self, backbone_cfg, hidden_dims: Sequence[int] = (128, 64), dropout: float = 0.0):
        """Initialize a point-cloud budget model.

        Args:
            backbone_cfg: Config for the point-cloud backbone, using the same
                schema as the main grasp model's ``algo.model.backbone``.
            hidden_dims: Hidden layer widths for the regression head.
            dropout: Dropout probability inserted after hidden activations.

        Returns:
            None.
        """
        super().__init__()
        self.backbone = eval(backbone_cfg.name)(backbone_cfg)
        self.regressor = GeometryBudgetHead(
            input_dim=int(backbone_cfg.out_feat_dim),
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

    def forward(self, data: dict) -> torch.Tensor:
        """Predict log budget multipliers from point clouds.

        Args:
            data: Batch dictionary accepted by the configured point-cloud
                backbone. It must include ``point_clouds``, sparse coordinates,
                sparse features, and ``quantize2original`` when using
                ``WrappedMinkUNet``.

        Returns:
            Tensor with shape ``(B,)`` containing predicted log multipliers.
        """
        global_feature, _ = self.backbone(data)
        return self.regressor(global_feature)
