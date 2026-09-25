import numpy as np
import torch


def numpy_normalize(vec):
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-6)


def numpy_quaternion_to_matrix(quaternions: np.ndarray) -> np.ndarray:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = np.split(quaternions, 4, -1)

    two_s = 2.0 / (quaternions * quaternions).sum(-1, keepdims=True)

    o = np.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )

    return o.reshape(quaternions.shape[:-1] + (3, 3))


def proper_svd(rot: torch.Tensor):
    """
    compute proper svd of rotation matrix
    rot: (B, 3, 3)
    return rotation matrix (B, 3, 3) with det = 1
    """
    u, s, v = torch.svd(rot.double())
    with torch.no_grad():
        sign = torch.sign(torch.det(torch.einsum("bij,bkj->bik", u, v)))
        diag = torch.stack(
            [torch.ones_like(s[:, 0]), torch.ones_like(s[:, 1]), sign], dim=-1
        )
        diag = torch.diag_embed(diag)
    return torch.einsum("bij,bjk,blk->bil", u, diag, v).to(rot.dtype)
