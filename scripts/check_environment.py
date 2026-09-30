"""Check installed dependencies, with optional CUDA and visualization checks."""

import argparse
import importlib
from importlib.metadata import version
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true", help="Run CUDA extension and backbone checks")
    parser.add_argument("--visualize", action="store_true", help="Import visualization and MANO modules")
    args = parser.parse_args()

    import torch
    from pytorch3d.transforms import matrix_to_quaternion

    for name in ("torch", "numpy", "scipy", "nflows", "diffusers", "pytorch3d"):
        importlib.import_module(name)
        print(f"{name}: {version(name)}")
    torch.testing.assert_close(matrix_to_quaternion(torch.eye(3)), torch.tensor([1., 0., 0., 0.]))

    if args.cuda:
        import MinkowskiEngine as ME
        from pytorch3d.ops import knn_points
        from dexlearn.network.backbones.mink_unet import WrappedMinkUNet

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; check the NVIDIA driver and CUDA_VISIBLE_DEVICES")
        torch.manual_seed(0)
        points = torch.randn(1, 16, 3, device="cuda", requires_grad=True)
        distances = knn_points(points, points.detach(), K=2).dists
        distances.sum().backward()
        assert torch.isfinite(distances).all() and torch.isfinite(points.grad).all()
        print("PyTorch3D CUDA KNN forward/backward passed")

        coords = ME.utils.batched_coordinates([torch.randint(0, 32, (128, 3))])
        features = torch.randn(len(coords), 3, device="cuda", requires_grad=True)
        sparse = ME.SparseTensor(features, coordinates=coords, device="cuda")
        layer = ME.MinkowskiConvolution(3, 8, kernel_size=3, dimension=3).cuda()
        result = layer(sparse).F
        result.square().mean().backward()
        assert torch.isfinite(result).all() and torch.isfinite(features.grad).all()
        backbone = WrappedMinkUNet(SimpleNamespace(out_feat_dim=16)).cuda().eval()
        with torch.no_grad():
            output = backbone.model(sparse).F
        torch.cuda.synchronize()
        assert output.shape == (len(sparse), 16) and torch.isfinite(output).all()
        print("MinkowskiEngine CUDA forward/backward and HUGS backbone forward passed")

    if args.visualize:
        for name in ("manopth.manolayer", "pytorch_kinematics", "mr_utils.robot.pk_helper",
                     "mr_utils.robot.pk_visualizer", "viser", "cv2"):
            importlib.import_module(name)
        print("Visualization imports passed (no window or MANO model loaded)")

    print("Environment checks passed")


if __name__ == "__main__":
    main()
