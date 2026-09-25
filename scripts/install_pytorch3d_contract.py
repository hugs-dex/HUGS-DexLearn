"""Install the verified official 0.7.8 archive into a Python 3.10 contract venv."""

import argparse
import hashlib
from pathlib import Path
import sys
import sysconfig
import tarfile

SHA256 = "9e8965ec8c9ac66cb8aa89ab7486f163619c67336d1dc74ae72018064cce151a"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 10) or sys.prefix == sys.base_prefix:
        raise RuntimeError("Use a dedicated Python 3.10 venv; system installation is not supported")
    import torch
    if torch.__version__.split("+")[0] != "2.2.2":
        raise RuntimeError("This contract archive requires PyTorch 2.2.2")
    if hashlib.sha256(args.archive.read_bytes()).hexdigest() != SHA256:
        raise ValueError("PyTorch3D archive SHA-256 mismatch")
    site = Path(sysconfig.get_paths()["purelib"])
    prefix = "lib/python3.10/site-packages/"
    with tarfile.open(args.archive) as archive:
        members = [m for m in archive if m.isfile() and m.name.startswith(prefix)]
        for member in members:
            relative = Path(member.name[len(prefix):])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe archive member")
            target = site / relative
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite an existing package file: {target}")
        for member in members:
            target = site / member.name[len(prefix):]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(member).read())
    from pytorch3d.transforms import matrix_to_quaternion
    assert torch.allclose(matrix_to_quaternion(torch.eye(3)), torch.tensor([1., 0., 0., 0.]))
    print("Installed PyTorch3D 0.7.8; CPU transform smoke passed. CUDA ops are not verified.")


if __name__ == "__main__":
    main()
