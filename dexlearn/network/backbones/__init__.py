import torch
import torch.nn as nn

def WrappedMinkUNet(*args, **kwargs):
    """Construct the unchanged sparse backbone only when selected."""
    from dexlearn.utils.dependencies import require_module
    require_module("MinkowskiEngine", "WrappedMinkUNet")
    from .mink_unet import WrappedMinkUNet as implementation
    return implementation(*args, **kwargs)
