import torch

from torch import Tensor

from diff_gaussian_rasterization_w_pose_depth import _C

import matplotlib.pyplot as plt
import warnings

warnings.simplefilter("ignore")

args = torch.load("snapshot_fw.dump")
args = [x.cuda() if isinstance(x, Tensor) else x for x in args]

(
    num_rendered,
    color,
    depth,
    alpha,
    radii,
    geomBuffer,
    binningBuffer,
    imgBuffer,
    mean3d_camera,
) = _C.rasterize_gaussians(*args)

color = color.clamp(0, 1)
loss = color.mean()
loss.backward()
