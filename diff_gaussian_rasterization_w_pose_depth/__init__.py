#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from . import _C  # noqa


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [
        item.cpu().clone() if isinstance(item, torch.Tensor) else item
        for item in input_tuple
    ]
    return tuple(copied_tensors)




def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
    camera_center: Tensor,
    camera_pose: Tensor,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        raster_settings.intrinsic_matrix,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        camera_center,
        camera_pose,
    )


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        K,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings: GaussianRasterizationSettings,
        camera_center,
        camera_pose,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            camera_center,
            raster_settings.prefiltered,
            raster_settings.debug,
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            # Copy them before they can be corrupted
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                (
                    num_rendered,
                    color,
                    depth,
                    alpha,
                    radii,
                    geomBuffer,
                    binningBuffer,
                    imgBuffer,
                    mean3d_cam,
                ) = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print(
                    "\nAn error occured in forward. Please forward snapshot_fw.dump for debugging."
                )
                raise ex
        else:
            (
                num_rendered,
                color,
                depth,
                alpha,
                radii,
                geomBuffer,
                binningBuffer,
                imgBuffer,
                mean3d_cam,
            ) = _C.rasterize_gaussians(*args)

        # save for the K gradient.

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(
            colors_precomp,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            geomBuffer,
            binningBuffer,
            imgBuffer,
            camera_center,
            camera_pose,
            alpha,
            mean3d_cam,
        )
        return color, depth, alpha, radii, mean3d_cam

    @staticmethod
    def backward(ctx, grad_out_color, grad_out_depth, grad_out_alpha, _, __):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        (
            colors_precomp,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            geomBuffer,
            binningBuffer,
            imgBuffer,
            camera_center,
            camera_pose,
            alpha,
            mean3d_cam,
        ) = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (
            raster_settings.bg,
            means3D,
            radii,
            colors_precomp,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.proj_k,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            grad_out_color,
            grad_out_depth,
            grad_out_alpha,
            sh,
            raster_settings.sh_degree,
            camera_center,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            alpha,
            raster_settings.debug,
        )

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            # Copy them before they can be corrupted
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                (
                    grad_means2D,
                    grad_colors_precomp,
                    grad_opacities,
                    grad_means3D,
                    grad_cov3Ds_precomp,
                    grad_sh,
                    grad_scales,
                    grad_rotations,
                    grad_camera_pose,
                ) = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print(
                    "\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n"
                )
                raise ex
        else:
            (
                grad_means2D,
                grad_colors_precomp,
                grad_opacities,
                grad_means3D,
                grad_cov3Ds_precomp,
                grad_sh,
                grad_scales,
                grad_rotations,
                grad_camera_pose,
            ) = _C.rasterize_gaussians_backward(*args)
        # compute the gradient with respect to K.
        with torch.no_grad():
            grad_uv_k4: Float[Tensor, "n 2 4"] = torch.zeros(
                mean3d_cam.shape[0], 2, 4, device="cuda", dtype=torch.float32
            )
            grad_uv_k4[:, 0, 0] = mean3d_cam[:, 0] / (mean3d_cam[:, 2] + 1e-4)  # du/dfx
            grad_uv_k4[:, 1, 1] = mean3d_cam[:, 1] / (mean3d_cam[:, 2] + 1e-4)  # dv/dfy
            grad_uv_k4[:, 0, 2] = 1  # du/dcx
            grad_uv_k4[:, 1, 3] = 1  # dv/dcy

            grad_k = torch.einsum("nj,njk->nk", grad_means2D[:, :2], grad_uv_k4).sum(
                dim=0
            )
            grad_K = torch.tensor([[grad_k[0],  grad_k[2], 0], [grad_k[1],  grad_k[3], 0], [0, 0, 1]], device="cuda", dtype=torch.float32)


        grads = (
            grad_means3D,
            grad_means2D,
            grad_K,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
            None,
            grad_camera_pose,
        )

        return grads


@dataclass(kw_only=True, slots=True)
class GaussianRasterizationSettings:
    image_height: int
    """image height"""
    image_width: int
    """image width"""
    tanfovx: float
    """tan of horizontal field of view"""
    tanfovy: float
    """tan of vertical field of view"""
    bg: Float[Tensor, "3"]
    """background color"""
    scale_modifier: float
    """scale modifier"""
    viewmatrix: Float[Tensor, "4 4"]
    """transpose of world2cam matrix"""
    projmatrix: Float[Tensor, "4 4"]
    """full projection matrix (proj@world2cam)^{T}"""
    proj_k: Float[Tensor, "4 4"]
    """ opengl projection matrix from camera space to NDC space"""
    intrinsic_matrix: Float[Tensor, "3 3"]
    """intrinsic matrix"""
    sh_degree: int
    """degree of spherical harmonics"""
    campos: Float[torch.Tensor, "3"]
    """camera position"""
    prefiltered: bool
    """not sure of the definition"""
    debug: bool


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings: GaussianRasterizationSettings):
        super().__init__()
        self.raster_settings: GaussianRasterizationSettings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions, raster_settings.viewmatrix, raster_settings.projmatrix
            )

        return visible

    def forward(
        self,
        means3D,
        means2D,
        opacities,
        shs=None,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
        camera_center=None,
        camera_pose=None,
    ):

        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (
            shs is not None and colors_precomp is not None
        ):
            raise Exception(
                "Please provide excatly one of either SHs or precomputed colors!"
            )

        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
            (scales is not None or rotations is not None) and cov3D_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!"
            )

        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])
        if camera_center is None:
            camera_center = torch.Tensor([])
        if camera_pose is None:
            camera_pose = torch.Tensor([])
        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
            camera_center,
            camera_pose,
        )
