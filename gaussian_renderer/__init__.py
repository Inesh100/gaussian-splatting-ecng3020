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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians and retain the rasterizer's complete output tuple.  # Keep this flexible so an instrumented extension can append contribution counts.
    if separate_sh:  # Use separate spherical-harmonic tensors when the sparse optimizer path requests them.
        rasterizer_outputs = rasterizer(  # Capture every value returned by the rasterizer instead of assuming a fixed tuple length.
            means3D = means3D,  # Supply the Gaussian centers in world coordinates.
            means2D = means2D,  # Supply the screen-space gradient placeholder used by 3DGS.
            dc = dc,  # Supply the low-order spherical-harmonic coefficients.
            shs = shs,  # Supply the remaining spherical-harmonic coefficients.
            colors_precomp = colors_precomp,  # Supply precomputed colors when Python color conversion is enabled.
            opacities = opacity,  # Supply activated Gaussian opacities.
            scales = scales,  # Supply Gaussian scales when covariance is not precomputed.
            rotations = rotations,  # Supply Gaussian rotations when covariance is not precomputed.
            cov3D_precomp = cov3D_precomp)  # Supply precomputed covariance when Python covariance conversion is enabled.
    else:  # Use the standard rasterizer call when separate spherical harmonics are not requested.
        rasterizer_outputs = rasterizer(  # Capture the outputs from the standard rasterization path.
            means3D = means3D,  # Supply the Gaussian centers in world coordinates.
            means2D = means2D,  # Supply the screen-space gradient placeholder used by 3DGS.
            shs = shs,  # Supply spherical-harmonic appearance coefficients.
            colors_precomp = colors_precomp,  # Supply precomputed colors when available.
            opacities = opacity,  # Supply activated Gaussian opacities.
            scales = scales,  # Supply Gaussian scales when covariance is not precomputed.
            rotations = rotations,  # Supply Gaussian rotations when covariance is not precomputed.
            cov3D_precomp = cov3D_precomp)  # Supply precomputed covariance when Python covariance conversion is enabled.
    if len(rasterizer_outputs) == 3:  # Preserve compatibility with the current rasterizer that returns image, radii, and depth.
        rendered_image, radii, depth_image = rasterizer_outputs  # Unpack the three standard rasterizer outputs.
        max_contribution_area = None  # Mark contribution counts as unavailable unless the rasterizer explicitly returns them.
    elif len(rasterizer_outputs) == 4:  # Accept a custom rasterizer extension that also returns maximum-contribution pixel counts.
        rendered_image, radii, depth_image, max_contribution_area = rasterizer_outputs  # Unpack image, radii, depth, and per-Gaussian contribution areas.
    else:  # Reject unexpected rasterizer APIs before training silently uses the wrong values.
        raise RuntimeError(f"Expected rasterizer to return 3 or 4 values, but received {len(rasterizer_outputs)}.")  # Report the extension mismatch clearly.
        
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image,  # Preserve the rendered depth map for depth-related training losses.
        "max_contribution_area": max_contribution_area  # Expose optional per-Gaussian maximum-contribution pixel counts from an instrumented rasterizer.
        }
    
    return out
