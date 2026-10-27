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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
    def create_from_pcd(self, pcd: BasicPointCloud, cam_infos: int, spatial_lr_scale: float, target_budget: int = 500000):  # Build the initial Gaussian set and cap its size.
        """Initialize Gaussians from a point cloud with a configurable point budget.

        If the input exceeds the budget, a random subset is selected without
        replacement. If it is already within budget, all input points are kept;
        training densification may add points later.
        """
        self.spatial_lr_scale = spatial_lr_scale  # Store the scene scale used by the position learning rate.

        # Reject invalid budgets and empty point clouds before sampling.
        if target_budget <= 0:  # A zero or negative budget cannot initialize a useful scene.
            raise ValueError("target_budget must be a positive integer.")  # Stop early with a clear error.

        points = np.asarray(pcd.points)  # Convert point positions to a NumPy array for sampling.
        colors = np.asarray(pcd.colors)  # Convert matching point colours to a NumPy array.
        num_points = points.shape[0]  # Count the available input points.

        if num_points == 0:  # Sampling cannot work if there are no input points.
            raise ValueError("The input point cloud is empty.")  # Explain why initialization cannot continue.
        if colors.shape[0] != num_points:  # Every point must have one matching colour.
            raise ValueError("The point and colour arrays must contain the same number of entries.")  # Reject misaligned data.

        # Downsample without replacement when the input exceeds the budget.
        if num_points > target_budget:  # Use this branch when the point cloud is too large.
            print(f"Point cloud has {num_points} points; downsampling to budget {target_budget}.")  # Log the sampling decision.
            sample_indices = np.random.choice(num_points, size=target_budget, replace=False)  # Randomly choose distinct point indices.
            points = points[sample_indices]  # Keep the selected positions only.
            colors = colors[sample_indices]  # Keep the colours at those same indices.

        # If the point cloud is already within budget, keep its points unchanged.
        # Do not duplicate points just to fill the budget; the budget is a maximum, not a target.

        # Convert the adjusted positions and matching colours to CUDA tensors.
        fused_point_cloud = torch.tensor(points, dtype=torch.float32, device="cuda")  # Store positions as GPU float tensors.
        fused_color = RGB2SH(torch.tensor(colors, dtype=torch.float32, device="cuda"))  # Convert GPU colours to spherical-harmonic DC values.

        # Create one spherical-harmonic feature vector for each sampled point.
        features = torch.zeros(  # Allocate zero-initialized appearance coefficients.
            (fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2),  # Shape is points × RGB channels × SH coefficients.
            dtype=torch.float32,  # Use 32-bit floating-point coefficients.
            device="cuda",  # Keep the feature tensor on the GPU.
        )
        features[:, :3, 0] = fused_color  # Fill the first SH coefficient with the converted colour.
        features[:, 3:, 1:] = 0.0  # Leave higher-order coefficients initialized to zero (the slice is empty for RGB).

        print("Number of points at initialisation:", fused_point_cloud.shape[0])  # Report the initial Gaussian count.

        # Calculate scales from the same sampled positions that initialize the model.
        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 1e-7)  # Estimate nearest-neighbour squared distances and avoid zeros.
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)  # Convert distances into log-scales for all three axes.
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")  # Allocate one quaternion per Gaussian.
        rots[:, 0] = 1  # Initialize each quaternion to the identity rotation.

        opacities = self.inverse_opacity_activation(  # Convert desired opacity into the model's trainable logit space.
            0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float32, device="cuda")  # Initialize every Gaussian at opacity 0.1.
        )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))  # Make positions trainable.
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))  # Store trainable DC colour coefficients.
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))  # Store trainable higher-order appearance coefficients.
        self._scaling = nn.Parameter(scales.requires_grad_(True))  # Make Gaussian scales trainable.
        self._rotation = nn.Parameter(rots.requires_grad_(True))  # Make Gaussian rotations trainable.
        self._opacity = nn.Parameter(opacities.requires_grad_(True))  # Make Gaussian opacity logits trainable.
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")  # Initialize screen-space radius history for every Gaussian.
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}  # Map image names to exposure indices.
        self.pretrained_exposures = None  # Indicate that no pretrained exposure values were loaded.
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)  # Create one identity exposure transform per camera.
        self._exposure = nn.Parameter(exposure.requires_grad_(True))  # Make camera exposure transforms trainable.

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def densify_and_blur_split(self, radius_threshold=2.0, N=2):
        """Split Gaussians with large projected footprints instead of using gradient thresholds."""
        # Combine the historical maximum projected radius with the current view's radius.
        # This lets a Gaussian qualify even when it is not visible in the current training view.
        blur_score = torch.maximum(self.max_radii2D, self.tmp_radii)  # Approximate image-space blur using projected footprint.
        selected_pts_mask = blur_score >= radius_threshold  # Select the Gaussians whose screen-space footprint is too large.
        selected_count = int(selected_pts_mask.sum().item())  # Count selected parents for logging and the no-op check.

        # Avoid tensor construction and optimizer updates when no Gaussian reaches the threshold.
        if selected_count == 0:
            return

        # Sample child positions around each selected parent using its current 3D scale.
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)  # Repeat each parent's scale once per child.
        means = torch.zeros((stds.size(0), 3), device=self.get_xyz.device)  # Center each sampling distribution at zero.
        samples = torch.normal(mean=means, std=stds)  # Sample local offsets inside each parent's Gaussian footprint.
        rotations = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)  # Rotate local offsets into world coordinates.
        new_xyz = torch.bmm(rotations, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)  # Place children around their parent.
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))  # Shrink children to distribute the parent's footprint.
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)  # Inherit parent orientations.
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)  # Preserve low-order colour coefficients.
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)  # Preserve higher-order appearance coefficients.
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)  # Initialize children with their parent's opacity logits.
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)  # Keep temporary radius bookkeeping aligned with the new children.

        # Append children first so optimizer state and model tensors stay synchronized.
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        # Remove each selected parent; its children now represent the finer footprint.
        parent_prune_mask = torch.cat((selected_pts_mask, torch.zeros(N * selected_count, device=self.get_xyz.device, dtype=torch.bool)))  # Build a full mask for parents plus children.
        self.prune_points(parent_prune_mask)  # Remove parents and preserve all newly generated children.
        print(f"Blur Split: split {selected_count} parents into {N * selected_count} children.")  # Record this densification event.

    def densify_and_prune(
        self, max_grad, min_opacity, extent, max_screen_size, radii,
        gaussian_budget=None, stochastic_prune_power=2.0,
        densification_method="gradient", blur_split_radius_threshold=2.0
    ):
        """Densify with gradient rules or Blur Split, prune, and enforce the Gaussian budget."""
        # Divide accumulated gradients by observation counts to get mean gradients.
        grads = self.xyz_gradient_accum / self.denom  # Compute each Gaussian's mean view-space gradient.
        # Replace undefined gradients with zero so they cannot trigger gradient densification.
        grads[grads.isnan()] = 0.0  # Prevent NaNs from selecting points accidentally.

        # Store the current view's projected radii for the split routine and pruning bookkeeping.
        self.tmp_radii = radii  # The array must remain aligned with the current Gaussian ordering.

        # Choose exactly one densification strategy for a controlled experiment.
        if densification_method == "gradient":  # Preserve the existing 3DGS gradient-based method.
            self.densify_and_clone(grads, max_grad, extent)  # Clone small high-gradient Gaussians.
            self.densify_and_split(grads, max_grad, extent)  # Split large high-gradient Gaussians.
        elif densification_method == "blur_split":  # Use projected footprint rather than gradient magnitude.
            if blur_split_radius_threshold <= 0:  # Reject thresholds that would select every nonnegative radius.
                raise ValueError("blur_split_radius_threshold must be greater than zero.")
            self.densify_and_blur_split(  # Split Gaussians whose projected footprints appear too broad.
                radius_threshold=blur_split_radius_threshold  # Interpret the threshold in projected pixel-radius units.
            )
        else:  # Fail clearly rather than silently running the wrong experiment.
            raise ValueError("densification_method must be 'gradient' or 'blur_split'.")

        # Apply the original 3DGS opacity rule after either densification strategy.
        prune_mask = (self.get_opacity < min_opacity).squeeze()  # Mark Gaussians below the opacity threshold.
        # Apply size-based pruning only when a screen-size threshold is active.
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size  # Mark Gaussians that are too large in screen space.
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent  # Mark Gaussians that are too large in world space.
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)  # Combine pruning criteria.
        self.prune_points(prune_mask)  # Remove points rejected by the baseline pruning rules.

        # Enforce the experiment's hard Gaussian budget after densification and ordinary pruning.
        if gaussian_budget is not None:
            if gaussian_budget <= 0:  # A nonpositive budget cannot represent a valid model.
                raise ValueError("gaussian_budget must be a positive integer.")
            if stochastic_prune_power <= 0:  # The configured stochastic pruning power must remain positive.
                raise ValueError("stochastic_prune_power must be greater than zero.")
            current_count = self.get_xyz.shape[0]  # Count the Gaussians remaining after normal pruning.
            excess_count = current_count - gaussian_budget  # Calculate how many must be removed to satisfy the cap.
            if excess_count > 0:  # Do extra pruning only if the model exceeds its maximum size.
                opacity_scores = self.get_opacity.squeeze(-1)  # Read activated opacity values in [0, 1].
                removal_weights = (1.0 - opacity_scores).clamp_min(1e-8)  # Prefer transparent Gaussians without zero weights.
                removal_weights = removal_weights.pow(stochastic_prune_power)  # Strengthen the preference for low-opacity points.
                removal_weights = removal_weights / removal_weights.sum()  # Convert weights to sampling probabilities.
                sampled_indices = torch.multinomial(removal_weights, num_samples=excess_count, replacement=False)  # Sample distinct removals.
                budget_prune_mask = torch.zeros(current_count, dtype=torch.bool, device=self.get_xyz.device)  # Initially keep every Gaussian.
                budget_prune_mask[sampled_indices] = True  # Mark the sampled Gaussians for removal.
                self.prune_points(budget_prune_mask)  # Update model and optimizer tensors together.
                print(f"Gaussian budget: pruned {excess_count}; remaining {self.get_xyz.shape[0]} / {gaussian_budget} (method={densification_method}).")  # Log hard-cap enforcement.

        # Clear temporary radius data after all pruning operations have completed.
        self.tmp_radii = None  # Prevent stale radii from being reused during the next densification cycle.
        torch.cuda.empty_cache()  # Release unused cached CUDA memory.

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
