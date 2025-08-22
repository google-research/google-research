# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-
"""Main inference class.

This code defines the implementation for generating long sequence of views
using an autoregressive render-refine-repeat technique.
"""
import glob
import os
import time

import imageio
import kornia
from models import fly_camera
from models.networks import co_mod_gan
import numpy as np
from splatting import splatting_function
import torch

MIN_DISP = 1e-2  # minimum disparity value
NORMALIZE_SOBEL = False  # normalize sobel gradient


class RenderModel(torch.nn.Module):
  """Render model."""

  def __init__(self, opt, is_train):
    super().__init__()
    self.device = torch.device('cuda:{}'.format(opt.local_rank))

    self.e_losses = {}
    self.opt = opt
    self.start_epoch = 0

    self.generator = self.initialize_networks(opt)

    self.generator = self.generator.to(self.device)

    self.start_epoch = 0

    self.start_global_step = self.pretrain_global_step = 0
    self.start_epoch = self.pretrain_epoch = 0

    if (not is_train or opt.continue_train):
      log_dir = './checkpoints'

      ckpt_path = sorted(glob.glob(os.path.join(log_dir, '*latest.tar')))[-1]
      print('Reloading from', ckpt_path)

      ckpt = torch.load(ckpt_path, map_location='cpu')
      self.start_global_step = ckpt['global_step']
      self.start_epoch = ckpt['epoch']

      print('start_global_step ', self.start_global_step)

      self.generator.load_state_dict(ckpt['netG_ema'])

      del ckpt

    self.generator.train().requires_grad_(False)
    torch.cuda.empty_cache()

    self.alpha_threshold = 0.3
    self.mask_threshold = 0.9

  def initialize_networks(self, opt):
    """Intialize Exponetial Moving Average (EMA) generator."""

    generator = co_mod_gan.Generator()

    return generator

  def render_forward_splat(self,
                           src_imgs, src_depths,
                           r_cam, t_cam, k_src,
                           k_dst):
    """3D render the image to the next viewpoint.

    Args:
      src_imgs: source images
      src_depths: source depth maps
      r_cam: relative camera rotation
      t_cam: relative camera translation
      k_src: source intrinsic matrix
      k_dst: destination intrinsic matrix

    Returns:
      warp_feature: the rendered RGB feature map
      warp_disp: the rendered disparity
      warp_mask: the rendered mask
    """
    batch_size = src_imgs.shape[0]

    rot = r_cam
    t = t_cam
    k_src_inv = k_src.inverse()

    x = np.arange(src_imgs[0].shape[1])
    y = np.arange(src_imgs[0].shape[0])
    coord = np.stack(np.meshgrid(x, y), -1)
    coord = np.concatenate((coord, np.ones_like(coord)[:, :, [0]]), -1)
    coord = coord.astype(np.float32)
    coord = torch.as_tensor(coord, dtype=k_src.dtype, device=k_src.device)
    coord = coord[None, Ellipsis, None].repeat(batch_size, 1, 1, 1, 1)

    depth = src_depths[:, :, :, None, None]

    # from reference to target viewpoint
    pts_3d_ref = depth * k_src_inv[:, None, None, Ellipsis] @ coord
    pts_3d_tgt = rot[:, None, None, Ellipsis] @ pts_3d_ref + t[:, None, None, :,
                                                          None]
    points = k_dst[:, None, None, Ellipsis] @ pts_3d_tgt
    points = points.squeeze(-1)

    new_z = points[:, :, :, [2]].clone().permute(0, 3, 1, 2)  # b,1,h,w
    points = points / torch.clamp(points[:, :, :, [2]], 1e-8, None)

    src_ims_ = src_imgs.permute(0, 3, 1, 2)
    num_channels = src_ims_.shape[1]

    flow = points - coord.squeeze(-1)
    flow = flow.permute(0, 3, 1, 2)[:, :2, Ellipsis]

    importance = 1. / (new_z)
    importance_min = importance.amin((1, 2, 3), keepdim=True)
    importance_max = importance.amax((1, 2, 3), keepdim=True)
    weights = (importance - importance_min) / (importance_max - importance_min +
                                               1e-6) * 20 - 10
    src_mask_ = torch.ones_like(new_z)

    input_data = torch.cat([src_ims_, (1. / (new_z)), src_mask_], 1)

    output_data = splatting_function('softmax', input_data, flow,
                                     weights.detach())

    warp_feature = output_data[:, 0:num_channels, Ellipsis]
    warp_disp = output_data[:, num_channels:num_channels + 1, Ellipsis]
    warp_mask = output_data[:, num_channels + 1:num_channels + 2, Ellipsis]

    return warp_feature, warp_disp, warp_mask

  def sobel_fg_alpha(self, disp, mode='sobel', beta=10.0):
    sobel_grad = kornia.filters.spatial_gradient(
        disp, mode=mode, normalized=NORMALIZE_SOBEL)
    sobel_mag = torch.sqrt(sobel_grad[:, :, 0, Ellipsis]**2 +
                           sobel_grad[:, :, 1, Ellipsis]**2)
    alpha = torch.exp(-1.0 * beta * sobel_mag).detach()

    return alpha

  def render_and_refine(self,
                        cur_rgba,
                        cur_disp,
                        fwd_rot,
                        fwd_t,
                        k_ref,
                        noise,
                        global_code,
                        mean_latent=None,
                        truncation=1,
                        return_all=False,
                        return_rast=False,
                        use_ema=False):
    """3D render the image to the next viewpoint followed by image refinement.

    Args:
      cur_rgba: RGBA image at previous viewpoint
      cur_disp: disparity at previous viewpoint
      fwd_rot: relative camera rotation
      fwd_t: relative camera translation
      k_ref: intrinsic matrix
      noise: injected Gaussian noise
      global_code: global latent code from encoder
      mean_latent: mean latent vector from monte carlo sampling
      truncation: truncation trick in StyleGAN
      return_all: return all rendered information
      return_rast: whether to return rendered RGBD images
      use_ema: whether to use exponential moving average model

    Returns:
      next_rgbd: A refined RGBD image rendered at next viewpoint
    """
    rast_rgba, rast_disp, rast_mask = self.render_forward_splat(
        cur_rgba.permute(0, 2, 3, 1),
        1. / cur_disp,
        fwd_rot,
        fwd_t,
        k_src=k_ref,
        k_dst=k_ref)

    rast_img = (rast_rgba[:, :3])

    if return_rast:
      disocc_mask = (rast_rgba[:, 3:4] >
                     self.alpha_threshold) * rast_mask > self.mask_threshold
      disocc_mask = disocc_mask.detach()

      rast_img_masked = rast_img * disocc_mask
      rast_disp_masked = rast_disp * disocc_mask

      inputs_rast = torch.cat([rast_img_masked, rast_disp_masked, disocc_mask],
                              dim=1)

      return inputs_rast

    disocc_mask = (rast_rgba[:, 3:4] > self.alpha_threshold) * (
        rast_mask > self.mask_threshold)
    disocc_mask = disocc_mask.detach()

    rast_disp = rast_disp.clamp(
        min=MIN_DISP,
        max=1.)  # make sure input is in [0, 1] because we have sigmoid!!!

    rast_img_masked = rast_img * disocc_mask
    rast_disp_masked = rast_disp * disocc_mask

    inputs_rast = torch.cat([rast_img_masked, rast_disp_masked, disocc_mask],
                            dim=1)
    inputs_rast_ = inputs_rast * 2. - 1.
    e_features = {}

    if use_ema:
      e_features = self.generator.feature_encoder(
          (inputs_rast_, e_features))
      next_rgbd = self.generator(
          e_features,
          global_code,
          noise,
          truncation=truncation,
          truncation_mean=mean_latent)
    else:
      e_features = self.generator.feature_encoder((inputs_rast_, e_features))
      next_rgbd = self.generator(e_features, global_code, noise)

    if return_all:
      return {
          'next_rgbd': next_rgbd,
          'rast_img': rast_img,
          'rast_disp': rast_disp,
          'disocc_mask': disocc_mask,
          'dlatents_in': self.generator.dlatents_in
      }

    return next_rgbd

  def view_generation(self, data, save_dir, sky_fraction, near_fraction, lerp,
                      num_steps):
    """Run view generation in auto-regressive manner with sky correction.

    Args:
     data: Batch consisting of input camera and RGBD information
     save_dir: directory to save the images
     sky_fraction: predefined fraction of sky content in auto-pilot
     near_fraction: predefined fraction of near content in auto-pilot
     lerp: pose inteporlation ratio
     num_steps: number of steps to render along a trajecotry

    Returns:
     bool: True on success

    """
    use_auto_pilot = True
    use_sky_correction = True

    cam_speed = 0.025
    horizon = 0.4

    k_ref = data['k_ref'].to(self.device)
    k_full = data['k_full'].to(self.device)
    index = data['index'][0].item()
    disp_gamma = data['disp_gamma'][0].item()
    offset_xx = data['offset_xx'][0].item()

    if disp_gamma < 0.1:
      return -1

    disp_gamma = data['disp_gamma']

    real_rgbd = data['real_rgbd'].to(self.device)

    real_rgbd_full = data['real_rgbd_full'].to(self.device)

    np.save(os.path.join(save_dir, 'inputs.npy'), real_rgbd_full.cpu().numpy())
    np.save(os.path.join(save_dir, 'inputs_crop.npy'), real_rgbd.cpu().numpy())

    sky_offset = data['sky_offset'][0].item()
    sky_threshold = sky_offset

    img_1 = real_rgbd[:, :3, Ellipsis]
    scaled_disp = real_rgbd[:, 3:4, Ellipsis]

    mode = 'sobel'
    beta = 4.
    cur_alpha = self.sobel_fg_alpha(scaled_disp,
                                    mode,
                                    beta=beta)

    cur_rgba = torch.cat([img_1, cur_alpha], dim=1)

    cur_disp = scaled_disp.squeeze(1)

    next_look_dir = torch.as_tensor(
        np.array([0.0, 0.0, 1.0]), dtype=torch.float32)
    next_look_dir = next_look_dir.unsqueeze(0).repeat(cur_disp.shape[0], 1)

    next_move_dir = torch.as_tensor(
        np.array([0.0, 0.0, 1.0]), dtype=torch.float32)

    next_move_dir = next_move_dir.unsqueeze(0).repeat(cur_disp.shape[0], 1)

    camera_down = torch.as_tensor(
        np.array([0.0, 1.0, 0.0]), dtype=torch.float32)
    camera_down = camera_down.unsqueeze(0).repeat(cur_disp.shape[0], 1)

    speed = cam_speed * 7.5 * disp_gamma

    cur_se3 = torch.eye(4).unsqueeze(0).repeat(cur_disp.shape[0], 1, 1)
    accumulate_se3 = torch.eye(4).unsqueeze(0).repeat(cur_disp.shape[0], 1, 1)

    # simple forward moving
    camera_pos = np.array([0.0, 0.0, 0.0])
    camera_dir = np.array([0.0, 0.0, 1.0])
    camera_up = np.array([0.0, 1.0, 0.0])

    _, show_t = fly_camera.pose_from_look_direction_np(camera_pos, camera_dir,
                                                       camera_up)
    t_c2_c1 = torch.as_tensor(show_t, dtype=torch.float32).unsqueeze(0)
    position = t_c2_c1.clone()

    real_rgbd_ = real_rgbd * 2. - 1.

    num_samples = 50000
    mean_latent = self.generator.mean_latent(num_samples, self.device)
    global_code = self.generator.style_encoder(real_rgbd_)

    init_rgbm = real_rgbd_full[:, :3, Ellipsis]
    init_disp = real_rgbd_full[:, 3, Ellipsis]

    noise = [torch.randn(1, 512).to(self.device)]

    time_list = []

    # use auto-pilot to navigate automatically
    for ii in range(num_steps):
      print(sky_threshold)
      print('index %d step %d' % (index, ii))
      time0 = time.time()

      if use_auto_pilot:
        pilot_data = fly_camera.auto_pilot(
            k_ref[0].cpu(),
            cur_disp.detach().cpu(),
            speed,
            next_look_dir,
            next_move_dir,
            position,
            camera_down,
            lerp=0.05,
            movelerp=0.2,
            horizon=horizon,
            sky_fraction=sky_fraction,
            near_fraction=near_fraction)

        next_rot = pilot_data['next_rot']
        next_t = pilot_data['next_t']
        next_look_dir = pilot_data['look_dir']
        next_move_dir = pilot_data['move_dir']
        position = pilot_data['position']

        # world to camera
        next_se3 = torch.cat([next_rot, next_t.unsqueeze(-1)], dim=-1)
        next_se3 = torch.cat(
            [next_se3, torch.zeros_like(next_se3[:, 0:1, :])], dim=1)
        next_se3[:, -1, -1] = 1.
        accumulate_se3 = next_se3

        camera_down = next_se3[:, 1, :3]
        # from current to next, delta pose
        fly_se3 = torch.matmul(next_se3, torch.inverse(cur_se3)).to(self.device)
        # update cur camera to world
        cur_se3 = next_se3

      # warp init rgbd
      warp_init_rgbm, warp_init_disp, warp_init_mask = self.render_forward_splat(
          init_rgbm.permute(0, 2, 3, 1),
          1. / init_disp.clamp(1e-2, 10.),
          accumulate_se3[:, :3, :3].to(self.device),
          t_c2_c1.to(self.device),  # no translation for sky
          k_src=k_full,
          k_dst=k_full)

      crop_size_w = (init_rgbm.shape[-1] - 128) // 2
      crop_size_h = (init_rgbm.shape[-2] - 128) // 2 + offset_xx // 2

      warp_init_rgbm = warp_init_rgbm[:, :, crop_size_h:crop_size_h + 128,
                                      crop_size_w:crop_size_w + 128]
      warp_init_disp = warp_init_disp[:, :, crop_size_h:crop_size_h + 128,
                                      crop_size_w:crop_size_w + 128]
      warp_init_mask = warp_init_mask[:, :, crop_size_h:crop_size_h + 128,
                                      crop_size_w:crop_size_w + 128]

      render_outs = self.render_and_refine(
          cur_rgba.clamp(min=0., max=1.),
          cur_disp.clamp(min=1e-2, max=1.),
          fly_se3[:, :3, :3],
          fly_se3[:, :3, 3],
          k_ref,
          noise,
          global_code,
          mean_latent=mean_latent,
          truncation=0.7,
          return_all=True,
          use_ema=True)

      next_rgbd = render_outs['next_rgbd']
      rast_disp = render_outs['rast_disp']
      disoccu_mask = render_outs['disocc_mask']

      rast_disp = rast_disp.squeeze(1)
      disoccu_mask = disoccu_mask.squeeze(1)

      refine_rgb = next_rgbd[:, 0:3, Ellipsis]
      refine_disp = next_rgbd[:, 3:4, Ellipsis]

      save_path = os.path.join(save_dir, '%03d.png' % (ii + 1))
      imageio.imwrite(
          save_path, next_rgbd[0, :3,
                               Ellipsis].detach().cpu().numpy().transpose(1, 2, 0))

      next_alpha = self.sobel_fg_alpha(refine_disp, mode=mode, beta=beta)

      if use_sky_correction:
        init_sky_mask = (warp_init_mask > 0.9) & (warp_init_disp < sky_offset)
        init_sky_mask = kornia.morphology.erosion(
            init_sky_mask.float(),
            torch.ones(7, 7).to(self.device))

        m_disp_2 = (refine_disp < sky_threshold).float() * init_sky_mask
        m_disp_2 = m_disp_2.float()
        m_disp_2_blur = kornia.filters.gaussian_blur2d(m_disp_2, (5, 5),
                                                       (1.5, 1.5))
        m_disp_2_blur = m_disp_2_blur * (warp_init_mask > 0.9).float()
        m_rgb_2_blur = m_disp_2_blur.repeat(1, 3, 1, 1)

        m_disp_2 = (refine_disp < sky_threshold).float() * init_sky_mask
        m_disp_2 = m_disp_2.float()

        if torch.mean(m_disp_2) < 0.05 or torch.mean(m_disp_2) > 0.6:
          os.system('rm -r %s' % save_dir)
          return False

        warp_init_rgb = warp_init_rgbm[:, :3]
        refine_rgb = refine_rgb * (
            1 - m_rgb_2_blur) + warp_init_rgb[:, :3, Ellipsis] * m_rgb_2_blur
        refine_disp = refine_disp * (
            1. - m_disp_2_blur) + warp_init_disp * m_disp_2_blur

      refine_disp = refine_disp.squeeze(1)

      refine_disp_align = refine_disp

      # update in the next iteration
      refine_disp_align = torch.clamp(refine_disp_align, 0., 1.)
      cur_disp = refine_disp_align.clone()  # update aligned disp
      cur_rgba = torch.cat([refine_rgb, next_alpha], dim=1)

      dt = time.time() - time0
      time_list.append(dt)

    return True
