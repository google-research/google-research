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

"""Data wrappers."""

import dataclasses

import chex
from einops import rearrange
from einops import repeat
from flax import linen as nn
import jax.numpy as jnp

from omnimatte3D.src.utils import depth_viz_utils
from omnimatte3D.src.utils import model_utils
from omnimatte3D.src.utils import render_utils


def over_compose(base_rgb, over_rgb, alpha):
  chex.assert_equal_shape([base_rgb, over_rgb])
  return (base_rgb * (1 - alpha)) + (over_rgb * alpha)


@chex.dataclass
class LDI:
  """Store the layers of the ldi along with camera information."""

  rgb_layers: chex.ArrayDevice
  fg_layers: chex.ArrayDevice  # b n l 2 h w c
  fg_alphas: chex.ArrayDevice
  disp_layers: chex.ArrayDevice
  mask_layers: chex.ArrayDevice
  camtoworld: chex.ArrayDevice  # [b, l, 4, 4]
  intrinsic: chex.ArrayDevice  # [b, 1, 4]

  # ------------------------------------------------------------------
  # Compositions.
  # ------------------------------------------------------------------
  def compose_rgb(self, white_bkgd):  # pylint: disable=unused-argument
    """Function to compose the LDI."""

    out_rgb = self.rgb_layers[:, :, -1:]
    num_objects = self.fg_layers.shape[3]
    # over composite the layers.
    for lidx in range(num_objects)[::-1]:
      out_rgb = over_compose(
          out_rgb,
          self.fg_layers[:, :, :, lidx],
          self.fg_alphas[:, :, :, lidx],
      )

    out_rgb = rearrange(out_rgb, 'b n 1 h w c -> b n h w c')
    return out_rgb

  # ------------------------------------------------------------------
  # Projections.
  # ------------------------------------------------------------------
  def project_rgb(
      self,
      tgt_camtoworld,
      tgt_intrinsic,
      white_bkgd,
      merge,
      use_rts,
  ):
    """Function to project the LDI to a target camera."""
    projection_list = []
    # Project each source image sperately
    for i in range(self.rgb_layers.shape[1]):
      projection_list.append(
          render_utils.splat_layers_to_target(
              self.rgb_layers[:, i],
              self.disp_layers[:, i],
              self.mask_layers[:, i],
              self.camtoworld[:, i],
              self.intrinsic,
              tgt_camtoworld,
              tgt_intrinsic,
              use_rts,
          )
      )

    if merge:
      proj_tgt_rgb, alpha = self.merge_projections_rgb(projection_list)
    else:
      proj_tgt_rgb = jnp.stack(projection_list, axis=1)[Ellipsis, :3]
      alpha = jnp.stack(projection_list, axis=1)[Ellipsis, -1:]

    proj_tgt_rgb = render_utils.rgba_to_rgb(proj_tgt_rgb, white_bkgd)
    return proj_tgt_rgb, alpha

  # ------------------------------------------------------------------
  # Merge Projections.
  # ------------------------------------------------------------------
  def merge_projections_rgb(self, projection_list):
    """Merge multiple projections."""
    projections = jnp.stack(projection_list, axis=1)  # [b, n, h, w ,c]
    projection_weights = projections[Ellipsis, -1:]
    projection_rgba = projections[Ellipsis, :-1]

    weights_sum = projection_weights.sum(axis=1)
    out_image = (projection_rgba * projection_weights).sum(axis=1) / (
        weights_sum + 1e-4
    )

    return out_image, weights_sum

  def mask_rgb_disp_layers(self, min_disp):
    self.rgb_layers = self.rgb_layers * self.mask_layers
    self.disp_layers = self.disp_layers * self.mask_layers + (
        (1.0 - self.mask_layers) * min_disp
    )

  @property
  def shape(self):
    shape_str = ''
    for field in dataclasses.fields(self):
      shape_str += '{}: {} \n'.format(
          field.name,
          getattr(self, field.name).shape,
      )
    return shape_str


def crop(img):
  crop_ratio = 20
  image_height, image_width = img.shape[-3:-1]
  h_crop = image_height // crop_ratio
  w_crop = image_width // crop_ratio
  return img[Ellipsis, h_crop:-h_crop, w_crop:-w_crop, :]


@chex.dataclass
class RenderResult:
  """Store the model prediction."""

  ldi: LDI
  bg_ldi: LDI
  ft_ldi: LDI
  pred_src_rgb: chex.ArrayDevice
  pred_tgt_rgb: chex.ArrayDevice
  pred_tgt_alpha: chex.ArrayDevice
  pred_rgb_proj_ft: chex.ArrayDevice
  pred_alpha_proj_ft: chex.ArrayDevice
  crop_projection: bool

  def disp_layer_loss(self, gt_disp_layers, gt_loss_mask):
    chex.assert_equal_shape([self.ldi.disp_layers, gt_disp_layers])
    chex.assert_rank(self.ldi.disp_layers, 6)
    chex.assert_equal_rank([gt_loss_mask, self.ldi.disp_layers])
    loss = (
        ((self.ldi.disp_layers - gt_disp_layers) * gt_loss_mask) ** 2
    ).mean()
    return loss

  def disp_smooth_loss(self):
    dx, dy = model_utils.gradient(self.ldi.disp_layers)
    dxx, dxy = model_utils.gradient(dx)
    dyx, dyy = model_utils.gradient(dy)

    return (
        jnp.mean(jnp.abs(dxx))
        + jnp.mean(jnp.abs(dxy))
        + jnp.mean(jnp.abs(dyx))
        + jnp.mean(jnp.abs(dyy))
    )

  def shadow_smooth_loss(self):
    dx, dy = model_utils.gradient(self.ldi.fg_alphas)
    dxx, dxy = model_utils.gradient(dx)
    dyx, dyy = model_utils.gradient(dy)
    return jnp.mean(dxx) + jnp.mean(dxy) + jnp.mean(dyx) + jnp.mean(dyy)

  def fg_rgb_smooth_loss(self):
    dx, dy = model_utils.gradient(self.ldi.fg_layers)
    dxx, dxy = model_utils.gradient(dx)
    dyx, dyy = model_utils.gradient(dy)
    return jnp.mean(dxx) + jnp.mean(dxy) + jnp.mean(dyx) + jnp.mean(dyy)

  def fg_alpha_reg_l1_loss(self):
    loss = jnp.mean(self.ldi.fg_alphas)
    return loss

  def fg_alpha_reg_l0_loss(self):
    l0_pred = (nn.sigmoid(self.ldi.fg_alphas * 5.0) - 0.5) * 2.0
    loss = jnp.mean(l0_pred)
    return loss

  def fg_mask_loss(self, gt_mask_layers):
    """Compute loss on fg alpha."""
    # the one here is due to poor design choices at earlier stage :(
    mask_pred = rearrange(self.ldi.fg_alphas, 'b n 1 m h w c -> b n m h w c')
    mask_tgt = gt_mask_layers

    chex.assert_equal_shape([mask_pred, mask_tgt])
    chex.assert_rank(mask_pred, 6)  # b n l h w c

    zero_mask = mask_tgt == 0.0
    one_mask = mask_tgt == 1.0

    l1_loss = jnp.abs(mask_pred - mask_tgt)
    chex.assert_equal_shape([l1_loss, zero_mask])
    chex.assert_equal_shape([l1_loss, one_mask])

    num_layers = mask_tgt.shape[-4]
    total_loss = 0
    # Compute loss for each layer seperately.
    for lidx in range(num_layers):
      layer_zero_loss = (l1_loss[:, :, lidx] * zero_mask[:, :, lidx]).sum() / (
          1 + zero_mask[:, :, lidx].sum()
      )
      layer_one_loss = (l1_loss[:, :, lidx] * one_mask[:, :, lidx]).sum() / (
          1 + one_mask[:, :, lidx].sum()
      )
      total_loss += 0.5 * (layer_zero_loss + layer_one_loss)

    return total_loss

  def src_rgb_grad_loss(self, gt_src):
    rgb_layers = rearrange(self.ldi.rgb_layers, 'b n 1 h w c -> b n h w c')
    chex.assert_equal_rank([rgb_layers, gt_src])
    pred_grad_x = model_utils.gradient(rgb_layers)[0]
    gt_grad_x = model_utils.gradient(gt_src)[0]
    loss = ((pred_grad_x - gt_grad_x) ** 2).mean()
    return loss

  def src_rgb_recon_loss(self, gt_src):
    chex.assert_equal_shape([self.pred_src_rgb, gt_src])
    loss = ((self.pred_src_rgb - gt_src) ** 2).mean()
    return loss

  def proj_far_rgb_loss(self):
    """Function to compute loss on projection to further away timestep."""
    assert self.pred_rgb_proj_ft.shape[1] == 2
    proj_img1 = self.pred_rgb_proj_ft[:, 0]  # b h w c
    proj_alpha1 = self.pred_alpha_proj_ft[:, 0]
    proj_img2 = self.pred_rgb_proj_ft[:, 1]  # b h w c
    proj_alpha2 = self.pred_alpha_proj_ft[:, 1]
    tgt_img = rearrange(self.ft_ldi.rgb_layers, 'b 1 h w c-> b h w c')

    chex.assert_equal_shape([proj_img1, tgt_img])
    chex.assert_equal_shape([proj_img2, tgt_img])
    chex.assert_equal_rank([proj_img1, proj_alpha1])
    chex.assert_equal_rank([proj_img2, proj_alpha2])

    if self.crop_projection:
      proj_img1 = crop(proj_img1)
      proj_img2 = crop(proj_img2)
      tgt_img = crop(tgt_img)

    loss_1 = (((proj_img1 - tgt_img) * proj_alpha1) ** 2).mean()
    loss_2 = (((proj_img2 - tgt_img) * proj_alpha2) ** 2).mean()
    return loss_1 + loss_2

  def compute_total_loss(self, batch, alpha_dict_all):
    """Compute the loss dict."""
    loss_dict = {}
    # -----------------------------------------------------------------------
    # disp layer loss.
    # -----------------------------------------------------------------------
    loss_dict['disp_smooth_loss'] = self.disp_smooth_loss()
    loss_dict['disp_layer_loss'] = self.disp_layer_loss(
        batch['src_disp'][:, :, None], batch['src_mask_loss_layer']
    )

    # fg smooth loss.
    loss_dict['shadow_smooth_loss'] = self.shadow_smooth_loss()
    loss_dict['fg_alpha_reg_l0_loss'] = self.fg_alpha_reg_l0_loss()
    loss_dict['fg_alpha_reg_l1_loss'] = self.fg_alpha_reg_l1_loss()
    loss_dict['fg_mask_loss'] = self.fg_mask_loss(batch['src_fg_mask_layer'])

    # -----------------------------------------------------------------------
    # src reconstruction.
    # -----------------------------------------------------------------------
    loss_dict['src_rgb_recon_loss'] = self.src_rgb_recon_loss(batch['src_rgb'])

    # -----------------------------------------------------------------------
    # projection loss tp further away timestep.
    # -----------------------------------------------------------------------
    loss_dict['proj_far_rgb_loss'] = self.proj_far_rgb_loss()

    total_loss = 0
    # To store only the active alphas for logging.
    alpha_dict = {}
    for loss_name, loss_value in loss_dict.items():
      alpha_name = loss_name[:-4] + 'alpha'  # remove loss and add alpha.
      alpha_dict[alpha_name] = alpha_dict_all.get(alpha_name)
      total_loss += loss_value * alpha_dict.get(alpha_name)

    stat_dict = loss_dict
    stat_dict.update(alpha_dict)

    return total_loss, stat_dict

  def get_log_dict(self, batch):
    """Get the data for logging."""
    log_dict = {}

    log_dict['tgt_images/gt'] = batch['rgb'][0]
    log_dict['tgt_images/pred'] = self.pred_tgt_rgb[0]
    log_dict['src_images/gt'] = batch['src_rgb'][0]
    log_dict['src_images/pred'] = self.pred_src_rgb[0]

    log_dict['tgt_images/alpha'] = self.pred_tgt_alpha[0]

    if 'src_rgb_layer' in batch.keys():
      has_layers = True
      chex.assert_equal_shape([self.ldi.rgb_layers, batch['src_rgb_layer']])
      rgb_layer_error = jnp.linalg.norm(
          (self.ldi.rgb_layers - batch['src_rgb_layer']),
          axis=-1,
          keepdims=True,
      )

      chex.assert_equal_shape([self.ldi.disp_layers, batch['src_disp_layer']])
      disp_layer_error = jnp.abs(self.ldi.disp_layers - batch['src_disp_layer'])

      mask_layer_error = jnp.abs(self.ldi.mask_layers - batch['src_mask_layer'])
    else:
      has_layers = False

    # Projection to further away timestep.
    proj_img_ft = self.pred_rgb_proj_ft[0]
    pred_img_ft = self.ft_ldi.rgb_layers[0]
    log_dict['ft_imgs/rgb_projs'] = proj_img_ft
    log_dict['ft_imgs/rgb_preds'] = pred_img_ft

    proj_alpha_ft = self.pred_alpha_proj_ft[0]
    log_dict['ft_imgs/alpha_projs'] = proj_alpha_ft
    proj_error = (
        jnp.linalg.norm(proj_img_ft - pred_img_ft, axis=-1, keepdims=True)
        * proj_alpha_ft
    )
    log_dict['ft_imgs/proj_error'] = proj_error

    for i in range(batch['src_rgb'].shape[1]):
      # rgb.
      log_dict['rgb_layers_{}/pred'.format(i)] = self.ldi.rgb_layers[0, i]
      if has_layers:
        log_dict['rgb_layers_{}/gt'.format(i)] = batch['src_rgb_layer'][0, i]
        log_dict['errors/rgb_layers_{}'.format(i)] = rgb_layer_error[0, i]

      log_dict['fg_layers_{}/pred'.format(i)] = self.ldi.fg_layers[0, i, 0]
      log_dict['fg_alphas_{}/pred'.format(i)] = self.ldi.fg_alphas[0, i, 0]
      # depth.
      log_dict['disp_input_layers_{}/gt'.format(i)] = (
          model_utils.normalize_depth(batch['src_disp_input_layer'][0, i])
      )
      log_dict['disp_layers_{}/pred'.format(i)] = model_utils.normalize_depth(
          self.ldi.disp_layers[0, i]
      )
      log_dict['raw_disp_layers_{}/pred'.format(i)] = self.ldi.disp_layers[0, i]
      if has_layers:
        log_dict['errors/disp_layers_{}'.format(i)] = disp_layer_error[0, i]

      # mask.
      log_dict['mask_layers_{}/pred'.format(i)] = self.ldi.mask_layers[0, i]
      if has_layers:
        log_dict['mask_layers_{}/gt'.format(i)] = batch['src_mask_layer'][0, i]
        log_dict['errors_mask/mask_layers_{}'.format(i)] = mask_layer_error[
            0, i
        ]

    return log_dict

  def get_video_dict(self, batch):
    """Construct video for logging."""
    video_dict = {}

    rgb_layers = rearrange(
        self.ldi.rgb_layers[:, 0], 'b l h w c -> (b h) (l w) c'
    )

    fg_layers = rearrange(
        self.ldi.fg_layers[:, 0], 'b l m h w c -> (b h) (l m w) c'
    )
    fg_alphas = repeat(
        self.ldi.fg_alphas[:, 0], 'b l m h w 1 -> (b h) (l m w) c', c=3
    )
    gt_mask = batch['src_fg_mask_layer'][:, 0]
    gt_mask = repeat(gt_mask, 'b l h w 1 -> (b h) (l w) c', c=3)

    fg_layers = fg_layers * fg_alphas

    # normalize disp with global value for all layers.
    disp_layers = rearrange(
        self.ldi.disp_layers[:, 0], 'b l h w c -> b h (l w) c'
    )
    disp_layers = depth_viz_utils.colorize_depth_map(disp_layers)
    disp_layers = rearrange(disp_layers, 'b h lw c -> (b h) lw c')

    all_layers = jnp.concatenate(
        [rgb_layers, fg_layers, fg_alphas, disp_layers], axis=1
    )
    video_dict['all_layers'] = all_layers
    video_dict['gt_mask'] = gt_mask

    src_pred = rearrange(self.ldi.rgb_layers[:, 0], '1 1 h w c -> 1 h w c')
    compose_rgb1 = over_compose(
        self.ldi.rgb_layers,
        self.ldi.fg_layers[:, :, :, 1],
        self.ldi.fg_alphas[:, :, :, 1],
    )[:, 0]
    compose_rgb1 = rearrange(compose_rgb1, '1 1 h w c -> 1 h w c')

    compose_rgb2 = over_compose(
        self.ldi.rgb_layers,
        self.ldi.fg_layers[:, :, :, 0],
        self.ldi.fg_alphas[:, :, :, 0],
    )[:, 0]
    compose_rgb2 = rearrange(compose_rgb2, '1 1 h w c -> 1 h w c')

    src_gt = batch['src_rgb'][:, 0]
    src_compare = jnp.concatenate(
        [src_pred, compose_rgb1, compose_rgb2, src_gt], axis=0
    )
    src_compare = rearrange(src_compare, 'n h w c -> h (n w) c', n=4)
    video_dict['src_compare'] = src_compare

    return video_dict
