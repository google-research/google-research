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

"""Davis Dataset."""

INTERNAL = False  # pylint: disable=g-statement-before-imports

import os
from os import path

import cv2
from einops import rearrange
import imageio
import jax
import numpy as np
from PIL import Image
import scipy

from omnimatte3D.src.datasets import base
from omnimatte3D.src.utils import file_utils


class Davis(base.Dataset):
  """Davis dataset."""

  def _load_renderings(self, config):
    self.data_dict, self.src_bg_dict1, self.src_bg_dict2 = load_davis_data(
        config
    )

    basedir = config.dataset.basedir
    scene = config.dataset.scene
    slamdir = path.join(basedir, 'casual_slam_all', scene, 'preprocess')
    with file_utils.open_file(
        os.path.join(slamdir, 'frame_overlap.np'), 'r'
    ) as fp:
      frame_overlap = np.load(fp)
      # since tgt state from time 1 to t-1
      frame_overlap = frame_overlap[1:-1, 1:-1]
      # Frame ranking.
      frame_rank = np.argsort(frame_overlap, axis=-1)[:, ::-1]
      assert frame_rank.shape[-1] > 10
      self.frame_rank = frame_rank[:, :10]  # choose the top 10 frames.

    self._n_examples = self.data_dict['rgb'].shape[0]
    assert self.frame_rank.shape[0] == self._n_examples

    return

  def _next_train(self):
    """Sample next training batch."""
    # We assume all images in the dataset are the same resolution, so we can use
    # the same width/height for sampling all pixels coordinates in the batch.
    # Get the target index.
    target_idx = np.random.randint(0, self._n_examples, (self._batch_size,))
    batch = jax.tree_util.tree_map(lambda x: x[target_idx], self.data_dict)

    # Get index for a further away timestep.
    if self.src_bg_dict1 is not None:
      shift_idx1 = np.concatenate(
          [np.random.choice(x, 1) for x in self.frame_rank[target_idx]], axis=0
      )
      assert np.all(
          shift_idx1 != target_idx
      ), 'Error in frame selection. Cannot project to same frame.'
      src_batch1 = jax.tree_util.tree_map(
          lambda x: x[shift_idx1],
          self.src_bg_dict1,
      )
      batch.update(src_batch1)

      # Further away image to project to.
      shift_idx2 = (target_idx + shift_idx1) // 2
      src_batch2 = jax.tree_util.tree_map(
          lambda x: x[shift_idx2], self.src_bg_dict2
      )
      batch.update(src_batch2)

    return batch

  def _next_test(self):
    """Sample next test batch (one full image)."""
    # Use the next camera index.
    cam_idx = self._test_camera_idx
    self._test_camera_idx = (self._test_camera_idx + 1) % self._n_examples

    batch = jax.tree_util.tree_map(lambda x: x[[cam_idx]], self.data_dict)

    # Get index for a further away timestep.
    if self.src_bg_dict1 is not None:
      shift_idx1 = self.frame_rank[cam_idx][0]
      src_batch1 = jax.tree_util.tree_map(
          lambda x: x[[shift_idx1]],
          self.src_bg_dict1,
      )
      batch.update(src_batch1)

      # Further away image to project to.
      shift_idx2 = (shift_idx1 + cam_idx) // 2
      src_batch2 = jax.tree_util.tree_map(
          lambda x: x[[shift_idx2]], self.src_bg_dict2
      )
      batch.update(src_batch2)

    return batch


def load_davis_data(config):
  """Function to load davis data."""
  basedir = config.dataset.basedir
  scene = config.dataset.scene
  imgdir = path.join(basedir, 'JPEGImages', '1080p', scene)

  slamdir = path.join(basedir, 'casual_slam_all', scene, 'preprocess')
  # slamdir = path.join(basedir, 'casual_slam', scene)
  if not INTERNAL:
    maskdir = path.join(basedir, 'Annotations', 'maskrcnn_mask', scene)

  # Load the disp first as they are save in np files and will provide meta data
  # for the number of images and resolution.
  disps = load_all_disp(slamdir)

  with file_utils.open_file(
      os.path.join(slamdir, 'min_max_depth.npy'), 'r'
  ) as fp:
    min_max_depth = np.load(fp)
    assert min_max_depth.shape == (2,)
    min_depth = float(min_max_depth[0])
    max_depth = float(min_max_depth[1])

  config.dataset.min_depth, config.dataset.max_depth = min_depth, max_depth
  num_images, image_height, image_width, _ = disps.shape
  # --------------------------------------------------------------------------
  # Load images and masks.
  images = load_all_images(imgdir, num_images, image_height, image_width)
  masks = load_all_maskrcnn_mask(maskdir, num_images, image_height, image_width)

  camtoworlds, intrinsic_matrix = load_camera_data(slamdir)

  intrinsic_arr = np.zeros((intrinsic_matrix.shape[0], 4))
  intrinsic_arr[:, 0] = intrinsic_matrix[:, 0, 0]
  intrinsic_arr[:, 1] = intrinsic_matrix[:, 1, 1]
  intrinsic_arr[:, 2] = intrinsic_matrix[:, 0, 2]
  intrinsic_arr[:, 3] = intrinsic_matrix[:, 1, 2]

  scene_dict = {
      'rgb': images,
      'disp': disps,
      'img_mask': masks,
      'camtoworld': camtoworlds,
      'intrinsic': intrinsic_arr,
      'time_step': (
          np.arange(images.shape[0])[:, None] / images.shape[0]
      ),  # normalize to 0-1.
  }
  config.dataset.num_objects = masks.shape[1]

  # mask out the forground object to create the input rgb and disp layers.
  # masks has shape (b h w 1) containing mask of foreground object.
  fg_masks = (masks.sum(1) >= 1.0) * 1.0
  fg_complement_mask = 1.0 - fg_masks[:, None]
  layer_masks = np.concatenate([masks, fg_complement_mask], axis=1)
  # the none below add the asix for layer dimension.
  layer_input_rgb = scene_dict['rgb'][:, None] * layer_masks + np.zeros_like(
      scene_dict['rgb'][:, None]
  ) * (1.0 - layer_masks)

  scene_dict['layer_input_rgb'] = layer_input_rgb

  if config.train.use_gt_rgb:
    scene_dict['layer_rgb'] = images[:, None].copy()

  layer_input_disp = scene_dict['disp'][:, None] * layer_masks + np.zeros_like(
      scene_dict['disp'][:, None]
  ) * (1.0 - layer_masks)
  scene_dict['layer_input_disp'] = layer_input_disp
  if config.train.use_gt_disp:
    scene_dict['layer_disp'] = disps[:, None].copy() + 1e-6

  scene_dict['layer_mask'] = layer_masks
  scene_dict['mask_loss_layer'] = fg_complement_mask  # b 1 h w 1
  scene_dict['bg_mask'] = fg_complement_mask[:, 0]

  dilate_mask = []
  for obj_masks in scene_dict['img_mask']:
    dilate_obj_mask = []
    for x in obj_masks:
      dilate_obj_mask.append(
          scipy.ndimage.binary_dilation(
              x[Ellipsis, 0], np.ones((7, 7)), iterations=1
          )
          * 1.0
      )
    dilate_obj_mask = np.stack(dilate_obj_mask, axis=0)
    dilate_mask.append(dilate_obj_mask)

  dilate_mask = rearrange(np.stack(dilate_mask), 'b n h w -> b n h w 1')
  uncertain_region = dilate_mask - scene_dict['img_mask']
  trimap_mask = (scene_dict['img_mask'] * (1 - uncertain_region)) + (
      0.5 * uncertain_region
  )
  scene_dict['fg_mask'] = trimap_mask

  scene_dict['flow_layers'] = np.zeros_like(images[:, None])  # b 1 h w 3

  # For the source image with prev and next time step.
  prev_scene_dict = jax.tree_util.tree_map(lambda x: x[:-2], scene_dict)
  next_scene_dict = jax.tree_util.tree_map(lambda x: x[2:], scene_dict)
  num_images = images.shape[0]

  # Create the src dict from the prev and next scene dict.
  def stack_fn(key1, key2):
    return np.stack([prev_scene_dict[key1], next_scene_dict[key2]], axis=1)

  src_dict = {
      'src_rgb': stack_fn('rgb', 'rgb'),
      'src_rgb_input_layer': stack_fn('layer_input_rgb', 'layer_input_rgb'),
      'src_disp': stack_fn('disp', 'disp'),
      'src_disp_input_layer': stack_fn('layer_input_disp', 'layer_input_disp'),
      'src_bg_mask': stack_fn('bg_mask', 'bg_mask'),
      'src_fg_mask_layer': stack_fn('fg_mask', 'fg_mask'),
      'src_mask_layer': stack_fn('layer_mask', 'layer_mask'),
      'src_mask_loss_layer': stack_fn('mask_loss_layer', 'mask_loss_layer'),
      'src_camtoworld': stack_fn('camtoworld', 'camtoworld'),
      'src_scene_flow': stack_fn('flow_layers', 'flow_layers'),
      'src_time_step': stack_fn('time_step', 'time_step'),
  }
  if config.train.use_gt_rgb:
    src_dict['src_rgb_layer'] = stack_fn('layer_rgb', 'layer_rgb')
  if config.train.use_gt_disp:
    src_dict['src_disp_layer'] = stack_fn('layer_disp', 'layer_disp')

  curr_scene_dict = jax.tree_util.tree_map(lambda x: x[1:-1], scene_dict)

  num_images = curr_scene_dict['rgb'].shape[0]

  src_bg_scene_dict1 = jax.tree_util.tree_map(
      lambda x: x.copy().astype(np.float32), curr_scene_dict
  )
  sub_keys = [
      'camtoworld',
      'layer_input_rgb',
      'layer_input_disp',
      'layer_mask',
  ]
  if config.train.use_gt_rgb:
    sub_keys.append('layer_rgb')
  if config.train.use_gt_disp:
    sub_keys.append('layer_disp')

  src_bg_scene_dict1 = {
      'bg_src_' + str(key): val
      for key, val in src_bg_scene_dict1.items()
      if key in sub_keys
  }

  src_bg_scene_dict2 = jax.tree_util.tree_map(
      lambda x: x.copy().astype(np.float32), curr_scene_dict
  )

  sub_keys.append('rgb')
  src_bg_scene_dict2 = {
      'bg_src_tgt_' + str(key): val
      for key, val in src_bg_scene_dict2.items()
      if key in sub_keys
  }
  scene_dict = dict(
      [(key, val) for key, val in scene_dict.items() if 'layer_' not in key]
  )
  curr_scene_dict = jax.tree_util.tree_map(lambda x: x[1:-1], scene_dict)

  curr_scene_dict.update(src_dict)

  curr_scene_dict = jax.tree_util.tree_map(
      lambda x: x.astype(np.float32), curr_scene_dict
  )

  return curr_scene_dict, src_bg_scene_dict1, src_bg_scene_dict2


def load_scene(config):
  """Load kubric data with reference images."""
  # Get the train data.
  train_scene_output = load_davis_data(config)
  test_scene_output = load_davis_data(config)

  return train_scene_output, test_scene_output


def load_all_images(imgdir, num_images, image_height, image_width):
  """Function to load all the images."""

  def load_single_image(filename):
    """Function to load a single image."""
    with file_utils.open_file(filename, 'rb') as fimg:
      colorimage = imageio.imread(fimg).astype(np.float32)
    colorimage = colorimage / 255.0
    colorimage = cv2.resize(colorimage, (image_width, image_height))
    return colorimage[Ellipsis, :3]

  imgfiles = [
      path.join(imgdir, '{0:05d}.jpg'.format(f)) for f in range(num_images)
  ]

  if not INTERNAL:
    images = [load_single_image(f) for f in imgfiles]

  images = np.stack(images, axis=0)
  return images


def load_all_disp(dispdir):
  """Disparity is saved as a np file."""
  disp_file = path.join(dispdir, 'all_disp.npy')
  with file_utils.open_file(disp_file, 'r') as fp:
    all_disp = np.load(fp)
  return all_disp


def load_all_maskrcnn_mask(maskdir, num_masks, image_height, image_width):
  """Function to load all masks."""

  def load_single_mask(filename):
    """Function to load a single mask file."""
    with file_utils.open_file(filename, 'rb') as fmask:
      maskimg = Image.open(fmask)
      maskimg = np.asarray(maskimg, dtype=np.float32) / 255.0
      maskimg = cv2.resize(
          maskimg,
          (image_width, image_height),
          interpolation=cv2.INTER_NEAREST,
      )
    return maskimg

  obj_mask_dirs = [
      os.path.join(maskdir, x)
      for x in sorted(file_utils.listdir(maskdir))
      if file_utils.isdir(os.path.join(maskdir, x))
  ][::-1]
  all_mask_list = []
  for obj_dir in obj_mask_dirs:
    maskfiles = [
        path.join(obj_dir, '{0:04d}.png'.format(f + 1))
        for f in range(num_masks)
    ]
    if not INTERNAL:
      masks = [load_single_mask(f) for f in maskfiles]

    masks = np.stack(masks, axis=0)[Ellipsis, None]
    assert masks.max() == 1.0, 'mask has no ones. scaling might be wrong.'
    all_mask_list.append(masks)
  masks = np.stack(all_mask_list, axis=1)

  assert masks.max() == 1.0, 'mask has no ones. scaling might be wrong.'
  assert masks.min() == 0.0, 'mask has no zeros. scaling might be wrong.'
  return masks


def load_camera_data(cameradir):
  """Cameara are stored as npz files."""

  camtoworld_file = path.join(cameradir, 'all_camtoworld.npz')
  intrinsic_file = path.join(cameradir, 'all_intrinsic_matrix.npz')

  with file_utils.open_file(camtoworld_file, 'r') as fp:
    camtoworlds = np.load(fp)['camtoworld']

  with file_utils.open_file(intrinsic_file, 'r') as fp:
    intrinsic_matrix = np.load(fp)['intrinsic_matrix']

  return camtoworlds, intrinsic_matrix
