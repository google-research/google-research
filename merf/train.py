# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Main script for training, baking, evaulation and exporting.

Trains, bakes, evaluates a MERF and finally exports the reconstructed scene
to a format that can be read by the MERF webviewer.
"""

import functools
import gc
import os
from os import path
import time

from absl import app
import flax
from flax.training import checkpoints
import gin
from internal import baked_render
from internal import baking
from internal import camera_utils
from internal import configs
from internal import datasets
from internal import export
from internal import grid_utils
from internal import image
from internal import math
from internal import models
from internal import quantize
from internal import train_utils
from internal import utils
import jax
from jax import random
import jax.numpy as jnp
import mediapy as media
import numpy as np
import skimage.measure

configs.define_common_flags()
jax.config.parse_flags_with_absl()


def main(unused_argv):
  dataset = test_dataset = None

  # Shift the numpy random seed by host_id() to shuffle data loaded by different
  # hosts.
  np.random.seed(20201473 + jax.process_index())

  config = configs.load_config()
  log_dir = config.checkpoint_dir

  # Everything below can be copied one-to-one from colab.
  if not utils.isdir(log_dir):
    utils.makedirs(log_dir)
  temp_dir = path.join(log_dir, 'temp')
  if not utils.isdir(temp_dir):
    utils.makedirs(temp_dir)

  if jax.host_id() == 0:
    with utils.open_file(path.join(log_dir, 'config.gin'), 'w') as f:
      f.write(gin.config_str())

  indoor_scenes = [
      'kitchenlego',
      'fulllivingroom',
      'officebonsai',
      'kitchencounter',
  ]
  for scene_name in indoor_scenes:
    if scene_name in config.data_dir:
      print('indoor scene, setting config.factor = 2')
      config.factor = 2

  # Load dataset.
  dataset = datasets.load_dataset('train', config.data_dir, config, dataset)
  test_dataset = datasets.load_dataset(
      'test', config.data_dir, config, test_dataset
  )

  # Load checkpoint.
  _, state, _, _, _ = train_utils.setup_model(
      config, random.PRNGKey(20200823), dataset
  )
  state = checkpoints.restore_checkpoint(log_dir, state)
  step = int(state.step) // config.gradient_accumulation_steps
  print(f'Loaded checkpoint from {log_dir}, step: {step}')

  # Train loop.
  if step < config.max_steps + 1:
    # Create model and training functions.
    _, state, render_eval_pfn, train_pstep, _ = train_utils.setup_model(
        config, random.PRNGKey(config.model_seed), dataset
    )
    pdataset = flax.jax_utils.prefetch_to_device(dataset, 3)
    pstate = flax.jax_utils.replicate(state)
    rngs = random.split(random.PRNGKey(1234567), jax.local_device_count())

    np_to_jax = lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x
    cameras = jax.tree_util.tree_map(np_to_jax, dataset.cameras)

    # Use first image of test dataset to track PSNR progress during training.
    val_idx = 0
    val_rays = datasets.cam_to_rays(test_dataset, val_idx)
    val_psnr_history = {}

    warmup_iters = 10  # only for iters/sec measurement.
    train_start_time = None

    data_loss_history = []
    for i in range(config.max_steps + 1):
      if i == warmup_iters:
        train_start_time = time.time()
      step = pstate.step[0] // config.gradient_accumulation_steps
      train_frac = jnp.clip((step - 1) / (config.max_steps - 1), 0, 1)
      stats = None
      for _ in range(config.gradient_accumulation_steps):
        batch = next(pdataset)
        pstate, stats, rngs = train_pstep(
            rngs, pstate, batch, cameras, train_frac
        )
      stats = flax.jax_utils.unreplicate(stats)

      if i > warmup_iters:
        iters_per_sec = (i - warmup_iters) / (time.time() - train_start_time)
      else:
        iters_per_sec = 'n/a'

      data_loss_history.append(-10 * np.log10(stats['losses']['data']))

      if i % config.print_every == 0 and i > 0:
        print(f'Iteration: {i}, iters/sec: {iters_per_sec}', flush=True)
        print(
            ' '.join(
                [
                    f'{k}: {-10*np.log10(v):.3f}'
                    for k, v in stats['losses'].items()
                ]
            )
        )

      if i % config.train_render_every == 0 and i > 0:
        state = flax.jax_utils.unreplicate(pstate)
        rendering = models.render_image(
            functools.partial(render_eval_pfn, state.params, train_frac),
            val_rays,
            rngs[0],
            config,
            verbose=False,
        )

        val_psnr = -10 * np.log10(
            np.mean(np.square(rendering['rgb'] - test_dataset.images[val_idx]))
        )
        val_psnr_history[i] = val_psnr
        utils.save_json(
            val_psnr_history, os.path.join(log_dir, 'val_psnr_history.json')
        )

      if (i % config.checkpoint_every == 0 or i == config.max_steps) and i > 0:
        print('Saving checkpoint', i)
        state = flax.jax_utils.unreplicate(pstate)
        checkpoints.save_checkpoint(log_dir, state, i, keep=1, overwrite=True)
  if config.stop_after_training:
    quit()

  # Test
  metric_harness = image.MetricHarness()
  _, _, render_eval_pfn, _, _ = train_utils.setup_model(
      config, random.PRNGKey(20200823), dataset
  )
  orig_render_dir = path.join(log_dir, 'orig_render')
  if not utils.isdir(orig_render_dir):
    utils.makedirs(orig_render_dir)

  # dataset_type is either "train" or "test".
  def render_entire_dataset(dataset_type):
    print(f'rendering {dataset_type} images')
    frames = []
    all_psnr = []
    all_ssim = []
    train_frac = 1.0
    rendered_dataset = dataset if dataset_type == 'train' else test_dataset
    for img_idx in range(len(rendered_dataset.images)):
      rays = datasets.cam_to_rays(rendered_dataset, img_idx)
      rendering = models.render_image(
          functools.partial(render_eval_pfn, state.params, train_frac),
          rays,
          random.PRNGKey(42),
          config,
          verbose=False,
      )
      cpu = jax.local_devices(backend='cpu')[0]
      metric = metric_harness(
          jax.device_put(rendering['rgb'], cpu),
          jax.device_put(rendered_dataset.images[img_idx], cpu),
      )
      all_psnr.append(metric['psnr'])
      all_ssim.append(metric['ssim'])
      print(f'image#: {img_idx} PSNR: {metric["psnr"]}, SSIM: {metric["ssim"]}')
      frames.append(rendering['rgb'])
      img_dir = os.path.join(
          orig_render_dir, f'rgb.{dataset_type}.{img_idx:03d}.png'
      )
      utils.save_img_u8(rendering['rgb'], img_dir)
      img_dir = os.path.join(
          orig_render_dir, f'gt.{dataset_type}.{img_idx:03d}.png'
      )
      utils.save_img_u8(rendered_dataset.images[img_idx], img_dir)

    # Write metrics as .json file to disk.
    all_metrics = {
        'psnr_avg': np.mean(all_psnr),
        'ssim_avg': np.mean(all_ssim),
        'psnr': all_psnr,
        'ssim': all_ssim,
    }
    utils.save_json(
        all_metrics,
        os.path.join(orig_render_dir, f'metrics.{dataset_type}.json'),
    )
    print(
        f'average over all {dataset_type} images: PSNR:'
        f' {all_metrics["psnr_avg"]}, SSIM: {all_metrics["ssim_avg"]}'
    )

    # For the synthetic dataset the test set images are from a smooth path.
    video_dir = os.path.join(orig_render_dir, 'test.mp4')
    media.write_video(video_dir, frames, fps=30)

  render_entire_dataset('test')

  # Additionally render smooth elliptic path.
  gc.collect()
  ellipse_path_frames = []
  height, width = test_dataset.images[0].shape[:2]
  for render_pose in camera_utils.generate_ellipse_path(
      dataset.poses, n_frames=200
  ):
    factor = 1
    rays = camera_utils.cast_pinhole_rays(
        render_pose,
        height / factor,
        width / factor,
        dataset.focal / factor,
        config.near,
        config.far,
        xnp=jnp,
    )
    train_frac = 1.0
    rendering = models.render_image(
        functools.partial(render_eval_pfn, state.params, train_frac),
        rays,
        random.PRNGKey(42),
        config,
        verbose=False,
    )
    ellipse_path_frames.append(rendering['rgb'])
  video_dir = os.path.join(orig_render_dir, 'path.mp4')
  media.write_video(video_dir, ellipse_path_frames, fps=30)

  # Optionally render train set (off by default).
  if config.render_train_set:
    render_entire_dataset('train')
  if config.stop_after_testing:
    quit()

  # Baking
  gc.collect()

  # Visibility culling: Mark only voxels as "alive" that contribute to the
  # rendering of a training image.
  planes_features = (
      planes_density
  ) = (
      sparse_grid_features
  ) = sparse_grid_density = sparse_grid_block_indices = None
  grid_config = grid_utils.calculate_grid_config(config)
  alpha_threshold = weight_threshold = 0.005
  use_triplanes = config.triplane_resolution > 0
  use_sparse_grid = config.sparse_grid_resolution > 0
  subsampling_factor = 1
  load_alive_voxels_from_disk = True
  save_alive_voxels_to_disk = True
  use_alpha_culling = True
  use_only_first_image = False  # Useful for fast debugging.
  alive_voxels_path = path.join(temp_dir, 'alive_voxels.npy')

  if load_alive_voxels_from_disk and utils.file_exists(alive_voxels_path):
    alive_voxels = utils.load_np(alive_voxels_path)
  else:
    alive_voxels = baking.compute_alive_voxels(
        state,
        dataset,
        config,
        grid_config,
        alpha_threshold,
        weight_threshold,
        use_alpha_culling,
        subsampling_factor,
        use_only_first_image,
    )
  if save_alive_voxels_to_disk:
    utils.save_np(alive_voxels, alive_voxels_path)

  print(
      '{:.1f}% voxels are occupied.'.format(
          100 * alive_voxels.sum() / alive_voxels.size
      )
  )
  if config.stop_after_compute_alive_voxels:
    quit()

  # Compute alive data blocks.
  data_block_size = config.data_block_size
  print('data_block_size:', data_block_size)
  if use_sparse_grid:
    if use_triplanes:
      downsampling_ratio_3d_grid = int(
          config.triplane_resolution / config.sparse_grid_resolution
      )
      alive_voxels_3d_grid = skimage.measure.block_reduce(
          alive_voxels,
          (
              downsampling_ratio_3d_grid,
              downsampling_ratio_3d_grid,
              downsampling_ratio_3d_grid,
          ),
          np.max,
      )
    else:
      alive_voxels_3d_grid = alive_voxels
    alive_macroblocks = skimage.measure.block_reduce(
        alive_voxels_3d_grid,
        (data_block_size, data_block_size, data_block_size),
        np.max,
    )
    num_alive_macroblocks = alive_macroblocks.sum()
    print('Sparse grid:')
    print(
        '{} out of {} ({:.1f}%) macroblocks are occupied.'.format(
            num_alive_macroblocks,
            alive_macroblocks.size,
            100 * num_alive_macroblocks / alive_macroblocks.size,
        )
    )

  # Bake sparse grid.
  batch_size_in_blocks = 2**8
  if use_sparse_grid:
    sparse_grid_features_1d, sparse_grid_density_1d = baking.bake_sparse_grid(
        state,
        grid_config,
        alive_macroblocks,
        data_block_size,
        batch_size_in_blocks,
    )

  # Reshape sparse grid into 3D volume atlas texture (for OpenGL) and
  # compute the indirection grid.
  if use_sparse_grid:
    sparse_grid_features, sparse_grid_density, sparse_grid_block_indices = (
        baking.reshape_into_3d_atlas_and_compute_indirection_grid(
            sparse_grid_features_1d,
            sparse_grid_density_1d,
            data_block_size,
            alive_macroblocks,
        )
    )

  # Bake triplanes.
  batch_size_triplane_baking = 2**20
  if use_triplanes:
    planes_features, planes_density = baking.bake_triplane(
        state,
        config,
        grid_config,
        batch_size_triplane_baking,
        alive_voxels,
    )

  # Compute VRAM consumption.
  vram_consumption = {}
  if use_sparse_grid:
    vram_consumption = {
        'sparse_3d_grid': math.as_mib(sparse_grid_features) + math.as_mib(
            sparse_grid_density
        ),
        'indirection_grid': math.as_mib(sparse_grid_block_indices),
    }
  if use_triplanes:
    # Assume that all three planes have the same size.
    vram_consumption['triplanes'] = 3 * (
        math.as_mib(planes_features[0]) + math.as_mib(planes_density[0])
    )
  vram_consumption['total'] = sum(vram_consumption.values())
  print('VRAM consumption:')
  for k in vram_consumption:
    print(f'{k}: {vram_consumption[k]:.2f} MiB')
  utils.save_json(vram_consumption, os.path.join(log_dir, 'vram.json'))

  # Load Deferred MLP.
  deferred_mlp_vars = state.params['params']['DeferredMLP_0']

  # Quantizing.
  planes_features, planes_density, sparse_grid_features, sparse_grid_density = (
      quantize.map_quantize(
          planes_features,
          planes_density,
          sparse_grid_features,
          sparse_grid_density,
      )
  )

  # Compute occupancy grids based on downsampling factors.
  occupancy_grid_factors = [8, 16, 32, 64, 128]
  occupancy_grids = []
  for occupancy_grid_factor in occupancy_grid_factors:
    occupancy_grid = skimage.measure.block_reduce(
        alive_voxels,
        (occupancy_grid_factor, occupancy_grid_factor, occupancy_grid_factor),
        np.max,
    )
    occupancy_grids.append((occupancy_grid_factor, occupancy_grid))

  # Specify finest occupancy grid used during rendering.
  occupancy_grid_index = 1  # downsampling factor = 16.
  occupancy_grid_factor = occupancy_grids[occupancy_grid_index][0]
  occupancy_grid = occupancy_grids[occupancy_grid_index][1]

  # Render from the baked representation.
  baked_render_dir = path.join(log_dir, 'baked_render')
  if not utils.isdir(baked_render_dir):
    utils.makedirs(baked_render_dir)

  max_steps = 2**11
  baked_render_chunk_size = 2**10
  bg_intensity = 0.5

  # Move representation to GPU.
  gc.collect()

  def to_gpu(*x):
    fn = lambda x: jnp.array(x) if x is not None else None
    return jax.tree_map(fn, x)

  (
      planes_features_gpu,
      planes_density_gpu,
      sparse_grid_features_gpu,
      sparse_grid_density_gpu,
      sparse_grid_block_indices_gpu,
      occupancy_grid_gpu,
  ) = to_gpu(
      planes_features,
      planes_density,
      sparse_grid_features,
      sparse_grid_density,
      sparse_grid_block_indices,
      occupancy_grid,
  )
  gc.collect()

  metric_harness = image.MetricHarness()
  all_psnr = []
  all_ssim = []
  for render_index in list(range(len(test_dataset.camtoworlds))):
    # Generate rays.
    rays = datasets.cam_to_rays(test_dataset, render_index)
    gc.collect()

    # Render.
    rendering = baked_render.render_rays(
        baked_render_chunk_size,
        rays.origins,
        rays.directions,
        sparse_grid_features_gpu,
        sparse_grid_density_gpu,
        sparse_grid_block_indices_gpu,
        planes_features_gpu,
        planes_density_gpu,
        deferred_mlp_vars,
        occupancy_grid_gpu,
        config,
        grid_config,
        max_steps,
        bg_intensity,
        occupancy_grid_factor,
    )

    # Record metrics.
    img_dir = path.join(baked_render_dir, f'rgb.test.{render_index:03d}.png')
    utils.save_img_u8(rendering['rgb'], img_dir)

    cpu = jax.local_devices(backend='cpu')[0]
    metric = metric_harness(
        jax.device_put(rendering['rgb'], cpu),
        jax.device_put(test_dataset.images[render_index], cpu),
    )
    all_psnr.append(metric['psnr'])
    all_ssim.append(metric['ssim'])
    print(
        f'image#: {render_index} PSNR: {metric["psnr"]}, SSIM: {metric["ssim"]}'
    )

  del (
      planes_features_gpu,
      planes_density_gpu,
      sparse_grid_features_gpu,
      sparse_grid_density_gpu,
      sparse_grid_block_indices_gpu,
      occupancy_grid_gpu,
  )
  gc.collect()

  # Write metrics as .json file to disk.
  all_metrics = {
      'psnr_avg': np.mean(all_psnr),
      'ssim_avg': np.mean(all_ssim),
      'psnr': all_psnr,
      'ssim': all_ssim,
  }
  utils.save_json(
      all_metrics, os.path.join(baked_render_dir, 'metrics.test.json')
  )
  print(
      f'average over all test images: PSNR {all_metrics["psnr_avg"]}, SSIM:'
      f' {all_metrics["ssim_avg"]}'
  )

  # Export scene.
  export.export_scene(
      log_dir,
      sparse_grid_features,
      sparse_grid_density,
      sparse_grid_block_indices,
      planes_features,
      planes_density,
      deferred_mlp_vars,
      occupancy_grids,
      config,
      grid_config,
      data_block_size,
  )


if __name__ == '__main__':
  with gin.config_scope('train'):
    app.run(main)
