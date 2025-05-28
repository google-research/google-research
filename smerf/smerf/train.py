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

"""Main script for training, baking, evaulation and exporting.

Trains, bakes, evaluates a MERF and finally exports the reconstructed scene
to a format that can be read by the MERF webviewer.
"""
# pylint: disable=logging-fstring-interpolation

from collections import defaultdict  # pylint: disable=g-importing-member
import dataclasses
import functools
import gc
import os
import time

from absl import app
from absl import logging
from camp_zipnerf.internal import configs as teacher_configs
from camp_zipnerf.internal import datasets as teacher_datasets
from camp_zipnerf.internal import models as teacher_models
from camp_zipnerf.internal import train_utils as teacher_train_utils
import flax
from flax.metrics import tensorboard
from flax.training import checkpoints
import gin
import jax
from jax import random
import jax.numpy as jnp
import mediapy as media
import numpy as np
from smerf.internal import baked_render as merf_baked_render
from smerf.internal import baking as merf_baking
from smerf.internal import coord as merf_coord
from smerf.internal import datasets as merf_datasets
from smerf.internal import distill as merf_distill
from smerf.internal import export as merf_export
from smerf.internal import grid_utils as merf_grid_utils
from smerf.internal import image as merf_image
from smerf.internal import models as merf_models
from smerf.internal import train_utils as merf_train_utils
from smerf.internal import utils as merf_utils
from smerf.internal.mock.concurrent import parallel
import tensorflow as tf


def main(unused_argv):
  logging.info('main() has started!')

  # Hide GPUs from TensorFlow. If we don't JAX computations will OOM.
  tf.config.set_visible_devices([], 'GPU')

  # Verify that GPUs are available.
  num_gpu_devices = sum([device.platform == 'gpu' for device in jax.devices()])
  if not num_gpu_devices:
    raise ValueError(
        'The following code will be unreasonably slow unless JAX has access to'
        ' one or more GPUs. Please make sure that JAX has been installed with'
        ' CUDA support.'
    )
  del num_gpu_devices

  dataset = test_dataset = render_dataset = None

  # Shift the numpy random seed by host_id() to shuffle data loaded by different
  # hosts.
  np.random.seed(20201473 + jax.process_index())

  # Initialize configs.
  logging.info('Loading Gin config...')
  merf_config, teacher_config = merf_distill.load_config(save_config=False)

  # Where to write all assets. If baking_checkpoint_dir is specified, use it
  # for writing baked assets such as baked parameters and renders.
  log_dir = merf_distill.log_dir(merf_config)
  baked_log_dir = merf_distill.baked_log_dir(merf_config)

  # Save Gin configs to log directories. Only write to the corresponding
  # directory if some assets will be written there.
  if merf_distill.should_write_prebaked_assets(merf_config):
    merf_distill.save_config(log_dir)
  if merf_distill.should_write_baked_assets(merf_config):
    merf_distill.save_config(baked_log_dir)

  # Everything below can be copied one-to-one from colab.
  log_dir.mkdir(parents=True, exist_ok=True)

  # Load train, test, and render dataset splits.
  logging.info('Loading dataset...')
  dataset = teacher_datasets.load_dataset(
      'train', teacher_config.data_dir, teacher_config
  )
  raybatcher = teacher_datasets.RayBatcher(dataset)

  test_dataset = teacher_datasets.load_dataset(
      'test', teacher_config.data_dir, teacher_config
  )
  render_dataset = teacher_datasets.load_dataset(
      'test',
      teacher_config.data_dir,
      dataclasses.replace(teacher_config, render_path=True),
  )

  # Setup dataset-specific fields in merf_config.
  merf_config = merf_grid_utils.initialize_grid_config(
      merf_config, [dataset, test_dataset]
  )

  # Load cameras
  np_to_jax = lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x
  cameras = dataset.get_train_cameras(teacher_config)
  cameras = jax.tree_util.tree_map(np_to_jax, cameras)
  pcameras = flax.jax_utils.replicate(cameras)

  # Load teacher.
  logging.info('Loading teacher...')
  rng = random.PRNGKey(teacher_config.jax_rng_seed)
  teacher_model, teacher_state, teacher_render_eval_pfn, _, _ = (
      teacher_train_utils.setup_model(teacher_config, rng, dataset=dataset)
  )
  teacher_state = checkpoints.restore_checkpoint(
      teacher_config.checkpoint_dir, teacher_state
  )
  logging.info(f'teacher step: {teacher_state.step}')

  # Initialize MERF model.
  logging.info('Initializing MERF...')
  merf_model, merf_state, _, merf_train_pstep, _ = (
      merf_train_utils.setup_model(
          merf_config, random.PRNGKey(merf_config.model_seed), dataset
      )
  )

  # Restore checkpoint. Free up VRAM mid-restore by moving the state
  # to Host RAM, then putting it back in VRAM.
  merf_state = jax.device_get(merf_state)
  merf_state = checkpoints.restore_checkpoint(log_dir, merf_state)
  merf_state = jax.device_put(merf_state)
  merf_step = int(merf_state.step) // merf_config.gradient_accumulation_steps
  logging.info(f'Loaded checkpoint from {log_dir}, step: {merf_step}')

  # Initialize MERF rendering function.
  merf_render_eval_pfn = merf_distill.create_prender_student(
      teacher_model=teacher_model,
      student_model=merf_model,
      merf_config=merf_config,
      alpha_threshold=merf_baking.final_alpha_threshold(merf_config),
      return_ray_results=True,
  )

  # Replicate teacher state. This operation is left for last as it will
  # significantly increase VRAM usage.
  teacher_pstate = flax.jax_utils.replicate(teacher_state)
  prender_teacher = merf_distill.create_prender_teacher(
      teacher_model, teacher_config
  )

  # Initialize TensorBoard summaries.
  summary_writer = tensorboard.SummaryWriter(merf_config.checkpoint_dir)
  summary_writer.text('gin_config', f'<pre>{gin.config_str()}</pre>', merf_step)

  # Train loop.
  def train(merf_step, merf_state):
    logging.info('Entering train()...')
    if merf_step >= merf_config.max_steps + 1:
      return merf_step, merf_state

    gc.collect()
    p_raybatcher = flax.jax_utils.prefetch_to_device(raybatcher, 3)
    merf_pstate = flax.jax_utils.replicate(merf_state)
    prng = random.split(random.PRNGKey(1234567), jax.local_device_count())
    grid_config = merf_config.grid_config

    # Use first image of test dataset corresponding to a submodel on this host
    # for tracking PSNR progress during training. If image is assigned to this
    # host, then use an arbitrary frame instead.
    val_idx = (
        merf_baking.find_cam_idx_for_submodel(
            test_dataset,
            grid_config['submodels_on_host'],
            merf_config,
            grid_config,
        )
        or 0
    )
    val_rays = merf_datasets.cam_to_rays(test_dataset, val_idx, xnp=jnp)
    val_rays = merf_datasets.preprocess_rays(
        rays=val_rays, mode='test', merf_config=merf_config, dataset=dataset
    )

    warmup_iters = merf_step + 10  # only for iters/sec measurement.
    train_start_time = None

    for i in range(merf_step, merf_config.max_steps + 1):
      if i == warmup_iters:
        train_start_time = time.time()
      merf_step = (
          int(merf_pstate.step[0]) // merf_config.gradient_accumulation_steps
      )
      train_frac = jnp.clip((merf_step - 1) / (merf_config.max_steps - 1), 0, 1)

      ######################################################################
      pstats = None

      with jax.profiler.StepTraceAnnotation('train', step_num=merf_step):
        for _ in range(merf_config.gradient_accumulation_steps):
          pbatch = next(p_raybatcher)

          # Ray preprocessing including ray casting, jittering, exposure
          # preprocessing, submodel assignment, and more.
          pbatch = pbatch.replace(
              rays=merf_datasets.preprocess_rays(
                  rays=pbatch.rays,
                  mode='train',
                  merf_config=merf_config,
                  dataset=dataset,
                  pcameras=pcameras,
                  prng=prng,
              ),
          )

          # Push ray origins forward till first teacher ray interval.
          pbatch = merf_train_utils.pshift_batch_forward(
              prng=prng,
              pbatch=pbatch,
              teacher_pstate=teacher_pstate,
              prender_teacher=prender_teacher,
              config=merf_config,
          )

          # Render teacher.
          teacher_prng = prng if merf_config.distill_teacher_use_rng else None
          pteacher_history = prender_teacher(
              teacher_prng, teacher_pstate, pbatch
          )

          # Update MERF params.
          merf_pstate, pstats, prng = merf_train_pstep(
              prng, merf_pstate, pbatch, pteacher_history, train_frac
          )

      ######################################################################
      stats = flax.jax_utils.unreplicate(pstats)
      for k, v in stats['losses'].items():
        summary_writer.scalar(f'train_losses_{k}', v, merf_step)

      ######################################################################
      if i > warmup_iters:
        iters_per_sec = (i - warmup_iters) / (time.time() - train_start_time)
        summary_writer.scalar('train_iters_per_sec', iters_per_sec, merf_step)
      else:
        iters_per_sec = 'n/a'

      ######################################################################
      if (
          merf_config.print_every > 0
          and i % merf_config.print_every == 0
      ):
        logging.info(f'Iteration: {i}, iters/sec: {iters_per_sec}')
        logging.info(
            ' '.join(
                [
                    f'{k}: {-10*np.log10(v):.3f}'
                    for k, v in stats['losses'].items()
                ]
            )
        )

      ######################################################################
      if (
          merf_config.train_render_every > 0
          and i % merf_config.train_render_every == 0
      ):
        teacher_rendering = teacher_models.render_image(
            functools.partial(
                teacher_render_eval_pfn,
                teacher_pstate.params,
                1.0,
                None,  # No cameras needed
            ),
            rays=val_rays,
            rng=prng[0],
            config=teacher_config,
            return_all_levels=True,
        )
        merf_rendering = merf_models.render_image(
            functools.partial(
                merf_render_eval_pfn,
                teacher_pstate.params,
                merf_pstate.params,
                1.0,
            ),
            rays=val_rays,
            rng=prng[0],
            config=merf_config,
            verbose=False,
            transfer_to_cpu=True,
        )
        val_psnr = merf_image.imgs_to_psnr(
            merf_rendering['rgb'], test_dataset.images[val_idx]
        )
        summary_writer.scalar('train_render_psnr', val_psnr, merf_step)
        summary_writer.image(
            'train_render_gt', test_dataset.images[val_idx], merf_step
        )
        summary_writer.image(
            'train_render_merf', merf_rendering['rgb'], merf_step
        )
        summary_writer.image(
            'train_render_merf_depth',
            merf_image.colorize_depth(merf_rendering['depth']),
            merf_step,
        )
        summary_writer.image(
            'train_render_teacher', teacher_rendering['rgb'], merf_step
        )

      ######################################################################
      if (
          merf_config.checkpoint_every > 0
          and (
              i % merf_config.checkpoint_every == 0
              or i == merf_config.max_steps
          )
      ) and i > 0:
        logging.info(f'Saving checkpoint {i}')
        merf_state = flax.jax_utils.unreplicate(merf_pstate)
        checkpoints.save_checkpoint(
            log_dir, merf_state, i, keep=10, overwrite=True
        )

    merf_state = flax.jax_utils.unreplicate(merf_pstate)
    merf_step = int(merf_state.step) // merf_config.gradient_accumulation_steps
    return merf_step, merf_state

  if merf_config.enable_train:
    merf_step, merf_state = train(merf_step, merf_state)

  ##############################################################################
  # Replicate MERF state across devices.

  # Replicate state once and for all. Doing this operation once reduces churn
  # in VRAM and reduces the chance of OOMs.
  merf_pstate = flax.jax_utils.replicate(merf_state)

  # Delete unreplicated state to avoid merf_state and merf_pstate from going
  # out of sync. Use flax.jax_utils.unreplicate() if you'd like to recover the
  # unreplicated state again.
  del merf_state

  ##############################################################################
  # Create directory for renderings
  orig_render_dir = log_dir / 'orig_render'
  orig_render_dir.mkdir(parents=True, exist_ok=True)

  def render_entire_dataset(dataset_type):
    assert dataset_type in ['train', 'test'], f'{dataset_type=}'
    logging.info(f'rendering {dataset_type=}...')
    gc.collect()
    metric_harness = merf_image.MetricHarness()
    all_metrics = defaultdict(lambda: [])
    train_frac = 1.0
    render_dataset = {'train': dataset, 'test': test_dataset}[dataset_type]
    grid_config = merf_config.grid_config
    threadpool = merf_utils.AsyncThreadPool()

    def build_img_path(prefix, img_idx):
      return orig_render_dir / f'{prefix}.{dataset_type}.{img_idx:03d}.png'

    def load_img(path):
      return merf_utils.load_img(os.fspath(path)) / 255.

    teacher_render_fn = functools.partial(
        teacher_render_eval_pfn,
        teacher_pstate.params,
        train_frac,
        None,  # No cameras needed
    )
    student_render_fn = functools.partial(
        merf_render_eval_pfn,
        teacher_pstate.params,
        merf_pstate.params,
        train_frac,
    )

    for img_idx in range(len(render_dataset.images)):
      # When training one submodel per host, only render a view if this
      # submodel is responsible for it.
      if grid_config['submodel_idx_override'] is not None:
        model_sm_idx = grid_config['submodel_idx_override']
        cam_sm_idx = merf_baking.sm_idx_for_camera(
            render_dataset, img_idx, merf_config, grid_config
        )
        if cam_sm_idx != model_sm_idx:
          logging.info(
              f'Skipping {img_idx=}. It belongs to submodel={cam_sm_idx}.'
          )
          continue

      try:
        # pylint: disable=cell-var-from-loop
        gt_rgb, teacher_rgb, merf_rgb, merf_depth = parallel.ParallelMap(
            lambda prefix: load_img(build_img_path(prefix, img_idx)),
            ['gt', 'teacher', 'rgb', 'depth'],
        )
      # pylint: enable=cell-var-from-loop
      except Exception:  # pylint: disable=broad-exception-caught
        # Construct camera rays. Ray casting happens here.
        rays = merf_datasets.cam_to_rays(render_dataset, img_idx, xnp=jnp)
        rays = merf_datasets.preprocess_rays(
            rays=rays,
            mode='test',
            merf_config=merf_config,
            dataset=render_dataset,
        )

        # Render the teacher.
        teacher_rendering = teacher_models.render_image(
            teacher_render_fn,
            rays=rays,
            rng=random.PRNGKey(42),
            config=teacher_config,
            return_all_levels=False,
        )

        # Render the student.
        merf_rendering = merf_models.render_image(
            student_render_fn,
            rays,
            random.PRNGKey(42),
            merf_config,
            verbose=False,
            transfer_to_cpu=True,
        )

        # Move arrays to host memory. This should free up some VRAM.
        teacher_rendering = jax.device_get(teacher_rendering)
        merf_rendering = jax.device_get(merf_rendering)

        gt_rgb = render_dataset.images[img_idx]
        teacher_rgb = teacher_rendering['rgb']
        merf_rgb = merf_rendering['rgb']
        merf_depth = merf_image.colorize_depth(merf_rendering['depth'])

      # Save images to disk
      for prefix, img in [
          ('gt', gt_rgb),
          ('teacher', teacher_rgb),
          ('rgb', merf_rgb),
          ('depth', merf_depth),
      ]:
        img_dir = build_img_path(prefix, img_idx)
        threadpool.submit(merf_utils.save_img_u8, img, os.fspath(img_dir))

      # Compute metrics.
      for prefix, img_gt in [('gt', gt_rgb), ('teacher', teacher_rgb)]:
        metric = metric_harness(merf_rgb, img_gt)
        for metric_name in sorted(metric.keys()):
          all_metrics[f'{prefix}.{metric_name}'].append(metric[metric_name])
        logging.info(f'{prefix} image#={img_idx} {metric}')

    # Compute per-frame averages of all metrics.
    for key, values in list(all_metrics.items()):
      all_metrics[f'{key}.avg'] = np.mean(values)

    # Write metrics as .json file to disk.
    threadpool.submit(
        merf_utils.save_json,
        all_metrics,
        os.fspath(orig_render_dir / f'metrics.{dataset_type}.json'),
    )
    logging.info(
        f'average over all {dataset_type} images: %s',
        {
            f'{k}={all_metrics[k]}'
            for k in sorted(all_metrics.keys())
            if k.endswith('.avg')
        },
    )

    # Write TensorBoard metrics for all images.
    for k, v in all_metrics.items():
      if k.endswith('.avg'):
        summary_writer.scalar(f'{dataset_type}.{k}', v, merf_step)

    # Finish all async operations.
    threadpool.flush()

  if merf_config.enable_eval:
    render_entire_dataset('test')

  ##############################################################################
  # Render render_path video
  def render_path_video():
    logging.info('Rendering video...')
    gc.collect()
    threadpool = merf_utils.AsyncThreadPool()

    # Create directory for rendered frames.
    video_dir = orig_render_dir / 'render_path'
    video_dir.mkdir(parents=True, exist_ok=True)

    # Render frames.
    num_frames = len(render_dataset.camtoworlds)
    frame_idxs = range(0, num_frames, merf_config.render_path_video_every)
    rendered_frames = []
    for i in frame_idxs:
      # Generate camera rays
      rays = merf_datasets.cam_to_rays(render_dataset, i)
      rays = merf_datasets.preprocess_rays(
          rays, mode='test', merf_config=merf_config, dataset=dataset
      )

      # Skip this camera if there's no submodel to render it.
      param_idxs = merf_coord.sm_idxs_to_params_idxs(
          rays.sm_idxs, merf_config, merf_config.grid_config
      )
      if np.any(param_idxs == merf_grid_utils.INVALID_IDX):
        logging.info(
            f'Skipping frame {i} of {num_frames}. One or more rays could not be'
            ' mapped to a submodel owned by this host.'
        )
        continue

      # Try to read a rendered frame from disk. If this fails for any reason,
      # render it instead.
      logging.info(f'Rendering render_path frame {i} of {num_frames}...')
      rgb_frame_path = video_dir / f'rgb.render_path.{i:05d}.png'
      depth_frame_path = video_dir / f'depth.render_path.{i:05d}.png'
      teacher_frame_path = video_dir / f'teacher.render_path.{i:05d}.png'
      try:
        rgb, depth, teacher_rgb = parallel.ParallelMap(
            lambda p: merf_utils.load_img(p) / 255.,
            [rgb_frame_path, depth_frame_path, teacher_frame_path]
        )
      except Exception:  # pylint: disable=broad-exception-caught
        train_frac = 1.0

        # Render teacher
        teacher_rendering = teacher_models.render_image(
            functools.partial(
                teacher_render_eval_pfn,
                teacher_pstate.params,
                1.0,
                None,  # No cameras needed
            ),
            rays=rays,
            rng=random.PRNGKey(42),
            config=teacher_config,
            return_all_levels=False,
        )
        teacher_rgb = jax.device_get(teacher_rendering['rgb'])

        # Render student
        student_rendering = merf_models.render_image(
            functools.partial(
                merf_render_eval_pfn,
                teacher_pstate.params,
                merf_pstate.params,
                train_frac,
            ),
            rays,
            random.PRNGKey(42),
            merf_config,
            verbose=False,
            transfer_to_cpu=True,
        )
        rgb = jax.device_get(student_rendering['rgb'])
        depth = jax.device_get(
            merf_image.colorize_depth(student_rendering['depth'])
        )

        # Write images to disk.
        for img, path in [
            (teacher_rgb, teacher_frame_path),
            (rgb, rgb_frame_path),
            (depth, depth_frame_path),
        ]:
          threadpool.submit(merf_utils.save_img_u8, img, os.fspath(path))

      rendered_frames.append(
          {'rgb': rgb, 'depth': depth, 'teacher': teacher_rgb}
      )

    # Construct mp4 files.
    for key in ['rgb', 'depth', 'teacher']:
      video_path = orig_render_dir / f'{key}.render_path.mp4'
      fps = 60. / merf_config.render_path_video_every
      frames = [frame[key] for frame in rendered_frames]
      media.write_video(video_path, frames, fps=fps)

    # Finish all asynchronous operations.
    threadpool.flush()

  if merf_config.enable_video or merf_config.enable_render_path_video:
    render_path_video()

  ##############################################################################
  # Baking

  def bake_and_export_submodel(sm_idx):
    # Construct function for rendering MERF.
    merf_render_fn = functools.partial(
        merf_render_eval_pfn, teacher_pstate.params, merf_pstate.params, 1.0
    )

    # Create output directory.
    sm_temp_dir = merf_baking.construct_sm_temp_dir(sm_idx, merf_config)
    sm_temp_dir.mkdir(parents=True, exist_ok=True)

    sm_export_dir = baked_log_dir / 'baked' / f'sm_{sm_idx:03d}'
    sm_export_dir.mkdir(parents=True, exist_ok=True)

    # Construct baked representation.
    (
        sparse_grid_features,
        sparse_grid_density,
        sparse_grid_block_indices,
        planes_features,
        planes_density,
        deferred_mlp_vars,
        _,  # occupancy_grids
        packed_occupancy_grids,
        distance_grids,
    ) = merf_baking.bake_submodel(
        sm_idx=sm_idx,
        merf_model=merf_model,
        merf_pstate=merf_pstate,
        dataset=dataset,
        merf_render_fn=merf_render_fn,
        merf_config=merf_config,
        sm_temp_dir=sm_temp_dir,
    )

    # Export scene.
    if merf_config.enable_baked_export:
      logging.info('Exporting baked model assets...')
      merf_export.export_scene(
          baked_dir=sm_export_dir,
          sparse_grid_features=sparse_grid_features,
          sparse_grid_density=sparse_grid_density,
          sparse_grid_block_indices=sparse_grid_block_indices,
          planes_features=planes_features,
          planes_density=planes_density,
          deferred_mlp_vars=deferred_mlp_vars,
          packed_occupancy_grids=packed_occupancy_grids,
          distance_grids=distance_grids,
          sm_idx=sm_idx,
          config=merf_config,
      )

      # Export test-set poses for performance benchmarking.
      merf_export.export_test_cameras(
          baked_dir=sm_export_dir,
          test_dataset=test_dataset,
          sm_idx=sm_idx,
          config=merf_config,
          grid_config=merf_config.grid_config,
      )

  if merf_config.enable_baking:
    grid_config = merf_config.grid_config
    num_submodels = grid_config['num_submodels']
    num_local_submodels = len(grid_config['submodels_on_host'])

    for i, sm_idx in enumerate(grid_config['submodels_on_host']):
      logging.info(
          f'Baking submodel #{sm_idx}/{num_submodels} (#{i} of'
          f' #{num_local_submodels})'
      )
      bake_and_export_submodel(sm_idx)

  ##############################################################################
  # Baked rendering

  def render_and_eval_baked_submodel(sm_idx):
    # Load baked params.
    sm_temp_dir = merf_baking.construct_sm_temp_dir(sm_idx, merf_config)
    sm_baked_params = merf_baking.load_baked_params(sm_temp_dir)

    # Prepare output directory.
    sm_render_dir = merf_baked_render.construct_sm_render_dir(
        baked_log_dir, sm_idx
    )
    sm_render_dir.mkdir(parents=True, exist_ok=True)

    # Unpack baked state
    (
        sparse_grid_features,
        sparse_grid_density,
        sparse_grid_block_indices,
        planes_features,
        planes_density,
        deferred_mlp_vars,
        occupancy_grids,
        _,  # packed_occupancy_grids
        _,  # distance_grids
    ) = sm_baked_params

    # Determine batch size. A batch size of 2^14 pixels works well for a
    # spatial resolution of 2048^3 on 8x V100s.
    target_resolution = merf_config.grid_config['resolution_to_use']
    batch_size = (2 ** 15) * (2048 / target_resolution)
    batch_size = int(batch_size) // merf_config.gradient_accumulation_steps

    # Start rendering.
    occupancy_grid_factor, occupancy_grid = occupancy_grids[1]
    merf_baked_render.render_dataset(
        config=merf_config,
        planes_features=planes_features,
        planes_density=planes_density,
        sparse_grid_features=sparse_grid_features,
        sparse_grid_density=sparse_grid_density,
        sparse_grid_block_indices=sparse_grid_block_indices,
        deferred_mlp_vars=deferred_mlp_vars,
        occupancy_grid_factor=occupancy_grid_factor,
        occupancy_grid=occupancy_grid,
        sm_idx=sm_idx,
        dataset=test_dataset,
        baked_render_dir=sm_render_dir,
        batch_size=batch_size,
    )

  if merf_config.enable_baked_eval:
    grid_config = merf_config.grid_config
    num_submodels = grid_config['num_submodels']
    num_local_submodels = len(grid_config['submodels_on_host'])

    # Render each submodel.
    for i, sm_idx in enumerate(grid_config['submodels_on_host']):
      logging.info(
          'Rendering baked representation for submodel'
          f' #{sm_idx}/{num_submodels} (#{i} of {num_local_submodels})'
      )
      render_and_eval_baked_submodel(sm_idx)

    # Merge photos collections; compute merged metrics.
    baked_metrics = merf_baked_render.merge_all_baked_renders(
        grid_config['submodels_on_host'],
        log_dir=log_dir,
        baked_log_dir=baked_log_dir,
    )

    # Update TensorBoard with baked metrics.
    for key, value in baked_metrics.items():
      if key.endswith('.avg'):
        summary_writer.scalar(f'test.{key}', value, merf_step)

  ##############################################################################
  # Baked render path video

  def render_path_video_baked(sm_idx):
    # Load baked params.
    sm_temp_dir = merf_baking.construct_sm_temp_dir(sm_idx, merf_config)
    sm_baked_params = merf_baking.load_baked_params(sm_temp_dir)

    # Prepare output directory.
    sm_render_dir = merf_baked_render.construct_sm_render_dir(
        baked_log_dir, sm_idx
    )
    sm_render_dir.mkdir(parents=True, exist_ok=True)

    # Unpack baked state
    (
        sparse_grid_features,
        sparse_grid_density,
        sparse_grid_block_indices,
        planes_features,
        planes_density,
        deferred_mlp_vars,
        occupancy_grids,
        _,  # packed_occupancy_grids
        _,  # distance_grids
    ) = sm_baked_params

    # Determine batch size. A batch size of 2^14 pixels works well for a
    # spatial resolution of 2048^3 on 8x V100s.
    target_resolution = merf_config.grid_config['resolution_to_use']
    batch_size = (2 ** 15) * (2048 / target_resolution)
    batch_size = int(batch_size) // merf_config.gradient_accumulation_steps

    # Start rendering.
    occupancy_grid_factor, occupancy_grid = occupancy_grids[1]
    merf_baked_render.render_path(
        config=merf_config,
        planes_features=planes_features,
        planes_density=planes_density,
        sparse_grid_features=sparse_grid_features,
        sparse_grid_density=sparse_grid_density,
        sparse_grid_block_indices=sparse_grid_block_indices,
        deferred_mlp_vars=deferred_mlp_vars,
        occupancy_grid_factor=occupancy_grid_factor,
        occupancy_grid=occupancy_grid,
        sm_idx=sm_idx,
        dataset=render_dataset,
        baked_render_dir=sm_render_dir,
        batch_size=batch_size,
    )

  if merf_config.enable_baked_video:
    grid_config = merf_config.grid_config
    num_submodels = grid_config['num_submodels']
    num_local_submodels = len(grid_config['submodels_on_host'])

    # Render each submodel.
    for i, sm_idx in enumerate(grid_config['submodels_on_host']):
      logging.info(
          'Rendering ellipse path of baked representation for submodel'
          f' #{sm_idx}/{num_submodels} (#{i} of {num_local_submodels})'
      )
      render_path_video_baked(sm_idx)

    # Merge all results together.
    merf_baked_render.merge_all_baked_path_renders(
        sm_idxs=grid_config['submodels_on_host'],
        baked_log_dir=baked_log_dir,
        config=merf_config,
    )

  ##############################################################################
  # Finalize TensorBoard summaries.
  summary_writer.flush()
  summary_writer.close()

  logging.info('All done.')


if __name__ == '__main__':
  # Limit the amount of VRAM that XLA is allowed to use. The excess space is
  # used by non-XLA operations, such as NCCL all-gather operations.
  os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.80'

  teacher_configs.define_common_flags()
  jax.config.parse_flags_with_absl()

  with gin.config_scope('train'):
    app.run(main)
