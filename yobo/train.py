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

# pylint: skip-file
"""Training script for mipNeRF360."""

import functools
import gc
import os
import time

from absl import app
from absl import logging
import chex
import flax
from flax.metrics import tensorboard
from flax.training import checkpoints
import gin
import jax
from jax import random
import jax.numpy as jnp
import numpy as np

from google_research.yobo.internal import camera_utils
from google_research.yobo.internal import configs
from google_research.yobo.internal import datasets
from google_research.yobo.internal import image
from google_research.yobo.internal import models
from google_research.yobo.internal import train_utils
from google_research.yobo.internal import utils
from google_research.yobo.internal import vis

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.65'

configs.define_common_flags()
jax.config.parse_flags_with_absl()

TIME_PRECISION = 1000  # Internally represent integer times in milliseconds.




def main(unused_argv):

  config = configs.load_config()

  rng = random.PRNGKey(config.jax_rng_seed)
  # Shift the numpy random seed by host_id() to shuffle data loaded by different
  # hosts.
  np.random.seed(config.np_rng_seed + jax.host_id())

  if config.disable_pmap_and_jit:
    chex.fake_pmap_and_jit().start()

  if config.batch_size % jax.device_count() != 0:
    raise ValueError('Batch size must be divisible by the number of devices.')

  dataset = datasets.load_dataset('train', config.data_dir, config)
  test_dataset = datasets.load_dataset('test', config.data_dir, config)
  test_raybatcher = datasets.RayBatcher(test_dataset)

  np_to_jax = lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x
  cameras = dataset.get_train_cameras(config)
  cameras = jax.tree_util.tree_map(np_to_jax, cameras)

  cameras_replicated = flax.jax_utils.replicate(cameras)

  if config.rawnerf_mode:
    postprocess_fn = test_dataset.metadata['postprocess_fn']
  else:
    postprocess_fn = lambda z, _=None: z

  rng, key = random.split(rng)
  model, state, render_eval_pfn, train_pstep, lr_fn = train_utils.setup_model(
      config, key, dataset=dataset
  )

  def fn(x):
    return x.shape if isinstance(x, jnp.ndarray) else train_utils.tree_len(x)

  param_summary = train_utils.summarize_tree(fn, state.params['params'])
  num_chars = max([len(x) for x in param_summary])
  logging.info('Optimization parameter sizes/counts:')
  for k, v in param_summary.items():
    logging.info('%s %s', k.ljust(num_chars), str(v))

  if config.model_type == configs.ModelType.DEFAULT:
    if dataset.size > model.num_glo_embeddings and model.num_glo_features > 0:
      raise ValueError(
          f'Number of glo embeddings {model.num_glo_embeddings} '
          'must be at least equal to number of train images '
          f'{dataset.size}'
      )

  metric_harness = image.MetricHarness(**config.metric_harness_train_config)

  if not utils.isdir(config.checkpoint_dir):
    utils.makedirs(config.checkpoint_dir)
  state = checkpoints.restore_checkpoint(config.checkpoint_dir, state)
  state = train_utils.restore_partial_checkpoint(
      config, state, prefixes=['CacheModel', 'LightModel']
  )
  # Resume training at the step of the last checkpoint.
  init_step = state.step + 1
  state = flax.jax_utils.replicate(state)

  if jax.host_id() == 0:
    summary_writer = tensorboard.SummaryWriter(config.checkpoint_dir)
    summary_writer.text(
        'gin_config', gin.config.markdown(gin.operative_config_str()), step=0
    )
    if config.rawnerf_mode:
      for name, data in zip(['train', 'test'], [dataset, test_dataset]):
        # Log shutter speed metadata in TensorBoard for debug purposes.
        for key in ['exposure_idx', 'exposure_values', 'unique_shutters']:
          summary_writer.text(f'{name}_{key}', str(data.metadata[key]), 0)

  # Prefetch_buffer_size = 3 x batch_size.
  raybatcher = datasets.RayBatcher(dataset)
  p_raybatcher = flax.jax_utils.prefetch_to_device(raybatcher, 3)
  rng = rng + jax.host_id()  # Make random seed separate across hosts.
  rngs = random.split(rng, jax.local_device_count())  # For pmapping RNG keys.
  gc.disable()  # Disable automatic garbage collection for efficiency.
  total_time = 0
  total_steps = 0
  reset_stats = True
  if config.early_exit_steps is not None:
    num_steps = config.early_exit_steps
  else:
    num_steps = config.max_steps
  
  batch = None

  for step in range(init_step, num_steps + 1):
    with jax.profiler.StepTraceAnnotation('train', step_num=step):
      if batch is None or step % config.grad_accum_steps == 0:
        batch = next(p_raybatcher)

      if reset_stats and (jax.host_id() == 0):
        stats_buffer = []
        train_start_time = time.time()
        reset_stats = False

      learning_rate = lr_fn(step)
      train_frac = jnp.clip((step - 1) / (config.max_steps - 1), 0, 1)

      state, stats, rngs = train_pstep(rngs, state, batch, cameras, train_frac)

      if step % config.gc_every == 0:
        gc.collect()  # Disable automatic garbage collection for efficiency.

      # Log training summaries. This is put behind a host_id check because in
      # multi-host evaluation, all hosts need to run inference even though we
      # only use host 0 to record results.
      if jax.host_id() == 0:
        stats = flax.jax_utils.unreplicate(stats)

        stats_buffer.append(stats)

        if step == init_step or step % config.print_every == 0:
          elapsed_time = time.time() - train_start_time
          steps_per_sec = config.print_every / elapsed_time
          rays_per_sec = config.batch_size * steps_per_sec

          # A robust approximation of training time in case of pre-emption.
          total_time += int(round(TIME_PRECISION * elapsed_time))
          total_steps += config.print_every
          approx_total_time = int(round(step * total_time / total_steps))

          # Transpose and stack stats_buffer along axis 0.
          fs = [
              flax.traverse_util.flatten_dict(s, sep='/') for s in stats_buffer
          ]
          stats_stacked = {
              k: jnp.stack([f[k] for f in fs]) for k in fs[0].keys()
          }

          # Split every statistic that isn't a vector into a set of statistics.
          stats_split = {}
          for k, v in stats_stacked.items():
            if v.ndim not in [1, 2] and v.shape[0] != len(stats_buffer):
              raise ValueError('statistics must be of size [n], or [n, k].')
            if v.ndim == 1:
              stats_split[k] = v
            elif v.ndim == 2:
              # The "ray_" stats are vectors of percentiles, which we would like
              # to log as a single histogram and so shouldn't be broken up.
              if k.startswith('ray_'):
                stats_split[k] = v
              else:
                for i, vi in enumerate(tuple(v.T)):
                  stats_split[f'{k}/{i}'] = vi

          if config.debug_mode:
            # Summarize the entire histogram of each statistic.
            for k, v in stats_split.items():
              summary_writer.histogram('train_' + k, v, step)

          # Take the mean and max of each statistic since the last summary.
          # We don't bother logging the average and max "ray_" stats as they are
          # unlikely to be informative.
          kv = [
              (k, v) for k, v in stats_split.items() if not k.startswith('ray_')
          ]
          avg_stats = {k: jnp.mean(v) for k, v in kv}
          max_stats = {k: jnp.max(v) for k, v in kv}

          summ_fn = lambda s, v: summary_writer.scalar(s, v, step)  # pylint:disable=cell-var-from-loop

          # Summarize the mean and max of each statistic.
          for k, v in avg_stats.items():
            summ_fn(f'train_avg_{k}', v)
          for k, v in max_stats.items():
            summ_fn(f'train_max_{k}', v)

          n = sum([np.prod(np.array(v)) for v in param_summary.values()])
          summ_fn('num_params', n)
          for k, v in param_summary.items():
            summ_fn(f'num_params/{k}', np.prod(np.array(v)))

          summ_fn('train_num_devices', len(jax.local_devices()))

          summ_fn('train_learning_rate', learning_rate)
          summ_fn('train_steps_per_sec', steps_per_sec)
          summ_fn('train_rays_per_sec', rays_per_sec)

          summary_writer.scalar(
              'train_avg_psnr_timed',
              avg_stats['psnr'],
              total_time // TIME_PRECISION,
          )
          summary_writer.scalar(
              'train_avg_psnr_timed_approx',
              avg_stats['psnr'],
              approx_total_time // TIME_PRECISION,
          )


          if config.model_type == configs.ModelType.DEFAULT:
            if dataset.metadata is not None and model.learned_exposure_scaling:
              params = state.params['params']
              scalings = params['exposure_scaling_offsets']['embedding'][0]
              num_shutter_speeds = dataset.metadata['unique_shutters'].shape[0]
              for i_s in range(num_shutter_speeds):
                for j_s, value in enumerate(scalings[i_s]):
                  summary_name = f'exposure/scaling_{i_s}_{j_s}'
                  summary_writer.scalar(summary_name, value, step)

          params = state.params['params']
          for key in params:
            if 'beta' in params[key]:
              summ_fn('_'.join([key, 'beta']), params[key]['beta'][0])

          if config.model_type == configs.ModelType.DEFAULT:
            if model.scheduled_beta:
              for i_level in range(len(model.final_betas)):
                beta = model.get_scheduled_beta(i_level, train_frac)
                summ_fn('beta_{}'.format(i_level), beta)

          precision = int(np.ceil(np.log10(config.max_steps))) + 1
          avg_loss = avg_stats['loss']
          avg_psnr = avg_stats['psnr']
          # Grab each "losses_{x}" field and print it as "x[:4]".
          # pylint:disable=g-complex-comprehension
          str_losses = [
              (
                  k[7:11],
                  (f'{v:0.5f}' if v >= 1e-4 and v < 10 else f'{v:0.1e}'),
              )
              for k, v in avg_stats.items()
              if k.startswith('losses/')
          ]
          msg = (
              f'%{precision}d/%d: loss=%0.5f, psnr=%6.3f, lr=%0.2e | '
              + ', '.join([f'{k}={s}' for k, s in str_losses])
              + ', %0.0f r/s'
          )
          logging.info(
              msg,
              step,
              config.max_steps,
              avg_loss,
              avg_psnr,
              learning_rate,
              rays_per_sec,
          )

          # Reset everything we are tracking between summarizations.
          reset_stats = True

        if (config.visualize_every > 0) and (
            step == 1 or step % config.visualize_every == 0
        ):
          vis_start_time = time.time()

          # Log histogram statistics for all trainable model parameters.
          params = flax.jax_utils.unreplicate(state.params['params'])
          params_flat = flax.traverse_util.flatten_dict(params)
          for name_tuple, param in params_flat.items():
            ps = jnp.percentile(param.flatten(), jnp.linspace(0, 100, 101))
            summary_writer.histogram('/'.join(name_tuple), ps, step)

          # Visualize some model parameters.
          for mlp_name in params:
            # If we're running an NGP model, summarize the histogram of features
            # at each scale and visualize x/y/z center-slices of each grid.
            ngp_key = mlp_name[: mlp_name.find('MLP')] + 'HashEncoding_0'
            if ngp_key in params[mlp_name]:
              ngp = params[mlp_name][ngp_key]
              for tensorname, tensor in ngp.items():
                # Plot a histogram of `tensor` and write it as an image.
                fig_array = image.render_histogram(
                    jax.device_get(tensor).flatten(), bins=128, log=True
                )
                tag = '/'.join(['param', mlp_name, ngp_key, tensorname])
                summary_writer.image(tag + '_loghist', fig_array, step)

                if tensorname.startswith('grid_'):
                  tag = '/'.join(['param', mlp_name, ngp_key, tensorname])
                  for d in range(tensor.shape[-1]):
                    x_slice = tensor[tensor.shape[0] // 2, :, :, d]
                    y_slice = tensor[:, tensor.shape[1] // 2, :, d]
                    z_slice = tensor[:, :, tensor.shape[2] // 2, d]
                    vis_fn = vis.colorize
                    summary_writer.image(tag + f'/x{d}', vis_fn(x_slice), step)
                    summary_writer.image(tag + f'/y{d}', vis_fn(y_slice), step)
                    summary_writer.image(tag + f'/z{d}', vis_fn(z_slice), step)

            # Visualize the learned vignette map, and its normalized log.
            ngp_key = 'VignetteWeights'
            if ngp_key in params[mlp_name]:
              coords = jnp.stack(
                  jnp.meshgrid(*[jnp.linspace(-0.5, 0.5, 64)] * 2), axis=-1
              )
              weights = params[mlp_name][ngp_key]
              vignette = image.compute_vignette(coords, weights)
              summary_writer.histogram(
                  'train_vignette', vignette.flatten(), step
              )
              tag = '/'.join(['param', mlp_name, ngp_key])
              summary_writer.image(tag, vignette, step)
              normalize = lambda x: (x - jnp.min(x)) / (jnp.max(x) - jnp.min(x))
              summary_writer.image(
                  tag + '_normalized', normalize(vignette), step
              )
          logging.info(
              'Model visualized in %0.3fs',
              time.time() - vis_start_time,
          )

      if step == 1 or step % config.checkpoint_every == 0:
        checkpoints.save_checkpoint_multiprocess(
            config.checkpoint_dir,
            jax.device_get(flax.jax_utils.unreplicate(state)),
            int(step),
            keep=config.checkpoint_keep,
        )

      # Test-set evaluation.
      if (
          config.train_render_every > 0
          and step % config.train_render_every == 0
      ):
        # We reuse the same random number generator from the optimization step
        # here on purpose so that the visualization matches what happened in
        # training.
        eval_start_time = time.time()
        eval_variables = state.params  # Do not unreplicate
        test_case = next(test_raybatcher)
        rendering = models.render_image(
            functools.partial(
                render_eval_pfn,
                eval_variables,
                train_frac,
                cameras_replicated,
            ),
            rays_or_pixels=test_case.rays,
            rng=rngs[0],
            config=config,
            return_all_levels=True,
        )

        # Log eval summaries on host 0.
        if jax.host_id() == 0:
          eval_time = time.time() - eval_start_time
          num_rays = np.prod(test_case.rays.near.shape[:-1])
          rays_per_sec = num_rays / eval_time
          summary_writer.scalar('test_rays_per_sec', rays_per_sec, step)
          logging.info(
              'Eval %d: %0.3fs., %0.0f rays/sec', step, eval_time, rays_per_sec
          )

          metric_start_time = time.time()
          metric = metric_harness(
              postprocess_fn(rendering['rgb']), postprocess_fn(test_case.rgb)
          )
          logging.info(
              'Metrics computed in %0.3fs', time.time() - metric_start_time
          )
          for name, val in metric.items():
            logging.info('%s = %.4f', name, val)
            summary_writer.scalar('train_metrics/' + name, val, step)

          residual = postprocess_fn(rendering['rgb']) - postprocess_fn(
              test_case.rgb
          )
          summary_writer.image(
              'test_residual', np.clip(residual + 0.5, 0, 1), step
          )
          residual_hist = image.render_histogram(
              np.array(residual).reshape([-1, 3]),
              bins=32,
              range=(-1, 1),
              log=True,
              color=('r', 'g', 'b'),
          )
          summary_writer.image('test_residual_hist', residual_hist, step)

          if config.vis_decimate > 1:
            d = config.vis_decimate
            decimate_fn = lambda x, d=d: None if x is None else x[::d, ::d]
          else:
            decimate_fn = lambda x: x
          rendering = jax.tree_util.tree_map(decimate_fn, rendering)
          test_case = jax.tree_util.tree_map(decimate_fn, test_case)
          vis_start_time = time.time()
          vis_suite = vis.visualize_suite(rendering)
          logging.info('Visualized in %0.3f', time.time() - vis_start_time)
          if config.rawnerf_mode:
            # Unprocess raw output.
            vis_suite['color_raw'] = rendering['rgb']
            # Autoexposed colors.
            vis_suite['color_auto'] = postprocess_fn(rendering['rgb'], None)
            summary_writer.image(
                'test_true_auto', postprocess_fn(test_case.rgb, None), step
            )
            # Exposure sweep colors.
            exposures = test_dataset.metadata['exposure_levels']
            for p, x in list(exposures.items()):
              vis_suite[f'color/{p}'] = postprocess_fn(rendering['rgb'], x)
              summary_writer.image(
                  f'test_true_color/{p}', postprocess_fn(test_case.rgb, x), step
              )
          summary_writer.image('test_true_color', test_case.rgb, step)
          if config.compute_normal_metrics:
            summary_writer.image(
                'test_true_normals', test_case.normals / 2.0 + 0.5, step
            )
          for k, v in vis_suite.items():
            if isinstance(v, list):
              for ii, vv in enumerate(v):
                summary_writer.image(f'test_output_{k}/{ii}', vv, step)
            else:
              summary_writer.image(f'test_output_{k}', v, step)

  if config.max_steps % config.checkpoint_every != 0:
    checkpoints.save_checkpoint_multiprocess(
        config.checkpoint_dir,
        jax.device_get(flax.jax_utils.unreplicate(state)),
        int(config.max_steps),
        keep=config.checkpoint_keep,
    )

  if jax.host_id() == 0 and config.dinnerf_report_events:


if __name__ == '__main__':
  with gin.config_scope('train'):
    app.run(main)
