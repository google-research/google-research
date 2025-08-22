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

"""Evaluation script for mipNeRF360."""

import functools
import gc
from os import path
import sys
import time

from absl import app
from absl import logging
import chex
import flax
from flax.metrics import tensorboard
from flax.training import checkpoints
import gin
from internal import alignment
from internal import camera_utils
from internal import configs
from internal import datasets
from internal import image_io
from internal import image_utils
from internal import models
from internal import ref_utils
from internal import train_utils
from internal import utils
from internal import vis
import jax
from jax import random
import jax.numpy as jnp
import jaxcam
import numpy as np


configs.define_common_flags()
jax.config.parse_flags_with_absl()


def plot_camera_metrics(
    *,
    summary_writer,
    camera_params,
    train_cameras,
    train_cameras_gt,
    config,
    step,
    tag,
):
  """Plots camera statistics to TensorBoard."""
  camera_delta = config.camera_delta_cls()
  optimized_cameras: jaxcam.Camera = camera_delta.apply(
      camera_params, train_cameras
  )
  diffs = camera_utils.compute_camera_metrics(
      train_cameras_gt, optimized_cameras
  )
  reduce_fns = {
      'mean': np.mean,
      'max': np.max,
      'std': np.std,
  }
  for reduce_name, reduce_fn in reduce_fns.items():
    for stat_name, stat in diffs.items():
      summary_writer.scalar(
          f'eval_train_camera_{tag}_{reduce_name}/{stat_name}',
          reduce_fn(np.array(stat)),
          step=step,
      )

  # pylint: disable=cell-var-from-loop
  return [
      jax.tree_util.tree_map(lambda x: float(x[i]), diffs)
      for i in range(len(train_cameras))
  ]
  # pylint: enable=cell-var-from-loop


def main(unused_argv):
  config = configs.load_config(save_config=False)

  train_dataset = datasets.load_dataset('train', config.data_dir, config)
  test_dataset = datasets.load_dataset('test', config.data_dir, config)

  key = random.PRNGKey(20200823)
  model, state, render_eval_pfn, _, _ = train_utils.setup_model(
      config, key, dataset=train_dataset
  )
  if config.rawnerf_mode:
    postprocess_fn = test_dataset.metadata['postprocess_fn']
  else:
    postprocess_fn = lambda z: z

  metric_harness = image_utils.MetricHarness(
      **config.metric_harness_eval_config
  )

  last_step = 0
  out_dir = path.join(
      config.checkpoint_dir,
      'path_renders' if config.render_path else 'test_preds',
  )
  path_fn = lambda x: path.join(out_dir, x)

  if not config.eval_only_once:
    summary_writer = tensorboard.SummaryWriter(
        path.join(config.checkpoint_dir, 'eval')
    )

  jnp_cameras = None
  if config.cast_rays_in_eval_step:
    np_to_jax = lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x
    jnp_cameras = jax.tree_util.tree_map(np_to_jax, test_dataset.cameras)

  jnp_cameras_replicated = flax.jax_utils.replicate(jnp_cameras)

  last_eval_time = time.time()
  while True:
    state = checkpoints.restore_checkpoint(config.checkpoint_dir, state)
    step = int(state.step)
    state_params_replicated = flax.jax_utils.replicate(state.params)

    if step <= last_step:
      if time.time() - last_eval_time > config.eval_checkpoint_wait_timeout_sec:
        raise RuntimeError(
            'Waited for a new checkpoint for'
            f' {config.eval_checkpoint_wait_timeout_sec} seconds, got no new'
            ' checkpoint. This likely means that the training script has died.'
            ' Exiting. If this is expected, increase'
            ' config.eval_checkpoint_wait_timeout_sec.'
        )
      logging.info(
          'Checkpoint step %d <= last step %d, sleeping.', step, last_step
      )
      time.sleep(10)
      continue

    last_eval_time = time.time()

    logging.info('Evaluating checkpoint at step %d.', step)
    if config.eval_save_output and (not utils.isdir(out_dir)):
      utils.makedirs(out_dir)

    num_eval = min(test_dataset.size, config.eval_dataset_limit)
    key = random.PRNGKey(0 if config.deterministic_showcase else step)
    perm = random.permutation(key, num_eval)
    showcase_indices = np.sort(perm[: config.num_showcase_images])

    metrics = []
    metrics_aligned_optimized = []
    metrics_aligned_procrustes = []
    metrics_cameras = []
    metrics_cameras_procrustes = []
    showcases = []
    render_times = []

    state_params_replicated = flax.jax_utils.replicate(state.params)

    compute_aligned_metric = (
        config.optimize_test_cameras and step == config.max_steps
    )

    procrustes_cameras = None
    if config.compute_procrustes_metric and config.optimize_cameras:
      test_image_sizes = np.array(
          [(x.shape[1], x.shape[0]) for x in test_dataset.images]
      )
      test_jax_cameras = jax.vmap(test_dataset.jax_camera_from_tuple_fn)(
          test_dataset.cameras, test_image_sizes
      )
      train_jax_cameras = train_dataset.get_train_cameras(
          config, return_jax_cameras=True
      )
      train_jax_cameras_gt = train_dataset.jax_cameras
      camera_params = state.params['camera_params']
      camera_delta = config.camera_delta_cls()
      train_jax_cameras_opt = camera_delta.apply(
          camera_params, train_jax_cameras
      )
      train_jax_cameras_procrustes, test_jax_cameras_procrustes = (
          alignment.compute_procrusted_aligned_cameras(
              train_jax_cameras_gt=train_jax_cameras_gt,
              train_jax_cameras_opt=train_jax_cameras_opt,
              test_jax_cameras=test_jax_cameras,
          )
      )
      metrics_cameras = plot_camera_metrics(
          summary_writer=summary_writer,
          camera_params=camera_params,
          train_cameras=train_jax_cameras,
          train_cameras_gt=train_jax_cameras_gt,
          config=config,
          step=step,
          tag='error',
      )
      metrics_cameras_procrustes = plot_camera_metrics(
          summary_writer=summary_writer,
          camera_params=camera_params,
          train_cameras=train_jax_cameras,
          train_cameras_gt=train_jax_cameras_procrustes,
          config=config,
          step=step,
          tag='error_procrustes',
      )
      # Convert to tuples.
      procrustes_cameras = jax.vmap(camera_utils.tuple_from_jax_camera)(
          test_jax_cameras_procrustes
      )
      procrustes_cameras = (*procrustes_cameras, *test_dataset.cameras[3:])

      procrustes_cameras_replicated = flax.jax_utils.replicate(
          procrustes_cameras
      )

    raybatcher = datasets.RayBatcher(test_dataset)
    for idx in range(test_dataset.size):
      gc.collect()
      with jax.profiler.StepTraceAnnotation('eval', step_num=idx):
        eval_start_time = time.time()
        batch = next(raybatcher)
        if idx >= num_eval:
          logging.info('Skipping image %d/%d', idx + 1, test_dataset.size)
          continue
        logging.info('Evaluating image %d/%d', idx + 1, test_dataset.size)
        rays = batch.rays
        train_frac = state.step / config.max_steps

        def _render_image(cameras, rays, train_frac):
          return models.render_image(  # pytype: disable=wrong-arg-types  # jnp-array
              functools.partial(
                  render_eval_pfn,
                  state_params_replicated,
                  train_frac,
                  cameras,
              ),
              rays=rays,
              rng=None,
              config=config,
              return_all_levels=True,
          )

        if compute_aligned_metric:
          jnp_camera_optimized = alignment.align_test_camera(
              model, state, idx, test_dataset, config
          )
          jnp_camera_optimized_replicated = flax.jax_utils.replicate(
              jnp_camera_optimized
          )
          rendering_aligned_optimized = _render_image(
              jnp_camera_optimized_replicated, rays, train_frac
          )
          rendering_aligned_optimized = jax.tree_util.tree_map(
              np.asarray, rendering_aligned_optimized
          )

        if procrustes_cameras is not None:
          rendering_aligned_procrustes = _render_image(
              procrustes_cameras_replicated, rays, train_frac
          )
          rendering_aligned_procrustes = jax.tree_util.tree_map(
              np.asarray, rendering_aligned_procrustes
          )

        rendering = _render_image(jnp_cameras_replicated, rays, train_frac)
        rendering = jax.tree_util.tree_map(np.asarray, rendering)
        rays = jax.tree_util.tree_map(np.asarray, rays)

        if jax.host_id() != 0:  # Only record via host 0.
          continue

        render_times.append((time.time() - eval_start_time))
        logging.info('Rendered in %0.3fs', render_times[-1])

        # Cast to 64-bit to ensure high precision for color correction function.
        gt_rgb = np.array(batch.rgb, dtype=np.float64)
        rendering['rgb'] = np.array(rendering['rgb'], dtype=np.float64)
        if compute_aligned_metric:
          rendering['rgb_aligned_optimized'] = np.array(
              rendering_aligned_optimized['rgb'], dtype=np.float64
          )
        if procrustes_cameras is not None:
          rendering['rgb_aligned_procrustes'] = np.array(
              rendering_aligned_procrustes['rgb'], dtype=np.float64
          )

        if not config.eval_only_once and idx in showcase_indices:
          showcase_idx = (
              idx if config.deterministic_showcase else len(showcases)
          )
          showcases.append((showcase_idx, rendering, batch))
        if not config.render_path:
          rgb = postprocess_fn(rendering['rgb'])
          if compute_aligned_metric:
            rgb_aligned_optimized = postprocess_fn(
                rendering['rgb_aligned_optimized']
            )

          if procrustes_cameras is not None:
            rgb_aligned_procrustes = postprocess_fn(
                rendering['rgb_aligned_procrustes']
            )
          rgb_gt = postprocess_fn(gt_rgb)

          if config.eval_quantize_metrics:
            # Ensures that the images written to disk reproduce the metrics.
            rgb = np.round(rgb * 255) / 255

          if config.eval_crop_borders > 0:
            crop_fn = lambda x, c=config.eval_crop_borders: x[c:-c, c:-c]
            rgb = crop_fn(rgb)
            rgb_gt = crop_fn(rgb_gt)
            if compute_aligned_metric:
              rgb_aligned_optimized = crop_fn(rgb_aligned_optimized)
            if 'rgb_aligned_procrustes' in rendering:
              rgb_aligned_procrustes = crop_fn(rgb_aligned_procrustes)

          metric = metric_harness(rgb, rgb_gt)
          if compute_aligned_metric:
            metric_aligned_optimized = metric_harness(
                rgb_aligned_optimized, rgb_gt
            )
          if procrustes_cameras is not None:
            metric_aligned_procrustes = metric_harness(
                rgb_aligned_procrustes, rgb_gt
            )

          if config.compute_disp_metrics:
            for tag in ['mean', 'median']:
              key = f'distance_{tag}'
              if key in rendering:
                disparity = 1 / (1 + rendering[key][-1])
                metric[f'disparity_{tag}_mse'] = float(
                    ((disparity - batch.disps) ** 2).mean()
                )

          if config.compute_normal_metrics:
            weights = rendering['acc'][-1] * batch.alphas
            normalized_normals_gt = ref_utils.l2_normalize(batch.normals)
            for key, val in rendering.items():
              if key.startswith('normals') and val is not None:
                normalized_normals = ref_utils.l2_normalize(val[-1])
                metric[key + '_mae'] = ref_utils.compute_weighted_mae(
                    weights, normalized_normals, normalized_normals_gt
                )

          for m, v in metric.items():
            logging.info('%s = %0.4f', m, v)

          metrics.append(metric)
          if compute_aligned_metric:
            metrics_aligned_optimized.append(metric_aligned_optimized)
          if procrustes_cameras is not None:
            metrics_aligned_procrustes.append(metric_aligned_procrustes)

        if config.eval_save_output and (config.eval_render_interval > 0):
          if (idx % config.eval_render_interval) == 0:
            image_io.save_img_u8(
                postprocess_fn(rendering['rgb']),
                path_fn(f'color_{idx:03d}.png'),
            )
            if compute_aligned_metric:
              image_io.save_img_u8(
                  postprocess_fn(rendering['rgb_aligned_optimized']),
                  path_fn(f'color_aligned_optimized_{idx:03d}.png'),
              )
            if procrustes_cameras is not None:
              image_io.save_img_u8(
                  postprocess_fn(rendering['rgb_aligned_procrustes']),
                  path_fn(f'color_aligned_procrustes_{idx:03d}.png'),
              )

            for key in ['distance_mean', 'distance_median']:
              if key in rendering:
                image_io.save_img_f32(
                    rendering[key][-1], path_fn(f'{key}_{idx:03d}.tiff')
                )

            for key in ['normals']:
              if key in rendering:
                image_io.save_img_u8(
                    rendering[key][-1] / 2.0 + 0.5,
                    path_fn(f'{key}_{idx:03d}.png'),
                )

            if 'acc' in rendering:
              image_io.save_img_f32(
                  rendering['acc'][-1], path_fn(f'acc_{idx:03d}.tiff')
              )

            if batch.masks is not None:
              image_io.save_img_u8(
                  batch.rgb * batch.masks,
                  path_fn(f'masked_input_{idx:03d}.png'),
              )

    if (not config.eval_only_once) and (jax.host_id() == 0):
      summary_writer.scalar(
          'eval_median_render_time', np.median(render_times), step
      )

      def summarize_metrics(metrics, metrics_suffix):
        for name in metrics[0]:
          scores = [m[name] for m in metrics]
          prefix = f'eval_metrics{metrics_suffix}/'
          summary_writer.scalar(prefix + name, np.mean(scores), step)
          summary_writer.histogram(prefix + 'perimage_' + name, scores, step)

      summarize_metrics(metrics, '')
      if compute_aligned_metric:
        summarize_metrics(metrics_aligned_optimized, '_aligned_optimized')
      if procrustes_cameras is not None:
        summarize_metrics(metrics_aligned_procrustes, '_aligned_procrustes')

      if config.multiscale_train_factors is not None:
        factors = [1] + list(config.multiscale_train_factors)
        n_images = len(metrics) // len(factors)
        # Split metrics into chunks of n_images (each downsampling level).
        for i, f in enumerate(factors):
          i0 = i * n_images
          i1 = (i + 1) * n_images
          image_shapes = np.array([z.shape for z in test_dataset.images[i0:i1]])
          if not np.all(image_shapes == image_shapes[0]):
            raise ValueError(
                'Not all image shapes match for downsampling '
                f'factor {f}x in evaluation'
            )
          summarize_metrics(metrics[i0:i1], f'_{f}x')
          if compute_aligned_metric:
            summarize_metrics(
                metrics_aligned_optimized[i0:i1], f'_{f}x_aligned_optimized'
            )
          if procrustes_cameras is not None:
            summarize_metrics(
                metrics_aligned_procrustes[i0:i1], f'_{f}x_aligned_procrustes'
            )

      for i, r, b in showcases:
        if config.vis_decimate > 1:
          d = config.vis_decimate
          decimate_fn = lambda x, d=d: None if x is None else x[::d, ::d]
        else:
          decimate_fn = lambda x: x
        r = jax.tree_util.tree_map(decimate_fn, r)
        b = jax.tree_util.tree_map(decimate_fn, b)
        for k, v in vis.visualize_suite(r).items():
          if isinstance(v, list):
            for ii, vv in enumerate(v):
              summary_writer.image(f'output_{k}_{i}/{ii}', vv, step)
          else:
            summary_writer.image(f'output_{k}_{i}', v, step)
        if b.masks is not None:
          mask_float_array = jax.numpy.asarray(b.masks, dtype=jax.numpy.float32)
          summary_writer.image(f'mask_{i}', mask_float_array, step)
          summary_writer.image(
              f'masked_image_{i}', b.rgb * mask_float_array, step
          )
        if not config.render_path:
          target = postprocess_fn(b.rgb)
          pred = postprocess_fn(r['rgb'])
          if compute_aligned_metric:
            pred_aligned_optimized = postprocess_fn(r['rgb_aligned_optimized'])
          if procrustes_cameras is not None:
            pred_aligned_procrustes = postprocess_fn(
                r['rgb_aligned_procrustes']
            )
          summary_writer.image(f'output_color_{i}', pred, step)
          if compute_aligned_metric:
            summary_writer.image(
                f'output_color_aligned_optimized_{i}',
                pred_aligned_optimized,
                step,
            )
          if procrustes_cameras is not None:
            summary_writer.image(
                f'output_color_aligned_procrustes_{i}',
                pred_aligned_procrustes,
                step,
            )
          summary_writer.image(f'true_color_{i}', target, step)
          residual = pred - target
          summary_writer.image(
              f'output_residual_{i}', np.clip(residual + 0.5, 0, 1), step
          )
          if compute_aligned_metric:
            residual_aligned_optimized = pred_aligned_optimized - target
            summary_writer.image(
                f'output_residual_aligned_{i}',
                np.clip(residual_aligned_optimized + 0.5, 0, 1),
                step,
            )
          if procrustes_cameras is not None:
            residual_aligned_procrustes = pred_aligned_procrustes - target
            summary_writer.image(
                f'output_residual_aligned_{i}',
                np.clip(residual_aligned_procrustes + 0.5, 0, 1),
                step,
            )
          residual_hist = image_utils.render_histogram(
              np.array(residual).reshape([-1, 3]),
              bins=32,
              range=(-1, 1),
              log=True,
              color=('r', 'g', 'b'),
          )
          summary_writer.image(f'output_residual_hist_{i}', residual_hist, step)
          if config.compute_normal_metrics:
            summary_writer.image(
                f'true_normals_{i}', b.normals / 2.0 + 0.5, step
            )

    if (
        config.eval_save_output
        and (not config.render_path)
        and (jax.host_id() == 0)
    ):
      with utils.open_file(path_fn(f'render_times_{step}.txt'), 'w') as f:
        f.write(' '.join([str(r) for r in render_times]))
      for name in metrics[0]:
        with utils.open_file(path_fn(f'metric_{name}_{step}.txt'), 'w') as f:
          f.write(' '.join([str(m[name]) for m in metrics]))
      if compute_aligned_metric:
        for name in metrics_aligned_optimized[0]:
          with utils.open_file(
              path_fn(f'metric_aligned_optimized_{name}_{step}.txt'), 'w'
          ) as f:
            f.write(' '.join([str(m[name]) for m in metrics_aligned_optimized]))
      if procrustes_cameras is not None:
        for name in metrics_aligned_procrustes[0]:
          with utils.open_file(
              path_fn(f'metric_aligned_procrustes_{name}_{step}.txt'), 'w'
          ) as f:
            f.write(
                ' '.join([str(m[name]) for m in metrics_aligned_procrustes])
            )
      if metrics_cameras:
        for name in metrics_cameras[0]:
          with utils.open_file(
              path_fn(f'metric_cameras_{name}_{step}.txt'), 'w'
          ) as f:
            f.write(' '.join([str(m[name]) for m in metrics_cameras]))
      if metrics_cameras_procrustes:
        for name in metrics_cameras_procrustes[0]:
          with utils.open_file(
              path_fn(f'metric_cameras_procrustes_{name}_{step}.txt'), 'w'
          ) as f:
            f.write(
                ' '.join([str(m[name]) for m in metrics_cameras_procrustes])
            )
      if config.eval_save_ray_data:
        for i, r, b in showcases:
          rays = {k: v for k, v in r.items() if 'ray_' in k}
          np.set_printoptions(threshold=sys.maxsize)
          with utils.open_file(path_fn(f'ray_data_{step}_{i}.txt'), 'w') as f:
            f.write(repr(rays))

    # A hack that forces Jax to keep all TPUs alive until every TPU is finished.
    x = jnp.ones([jax.local_device_count()])
    x = jax.device_get(jax.pmap(lambda x: jax.lax.psum(x, 'i'), 'i')(x))
    print(x)

    if config.eval_only_once:
      logging.info('Eval only once enabled, shutting down.')
      break
    if config.early_exit_steps is not None:
      num_steps = config.early_exit_steps
    else:
      num_steps = config.max_steps
    if int(step) >= num_steps:
      logging.info('Termination num steps reached (%d).', num_steps)
      break
    last_step = step


if __name__ == '__main__':
  with gin.config_scope('eval'):
    app.run(main)
