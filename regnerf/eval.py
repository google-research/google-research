# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Evaluation script for RegNeRF."""
import functools
from os import path
import time

from absl import app
import flax
from flax.metrics import tensorboard
from flax.training import checkpoints
from internal import configs, datasets, math, models, utils, vis  # pylint: disable=g-multiple-import
import jax
from jax import random
import numpy as np
from skimage.metrics import structural_similarity
import tensorflow as tf

CENSUS_EPSILON = 1 / 256  # Guard against ground-truth quantization.

configs.define_common_flags()
jax.config.parse_flags_with_absl()


def main(unused_argv):

  tf.config.experimental.set_visible_devices([], 'GPU')
  tf.config.experimental.set_visible_devices([], 'TPU')

  config = configs.load_config(save_config=False)

  dataset = datasets.load_dataset('test', config.data_dir, config)
  model, init_variables = models.construct_mipnerf(
      random.PRNGKey(20200823),
      dataset.peek()['rays'],
      config)
  optimizer = flax.optim.Adam(config.lr_init).create(init_variables)
  state = utils.TrainState(optimizer=optimizer)
  del optimizer, init_variables

  # Rendering is forced to be deterministic even if training was randomized, as
  # this eliminates 'speckle' artifacts.
  def render_eval_fn(variables, _, rays):
    return jax.lax.all_gather(
        model.apply(
            variables,
            None,  # Deterministic.
            rays,
            resample_padding=config.resample_padding_final,
            compute_extras=True), axis_name='batch')

  # pmap over only the data input.
  render_eval_pfn = jax.pmap(
      render_eval_fn,
      in_axes=(None, None, 0),
      donate_argnums=2,
      axis_name='batch',
  )

  def ssim_fn(x, y):
    return structural_similarity(x, y, multichannel=True)

  census_fn = jax.jit(
      functools.partial(math.compute_census_err, epsilon=CENSUS_EPSILON))

  print('WARNING: LPIPS calculation not supported. NaN values used instead.')
  if config.eval_disable_lpips:
    lpips_fn = lambda x, y: np.nan
  else:
    lpips_fn = lambda x, y: np.nan

  last_step = 0
  out_dir = path.join(config.checkpoint_dir,
                      'path_renders' if config.render_path else 'test_preds')
  path_fn = lambda x: path.join(out_dir, x)

  if not config.eval_only_once:
    summary_writer = tensorboard.SummaryWriter(
        path.join(config.checkpoint_dir, 'eval'))
  while True:
    # Fix for loading pre-trained models.
    try:
      state = checkpoints.restore_checkpoint(config.checkpoint_dir, state)
    except:  # pylint: disable=bare-except
      print('Using pre-trained model.')
      state_dict = checkpoints.restore_checkpoint(config.checkpoint_dir, None)
      for i in [9, 17]:
        del state_dict['optimizer']['target']['params']['MLP_0'][f'Dense_{i}']
      state_dict['optimizer']['target']['params']['MLP_0'][
          'Dense_9'] = state_dict['optimizer']['target']['params']['MLP_0'][
              'Dense_18']
      state_dict['optimizer']['target']['params']['MLP_0'][
          'Dense_10'] = state_dict['optimizer']['target']['params']['MLP_0'][
              'Dense_19']
      state_dict['optimizer']['target']['params']['MLP_0'][
          'Dense_11'] = state_dict['optimizer']['target']['params']['MLP_0'][
              'Dense_20']
      del state_dict['optimizerd']
      state = flax.serialization.from_state_dict(state, state_dict)

    step = int(state.optimizer.state.step)
    if step <= last_step:
      print(f'Checkpoint step {step} <= last step {last_step}, sleeping.')
      time.sleep(10)
      continue
    print(f'Evaluating checkpoint at step {step}.')
    if config.eval_save_output and (not utils.isdir(out_dir)):
      utils.makedirs(out_dir)

    key = random.PRNGKey(0 if config.deterministic_showcase else step)
    perm = random.permutation(key, dataset.size)
    showcase_indices = np.sort(perm[:config.num_showcase_images])

    metrics = []
    showcases = []
    for idx in range(dataset.size):
      print(f'Evaluating image {idx+1}/{dataset.size}')
      eval_start_time = time.time()
      batch = next(dataset)
      rendering = models.render_image(
          functools.partial(render_eval_pfn, state.optimizer.target),
          batch['rays'],
          None,
          config)
      print(f'Rendered in {(time.time() - eval_start_time):0.3f}s')

      if jax.host_id() != 0:  # Only record via host 0.
        continue
      if not config.eval_only_once and idx in showcase_indices:
        showcase_idx = idx if config.deterministic_showcase else len(showcases)
        showcases.append((showcase_idx, rendering, batch))
      if not config.render_path:
        metric = {}
        metric['psnr'] = float(
            math.mse_to_psnr(((rendering['rgb'] - batch['rgb'])**2).mean()))
        metric['ssim'] = float(ssim_fn(rendering['rgb'], batch['rgb']))
        metric['lpips'] = float(lpips_fn(rendering['rgb'], batch['rgb']))
        metric['avg_err'] = float(
            math.compute_avg_error(
                psnr=metric['psnr'],
                ssim=metric['ssim'],
                lpips=metric['lpips'],
            ))
        metric['census_err'] = float(census_fn(rendering['rgb'], batch['rgb']))

        if config.compute_disp_metrics:
          disp = 1 / (1 + rendering['distance_mean'])
          metric['disp_mse'] = float(((disp - batch['disps'])**2).mean())

        if config.compute_normal_metrics:
          one_eps = 1 - np.finfo(np.float32).eps
          metric['normal_mae'] = float(
              np.arccos(
                  np.clip(
                      np.sum(batch['normals'] * rendering['normals'], axis=-1),
                      -one_eps, one_eps)).mean())

        if config.dataset_loader == 'dtu':
          rgb = batch['rgb']
          rgb_hat = rendering['rgb']
          mask = batch['mask']
          mask_bin = (mask == 1.)

          rgb_fg = rgb * mask + (1 - mask)
          rgb_hat_fg = rgb_hat * mask + (1 - mask)

          metric['psnr_masked'] = float(
              math.mse_to_psnr(((rgb - rgb_hat)[mask_bin]**2).mean()))
          metric['ssim_masked'] = float(ssim_fn(rgb_hat_fg, rgb_fg))
          metric['lpips_masked'] = float(lpips_fn(rgb_hat_fg, rgb_fg))
          metric['avg_err_masked'] = float(
              math.compute_avg_error(
                  psnr=metric['psnr_masked'],
                  ssim=metric['ssim_masked'],
                  lpips=metric['lpips_masked'],
              ))

        for m, v in metric.items():
          print(f'{m:10s} = {v:.4f}')
        metrics.append(metric)

      if config.eval_save_output and (config.eval_render_interval > 0):
        if (idx % config.eval_render_interval) == 0:
          utils.save_img_u8(rendering['rgb'], path_fn(f'color_{idx:03d}.png'))
          utils.save_img_u8(rendering['normals'] / 2. + 0.5,
                            path_fn(f'normals_{idx:03d}.png'))
          utils.save_img_f32(rendering['distance_mean'],
                             path_fn(f'distance_mean_{idx:03d}.tiff'))
          utils.save_img_f32(rendering['distance_median'],
                             path_fn(f'distance_median_{idx:03d}.tiff'))
          utils.save_img_f32(rendering['acc'], path_fn(f'acc_{idx:03d}.tiff'))

    if (not config.eval_only_once) and (jax.host_id() == 0):
      for name in list(metrics[0].keys()):
        summary_writer.scalar(name, np.mean([m[name] for m in metrics]), step)
      for i, r, b in showcases:
        for k, v in vis.visualize_suite(r, b['rays'], config).items():
          summary_writer.image(f'pred_{k}_{i}', v, step)
        if not config.render_path:
          summary_writer.image(f'target_{i}', b['rgb'], step)
    if (config.eval_save_output and (not config.render_path) and
        (jax.host_id() == 0)):
      for name in list(metrics[0].keys()):
        with utils.open_file(path_fn(f'metric_{name}_{step}.txt'), 'w') as f:
          f.write(' '.join([str(m[name]) for m in metrics]))
    if config.eval_only_once:
      break
    if int(step) >= config.max_steps:
      break
    last_step = step


if __name__ == '__main__':
  app.run(main)
