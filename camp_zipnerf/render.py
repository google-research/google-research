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

"""Render script for mipNeRF360."""

import dataclasses
import functools
import os
import time

from absl import app
from absl import flags
from absl import logging
import flax
from flax.training import checkpoints
import gin
from internal import configs
from internal import datasets
from internal import image_io
from internal import models
from internal import train_utils
from internal import utils
from internal import videos_utils
import jax
from jax import random
import jax.numpy as jnp
import numpy as np

configs.define_common_flags()
jax.config.parse_flags_with_absl()


def render_pipeline(config):
  """Renders all eligible test frames and writes them to disk."""
  dataset = datasets.load_dataset('test', config.data_dir, config)

  key = random.PRNGKey(20200823)
  _, state, render_eval_pfn, _, _ = train_utils.setup_model(
      config, key, dataset=dataset
  )

  if config.rawnerf_mode:
    postprocess_fn = dataset.metadata['postprocess_fn']
  else:
    postprocess_fn = lambda z: z

  state = checkpoints.restore_checkpoint(config.checkpoint_dir, state)
  step = int(state.step)
  logging.info('Rendering checkpoint at step %d.', step)

  out_name = 'path_renders' if config.render_path else 'test_preds'
  out_name = f'{out_name}_step_{step}'
  base_dir = config.render_dir
  if base_dir is None:
    base_dir = os.path.join(config.checkpoint_dir, 'render')
  out_dir = os.path.join(base_dir, out_name)
  if not utils.isdir(out_dir):
    utils.makedirs(out_dir)

  if jax.host_id() == 0:
    # Save out numpy array of the render poses.
    posefile = os.path.join(base_dir, 'render_poses.npy')
    with utils.open_file(posefile, 'wb') as fp:
      np.save(fp, np.array(dataset.camtoworlds)[:, :3, :4])

  path_fn = lambda x: os.path.join(out_dir, x)

  # Ensure sufficient zero-padding of image indices in output filenames.
  zpad = max(3, len(str(dataset.size - 1)))
  idx_to_str = lambda idx: str(idx).zfill(zpad)

  replicated_state = flax.jax_utils.replicate(state.params)

  jnp_cameras = None
  if config.cast_rays_in_eval_step:
    np_to_jax = lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x
    jnp_cameras = jax.tree_util.tree_map(np_to_jax, dataset.cameras)

  jnp_cameras = flax.jax_utils.replicate(jnp_cameras)

  for idx in range(dataset.size):
    rays = dataset.generate_ray_batch(idx).rays
    idx_str = idx_to_str(idx)
    logging.info('Evaluating image %d/%d', idx + 1, dataset.size)
    eval_start_time = time.time()
    train_frac = 1.0
    rendering = models.render_image(  # pytype: disable=wrong-arg-types  # jnp-array
        functools.partial(
            render_eval_pfn,
            replicated_state,
            train_frac,
            jnp_cameras,
        ),
        rays,
        None,
        config,
    )
    logging.info('Rendered in %0.3fs', time.time() - eval_start_time)

    if jax.host_id() != 0:  # Only record via host 0.
      continue

    rendering['rgb'] = postprocess_fn(rendering['rgb'])

    image_io.save_img_u8(rendering['rgb'], path_fn(f'color_{idx_str}.png'))

    if not config.render_rgb_only:
      if 'normals' in rendering:
        image_io.save_img_u8(
            rendering['normals'] / 2.0 + 0.5, path_fn(f'normals_{idx_str}.png')
        )
      if 'normals_rectified' in rendering:
        image_io.save_img_u8(
            rendering['normals_rectified'] / 2.0 + 0.5,
            path_fn(f'normals_rectified_{idx_str}.png'),
        )
      image_io.save_img_f32(
          rendering['distance_mean'], path_fn(f'distance_mean_{idx_str}.tiff')
      )
      image_io.save_img_f32(
          rendering['distance_median'],
          path_fn(f'distance_median_{idx_str}.tiff'),
      )
      image_io.save_img_u8(rendering['acc'], path_fn(f'acc_{idx_str}.png'))

  logging.info('Creating videos.')
  videos_utils.create_videos(config, base_dir, out_dir, out_name, dataset.size)


def render_config(config):
  """Renders the pipeline given a config."""
  logging.info(
      'Rendering config:\n%s',
      gin.operative_config_str(),
  )

  base_render_dir = None
  if not config.render_spline_keyframes_choices:
    # Default codepath. Render one of the following,
    #   1) config.render_spline_keyframes is defined: use spline path
    #   2) config.render_spline_keyframes isn't defined: use spiral path
    base_render_dir = config.render_dir  # For dashboard reporting
    render_pipeline(config)

  else:  # config.render_spline_keyframes_choices is not None
    if config.render_spline_keyframes:
      raise ValueError(
          'Both Config.render_spline_keyframes and '
          'Config.render_spline_keyframes_choices have been set. Please use '
          'only one of the two and try again.'
      )

    # Render once per value in render_spline_keyframes_choices.
    base_render_dir = config.render_dir or os.path.join(
        config.checkpoint_dir, 'render'
    )
    render_spline_keyframes_choices = (
        config.render_spline_keyframes_choices.split(',')
    )
    logging.info('Found %d spline paths', len(render_spline_keyframes_choices))
    for render_spline_keyframes in render_spline_keyframes_choices:
      logging.info('Rendering spline path: %s', render_spline_keyframes)
      render_spline_name = os.path.basename(render_spline_keyframes)
      render_dir = os.path.join(base_render_dir, render_spline_name)
      new_config = dataclasses.replace(
          config,
          render_spline_keyframes=render_spline_keyframes,
          render_dir=render_dir,
          render_spline_keyframes_choices=None,
      )
      render_pipeline(new_config)


def main(unused_argv):
  gin.parse_config_files_and_bindings(
      flags.FLAGS.gin_configs,
      flags.FLAGS.gin_bindings,
      skip_unknown=True,
      finalize_config=False,
  )
  config = configs.Config()
  render_config(config)


if __name__ == '__main__':
  with gin.config_scope('eval'):  # Use the same scope as eval.py
    app.run(main)
