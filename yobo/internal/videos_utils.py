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
"""Utils for operating with videos."""

import os
import tempfile

from absl import flags
from absl import logging
from etils import epath
from google_research.yobo.internal import configs
from google_research.yobo.internal import math
from google_research.yobo.internal import utils
from matplotlib import cm
import mediapy as media
import numpy as np

_FILE_EXTENSION_TO_CODEC = {'mp4': 'h264', 'gif': 'gif', 'webm': 'vp9'}
_IMAGE_FILE_TYPE_TO_EXTENSION = {
    'color': 'png',
    'normals': 'png',
    'normals_rectified': 'png',
    'acc': 'png',
    'distance_mean': 'tiff',
    'distance_median': 'tiff',
    'semantic': 'png',
}


def create_videos(
    config,
    base_dir,
    out_dir,
    out_name,
    num_frames,
):
  """Creates videos out of the images saved to disk.

  After the function is called, the base_dir will contain the rendered mp4
  video.

  Args:
    config: Loaded gin config.
    base_dir: Base directory for rendered pngs, mp4 files, etc.
    out_dir: Directory with all rendered frames.
    out_name: Base name for rendered video prefix.
    num_frames: Number of all rendered frames.
  """
  names = [n for n in config.checkpoint_dir.split('/') if n]
  # Last two parts of checkpoint path are experiment name and scene name.
  if 'is_xm_sweep' in flags.FLAGS and flags.FLAGS.is_xm_sweep:
    # Just use `out_name` for sweeps since experiment names can be super long.
    video_prefix = out_name
  else:
    exp_name, scene_name = names[-2:]
    video_prefix = f'{scene_name}_{exp_name}_{out_name}'

  zpad = max(3, len(str(num_frames - 1)))
  idx_to_str = lambda idx: str(idx).zfill(zpad)

  def render_dist_curve_fn(z):
    return math.power_ladder(z, **config.render_dist_vis_params)

  with tempfile.TemporaryDirectory() as tmp_dir:
    # Copy all images files locally.
    logging.info('Copying data locally')
    CopyDir(out_dir, tmp_dir, overwrite=True)
    video_dir = os.path.join(tmp_dir, 'videos')
    utils.makedirs(video_dir)

    # Load one example frame to get image shape and depth range.
    if config.render_rgb_only:
      img_file = os.path.join(tmp_dir, f'color_{idx_to_str(0)}.png')
      shape = utils.load_img(img_file).shape
    else:
      depth_file = os.path.join(tmp_dir, f'distance_mean_{idx_to_str(0)}.tiff')
      depth_frame = utils.load_img(depth_file)
      shape = depth_frame.shape
      if config.render_dist_adaptive:
        p = config.render_dist_percentile
        distance_limits = np.percentile(depth_frame.flatten(), [p, 100 - p])
      else:
        distance_limits = config.near, config.far
      lo, hi = [render_dist_curve_fn(x) for x in distance_limits]
    logging.info('Video shape is %s', str(shape[:2]))

    for video_ext in config.render_video_exts:
      if video_ext not in _FILE_EXTENSION_TO_CODEC:
        raise ValueError(
            f"Invalid video format: '{video_ext}'. "
            "Must be either 'mp4', 'webm' or 'gif'"
        )
      else:
        video_codec = _FILE_EXTENSION_TO_CODEC[video_ext]

      video_kwargs = {
          'shape': shape[:2],
          'codec': video_codec,
          'fps': config.render_video_fps,
          'crf': config.render_video_crf,
      }

      keys_to_render = [
          'color',
          'normals',
          'normals_rectified',
          'acc',
          'distance_mean',
          'distance_median',
      ]
      if config.semantic_dir:
        keys_to_render.append('semantic')

      for k in keys_to_render:
        looped_suffix = 'looped_' if config.render_looped_videos else ''
        video_file = os.path.join(
            video_dir, f'{video_prefix}_{looped_suffix}{k}.{video_ext}'
        )
        input_format = 'gray' if k == 'acc' else 'rgb'
        file_ext = _IMAGE_FILE_TYPE_TO_EXTENSION[k]
        file0 = os.path.join(tmp_dir, f'{k}_{idx_to_str(0)}.{file_ext}')
        if not utils.file_exists(file0):
          logging.info('Images missing for tag %s', k)
          continue
        logging.info('Making video %s...', video_file)
        with media.VideoWriter(
            video_file, **video_kwargs, input_format=input_format
        ) as writer:
          indices = list(range(num_frames))
          if config.render_looped_videos:
            indices += reversed(indices)
          for idx in indices:
            img_file = os.path.join(
                tmp_dir, f'{k}_{idx_to_str(idx)}.{file_ext}'
            )
            if not utils.file_exists(img_file):
              raise ValueError(f'Image file {img_file} does not exist.')
            img = utils.load_img(img_file)
            if k in [
                'acc',
                'color',
                'normals',
                'normals_rectified',
                'semantic',
            ]:
              img = img / 255.0
            elif k.startswith('distance'):
              img = render_dist_curve_fn(img)
              img = np.clip((img - lo) / (hi - lo), 0, 1)
              img = cm.get_cmap('turbo')(img)[Ellipsis, :3]

            frame = (np.clip(np.nan_to_num(img), 0.0, 1.0) * 255.0).astype(
                np.uint8
            )
            writer.add_image(frame)

    # Copy videos back to server.
    logging.info('Copying video files back to server')
    CopyDir(video_dir, base_dir, overwrite=True)
