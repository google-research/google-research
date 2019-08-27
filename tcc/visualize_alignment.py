# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Visualize alignment based on nearest neighbor in embedding space."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from absl import app
from absl import flags
from absl import logging

from dtw import dtw
import matplotlib
matplotlib.use('Agg')
from matplotlib.animation import FuncAnimation  # pylint: disable=g-import-not-at-top
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v2 as tf

gfile = tf.io.gfile

EPSILON = 1e-7

flags.DEFINE_string('video_path', None, 'Path to aligned video.')
flags.DEFINE_string('embs_path', None, 'Path to '
                    'embeddings. Can be regex.')
flags.DEFINE_boolean('use_dtw', False, 'Use dynamic time warping.')
flags.DEFINE_integer('reference_video', 0, 'Reference video.')
flags.DEFINE_integer('switch_video', 10, 'Reference video.')
flags.DEFINE_integer('candidate_video', None, 'Target video.')
flags.DEFINE_boolean(
    'normalize_embeddings', False, 'If True, L2 normalizes the embeddings '
    'before aligning.')
flags.DEFINE_boolean(
    'grid_mode', True, 'If False, switches to dynamically '
    'jumping between videos.')
flags.DEFINE_integer('interval', 50, 'Time in ms b/w consecutive frames.')

flags.mark_flag_as_required('video_path')
flags.mark_flag_as_required('embs_path')

FLAGS = flags.FLAGS


def dist_fn(x, y):
  dist = np.sum((x-y)**2)
  return dist


def get_nn(embs, query_emb):
  dist = np.linalg.norm(embs - query_emb, axis=1)
  assert len(dist) == len(embs)
  return np.argmin(dist), np.min(dist)


def unnorm(query_frame):
  min_v = query_frame.min()
  max_v = query_frame.max()
  query_frame = (query_frame - min_v) / (max_v - min_v)
  return query_frame


def align(query_feats, candidate_feats, use_dtw):
  """Align videos based on nearest neighbor or dynamic time warping."""
  if use_dtw:
    _, _, _, path = dtw(query_feats, candidate_feats, dist=dist_fn)
    _, uix = np.unique(path[0], return_index=True)
    nns = path[1][uix]
  else:
    nns = []
    for i in range(len(query_feats)):
      nn_frame_id, _ = get_nn(candidate_feats, query_feats[i])
      nns.append(nn_frame_id)
  return nns


def create_video(embs, frames, video_path, use_dtw, query, candidate, interval):
  """Create aligned videos."""
  # If candiidate is not None implies alignment is being calculated between
  # 2 videos only.
  if candidate is not None:
    fig, ax = plt.subplots(ncols=2, figsize=(10, 10), tight_layout=True)
    nns = align(embs[query], embs[candidate], use_dtw)

    def update(i):
      """Update plot with next frame."""
      logging.info('%s/%s', i, len(embs[query]))
      ax[0].imshow(unnorm(frames[query][i]))
      ax[1].imshow(unnorm(frames[candidate][nns[i]]))
      # Hide grid lines
      ax[0].grid(False)
      ax[1].grid(False)

      # Hide axes ticks
      ax[0].set_xticks([])
      ax[1].set_xticks([])
      ax[0].set_yticks([])
      ax[1].set_yticks([])
      plt.tight_layout()
  else:
    ncols = int(math.sqrt(len(embs)))
    fig, ax = plt.subplots(
        ncols=ncols,
        nrows=ncols,
        figsize=(5 * ncols, 5 * ncols),
        tight_layout=True)
    nns = []
    for candidate in range(len(embs)):
      nns.append(align(embs[query], embs[candidate], use_dtw))
    ims = []

    def init():
      k = 0
      for k in range(ncols):
        for j in range(ncols):
          ims.append(ax[j][k].imshow(
              unnorm(frames[k * ncols + j][nns[k * ncols + j][0]])))
          ax[j][k].grid(False)
          ax[j][k].set_xticks([])
          ax[j][k].set_yticks([])
      return ims

    ims = init()

    def update(i):
      logging.info('%s/%s', i, len(embs[query]))
      for k in range(ncols):
        for j in range(ncols):
          ims[k * ncols + j].set_data(
              unnorm(frames[k * ncols + j][nns[k * ncols + j][i]]))
      plt.tight_layout()
      return ims

  anim = FuncAnimation(
      fig,
      update,
      frames=np.arange(len(embs[query])),
      interval=interval,
      blit=False)
  anim.save(video_path, dpi=80)


def create_dynamic_video(embs, frames, video_path, use_dtw, query, switch_video,
                         interval):
  """Create aligned videos."""
  fig, ax = plt.subplots(ncols=2, figsize=(10, 10), tight_layout=True)

  nns = []
  for candidate in range(len(embs)):
    nns.append(align(embs[query], embs[candidate], use_dtw))

  def update(i):
    """Update plot with next frame."""
    logging.info('%s/%s', i, len(embs[query]))
    candidate = i // switch_video + 1

    ax[0].imshow(unnorm(frames[query][i]))
    ax[1].imshow(unnorm(frames[candidate][nns[candidate][i]]))
    # Hide grid lines
    ax[0].grid(False)
    ax[1].grid(False)

    # Hide axes ticks
    ax[0].set_xticks([])
    ax[1].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_yticks([])
    plt.tight_layout()

  anim = FuncAnimation(
      fig,
      update,
      frames=np.arange(len(embs[query])),
      interval=interval,
      blit=False)
  anim.save(video_path, dpi=80)


def visualize():
  """Visualize alignment."""
  all_files = sorted(gfile.glob(FLAGS.embs_path))
  logging.info('Found files: %s', all_files)

  # Load embeddings and frames.
  embs = []
  frames = []
  for i in range(len(all_files)):
    file_obj = gfile.GFile(all_files[i], 'rb')
    query_dict = np.load(file_obj, allow_pickle=True).item()

    for j in range(len(query_dict['embs'])):
      curr_embs = query_dict['embs'][j]
      if FLAGS.normalize_embeddings:
        curr_embs = [x/(np.linalg.norm(x) + EPSILON) for x in curr_embs]
      embs.append(curr_embs)
      frames.append(query_dict['frames'][j])

  if FLAGS.grid_mode:
    create_video(
        embs,
        frames,
        FLAGS.video_path,
        FLAGS.use_dtw,
        query=FLAGS.reference_video,
        candidate=FLAGS.candidate_video,
        interval=FLAGS.interval)
  else:
    create_dynamic_video(
        embs,
        frames,
        FLAGS.video_path,
        FLAGS.use_dtw,
        query=FLAGS.reference_video,
        switch_video=FLAGS.switch_video,
        interval=FLAGS.interval)


def main(_):
  visualize()


if __name__ == '__main__':
  app.run(main)
