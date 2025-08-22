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

"""Scrape the blender results and generate a table."""

import glob
import os

import numpy as np
import tensorflow as tf


def scrape_folder(folder, num_iters, metric_names=('psnr', 'ssim')):
  """Scrape a folder of results and pull out metrics and timings."""
  stats = {}
  for metric_name in metric_names:
    filename = os.path.join(
        folder, 'test_preds', f'metric_{metric_name}_{num_iters}.txt'
    )
    with open(filename) as f:
      v = np.array([float(s) for s in f.readline().split(' ')])
    stats[metric_name] = np.mean(v)

  grab_tags = ['train_steps_per_sec', 'eval_median_render_time']
  grabbed_tags = {k: [] for k in grab_tags}
  for pattern in ['events*']:  # , 'eval/events*']:
    for event_file in glob.glob(os.path.join(folder, pattern)):
      for event in tf.compat.v1.train.summary_iterator(event_file):
        value = event.summary.value
        if len(value) > 0:
          tag = event.summary.value[0].tag
          if tag in grab_tags:
            grabbed_tags[tag].append(
                np.array(tf.make_ndarray(event.summary.value[0].tensor)).item()
            )

  if grabbed_tags['train_steps_per_sec']:
    steps_per_sec = np.percentile(
        np.array(grabbed_tags['train_steps_per_sec']), 95
    )
    stats['num_hours'] = (num_iters / steps_per_sec) / (60**2)
  else:
    stats['num_hours'] = np.nan

  return stats


def render_table(names, data, precisions, rank_order, suffixes=None, hlines=[]):
  """Render a table of results into latex."""

  def rankify(x, order):
    assert len(x.shape) == 1
    if order == 0:
      return np.full_like(x, 1e5, dtype=np.int32)
    u = np.sort(np.unique(x))
    if order == 1:
      u = u[::-1]
    r = np.zeros_like(x, dtype=np.int32)
    for ui, uu in enumerate(u):
      mask = x == uu
      r[mask] = ui
    return np.int32(r)

  tags = [
      r'   \cellcolor{red}',
      r'\cellcolor{orange}',
      r'\cellcolor{yellow}',
      r'                  ',
  ]

  max_len = max([len(v) for v in list(names)])
  names_padded = [v + ' ' * (max_len - len(v)) for v in names]

  data_quant = np.round(
      (data * 10.0 ** (np.array(precisions)[None, :]))
  ) / 10.0 ** (np.array(precisions)[None, :])
  if suffixes is None:
    suffixes = [''] * len(precisions)

  tagranks = []
  for d in range(data_quant.shape[1]):
    tagranks.append(
        np.clip(rankify(data_quant[:, d], rank_order[d]), 0, len(tags) - 1)
    )
  tagranks = np.stack(tagranks, -1)

  for i_row in range(len(names)):
    line = ''
    if i_row in hlines:
      line += '\\hline\n'
    line += names_padded[i_row]
    for d in range(data_quant.shape[1]):
      line += ' & '
      if rank_order[d] != 0 and not np.isnan(data[i_row, d]):
        line += tags[tagranks[i_row, d]]
      if np.isnan(data[i_row, d]):
        line += ' - '
      else:
        assert precisions[d] >= 0
        line += ('{:' + f'0.{precisions[d]}f' + '}').format(
            data_quant[i_row, d]
        ) + suffixes[d]
    if i_row < (len(names) - 1):
      line += ' \\\\'
    print(line)


if __name__ == '__main__':
  scene_names = [
      'chair',
      'drums',
      'ficus',
      'hotdog',
      'lego',
      'materials',
      'mic',
      'ship',
  ]

  models_meta = {}  # folder : latex_name
  models_meta['~/tmp/zipnerf/blender'] = 'Zip-NeRF', 200000

  all_stats = []
  avg_stats = []
  for model_path, (_, num_iters) in models_meta.items():
    scene_stats = []
    for scene_name in scene_names:
      folder = os.path.expanduser(os.path.join(model_path, scene_name))
      stats = scrape_folder(folder, num_iters)
      print(model_path, scene_name, stats)
      scene_stats.append(stats)
    avg_stats.append({
        k: type(scene_stats[0][k])(np.mean([s[k] for s in scene_stats]))
        for k in scene_stats[0].keys()
    })
    all_stats.append(scene_stats)
    print(model_path, avg_stats[-1])

  names = [x[0] for x in list(models_meta.values())]

  precisions = [2, 3]
  rank_orders = [1, 1]

  for i_metric, metric in enumerate(['psnr', 'ssim']):
    print(metric)
    precision = precisions[i_metric]
    rank_order = rank_orders[i_metric]

    print(
        ' & '
        + ' & '.join(['\\textit{' + s + '}' for s in scene_names + ['avg']])
        + ' \\\\\\hline'
    )
    data = np.array([
        np.array([s[metric] for s in scene_stats]) for scene_stats in all_stats
    ])
    data = np.concatenate(
        [data, np.array([a[metric] for a in avg_stats])[:, None]], axis=1
    )  # Add averages
    n = len(scene_names) + 1
    render_table(
        names, data, [precision] * n, [rank_order] * n, hlines=[len(names) - 1]
    )
