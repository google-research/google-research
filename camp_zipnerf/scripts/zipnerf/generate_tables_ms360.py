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

"""Scrape the multi-scale 360 results and generate a table."""

import glob
import os

import numpy as np
import tensorflow as tf

NUM_SCALES = 4


def scrape_folder(folder, num_iters, metric_names=['psnr', 'ssim']):
  stats = {}
  for i_metric, metric_name in enumerate(metric_names):
    filename = os.path.join(
        folder, 'test_preds', f'metric_{metric_name}_{num_iters}.txt'
    )
    with open(filename) as f:
      v = np.array([float(s) for s in f.readline().split(' ')])

    for i, vi in enumerate(
        tuple(np.mean(np.reshape(v, [NUM_SCALES, -1]), axis=-1))
    ):
      stats[f'{metric_name}_{i}'] = vi

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


def render_table(
    names, data, precisions, rank_order, suffixes=None, hlines=[], colorful=True
):
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
        if colorful:
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
      'bicycle',
      'flowerbed',
      'gardenvase',
      'stump',
      'treehill',
      'fulllivingroom',
      'kitchencounter',
      'kitchenlego',
      'officebonsai',
  ]

  models_meta = {}  # folder : latex_name
  models_meta['~/tmp/zipnerf/ms360'] = 'Our Model', 200000

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

  val_names = []
  for i in range(NUM_SCALES):
    for metric in ['psnr', 'ssim']:
      val_names.append(f'{metric}_{i}')
  val_names.append('num_hours')

  names = [x[0] for x in list(models_meta.values())]
  data = np.array([[s[n] for n in val_names] for s in avg_stats])
  precisions = [2, 3] * NUM_SCALES + [2]
  rank_order = [1, 1] * NUM_SCALES + [
      0
  ]  # +1 = higher is better, -1 = lower is better, 0 = do not color code
  suffixes = ['', ''] * NUM_SCALES + ['']
  print('Average Results:')
  render_table(names, data, precisions, rank_order, suffixes=suffixes)

  print('Per-Scene Results:')

  for metric in ['psnr', 'ssim']:
    print(metric)
    print()
    val_names = []
    for i in range(NUM_SCALES):
      val_names.append(f'{metric}_{i}')

    names = [x[0] for x in list(models_meta.values())]
    data = np.array([
        np.array([[s[n] for n in val_names] for s in scene_stats]).flatten()
        for scene_stats in all_stats
    ])

    precisions = [2 if metric == 'psnr' else 3] * NUM_SCALES * len(scene_names)
    rank_order = [+1] * NUM_SCALES * len(scene_names)
    render_table(names, data, precisions, rank_order, colorful=False)
    print()
