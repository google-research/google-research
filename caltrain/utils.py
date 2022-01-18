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

"""Utility functions and definitions."""
import enum
import matplotlib
import numpy as np
from scipy.special import expit  # pylint: disable=no-name-in-module
from sklearn.utils.extmath import softmax

import caltrain as caltrain

matplotlib.use('Agg')
font = {'size': 26}
matplotlib.rc('font', **font)


def export_legend(legend, filename=None, expand=None):
  fig = legend.figure
  fig.canvas.draw()
  bbox = legend.get_window_extent()
  if expand:
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
  bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
  fig.savefig(filename, dpi='figure', bbox_inches=bbox)


def get_hash_key(config):
  """Compute hash key given simulation config."""
  dataset = config['dataset']
  assert dataset in caltrain.TRUE_DATASETS
  a = config['a']
  b = config['b']
  alpha = config['alpha']
  beta = config['beta']
  d = config['d']
  split = config['split']
  num_samples = config['num_samples']
  calibration_method = config['calibration_method']
  ce_type = config['ce_type']
  num_bins = config['num_bins']
  bin_method = config['bin_method']
  norm = config['norm']
  num_reps = config['num_reps']

  if dataset in ['polynomial', 'flip_polynomial']:
    hash_key = (
        f'{dataset}(a={a},b={b},d={d})_{split}_{num_samples}_{calibration_method}'
        f'_{ce_type}_bins={num_bins}_{bin_method}_norm={norm}_reps={num_reps}')
  elif dataset in [
      'logistic_beta', 'logistic_log_odds', 'two_param_polynomial',
      'two_param_flip_polynomial', 'logistic_two_param_flip_polynomial'
  ]:
    hash_key = (f'{dataset}(a={a},b={b},alpha={alpha},beta={beta})_{split}'
                f'_{num_samples}_{calibration_method}_{ce_type}_bins={num_bins}'
                f'_{bin_method}_norm={norm}_reps={num_reps}')
  elif dataset == 'logistic':
    hash_key = (f'{dataset}(a={a},b={b})_{split}_{num_samples}_'
                f'{calibration_method}_{ce_type}_bins={num_bins}_{bin_method}'
                f'_norm={norm}_reps={num_reps}')
  else:
    raise NotImplementedError
  return hash_key


def to_softmax(logits):
  num_classes = logits.shape[1]
  if num_classes == 1:
    scores = expit(logits)
  else:
    scores = softmax(logits)
  return scores


class Enum(enum.Enum):

  @classmethod
  def items(cls):
    return ((x.name, x.value) for x in cls)
