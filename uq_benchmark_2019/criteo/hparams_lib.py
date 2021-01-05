# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

# Lint as: python2, python3
"""Library of tuned hparams and functions for converting to ModelOptions."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
from absl import logging

from uq_benchmark_2019.criteo import data_lib
from uq_benchmark_2019.criteo import models_lib

HParamsA = collections.namedtuple(
    'CriteoHparamsA', [
        'batch_size', 'learning_rate', 'dropout_rate',
        'max_hash_buckets', 'num_embed_dims'])
HParamsC = collections.namedtuple(
    'CriteoHparamsC', ['batch_size', 'learning_rate', 'dropout_rate'])


_HPS_DICT_A = dict(
    vanilla=HParamsA(
        1024, 1e-3, dropout_rate=0.1,
        max_hash_buckets=200, num_embed_dims=8),
    ll_dropout=HParamsA(
        1024, 1e-3, dropout_rate=0.1,
        max_hash_buckets=200, num_embed_dims=8),
    ll_svi=HParamsA(
        1024, 1e-3, dropout_rate=0.1,
        max_hash_buckets=200, num_embed_dims=8),
    dropout=HParamsA(
        1024, 1e-3, dropout_rate=0.1,
        max_hash_buckets=200, num_embed_dims=8),
    svi=HParamsA(
        1024, 1e-3, dropout_rate=0.1,
        max_hash_buckets=200, num_embed_dims=8),
)
_HPS_DICT_C = dict(
    vanilla=HParamsC(869, 0.00028763, dropout_rate=0.039751),
    dropout=HParamsC(3911, 0.000577941, dropout_rate=0.05),
    ll_dropout=HParamsC(1991, 0.000267056, dropout_rate=0.05),
    svi=HParamsC(3746, 0.000490527, dropout_rate=0.0611851),
    ll_svi=HParamsC(23145, 0.000485135, dropout_rate=0.0602327),
)

NUM_EMBED_DIMS_TMPL = 'num_embed_dims_%02d'
NUM_HASH_BUCKETS_TMPL = 'num_hash_buckets_%02d'

_TUNED_NUM_EMBED_DIMS = [3, 9, 29, 11, 17, None, 14, 4, None, 12, 19, 24, 29, None, 13, 25, None, 8, 29, None, 22, None, None, 31, None, 29]
_TUNED_NUM_HASH_BUCKETS = [1373, 2148, 4847, 9781, 396, 28, 3591, 2798, 14, 7403, 2511, 5598, 9501, 46, 4753, 4056, 23, 3828, 5856, 12, 4226, 23, 61, 3098, 494, 5087]
_TUNED_LAYER_SIZES = [2572, 1454, 1596]
# pylint: enable=line-too-long


def get_tuned_hparams(method, parameterization):
  if parameterization.upper() == 'A':
    logging.warn('Note: HParams-A are untuned.')
    return _HPS_DICT_A[method]
  elif parameterization.upper() == 'C':
    return _HPS_DICT_C[method]
  raise NotImplementedError('Unsupported parameterization: %s' %
                            parameterization)


def model_opts_from_hparams(hps, method, parameterization, fake_training=False):
  """Returns a ModelOptions instance using given hyperparameters."""

  if parameterization.upper() == 'A':
    num_uniq = [data_lib.get_categorical_num_unique(i)
                for i in data_lib.CAT_FEATURE_INDICES]
    num_hash_buckets = [min(hps.max_hash_buckets, 2 * n) for n in num_uniq]
    num_embed_dims = [None if n < 100 else hps.num_embed_dims
                      for n in num_hash_buckets]
    layer_sizes = [200, 200]
  elif parameterization.upper() == 'B':
    num_hash_buckets, num_embed_dims = [], []
    for idx in data_lib.CAT_FEATURE_INDICES:
      num_uniq_i = data_lib.get_categorical_num_unique(idx)
      num_hash_buckets_i = getattr(hps, NUM_HASH_BUCKETS_TMPL % idx)
      num_embed_dims_i = (None if num_uniq_i < 110 else
                          getattr(hps, NUM_EMBED_DIMS_TMPL % idx))
      num_hash_buckets.append(num_hash_buckets_i)
      num_embed_dims.append(num_embed_dims_i)
    layer_sizes = [hps.layer_size_1, hps.layer_size_2, hps.layer_size_3]
  elif parameterization.upper() == 'C':
    num_hash_buckets = _TUNED_NUM_HASH_BUCKETS
    num_embed_dims = _TUNED_NUM_EMBED_DIMS
    layer_sizes = _TUNED_LAYER_SIZES
  else:
    raise ValueError('Unrecognized parameterization: %s' % parameterization)

  return models_lib.ModelOptions(
      # Modeling parameters.
      method=method,
      layer_sizes=[100, 100] if fake_training else layer_sizes,
      num_hash_buckets=num_hash_buckets,
      num_embed_dims=num_embed_dims,
      dropout_rate=hps.dropout_rate,

      # Training parameters.
      batch_size=32 if fake_training else hps.batch_size,
      learning_rate=hps.learning_rate,
  )
