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

from uq_benchmark_2019.cifar import data_lib
from uq_benchmark_2019.cifar import models_lib


HParams = collections.namedtuple(
    'CifarHparams', ['batch_size', 'init_learning_rate', 'dropout_rate',
                     'init_prior_scale_mean',
                     'init_prior_scale_std', 'std_prior_scale'])

_HPS_VANILLA = HParams(7, 0.000717, 0, None, None, None)
_HPS_DROPOUT = HParams(5, 0.000250, 0.054988, None, None, None)

_HPS_LL_SVI = HParams(
    16, 0.00115285, 0,
    init_prior_scale_mean=-2.73995,
    init_prior_scale_std=-3.61795,
    std_prior_scale=4.85503)

_HPS_SVI = HParams(
    107, 0.001189, 0,
    init_prior_scale_mean=-1.9994,
    init_prior_scale_std=-0.30840,
    std_prior_scale=3.4210)

_HPS_LL_DROPOUT = HParams(16, 0.000313, 0.319811, None, None, None)

HPS_DICT = dict(
    vanilla=_HPS_VANILLA,
    dropout=_HPS_DROPOUT,
    dropout_nofirst=_HPS_DROPOUT,
    svi=_HPS_SVI,
    ll_dropout=_HPS_LL_DROPOUT,
    ll_svi=_HPS_LL_SVI,
    wide_dropout=_HPS_DROPOUT,
)


def model_opts_from_hparams(hps, method, fake_training=False):
  """Returns a ModelOptions instance using given hyperparameters."""
  dropout_rate = hps.dropout_rate if hasattr(hps, 'dropout_rate') else 0
  variational = method in ('svi', 'll_svi')

  model_opts = models_lib.ModelOptions(
      # Modeling params
      method=method,
      resnet_depth=20,
      num_resnet_filters=32 if method == 'wide_dropout' else 16,
      # Data params.
      image_shape=data_lib.CIFAR_SHAPE,
      num_classes=data_lib.CIFAR_NUM_CLASSES,
      examples_per_epoch=data_lib.CIFAR_NUM_TRAIN_EXAMPLES,
      # SGD params
      train_epochs=200,
      batch_size=hps.batch_size,
      dropout_rate=dropout_rate,
      init_learning_rate=hps.init_learning_rate,
      # Variational params
      std_prior_scale=hps.std_prior_scale if variational else None,
      init_prior_scale_mean=hps.init_prior_scale_mean if variational else None,
      init_prior_scale_std=hps.init_prior_scale_std if variational else None,
  )

  if fake_training:
    model_opts.batch_size = 32
    model_opts.examples_per_epoch = 256
    model_opts.train_epochs = 1
  return model_opts
