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
from uq_benchmark_2019.mnist import data_lib
from uq_benchmark_2019.mnist import models_lib

_PREDICTIONS_PER_EXAMPLE = 300
_TRAIN_EPOCHS = 20

HParams = collections.namedtuple(
    'MnistHparams', ['batch_size', 'learning_rate', 'dropout_rate'])

_HPS_DICT_MLP = dict(
    vanilla=HParams(64, 0.000463823361691, 0.230510371879),
    ll_dropout=HParams(2031, 0.00506288988719, 0.180932012085),
    ll_svi=HParams(1411, 0.00273346709332, 0.134455710557),
    dropout=HParams(2, 0.000114126279919, 0.12086438944),
    svi=HParams(2048, 0.00342926092994, 4.18893329913e-09),
)

_HPS_DICT_LENET = dict(
    vanilla=HParams(1620, 0.00101964166073, 0.285357549021),
    ll_dropout=HParams(2046, 0.00612961880078, 0.383790834726),
    ll_svi=HParams(2016, 0.00234115479416, 0.310549422253),
    dropout=HParams(1136, 0.00502635774141, 0.3033871646),
    svi=HParams(2041, 0.00219023286198, 0.10784867696),
)


def get_tuned_model_options(architecture, method,
                            fake_data=False, fake_training=False):
  hps = {'mlp': _HPS_DICT_MLP, 'lenet': _HPS_DICT_LENET}[architecture][method]
  return model_opts_from_hparams(hps, method, architecture,
                                 fake_data=fake_data,
                                 fake_training=fake_training)


def model_opts_from_hparams(hps, method, architecture,
                            fake_data=False, fake_training=False):
  """Returns a ModelOptions instance using given hyperparameters."""
  num_train_examples = (data_lib.DUMMY_DATA_SIZE if fake_data else
                        data_lib.NUM_TRAIN_EXAMPLES)
  model_opts = models_lib.ModelOptions(
      method=method,
      architecture=architecture,
      train_epochs=1 if fake_training else _TRAIN_EPOCHS,
      num_train_examples=num_train_examples,
      batch_size=hps.batch_size,
      learning_rate=hps.learning_rate,
      mlp_layer_sizes=[200, 200],
      dropout_rate=hps.dropout_rate,
      num_examples_for_predict=55 if fake_training else int(1e4),
      predictions_per_example=4 if fake_training else _PREDICTIONS_PER_EXAMPLE,
  )

  if method == 'vanilla':
    model_opts.predictions_per_example = 1
  return model_opts
