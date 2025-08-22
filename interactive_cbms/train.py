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

"""Interactive bottleneck training and evaluation."""

import os
from typing import Sequence

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
from tensorflow_addons import optimizers as tfa_optimizers

from interactive_cbms import enum_utils
from interactive_cbms import network
from interactive_cbms.datasets import chexpert_dataset
from interactive_cbms.datasets import cub_dataset
from interactive_cbms.datasets import oai_dataset

tfk = tf.keras

_ARCH = flags.DEFINE_enum_class(
    'arch',
    default=enum_utils.Arch.X_TO_C_TO_Y,
    enum_class=enum_utils.Arch,
    help='Architecture to use for training.')
_NON_LINEAR_CTOY = flags.DEFINE_bool(
    'non_linear_ctoy',
    default=False,
    help='Whether to use a non-linear CtoY model.')
_DATASET = flags.DEFINE_enum_class(
    'dataset',
    default=enum_utils.Dataset.CUB,
    enum_class=enum_utils.Dataset,
    help='Dataset to use for training.')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', default=32, help='Batch Size')
_MERGE_TRAIN_AND_VAL = flags.DEFINE_bool(
    'merge_train_and_val',
    default=False,
    help='Whether to merge training and validation sets for training.')
_OPTIMIZER = flags.DEFINE_enum(
    'optimizer',
    default='sgd',
    enum_values=['sgd', 'adam'],
    help='Optimizer to use for training.')
_LR = flags.DEFINE_float('lr', default=1e-3, help='Learning rate.')
_WD = flags.DEFINE_float('wd', default=0, help='Weight decay.')
_LOSS_WEIGHTS = flags.DEFINE_list(
    'loss_weights', default=None, help='Loss weights')
_EPOCHS = flags.DEFINE_integer('epochs', default=100, help='No. of epochs.')
_STOPPING_PATIENCE = flags.DEFINE_integer(
    'stopping_patience',
    default=100,
    help='Patience (in no. of epochs) for early stopping.')
_EXPERIMENT_DIR = flags.DEFINE_string(
    'experiment_dir',
    default='ICBM_checkpoints',
    help='Experiment directory to save models and results.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if _DATASET.value is enum_utils.Dataset.CUB:
    dataset = cub_dataset
  elif _DATASET.value is enum_utils.Dataset.CHEXPERT:
    dataset = chexpert_dataset
  elif _DATASET.value is enum_utils.Dataset.OAI:
    dataset = oai_dataset
  ds_train, ds_test, _ = dataset.load_dataset(
      batch_size=_BATCH_SIZE.value,
      merge_train_and_val=_MERGE_TRAIN_AND_VAL.value)

  model = network.InteractiveBottleneckModel(
      arch=_ARCH.value,
      n_concepts=dataset.Config.n_concepts,
      n_classes=dataset.Config.n_classes,
      non_linear_ctoy=_NON_LINEAR_CTOY.value)
  logging.info('%s model initialized', model.arch)

  if _OPTIMIZER.value == 'sgd':
    loss_weights = _LOSS_WEIGHTS.value
    if loss_weights is not None:
      loss_weights = [[float(w)] for w in loss_weights]
    model.compile(
        optimizer=tfa_optimizers.SGDW(
            learning_rate=_LR.value, momentum=0.9, weight_decay=_WD.value),
        loss_weights=loss_weights)
  else:
    model.compile(
        optimizer=tfa_optimizers.AdamW(
            learning_rate=_LR.value, weight_decay=_WD.value))

  logging.info('Using loss weights: %s', str(model.loss_weights))

  checkpoint_dir = os.path.join(
      _EXPERIMENT_DIR.value, _DATASET.value, _ARCH.value,
      f'{_OPTIMIZER.value}_lr-{_LR.value}_wd-{_WD.value}')

  model.fit(
      ds_train,
      validation_data=ds_test,
      epochs=_EPOCHS.value,
      verbose=2,
      callbacks=[
          tfk.callbacks.EarlyStopping(
              monitor=model.custom_loss_metrics[-1][0].name,
              patience=_STOPPING_PATIENCE.value,
              verbose=2,
              mode='min',
              restore_best_weights=False),
          tfk.callbacks.ModelCheckpoint(
              os.path.join(checkpoint_dir, 'checkpoint_class'),
              monitor=f'val_{model.custom_alt_metrics[-1][0].name}',
              mode='max',
              save_weights_only=True,
              save_best_only=True),
          tfk.callbacks.ModelCheckpoint(
              os.path.join(checkpoint_dir, 'checkpoint_concept'),
              monitor=f'val_{model.custom_alt_metrics[0][0].name}',
              mode='max',
              save_weights_only=True,
              save_best_only=True),
          tfk.callbacks.ModelCheckpoint(
              os.path.join(checkpoint_dir, 'checkpoint_trainloss'),
              monitor=f'{model.custom_loss_metrics[-1][0].name}',
              mode='min',
              save_weights_only=True,
              save_best_only=True),
          tfk.callbacks.TensorBoard(
              log_dir=(f'{_EXPERIMENT_DIR.value}/{_DATASET.value}/logs/'
                       f'{_ARCH.value}/{_OPTIMIZER.value}_lr-{_LR.value}_'
                       f'wd-{_WD.value}/'))
      ])

  model.load_weights(os.path.join(checkpoint_dir, 'checkpoint_trainloss'))
  model.evaluate(ds_test, verbose=2)


if __name__ == '__main__':
  app.run(main)
