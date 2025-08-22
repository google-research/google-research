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

"""Script to train models using Instance MIR."""

from collections.abc import Sequence
import functools
import os

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from mir_uai24 import dataset
from mir_uai24 import enum_utils
from mir_uai24 import network

# pylint: disable=g-bad-import-order

tfk = tf.keras

_DATASET = flags.DEFINE_enum_class(
    'dataset',
    default=enum_utils.Dataset.SYNTHETIC,
    enum_class=enum_utils.Dataset,
    help='Dataset to use.',)
_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size', default=64, help='No. of bags in a batch.')
_EMBEDDING_DIM = flags.DEFINE_integer(
    'embedding_dim', default=32, help='Embedding dimension.')
_NUM_HIDDEN_LAYERS = flags.DEFINE_integer(
    'num_hidden_layers',
    default=2,
    help='No. of hidden layers to use in the MLP model.')
_NUM_HIDDEN_UNITS = flags.DEFINE_integer(
    'num_hidden_units',
    default=32,
    help='No. of hidden units to use in the MLP hidden layers.')
_OPTIMIZER = flags.DEFINE_enum(
    'optimizer',
    default='adam',
    enum_values=['sgd', 'adam'],
    help='Optimizer to use for training.',)
_LR = flags.DEFINE_float('lr', default=1e-3, help='Learning rate.')
_WD = flags.DEFINE_float('wd', default=0, help='Weight decay.')
_EPOCHS = flags.DEFINE_integer('epochs', default=100, help='No. of epochs.')
_EXPERIMENT_DIR = flags.DEFINE_string(
    'experiment_dir',
    default='mir_uai24/artifacts/instance_mir',
    help='Experiment directory.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  (ds_train, ds_val, ds_test), dataset_info = dataset.load(
      _DATASET.value, False, _BATCH_SIZE.value, with_info=True)
  assert isinstance(dataset_info, enum_utils.DatasetInfo)

  ds_train = ds_train.map(functools.partial(
      dataset.instancemir_transform, dataset_info=dataset_info))

  optimizer = {
      'adam': tfk.optimizers.AdamW,
      'sgd': tfk.optimizers.SGD
  }[_OPTIMIZER.value](learning_rate=_LR.value, weight_decay=_WD.value)

  model = network.InstanceMLPModel(
      _EMBEDDING_DIM.value, _NUM_HIDDEN_LAYERS.value, _NUM_HIDDEN_UNITS.value,
      dataset_info)
  model.compile(optimizer=optimizer, run_eagerly=True)
  experiment_dir = os.path.join(
      _EXPERIMENT_DIR.value,
      '{}',
      (
          f'bs-{_BATCH_SIZE.value}_{_OPTIMIZER.value}_lr-{_LR.value}_'
          f'wd-{_WD.value}_u-{_NUM_HIDDEN_UNITS.value}')
  )

  logging.info('Storing checkpoints and logs in %s', experiment_dir.format('*'))
  checkpoint_dir = experiment_dir.format('checkpoints')
  log_dir = experiment_dir.format('logs')

  callbacks = [
      tfk.callbacks.TensorBoard(log_dir=log_dir),
      tfk.callbacks.ModelCheckpoint(
          filepath=os.path.join(
              checkpoint_dir, 'checkpoint_val_instance_mse-{epoch:03d}'),
          monitor='val_instance_mse',
          mode='min',
          save_best_only=True,
          save_weights_only=True
          ),
  ]

  model.fit(
      ds_train,
      epochs=_EPOCHS.value,
      validation_data=ds_val,
      callbacks=callbacks)



if __name__ == '__main__':
  app.run(main)
