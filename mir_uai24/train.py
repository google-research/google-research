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

"""Script to train models using wtd-Assign."""

from collections.abc import Sequence
import json
import os

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from mir_uai24 import dataset
from mir_uai24 import enum_utils
from mir_uai24 import network
from mir_uai24 import priors

# pylint: disable=g-bad-import-order

tfk = tf.keras

_TRAIN_INSTANCE = flags.DEFINE_boolean(
    'train_instance',
    default=False,
    help='Train on instance-level data.')
_DATASET = flags.DEFINE_enum_class(
    'dataset',
    default=enum_utils.Dataset.SYNTHETIC,
    enum_class=enum_utils.Dataset,
    help='Dataset to use.')
_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size',
    default=100,
    help='No. of bags in a batch.')
_EMBEDDING_DIM = flags.DEFINE_integer(
    'embedding_dim',
    default=0,
    help='Embedding dimension.')
_NUM_HIDDEN_LAYERS = flags.DEFINE_integer(
    'num_hidden_layers',
    default=1,
    help='No. of hidden layers to use in the MLP model.')
_NUM_HIDDEN_UNITS = flags.DEFINE_integer(
    'num_hidden_units',
    default=1024,
    help='No. of hidden units to use in the MLP hidden layers.')
_OPTIMIZER = flags.DEFINE_enum(
    'optimizer',
    default='adam',
    enum_values=['sgd', 'adam'],
    help='Optimizer to use for training.')
_LR = flags.DEFINE_float('lr', default=1e-3, help='Learning rate.')
_WD = flags.DEFINE_float('wd', default=0, help='Weight decay.')
_POSTERIOR_LR = flags.DEFINE_float(
    'posterior_lr',
    default=None,
    help='Posterior learning rate.')
_ERM_LOSS_WEIGHTS = flags.DEFINE_string(
    'erm_loss_weights',
    default=json.dumps({
        'bag_mse': 1.0, 'posterior_bce_loss': 1.0,
        'posterior_sum_1': 1.0, 'overlap_posterior_max_sum_1': 1.0}),
    help='Weights for ERM loss.')
_EPOCHS = flags.DEFINE_integer('epochs', default=100, help='No. of epochs.')
_EXPERIMENT_DIR = flags.DEFINE_string(
    'experiment_dir',
    default=('mir_uai24/artifacts/wtd-Assign'),
    help='Experiment directory.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  (ds_train, ds_val, ds_test), dataset_info = dataset.load(
      _DATASET.value, _TRAIN_INSTANCE.value, _BATCH_SIZE.value, with_info=True)

  optimizer = {
      'adam': tfk.optimizers.AdamW,
      'sgd': tfk.optimizers.SGD
  }[_OPTIMIZER.value](learning_rate=_LR.value, weight_decay=_WD.value)

  if _TRAIN_INSTANCE.value:
    model = network.InstanceMLPModel(
        _EMBEDDING_DIM.value, _NUM_HIDDEN_LAYERS.value, _NUM_HIDDEN_UNITS.value,
        dataset_info)
    model.compile(optimizer=optimizer, run_eagerly=True)
    experiment_dir = os.path.join(
        _EXPERIMENT_DIR.value,
        '{}',
        (
            f'nunits-{_NUM_HIDDEN_UNITS.value}_bs-{_BATCH_SIZE.value}_'
            f'{_OPTIMIZER.value}_lr-{_LR.value}_wd-{_WD.value}')
        )
  else:
    prior = priors.uniform(dataset_info)
    model = network.BagMLPModelWithERM(
        _EMBEDDING_DIM.value, _NUM_HIDDEN_LAYERS.value, _NUM_HIDDEN_UNITS.value,
        dataset_info, prior)
    if _POSTERIOR_LR.value is not None:
      posterior_lr = _POSTERIOR_LR.value
    else:
      posterior_lr = _LR.value

    posterior_optimizer = {
        'adam': tfk.optimizers.Adam,
        'sgd': tfk.optimizers.SGD
    }[_OPTIMIZER.value](learning_rate=posterior_lr)

    erm_loss_weights = json.loads(_ERM_LOSS_WEIGHTS.value)
    model.compile(
        optimizer=optimizer,
        posterior_optimizer=posterior_optimizer,
        loss_weights=erm_loss_weights,
        run_eagerly=True)

    experiment_dir = os.path.join(
        _EXPERIMENT_DIR.value,
        '{}',
        (
            f'bs-{_BATCH_SIZE.value}_{_OPTIMIZER.value}_lr-{_LR.value}_'
            f'{_POSTERIOR_LR.value}_wd-{_WD.value}_erm_loss_w-'
            f'{erm_loss_weights["bag_mse"]}_'
            f'{erm_loss_weights["posterior_bce_loss"]}_'
            f'{erm_loss_weights["posterior_sum_1"]}_'
            f'{erm_loss_weights["overlap_posterior_max_sum_1"]}'
        ),
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
      callbacks=callbacks
  )


if __name__ == '__main__':
  app.run(main)
