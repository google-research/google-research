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

"""Script to train models using Balanced Pruning MIR."""

from collections.abc import Sequence
import functools
import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
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
    help='Dataset to use.')
_PRED_AGGREGATION = flags.DEFINE_enum(
    'pred_aggregation',
    default='mean',
    enum_values=['mean', 'median'],
    help='Aggregation method for predictions.')
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
    help='Optimizer to use for training.')
_LR = flags.DEFINE_float('lr', default=1e-3, help='Learning rate.')
_WD = flags.DEFINE_float('wd', default=0, help='Weight decay.')
_EPOCHS = flags.DEFINE_integer('epochs', default=100, help='No. of epochs.')
_EXPERIMENT_DIR = flags.DEFINE_string(
    'experiment_dir',
    default='mir_uai24/artifacts/bp_mir',
    help='Experiment directory.')


def filter_fn(
    batch,
    bag_prune_map,
    dataset_info):
  """Prunes the batch according to the provided bag prune map."""

  mask = (
      bag_prune_map[batch[dataset_info.bag_id]]
      != batch[dataset_info.instance_id][:, 0]
  )
  batch[dataset_info.bag_id] = tf.boolean_mask(batch[dataset_info.bag_id], mask)
  batch[dataset_info.instance_id] = tf.boolean_mask(
      batch[dataset_info.instance_id], mask
  )
  batch[dataset_info.bag_id_x_instance_id] = tf.boolean_mask(
      batch[dataset_info.bag_id_x_instance_id], mask
  )
  for feature in dataset_info.features:
    batch[feature.key] = tf.boolean_mask(batch[feature.key], mask)
  batch[dataset_info.label] = tf.boolean_mask(batch[dataset_info.label], mask)
  return batch


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

  model = network.BagMLPModelWithBP(
      _PRED_AGGREGATION.value,
      _EMBEDDING_DIM.value,
      _NUM_HIDDEN_LAYERS.value,
      _NUM_HIDDEN_UNITS.value,
      dataset_info)
  model.compile(optimizer=optimizer, run_eagerly=True)
  experiment_dir = os.path.join(
      _EXPERIMENT_DIR.value,
      '{}',
      (
          f'bs-{_BATCH_SIZE.value}_{_OPTIMIZER.value}_lr-{_LR.value}_'
          f'wd-{_WD.value}_units-{_NUM_HIDDEN_UNITS.value}-'
          f'{_PRED_AGGREGATION.value}'
      )
  )

  logging.info('Storing checkpoints and logs in %s', experiment_dir.format('*'))
  checkpoint_dir = experiment_dir.format('checkpoints')
  log_dir = experiment_dir.format('logs')

  callbacks = [
      tfk.callbacks.TensorBoard(log_dir=log_dir),
      tfk.callbacks.ModelCheckpoint(
          filepath=os.path.join(
              checkpoint_dir, 'checkpoint_bpmir_val_instance_mse'),
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

  model.load_weights(
      os.path.join(checkpoint_dir, 'checkpoint_bpmir_val_instance_mse'))

  mse_loss = tfk.losses.MeanSquaredError(reduction=tfk.losses.Reduction.NONE)
  assert dataset_info is not None
  for i in range(dataset_info.bag_size - 1):
    bag_ids = []
    pruned_instance_ids = []
    for bag_batch in ds_train:
      bag_range_ids = tf.unique(bag_batch['bag_id'])[1]
      preds = model(bag_batch, training=False)
      agg_preds = model.pred_aggregation_fn(preds, bag_range_ids)
      agg_preds = tf.gather(agg_preds, bag_range_ids)
      loss = mse_loss(preds, agg_preds[:, None])

      n_bags = tf.reduce_max(bag_range_ids) + 1
      mask = bag_range_ids[:, None] == tf.range(n_bags)[None]
      loss = loss[:, None] * tf.cast(mask, dtype=tf.float32)
      loss = tf.where(mask, loss, -tf.ones_like(mask, dtype=tf.float32)*np.inf)
      pruned_i = tf.argmax(loss, axis=0)
      pruned_instance_ids.append(
          tf.gather(bag_batch['instance_id'], pruned_i)[:, 0])
      bag_ids.append(tf.unique(bag_batch['bag_id'])[0])
    bag_ids = tf.concat(bag_ids, axis=0)
    pruned_instance_ids = tf.concat(pruned_instance_ids, axis=0)
    bag_prune_map = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(bag_ids, pruned_instance_ids),
        default_value=-1)
    ds_train = ds_train.map(functools.partial(
        filter_fn, bag_prune_map=bag_prune_map, dataset_info=dataset_info))
    model.fit(
        ds_train,
        initial_epoch=(i+1)*_EPOCHS.value,
        epochs=(i+2)*_EPOCHS.value,
        validation_data=ds_val,
        callbacks=callbacks)


if __name__ == '__main__':
  app.run(main)
