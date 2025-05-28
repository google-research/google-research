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

"""Train graph transformer model."""

from collections.abc import Mapping, Sequence
import json
import os
import re
from typing import Any, Optional

from absl import app
from absl import flags
import tensorflow as tf
import tensorflow_probability as tfp

from CardBench_zero_shot_cardinality_training.graph_transformer import constants
from CardBench_zero_shot_cardinality_training.graph_transformer.models import graph_transformer

_TRAINING_DATASET_NAMES = flags.DEFINE_list(
    'training_dataset_names',
    None,
    'Name of the training dataset.',
    required=True,
)

_TEST_DATASET_NAME = flags.DEFINE_string(
    'test_dataset_name', None, 'Name of the test dataset.', required=True
)

_INPUT_DATASET_PATH = flags.DEFINE_string(
    'input_dataset_path',
    None,
    'Input dataset path.',
    required=True,
)

_MODEL_PATH = flags.DEFINE_string(
    'model_path',
    None,
    'Model path.',
    required=True,
)

_DATASET_TYPE = flags.DEFINE_enum(
    'dataset_type',
    None,
    ['binary_join', 'single_table'],
    'Dataset type.',
    required=True,
)

_TRAIN_RATIO = flags.DEFINE_float(
    'train_ratio', 0.85, 'Train ratio for training/val dataset split'
)

_SCALING_STRATEGY_FILENAME = flags.DEFINE_string(
    'scaling_strategy_filename', None, 'Input scaling strategy'
)

_NUM_EPOCHS = flags.DEFINE_integer('num_epochs', 200, 'Number of epochs')

_INIT_LEARNING_RATE = flags.DEFINE_float(
    'init_lr', 1e-3, 'Initial learning rate'
)
_MIN_LEARNING_RATE = flags.DEFINE_float('min_lr', 1e-5, 'Min learning rate')

_NUM_ENCODING_LAYERS = flags.DEFINE_integer(
    'num_encoding_layers', 16, 'Number of encoding layers'
)

_NUM_EMBEDDING_LAYERS = flags.DEFINE_integer(
    'num_embedding_layers', 3, 'Number of embedding layers'
)

_NUM_OUTPUT_LAYERS = flags.DEFINE_integer(
    'num_output_layers', 3, 'Number of output layers'
)

_MODEL_DIM = flags.DEFINE_integer('model_dim', 128, 'Model dimension')

_NUM_HEADS = flags.DEFINE_integer('num_heads', 8, 'Number of heads')

_DROPOUT = flags.DEFINE_float('dropout', 0.0, 'Dropout rate')

_MASK_TYPE = flags.DEFINE_string(
    'mask_type', 'ancestor_causal_mask', 'Mask type'
)


_REDUCE_LR_PATIENCE = flags.DEFINE_integer(
    'reduce_lr_patience', 5, 'Reduce learning rate patience'
)

_REDUCE_LR_FACTOR = flags.DEFINE_float(
    'reduce_lr_factor', 0.7, 'Reduce learning rate factor'
)

_EARLY_STOPPING_PATIENCE = flags.DEFINE_integer(
    'early_stopping_patience', 10, 'Early stopping patience'
)

_BASE_MODEL_CHECKPOINT_PATH = flags.DEFINE_string(
    'base_model_checkpoint_path',
    None,
    'Base model checkpoint path',
)

_BATCH_SIZE = flags.DEFINE_integer('batch_size', 64, 'BATCH_SIZE')

_TRAIN_VAL_SAMPLE_SIZE = flags.DEFINE_integer(
    'train_val_sample_size',
    5000,
    'Training and validation dataset sample size.',
)

_TEST_SAMPLE_SIZE = flags.DEFINE_integer(
    'test_sample_size', 500, 'Test dataset sample size.'
)

_LABEL = flags.DEFINE_enum(
    'label',
    'cardinality',
    ['exec_time', 'cardinality'],
    'Label name',
)


def parse_example(
    example, label_name
):
  """Parse a single example from tfrecord serilized data."""

  data = {
      'node': tf.io.FixedLenFeature([], tf.string),
      'query_id': tf.io.FixedLenFeature([], tf.string),
      'node_padding': tf.io.FixedLenFeature([], tf.string),
      'topological_order': tf.io.FixedLenFeature([], tf.string),
      'parent_causal_mask': tf.io.FixedLenFeature([], tf.string),
      'ancestor_causal_mask': tf.io.FixedLenFeature([], tf.string),
      'spatial_encoding': tf.io.FixedLenFeature([], tf.string),
      'cardinality': tf.io.FixedLenFeature([], tf.string),
      'exec_time': tf.io.FixedLenFeature([], tf.string),
  }

  content = tf.io.parse_single_example(example, data)
  x = {}
  y = tf.constant([0.0], dtype=tf.float32)
  for k in content:
    if k == 'query_id':
      pass
    elif k == label_name:
      y = tf.io.parse_tensor(content[k], out_type=tf.float32)
    elif k not in constants.LABELS:
      x[k] = tf.io.parse_tensor(content[k], out_type=tf.float32)

  return x, y


def read_data(
    batch_size,
    train_val_dataset_paths,
    test_dataset_path,
    label_name,
    train_val_size,
    test_size,
    train_ratio,
    random_seed = 12345,
):
  """Read data from tfrecord files, split the dataset into train/val/test sets."""

  train_ds = None
  val_ds = None
  test_ds = None
  train_size = int(train_ratio * train_val_size)

  for train_val_dataset_path in train_val_dataset_paths:
    ds = tf.data.TFRecordDataset(train_val_dataset_path).map(
        lambda x: parse_example(x, label_name),
        num_parallel_calls=8,
        deterministic=True,
    )
    ds = ds.shuffle(10000, seed=random_seed, reshuffle_each_iteration=False)

    # If test dataset path is the same as train/val dataset path, we split the
    # dataset into train/val/test sets.
    if test_dataset_path == train_val_dataset_path:
      test_ds = ds.take(test_size)
    ds = ds.skip(test_size)

    # Build training set by taking the first portion of train+val set.
    if train_ds is None:
      train_ds = ds.take(train_size)
    else:
      train_ds = train_ds.concatenate(ds.take(train_size))

    # Build val set by taking the second portion of train+val set.
    if val_ds is None:
      val_ds = ds.skip(train_size)
    else:
      val_ds = val_ds.concatenate(ds.skip(train_size))

  # If test dataset path is not the same as train/val dataset path, we read the
  # test dataset separately.
  if test_dataset_path not in train_val_dataset_paths:
    test_ds = tf.data.TFRecordDataset(test_dataset_path).map(
        lambda x: parse_example(x, label_name),
        num_parallel_calls=8,
        deterministic=True,
    )
    test_ds = test_ds.shuffle(
        len(list(test_ds)), seed=random_seed, reshuffle_each_iteration=False
    )
    test_ds = test_ds.take(test_size)

  padded_shapes = {
      'node': [constants.MAX_NUM_NODES, constants.NODE_FEATURE_DIM],
      'topological_order': [constants.MAX_NUM_NODES, 1],
      'parent_causal_mask': [constants.MAX_NUM_NODES, constants.MAX_NUM_NODES],
      'ancestor_causal_mask': [
          constants.MAX_NUM_NODES,
          constants.MAX_NUM_NODES,
      ],
      'spatial_encoding': [constants.MAX_NUM_NODES, constants.MAX_NUM_NODES],
      'node_padding': [constants.MAX_NUM_NODES],
  }

  if not train_ds:
    raise ValueError('train dataset is empty')
  if not val_ds:
    raise ValueError('val dataset is empty')
  if not test_ds:
    raise ValueError('test dataset is empty')

  # Make sure to shuffle the training dataset every iteration during training.
  train_ds = train_ds.shuffle(
      10000, seed=random_seed, reshuffle_each_iteration=True
  )

  train_ds_batched = train_ds.padded_batch(
      batch_size=batch_size, padded_shapes=(padded_shapes, [1])
  )
  val_ds_batched = val_ds.padded_batch(
      batch_size=batch_size, padded_shapes=(padded_shapes, [1])
  )
  test_ds_batched = test_ds.padded_batch(
      batch_size=batch_size, padded_shapes=(padded_shapes, [1])
  )

  return (
      train_ds_batched,
      val_ds_batched,
      test_ds_batched,
  )


class QErrorMetric(tf.keras.metrics.Metric):
  """QError metric."""

  def __init__(
      self,
      name,
      scaling_strategy,
      label_name,
      dtype = None,
  ):
    super().__init__(name=name, dtype=dtype)
    self.scaling_strategy = scaling_strategy
    self.label_name = label_name
    self.q_errors = []

  def update_state(
      self,
      y_true,
      y_pred,
      sample_weight = None,
  ):
    """Updates the state of the metric."""
    y_true = tf.convert_to_tensor(y_true, dtype=self._dtype)
    y_pred = tf.convert_to_tensor(y_pred, dtype=self._dtype)

    unscaled_y_true = tf.pow(
        10.0,
        y_true * self.scaling_strategy['g'][self.label_name]['log_std']
        + self.scaling_strategy['g'][self.label_name]['log_mean'],
    )
    unscaled_y_pred = tf.pow(
        10.0,
        y_pred * self.scaling_strategy['g'][self.label_name]['log_std']
        + self.scaling_strategy['g'][self.label_name]['log_mean'],
    )
    div1 = tf.math.divide(unscaled_y_true, unscaled_y_pred)
    div2 = tf.math.divide(unscaled_y_pred, unscaled_y_true)
    self.q_errors = tf.math.maximum(div1, div2)

  def result(self):
    if self.name.startswith('mean'):
      return tf.reduce_mean(self.q_errors)
    elif self.name.startswith('p'):
      quantile = float(re.findall(r'\d+', self.name)[0])
      return tfp.stats.percentile(self.q_errors, quantile)
    else:
      raise NotImplementedError(f'Unsupported q_error metric type: {self.name}')


class TestModelCallback(tf.keras.callbacks.Callback):
  """Log test errors after each epoch."""

  def __init__(
      self,
      model,
      test_ds_batched,
      scaling_strategy,
      label_name,
  ):
    super().__init__()
    self.model = model
    self.test_ds_batched = test_ds_batched
    self.scaling_strategy = scaling_strategy
    self.label_name = label_name

  def on_epoch_end(self, epoch, logs=None):
    mean_q_error = QErrorMetric(
        'mean_q_error', self.scaling_strategy, self.label_name
    )
    p50_q_error = QErrorMetric(
        'p50_q_error', self.scaling_strategy, self.label_name
    )
    p75_q_error = QErrorMetric(
        'p75_q_error', self.scaling_strategy, self.label_name
    )
    p90_q_error = QErrorMetric(
        'p90_q_error', self.scaling_strategy, self.label_name
    )
    p95_q_error = QErrorMetric(
        'p95_q_error', self.scaling_strategy, self.label_name
    )
    p99_q_error = QErrorMetric(
        'p99_q_error', self.scaling_strategy, self.label_name
    )
    for x, y in self.test_ds_batched:
      y_pred = self.model(x, training=False)
      mean_q_error.update_state(y, y_pred)
      p50_q_error.update_state(y, y_pred)
      p75_q_error.update_state(y, y_pred)
      p90_q_error.update_state(y, y_pred)
      p95_q_error.update_state(y, y_pred)
      p99_q_error.update_state(y, y_pred)

    logs['test_mean_q_error'] = mean_q_error.result()
    logs['test_p50_q_error'] = p50_q_error.result()
    logs['test_p75_q_error'] = p75_q_error.result()
    logs['test_p90_q_error'] = p90_q_error.result()
    logs['test_p95_q_error'] = p95_q_error.result()
    logs['test_p99_q_error'] = p99_q_error.result()


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  train_dataset_paths = []

  for dataset_name in _TRAINING_DATASET_NAMES.value:
    train_path = os.path.join(
        _INPUT_DATASET_PATH.value,
        _DATASET_TYPE.value,
        f'{dataset_name}_{_DATASET_TYPE.value}.tfrecord',
    )
    train_dataset_paths.append(train_path)

  test_dataset_path = os.path.join(
      _INPUT_DATASET_PATH.value,
      _DATASET_TYPE.value,
      f'{_TEST_DATASET_NAME.value}_{_DATASET_TYPE.value}.tfrecord',
  )

  checkpoint_path = os.path.join(
      _MODEL_PATH.value,
      'graph_transformer.ckpt',
  )

  os.makedirs(_MODEL_PATH.value, exist_ok=True)

  # Load scaling strategy
  with open(
      os.path.join(
          _INPUT_DATASET_PATH.value,
          _DATASET_TYPE.value,
          _SCALING_STRATEGY_FILENAME.value,
      )
  ) as f:
    scaling_strategy = json.load(f)

  mean_q_error = QErrorMetric('mean_q_error', scaling_strategy, _LABEL.value)
  p50_q_error = QErrorMetric('p50_q_error', scaling_strategy, _LABEL.value)
  p75_q_error = QErrorMetric('p75_q_error', scaling_strategy, _LABEL.value)
  p90_q_error = QErrorMetric('p90_q_error', scaling_strategy, _LABEL.value)
  p95_q_error = QErrorMetric('p95_q_error', scaling_strategy, _LABEL.value)
  p99_q_error = QErrorMetric('p99_q_error', scaling_strategy, _LABEL.value)

  # Load and split training/val/test datasets
  train_ds, val_ds, test_ds = read_data(
      batch_size=_BATCH_SIZE.value,
      train_val_dataset_paths=train_dataset_paths,
      test_dataset_path=test_dataset_path,
      label_name=_LABEL.value,
      train_val_size=_TRAIN_VAL_SAMPLE_SIZE.value,
      test_size=_TEST_SAMPLE_SIZE.value,
      train_ratio=_TRAIN_RATIO.value,
  )

  # Model initialization
  model = graph_transformer.Predictor(
      num_encoder_layer=_NUM_ENCODING_LAYERS.value,
      num_embedding_layer=_NUM_EMBEDDING_LAYERS.value,
      num_output_layer=_NUM_OUTPUT_LAYERS.value,
      model_dim=_MODEL_DIM.value,
      num_heads=_NUM_HEADS.value,
      hidden_dim=_MODEL_DIM.value,
      output_dim=1,
      num_nodes=constants.MAX_NUM_NODES,
      node_feature_dim=constants.NODE_FEATURE_DIM,
      num_node_types=len(constants.NODE_TYPES),
      num_edge_types=len(constants.EDGE_TYPES),
      mask_type=_MASK_TYPE.value,
      dropout_prob=_DROPOUT.value,
  )

  optimizer = tf.keras.optimizers.Adam(learning_rate=_INIT_LEARNING_RATE.value)
  loss = tf.keras.losses.MeanAbsoluteError()
  metrics = [
      mean_q_error,
      p50_q_error,
      p75_q_error,
      p90_q_error,
      p95_q_error,
      p99_q_error,
  ]
  model.compile(optimizer, loss=loss, metrics=metrics)

  # Load model checkpoint if exists
  if _BASE_MODEL_CHECKPOINT_PATH.value:
    model.load_weights(_BASE_MODEL_CHECKPOINT_PATH.value)
    model.optimizer.learning_rate = _INIT_LEARNING_RATE.value

  # Eerly stopping if the validation loss does not improve after
  # _EARLY_STOPPING_PATIENCE epochs.
  earlystopping_callback = tf.keras.callbacks.EarlyStopping(
      monitor='val_loss',
      patience=_EARLY_STOPPING_PATIENCE.value,
      restore_best_weights=True,
  )

  # Reduce learning rate if the validation loss does not improve after
  # _REDUCE_LR_PATIENCE epochs.
  reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
      monitor='val_loss',
      factor=_REDUCE_LR_FACTOR.value,
      patience=_REDUCE_LR_PATIENCE.value,
      min_lr=_MIN_LEARNING_RATE.value,
      min_delta=1e-9,
  )

  cp_callback = tf.keras.callbacks.ModelCheckpoint(
      monitor='val_loss',
      filepath=checkpoint_path,
      verbose=1,
      save_weights_only=True,
      save_freq='epoch',
  )

  test_model_callback = TestModelCallback(
      model,
      test_ds,
      scaling_strategy,
      _LABEL.value,
  )

  model.fit(
      train_ds,
      epochs=_NUM_EPOCHS.value,
      validation_data=val_ds,
      callbacks=[
          earlystopping_callback,
          reduce_lr_callback,
          cp_callback,
          test_model_callback,
      ],
  )


if __name__ == '__main__':
  app.run(main)
