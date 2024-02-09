# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Common functions for building TF models."""

import glob
import json
import logging
import os
import tempfile
import time
from typing import Text, Dict, Any, Tuple, Optional

import tensorflow as tf
from tensorflow.compat.v2 import keras


from simulation_research.next_day_wildfire_spread import file_util
from tensorflow.contrib import training as contrib_training

CONV2D_FILTERS_DEFAULT = 64
CONV2D_KERNEL_SIZE_DEFAULT = 3
CONV2D_STRIDES_DEFAULT = 1
CONV2D_PADDING_DEFAULT = 'same'
CONV2D_BIAS_DEFAULT = False
RES_SHORTCUT_KERNEL_SIZE = 1
RES_STRIDES_LIST_DEFAULT = (2, 1)
RES_DECODER_STRIDES = (1, 1)
RES_POOL_SIZE_DEFAULT = 2
DROPOUT_DEFAULT = 0.0
BATCH_NORM_DEFAULT = 'none'
L1_REGULARIZATION_DEFAULT = 0.0
L2_REGULARIZATION_DEFAULT = 0.0
CLIPNORM_DEFAULT = 1e6  # Large value used as placeholder for 'None'.

f_open = open
f_glob = glob.glob


def conv2d_layer(
    filters = CONV2D_FILTERS_DEFAULT,
    kernel_size = CONV2D_KERNEL_SIZE_DEFAULT,
    strides = CONV2D_STRIDES_DEFAULT,
    padding = CONV2D_PADDING_DEFAULT,
    use_bias = CONV2D_BIAS_DEFAULT,
    bias_initializer = keras.initializers.zeros(
    ),
    l1_regularization = L1_REGULARIZATION_DEFAULT,
    l2_regularization = L2_REGULARIZATION_DEFAULT
):
  """Creates convolution 2D layer.

  Args:
    filters: The dimensionality of the output space (i.e. the number of output
      filters in the convolution).
    kernel_size: Height and width of the 2D convolution window.
    strides: Strides of the convolution along the height and width.
    padding: one of `valid` or `same` (case-insensitive). `valid` means no
      padding. `same` results in padding evenly to the left/right and/or up/down
      of the input such that output has the same height/width dimension as the
      input.
    use_bias: Whether to use bias vector.
    bias_initializer: Initializer for the bias vector.
    l1_regularization: L1 regularization factor applied on the kernel.
    l2_regularization: L2 regularization factor applied on the kernel.

  Returns:
    Conv2D layer.
  """
  return keras.layers.Conv2D(
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      use_bias=use_bias,
      bias_initializer=bias_initializer,
      kernel_regularizer=keras.regularizers.l1_l2(
          l1=l1_regularization, l2=l2_regularization))


def res_block(
    input_tensor,
    filters,
    strides = RES_STRIDES_LIST_DEFAULT,
    pool_size = RES_POOL_SIZE_DEFAULT,
    dropout = DROPOUT_DEFAULT,
    batch_norm = BATCH_NORM_DEFAULT,
    l1_regularization = L1_REGULARIZATION_DEFAULT,
    l2_regularization = L2_REGULARIZATION_DEFAULT):
  """Creates convolution layer blocks with residual connections.

  Args:
    input_tensor: Input to the residual block.
    filters: Filters to use in successive layers.
    strides: Strides to use in successive layers.
    pool_size: Size of the max pool window.
    dropout: Dropout rate.
    batch_norm: Controls batch normalization layers.
    l1_regularization: L1 regularization factor applied on the kernel.
    l2_regularization: L2 regularization factor applied on the kernel.

  Returns:
    Output of the residual block.
  """
  res_path = input_tensor
  if batch_norm != 'none':
    res_path = keras.layers.BatchNormalization()(res_path)
  res_path = keras.layers.LeakyReLU()(res_path)
  res_path = keras.layers.Dropout(dropout)(res_path)
  if strides[0] == 1:
    res_path = conv2d_layer(
        filters=filters[0],
        l1_regularization=l1_regularization,
        l2_regularization=l2_regularization)(
            res_path)
  else:
    res_path = keras.layers.MaxPooling2D(
        pool_size=pool_size, strides=strides[0])(
            res_path)

  if batch_norm == 'all':
    res_path = keras.layers.BatchNormalization()(res_path)
  res_path = keras.layers.LeakyReLU()(res_path)
  res_path = keras.layers.Dropout(dropout)(res_path)
  res_path = conv2d_layer(
      filters=filters[1],
      strides=strides[1],
      l1_regularization=l1_regularization,
      l2_regularization=l2_regularization)(
          res_path)

  # Construct the residual link that bypasses this block.
  shortcut = conv2d_layer(
      filters=filters[1],
      kernel_size=RES_SHORTCUT_KERNEL_SIZE,
      strides=strides[0],
      l1_regularization=l1_regularization,
      l2_regularization=l2_regularization)(
          input_tensor)
  if batch_norm == 'all':
    shortcut = keras.layers.BatchNormalization()(shortcut)
  res_path = keras.layers.Dropout(dropout)(res_path)

  res_path = shortcut + res_path
  return res_path


def save_hparams(hparams,
                 path,
                 indent = None,
                 separators = None,
                 sort_keys = False):
  file_util.maybe_make_dirs(path)
  with f_open(path, 'w') as f:
    f.write(hparams.to_json(indent, separators, sort_keys))


def save_keras_model(model,
                     path,
                     include_optimizer = False):
  """Save the Keras model to a file.

  Note: It's possible the saved model will not be TF2-compatible. If you get
  errors using this function, try `save_keras_model_as_h5` instead.

  Args:
    model: A Keras model.
    path: Where to save the model.
    include_optimizer: Whether the optimizer should also be saved.
  """
  file_util.maybe_make_dirs(path)
  with tempfile.TemporaryDirectory() as temp_dir:
    tf.keras.models.save_model(
        model, temp_dir, include_optimizer=include_optimizer)
    gfile.RecursivelyCopyDir(temp_dir, path, overwrite=True)


def save_keras_model_as_h5(model,
                           filename,
                           include_optimizer = False):
  """Save the Keras model to the given filename in the h5 format.

  It is not preferable to save in the `h5` format. You should only use this
  function if `save_keras_model` does not work for you.

  Args:
    model: the model to save.
    filename: the name of the file, including the full path.
    include_optimizer: whether the optimizer should also be saved.
  """
  file_util.maybe_make_dirs(filename)
  with tempfile.NamedTemporaryFile(suffix='.h5') as f:
    tf.keras.models.save_model(model, f, include_optimizer=include_optimizer)
    gfile.Copy(f.name, filename, overwrite=True)


def save_dict_to_json(data, path):
  """Saves dict as JSON file."""
  json_str = json.dumps(data, indent=2) + '\n'
  file_util.maybe_make_dirs(path)
  with gfile.Open(path, 'w') as f:
    f.write(json_str)


class BestModelExporter(tf.keras.callbacks.Callback):
  """Like Keras's ModelCheckPoint with `save_best_only` set to True.

  Saves Keras models and metrics in files with names of the form
  `<timestamp>.h5`.
  Can recover best metric after failure.
  """

  def __init__(self,
               metric_key,
               min_or_max,
               output_dir,
               use_h5 = False):
    self.metric_key = metric_key
    self.output_dir = output_dir
    if min_or_max not in ('min', 'max'):
      raise ValueError('min_or_max must be specified as \'min\' or \'max\'')
    self.mode = min_or_max
    self.best = None
    self.use_h5 = use_h5

  def on_train_begin(self, logs = None):
    if self.best is None:
      # Check whether to restore best from saved metrics file.
      metrics_file_pattern = os.path.join(self.output_dir, 'metrics_*.json')
      metrics_file_list = sorted(f_glob(metrics_file_pattern))
      if metrics_file_list:
        latest_metrics_filepath = metrics_file_list[-1]
        with f_open(latest_metrics_filepath, 'r') as f:
          metrics_dict = json.load(f)
          best_val_txt = metrics_dict[self.metric_key]
          if best_val_txt:
            self.best = float(best_val_txt)
            logging.info('Best metric value (%s=%s) restored on_train_begin.',
                         self.metric_key, self.best)

  def on_epoch_end(self, epoch, logs):  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
    metric = logs[self.metric_key]
    if (self.best is None or (self.mode == 'min' and metric < self.best) or
        (self.mode == 'max' and metric > self.best)):
      timestamp = int(time.time())
      if self.use_h5:
        filename = f'model_{timestamp}.h5'
        save_keras_model_as_h5(self.model,
                               os.path.join(self.output_dir, filename))
      else:
        filename = f'{timestamp}/'
        save_keras_model(self.model, os.path.join(self.output_dir, filename))
      logging.info(
          'Best model saved as %s after epoch %d (%s=%s, '
          'previous best=%s)', filename, epoch, self.metric_key, metric,
          self.best)
      self.best = metric
      # Make ndarrays/tf.tensors metrics JSON serializable.
      for metric_key in logs:
        if hasattr(logs[metric_key], 'dtype'):
          if tf.is_tensor(logs[metric_key]):
            logs[metric_key] = logs[metric_key].numpy()
          logs[metric_key] = logs[metric_key].item()
      # Save best metrics and epoch.
      metrics_filename = os.path.join(self.output_dir,
                                      f'metrics_{timestamp}.json')
      metrics_dict = logs
      metrics_dict['epoch'] = epoch
      save_dict_to_json(metrics_dict, metrics_filename)
    else:
      logging.info('Model not saved after epoch %d (%s=%s, best=%s)', epoch,
                   self.metric_key, metric, self.best)
