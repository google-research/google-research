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

"""Utilities and common steps for model export."""

import os

from typing import Any, Dict, List, Optional

from absl import flags
from absl import logging

import numpy as np
import tensorflow as tf

from non_semantic_speech_benchmark.data_prep import data_prep_utils
from non_semantic_speech_benchmark.distillation import frontend_lib
from non_semantic_speech_benchmark.distillation import models


def get_experiment_dirs(experiment_dir):
  """Returns a list of experiment directories.

  NOTE: This assumes that only folders with hyperparams in their name occur in
  the working dict.

  Args:
    experiment_dir: Base for all directories.

  Returns:
    List of specific experiment subdirs.
  """
  if not tf.io.gfile.exists(experiment_dir):
    raise ValueError(f'Experiment dir doesn\'t exist: {experiment_dir}')
  experiment_dirs = [f for f in tf.io.gfile.listdir(experiment_dir)
                     if tf.io.gfile.isdir(os.path.join(experiment_dir, f))]
  return experiment_dirs


def get_params(experiment_dir_str):
  """Extracts hyperparams from experiment directory string.

  Args:
    experiment_dir_str: The folder-name for the set of hyperparams. Eg:
      '1-al=1.0,ap=False,lr=0.0001,ms=small,tbs=512'

  Returns:
    A dict mapping param key (str) to eval'ed value (float/eval/string).
  """
  parsed_params = {}
  start_idx = experiment_dir_str.find('-') + 1
  for kv in experiment_dir_str[start_idx:].split(','):
    cur_split = kv.split('=')
    if len(cur_split) != 2:
      raise ValueError(f'Folder doesn\'t split properly: {kv}')
    key, value = cur_split
    try:
      value = eval(value)  # pylint: disable=eval-used
    except:  # pylint: disable=bare-except
      pass
    parsed_params[key] = value
  return parsed_params


def get_model(checkpoint_folder_path,
              params,
              tflite_friendly,
              checkpoint_number = None,
              include_frontend = False):
  """Given folder & training params, exports SavedModel without frontend."""
  # Optionally override frontend flags from
  # `non_semantic_speech_benchmark/export_model/tf_frontend.py`
  override_flag_names = [
      'frame_hop', 'n_required', 'num_mel_bins', 'frame_width', 'pad_mode'
  ]
  for flag_name in override_flag_names:
    if flag_name in params:
      setattr(flags.FLAGS, flag_name, params[flag_name])

  static_model = models.get_keras_model(
      params['mt'],
      output_dimension=1024,
      truncate_output=params['tr'] if 'tr' in params else False,
      frontend=include_frontend,
      tflite=tflite_friendly)
  checkpoint = tf.train.Checkpoint(model=static_model)
  if checkpoint_number:
    checkpoint_to_load = os.path.join(
        checkpoint_folder_path, f'ckpt-{checkpoint_number}')
    assert tf.train.load_checkpoint(checkpoint_to_load)
  else:
    checkpoint_to_load = tf.train.latest_checkpoint(checkpoint_folder_path)
  checkpoint.restore(checkpoint_to_load).expect_partial()
  return static_model


def convert_tflite_model(model, quantize,
                         model_path):
  """Uses TFLiteConverter to convert a Keras Model.

  Args:
    model: Keras model obtained from get_tflite_friendly_model.
    quantize: Whether to quantize TFLite model using dynamic quantization. See:
      https://www.tensorflow.org/lite/performance/post_training_quant
    model_path: Path for TFLite file.
  """
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  converter.target_spec.supported_ops = [
      tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
      # There is a GatherV2 op in the frontend that isn't supported by TFLite
      # as a builtin op. (It works as a TFLite builtin only if the sample size
      # to the frontend is a constant)
      # However, TFLite supports importing some relevant operators from TF,
      # at the cost of binary size (~ a few MB).
      # See: https://www.tensorflow.org/lite/guide/ops_select
      # NOTE: This has no effect on the model/binary size if the graph does not
      # required the extra TF ops (for example, for no-frontend versio
      tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
  ]
  if quantize:
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
  tflite_buffer = converter.convert()

  with tf.io.gfile.GFile(model_path, 'wb') as f:
    f.write(tflite_buffer)
  logging.info('Exported TFLite model to %s.', model_path)


def sanity_check(
    include_frontend,
    model_path,
    embedding_dim,
    tflite,
    n_required = None,
    frame_width = None,
    num_mel_bins = None):
  """Sanity check model by running dummy inference."""
  n_required = n_required or flags.FLAGS.n_required
  frame_width = frame_width or flags.FLAGS.frame_width
  num_mel_bins = num_mel_bins or flags.FLAGS.num_mel_bins

  if include_frontend:
    input_shape = (1, 2 * n_required)
    expected_output_shape = (7, embedding_dim)
  else:
    feats_inner_dim = frontend_lib.get_frontend_output_shape()[0] * frame_width
    input_shape = (1, feats_inner_dim, num_mel_bins, 1)
    expected_output_shape = (1, embedding_dim)
  logging.info('Input shape: %s. Expected output shape: %s', input_shape,
               expected_output_shape)
  model_input = np.zeros(input_shape, dtype=np.float32)

  if tflite:
    logging.info('Building tflite interpreter...')
    interpreter = data_prep_utils.build_tflite_interpreter(
        model_path)
    logging.info('Running inference...')
    output = data_prep_utils.samples_to_embedding_tflite(
        model_input, sample_rate=16000, interpreter=interpreter, output_key='0',
        name='sanity_check')
  else:
    logging.info('Loading and running inference with SavedModel...')
    model = tf.saved_model.load(model_path)
    output = model(model_input)['embedding'].numpy()
  np.testing.assert_array_equal(output.shape, expected_output_shape)
  logging.info('Model "%s" worked.', model_path)
