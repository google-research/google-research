# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Utilities for model inference.

Assumptions:
* All residue encoding is one hot for all models.
* Logistic and CNN models use fixed length sequence encodings
* RNN model uses variable length sequence encodings
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import os

import dataset_utils
from ..model_training import cnn
from ..model_training import lr
from ..model_training import rnn
import tensorflow as tf


def load_hparams(model_dir):
  """Load serialized hyperparameters from a model dir.

  Note: HParam backfill assumptions due to incomplete hparam metadata originally
  being emitted during model training (used for inferring the type of model from
  hparams stored in a model_dir):
  * num_units hparams only specified for rnn models
  * conv_depth hparam only specified for cnn models

  Args:
    model_dir: (str) Path to a tf.Estimator model dir.
  Returns:
    (dict) Model hyperparameters.
  """
  with open(os.path.join(model_dir, 'hparams.json')) as f:
    hparams = json.load(f)

  if 'num_units' in hparams:
    hparams['model'] = 'rnn'
    hparams['seq_encoder'] = 'varlen-id'
  elif 'conv_depth' in hparams:
    hparams['model'] = 'cnn'
    hparams['seq_encoder'] = 'fixedlen-id'
  else:
    hparams['model'] = 'logistic'
    hparams['seq_encoder'] = 'fixedlen-id'

  return hparams


def load_model(model_dir, inference_batch_size=1024):
  """Loads serialized tf.Estimator model from a dir for inference.

  Args:
    model_dir: (str) Path to a tf.Estimator model dir.
    inference_batch_size: (int) Batch size to use when calling model.predict().
  Returns:
    (tf.Estimator, dict) A tuple of (model, hparams).
  """
  hparams = load_hparams(model_dir)
  hparams['batch_size'] = inference_batch_size

  model_fn = None
  if hparams['model'] == 'rnn':
    model_fn = rnn.rnn_model_fn
  elif hparams['model'] == 'cnn':
    cnn_refs = {}  # These are just for debugging and introspection
    model_fn = functools.partial(cnn.cnn_model_fn, refs=cnn_refs)
  elif hparams['model'] == 'logistic':
    model_fn = lr.logistic_regression_model_fn
  else:
    raise ValueError('Model type "%s" is not supported' % hparams['model'])

  model = tf.estimator.Estimator(
      model_fn=model_fn,
      params=hparams,
      model_dir=model_dir,
  )

  return model, hparams


def load_dataset(
    dataset_name=None,
    model_type=None,
    dataset_dirpath='/path/to/datasets',
    dataset_filepath=None,
    batch_size=1024):
  """Loads a TFRecord dataset and wraps it in a tf.Estimator model input_fn.

  Note: assumes that all residue encoding is one hot.

  Args:
    dataset_name: The name of the dataset to load; the full path that will be
      loaded will be "dataset_dirpath/dataset_name.tfrecord".
    model_type: (str) The type of model for which to construct the input_fn;
      e.g., 'rnn' or 'cnn'.
    dataset_dirpath: (str) The directory from which to load the dataset.
    dataset_filepath: (str) The full filepath to a TFRecords file to load;
      if specified, overrides dataset_name and dataset_dirpath.
    batch_size: (int) The batch size to use for the model input_fn.
  Returns:
    (func) A tf.Estimator input_fn.
  """
  if dataset_filepath is None:
    dataset_filepath = os.path.join(
        dataset_dirpath, '%s.tfrecord' % dataset_name)

  def wrapped_input_fn():
    """A tf.Estimator model input_fn.

    Combines the graph operations for loading TFRecords and preprocessing the
    tf.Examples into a single input_fn, so that all tf operations are
    instantiated in the same graph whenever input_fn is invoked.

    Returns:
      (features, labels) Tensors representing batches of features and labels.
    """
    dataset = dataset_utils.read_tfrecord_dataset(dataset_filepath)

    # Note that the datasets below are loaded for a single epoch because they
    # are used for model inference in this case (versus training) and we only
    # want to make a single pass over a dataset for inference.
    input_fn = None
    if model_type == 'rnn':
      input_fn = dataset_utils.as_estimator_input_fn(
          dataset.map(lambda ex: dataset_utils.encode_varlen(  # pylint: disable=g-long-lambda
              ex, dataset_utils.ONEHOT_VARLEN_SEQUENCE_ENCODER)),
          batch_size,
          num_epochs=1,
          shuffle=False,
          drop_partial_batches=True,
          sequence_element_encoding_shape=20,
      )
    elif model_type in ('cnn', 'logistic'):
      input_fn = dataset_utils.as_estimator_input_fn(
          dataset.map(lambda ex: dataset_utils.encode_fixedlen(  # pylint: disable=g-long-lambda
              ex, dataset_utils.ONEHOT_FIXEDLEN_MUTATION_ENCODER)),
          batch_size,
          num_epochs=1,
          shuffle=False,
          drop_partial_batches=True,
      )
    else:
      raise ValueError('Model type "%s" is not supported' % model_type)
    return input_fn()

  return wrapped_input_fn
