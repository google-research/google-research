# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Utilities for computing the total variation.

This file contains function for computing the total variation
of hidden layer over the entire training set.

  Typical usage example:

  root_dir = # directory where the dataset is at
  model_config = ModelConfig(...)
  input_fn = data_util.get_input(
      data=model_config.dataset, data_format=model_config.data_format)
  h1_total_variation = compute_total_variation(input_fn, root_dir, 'h1',
                                               model_config)
"""

from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf


IMG_HEIGHT = 32
IMG_WIDTH = 32
IMG_CHANNELS = 3


def compute_total_variation(
    input_fn, root_dir, model_config, layer='inputs', sess=None,
    batchsize=50, dataset_size=50000):
  """Compute the total variation of a hidden layer on all input data.

  Loads a given model from given directory and load the parameters in the given
  scope. Iterates over the entire training dataset and computes the total
  variation of the layer over the entire training set.

  Args:
    input_fn:  function that produces the input and label tensors
    root_dir:  the directory where the dataset is at
    model_config: a ModelConfig object that specifies the model
    layer: name of the hidden layer at which the total variation is computed.
      Only 1 layer at a time due to memory constraints. Available options
      include inputs, h1, h2, and h3.
    sess: optional tensorflow session
    batchsize: batch size with which the margin is computed
    dataset_size: number of data points in the dataset

  Returns:
    A scalar that is the total variation at the specified layer.
  """
  param_path = model_config.get_model_dir_name(root_dir)
  model_fn = model_config.get_model_fn()

  if not sess:
    sess = tf.Session()

  data_format = model_config.data_format
  image_iter, label_iter = input_fn()
  if data_format == 'HWC':
    img_dim = [None, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS]
  else:
    img_dim = [None, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH]
  image = tf.placeholder(tf.float32, shape=img_dim, name='image')
  label = tf.placeholder(
      tf.float32, shape=[None, model_config.num_class], name='label')

  loss_layers = [layer]
  end_points_collection = {}
  _ = model_fn(image, False, perturb_points=loss_layers,
               normalizer_collection=None,
               end_points_collection=end_points_collection)

  layer_activations = [end_points_collection[l] for l in loss_layers]

  # load model parameters
  sess.run(tf.global_variables_initializer())
  model_config.load_parameters(param_path, sess)

  count = 0
  all_activation = []
  while count < dataset_size:
    try:
      count += batchsize
      image_batch, label_batch = sess.run([image_iter, label_iter])
      label_batch = np.reshape(label_batch, [-1, model_config.num_class])
      fd = {image: image_batch, label: label_batch.astype(np.float32)}
      activation = np.squeeze(list(sess.run(layer_activations, feed_dict=fd)))
      all_activation.append(activation)
    except tf.errors.OutOfRangeError:
      print('reached the end of the data')
      break

  all_activation = np.concatenate(all_activation, axis=0)
  response_flat = all_activation.reshape([all_activation.shape[0], -1])
  response_std = np.std(response_flat, axis=0)
  total_variation_unnormalized = (np.sum(response_std ** 2)) ** 0.5
  return total_variation_unnormalized / all_activation.shape[0]
