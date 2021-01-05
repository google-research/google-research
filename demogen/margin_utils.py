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

"""Utilities for computing the margin of a model specified by a model config.

This file contains function for computing the linear approximation of
the distance to the decision boundary at specified hidden layer over
the training dataset.

  Typical usage example:

  root_dir = # directory where the dataset is at
  model_config = ModelConfig(...)
  input_fn = data_util.get_input(
      data=model_config.dataset, data_format=model_config.data_format)
  margins = compute_margin(input_fn, root_dir, model_config)
  input_margins = margins['inputs']
  h1_margins = margins['h1']
  h2_margins = margins['h2']
  h3_margins = margins['h3']
"""

from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf


IMG_HEIGHT = 32
IMG_WIDTH = 32
IMG_CHANNELS = 3


def compute_margin(
    input_fn, root_dir, model_config, sess=None,
    batchsize=50, dataset_size=50000):
  """Compute the margins of a model on all input data.

  Loads a given model from given directory and load the parameters in the given
  scope. Iterates over the entire training dataset and computes the upper bound
  on the margin by doing a line search.

  Args:
    input_fn:  function that produces the input and label tensors
    root_dir:  the directory containing the dataset
    model_config: a ModelConfig object that specifies the model
    sess: optional tensorflow session
    batchsize: batch size with which the margin is computed
    dataset_size: number of data points in the dataset

  Returns:
    A dictionary that maps each layer's name to the margins at that layer
    over the entire training set.
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

  loss_layers = ['inputs', 'h1', 'h2', 'h3']
  end_points_collection = {}
  logits = model_fn(image, is_training=False, perturb_points=loss_layers,
                    normalizer_collection=None,
                    end_points_collection=end_points_collection)

  # set up the graph for computing margin
  layer_activations = [end_points_collection[l] for l in loss_layers]
  layer_margins = margin(logits, label, layer_activations)

  # load model parameters
  sess.run(tf.global_variables_initializer())
  model_config.load_parameters(param_path, sess)

  count = 0
  margin_values = []
  while count < dataset_size:
    try:
      count += batchsize
      image_batch, label_batch = sess.run([image_iter, label_iter])
      label_batch = np.reshape(label_batch, [-1, model_config.num_class])
      fd = {image: image_batch, label: label_batch.astype(np.float32)}
      batch_margin = np.squeeze(list(sess.run(layer_margins, feed_dict=fd)))
      margin_values.append(batch_margin)
    except tf.errors.OutOfRangeError:
      print('reached the end of the data')
      break

  margin_values = np.concatenate(margin_values, axis=1)
  margin_values_map = {}
  for ln, lm in zip(loss_layers, margin_values):
    margin_values_map[ln] = lm

  return margin_values_map


def margin(logits, labels, layer_activations, dist_norm=2, epsilon=1e-6):
  """Build graphs for margins at hidden layers.

  Args:
    logits:  logits of the model
    labels:  ground truth label of the input
    layer_activations:  List of tensors representing the activations at the
        hidden layers for which the margins are computed.
    dist_norm: type of norm the margin is computed with
    epsilon: epsilon for numerical stability of division

  Returns:
    A list that contains linear margin approximation at each layer in layers.
  """
  num_classes = logits.get_shape().as_list()[1]
  labels_int = tf.argmax(labels, axis=1, output_type=tf.int32)
  bs_lin = tf.range(0, tf.shape(logits)[0])
  indices_true = tf.stop_gradient(tf.transpose(tf.stack([bs_lin, labels_int])))
  values_true = tf.gather_nd(logits, indices_true)

  values, indices = tf.nn.top_k(logits, k=2)
  indices = tf.stop_gradient(indices)
  # indicator if the highest class matches the ground truth
  true_match_float = tf.cast(
      tf.equal(indices[:, 0], labels_int), dtype=tf.float32)
  # if zero match the true class then we take the next class, otherwise we use
  # the highest class
  values_c = (values[:, 1] * true_match_float +
              values[:, 0] * (1 - true_match_float))
  true_match = tf.cast(true_match_float, dtype=tf.int32)
  indices_c = indices[:, 1] * true_match + indices[:, 0] * (1 - true_match)
  grad_ys = tf.one_hot(labels_int, num_classes)
  grad_ys -= tf.one_hot(indices_c, num_classes)
  grad_ys = tf.stop_gradient(grad_ys)
  # numerator of the distance
  numerator = values_true - values_c
  # compute gradient wrt all hidden layers at once
  g_wrt_all_layers = tf.gradients(logits, layer_activations, grad_ys)
  layer_dims = [l.shape.rank for l in layer_activations]
  g_norm_all_layers = []
  for i, g in enumerate(g_wrt_all_layers):
    if dist_norm == 0:  # l infinity
      g_norm_all_layers.append(epsilon + tf.reduce_max(
          tf.abs(g), axis=np.arange(1, layer_dims[i])))
    elif dist_norm == 1:
      g_norm_all_layers.append(epsilon + tf.reduce_sum(
          tf.abs(g), axis=np.arange(1, layer_dims[i])))
    elif dist_norm == 2:
      g_norm_all_layers.append(tf.sqrt(epsilon + tf.reduce_sum(
          g*g, axis=np.arange(1, layer_dims[i]))))
    else:
      raise ValueError('only norms supported are 1, 2, and infinity')
  g_norm_all_layers = [tf.stop_gradient(g) for g in g_norm_all_layers]
  dist_to_db_all_layers = [numerator / gn for gn in g_norm_all_layers]
  return dist_to_db_all_layers
