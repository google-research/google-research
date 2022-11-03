# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Efficient implementation of CKA based on minibatch statistics"""

from absl import logging
import numpy as np

import tensorflow.compat.v2 as tf

tf.enable_v2_behavior()


class MinibatchCKA(tf.keras.metrics.Metric):

  def __init__(self,
               num_layers,
               num_layers2=None,
               across_models=False,
               dtype=tf.float32):
    super(MinibatchCKA, self).__init__()
    if num_layers2 is None:
      num_layers2 = num_layers
    self.hsic_accumulator = self.add_weight(
        'hsic_accumulator',
        shape=(num_layers, num_layers2),
        initializer=tf.keras.initializers.zeros,
        dtype=dtype)
    self.across_models = across_models
    if across_models:
      self.hsic_accumulator_model1 = self.add_weight(
          'hsic_accumulator_model1',
          shape=(num_layers,),
          initializer=tf.keras.initializers.zeros,
          dtype=dtype)
      self.hsic_accumulator_model2 = self.add_weight(
          'hsic_accumulator_model2',
          shape=(num_layers2,),
          initializer=tf.keras.initializers.zeros,
          dtype=dtype)

  def _generate_gram_matrix(self, x):
    """Generate Gram matrix and preprocess to compute unbiased HSIC.

    This formulation of the U-statistic is from Szekely, G. J., & Rizzo, M.
    L. (2014). Partial distance correlation with methods for dissimilarities.
    The Annals of Statistics, 42(6), 2382-2412.

    Args:
      x: A [num_examples, num_features] matrix.

    Returns:
      A [num_examples ** 2] vector.
    """
    x = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(x, x, transpose_b=True)
    n = tf.shape(gram)[0]
    gram = tf.linalg.set_diag(gram, tf.zeros((n,), gram.dtype))
    gram = tf.cast(gram, self.hsic_accumulator.dtype)
    means = tf.reduce_sum(gram, 0) / tf.cast(n - 2, self.hsic_accumulator.dtype)
    means -= tf.reduce_sum(means) / tf.cast(2 * (n - 1),
                                            self.hsic_accumulator.dtype)
    gram -= means[:, None]
    gram -= means[None, :]
    gram = tf.linalg.set_diag(gram, tf.zeros((n,), self.hsic_accumulator.dtype))
    gram = tf.reshape(gram, (-1,))
    return gram

  def update_state(self, activations):
    """Accumulate minibatch HSIC values.

    Args:
      activations: A list of activations for all layers.
    """
    # tf.assert_equal(
    #     tf.shape(self.hsic_accumulator)[0], len(activations),
    #     'Number of activation vectors does not match num_layers.')
    layer_grams = [self._generate_gram_matrix(x) for x in activations]
    layer_grams = tf.stack(layer_grams, 0)
    self.hsic_accumulator.assign_add(
        tf.matmul(layer_grams, layer_grams, transpose_b=True))

  def update_state_across_models(self, activations1, activations2):
    """Accumulate minibatch HSIC values from different models.

    Args:
      activations1: A list of activations for all layers in model 1.
      activations2: A list of activations for all layers in model 2.
    """
    tf.assert_equal(
        tf.shape(self.hsic_accumulator)[0], len(activations1),
        'Number of activation vectors does not match num_layers.')
    tf.assert_equal(
        tf.shape(self.hsic_accumulator)[1], len(activations2),
        'Number of activation vectors does not match num_layers.')
    layer_grams1 = [self._generate_gram_matrix(x) for x in activations1]
    layer_grams1 = tf.stack(layer_grams1, 0)  #(n_layers, n_examples ** 2)
    layer_grams2 = [self._generate_gram_matrix(x) for x in activations2]
    layer_grams2 = tf.stack(layer_grams2, 0)
    self.hsic_accumulator.assign_add(
        tf.matmul(layer_grams1, layer_grams2, transpose_b=True))
    self.hsic_accumulator_model1.assign_add(
        tf.einsum('ij,ij->i', layer_grams1, layer_grams1))
    self.hsic_accumulator_model2.assign_add(
        tf.einsum('ij,ij->i', layer_grams2, layer_grams2))

  def result(self):
    mean_hsic = tf.convert_to_tensor(
        self.hsic_accumulator)  #(num_layers, num_layers2)
    if self.across_models:
      normalization1 = tf.sqrt(
          tf.convert_to_tensor(self.hsic_accumulator_model1))  #(num_layers,)
      normalization2 = tf.sqrt(
          tf.convert_to_tensor(self.hsic_accumulator_model2))  #(num_layers2,)
      mean_hsic /= normalization1[:, None]
      mean_hsic /= normalization2[None, :]
    else:
      normalization = tf.sqrt(tf.linalg.diag_part(mean_hsic))
      mean_hsic /= normalization[:, None]
      mean_hsic /= normalization[None, :]
    return mean_hsic


def test_CKA(n_layers,
             n_layers2,
             activations1,
             activations2,
             cka1=None,
             cka2=None):
  """Test for checking that update_state_across_models() works as intended"""
  if cka1 is None:
    cka1 = MinibatchCKA(n_layers, n_layers2, across_models=True)
  if cka2 is None:
    cka2 = MinibatchCKA(n_layers + n_layers2)

  cka1.update_state_across_models(activations1, activations2)
  cka1_result = cka1.result().numpy()

  combined_activations = activations1
  combined_activations.extend(activations2)
  cka2.update_state(combined_activations)
  cka2_result = cka2.result().numpy()[:n_layers, -n_layers2:]
  assert (np.max(np.abs(cka2_result - cka1_result)) < 1e-5)
