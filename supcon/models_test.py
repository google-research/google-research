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

"""Tests for supcon.models."""

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf

from supcon import enums
from supcon import models


class ResNetTest(tf.test.TestCase, parameterized.TestCase):

  def testInvalidArchitecture(self):
    with self.assertRaisesRegex(ValueError, 'Architecture should be one of'):
      _ = models.ContrastiveModel(
          architecture='skynet',
          encoder_kwargs={'depth': 18},
          projection_head_kwargs={'feature_dims': (128,)},
          classification_head_kwargs={'num_classes': 10})

  def testResNetV1(self):
    inputs = tf.random_uniform(
        shape=(2, 224, 224, 3), minval=-1., maxval=1., dtype=tf.float32)
    model = models.ContrastiveModel(
        architecture=enums.EncoderArchitecture.RESNET_V1,
        encoder_kwargs={'depth': 18},
        projection_head_kwargs={'feature_dims': (128,)},
        classification_head_kwargs={'num_classes': 10})
    embedding, normalized_embedding, projection, classification = model(
        inputs, training=True)
    self.assertListEqual([2, 512], embedding.shape.as_list())
    self.assertListEqual([2, 512], normalized_embedding.shape.as_list())
    self.assertListEqual([2, 128], projection.shape.as_list())
    self.assertListEqual([2, 10], classification.shape.as_list())

  def testResNext(self):
    inputs = tf.random_uniform(
        shape=(2, 224, 224, 3), minval=-1., maxval=1., dtype=tf.float32)
    model = models.ContrastiveModel(
        architecture=enums.EncoderArchitecture.RESNEXT,
        encoder_kwargs={'depth': 50},
        projection_head_kwargs={'feature_dims': (128,)},
        classification_head_kwargs={'num_classes': 10})
    embedding, normalized_embedding, projection, classification = model(
        inputs, training=True)
    self.assertListEqual([2, 2048], embedding.shape.as_list())
    self.assertListEqual([2, 2048], normalized_embedding.shape.as_list())
    self.assertListEqual([2, 128], projection.shape.as_list())
    self.assertListEqual([2, 10], classification.shape.as_list())

  def testStopGradientBeforeClassificationHeadFalse(self):
    inputs = tf.random_uniform(
        shape=(2, 224, 224, 3), minval=-1., maxval=1., dtype=tf.float32)
    model = models.ContrastiveModel(
        architecture=enums.EncoderArchitecture.RESNET_V1,
        stop_gradient_before_classification_head=False,
        encoder_kwargs={'depth': 18},
        projection_head_kwargs={'feature_dims': (128,)},
        classification_head_kwargs={'num_classes': 10})
    embedding, _, _, classification = model(inputs, training=True)
    input_grads = tf.gradients(classification, inputs)[0]
    embedding_grads = tf.gradients(classification, embedding)[0]
    self.assertIsNotNone(input_grads)
    self.assertIsNotNone(embedding_grads)
    with self.cached_session() as sess:
      sess.run(tf.global_variables_initializer())
      input_grads_np, embedding_grads_np = sess.run(
          (input_grads, embedding_grads))
      # Make sure that there are no NaNs
      self.assertFalse(np.isnan(input_grads_np).any())
      self.assertFalse(np.isnan(embedding_grads_np).any())

  def testStopGradientBeforeClassificationHeadTrue(self):
    inputs = tf.random_uniform(
        shape=(2, 224, 224, 3), minval=-1., maxval=1., dtype=tf.float32)
    model = models.ContrastiveModel(
        architecture=enums.EncoderArchitecture.RESNET_V1,
        stop_gradient_before_classification_head=True,
        encoder_kwargs={'depth': 18},
        projection_head_kwargs={'feature_dims': (128,)},
        classification_head_kwargs={'num_classes': 10})
    embedding, _, _, classification = model(inputs, training=True)
    input_grads = tf.gradients(classification, inputs)[0]
    embedding_grads = tf.gradients(classification, embedding)[0]
    self.assertIsNone(input_grads)
    self.assertIsNone(embedding_grads)

  def testNormalizeClassificationHeadInputsTrue(self):
    inputs = tf.random_uniform(
        shape=(2, 224, 224, 3), minval=-1., maxval=1., dtype=tf.float32)
    model = models.ContrastiveModel(
        architecture=enums.EncoderArchitecture.RESNET_V1,
        stop_gradient_before_classification_head=False,
        normalize_classification_head_input=True,
        encoder_kwargs={'depth': 18},
        projection_head_kwargs={'feature_dims': (128,)},
        classification_head_kwargs={'num_classes': 10})
    embedding, normalized_embedding, _, classification = model(
        inputs, training=True)
    normalized_embedding_grads = tf.gradients(classification,
                                              normalized_embedding)[0]
    embedding_grads = tf.gradients(classification, embedding)[0]
    self.assertIsNotNone(normalized_embedding_grads)
    self.assertIsNotNone(embedding_grads)
    with self.cached_session() as sess:
      sess.run(tf.global_variables_initializer())
      normalized_embedding_grads_np, embedding_grads_np = sess.run(
          (normalized_embedding_grads, embedding_grads))
      # Make sure that there are no NaNs
      self.assertFalse(np.isnan(normalized_embedding_grads_np).any())
      self.assertFalse(np.isnan(embedding_grads_np).any())

  def testNormalizeClassificationHeadInputsFalse(self):
    inputs = tf.random_uniform(
        shape=(2, 224, 224, 3), minval=-1., maxval=1., dtype=tf.float32)
    model = models.ContrastiveModel(
        architecture=enums.EncoderArchitecture.RESNET_V1,
        stop_gradient_before_classification_head=False,
        normalize_classification_head_input=False,
        encoder_kwargs={'depth': 18},
        projection_head_kwargs={'feature_dims': (128,)},
        classification_head_kwargs={'num_classes': 10})
    embedding, normalized_embedding, _, classification = model(
        inputs, training=True)
    normalized_embedding_grads = tf.gradients(classification,
                                              normalized_embedding)[0]
    embedding_grads = tf.gradients(classification, embedding)[0]
    self.assertIsNone(normalized_embedding_grads)
    self.assertIsNotNone(embedding_grads)
    with self.cached_session() as sess:
      sess.run(tf.global_variables_initializer())
      embedding_grads_np = sess.run(embedding_grads)
      # Make sure that there are no NaNs
      self.assertFalse(np.isnan(embedding_grads_np).any())

  def testNormalizeProjectionHeadInputsTrue(self):
    inputs = tf.random_uniform(
        shape=(2, 224, 224, 3), minval=-1., maxval=1., dtype=tf.float32)
    model = models.ContrastiveModel(
        architecture=enums.EncoderArchitecture.RESNET_V1,
        normalize_projection_head_input=True,
        encoder_kwargs={'depth': 18},
        projection_head_kwargs={'feature_dims': (128,)},
        classification_head_kwargs={'num_classes': 10})
    embedding, normalized_embedding, projection, _ = model(
        inputs, training=True)
    normalized_embedding_grads = tf.gradients(projection,
                                              normalized_embedding)[0]
    embedding_grads = tf.gradients(projection, embedding)[0]
    self.assertIsNotNone(normalized_embedding_grads)
    self.assertIsNotNone(embedding_grads)
    with self.cached_session() as sess:
      sess.run(tf.global_variables_initializer())
      normalized_embedding_grads_np, embedding_grads_np = sess.run(
          (normalized_embedding_grads, embedding_grads))
      # Make sure that there are no NaNs
      self.assertFalse(np.isnan(normalized_embedding_grads_np).any())
      self.assertFalse(np.isnan(embedding_grads_np).any())

  def testNormalizeProjectionHeadInputsFalse(self):
    inputs = tf.random_uniform(
        shape=(2, 224, 224, 3), minval=-1., maxval=1., dtype=tf.float32)
    model = models.ContrastiveModel(
        architecture=enums.EncoderArchitecture.RESNET_V1,
        normalize_projection_head_input=False,
        encoder_kwargs={'depth': 18},
        projection_head_kwargs={'feature_dims': (128,)},
        classification_head_kwargs={'num_classes': 10})
    embedding, normalized_embedding, projection, _ = model(
        inputs, training=True)
    normalized_embedding_grads = tf.gradients(projection,
                                              normalized_embedding)[0]
    embedding_grads = tf.gradients(projection, embedding)[0]
    self.assertIsNone(normalized_embedding_grads)
    self.assertIsNotNone(embedding_grads)
    with self.cached_session() as sess:
      sess.run(tf.global_variables_initializer())
      embedding_grads_np = sess.run(embedding_grads)
      # Make sure that there are no NaNs
      self.assertFalse(np.isnan(embedding_grads_np).any())


if __name__ == '__main__':
  tf.test.main()
