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

# Lint as: python3
"""The contrastive model."""

import tensorflow.compat.v1 as tf
from supcon import classification_head
from supcon import enums
from supcon import projection_head
from supcon import resnet


class ContrastiveModel(tf.layers.Layer):
  """A model suitable for contrastive training with different backbone networks.

  Attributes:
    architecture: An enums.EncoderArchitecture. The type of the architecture to
      use for the encoder.
    normalize_projection_head_input: Whether the encoder output that is the
      input to the projection head should be normalized.
    normalize_classification_head_input: Whether the encoder output that is the
      input to the classification head should be normalized.
    jointly_train_classification_head: Whether the classification head is
      trained simultaneously with the encoder. If false, a stop_gradient is
      added between the classification head and the encoder.
    encoder_kwargs: Keyword arguments that are passed on to the constructor of
      the encoder. The specific encoder implementation is determined by
      `architecture`.
    projection_head_kwargs: Keyword arguments that are passed on to the
      constructor of the projection head. These are the arguments to
      `projection_head.ProjectionHead`.
    classification_head_kwargs: Keyword arguments that are passed on to the
      constructor of the classification head. These are the arguments to
      `classification_head.ClassificationHead`.
    name: A name for this object.
  """

  def __init__(self,
               architecture=enums.EncoderArchitecture.RESNET_V1,
               normalize_projection_head_input=True,
               normalize_classification_head_input=True,
               stop_gradient_before_projection_head=False,
               stop_gradient_before_classification_head=True,
               encoder_kwargs=None,
               projection_head_kwargs=None,
               classification_head_kwargs=None,
               name='ContrastiveModel',
               **kwargs):
    super(ContrastiveModel, self).__init__(name=name, **kwargs)

    self.normalize_projection_head_input = normalize_projection_head_input
    self.normalize_classification_head_input = (
        normalize_classification_head_input)
    self.stop_gradient_before_projection_head = (
        stop_gradient_before_projection_head)
    self.stop_gradient_before_classification_head = (
        stop_gradient_before_classification_head)

    encoder_fns = {
        enums.EncoderArchitecture.RESNET_V1: resnet.ResNetV1,
        enums.EncoderArchitecture.RESNEXT: resnet.ResNext,
    }
    if architecture not in encoder_fns:
      raise ValueError(f'Architecture should be one of {encoder_fns.keys()}, '
                       f'found: {architecture}.')
    encoder_fn = encoder_fns[architecture]

    assert encoder_kwargs is not None
    projection_head_kwargs = projection_head_kwargs or {}
    classification_head_kwargs = classification_head_kwargs or {}

    self.encoder = encoder_fn(name='Encoder', **encoder_kwargs)
    self.projection_head = projection_head.ProjectionHead(
        **projection_head_kwargs)
    self.classification_head = classification_head.ClassificationHead(
        **classification_head_kwargs)

  def call(self, inputs, training):
    embedding = self.encoder(inputs, training)
    normalized_embedding = tf.nn.l2_normalize(embedding, axis=1)

    projection_input = (
        normalized_embedding
        if self.normalize_projection_head_input else embedding)
    if self.stop_gradient_before_projection_head:
      projection_input = tf.stop_gradient(projection_input)
    projection = self.projection_head(projection_input, training)

    classification_input = (
        normalized_embedding
        if self.normalize_classification_head_input else embedding)
    if self.stop_gradient_before_classification_head:
      classification_input = tf.stop_gradient(classification_input)
    classification = self.classification_head(classification_input, training)

    return embedding, normalized_embedding, projection, classification
