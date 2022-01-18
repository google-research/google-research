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
"""Factory to build video classification model."""
import collections
import functools
import logging
from typing import Any, Dict, Mapping, Text, Union, Optional

import tensorflow as tf

from vatt.configs import factory as configs_factory
from vatt.configs import video as video_config
from vatt.modeling.backbones.video import i3d
from vatt.modeling.backbones.video import vitx3d


def get_shape(x):
  """Deal with dynamic shape in tensorflow cleanly."""
  static = x.shape.as_list()
  dynamic = tf.shape(x)
  return [dynamic[i] if s is None else s for i, s in enumerate(static)]


class PredictionAggregator(tf.keras.layers.Layer):
  """Aggregates test predictions."""

  def __init__(self,
               num_test_clips = 1,
               name = 'aggregate_clips'):

    super(PredictionAggregator, self).__init__(name=name)
    self._num_test_clips = num_test_clips

  def call(self,
           inputs,
           training = None):
    if training or self._num_test_clips == 1:
      return inputs

    else:
      def aggregate(inputs):
        d_features = get_shape(inputs)[-1]
        return tf.reduce_mean(
            tf.reshape(inputs,
                       [-1, self._num_test_clips, d_features]),
            axis=1,
            )

      return tf.nest.map_structure(aggregate, inputs)


class VideoModel(tf.keras.Model):
  """Constructs Video model with (potential) head."""

  def __init__(self,
               base_model,
               params,
               pred_aggregator = None):
    """VideoModel."""

    super(VideoModel, self).__init__(name='video_module')
    self._model_name = params.name
    self._freeze_backbone = params.freeze_backbone
    self._dropout_rate = params.cls_dropout_rate
    self._final_endpoint = params.final_endpoint
    self._num_classes = params.num_classes
    self._ops = collections.OrderedDict()

    base_kwargs = params.as_dict()
    if 'backbone_config' in base_kwargs:
      base_kwargs['backbone_config'] = params.backbone_config

    self._base = base_model(**base_kwargs)

    if self._freeze_backbone:
      self._base.trainable = False

    if self._num_classes is not None:
      self._ops['dropout'] = tf.keras.layers.Dropout(rate=self._dropout_rate)
      cls_name = 'classification/weights'
      self._ops['cls'] = tf.keras.layers.Dense(
          self._num_classes,
          kernel_initializer='glorot_normal',
          name=cls_name,
          )
      pred_name = 'classification/probabilities'
      self._ops['softmax'] = functools.partial(tf.nn.softmax, name=pred_name)
      self._loss_object = tf.keras.losses.CategoricalCrossentropy(
          reduction=tf.keras.losses.Reduction.NONE
          )

    self.pred_aggregator = pred_aggregator

  def loss_fn(self,
              labels,
              outputs,
              replicator):

    del replicator
    loss = self._loss_object(labels['one_hot'], outputs['probabilities'])
    loss = tf.reduce_mean(loss)

    losses = {'model_loss': loss}
    l2_loss = tf.reduce_sum(self.losses) / 2
    total_loss = losses['model_loss'] + tf.cast(l2_loss,
                                                losses['model_loss'].dtype)

    losses.update({'regularization_loss': l2_loss,
                   'total_loss': total_loss})

    return losses

  def call(self,
           inputs,
           training = None):
    """Call the layer.

    Args:
      inputs: input tensors of different modalities. E.g., RGB, optical flow.
      training: True for in the training mode.

    Returns:
      output_dict: a dict of model outputs, including one of the features,
      logits and probabilities, depending on the configs
    """
    if isinstance(inputs, dict):
      data = inputs['images']
    else:
      data = inputs

    # for dropout and batch_norm. Especially for fuse logits layers.
    features_pooled, end_points = self._base(data, training=training)
    features = end_points[self._final_endpoint]

    if self._freeze_backbone:
      features = tf.stop_gradient(features)
      features_pooled = tf.stop_gradient(features_pooled)

    outputs = {'features': features,
               'features_pooled': features_pooled}

    if self._num_classes is None:
      return outputs

    features_pooled = self._ops['dropout'](features_pooled, training)
    logits = self._ops['cls'](features_pooled)
    if self.pred_aggregator is not None:
      logits = self.pred_aggregator(logits, training)
    probabilities = self._ops['softmax'](logits)

    outputs = {
        'logits': logits,
        'probabilities': probabilities
    }

    return outputs


def build_model(
    params = None,
    override_params = None,
    backbone = None,
    mode = 'embedding',
    ):
  """Build model by name."""
  if params is None:
    assert backbone is not None, 'either params or backbone should be specified'
    params = configs_factory.build_model_configs(backbone)

  if override_params is not None:
    params.override(override_params)

  model_name = params.name
  if model_name.startswith('i3d'):
    base_model = i3d.InceptionI3D
  elif model_name.startswith('vit'):
    base_model = vitx3d.ViTx3D
  else:
    raise ValueError('Unknown backbone {!r}'.format(model_name))

  if mode == 'predict':
    pred_aggregator = PredictionAggregator(
        num_test_clips=params.num_test_samples
        )
  else:
    pred_aggregator = None

  model = VideoModel(
      base_model=base_model,
      params=params,
      pred_aggregator=pred_aggregator,
      )

  logging.info('Video model %s created successfully.', params.name)

  return model
