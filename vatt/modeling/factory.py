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

"""Main factory for building vatt Models."""

import logging
from typing import Any, Dict, Mapping, Optional, Text

import tensorflow as tf

from vatt.configs import factory as configs_factory
from vatt.configs import multimodal as vatt_config
from vatt.modeling.backbones import multimodal as vatt_models
from vatt.modeling.heads import factory as head_factory
from vatt.utils.train import objectives


class MMRLModel(tf.keras.Model):
  """Constructs VATT model with (potential) heads.

  This class constructs all three modalities and returns the video, audio, and
  text embeddings. This class also contains a general loss_fn attribute that
  connects generic controllers to vatt-specific losses.
  """

  def __init__(self,
               base_module,
               head_module,
               params):
    """MMRLModel.

    Args:
      base_module: the base module containing video+audio+text layers.
      head_module: the head module containing projection heads.
      params: Hyperparameters of the model.
    """
    super(MMRLModel, self).__init__(name=params.model_name)
    self._params = params

    self._base_layer = base_module(**params.backbone_config.as_dict())
    self._head_layer = head_module
    self._loss_lib = self._build_losses(params)

  def _build_losses(self, params):
    all_losses = {'bridge_losses': []}

    for bridge_params in params.loss_config.bridge:
      all_losses['bridge_losses'].append(
          objectives.build_loss(bridge_params))

    return all_losses

  def loss_fn(self,
              labels,
              outputs,
              replicator):

    losses = {}

    def maybe_initialize(losses, metric_name):
      if metric_name not in losses:
        losses[metric_name] = tf.convert_to_tensor(0., dtype=tf.float32)
      return losses

    for bridge_loss_fn in self._loss_lib['bridge_losses']:
      predictions = outputs['head_stack']['bridge']
      loss, metrics_to_log = bridge_loss_fn(labels,
                                            predictions,
                                            True,
                                            replicator)
      losses = maybe_initialize(losses, 'bridge_losses/total_loss')
      losses['bridge_losses/total_loss'] += (
          loss * bridge_loss_fn.loss_weight
          )
      for log_name in metrics_to_log:
        metric_name = '/'.join(['bridge_losses',
                                bridge_loss_fn.name,
                                log_name])
        losses[metric_name] = metrics_to_log[log_name]

    losses['model_loss'] = losses['bridge_losses/total_loss']
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
    video_inputs = inputs['video']
    audio_inputs = inputs['audio']
    text_inputs = inputs['text']

    features = self._base_layer(
        video=video_inputs,
        audio=audio_inputs,
        word_ids=text_inputs,
        training=training,
        )
    features['current_step'] = inputs.get('current_step', -1)
    heads_outputs = self._head_layer(inputs=features,
                                     training=training,
                                     mask=None)

    del features['current_step']
    outputs = features
    outputs['head_stack'] = heads_outputs

    return outputs


def build_model(
    params = None,
    override_params = None,
    model_arch = None,
    ):
  """Build model by name."""
  if params is None:
    assert model_arch is not None, ('either params or model_arch should be '
                                    'specified')
    params = configs_factory.build_model_configs(model_arch)

  if override_params is not None:
    params.override(override_params)

  backbone_name = params.backbone_config.name
  if backbone_name.startswith('unified_backbone'):
    base_module = vatt_models.UnifiedFusion
  else:
    base_module = vatt_models.AudioTextVideoFusion

  head_module = head_factory.build_model(params=params.head_config)

  model = MMRLModel(
      base_module=base_module,
      head_module=head_module,
      params=params,
  )

  logging.info('Entire MM model %s created successfully.', params.model_name)

  return model
