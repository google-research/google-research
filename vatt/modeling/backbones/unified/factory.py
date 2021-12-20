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

# Lint as: python3
"""Factory to build video classification model."""
import collections
import functools
import logging
from typing import Any, Dict, Text, Optional

import tensorflow as tf

from vatt.configs import factory as configs_factory
from vatt.configs import unified as unified_config
from vatt.modeling.backbones.unified import uvatt


class UnifiedModule(tf.keras.Model):
  """Constructs Unified model with (potential) classification head.

  This class constructs the video backbone and either returns the overall
  output or adds a fc layer on top of a user-specified modalitiy outputs
  and returns the logtis/probs.
  """

  def __init__(self,
               base_model,
               params):
    """Unified Backbone Model.

    Args:
      base_model: the base model.
      params: Hyperparameters of the model.
    """
    super(UnifiedModule, self).__init__(name="unified_module")
    self._model_name = params.name
    self._dropout_rate = params.cls_dropout_rate
    self._modality = params.modality
    self._num_classes = params.num_classes
    self._ops = collections.OrderedDict()

    self.unified_transformer = base_model(**params.as_dict())

    if self._num_classes is not None:
      self._ops["dropout"] = tf.keras.layers.Dropout(rate=self._dropout_rate)
      cls_name = "classification/weights"
      self._ops["cls"] = tf.keras.layers.Dense(self._num_classes, name=cls_name)
      pred_name = "classification/probabilities"
      self._ops["softmax"] = functools.partial(tf.nn.softmax, name=pred_name)

  def call(self,
           video,
           audio,
           word_ids,
           txt_attn_mask,
           training = None):
    """Call the layer.

    Args:
      video: raw video frames
      audio: raw audio samples
      word_ids: raw text ids
      txt_attn_mask: padding mask for text ids
      training: True for in the training mode.

    Returns:
      output_dict: a dict of model outputs,
    """
    base_inputs = {
        "video": {"data": video},
        "audio": {"data": audio},
        "text": {"data": word_ids,
                 "attention_mask": txt_attn_mask},
    }

    outputs = self.unified_transformer(base_inputs, training=training)

    if self._num_classes is None:
      return outputs

    features_pooled = outputs[self._modality]["features_pooled"]
    features_pooled = self._ops["dropout"](features_pooled, training)
    logits = self._ops["cls"](features_pooled)
    probabilities = self._ops["softmax"](logits)

    outputs = {
        "logits": logits,
        "probabilities": probabilities
    }

    return outputs


def build_model(
    backbone,
    params = None,
    override_params = None,
    ):
  """Build model by name."""
  if params is None:
    assert backbone is not None, "either params or backbone should be specified"
    params = configs_factory.build_model_configs(backbone)

  if override_params is not None:
    params.override(override_params)

  if backbone.startswith("ut"):
    base_model = uvatt.UniversalVATT
  else:
    raise ValueError("Unknown backbone {!r}".format(backbone))

  model = UnifiedModule(
      base_model=base_model,
      params=params,
      )

  logging.info("Unified backbone %s created successfully.", params.name)

  return model
