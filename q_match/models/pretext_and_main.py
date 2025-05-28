# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Models here have both prext outputs and main outputs for downstream tasks.


In the setup method, the pretext model should be named 'pretext', the
finetuning head should be named 'finetune_head', and the linear classifier which
has a stop gradient defined on its inputs should be name 'linear_head'.
This will ensure that variables defined in each have the correct ancesestors
in the variable dict.

Furthermore, the 'finetune_head' and 'linear_head' should both be contained
within a 'prediction_head'.
"""
from typing import Optional

import flax.linen as nn

from q_match.models import main_models
from q_match.models import pretext_models


class Template(nn.Module):

  def setup(self):
    self.pretext = Ellipsis
    self.fine_tune = Ellipsis
    self.linear_classifier = Ellipsis

  def __call__(self, inputs):
    return {'pretext': None, 'main': None}


class ImixClassifier(nn.Module):
  """The Pretext model with a linear classifier for downstream predictions.

  Output will be the pretext outputs, (encoded value, reconstruction guess,
  mask guess, mask) and the logits.

  Uses the iMix architecture for the prext network.

  The input to the linear classifier is the vime encoded value.

  Attributes:
    training: If training parameters or running inference.
    num_classes: Number of classes for downstream classification.
    algorithm: The algorithm being used for pretext training.
  """

  training: bool
  num_classes: int
  algorithm: Optional[str] = None

  def setup(self,):
    self.pretext = pretext_models.Pretext(training=self.training,
                                          algorithm=self.algorithm)
    self.prediction_head = main_models.LinearClassifier(
        training=self.training, num_classes=self.num_classes)
    self.prediction_head_over_features = main_models.LinearClassifier(
        training=self.training, num_classes=self.num_classes)
    self.initial_lc_over_inputs_batch_norm_layer = nn.BatchNorm(
        use_running_average=True)

  def __call__(self, inputs, linear_over_features=False):
    """Forward pass.

    Args:
      inputs: the input data
      linear_over_features: Whether to compute the linear head over the input
        features
    Returns:
      Dict with keys 'pretext', 'main'
    """
    pretext_outputs = self.pretext(inputs,)
    z = pretext_outputs['pretext']['encoded']
    main_over_inputs = self.prediction_head_over_features(
        self.initial_lc_over_inputs_batch_norm_layer(inputs))
    main_outputs = self.prediction_head(z)
    if linear_over_features:
      main_outputs = main_over_inputs

    return {**pretext_outputs, **main_outputs}


class ResnetClassifier(nn.Module):
  """Classifier using the resnet pretext architecure."""

  training: bool
  num_classes: int
  algorithm: Optional[str] = None

  def setup(self,):
    self.pretext = pretext_models.ResnetPretext(
        training=self.training, algorithm=self.algorithm)
    self.prediction_head = main_models.LinearClassifier(
        training=self.training, num_classes=self.num_classes)
    self.prediction_head_over_features = main_models.LinearClassifier(
        training=self.training, num_classes=self.num_classes)
    self.initial_lc_over_inputs_batch_norm_layer = nn.BatchNorm(
        use_running_average=True)

  def __call__(self, inputs, linear_over_features=False):
    """Forward pass.

    Args:
      inputs: the input data
      linear_over_features: Whether to compute the linear head over the input
        features
    Returns:
      Dict with keys 'pretext', 'main'
    """
    pretext_outputs = self.pretext(inputs,)
    z = pretext_outputs['pretext']['encoded']
    main_over_inputs = self.prediction_head_over_features(
        self.initial_lc_over_inputs_batch_norm_layer(inputs))
    main_outputs = self.prediction_head(z)

    if linear_over_features:
      main_outputs = main_over_inputs

    return {**pretext_outputs, **main_outputs}
