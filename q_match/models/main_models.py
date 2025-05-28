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

"""Main Models.

All models defined here are used for main tasks (e.g., classification).

All outputs will have a dictionary with a key of 'main' and nested in here
can be another dictionary of values, which should contain either 'logits' for
classification models or 'output' for regression tasks.

TODO(tmulc): refactor models to make the model call accept the 'training'
argument instead of the model instantiation.
"""
from typing import Sequence

import flax.linen as nn
import jax

from q_match.models import models


class ClassifierTemplate(nn.Module):

  def __call__(self,):
    return {'main': {'logits': None}}


class RegressionTemplate(nn.Module):

  def __call__(self,):
    return {'main': {'output': None}}


class LinearClassifier(nn.Module):
  """Linear classifier over input features with and without the stop gradient.

  The finetuning head will create variables with 'finetune_head' as an ancestor
  and the linear_head will create variables with 'linear_head' as an ancestor,
  both per our convention.

  Output in both cases in the logits.
  """

  training: bool
  num_classes: int

  def setup(self):
    self.finetune_head = nn.Dense(self.num_classes)
    self.linear_head = nn.Dense(self.num_classes)

  def __call__(self, inputs):
    return {
        'main': {
            'finetune_head': {
                'logits': self.finetune_head(inputs)
            },
            'linear_head': {
                'logits': self.linear_head(jax.lax.stop_gradient(inputs))
            }
        }
    }


class BaseMLPClassifier(nn.Module):
  """Base MLP classifier.  Output is class probabilities."""

  layer_sizes: Sequence[int]
  training: bool
  num_classes: int

  def setup(self):
    self.mlp = models.BaseMLP(layer_sizes=self.layer_sizes,
                              training=self.training)
    self.fc = nn.Dense(self.num_classes)

  def __call__(self, inputs,):
    x = self.mlp(inputs)
    logits = self.fc(x)

    return {'main': {'logits': logits, 'probabilities': nn.softmax(logits)}}


# TODO(tmulc): Delete this if it isn't used.
# class ImixMLPClassifier(nn.Module):
#   """iMix body with classification head.
#   """

#   training: bool
#   num_classes: int

#   def setup(self):
#     self.imix_mlp = models.ImixMLP(training=self.training)
#     self.fc = nn.Dense(self.num_classes)

#   def __call__(self, inputs):
#     x = self.imix_mlp(inputs)
#     x = self.fc(x)
#     return {'main': {'logits': x}}
