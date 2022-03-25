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

"""Contains commonly used loss functions."""

import functools

from flax.training import common_utils

import jax
import jax.numpy as jnp


def binary_logistic_loss(logit, label, num_classes=2):
  """Computes the logistic loss for one datapoint.

  Args:
    logit: logit predicted by the model
    label: true class label: 0 or 1.
    num_classes: not used

  Returns:
    loss: value of the loss
  """
  del num_classes
  return (jax.nn.softplus(logit) - label * logit).sum()


@functools.partial(jax.jit, static_argnums=(2,))
def cross_entropy_loss(logprobs, label,
                       num_classes):
  """Computes the cross entropy loss for one datapoint.

  Args:
    logprobs: log probabilities predicted by the model
    label: true class label
    num_classes: number of classes in the task

  Returns:
    loss: value of the loss.
  """
  one_hot_labels = common_utils.onehot(label, num_classes=num_classes)
  return -jnp.sum(one_hot_labels * logprobs)
