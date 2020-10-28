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

"""Common interfaces for multi-objective problems."""
import abc
import six


@six.add_metaclass(abc.ABCMeta)
class Problem(object):
  """Abstract class for multi-loss optimization problems."""

  @abc.abstractmethod
  def losses_and_metrics(self, inputs, inputs_extra=None, training=False):
    """Compute the losses and some additional metrics.

    Args:
      inputs: Dict[ Str: tf.Tensor]. Maps input names (for instance, "image" or
        "label") to their values.
      inputs_extra: tf.Tensor. Additional conditioning inputs.
      training: Bool. Whether to run the model in the training mode (mainly
        important for models with batch normalization).

    Returns:
      losses: Dict[ Str: tf.Tensor]. A dictionary mapping loss names to tensors
        of their per-sample values.
      metrics: Dict[ Str: tf.Tensor]. A dictionary mapping metric names to
        tensors of their per-sample values.
    """

  @abc.abstractmethod
  def initialize_model(self):
    pass

  @abc.abstractproperty
  def losses_keys(self):
    """Names of the losses used in the problem (keys in the dict of losses)."""

  @abc.abstractproperty
  def module_spec(self):
    """TF Hub Module spec."""
