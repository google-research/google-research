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

"""Matrix initialization utilities."""
import abc
import numpy as np
import six


# TODO(tgale): It would be better if we just used the weight initializers
# from TensorFlow. Currently we do this because our weight connectors are
# not TensorFlow ops, and this gives us flexibility to add weird weight
# connectors that may be difficult to do in TensorFlow.
class Initializer(six.with_metaclass(abc.ABCMeta)):
  """Defines API for a weight initializer."""

  def __init__(self):
    """Initialization API for weight initializer.

    This method can be overridden to save input
    keyword arguments for the specific initializer.
    """
    pass

  @abc.abstractmethod
  def __call__(self, shape):
    pass


class Uniform(Initializer):

  def __init__(self, low=0.0, high=1.0):
    super(Uniform, self).__init__()
    self.low = low
    self.high = high

  def __call__(self, shape):
    return np.reshape(np.random.uniform(
        self.low, self.high, np.prod(shape)), shape)


class Range(Initializer):

  def __call__(self, shape):
    # NOTE: We offset the initial values by 1 s.t. none of the
    # weights are zero valued to begin with.
    return np.reshape(np.arange(np.prod(shape)) + 1, shape)
