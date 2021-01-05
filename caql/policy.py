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

"""Policy abstract class."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import abc
import six


@six.add_metaclass(abc.ABCMeta)
class BasePolicy(object):
  """Abstract base class for policies."""

  def __init__(self, action_spec):
    self._action_spec = action_spec

  def action(self, state, use_action_function=False, batch_mode=False):
    """Returns the action for the given state.

    Args:
      state: list or np.ndarray. A vector representing a state.
      use_action_function: bool. If True, predict the action using the action
        function. Otherwise, compute the argmax_a Q(s,a) directly.
      batch_mode: bool. If True, collect multiple actions from policy.

    Returns:
      np.ndarray. The action to use for the given state.
    """
    return self._action(state, use_action_function, batch_mode)

  def update_params(self):
    """Updates the parameters of a policy."""
    return self._update_params()

  def params_debug_str(self):
    """Returns the debug string for policy parameters."""
    return self._params_debug_str()

  @property
  def action_spec(self):
    return self._action_spec

  @property
  def continuous_action(self):
    return bool(self._action_spec.shape)

  @abc.abstractmethod
  def _action(self, state, use_action_function, batch_mode=False):
    """Implementation for `action`."""

  @abc.abstractmethod
  def _update_params(self):
    """Implementation for `update_params`."""

  @abc.abstractmethod
  def _params_debug_str(self):
    """Implementation for `params_debug_str`."""
