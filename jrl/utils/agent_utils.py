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

"""Utilities used for building agents
"""
from typing import Any
import abc

from acme.agents.jax import builders


class RLComponents(abc.ABC):
  """A dataclass that defines components of a RL algorithm."""

  @abc.abstractmethod
  def make_builder(self):
    """Builder."""

  @abc.abstractmethod
  def make_networks(self):
    """Networks."""

  @abc.abstractmethod
  def make_behavior_policy(self, network):
    """Behavior policy."""

  @abc.abstractmethod
  def make_eval_behavior_policy(self, network, **kwargs):
    """Eval behavior policy."""

#   def policy_variable_name(self) -> str:
#     """A variable name where the learner stores its policy params."""
#     return 'policy'

#   def make_logpi_fn(self, spec: specs.EnvironmentSpec) -> Any:  # pylint: disable=unused-argument
#     """logpi."""
#     return None
