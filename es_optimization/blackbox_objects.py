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
r"""Library for creating different blackbox objects.

Library for creating different blackbox objects providing ways to manipulate
blackbox functions to be optimized (obtaining starting points for their
optimization, executing them and getting metaparameters needed to set
up certain algorithms optimizing them).
"""

import abc
from typing import List, Any
import numpy as np


class BlackboxObject(abc.ABC):
  """Abstract class used to execute and restart blackbox functions.

  Abstract class responsible for executing and restarting blackbox fuctions.
  """

  @abc.abstractmethod
  def get_initial(self):
    """Returns input where optimization starts.

    Returns input where optimization starts.

    Args:

    Returns:
      Initial input where optimization starts.
    """
    raise NotImplementedError('Abstract method')

  @abc.abstractmethod
  def execute(self, params, hyperparams):
    """Executes blackbox function.

    Executes blackbox function.

    Args:
      params: parameters of the blackbox function
      hyperparams: hyperparameters of the blackbox function

    Returns:
      The value of the blackbox function for given parameters and
      hyperparameters as well as evaluation statistics from the evaluation.
    """
    raise NotImplementedError('Abstract method')

  @abc.abstractmethod
  def get_metaparams(self):
    """Returns metaparameters of the blackbox function.

    Returns the list of metaparameters of the blackbox function.

    Args:

    Returns:
      The list of metaparameters of the blackbox function.
    """
    raise NotImplementedError('Abstract method')
