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

"""Base class for defining downstream evaluation tasks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc


class Task(object):
  """Base class for defining algorithms."""
  _metaclass_ = abc.ABCMeta

  def __init__(self, downstream_task=True):
    """If it is a downstream task then evluation will be done on emebeddings."""
    self.downstream_task = downstream_task

  def evaluate(self, algo, global_step, iterators=None,
               embeddings_dataset=None):
    """Evaluate a checkpoint.

    Args:
      algo: Algorithm, a training algo that contains the model to evaluate.
      global_step: Tensor, integer containing training steps till now.
      iterators: dict, dict of train and val data iterators.
      embeddings_dataset: dict, dict conisting of train and val embeddings and
        labels.

    Raises:
      ValueError: In case invalid configs are passed.
    """
    if iterators and embeddings_dataset:
      raise ValueError('Dataset can either be iterator or embedding. Not both.')

    if not iterators and not embeddings_dataset:
      raise ValueError('Neither embeddings not iterator passed.')

    if self.downstream_task:
      return self.evaluate_embeddings(algo, global_step, embeddings_dataset)
    else:
      return self.evaluate_iterators(algo, global_step, iterators)

  @abc.abstractmethod
  def evaluate_embeddings(self, algo, global_step, embeddings_dataset):
    raise NotImplementedError

  @abc.abstractmethod
  def evaluate_iterators(self, algo, global_step, iterators):
    raise NotImplementedError
