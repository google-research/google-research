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

"""Common interfaces for multi-loss optimizers."""

import abc
import six


@six.add_metaclass(abc.ABCMeta)
class MultiLossOptimizer(object):
  """Abstract class for first-order multi-loss optimizers.

  Namely, these are optimizers which take a problem with multiple losses and
  optimize them to (ideally) end up on the Pareto frontier. This interface is
  for first-order optimizers, which take a set of losses and a first order
  optimizer and return an update op that should be applied to the graph.
  """

  def __init__(self, problem, batch_size=None):
    """Initializes the optimizer.

    Args:
      problem: An instance of `yoto.problems.Problem`.
      batch_size: Integer, the batch size of the inputs passed to
        `compute_train_loss_and_update_op`.
    """
    self._problem = problem
    self._losses_names = list(sorted(self._problem.losses_keys))
    self._batch_size = batch_size

  @abc.abstractmethod
  def compute_train_loss_and_update_op(self, inputs, base_optimizer):
    """Computes the training loss and the update operation.

    Args:
      inputs: Tensor or dict mapping strings to tensors that will be passed to
        the problem given in the initializer.
      base_optimizer: An instance of `tf.train.Optimizer` which will be used
        to compute the update op.
    Returns:
      A tuple of a scalar tensor holding the current loss and an update op.
    """

  def _check_weights_dict(self, weights_dict):
    """Checks if the given dictionary has the correct keys (loss names).

    Args:
      weights_dict: A dictionary with string keys.
    Raises:
      ValueError if the keys are not the same as the names of the losses
      of the problem given in the initializer.
    """
    try:
      weights_keys_set = set(weights_dict.keys())
    except AttributeError:
      raise TypeError("The weights must be a dictionary-like object.")
    names_set = set(self._losses_names)
    if weights_keys_set != names_set:
      raise ValueError(
          "Exactly one weight must be provided for each key, but got {} "
          "while the losses are {}".format(weights_keys_set, names_set))


class MultiLossOptimizerWithConditioning(MultiLossOptimizer):
  """Abstract class for conditional multi-loss optimizers.

  These optimizers condition the problem by passing extra inputs. As these
  additional inputs impact how much importance is assigned to each loss, at eval
  time they have to provide a function that computes the losses for a specific
  set of weights indicating which losses are deemed more important.
  """

  @abc.abstractmethod
  def compute_eval_losses_and_metrics_for_weights(self, inputs, weights_dict):
    """Compute the losses for the given inputs and weights.

    The provided weights indicate how much importance do we assign to the
    individual losses. For example, if loss_1 has a higher weight than loss_2,
    this implies that we care more about minimizing loss_1 than loss_2.

    Args:
      inputs: The inputs to be passed to the underlying problem.
      weights_dict: A dictionary mapping strings to scalars.
    Returns:
      losses: A dict mapping the names of the losses to the corresponding
        values (scalar tensorflow tensors).
      metrics: A dict mapping the names of the metrics to the corresponding
        values (scalar tensorflow tensors).
    """
