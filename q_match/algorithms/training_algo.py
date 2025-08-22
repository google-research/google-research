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

"""Base training classes.
"""

import abc
import functools

import flax
import jax
import optax


def l2_normalize(
    x,
    axis = -1,
    epsilon = 1e-12,
):
  """L2 normalize a tensor on an axis with numerical stability."""
  norm = jax.numpy.linalg.norm(x, ord=2, axis=axis, keepdims=True)
  return x/jax.numpy.maximum(norm, epsilon)


class TrainingAlgo(abc.ABC):
  """Training Algorithm.

  Attributes:
    logdir: location of the log directory.
    dataset: tf dataset to train.
    batch_size: batch size for training.
    model: the flax model to train.
    eval_model: the eval/inference version of the flax model.
    learning_rate: the learning rate for training.
    epochs: number of epochs to train for
    params: Optional params to start training from.  If None, random params
      are initialized.
    state: Optional state to start training from.
    writer: Writer for writing to tensorboard.
    optimizer: Optimizer for training using gradients.
    optimizer_state: State of the optimizer.
    weight_decay: weight decay coeffecient.
    weigh_decay_mask: Mask to use for weight decay. False values
      means the parameter should be excluded by weight decay.  This mask is
      in addition to a mask on batch norm parameters and bias parameters.
    rngs: The PRNGs for model applies.
  """

  # pylint: disable=unused-argument
  def __init__(self,
               logdir,
               dataset,
               batch_size,
               model,
               eval_model,
               learning_rate,
               epochs,
               params=None,
               state=None,
               writer=None,
               weight_decay=0.,
               weight_decay_mask=None,
               rngs=None,
               **kwargs):
    self.logdir = logdir
    self.dataset = dataset
    self.batch_size = batch_size
    self.model = model
    self.eval_model = eval_model
    self.epochs = epochs
    self.params = params
    self.state = state
    self.learning_rate = learning_rate
    self.writer = writer

    self.rngs = {'dropout': jax.random.PRNGKey(0)}

    batch_norm_mask = jax.tree.map(
        lambda x: not x,
        self.generate_parameter_ancestors(self.params, 'batch_norm'))
    bias_mask = jax.tree.map(
        lambda x: not x, self.is_leaf_name(self.params, 'bias'))
    bias_and_bn_mask = jax.tree.map(lambda x, y: x and y, bias_mask,
                                    batch_norm_mask)

    if weight_decay_mask is None:
      weight_decay_mask = bias_and_bn_mask
    else:
      weight_decay_mask = jax.tree.map(lambda x, y: x and y,
                                       weight_decay_mask, bias_and_bn_mask)

    optimizer = optax.adamw(
        learning_rate=self.learning_rate,
        weight_decay=weight_decay,
        mask=weight_decay_mask,
    )
    self.optimizer = optax.chain(optimizer, optax.zero_nans())
  # pylint: enable=unused-argument

  @abc.abstractmethod
  def _loss(self, *args, **kwargs):
    """Loss function that calls model using params.

    Should return the scalar loss as the first value, and a tuple of
    other auxilary values, such as the updated model state.

    Args:
      *args: Positional arguments.
      **kwargs: Keyword arguments.
    """

  @functools.partial(jax.jit, static_argnums=(0,))
  def loss(self, *args, **kwargs):
    """Jitted version of the private loss."""
    return self._loss(*args, **kwargs)

  @functools.partial(jax.jit, static_argnums=(0,))
  def update_model(self, params, gradients, optimizer_state):
    updates, optimizer_state = self.optimizer.update(gradients,
                                                     optimizer_state,
                                                     params=params)
    params = optax.apply_updates(params, updates)

    return params, optimizer_state

  def get_grad_fn(self,):
    return jax.jit(jax.grad(self.loss, has_aux=True))

  @abc.abstractmethod
  def run(self,):
    """Runs a training algorithm through a dataset for a fixed number of epochs.

    Returns:
      params: Parameters after training
      state: Model state of training.
    """

  def update_rngs(self,):
    """Updates the rngs with new values."""
    new_rngs = {}
    for k, rng in self.rngs.items():
      rng, _ = jax.random.split(rng, 2)
      new_rngs[k] = rng
    self.rngs = new_rngs

  def generate_parameter_ancestors(self, params, name):
    """Returns a Pytree inidicated if the leaf is has an ancestor with name.

    Has the same structure as params, except each leaf is a boolean value
    where True indicates the parameter is a parameter with name as an ancestor.
    This is useful for identifying parameters that should be excluded from
    weight decay.

    Args:
      params: A FrozenDict of parameter values.
      name: The name to match.
    """
    flattened = flax.traverse_util.flatten_dict(params.unfreeze())
    flattened_mask = {k: True if any([name in pname for pname in k]) else False
                      for k in flattened.keys()}
    mask = flax.core.FrozenDict(
        flax.traverse_util.unflatten_dict(flattened_mask))
    return mask

  def is_leaf_name(self, params, name):
    """Returns a Pytree inidicated if the leaf is named name.

    Has the same structure as params, except each leaf is a boolean value
    where True indicates the parameter has name.
    This is useful for identifying parameters that should be excluded from
    weight decay.

    Args:
      params: A FrozenDict of parameter values.
      name: The name to match.
    """
    flattened = flax.traverse_util.flatten_dict(params.unfreeze())
    flattened_mask = {k: True if k[-1] == name else False
                      for k in flattened.keys()}
    mask = flax.core.FrozenDict(
        flax.traverse_util.unflatten_dict(flattened_mask))
    return mask


class PretextTrainingAlgo(TrainingAlgo):
  """Pretext Training Algo.

  Takes care of generating the weight decay masks for pretext parameters.
  """

  def __init__(self,
               logdir,
               dataset,
               batch_size,
               model,
               eval_model,
               learning_rate,
               epochs,
               params=None,
               state=None,
               writer=None,
               weight_decay=0.,
               weight_decay_mask=None,
               patience=32,):
    # Only apply weight decay to pretext parameters.
    pretext_mask = self.generate_parameter_ancestors(params, 'pretext')
    super(PretextTrainingAlgo, self).__init__(
        logdir,
        dataset,
        batch_size,
        model,
        eval_model,
        learning_rate,
        epochs,
        params=params,
        state=state,
        writer=writer,
        weight_decay=weight_decay,
        weight_decay_mask=pretext_mask,
    )
    self.patience = patience
    self.early_stop_params = self.params
    self.early_stop_state = self.state
    self.best_early_stop_loss = float('inf')
    self.patience_counter = 0
