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

"""Contains code for the main Trainer class for our models."""

import dataclasses
import functools
import operator as op
from typing import Any, Callable, Dict, Optional, Type

from absl import logging
import chex
import flax
from flax import linen as nn
from flax import optim
import flax.serialization
from flax.training import checkpoints
import gin
import jax
import jax.example_libraries.optimizers
import jax.numpy as jnp
import jax.random as jrandom

from d3pm.text import datasets  # pylint: disable=unused-import
from d3pm.text import types
from d3pm.text import utils


def l2_norm(params):
  return jax.tree_util.tree_map(lambda x: jnp.sum(x**2), params)


def _pmap_preprocess_batch(batch,
                           features=None,
                           disabled=False,
                           max_batch_size=None):
  """Adds an extra device dimension to a pytree to support jax.pmap.

  Args:
    batch: an arbitrary pytree whose first axis is a batch axis to be split.
    features: if not None, a set of features to select from the batch.
    disabled: if True, makes this a no-op.
    max_batch_size: Maximum batch size to allow per device. If provided, extra
      elements will be dropped silently.

  Returns:
    a new pytree where the batch axis has been split into two axes.
  """
  if disabled:
    return batch

  if features is not None:
    batch = {k: batch[k] for k in features}

  def reshape_arr(arr):
    reshaped = arr.reshape((jax.local_device_count(), arr.shape[0] //
                            jax.local_device_count()) + arr.shape[1:])
    if max_batch_size is not None:
      reshaped = reshaped[:, :max_batch_size, Ellipsis]

    return reshaped

  return jax.tree.map(reshape_arr, batch)


@flax.struct.dataclass
class TrainState:
  optimizer: flax.optim.Optimizer
  step: chex.Array

  # parameters for handling outlier rejection
  ema_loss: chex.Array
  ema_variance: chex.Array


def _get_batch_size(batch):
  """Returns the batch size from a batch dictionary."""
  if not batch:
    raise ValueError('Cannot get batch size for an empty batch dictionary.')

  example = next(iter(batch.values()))
  return example.shape[0]


def build_vmapped_loss(
    loss_fn,
    batch,
    rng_key,
    dynamic_state,
    *,
    is_eval,
    model_apply,
    static_state,
    vmap_batch=False,
):
  """Create an (optionally) vmapped version of the loss function."""

  loss_fn = functools.partial(
      loss_fn,
      model_apply=model_apply,
      is_eval=is_eval,
      **static_state,
      **dynamic_state,
  )

  def batch_loss_fn(params, batch, rng_key):
    return loss_fn(params, **batch, rng_key=rng_key)

  if vmap_batch:
    batch_loss_fn = jax.vmap(batch_loss_fn, in_axes=(None, 0, 0))
    batch_size = _get_batch_size(batch)
    rng_key = jrandom.split(rng_key, num=batch_size)

  def full_loss_fn(params):
    """A simple loss function which averages the loss from each device."""

    (loss, denominator), (metrics,
                          extras) = batch_loss_fn(params, batch, rng_key)

    loss, denominator = jnp.asarray(loss), jnp.asarray(denominator)

    return loss.sum() / denominator.sum(), (metrics, extras)

  return full_loss_fn


def standard_train_step(
    state,
    batch,
    rng_key,
    dynamic_state,
    *,
    static_state,
    loss_fn,
    learning_rate_fn,
    model_cls,
    grad_clip = None,
    use_bfloat16 = False,
    parallel=True,
    vmap_batch=False,
    ema_decay_rate = 0.9,
    ema_burn_in = 1000,
    threshold = 0.0,
):
  """Perform a single standard training step.

  Args:
    state: a TrainState object containing the optimizer and EMA params.
    batch: dictionary or tuple
    rng_key: Jax RNG for model Dropout and additional RNG.
    dynamic_state: a dict of dynamic objects that should be passed to the model.
    static_state: any additional state to be passed to the model. The model will
      be recompiled when this changes.
    loss_fn: loss function which takes a function and batch and returns a loss.
    learning_rate_fn: function that returns the learning rate for a given
      iteration.
    model_cls: an nn.Module type to use for training. Must have a train attr.
    grad_clip: if not None, a float which determines the grad clipping norm.
    use_bfloat16: if True, round gradients to bfloat16 during training.
    parallel: if True, pmean reduces across device dimension.
    vmap_batch: if True, apply vmap over the batch axis.
    ema_decay_rate: the rate at which ema stats decay.
    ema_burn_in: the number of steps to skip before rejecting outliers.
    threshold: the probability below which any loss sample will be rejected. Set
      to 0 to disable EMA outlier rejection. Note that this currently doesn't
      work because we don't update the loss across devices properly. So please
      do not enable this.

  Returns:
    the updated optimizer, a metrics dict, and the new Jax RNG key.
  """
  logging.info('Recompiling train_step.')  # only called when recompiling

  optimizer = state.optimizer

  # We handle PRNG splitting inside the top pmap to improve efficiency.
  step = state.step
  lr = learning_rate_fn(step)

  model = model_cls(train=True)
  apply_key, rng_key = jrandom.split(rng_key)
  loss_key, rng_key = jrandom.split(rng_key)

  model_apply = utils.make_model_apply(model, apply_key)

  loss_fn = build_vmapped_loss(
      loss_fn,
      batch,
      loss_key,
      dynamic_state,
      is_eval=False,
      model_apply=model_apply,
      static_state=static_state,
      vmap_batch=vmap_batch)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, (metrics, _)), grad = grad_fn(optimizer.target)

  if use_bfloat16:
    grad = jax.tree.map(lambda x: x.astype(jnp.bfloat16), grad)

  if parallel:
    grad = jax.lax.pmean(grad, 'device')

  if grad_clip is not None:
    # Clip gradients after pmean aggregation
    unclipped_grad = grad
    grad = jax.example_libraries.optimizers.clip_grads(grad, grad_clip)

  new_optimizer = optimizer.apply_gradient(grad, learning_rate=lr)
  metrics['nn/learning_rate'] = lr

  # Gradient norms
  grad_l2_tree = l2_norm(grad)
  grad_l2_sum = jax.tree_util.tree_reduce(op.add, grad_l2_tree)
  grad_l2_max = jax.tree_util.tree_reduce(jnp.maximum, grad_l2_tree)
  metrics['nn/l2_grad_sum'] = grad_l2_sum
  metrics['nn/l2_grad_max'] = grad_l2_max

  if grad_clip is not None:
    # Unclipped gradient norms (if applicable).
    grad_l2_tree = l2_norm(unclipped_grad)
    grad_l2_sum = jax.tree_util.tree_reduce(op.add, grad_l2_tree)
    grad_l2_max = jax.tree_util.tree_reduce(jnp.maximum, grad_l2_tree)
    metrics['nn/l2_noclip_grad_sum'] = grad_l2_sum
    metrics['nn/l2_noclip_grad_max'] = grad_l2_max

  if threshold > 0:
    normal_pdf = jax.scipy.stats.norm.pdf(
        loss, loc=state.ema_loss, scale=jnp.sqrt(state.ema_variance))
    metrics['nn/normal_pdf'] = normal_pdf
    should_replace = (normal_pdf > threshold) | (state.step < ema_burn_in)
  else:
    should_replace = True

  grads_ok = jnp.all(
      jnp.asarray(
          [jnp.all(jnp.isfinite(p)) for p in jax.tree.leaves(new_optimizer)]))

  loss_ok = jnp.all(jnp.isfinite(loss))
  should_replace = should_replace & grads_ok & loss_ok

  metrics['nn/step_skipped'] = 1 - should_replace
  metrics['nn/ema_loss'] = state.ema_loss
  metrics['nn/ema_variance'] = state.ema_variance
  metrics['nn/step'] = state.step
  metrics['nn/grads_ok'] = grads_ok
  metrics['nn/loss_ok'] = loss_ok

  delta = (loss - state.ema_loss)

  new_state = TrainState(
      optimizer=new_optimizer,
      step=state.step + 1,
      ema_loss=state.ema_loss * ema_decay_rate + (1 - ema_decay_rate) * loss,
      ema_variance=state.ema_variance * ema_decay_rate +
      (1 - ema_decay_rate) * delta**2,
  )

  new_state = jax.tree.map(
      lambda a, b: jnp.where(should_replace, a, b),
      new_state,
      state,
  )

  return new_state, metrics, rng_key


def eval_step(optimizer,
              batch,
              rng_key,
              dynamic_state,
              *,
              static_state,
              loss_fn,
              model_cls,
              vmap_batch=False):
  """Calculate evaluation metrics on a batch (without updating params).

  Args:
    optimizer: the optimizer of the model to use.
    batch: a dictionary of the form {'inputs': ..., 'targets': ...}
    rng_key: a Jax PRNGKey.
    dynamic_state: a dict of dynamic objects that should be passed to the model.
    static_state: a dict of static objects that should be passed to the model.
    loss_fn: a loss function which takes a callable and returns a loss.
    model_cls: an nn.Module type for use in evaluation.
    vmap_batch: if True, will vmap over the batch dimension

  Returns:
    a metrics dictionary.
  """
  logging.info('Recompiling eval_step.')

  apply_key, rng_key = jrandom.split(rng_key)
  loss_key, rng_key = jrandom.split(rng_key)

  model = model_cls(train=False)
  model_apply = utils.make_model_apply(model, apply_key)

  loss_fn = build_vmapped_loss(
      loss_fn,
      batch,
      loss_key,
      is_eval=True,
      model_apply=model_apply,
      dynamic_state=dynamic_state,
      static_state=static_state,
      vmap_batch=vmap_batch)

  _, (metrics, extras) = loss_fn(optimizer.target)

  return metrics, extras, rng_key


def predict_step(params,
                 batch,
                 rng_key,
                 dynamic_state,
                 *,
                 static_state,
                 predict_fn,
                 model_cls,
                 dataset_info,
                 vmap_batch=False,
                 **kwargs):
  """Calculate predictions metrics for non-classification tasks.

  Args:
    params: parameters of the model to use.
    batch: inputs to pass to the model.
    rng_key: a Jax PRNGKey.
    dynamic_state: a dict of dynamic objects that should be passed to the model.
    static_state: a dict of static objects that should be passed to the model.
    predict_fn: a prediction function.
    model_cls: an nn.Module type for use in evaluation.
    dataset_info: information about the dataset, including the vocab
    vmap_batch: if True, vmap over the batch axis.
    **kwargs: kwargs for the predict_fn.

  Returns:
    a metrics dictionary and a new RNG key.
  """
  logging.info('Recompiling predict_step.')

  model = model_cls(train=False)

  rng_key, predict_key = jrandom.split(rng_key)

  predict_fn = functools.partial(
      predict_fn,
      model=model,
      dataset_info=dataset_info,
      **kwargs,
      **static_state,
      **dynamic_state,
  )

  def batch_predict_fn(params, batch, rng_key):
    return predict_fn(params, **batch, rng_key=rng_key)

  if vmap_batch:
    batch_predict_fn = jax.vmap(batch_predict_fn, in_axes=(None, 0, 0))
    batch_size = _get_batch_size(batch)
    predict_key = jrandom.split(predict_key, num=batch_size)

  predictions = batch_predict_fn(params, batch, predict_key)

  return predictions, rng_key


def initialize_params_and_optimizer(
    dataset_info,
    init_features,
    optimizer_def,
    model_cls,
    init_rng,
    vmap_batch=False,
):
  """Initializes the optimizer and state with a given RNG and model."""
  model = model_cls(train=False)

  # we support a custom init method in case a network has complex
  # initialization logic (for example, diffusion models that take noise.
  if hasattr(model, 'custom_init'):
    init_method = model.custom_init
  elif hasattr(model, '__call__'):
    init_method = model.__call__
  else:
    raise ValueError('Model must support __call__ or a custom_init method.')

  batch = utils.build_batch_from_info(dataset_info)
  init_batch = {k: v for k, v in batch.items() if k in init_features}

  if vmap_batch:
    init_batch = {k: v[0] for k, v in init_batch.items()}

  params = model.init(init_rng, **init_batch, method=init_method)

  logging.info('model has %d parameters.', utils.num_params(params))

  optimizer = optimizer_def.create(params)

  state = TrainState(
      optimizer=optimizer,
      step=jnp.array(0),
      ema_loss=jnp.array(0.0),
      ema_variance=jnp.array(1.0),
  )

  return state


gin.external_configurable(optim.Adam, name='Adam')
gin.external_configurable(optim.Adafactor, name='Adafactor')


@gin.configurable(denylist=['dataset_info', 'task'], module='trainers')
class Trainer:
  """Base Trainer class for supervised learning."""

  def __init__(
      self,
      dataset_info,
      task,
      *,
      model_cls,
      optimizer_cls=optim.Adam,
      learning_rate_fn=utils.create_learning_rate_scheduler,
      random_seed = 42,
      disable_pmap = False,
      limit_predict_batch_to = None,
      state=None,
      **train_kwargs,
  ):
    """Trainer class for a simple supervised learning model.

    Args:
      dataset_info: a struct providing information about the dataset
      task: a Task object specifying the loss_fn and predict_fn.
      model_cls: an nn.Module type to use for training.
      optimizer_cls: OptimizerDef type to use to create optimizer.
      learning_rate_fn: a learning rate function to use during training.
      random_seed: random seed for model and parameter init.
      disable_pmap: if True, disables all pmap and JIT logic to make debugging
        easier.
      limit_predict_batch_to: Max batch size per device for predictions. Useful
        if prediction uses more memory than other methods.
      state: if not None, a TrainState object to use. This will reduce the
        loading time.
      **train_kwargs: kwargs to be passed to train_step. Some of these are
        significant, for instance the use of EMA outlier-rejection on updates.
    """
    logging.info(
        'Creating a Trainer class with disable_pmap: %s, dataset_info %s, and task %s.',
        disable_pmap, dataset_info, task)

    self.dataset_info = dataset_info
    self.task = task

    self.aux_state = self.task.state_init_fn(
        dataset_info=dataset_info, task=task)

    for feature in self.task.input_features:
      if feature not in self.dataset_info.features:
        raise ValueError(
            f'Dataset and task are incompatible. Task expected feature '
            f'{feature} which was not supplied by the dataset.')

    self.disable_pmap = disable_pmap
    self.limit_predict_batch_to = limit_predict_batch_to

    self.rng_key = jrandom.PRNGKey(random_seed)
    if self.disable_pmap:
      self.model_rng = jax.random.PRNGKey(random_seed)
    else:
      self.model_rng = jrandom.split(
          jax.random.PRNGKey(random_seed), num=jax.local_device_count())

    self.optimizer_def = optimizer_cls(learning_rate=None)

    if self.task.model_init_fn is not None:
      model_cls = self.task.model_init_fn(
          model_cls=model_cls, task=self.task, dataset_info=self.dataset_info)

    self.model_cls = model_cls

    if state is None:
      init_rng, self.rng_key = jrandom.split(self.rng_key)
      state = initialize_params_and_optimizer(
          dataset_info=self.dataset_info,
          init_features=task.init_features,
          optimizer_def=self.optimizer_def,
          model_cls=self.model_cls,
          init_rng=init_rng,
          vmap_batch=task.vmap_batch,
      )

    self.set_state(state, replicate=True)

    train_fn = functools.partial(
        standard_train_step,
        learning_rate_fn=learning_rate_fn(),
        model_cls=self.model_cls,
        loss_fn=task.loss_fn,
        parallel=not self.disable_pmap,
        static_state=self.aux_state.static_state,
        vmap_batch=task.vmap_batch,
        **train_kwargs)
    self.train_fn = self._apply_pmap(
        train_fn, in_axes=(0, 0, 0, 0), axis_name='device', donate_argnums=(0,))

    eval_fn = functools.partial(
        eval_step,
        model_cls=self.model_cls,
        loss_fn=task.loss_fn,
        static_state=self.aux_state.static_state,
        vmap_batch=task.vmap_batch)

    self.eval_fn = self._apply_pmap(
        eval_fn, in_axes=(0, 0, 0, 0), axis_name='device')

    if task.predict_fn is None:
      self.predict_fn = None
    else:
      self.partial_predict_fn = functools.partial(
          predict_step,
          dataset_info=self.dataset_info,
          model_cls=self.model_cls,
          predict_fn=task.predict_fn,
          static_state=self.aux_state.static_state,
          vmap_batch=task.vmap_batch)
      self.predict_fn = self._apply_pmap(
          self.partial_predict_fn, in_axes=(0, 0, 0, 0), axis_name='device')

    # update the state if provided.
    if self.aux_state and self.aux_state.dynamic_update_fn:
      self.state_update_fn = functools.partial(
          self.aux_state.dynamic_update_fn,
          static_state=self.aux_state.static_state)

      if self.aux_state.jit_update:
        self.state_update_fn = self._apply_pmap(
            self.state_update_fn, in_axes=(0, 0), donate_argnums=(0,))
    else:
      self.state_update_fn = None

    self.update_aux_state()

  def update_aux_state(self):
    """A helper function to update the auxiliary train state."""

    step = self.step

    if self.state_update_fn and (step == 0 or step %
                                 self.aux_state.dynamic_update_freq == 0):
      new_dynamic_state = self.state_update_fn(self.aux_state.dynamic_state,
                                               self.state.optimizer.target)

      self.aux_state = dataclasses.replace(
          self.aux_state, dynamic_state=new_dynamic_state)

  def fit_batch(self, batch):
    """Updates the optimizer state to fit a batch of data."""

    batch = _pmap_preprocess_batch(
        batch, features=self.task.input_features, disabled=self.disable_pmap)

    self.update_aux_state()

    self.state, metrics, self.model_rng = self.train_fn(
        self.state, batch, self.model_rng, self.aux_state.dynamic_state)

    return metrics

  def evaluate_batch(self, batch):
    """Returns metrics for a batch of data without updating the state."""

    batch = _pmap_preprocess_batch(
        batch, features=self.task.input_features, disabled=self.disable_pmap)

    metrics, extras, self.model_rng = self.eval_fn(self.state.optimizer, batch,
                                                   self.model_rng,
                                                   self.aux_state.dynamic_state)
    return metrics, extras

  @functools.lru_cache()
  def _get_predict_fn(self, **kwargs):
    print('Reapplying pmap to predict_batch to include kwargs.')
    partial_predict_fn = functools.partial(self.partial_predict_fn, **kwargs)
    predict_fn = self._apply_pmap(
        partial_predict_fn, in_axes=(0, 0, 0, 0), axis_name='device')
    return predict_fn

  def predict_batch(self, batch, **kwargs):
    """Evaluate/use a trained model by predicting an output.

    Args:
      batch: a batch to pass to the model to condition the prediction step.
      **kwargs: kwargs to pass to predict_fn.

    Returns:
      a dictionary containing metrics.
    """
    if self.predict_fn is None:
      logging.info(
          'predict_batch is disabled because predict_fn was not provided.')
      predictions = {}
      return predictions

    if kwargs:
      predict_fn = self._get_predict_fn(**kwargs)
    else:
      predict_fn = self.predict_fn

    batch = _pmap_preprocess_batch(
        batch,
        features=self.task.predict_features,
        disabled=self.disable_pmap,
        max_batch_size=self.limit_predict_batch_to,
    )

    predictions, self.model_rng = predict_fn(self.state.optimizer.target, batch,
                                             self.model_rng,
                                             self.aux_state.dynamic_state)

    return predictions

  @property
  def step(self):
    if self.disable_pmap:
      return int(self.state.step)
    else:
      return int(self.state.step[0])

  def num_params(self, unreplicate=True):
    unreplicate = unreplicate and not self.disable_pmap
    return utils.num_params(
        self.state.optimizer.target, unreplicate=unreplicate)

  def _apply_pmap(self, fn, *args, **kwargs):
    if self.disable_pmap:
      return fn
    else:
      return jax.pmap(fn, *args, **kwargs)

  def get_state(self, unreplicate=True):
    """Returns the (optionally unreplicated) state."""
    if not unreplicate or self.disable_pmap:
      return self.state

    return flax.jax_utils.unreplicate(self.state)

  def set_state(self, state, replicate=True):
    """Set the state to a new value with replication."""
    if replicate and not self.disable_pmap:
      state = flax.jax_utils.replicate(state)

    self.state = state

  def save_checkpoint(self, ckpt_dir, keep=1, **kwargs):
    """Saves current optimizer parameters to ckpt_dir.

    Args:
      ckpt_dir: directory to save checkpoints in.
      keep: if True, will keep all checkpoints. If False, will overwrite.
      **kwargs: additional arguments to pass to `checkpoints.save_checkpoint()`.
    """

    state = self.get_state(unreplicate=True)
    step = int(state.step)

    checkpoints.save_checkpoint(
        ckpt_dir, target=state, step=step, keep=keep, **kwargs)

  def load_checkpoint(self, ckpt_dir, **kwargs):
    """Loads checkpoints found in ckpt_dir.

    Args:
      ckpt_dir: a directory containing checkpoints to load with Flax. If None
        are found, will use the original optimizer.
      **kwargs: additional kwargs to pass to `load_checkpoint`.
    """
    state = self.get_state(unreplicate=True)

    restored_state = utils.load_checkpoint(state, ckpt_dir, **kwargs)

    if state is not restored_state:
      logging.info('Successfully restored model from checkpoint %s.', ckpt_dir)
      self.set_state(restored_state, replicate=True)
    else:
      logging.info('No checkpoint found in directory %s. Using new params.',
                   ckpt_dir)
