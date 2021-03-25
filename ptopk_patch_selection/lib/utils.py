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

"""Utility functions for training and evaluation."""

import collections
import contextlib
import re
import time
from typing import Any, Callable, Dict, List, Sequence, Tuple

from absl import logging
from clu import parameter_overview
from flax import nn
import jax
from jax import numpy as jnp
import ml_collections
import optax


def create_grid(
    samples_per_dim, value_range = (-1.0, 1.0)
):
  """Creates a tensor with equidistant entries from -1 to +1 in each dim.

  Args:
    samples_per_dim: Number of points to have along each dimension.
    value_range: In each dimension, points will go from range[0] to range[1]

  Returns:
      A tensor of shape [samples_per_dim] + [len(samples_per_dim)].
  """
  s = [jnp.linspace(value_range[0], value_range[1], n) for n in samples_per_dim]
  pe = jnp.stack(jnp.meshgrid(*s, sparse=False, indexing="ij"), axis=-1)
  return jnp.array(pe)


def squared_distance(a, b):
  """Calculates the squared distance for each pair of rows from a and b."""
  a_sq = jnp.sum(a * a, axis=-1)
  b_sq = jnp.sum(b * b, axis=-1)
  ab = jnp.einsum("...kd,...xd->...kx", a, b)
  d = a_sq[Ellipsis, None] + b_sq[Ellipsis, None, :] - 2 * ab
  return d


def cosine_decay(lr,
                 step,
                 total_steps,
                 epsilon = 1e-5):
  ratio = jnp.maximum(0, step / (total_steps + epsilon))
  mult = 0.5 * (1. + jnp.cos(jnp.pi * ratio))
  return mult * lr


def linear_warmup_learning_rate(step,
                                lr,
                                warmup_steps):
  """Linear warmup of learning rate at beginning of training."""
  warmup = jnp.minimum(1., step / warmup_steps)
  return lr * warmup


def get_cosine_learning_rate(step,
                             base_learning_rate,
                             num_steps,
                             warmup_steps = 100):
  """Cosine learning rate schedule."""
  lr = cosine_decay(base_learning_rate, step - warmup_steps,
                    num_steps - warmup_steps)
  lr = linear_warmup_learning_rate(step, lr, warmup_steps)
  return lr


@jax.jit
def to_tree_arrays(list_of_trees):
  """Convert a list of pytrees into a pytree of stacked jnp.arrays.

  Args:
    list_of_trees: A list of pytrees containing numbers as leaves.

  Returns:
    A pytree of jnp.arrays having the same structure as the elements of
    `list_of_trees`

  Example:
    >>> to_tree_arrays([
        (1, {"a": jnp.array([1,2])}),
        (2, {"a": jnp.array([3,4])})
      ])
    (DeviceArray([1, 2], dtype=int32),
     {'a': DeviceArray([[1, 2],
                        [3, 4]], dtype=int32)})
  """
  if not list_of_trees:
    return list_of_trees

  trees_list = jax.tree_transpose(
      jax.tree_structure([0] * len(list_of_trees)),
      jax.tree_structure(list_of_trees[0]), list_of_trees)

  trees_array = jax.tree_multimap(lambda _, ls: jnp.stack(ls), list_of_trees[0],
                                  trees_list)

  return trees_array


def check_epochs_and_steps(config,
                           preferred_option=None):
  """Make sure that either num_epochs or num_train_steps is defined.

  Throws error if both or neither are defined.

  Args:
    config: Configuration
    preferred_option: Whether num_epochs or num_train_steps should be defined.
  """

  num_epochs_defined = True
  num_train_steps_defined = True

  if config.get("num_epochs") is None or config.get("num_epochs") == -1:
    num_epochs_defined = False

  if config.get("num_train_steps") is None or config.get(
      "num_train_steps") == -1:
    num_train_steps_defined = False

  if ((num_epochs_defined and num_train_steps_defined) or
      (not num_epochs_defined and not num_train_steps_defined)):
    raise ValueError("Need to define either num_epochs or num_train_steps, "
                     f"Now, num_epochs is {config.get('num_epochs')!r}, "
                     f"num_train_steps is {config.get('num_train_steps')!r}.")

  if preferred_option is not None:
    if preferred_option == "num_epochs":
      if not num_epochs_defined:
        raise ValueError("You should define hyperparameters in terms of "
                         "num_epochs, but now it is "
                         f"{config.get('num_epochs')!r}.")
    elif preferred_option == "num_train_steps":
      if not num_train_steps_defined:
        raise ValueError("You should define hyperparameters in terms of "
                         "num_train_steps, but now it is "
                         f"{config.get('num_train_steps')!r} ")
    else:
      raise ValueError(f"Preferred option {preferred_option!r} not available.")


def get_optax_schedule_fn(
    *, warmup_ratio, num_train_steps, decay,
    decay_at_steps,
    cosine_decay_schedule):
  """Learning rate schedule compatible with optax.scale_by_schedule op.

  Args:
    warmup_ratio: Fraction of total schedule that has a linear warmup.
    num_train_steps: Number of training steps in total.
    decay: Amount of decay to apply at each `decay_at_steps`.
    decay_at_steps: Piecewise constant weight decay is applied at these steps.
    cosine_decay_schedule: Whether to use a cosine decay schedule. This is
      mutually exclusive to `decay_at_steps`.

  Raises:
    NotImplementedError if both `cosine_decay_schedule` is True and
    `decay_at_steps` is not empty.

  Returns:
    Return a scedule_fn which takes `count` as input and returns `step_size`.
  """
  if cosine_decay_schedule and decay_at_steps:
    raise NotImplementedError(
        "Joint decay_at_steps and cosine_decay_schedule is not implemented.")

  def schedule_fn(count):
    progress = count / num_train_steps

    if cosine_decay_schedule:
      logging.info("Uses cosine decay learning rate with linear warmup on %f "
                   "first steps.", warmup_ratio)
      def cosine_decay_fn(_):
        return (jnp.cos(jnp.pi * (progress - warmup_ratio) /
                        (1 - warmup_ratio)) + 1) / 2

      return jax.lax.cond(
          progress < warmup_ratio,
          lambda _: progress / warmup_ratio,  # linear warmup
          cosine_decay_fn,
          None)

    if decay_at_steps:
      logging.info(
          "Learning rate will be multiplied by %f at steps %s [total steps %d]",
          decay, decay_at_steps, num_train_steps)
      logging.info("Learning rate has a linear warmup on %f first steps.",
                   warmup_ratio)
      decay_at_steps_dict = {s: decay for s in decay_at_steps}
      fn = optax.piecewise_constant_schedule(1., decay_at_steps_dict)
      step_size = fn(count)
      if warmup_ratio > 0.0:
        step_size *= jnp.minimum(progress / warmup_ratio, 1.0)
      return step_size

    if warmup_ratio > 0.0:
      return min(progress / warmup_ratio, 1.0)
    else:
      return 1.0

  return schedule_fn


class DecoupledWeightDecayState(optax.OptState):
  """Maintains count for scale scheduling."""
  count: jnp.ndarray  # shape=(), dtype=jnp.int32
  step_size: jnp.ndarray  # shape=(), dtype=jnp.float32


def decoupled_weight_decay(decay,
                           step_size_fn):
  """Adds decay * step_size_fn(count) to updates.

  Args:
    decay: the decay coefficient for weight decay.
    step_size_fn: a function that takes an update count as input and proposes
      the step_size to multiply the params by.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(_):
    return DecoupledWeightDecayState(count=jnp.zeros([], jnp.int32),
                                     step_size=jnp.zeros([], jnp.float32))

  def update_fn(updates, state, params=None):
    step_size = step_size_fn(state.count) * decay
    updates = jax.tree_multimap(lambda u, p: u - step_size * p, updates, params)

    # does a _safe_int32_increment
    max_int32_value = jnp.iinfo(jnp.int32).max
    new_count = jnp.where(state.count < max_int32_value,
                          state.count + 1,
                          max_int32_value)
    new_state = DecoupledWeightDecayState(count=new_count, step_size=step_size)

    return updates, new_state

  return optax.GradientTransformation(init_fn, update_fn)


class Means:
  """Small helper class for collecting mean values (e.g. during eval)."""

  def __init__(self):
    self.reset()

  def append(self, metrics):
    for k, v in metrics.items():
      self._means[k].append(v)

  def result(self):
    return {k: jnp.stack(v).mean().item() for k, v in self._means.items()}

  def reset(self):
    self._means = collections.defaultdict(list)


class StepTimer:
  """Keeps track of steps_per_sec and examples_per_sec."""

  def __init__(self, *, batch_size, initial_step):
    self.t0 = time.time()
    self.batch_size = batch_size
    self.last_step = initial_step

  @contextlib.contextmanager
  def paused(self):
    """Pause the timer for a `with` block."""
    t0 = time.time()
    try:
      yield
    finally:
      self.t0 += time.time() - t0

  def get_and_reset(self, step):
    """Returns steps_per_sec and examples_per_sec and starts a new period."""
    steps = step - self.last_step
    secs = time.time() - self.t0
    self.last_step = step
    self.t0 = time.time()
    return dict(
        steps_per_sec=steps / secs,
        examples_per_sec=self.batch_size * steps / secs,
    )


def flatten_dict(config_dict, parent_key="", sep="_"):
  """Flatten config_dict such that it can be used by hparams.

  Args:
    config_dict: Config-dict.
    parent_key: String used in recursion.
    sep: String used to separate parent and child keys.

  Returns:
   Flattened dict.
  """
  items = []
  for k, v in config_dict.items():
    new_key = parent_key + sep + k if parent_key else k

    # Take special care of things hparams cannot handle.
    if v is None:
      v = "None"
    if isinstance(v, list):
      v = str(v)
    if isinstance(v, tuple):
      v = str(v)

    # Recursively flatten the dict.
    if isinstance(v, dict):
      items.extend(flatten_dict(v, new_key, sep=sep).items())
    else:
      items.append((new_key, v))
  return dict(items)


def l2_normalize(x, dim=-1, epsilon=1e-12):
  """L2-normalize vector x along dimension dim.

  Args:
    x: Vector to be normalized.
    dim: Dimension along which to normalize.
    epsilon: Small constant to avoid division by zero.

  Returns:
    Normalized Vector.
  """
  divisor = jnp.maximum(
      jnp.linalg.norm(x, ord=2, axis=dim, keepdims=True), epsilon)
  return x / divisor


def extract_images_patches(images,
                           window_size,
                           stride = (1, 1)):
  """Extracts patches from an image using a convolution operator.

  Args:
    images: A tensor of images of shapes (B, H, W, C).
    window_size: The size of the patches to extract (h, w).
    stride: The shift between extracted patches (s1, s2)

  Returns:
    All the patches in a tensor of dimension
      (B, (H - h + 1) // s1, (W - w + 1) // s2, h, w, C).
  """
  # batch, channels, height, width
  images = jnp.moveaxis(images, -1, 1)
  d = images.shape[1]
  h, w = window_size

  # construct the lookup conv weights
  dim_out = jnp.arange(d * h * w).reshape((-1, 1, 1, 1))
  dim_in = jnp.arange(d).reshape((1, -1, 1, 1))
  i = jnp.arange(h).reshape((1, 1, -1, 1))
  j = jnp.arange(w).reshape((1, 1, 1, -1))
  weights = ((w * i + j) * d + dim_in == dim_out).astype(jnp.float32)

  # batch, h * w * d, (H - h + 1) // s1, (W - w + 1) // s2
  concatenated_patches = jax.lax.conv(images,
                                      weights,
                                      window_strides=stride,
                                      padding="VALID")

  # batch, (H - h + 1) // s1, (W - w + 1) // s2, h * w * d
  concatenated_patches = jnp.moveaxis(concatenated_patches, 1, -1)

  # batch, (H - h + 1) // s1, (W - w + 1) // s2, h, w, d
  shape = concatenated_patches.shape[:3] + (h, w, d)
  patches = concatenated_patches.reshape(shape)
  return patches


def get_constant_initializer(constant):
  """Returns an initializer function that initializes with a constant."""
  def init(key, shape, dtype=jnp.float32):
    return constant * jax.nn.initializers.ones(key, shape, dtype)
  return init


def extract_windows(inputs, window_size):
  """Extracts windows from a matrix.

  Args:
    inputs: A tensor of shape (n, d).
    window_size: The size of the windows to extract w.

  Returns:
    All the windows in a tensor of dimension (n - w + 1, w, d).
  """
  n, d = inputs.shape
  inputs = jnp.moveaxis(inputs, 0, 1)
  inputs = inputs[jnp.newaxis, :, :]

  # construct the lookup conv weights
  dim_out = jnp.arange(d * window_size).reshape((-1, 1, 1))
  dim_in = jnp.arange(d).reshape((1, -1, 1))
  i = jnp.arange(window_size).reshape((1, 1, -1))
  weights = (i * d + dim_in == dim_out).astype(jnp.float32)

  # -- 1, n * d, n - w
  windows = jax.lax.conv(inputs,
                         weights,
                         window_strides=(1,),
                         padding="VALID")

  windows = jnp.moveaxis(windows, 1, -1)
  shape = (n - window_size + 1, window_size, d)
  windows = windows.reshape(shape)
  return windows


@jax.partial(jax.jit, static_argnums=(0, 1))
def position_offsets(height, width):
  """Generates a (height, width, 2) tensor containing pixel indices."""
  position_offset = jnp.indices((height, width))
  position_offset = jnp.moveaxis(position_offset, 0, -1)
  return position_offset


def get_mse_loss(preds, targets):
  return jnp.mean((preds - targets)**2)


def unflatten_dict(d, delimiter = "/"):
  """Creates a hierarchical dictionary by splitting the keys at delimiter."""
  new_dict = {}
  for path, v in d.items():
    current_dict = new_dict
    keys = path.split(delimiter)
    for key in keys[:-1]:
      if key not in current_dict:
        current_dict[key] = {}
      current_dict = current_dict[key]
    current_dict[keys[-1]] = v
  return new_dict


def add_prefix_to_dict_keys(d, prefix):
  """Returns a new dict with `prefix` before each key."""
  if not prefix:
    return d
  if isinstance(d, nn.Collection):
    items = d.as_dict().items()
  else:
    items = d.items()
  dict_class = type(d)
  return dict_class({f"{prefix}{k}": v for k, v in items})


def override_dict(d, override):
  """Creates a new dict replacing values from override in d."""
  dict_class = type(d)
  if isinstance(d, nn.Collection):
    d = d.as_dict()
  if isinstance(override, nn.Collection):
    override = override.as_dict()

  available_keys = set(d.keys())
  ignored_keys = set(override.keys()).difference(available_keys)
  d.update({k: v for k, v in override.items() if k in available_keys})
  return dict_class(d), ignored_keys


class MultipliersState(optax.OptState):
  """State for per weight learning rate multiplier optimizer."""
  multipliers: Any


def freeze(regex_frozen_weights):
  """Creates an optimizer that set learning rate to 0. for some weights.

  Args:
    regex_frozen_weights: The regex that matches the (flatten) parameters
      that should not be optimized.

  Returns:
    A chainable optimizer.
  """
  return scale_selected_parameters(regex_frozen_weights, multiplier=0.)


def scale_selected_parameters(regex, multiplier):
  """Creates an optimizer that multiply a selection of weights by `multiplier`.

  Args:
    regex: The regex that matches the (flatten) parameters whose learning rate
      should be scaled.
    multiplier: The scaling factor applied to matching parameters.

  Returns:
    A chainable optimizer.
  """
  def init_fn(params):
    flatten_params = parameter_overview.flatten_dict(params)
    multipliers = {k: multiplier if re.match(regex, k) else 1.
                   for k, _ in flatten_params.items()}
    logging.info("Optimizer uses multiplier %f for weights: %s",
                 multiplier, sorted([k for k, _ in flatten_params.items()
                                     if re.match(regex, k)]))
    multipliers = unflatten_dict(multipliers)
    return MultipliersState(multipliers)

  def update_fn(updates, state, params=None):
    del params
    multiplied_updates = jax.tree_multimap(
        lambda m, update: jax.tree_map(lambda u: u * m, update),
        state.multipliers, updates)
    return multiplied_updates, state

  return optax.GradientTransformation(init_fn, update_fn)


def has_inf_or_nan(x):
  return jnp.isinf(x).any() or jnp.isnan(x).any()


def has_any_inf_or_nan(x):
  return any(map(has_inf_or_nan, jax.tree_leaves([x])))
