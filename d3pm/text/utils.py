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

"""Common utilities for interacting with D3PM models."""

import itertools
import operator
import time
from typing import Any, Dict, Mapping, Optional

from absl import logging
import flax
from flax import optim
from flax import serialization
import flax.jax_utils
from flax.training import checkpoints
import gin
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import tensorflow as tf
import tree

from d3pm.text import types


def build_dataset(ds,
                  vocab = None):
  """Create a D3PM Dataset object.

  Args:
    ds: a tf.data.Dataset object.
    vocab: optionally, a vocabulary to include in the dataset.

  Returns:
    a types.Dataset dataset object.
  """
  if ds is None:
    return None

  return types.Dataset(
      dataset=ds,
      _vocab=vocab,
  )


def wrap_datasets(
    train = None,
    valid = None,
    test = None,
    *,
    vocab = None):
  """Create a D3PM Dataset object.

  Args:
    train: train dataset to use for training.
    valid: a dataset to evaluate on. If not provided, will default to test.
    test: a test dataset to test on.
    vocab: optionally, a vocabulary to include in the dataset.

  Returns:
    a types.Dataset dataset object.
  """

  datasets = {"train": train, "valid": valid, "test": test}

  return {
      k: build_dataset(v, vocab=vocab)
      for k, v in datasets.items()
      if v is not None
  }


get_dataset_info = types.get_dataset_info


def get_dataset_info_from_batch(batch,
                                *,
                                vocab = None):
  """Wraps a set of TFDS datasets with vocabularies."""
  shapes = jax.eval_shape(lambda: batch)

  dataset_info = types.DatasetInfo(
      features=batch.keys(), vocab=vocab, shapes=shapes)

  return dataset_info


def build_batch_from_info(dataset_info,
                          prune_batch_dim=True):
  """Given a shape spec, returns an object of the correct shape.

  Args:
    dataset_info: a DatasetInfo object to use to construct an example batch.
    prune_batch_dim: if True, will set the first dim of any batch to size 1.

  Returns:
    a batch with the same shape as dataset_info.shapes.
  """

  def _initialize(spec):
    if prune_batch_dim:
      shape = (1,) + spec.shape[1:]
    else:
      shape = spec.shape

    return jnp.zeros(shape, dtype=spec.dtype)

  return jax.tree.map(_initialize, dataset_info.shapes)


def make_model_apply(model, rng_key):
  """Create a custom model_apply method that supports apply call by name.

  This returns a model_apply function which can be called with
  `model_apply(params, input)` or `model_apply(..., method='embed'). This allows
  this modely_apply function to be passed with RNGs included instead of the
  model itself.

  Args:
    model: a Flax (Linen) module.
    rng_key: an RNG key to use to create RNGs for the model_apply call.

  Returns:
    a Callable model_apply function and a new RNG key.
  """
  dropout_rng, extra_rng = jrandom.split(rng_key)

  def model_apply(params,
                  *args,
                  method="__call__",
                  allow_missing=False,
                  **kwargs):
    if not hasattr(model, method):
      if allow_missing:
        return args[0]
      else:
        raise ValueError(f"Model has no such method {method}.")

    method = getattr(model, method)

    return model.apply(
        params,
        *args,
        **kwargs,
        method=method,
        rngs={
            "dropout": dropout_rng,
            "extra": extra_rng,
        })

  return model_apply


def stack_forest(forest):
  stack_args = lambda *args: np.stack(args)
  return jax.tree.map(stack_args, *forest)


def get_metrics(device_metrics):
  # We select the first element of x in order to get a single copy of a
  # device-replicated metric.
  metrics_np = jax.device_get(device_metrics)
  return stack_forest(metrics_np)


def combine_metrics(step_metrics):
  """Given a list of metric dicts, combine to a single summary metrics dict.

  This method only takes the first device entry from each metric, so if each
  dict has values replicated across a device axis, this will only combine and
  evaluate results for the first device. Furthermore, denominator will be
  multiplied by the number of steps averaged over (usually 100 by default).

  Args:
    step_metrics: A list of dicts with (metric name, metric value) items.
      Contains summed metrics and the corresponding denominator (the number of
      next-token prediction instances). Each metric value must have at least one
      dimension.

  Returns:
    A dict with (metric name, metric value) items containing combined metrics.
  """
  metrics_all = get_metrics(step_metrics)
  lr = None
  if "learning_rate" in metrics_all:
    lr = metrics_all.pop("learning_rate").mean()

  # Pull out metrics which we want to avoid normalizing by denominator.
  # example: activation norms.
  nonorm = dict()
  for k in metrics_all:
    if "nn/" in k:
      nonorm[k] = metrics_all[k]
  for k in nonorm:
    metrics_all.pop(k)
  nonorm = jax.tree.map(jnp.mean, nonorm)

  metrics_all = jax.tree.map(lambda x: jnp.asarray(x, dtype=np.float32),
                             metrics_all)
  metrics_sums = jax.tree.map(jnp.sum, metrics_all)
  denominator = metrics_sums.pop("denominator")
  summary = jax.tree.map(lambda x: x / denominator, metrics_sums)  # pylint: disable=cell-var-from-loop
  summary["denominator_sum"] = denominator
  if lr is not None:
    summary["learning_rate"] = lr

  # Add back in unnormalized metrics.
  for k in nonorm:
    summary[k] = nonorm[k]

  # Calculate (clipped) perplexity after averaging log-perplexities:
  if "loss" in summary:
    summary["perplexity"] = jnp.clip(jnp.exp(summary["loss"]), max=1.0e4)

  return summary


@gin.configurable(module="utils")
def create_learning_rate_scheduler(
    factors="constant * linear_warmup * rsqrt_normalized_decay",
    base_learning_rate=1e-3,
    warmup_steps=5000,
    decay_factor=0.5,
    steps_per_decay=20000,
    steps_per_cycle=100000,
    min_learning_rate=1e-8,
    max_steps=1000000,
):
  """Creates learning rate schedule.

  Interprets factors in the factors string which can consist of:
  * constant: interpreted as the constant value,
  * linear_warmup: interpreted as linear warmup until warmup_steps,
  * linear_warmup_from: linear warmup from min_lr over warmup_steps
  * rsqrt_decay: divide by square root of max(step, warmup_steps)
  * decay_every: Every k steps decay the learning rate by decay_factor.
  * cosine_decay: Cyclic cosine decay, uses steps_per_cycle parameter.
  * linear_decay: Linearly decay to min_factor * base_learning_rate over
    max_steps.

  Args:
    factors: A string with factors separated by '*' that defines the schedule.
    base_learning_rate: Float, the starting constant for the lr schedule.
    warmup_steps: How many steps to warm up for in the warmup schedule.
    decay_factor: The amount to decay the learning rate by.
    steps_per_decay: How often to decay the learning rate.
    steps_per_cycle: Steps per cycle when using cosine decay.
    min_learning_rate: minimum learning rate to allow. All rates are clipped to
      this value, and linear_warmup_from uses this as its starting value.
    max_steps: number of steps over which to decay for linear_decay.

  Returns:
    a function learning_rate(step): float -> {'learning_rate': float}, the
    step-dependent lr.
  """
  factors = [n.strip() for n in factors.split("*")]
  base_lr = base_learning_rate
  min_lr = min_learning_rate
  min_factor = base_lr / min_lr

  def step_fn(step):
    """Step to learning rate function."""
    step = jnp.asarray(step)

    ret = 1.0
    interp_factor = jnp.where(warmup_steps == 0, 1.0, step / warmup_steps)
    for name in factors:
      if name == "constant":
        ret *= base_learning_rate
      elif name == "linear_warmup":  # warm up to constant
        ret *= jnp.minimum(1.0, interp_factor)
      elif name == "linear_warmup_from":  # warm up to constant
        interpolation = jnp.minimum(1.0, interp_factor)

        # derivation: lr = base_lr / min_factor * (1 - interpolation) +
        #                                       (1 - interpolation)
        # lr = base_lr * (interpolation + (1 - interpolation) / min_factor)

        ret *= interpolation + (1 - interpolation) / min_factor
      elif name == "rsqrt_decay":
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == "rsqrt_normalized_decay":
        ret *= jnp.sqrt(warmup_steps)
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == "decay_every":
        ret *= (decay_factor**(step // steps_per_decay))
      elif name == "cosine_decay":
        progress = jnp.maximum(0.0,
                               (step - warmup_steps) / float(steps_per_cycle))
        ret *= jnp.maximum(0.0,
                           0.5 * (1.0 + jnp.cos(jnp.pi * (progress % 1.0))))

      elif name == "linear_decay":
        interpolation = (max_steps - step + warmup_steps) / max_steps
        interpolation = jnp.minimum(
            1.0,
            interpolation)  # now 1 until warmup_steps, then linear decay to 0

        # derivation: interp = base_lr * interpolation +
        #             (1 - interpolation) * base_lr / min_factor
        ret *= (interpolation + (1 - interpolation) / min_factor)
      else:
        raise ValueError("Unknown factor %s." % name)

    ret = jnp.maximum(min_lr, ret)
    return jnp.asarray(ret, dtype=jnp.float32)

  return step_fn


def evaluate(model, eval_ds, num_eval_steps=None):
  """Evaluates model on eval_ds for num_eval_steps.

  Args:
    model: A model to use for evaluation. Must have an evaluate_batch() method.
    eval_ds: A tensorflow dataset containing the data to be used for evaluation.
    num_eval_steps: If given, evaluate for this many steps, otherwise use the
      entire dataset.

  Returns:
    A dictionary with (metric name, metric value) items.
  """
  start = time.time()
  eval_metrics = []
  eval_iter = iter(eval_ds)
  if num_eval_steps is None:
    num_iter = itertools.repeat(1)
  else:
    num_iter = range(num_eval_steps)
  for _, eval_batch in zip(num_iter, eval_iter):
    eval_batch = jax.tree.map(np.asarray, eval_batch)
    metrics, _ = model.evaluate_batch(eval_batch)  # ignore extras
    eval_metrics.append(metrics)
  finish = time.time()
  eval_summary = combine_metrics(eval_metrics)
  eval_summary["runtime"] = finish - start
  eval_summary["steps_per_second"] = len(eval_metrics) / (finish - start)
  return eval_summary


def predict(model, eval_ds, metric_fns, dataset_info, num_predict_steps=None):
  """Collections predictions for a model on eval_ds for num_predict_steps.

  Args:
    model: A model to use for prediction. Must have an predict_batch() method.
    eval_ds: A tensorflow dataset containing the data to be used for evaluation.
    metric_fns: functions to call on predictions.
    dataset_info: a types.DatasetInfo object.
    num_predict_steps: If given, predict for this many steps, otherwise use the
      entire dataset.

  Returns:
    A dictionary with (metric name, metric value) items.
  """
  predictions = []

  eval_iter = iter(eval_ds)
  if num_predict_steps is None:
    num_iter = itertools.repeat(1)
  else:
    num_iter = range(num_predict_steps)
  for _, eval_batch in zip(num_iter, eval_iter):
    eval_batch = jax.tree.map(np.asarray, eval_batch)
    prediction = model.predict_batch(eval_batch)

    for metric_fn in metric_fns:
      metrics = metric_fn(prediction, eval_batch, dataset_info=dataset_info)
      for m in metrics:
        predictions.append(m)

  return predictions


def tree_add(pytree):
  """Sums over the leaf nodes of a pytree."""
  return jax.tree_util.tree_reduce(operator.add, pytree, 0)


def param_shapes(params):
  """Returns a new pytree with the shapes of each leaf node."""
  return jax.tree_util.tree_map(jnp.shape, params)


def num_params(params, unreplicate=False):
  """Returns the total number of parameters for a pytree with NDArray leaves."""
  if unreplicate:
    params = flax.jax_utils.unreplicate(params)

  per_node_params = jax.tree_util.tree_map(jnp.size, params)
  return tree_add(per_node_params)


class FilterOptimizer(optim.OptimizerDef):
  """A version of MultiOptimizer with same semantics as standard optimizer."""

  def __init__(self, traversal, optimizer):
    """Create a new FilterOptimizer.

    See docstring of `MultiOptimizer` for more details.

    Args:
      traversal: a flax.traverse_util.Traversal object.
      optimizer: a flax.optim.OptimizerDef instance.
    """
    super().__init__(optimizer.hyper_params)
    self.traversal = traversal
    self.opt = optimizer

  def init_state(self, params):
    params_t = list(self.traversal.iterate(params))
    state = self.opt.init_state(params_t)
    return state

  def apply_gradient(self, hyper_params, params, states, grads):
    new_params = params

    p = list(self.traversal.iterate(params))
    g = list(self.traversal.iterate(grads))
    new_p, new_states = self.opt.apply_gradient(hyper_params, p, states, g)
    new_params = self.traversal.set(new_p, new_params)

    return new_params, new_states


@gin.configurable()
def load_checkpoint(state, ckpt_dir, allow_missing=False):
  """Load checkpoint from directory into optimizer.

  Args:
    state: Flax optimizer state.
    ckpt_dir: Directory to load checkpoint from.
    allow_missing: Allows missing keys in checkpoint.

  Returns:
    deserialized optimizer.
  """
  del allow_missing

  ckpt = checkpoints.restore_checkpoint(ckpt_dir, target=None)
  print("-- load called --")
  if ckpt is None:
    logging.info("No checkpoint in %s.", ckpt_dir)
    return state

  print("Loading model - is not None")

  optimizer = _load_optimizer(state.optimizer, ckpt["optimizer"])

  keys = [key for key in state.keys() if key != "optimizer"]
  state_dict = {
      k: serialization.from_state_dict(getattr(state, k), ckpt[k]) for k in keys
  }

  state_dict["optimizer"] = optimizer

  return state.replace(**state_dict)


def _load_optimizer(optimizer, ckpt, allow_missing=False):
  """Loads the optimizer from the state dict."""
  init_keys = set(dict(tree.flatten_with_path(ckpt["target"])))
  model_keys = set(dict(tree.flatten_with_path(optimizer.target)))
  missing_in_model = init_keys.difference(model_keys)
  missing_in_init = model_keys.difference(init_keys)
  missing = model_keys.symmetric_difference(init_keys)
  print("init - model keys: %s", str(missing_in_model))
  print("model - init keys: %s", str(missing_in_init))
  print("difference: %s", str(missing))

  if not allow_missing:
    if missing_in_init:
      raise ValueError(
          "Checkpoints must match exactly if `allow_missing=False`. "
          "Checkpoint missing %s" % str(missing_in_init))

  for param_path in missing_in_init:

    def get_path(d, path):
      print(path)
      print("get")
      for k in path:
        print(k)
        d = d[k]
      return d

    def set_path(d, path, value):
      print("set")
      for k in path[:-1]:
        if k not in d:
          d[k] = dict()
        d = d[k]
      k = path[-1]
      if k in d:
        if value.shape != d[k].shape:
          raise ValueError("Shape mismatch: %s" % str(
              (k, value.shape, d[k].shape)))
      d[k] = value
      return d

    target_param = get_path(optimizer.target, param_path)
    set_path(ckpt["target"], param_path, target_param)

    try:
      target_opt_state = get_path(optimizer.state.param_states, param_path)
      target_opt_state = serialization.to_state_dict(target_opt_state)
      set_path(ckpt["state"]["param_states"], param_path, target_opt_state)
    except TypeError:
      print(f"unable to restore state for {param_path}. Resetting state.")
      ckpt["state"] = serialization.to_state_dict(optimizer.state)

  return serialization.from_state_dict(optimizer, ckpt)


def summarize_tree(pytree, short=True):
  """Summarize a pytree (including type, shape, and nan values).

  Args:
    pytree: a Jax PyTree (must support tree_map).
    short: if True, show a compressed version of the summary.

  Returns:
    a statistics pytree containing information about elements of the tree.
  """

  def _describe(x):
    if (hasattr(x, "shape") and x.shape):
      if isinstance(x, jax.core.Tracer):
        return f"{np.dtype(x.dtype).name}{x.shape} {x._trace}"  # pylint: disable=protected-access
      else:
        flat = np.asarray(x).reshape([-1])
        if flat.shape[0] == 0:
          return f"{np.dtype(x.dtype).name}{x.shape}"
        if short:
          info = f"{np.dtype(x.dtype).name}{x.shape}"
          if np.issubdtype(x.dtype, np.floating):
            info += (f" {np.mean(flat):.2} ±{np.std(flat):.2} "
                     f"[{np.min(flat):.2}, {np.max(flat):.2}]")
            if np.any(flat == 0):
              info += f" nz:{np.count_nonzero(flat) / len(flat):.2}"
            if np.any(np.isnan(flat)):
              info += f" nan:{np.count_nonzero(np.isnan(flat)) / len(flat):.2}"
            if np.any(flat == np.inf):
              info += f" inf:{np.count_nonzero(flat == np.inf) / len(flat):.2}"
            if np.any(flat == -np.inf):
              info += (
                  f" -inf:{np.count_nonzero(flat == -np.inf) / len(flat):.2}")
          elif np.issubdtype(x.dtype, np.integer):
            info += (f" [{np.min(flat)}, {np.max(flat)}] "
                     f"nz:{np.count_nonzero(flat) / len(flat):.2}")
          elif np.issubdtype(x.dtype, bool):
            info += f" T:{np.count_nonzero(flat) / len(flat):.2}"
        else:
          info = {
              "shape": x.shape,
              "dtype": x.dtype,
              "min": np.min(flat),
              "max": np.max(flat),
              "nonzero": np.count_nonzero(flat) / len(flat),
          }
          if np.issubdtype(x.dtype, np.floating):
            info.update({
                "mean": np.mean(flat),
                "median": np.median(flat),
                "std": np.std(flat),
                "finite": np.mean(np.isfinite(flat).astype("float32")),
            })
        return info
    else:
      return x

  return jax.tree.map(_describe, pytree)


def apply_ema(decay, avg, new):
  return jax.tree.map(lambda a, b: decay * a + (1. - decay) * b, avg, new)


def expand_by_shape(arr, shape):
  """Appends dimensions to arr according to the shape tuple.

  For example, if shape is (8, 16), will add two extra dimensions with these
  shapes by tiling the existing array.

  Args:
    arr: a jax ndarray.
    shape: a shape tuple.

  Returns:
    expanded tensor.
  """
  ndim = arr.ndim

  arr = jnp.expand_dims(arr, tuple(range(ndim, ndim + len(shape))))
  arr = jnp.tile(arr, (1,) * ndim + shape)
  return arr
