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

"""Utility methods for manipulating evaluation functions (e.g. loss/metric)."""

from typing import Any, Callable, Sequence, SupportsFloat, Union

import chex
import jax
import jax.numpy as jnp

Numeric = SupportsFloat

EvalArgs = Any
SingleSampleEvalOutput = Numeric
SingleSampleEvalFn = Callable[Ellipsis, SingleSampleEvalOutput]
VectorizedEvalOutput = chex.Array
VectorizedEvalFn = Callable[Ellipsis, VectorizedEvalOutput]
AveragedEvalOutput = Numeric
AveragedEvalFn = Callable[Ellipsis, AveragedEvalOutput]

Batch = chex.ArrayTree  # Batches are usually dictionaries of `jnp.ndarrays`.
Params = chex.ArrayTree  # Parameters are arbitrary nests of `jnp.ndarrays`.


@jax.jit
def basic_pred_label_extractor(
    params,
    batch,
    model,
):
  """Applies a model to a batch of data and returns predictions and labels.

  Note that this is just a sample method that's provided for convenience.
  This method should be modified to return all the information that the
  evalution function needs to evaluate the performance of the given model
  params.

  Args:
    params: The parameters of the model that needs to be applied to the batch of
      data.
    batch: The batch of data that we are going to use to evaluate the model.
    model: An object with a method called `apply_fn` that encodes the model
      graph. This function should take the model parameters and the batch
      features as input and return the model predictions. An example is an
      object of type flax.training.train_state.TrainState.

  Returns:
    A dictionary containing the predictions of the model and the labels in the
    batch.
  """
  model_outputs = model.apply_fn(params, batch['feature'], train=False)
  labels = batch['label']
  return model_outputs, labels


@jax.jit
def basic_pred_label_extractor_bn(
    params,
    batch,
    model,
):
  """Applies a model and returns predictions, batch norm parameters, and labels.

  Note that this is just a sample method that's provided for convenience.
  This method should be modified to return all the information that the
  evalution function needs to evaluate the performance of the given model
  params.

  Args:
    params: The parameters of the model that needs to be applied to the batch of
      data.
    batch: The batch of data that we are going to use to evaluate the model.
    model: An object with a method called `apply_fn` that encodes the model
      graph. This function should take the model parameters and the batch
      features as input and return the model predictions. An example is an
      object of type flax.training.train_state.TrainState.

  Returns:
    A dictionary containing the predictions of the model, the updated
    batch_stats collection from all flax.linen.BatchNorm modules, and the labels
    in the batch.
  """
  model_outputs, batch_stats = model.apply_fn(
      params, batch['feature'], train=True, mutable=['batch_stats']
  )
  labels = batch['label']
  return model_outputs, batch_stats, labels


def _get_batch_size(
    args, batch_axes
):
  """Extract the number of samples in a batch of data."""
  if isinstance(batch_axes, int):
    batch_axes_ = [batch_axes for _ in args]
  else:
    batch_axes_ = batch_axes

  arg_batch_sizes = [
      arg.shape[ax] for ax, arg in zip(batch_axes_, args) if ax is not None
  ]
  assert max(arg_batch_sizes) == min(arg_batch_sizes), (
      'Note: This is the default method for deducing the size of the batch '
      'from the arguments passed to the evaluation function. This method '
      'assumes that the arguments passed to the eval function are JAX trees '
      f'all of whose leaves are arrays where dimensions {batch_axes} enumerate '
      'the elements in the batch. This does not seem to be the case '
      'for the current arguments, where the shapes of the arrays along '
      f'dimension {batch_axes} are as follows:\n'
      f'{arg_batch_sizes}\n'
      'You probably need to pass your own method for deducing the number of '
      'samples in the batch from the evaluation function arguments.'
  )
  return arg_batch_sizes[0]


def vectorize_eval_fn(
    single_sample_eval_fn,
    batch_axes = 0,
    use_vmap = True,
    get_batch_size_fn = _get_batch_size,
):
  """A function that vectorizes an evaluation function (e.g. loss/metric).

  Args:
    single_sample_eval_fn: A function computing an eval function on a single
      sample and returns a single number rather than a single-element array.
    batch_axes: The axes along which the batch samples appear.
    use_vmap: Whether single_sample_eval_fn can be vmapped, i.e. if it is
      compilable by JAX. If so, this method loops over the samples in the batch
      using jax.vmap. If not, it uses a regular for loop.
    get_batch_size_fn: A method that extracts the size of the batch of data
      being passed to the eval function. This is needed only when use_vmap is
      set to False.

  Returns:
    A function computing an eval function on a batch of data and returning an
    array with a number for each sample in the batch.
  """
  if use_vmap:
    return jax.jit(jax.vmap(single_sample_eval_fn, in_axes=batch_axes))
  else:
    if not isinstance(batch_axes, int) or batch_axes != 0:
      raise NotImplementedError(
          'If you metric or loss is not vmappable, then the batch axis needs '
          'to be 0 for all of its input variables.'
      )

    def eval_nth_sample(
        single_sample_eval_fn,
        batch_eval_args,
        n,
    ):
      sample_pred_label = jax.tree.map(lambda arr: arr[n, Ellipsis], batch_eval_args)
      return single_sample_eval_fn(*sample_pred_label)

    def vectorized_eval_fn(*batch_eval_args):
      batch_size = get_batch_size_fn(batch_eval_args, batch_axes)
      return jnp.array(
          [
              eval_nth_sample(single_sample_eval_fn, batch_eval_args, ind)
              for ind in range(batch_size)
          ]
      )

    return vectorized_eval_fn


def get_mean_eval_fn(
    single_sample_eval_fn,
    batch_axes = 0,
    use_vmap = True,
    get_batch_size_fn = _get_batch_size,
):
  """Converts a single sample eval fn into the mean eval fn.

  Args:
    single_sample_eval_fn: An function that evaluates the prediction (e.g.
      produced by a model) for a single data sample.
    batch_axes: The axes along which the batch samples appear.
    use_vmap: Whether single_sample_eval_fn can be vmapped, i.e. if it is
      compilable by JAX.
    get_batch_size_fn: A method that extracts the size of the batch of data
      being passed to the eval function.

  Returns:
    A function that computes the average evaluation of the predictions for a
    batch of data.
  """

  def averaged_eval_fn(*batch_eval_args):
    vectorized_eval_fn = vectorize_eval_fn(
        single_sample_eval_fn,
        batch_axes=batch_axes,
        use_vmap=use_vmap,
        get_batch_size_fn=get_batch_size_fn,
    )
    vectorized_eval_results = vectorized_eval_fn(*batch_eval_args)
    assert len(vectorized_eval_results.shape) == 1, (
        'This eval function has not been vectorized properly: the ouput vector '
        'should be 1-dimensional, not '
        f'{len(vectorized_eval_results.shape)}-dimensional.'
    )
    return vectorized_eval_results.mean()

  return averaged_eval_fn


def sample_batch(
    rng,
    batch_size,
    batch_axes,
    *eval_args,
):
  """A method for sampling a batch."""
  if isinstance(batch_axes, int):
    batch_axes_ = [batch_axes for _ in eval_args]
  else:
    batch_axes_ = batch_axes

  def sample_array(arr, rng, batch_size, batch_axis):
    return jax.random.choice(
        rng, arr, shape=(batch_size,), replace=True, axis=batch_axis
    )

  def sampled_tree_or_tree(arg, ind):
    batch_axis = batch_axes_[ind]
    if batch_axis is not None:
      return jax.tree.map(
          lambda arr: sample_array(arr, rng, batch_size, batch_axis), arg
      )
    else:
      return arg

  return tuple(
      [sampled_tree_or_tree(arg, ind) for ind, arg in enumerate(eval_args)]
  )


def bootstrap_averaged_eval_fn(
    eval_fn,
    rand_key,
    num_eval_args,
    batch_axes = 0,
    use_vmap = True,
    get_batch_size_fn = _get_batch_size,
    sample_batch_fn=sample_batch,
):
  """Bootstraps a mean only metric to return samples from a distribution.

  Args:
    eval_fn: A function with inputs (preds, labels), which are the predictions
      made by a model and the ground truth labels, and the output is a number
      that calculates the mean of a metric measuring the performance of the
      model on the batch of labeled data.
    rand_key: A seed for randomly sampling from the batch.
    num_eval_args: Number of arguments that eval_fn takes.
    batch_axes: The axes along which the batch samples appear.
    use_vmap: Whether eval_fn can be vmapped, i.e. if it is compilable by JAX.
    get_batch_size_fn: A method that extracts the size of the batch of data
      being passed to the eval function.
    sample_batch_fn: `Callable[[chex.PRNGKey, int, int, *EvalArgs], EvalArgs]` A
      method that takes in a set of arguments for eval_fn and returns another
      set of arguments for eval_fn but for a sampled batch.

  Returns:
    A function with the same inputs as eval_fn that returns a 1-dimensional
    array that has as many elements as the number of labeled samples in the
    batch.
  """

  def sample_eval_fn(
      rng,
      batch_size,
      batch_axes,
      *batch_eval_args,
  ):
    sampled_batch_eval_args = sample_batch_fn(
        rng, batch_size, batch_axes, *batch_eval_args
    )
    return eval_fn(*sampled_batch_eval_args)

  if use_vmap:
    mapped_sample_eval_fn = jax.jit(
        jax.vmap(
            sample_eval_fn, in_axes=[0, None, None] + [None] * num_eval_args
        ),
        static_argnames=['batch_size', 'batch_axes'],
    )
  else:

    def mapped_sample_eval_fn(
        rngs,
        batch_size,
        batch_axes,
        *batch_eval_args,
    ):
      return jnp.array(
          [
              sample_eval_fn(rng, batch_size, batch_axes, *batch_eval_args)
              for rng in rngs
          ]
      )

  def bootstrapped_eval_fn(*batch_eval_args):
    batch_size = get_batch_size_fn(batch_eval_args, batch_axes)
    rngs = jax.random.split(rand_key, batch_size)
    return mapped_sample_eval_fn(
        rngs, int(batch_size), batch_axes, *batch_eval_args
    )

  return bootstrapped_eval_fn
