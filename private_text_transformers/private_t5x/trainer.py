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

"""T5X trainers that support differential privacy."""
import functools
from typing import Any, Dict, Mapping, Optional, Sequence, TYPE_CHECKING, Tuple

from absl import logging
import cached_property
import jax
from jax import config
from jax import numpy as jnp
from jax.experimental import maps
from t5x import metrics as t5x_metrics
from t5x import models
from t5x import partitioning
from t5x import train_state as train_state_lib
from t5x import trainer as t5x_trainer
import tensorflow as tf

from flaxformer import activation_partitioning

MetricsMap = t5x_metrics.MetricsMap

# Because we use xmap with pjit.
config.update("experimental_xmap_spmd_lowering", True)

# Disable all sharding annotations in Flaxformer as they don't work with xmap.
# TODO(bastings): See if these are still needed.
activation_partitioning.with_sharding = lambda x, _: x

if TYPE_CHECKING:  # See b/163639353
  cached_property = property  # pylint: disable=invalid-name
else:
  cached_property = cached_property.cached_property

BatchType = Mapping[str, jnp.ndarray]
DropoutRng = jnp.ndarray
MetricMapType = Mapping[str, jnp.ndarray]
ModelWeights = Any
MutableMetricMapType = Dict[str, jnp.ndarray]
P = partitioning.PartitionSpec
TensorBatchType = Mapping[str, tf.Tensor]
LearningRateCallable = t5x_trainer.LearningRateCallable
LogicalAxisRules = partitioning.LogicalAxisRules


def _remove_batch_rule(rules):
  """Removes the batch rule and returns the rest."""
  return [(k, v) for (k, v) in rules if k != "batch"]


def standard_logical_axis_rules_without_batch(
    activation_partitioning_dims = 1,
    parameter_partitioning_dims = 1,
    additional_rules = None):
  """Default sharding rules for T5X model in terms of logical axis names.

  Args:
    activation_partitioning_dims: enables 2-D activation sharding when set to 2.
    parameter_partitioning_dims: enables 2-D parameter sharding when set to 2.
    additional_rules: additional rules (a sequence of tuples) that will be
      appended to the standard rules.

  Returns:
    Sequence of logical axis rules
  """
  rules = partitioning.standard_logical_axis_rules(
      activation_partitioning_dims=activation_partitioning_dims,
      parameter_partitioning_dims=parameter_partitioning_dims,
      additional_rules=additional_rules)
  return _remove_batch_rule(rules)


def clip_grad_norm(grads, l2_norm_clip):
  """Clip grad."""
  logging.info("We are clipping the grads to %f", l2_norm_clip)

  nonempty_grads, tree_def = jax.tree.flatten(grads)
  total_grad_norm = jnp.linalg.norm(
      [jnp.linalg.norm(neg.ravel()) for neg in nonempty_grads])

  divisor = jnp.maximum(total_grad_norm / l2_norm_clip, 1.)
  normalized_nonempty_grads = [g / divisor for g in nonempty_grads]
  return jax.tree.unflatten(tree_def, normalized_nonempty_grads)


def add_grad_noise(rng, grads, noise_multiplier, l2_norm_clip,
                   global_batch_size):
  """Adds random noise to the clipped, averaged grads."""
  logging.info("We are adding the noise %f", noise_multiplier)
  grads_flat, grads_treedef = jax.tree.flatten(grads)
  rngs = jax.random.split(rng, len(grads_flat))
  # The grads are already normalized by the batch size (because for DP we
  # should use loss with normalize_loss_by_num_nonpadding_tokens=True.
  factor = l2_norm_clip * noise_multiplier / global_batch_size
  noised_grads = [
      g + factor * jax.random.normal(r, g.shape)
      for r, g in zip(rngs, grads_flat)
  ]
  return jax.tree.unflatten(grads_treedef, noised_grads)


def add_grad_noise_and_normalize(rng, grads, noise_multiplier,
                                 l2_norm_clip, global_batch_size):
  """Adds noise to the summed clipped grads and normalizes by batch size."""
  grads_flat, grads_treedef = jax.tree.flatten(grads)
  rngs = jax.random.split(rng, len(grads_flat))
  factor = l2_norm_clip * noise_multiplier
  noised_grads = [
      g + factor * jax.random.normal(r, g.shape)
      for r, g in zip(rngs, grads_flat)
  ]
  normalized_noised_grads = [g / global_batch_size for g in noised_grads]
  return jax.tree.unflatten(grads_treedef, normalized_noised_grads)


def accumulate_grads_microbatched(
    model,
    train_state,
    batch,
    dropout_rng,
    num_microbatches,
    use_dp,
    dp_l2_clip_norm,
    dp_noise_multiplier,
):
  """Implements optional microbatched gradient accumulation.

  Args:
    model: the instantiation of `BaseModel` to train.
    train_state: internal training state.
    batch: input batch consisting of either - simply-padded batched features
      'encoder_input_tokens', 'decoder_input_tokens' 'decoder_target_tokens'
      'decoder_loss_weights'- packed, batched features with additional
      "(encoder|decoder)_segment_id", "(encoder|decoder)_position"
    dropout_rng: jax PRNGKey for dropout.
    num_microbatches: the number of microbatches to use, or None for direct
      training.
    use_dp: Whether to use differentially private training. If num_microbatches
      is not provided, we will set it to the batch size to clip individual
      gradients. If it is provided, we will work with those minibatches.
    dp_l2_clip_norm: the max l2 norm to clip (individual or minibatch) gradients
      to.
    dp_noise_multiplier: Noise multiplier for the dp training.

  Returns:
   Accumulated gradients and incremental metrics.
  """
  logging.info(
      "use_dp=%s, dp_l2_clip_norm=%f, "
      "dp_noise_multiplier=%f, num_microbatches=%r", use_dp, dp_l2_clip_norm,
      dp_noise_multiplier, num_microbatches)

  batch_size = next(iter(batch.values())).shape[0]
  num_devices = jax.device_count()
  logging.info("Using %d devices", num_devices)

  grad_fn = jax.value_and_grad(model.loss_fn, has_aux=True)

  if num_microbatches is None or num_microbatches <= 1:

    if use_dp:  # DP with a single batch and per-example gradients.
      logging.info("Using xmap for per-example grads.")

      def grad_fn_with_clipping(params, batch, rng):
        """Wrapper to add grad clipping to grad_fn."""
        # Adding batch dimension since this code is run inside vmap
        batch = jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), batch)
        aux, grads = grad_fn(params, batch, rng)
        grads = clip_grad_norm(grads, dp_l2_clip_norm)
        return aux, grads

      # We use xmap so that we get vmap using multiple devices.
      per_example_grad_fn = maps.xmap(
          grad_fn_with_clipping,
          in_axes=({}, ["batch", Ellipsis], {}),
          out_axes=["batch", Ellipsis],
          axis_resources={"batch": "data"})

      dropout_rng, noise_rng = jax.random.split(dropout_rng)

      (loss, metrics), grad_accum = per_example_grad_fn(train_state.params,
                                                        batch, dropout_rng)

      # Sum per-example metrics to become batch metrics (as if without vmap).
      metrics = jax.tree.map(lambda metric: jnp.sum(metric, axis=0), metrics)

      # Sum per-example clipped gradients.
      grad_accum = jax.tree.map(functools.partial(jnp.sum, axis=0), grad_accum)

      # Add DP noise and normalize by global batch size.
      assert batch_size > 0, "batch size <= 0 error"
      grad_accum = add_grad_noise_and_normalize(noise_rng, grad_accum,
                                                dp_noise_multiplier,
                                                dp_l2_clip_norm, batch_size)
    else:  # Single batch without xmap/DP.
      (loss, metrics), grad_accum = grad_fn(train_state.params, batch,
                                            dropout_rng)

    del loss
  else:  # Loop over microbatches.
    assert batch_size % num_microbatches == 0, (
        "Batch size isn't divided evenly by num_microbatches.")
    microbatch_size = batch_size // num_microbatches
    logging.info("using microbatches: %d microbatches, %d size",
                 num_microbatches, microbatch_size)

    def get_microbatch(batch, idx):
      """Fetch microbatch slice from possibly-packed input data."""
      offset = idx * microbatch_size
      length = microbatch_size
      starts = {k: [offset] + [0] * (b.ndim - 1) for k, b in batch.items()}
      limits = {k: [length] + list(b.shape[1:]) for k, b in batch.items()}
      return {
          k: jax.lax.dynamic_slice(b, starts[k], limits[k])
          for k, b in batch.items()
      }

    def metrics_and_grad(
        loop_cnt,
        dropout_rng):
      dropout_rng, sub_dropout_rng = jax.random.split(dropout_rng)

      mbatch = get_microbatch(batch, loop_cnt)

      # We need to annotate the microbatch sharding as we would a batch.
      mbatch = jax.tree.map(
          lambda x: partitioning.with_sharding_constraint(  # pylint: disable=g-long-lambda
              x, partitioning.PartitionSpec("data")),
          mbatch)

      (loss, metrics), grad = grad_fn(train_state.params, mbatch,
                                      sub_dropout_rng)

      # Clip the noise.
      if use_dp:
        grad = clip_grad_norm(grad, dp_l2_clip_norm)

      del loss

      return metrics, grad

    def per_microbatch_train_step(
        loop_cnt, state
    ):
      (dropout_rng, grad_accum, prev_metrics) = state
      metrics, grad = metrics_and_grad(loop_cnt, dropout_rng)
      grad_accum = jax.tree.map(jnp.add, grad_accum, grad)
      metrics = jax.lax.cond(
          loop_cnt == 0, lambda _: metrics,
          lambda _: t5x_trainer.merge_metrics(prev_metrics, metrics), None)
      return dropout_rng, grad_accum, metrics

    # Initialize gradient accumulation loop state.
    accum_dtype = jnp.float32
    grad_accum_init = jax.tree.map(lambda x: jnp.zeros(x.shape, accum_dtype),
                                   train_state.params)
    initial_metrics_shape, _ = jax.eval_shape(
        metrics_and_grad, loop_cnt=0, dropout_rng=dropout_rng)
    initial_metrics = {
        k: t5x_metrics.shape_obj_to_defined_obj(v)
        for k, v in initial_metrics_shape.items()
    }
    loop_init = (dropout_rng, grad_accum_init, initial_metrics)
    # Run gradient accumulation loop.
    new_dropout_rng, grad_accum, metrics = jax.lax.fori_loop(
        0, num_microbatches, per_microbatch_train_step, loop_init)
    del new_dropout_rng

    # Divide the grads by the num of microbatches.
    grad_accum_flat, tree_def = jax.tree.flatten(grad_accum)
    grad_accum_flat = [g / num_microbatches for g in grad_accum_flat]
    grad_accum = jax.tree.unflatten(tree_def, grad_accum_flat)

    if use_dp:  # Add DP noise to accumulated grad from microbatches.
      noise_rng = jax.random.fold_in(jax.random.PRNGKey(0), train_state.step)  # pytype: disable=wrong-arg-types  # jax-ndarray
      grad_accum = add_grad_noise(noise_rng, grad_accum, dp_noise_multiplier,
                                  dp_l2_clip_norm, microbatch_size)

  return grad_accum, metrics


class PrivateTrainer(t5x_trainer.Trainer):
  """A T5X trainer that supports differential privacy."""

  def __init__(self,
               model,
               train_state,
               partitioner,
               eval_names,
               summary_dir,
               train_state_axes,
               rng,
               learning_rate_fn,
               num_microbatches,
               weight_metrics_computer = None,
               use_dp = False,
               dp_l2_clip_norm = 0.,
               dp_noise_multiplier = 0.):
    """Trainer constructor.

    Args:
      model: the instantiation of `BaseModel` to train.
      train_state: a train state with parameters and optimizer state.
      partitioner: the partitioner to use.
      eval_names: names of evaluation datasets, which must match the keys of the
        mapping passed to `eval`.
      summary_dir: optional directory to write TensorBoard metrics to.
      train_state_axes: partitioning info for the optimizer to be used.
      rng: jax PRNGKey seed for dropout.
      learning_rate_fn: returns the learning rate given the current step.
      num_microbatches: the number of microbatches to use, or None for direct
        training.
      weight_metrics_computer: A WeightMetricsComputer instance, or None, to
        decide what metrics, if any, to log about weights and weight updates
        during training.
      use_dp: Whether to use differentially private training. If
        num_microbatches is not provided, we will set it to the batch size to
        clip individual gradients. If it is provided, we will work with those
        minibatches.
      dp_l2_clip_norm: the max l2 norm to clip (individual or minibatch)
        gradients to.
      dp_noise_multiplier: Noise multiplier for the dp training.
    """
    logging.info(
        "Using custom trainer, use_dp=%s, dp_l2_clip_norm=%f, dp_noise_multiplier=%f",
        use_dp, dp_l2_clip_norm, dp_noise_multiplier)
    self._use_dp = use_dp
    self._dp_l2_clip_norm = dp_l2_clip_norm
    self._dp_noise_multiplier = dp_noise_multiplier

    super().__init__(
        model=model,
        train_state=train_state,
        partitioner=partitioner,
        eval_names=eval_names,
        summary_dir=summary_dir,
        train_state_axes=train_state_axes,
        rng=rng,
        learning_rate_fn=learning_rate_fn,
        num_microbatches=num_microbatches,
        weight_metrics_computer=weight_metrics_computer)

  @cached_property
  def _partitioned_train_step(self):

    def train_with_lr(train_state,
                      batch):
      learning_rate = self._learning_rate_fn(train_state.step)
      dropout_rng = self._get_step_rng(train_state.step)  # pytype: disable=wrong-arg-types  # jax-ndarray

      grad_accum, metrics = (
          accumulate_grads_microbatched(
              model=self._model,
              train_state=train_state,
              batch=batch,
              dropout_rng=dropout_rng,
              num_microbatches=self._num_microbatches,
              use_dp=self._use_dp,
              dp_l2_clip_norm=self._dp_l2_clip_norm,
              dp_noise_multiplier=self._dp_noise_multiplier))
      new_train_state, metrics = t5x_trainer.apply_grads(  # pytype: disable=wrong-arg-types  # jax-ndarray
          train_state, grad_accum, metrics, learning_rate,
          self._weight_metrics_computer)
      return new_train_state, metrics

    return self._partitioner.partition(
        train_with_lr,
        in_axis_resources=(self._train_state_axes, P("data",)),
        out_axis_resources=(self._train_state_axes, None),
        donate_argnums=(0,))
