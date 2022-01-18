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

"""Mutli environment trainer."""

import functools
import time

from flax import jax_utils
from flax.deprecated import nn
import jax
import jax.numpy as jnp

from gift.data import dataset_utils
from gift.pipelines import multi_env_trainer
from gift.pipelines import pipeline_utils
from gift.utils import shard_util


class MultiEnvEnd2End(multi_env_trainer.MultiEnvTrainer):
  """Multi environment end2end trainer."""

  def setup_pmapped_tain_and_eval_steps(self):
    super().setup_pmapped_tain_and_eval_steps()
    self.pmapped_forward_pass = jax.pmap(
        self.forward_pass, axis_name='batch', in_axes=(0, 0, 0, 0))

  def training_loss_fn(self, flax_model, train_state, batch, dropout_rng,
                       env_ids):
    """Runs forward pass and computes loss.

    Args:
      flax_model: A flax module.
      train_state: TrainState, the state of training including the current
        global_step, model_state, rng, and optimizer.
      batch: Batches from different environments.
      dropout_rng: FLAX PRNG key.
      env_ids: list(int); List of environment codes.

    Returns:
      loss, new_module_state and computed logits for each batch.
    """
    del env_ids
    inputs = pipeline_utils.get_multi_env_inputs(batch, 'inputs')

    with nn.stochastic(dropout_rng):
      env_logits, new_model_state = pipeline_utils.vmapped_flax_module_train(
          flax_model, train_state.model_state, inputs)

    #  Model state, e.g. batch statistics, are averaged over all environments
    #  because we use vmapped_flax_module_train.
    new_model_state = jax.tree_util.tree_map(
        functools.partial(jnp.mean, axis=0), new_model_state)

    loss = self.task.loss_function(env_logits, batch, flax_model.params,
                                   train_state.global_step)
    logs = None
    return loss, (new_model_state, env_logits, logs)


class MultiEnvReps2Reps(multi_env_trainer.MultiEnvTrainer):
  """Training pipeline for multiple environments using representational loss."""

  def __init__(self, model_cls, task, hparams, experiment_dir,
               tb_summary_writer, rng):
    rng, init_rng = jax.random.split(rng)
    super().__init__(model_cls, task, hparams, experiment_dir,
                     tb_summary_writer, init_rng)

    # Set up state transformers to compute the representation based
    # auxilary loss.

    # Get sample batch
    # TODO(samiraabnar): Refactor this by implementing a sample_batch for task.
    _, train_iters = list(
        zip(*dict(self.task.dataset.data_iters['train']).items()))
    init_batch = self.get_next_batch(train_iters)

    # Run the forward pass once to get the representations and their dimensions.
    flax_model = self.train_state.optimizer.target
    with nn.stochastic(rng):
      _, _, selected_env_reps, _ = jax.pmap(
          self.forward_pass,
          axis_name='batch')(flax_model, self.train_state, init_batch,
                             self.train_state.rng)
      self.task.setup_transformers(hidden_reps_dim=selected_env_reps.shape[-1])

  def training_loss_fn(self, flax_model, train_state, batch, dropout_rng,
                       env_ids):
    """Runs forward pass and computes loss.

    Args:
      flax_model: A flax module.
      train_state: TrainState, the state of training including the current
        global_step, model_state, rng, and optimizer.
      batch: Batches from different environments.
      dropout_rng: FLAX PRNG key.
      env_ids: list[int]; List of env codes.

    Returns:
      loss, new_module_state and computed logits for each batch.
    """
    env_logits, _, selected_env_reps, new_model_state = self.forward_pass(
        flax_model, train_state, batch, dropout_rng)
    #  Model state, e.g. batch statistics, are averaged over all environments
    #  because we use vmapped_flax_module_train.
    new_model_state = jax.tree_util.tree_map(
        functools.partial(jnp.mean, axis=0), new_model_state)

    with nn.stochastic(dropout_rng):
      # Compute the total loss (inside nn.stochastic):
      loss = self.task.loss_function(env_logits, selected_env_reps, batch,
                                     env_ids, flax_model.params,
                                     train_state.global_step)
    logs = None
    return loss, (new_model_state, env_logits, logs)


class MultiEnvReps2RepsWithHungarianMatching(MultiEnvReps2Reps):
  """Training pipeline for multiple environments using representational loss."""

  def setup_pmapped_tain_and_eval_steps(self):
    eval_env_ids = list(
        map(int, self.task.dataset.data_iters.validation.keys()))
    train_env_ids, _ = list(
        zip(*dict(self.task.dataset.data_iters['train']).items()))
    train_env_ids = list(map(int, train_env_ids))

    self.p_train_step = functools.partial(
        self.train_step, env_ids=train_env_ids)
    self.p_eval_step = functools.partial(
        self.eval_step, all_env_ids=eval_env_ids)

    self.pmapped_train_step = jax.pmap(
        self.p_train_step, axis_name='batch', in_axes=(0, 0, 0))
    self.pmapped_eval_step = jax.pmap(
        self.p_eval_step, axis_name='batch', in_axes=(0, 0))

    self.pmapped_forward_pass = jax.pmap(
        self.forward_pass, axis_name='batch', in_axes=(0, 0, 0, 0))

  def get_env_aligned_pairs_idx(self, env_reps, env_batches, env_ids):
    """Computes alignments between all environment pairs.

    Args:
      env_reps: jnp array; Reps for different environments (sharded).
      env_batches: list of dict; Batches of different environments (sharded).
      env_ids: jnp array; Environment ids.

    Returns:
      alignment between batches of environment pairs (sharded).
    """
    # TODO(riannevdberg, samiraabnar): aligning is done on the total
    #  unsharded batch, but that requires access between local batches
    #  when computing the loss. Unsure why this works! To be compatible
    #  with random alignment and sinkhorn soft alignment we should do
    #  alignment only within local batches.
    env_reps = shard_util.unshard_env_batch(env_reps)
    env_batches = shard_util.unshard(env_batches)
    with nn.stochastic(jax_utils.unreplicate(self.train_state.rng)):
      alignments = self.task.get_env_aligned_pairs_idx(env_reps, env_batches,
                                                       env_ids)
    alignments = dataset_utils.shard(alignments)

    return alignments

  def training_loss_fn(self, flax_model, train_state, batch, dropout_rng,
                       env_aligned_pairs_idx, env_ids):
    """Runs forward pass and computes loss.

    Args:
      flax_model: A flax module.
      train_state: TrainState, the state of training including the current
        global_step, model_state, rng, and optimizer.
      batch: Batches from different environments.
      dropout_rng: FLAX PRNG key.
      env_aligned_pairs_idx: dict; Alignments between examples of each
        environment pair (env_pair --> alignment).
      env_ids: list[int]; List of env codes.

    Returns:
      loss, new_module_state and computed logits for each batch.
    """

    env_logits, _, selected_env_reps, new_model_state = self.forward_pass(
        flax_model, train_state, batch, dropout_rng)
    #  Model state, e.g. batch statistics, are averaged over all environments
    #  because we use vmapped_flax_module_train.
    new_model_state = jax.tree_util.tree_map(
        functools.partial(jnp.mean, axis=0), new_model_state)
    with nn.stochastic(dropout_rng):
      # Compute the total loss (inside nn.stochastic):
      loss = self.task.loss_function(
          env_logits,
          selected_env_reps,
          batch,
          env_ids,
          flax_model.params,
          train_state.global_step,
          env_aligned_pairs_idx=env_aligned_pairs_idx)

    logs = None
    return loss, (new_model_state, env_logits, logs)

  def train_step(self, train_state, batch, env_aligned_pairs_idx, env_ids):
    """Runs a single step of training.

    Given the state of the training and a batch of data, computes
    the loss and updates the parameters of the model.

    Args:
      train_state: TrainState, the state of training including the current
        global_step, model_state, rng, and optimizer.
      batch: A single batch of data.
      env_aligned_pairs_idx: dict; Alignments between examples of each
        environment pair (env_pair --> alignment).
      env_ids: list(int): List of training environments codes.

    Returns:
      Updated state of training and calculated metrics.

    """
    max_grad_norm = self.hparams.get('max_grad_norm', None)
    new_rng, rng = jax.random.split(train_state.rng)

    # bind the rng to the host/device we are on.
    dropout_rng = pipeline_utils.bind_rng_to_host_device(
        rng, axis_name='batch', bind_to=['host', 'device'])

    train_loss_fn = functools.partial(
        self.training_loss_fn,
        train_state=train_state,
        batch=batch,
        dropout_rng=dropout_rng,
        env_aligned_pairs_idx=env_aligned_pairs_idx,
        env_ids=env_ids)

    new_train_state, metrics = self.compute_grads_and_update(
        batch, env_ids, max_grad_norm, new_rng, train_loss_fn, train_state)

    return new_train_state, metrics

  def train(self):
    """Training loop."""

    master = jax.host_id() == 0

    train_metrics = []
    train_summary, eval_summary = None, None
    tick = time.time()
    eval_env_ids = list(
        map(int, self.task.dataset.data_iters.validation.keys()))
    train_env_ids, train_iters = list(
        zip(*dict(self.task.dataset.data_iters['train']).items()))
    train_env_ids = list(map(int, train_env_ids))
    for step in range(self.start_step + 1, self.total_steps + 1):
      train_batches = self.get_next_batch(train_iters)

      _, _, selected_env_reps, _ = self.pmapped_forward_pass(
          self.train_state.optimizer.target, self.train_state, train_batches,
          self.train_state.rng)
      env_aligned_pairs_idx = self.get_env_aligned_pairs_idx(
          selected_env_reps, train_batches, train_env_ids)
      self.train_state, t_metrics = self.pmapped_train_step(
          self.train_state, train_batches, env_aligned_pairs_idx, train_env_ids)
      train_metrics.append(t_metrics)

      eval_summary, train_metrics, train_summary, tick = self.maybe_eval_and_log(
          eval_env_ids, eval_summary, master, step, tick, train_metrics,
          train_summary)

      # sync and save
      self.checkpoint(self.train_state, step)

    # wait until computations are done before exiting (for timing!)
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

    # return the train and eval summary after last step for regresesion testing
    return train_summary, eval_summary
