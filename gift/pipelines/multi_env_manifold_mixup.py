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

"""training pipeline for multi environment manifold mixup."""

import functools
import time

from flax.deprecated import nn
import jax
import jax.numpy as jnp

from gift.pipelines import multi_env_end2end
from gift.pipelines import pipeline_utils
from gift.utils import tensor_util


class MultiEnvManifoldMixup(multi_env_end2end.MultiEnvReps2Reps):
  """Training pipeline for multiple environments using manifold mixup."""

  def setup_pmapped_tain_and_eval_steps(self):
    eval_env_ids = list(
        map(int, self.task.dataset.data_iters.validation.keys()))
    train_env_ids, _ = list(
        zip(*dict(self.task.dataset.data_iters['train']).items()))
    train_env_ids = list(map(int, train_env_ids))

    self.p_train_step = functools.partial(
        self.train_step, env_ids=train_env_ids)
    self.pmapped_train_step = jax.pmap(
        self.p_train_step,
        axis_name='batch',
        in_axes=(0, 0),
        static_broadcasted_argnums=(2,),
        donate_argnums=(0, 1))

    self.p_eval_step = functools.partial(
        self.eval_step, all_env_ids=eval_env_ids)
    self.pmapped_eval_step = jax.pmap(
        self.p_eval_step,
        axis_name='batch',
        in_axes=(0, 0),
        static_broadcasted_argnums=(2,))

    self.pmapped_forward_pass = jax.pmap(
        self.forward_pass, axis_name='batch', in_axes=(0, 0, 0, 0))

  def training_loss_fn(self, flax_model, train_state, batch, dropout_rng,
                       env_ids, sampled_layer):
    """Runs forward pass and computes loss.

    Args:
      flax_model: A flax module.
      train_state: TrainState, the state of training including the current
        global_step, model_state, rng, and optimizer.
      batch: Batches from different environments.
      dropout_rng: FLAX PRNG key.
      env_ids: list[int]; List of env codes.
      sampled_layer: str; Name of the layer on which mixup is applied.

    Returns:
      loss, new_module_state and computed logits for each batch.
    """
    dropout_rng, new_rng = jax.random.split(dropout_rng)
    with nn.stochastic(dropout_rng):
      # Run student forward pass:
      (all_env_reps, env_logits, selected_env_reps,
       train_state) = self.stateful_forward_pass(flax_model, train_state, batch)
      new_model_state = train_state.model_state

    sampled_reps = all_env_reps[sampled_layer]
    interpolate_fn = jax.vmap(
        pipeline_utils.interpolate,
        in_axes=(0, 0, 0, 0, None, None, None, None))

    interpolate_rng, new_rng = jax.random.split(new_rng)
    with nn.stochastic(interpolate_rng):
      (interpolated_batches, interpolated_logits, sampled_lambdas,
       train_state) = self.maybe_inter_env_interpolation(
           batch, env_ids, flax_model, interpolate_fn, sampled_layer,
           sampled_reps, selected_env_reps, train_state)

      (same_env_interpolated_batches, same_env_interpolated_logits, _,
       train_state) = self.maybe_intra_env_interpolation(
           batch, env_ids, flax_model, interpolate_fn, sampled_layer,
           sampled_reps, train_state)

    loss_rng, new_rng = jax.random.split(new_rng)
    with nn.stochastic(loss_rng):
      # Compute the total loss (inside nn.stochastic):
      loss = self.task.loss_function(env_logits, selected_env_reps, batch,
                                     env_ids, flax_model.params,
                                     train_state.global_step)
      # Add the loss for cross environment interpolated states:
      if len(env_ids) > 1 and self.hparams.get('inter_env_interpolation', True):
        inter_mixup_factor = self.hparams.get('inter_mixup_factor', 1.0)
        loss += self.task.loss_function(
            interpolated_logits, None, interpolated_batches, None, None,
            train_state.global_step) * inter_mixup_factor

      # Add the loss for same environment interpolated states:
      if self.hparams.get('intra_env_interpolation', True):
        intra_mixup_factor = self.hparams.get('intra_mixup_factor', 1.0)
        loss += self.task.loss_function(
            same_env_interpolated_logits, None, same_env_interpolated_batches,
            None, None, train_state.global_step) * intra_mixup_factor

    logs = {'sampled_lambdas': sampled_lambdas}

    return loss, (new_model_state, env_logits, logs)

  # TODO(samiraabnar): Try to avoid code duplication when overriding this fn.
  def train_step(self, train_state, batch, env_ids, sampled_layer):
    """Runs a single step of training.

    Given the state of the training and a batch of data, computes
    the loss and updates the parameters of the model.

    Args:
      train_state: TrainState, the state of training including the current
        global_step, model_state, rng, and optimizer.
      batch: A single batch of data.
      env_ids: list(int): List of training environments codes.
      sampled_layer: str; Name of the layer on which mixup is applied.

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
        env_ids=env_ids,
        sampled_layer=sampled_layer)

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

    # Prepare arguments for layer sampling:
    sample_batch = self.get_next_batch(train_iters)
    _, all_env_reps, _, _ = self.pmapped_forward_pass(
        self.train_state.optimizer.target, self.train_state, sample_batch,
        self.train_state.rng)
    layer_keys, mixup_layers = pipeline_utils.get_sample_layer_params(
        self.hparams, all_env_reps)

    # Train loop:
    for step in range(self.start_step + 1, self.total_steps + 1):
      train_batches = self.get_next_batch(train_iters)
      sampled_layer = pipeline_utils.sample_layer(
          layer_keys, mixup_layers=mixup_layers)
      self.train_state, t_metrics = self.pmapped_train_step(
          self.train_state, train_batches, train_env_ids, sampled_layer)
      t_metrics = jax.tree_map(lambda x: x[0], t_metrics)
      train_metrics.append(t_metrics)

      (eval_summary, train_metrics, train_summary,
       tick) = self.maybe_eval_and_log(eval_env_ids, eval_summary, master, step,
                                       tick, train_metrics, train_summary)
      # Sync and save
      self.checkpoint(self.train_state, step)

    # wait until computations are done before exiting (for timing!)
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

    # return the train and eval summary after last step for regresesion testing
    return train_summary, eval_summary

  def maybe_intra_env_interpolation(self, batch, env_ids, flax_model,
                                    interpolate_fn, sampled_layer, sampled_reps,
                                    train_state):
    if self.hparams.get('intra_env_interpolation', True):
      # Set alpha ans beta for sampling lambda:
      beta_params = pipeline_utils.get_weight_param(self.hparams, 'beta', 1.0)
      alpha_params = pipeline_utils.get_weight_param(self.hparams, 'alpha', 1.0)
      step = train_state.global_step
      beta = pipeline_utils.scheduler(step, beta_params)
      alpha = pipeline_utils.scheduler(step, alpha_params)

      # This is just a random matching (similar to manifold mixup paper).
      self_aligned_matching_matrix, self_pair_ids = self.get_intra_env_matchings(
          batch, sampled_reps, env_ids)

      # Compute interpolated representations of sampled layer:
      same_env_inter_reps, sample_lambdas = interpolate_fn(
          jax.random.split(nn.make_rng(), len(sampled_reps)),
          self_aligned_matching_matrix, sampled_reps, sampled_reps,
          self.hparams.get('num_of_lambdas_samples_for_mixup',
                           1), alpha, beta, -1)

      # Get interpolated batches (interpolated inputs, labels, and weights)
      same_env_interpolated_batches = self.get_interpolated_batches(
          batch, same_env_inter_reps, self_pair_ids, sample_lambdas,
          self.hparams.get('intra_interpolation_method',
                           'plain_convex_combination'))

      if self.hparams.get('stop_grad_for_intra_mixup', True):
        same_env_interpolated_batches = jax.lax.stop_gradient(
            same_env_interpolated_batches)

      # Compute logits for the interpolated states:
      (_, same_env_interpolated_logits, _,
       train_state) = self.stateful_forward_pass(flax_model, train_state,
                                                 same_env_interpolated_batches,
                                                 sampled_layer)

      return (same_env_interpolated_batches, same_env_interpolated_logits,
              sample_lambdas, train_state)

    return None, None, 0, train_state

  def maybe_inter_env_interpolation(self, batch, env_ids, flax_model,
                                    interpolate_fn, sampled_layer, sampled_reps,
                                    selected_env_reps, train_state):
    if len(env_ids) > 1 and self.hparams.get('inter_env_interpolation', True):
      # We call the alignment method of the task class:
      aligned_pairs = self.task.get_env_aligned_pairs_idx(
          selected_env_reps, batch, env_ids)
      pair_keys, alignments = zip(*aligned_pairs.items())

      # Convert alignments which is the array of aligned indices to match mat.
      alignments = jnp.asarray(alignments)
      num_env_pairs = alignments.shape[0]
      batch_size = alignments.shape[2]
      matching_matrix = jnp.zeros(
          shape=(num_env_pairs, batch_size, batch_size), dtype=jnp.float32)
      matching_matrix = matching_matrix.at[:, alignments[:, 0],
                                           alignments[:, 1]].set(1.0)

      # Convert pair keys to pair ids (indices in the env_ids list).
      pair_ids = [(env_ids.index(int(x[0])), env_ids.index(int(x[1])))
                  for x in pair_keys]

      # Get sampled layer activations and group them similar to env pairs.
      paired_reps = jnp.array([
          (sampled_reps[envs[0]], sampled_reps[envs[1]]) for envs in pair_ids
      ])

      # Set alpha and beta for sampling lambda:
      beta_params = pipeline_utils.get_weight_param(self.hparams, 'inter_beta',
                                                    1.0)
      alpha_params = pipeline_utils.get_weight_param(self.hparams,
                                                     'inter_alpha', 1.0)
      beta = pipeline_utils.scheduler(train_state.global_step, beta_params)
      alpha = pipeline_utils.scheduler(train_state.global_step, alpha_params)

      # Get interpolated reps for each env pair:
      inter_reps, sample_lambdas = interpolate_fn(
          jax.random.split(nn.make_rng(), len(paired_reps[:, 0])),
          matching_matrix, paired_reps[:, 0], paired_reps[:, 1],
          self.hparams.get('num_of_lambdas_samples_for_inter_mixup',
                           1), alpha, beta, -1)

      # Get interpolated batches for each env pair:
      interpolated_batches = self.get_interpolated_batches(
          batch, inter_reps, pair_ids, sample_lambdas,
          self.hparams.get('intra_interpolation_method',
                           'plain_convex_combination'))

      if self.hparams.get('stop_grad_for_inter_mixup', True):
        interpolated_batches = jax.lax.stop_gradient(interpolated_batches)

      # Compute logits for the interpolated states:
      _, interpolated_logits, _, train_state = self.stateful_forward_pass(
          flax_model, train_state, interpolated_batches, sampled_layer)

      return (interpolated_batches, interpolated_logits, sample_lambdas,
              train_state)

    return None, None, 0, train_state

  def get_intra_env_matchings(self, batch, reps, env_ids):
    """This functions returns alignment for matching example of single envs.

    For now, this is only returning random permutations.

    Args:
      batch: list(dict); List of environment batches.
      reps: list(jnp array); representations of a selected layer for each env
        batch.
      env_ids: list(int); list of environment ids.

    Returns:
      self_aligned_matching_matrix, self_pair_ids
    """
    self_aligned_matching_matrix = []
    self_pair_ids = []

    for env_id, env_batch, env_reps in zip(env_ids, batch, reps):
      self_aligned_matching_matrix.append(
          pipeline_utils.get_self_matching_matrix(
              env_batch,
              env_reps,
              mode=self.hparams.get('intra_mixup_mode', 'random'),
              label_cost=self.hparams.get('intra_mixup_label_cost', 1.0),
              l2_cost=self.hparams.get('intra_mixup_l2_cost', 0.001)))
      self_pair_ids.append((env_ids.index(env_id), env_ids.index(env_id)))
    self_aligned_matching_matrix = jnp.array(self_aligned_matching_matrix)
    return self_aligned_matching_matrix, self_pair_ids

  def get_interpolated_batches(self,
                               batch,
                               new_reps,
                               pair_ids,
                               sample_lambdas,
                               interpolation_method='plain_convex_combination'):
    interpolated_batch_keys = []
    keys = []
    # Batch keys that should be interpolated:
    key = 'label'
    paired_batch_keys = jnp.array([
        (batch[x[0]][key], batch[x[1]][key]) for x in pair_ids
    ])
    if interpolation_method == 'plain_convex_combination':
      interpolated_batch_keys.append(
          jax.vmap(tensor_util.convex_interpolate)(paired_batch_keys[:, 0],
                                                   paired_batch_keys[:, 1],
                                                   sample_lambdas))
    else:
      # If the interpolation method is wasserstein or something else, we will
      # assuming the interpolation is label preserving, hence use the label of
      # of the source examples.
      interpolated_batch_keys.append(paired_batch_keys[:, 0])

    keys.append(key)

    # If batch has weights attribute:
    if batch[0].get('weights') is not None:
      key = 'weights'
      paired_batch_keys = jnp.array([
          (batch[x[0]][key], batch[x[1]][key]) for x in pair_ids
      ])
      interpolated_batch_keys.append(
          jax.vmap(tensor_util.convex_interpolate)(paired_batch_keys[:, 0],
                                                   paired_batch_keys[:, 1],
                                                   sample_lambdas))
      keys.append(key)

    # Set env_name to the env_name of first batch:
    key = 'env_name'
    paired_batch_keys = jnp.array(
        list(map(lambda x: (batch[x[0]][key], batch[x[1]][key]), pair_ids)))
    interpolated_batch_keys.append(paired_batch_keys[:, 0])
    keys.append(key)

    # Set inputs to interpolated reps:
    key = 'inputs'
    interpolated_batch_keys.append(new_reps)
    keys.append(key)

    interpolated_batches = []
    for j in range(len(pair_ids)):
      new_batch = {}
      for i in range(len(keys)):
        new_batch[keys[i]] = interpolated_batch_keys[i][j]
      interpolated_batches.append(new_batch)

    return interpolated_batches

  def stateful_forward_pass(self,
                            flax_model,
                            train_state,
                            batch,
                            input_key='input',
                            train=True):
    (env_logits, all_env_reps, selected_env_reps,
     new_model_state) = self.forward_pass(flax_model, train_state, batch,
                                          nn.make_rng(), input_key, train)
    #  Model state, e.g. batch statistics, are averaged over all environments
    #  because we use vmapped_flax_module_train.
    new_model_state = jax.tree_util.tree_map(
        functools.partial(jnp.mean, axis=0), new_model_state)
    # Update the model state already, since there is going to be another forward
    # pass.
    train_state = train_state.replace(model_state=new_model_state)
    return all_env_reps, env_logits, selected_env_reps, train_state
