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

"""Training pipeline Domain Adverserial Neural Networks.

Paper: https://arxiv.org/abs/1505.07818
"""

import functools
import time

from flax.deprecated import nn
from flax.training import common_utils
import jax
import jax.numpy as jnp

from gift.pipelines import multi_env_trainer
from gift.pipelines import pipeline_utils


class MultiEnvDomainAdverserial(multi_env_trainer.MultiEnvTrainer):
  """Training pipeline for multiple environments using dann loss."""

  def __init__(self, model_cls, task, hparams, experiment_dir,
               tb_summary_writer, rng):
    rng, init_rng = jax.random.split(rng)
    super().__init__(model_cls, task, hparams, experiment_dir,
                     tb_summary_writer, init_rng)

    self.labeled_envs = hparams.get('labeled_environments',
                                    None) or task.dataset.train_environments
    self.unlabeled_envs = hparams.get('unlabeled_environments', [])

  def setup_pmapped_tain_and_eval_steps(self):
    eval_env_ids = list(
        map(int, self.task.dataset.data_iters.validation.keys()))

    print('eval env ids', eval_env_ids)

    # self.pmapped_train_step = jax.pmap(
    #     self.train_step,
    #     axis_name='batch',
    #     in_axes=(0, 0, 0),
    #     static_broadcasted_argnums=(3, 4),
    #     donate_argnums=(1, 2))
    # self.pmapped_eval_step = jax.pmap(
    #     self.eval_step,
    #     axis_name='batch',
    #     in_axes=(0, 0),
    #     static_broadcasted_argnums=(2,))
    self.p_eval_step = functools.partial(
        self.eval_step, all_env_ids=eval_env_ids)

    self.pmapped_eval_step = jax.pmap(
        self.p_eval_step,
        axis_name='batch',
        in_axes=(0, 0),
        static_broadcasted_argnums=(2,))

  def dann_forward_pass(self,
                        flax_model,
                        train_state,
                        batch,
                        rng,
                        input_layer_key='input',
                        train=True,
                        discriminator=False):
    # bind the rng to the host/device we are on.
    rng = pipeline_utils.bind_rng_to_host_device(
        rng, axis_name='batch', bind_to=['host', 'device'])

    inputs = pipeline_utils.get_multi_env_inputs(batch, 'inputs')

    with nn.stochastic(rng):
      (env_logits, domain_logits, all_env_reps, selected_env_reps,
       new_model_state) = pipeline_utils.vmapped_dann_flax_module(
           inputs, flax_model, train_state.model_state, input_layer_key, train)

    selected_env_reps = selected_env_reps.reshape(
        (selected_env_reps.shape[0], selected_env_reps.shape[1], -1))

    return (env_logits, domain_logits, all_env_reps, selected_env_reps,
            new_model_state)

  def training_loss_fn(self, flax_model, train_state, batches,
                       unlabeled_batches, dropout_rng, env_ids,
                       unlabeled_env_ids):
    """Runs forward pass and computes loss.

    Args:
      flax_model: A flax module.
      train_state: TrainState, the state of training including the current
        global_step, model_state, rng, and optimizer.
      batches: Batches from labeled environments.
      unlabeled_batches: Batches from unlabeled environments.
      dropout_rng: FLAX PRNG key.
      env_ids: list[int]; List of labeled env ids.
      unlabeled_env_ids: list[int]; List of unlabeled env ids.

    Returns:
      loss, new_module_state and computed logits for each batch.
    """
    (env_logits, env_domain_logits, _, selected_env_reps,
     new_model_state) = self.dann_forward_pass(flax_model, train_state, batches,
                                               dropout_rng)

    #  Model state, e.g. batch statistics, are averaged over all environments
    #  because we use vmapped_flax_module_train.
    new_model_state = jax.tree_util.tree_map(
        functools.partial(jnp.mean, axis=0), new_model_state)
    train_state = train_state.replace(model_state=new_model_state)

    (_, unlabeled_env_domain_logits, _, _,
     new_model_state) = self.dann_forward_pass(flax_model, train_state,
                                               unlabeled_batches, dropout_rng)

    new_model_state = jax.tree_util.tree_map(
        functools.partial(jnp.mean, axis=0), new_model_state)

    if self.hparams.get('dann_factor_params', None):
      dann_factor_params = pipeline_utils.get_weight_param(
          self.hparams, 'dann_factor', .0)
      p = train_state.global_step.astype(
          jnp.float32) / dann_factor_params.total_steps
      # This is the last equation in page 21 in the DANN paper:
      # https://arxiv.org/pdf/1505.07818.pdf
      dann_factor = 2.0 / (1 +
                           jnp.exp(-1.0 * dann_factor_params.gamma * p)) - 1.0
    else:
      dann_factor = 1.0

    with nn.stochastic(dropout_rng):

      # Set domain labels based on env_ids (env_ids list is align with env_reps
      # and env_batches)
      bs = selected_env_reps[0].shape[0]
      domain_labels = []
      for env_id in env_ids:
        domain_labels.append(jnp.zeros_like(env_id))
      for env_id in unlabeled_env_ids:
        domain_labels.append(jnp.ones_like(env_id))

      domain_labels = common_utils.onehot(
          jnp.tile(jnp.asarray(domain_labels)[Ellipsis, None], (1, bs)), 2)

      all_batches = batches + unlabeled_batches
      # Get domain logits
      domain_logits = jnp.concatenate(
          [env_domain_logits, unlabeled_env_domain_logits], axis=0)

      # Compute the total loss (inside nn.stochastic):
      loss = self.task.loss_function(
          env_logits,
          batches,
          domain_logits,
          domain_labels,
          all_batches,
          dann_factor=dann_factor,
          model_params=flax_model.params,
          step=train_state.global_step)

    logs = {'dann_factor': dann_factor}
    return loss, (new_model_state, env_logits, logs)

  def train_step(self, train_state, batches, unlabeled_batches, env_ids,
                 unlabeled_env_ids):
    """Runs a single step of training.

    Given the state of the training and a batch of data, computes
    the loss and updates the parameters of the model.

    Args:
      train_state: TrainState; The state of training including the current
        global_step, model_state, rng, and optimizer.
      batches: list(dict); A batch of data for each labeled environment.
      unlabeled_batches: list(dict); A batch of data for each unlabeled
        environment.
      env_ids: list(int); List of labeled training environments ids.
      unlabeled_env_ids: list(int); List of unlabeled environments ids.

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
        batches=batches,
        unlabeled_batches=unlabeled_batches,
        dropout_rng=dropout_rng,
        env_ids=env_ids,
        unlabeled_env_ids=unlabeled_env_ids)

    new_train_state, metrics_dict = self.compute_grads_and_update(
        batches, env_ids, max_grad_norm, new_rng, train_loss_fn, train_state)

    return new_train_state, metrics_dict

  def train(self):
    """Training loop."""

    master = jax.host_id() == 0

    eval_env_ids = list(
        map(int, self.task.dataset.data_iters.validation.keys()))

    labeled_envs_ids = [
        self.task.dataset.env2id(env) for env in self.labeled_envs
    ]
    unlabeled_envs_ids = [
        self.task.dataset.env2id(env) for env in self.unlabeled_envs
    ]
    labeled_env_dict = {
        str(env_id): self.task.dataset.data_iters.train[str(env_id)]
        for env_id in labeled_envs_ids
    }
    unlabeled_env_dict = {
        str(env_id): self.task.dataset.data_iters.train[str(env_id)]
        for env_id in unlabeled_envs_ids
    }

    labeled_env_ids, labeled_iters = list(zip(*labeled_env_dict.items()))
    labeled_env_ids = list(map(int, labeled_env_ids))

    unlabeled_env_ids, unlabeled_iters = list(zip(*unlabeled_env_dict.items()))
    unlabeled_env_ids = list(map(int, unlabeled_env_ids))

    self.p_train_step = functools.partial(
        self.train_step,
        env_ids=labeled_env_ids,
        unlabeled_env_ids=unlabeled_env_ids)
    print('eval env ids', eval_env_ids)
    self.p_eval_step = functools.partial(
        self.eval_step, all_env_ids=eval_env_ids)

    self.pmapped_train_step = jax.pmap(
        self.p_train_step,
        axis_name='batch',
        in_axes=(0, 0, 0),
        donate_argnums=(1, 2))
    self.pmapped_eval_step = jax.pmap(
        self.p_eval_step,
        axis_name='batch',
        in_axes=(0, 0),
        static_broadcasted_argnums=(2))

    train_summary, eval_summary = self._train_loop(eval_env_ids,
                                                   labeled_env_ids,
                                                   labeled_iters,
                                                   unlabeled_env_ids,
                                                   unlabeled_iters, master)

    # wait until computations are done before exiting (for timing!)
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

    # return the train and eval summary after last step for regresesion testing
    return train_summary, eval_summary

  def _train_loop(
      self,
      eval_env_ids,
      labeled_env_ids,
      labeled_iters,
      unlabeled_env_ids,
      unlabeled_iters,
      master,
  ):

    train_metrics = []
    train_summary, eval_summary = None, None
    tick = time.time()

    for step in range(self.start_step + 1, self.total_steps + 1):

      labeled_batches = self.get_next_batch(labeled_iters)
      unlabeled_batches = self.get_next_batch(unlabeled_iters)

      self.train_state, t_metrics = self.pmapped_train_step(
          self.train_state, labeled_batches, unlabeled_batches)

      t_metrics = jax.tree_map(lambda x: x[0], t_metrics)
      train_metrics.append(t_metrics)

      (eval_summary, train_metrics, train_summary,
       tick) = self.maybe_eval_and_log(eval_env_ids, eval_summary, master, step,
                                       tick, train_metrics, train_summary)

      # Sync and save
      self.checkpoint(self.train_state, step)

    return eval_summary, train_summary
