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

"""Pipeline for manifold mixup experiments."""

import copy
import functools
import time

from flax.deprecated import nn
import jax
import jax.numpy as jnp
from gift.pipelines import end2end
from gift.pipelines import pipeline_utils
from gift.utils import tensor_util


class ManifoldMixup(end2end.End2end):
  """Training with manifold mixup.

  In this training pipeline, we have two forward passes in the train step, one
  that get the logits for the given inputs, and one that gets logits for the
  interpolated states (activations of the hidden layers) of the model.
  """

  def setup_pmapped_tain_and_eval_steps(self):
    self.pmapped_train_step = jax.pmap(
        self.train_step,
        axis_name='batch',
        in_axes=(0, 0),
        static_broadcasted_argnums=(2,),
    )
    self.pmapped_eval_step = jax.pmap(
        self.eval_step,
        axis_name='batch',
        in_axes=(0, 0),
    )

  def training_loss_fn(self, flax_module, train_state, batch, dropout_rng,
                       mixup_rng, sampled_layer):
    """Runs forward pass and computes loss.

    Args:
      flax_module: A flax module.
      train_state: TrainState, the state of training including the current
        global_step, model_state, rng, and optimizer.
      batch: Batches from different environments.
      dropout_rng: FLAX PRNG key.
      mixup_rng: FLAX PRNG key.
      sampled_layer: str; Name of the layer on which mixup will be applied.

    Returns:
      loss, new_module_state and computed logits for each batch.
    """

    with nn.stochastic(dropout_rng):
      with nn.stateful(train_state.model_state) as new_model_state:
        logits, reps, _ = flax_module(
            batch['inputs'], train=True, return_activations=True)

        # Get mathing between examples from the mini batch:
        matching_matrix = pipeline_utils.get_self_matching_matrix(
            batch,
            reps[sampled_layer],
            mode=self.hparams.get('intra_mixup_mode', 'random'),
            label_cost=self.hparams.get('intra_mixup_label_cost', 1.0),
            l2_cost=self.hparams.get('intra_mixup_l2_cost', 0.001))

    beta_params = self.hparams.get('beta_schedule_params') or {
        'initial_value': 1.0,
        'mode': 'constant'
    }
    alpha_params = self.hparams.get('alpha_schedule_params') or {
        'initial_value': 1.0,
        'mode': 'constant'
    }
    step = train_state.global_step
    beta = pipeline_utils.scheduler(step, beta_params)
    alpha = pipeline_utils.scheduler(step, alpha_params)

    with nn.stochastic(mixup_rng):
      with nn.stateful(new_model_state) as new_model_state:
        new_logits, sample_lambdas = self.interpolate_and_predict(
            nn.make_rng(), flax_module, matching_matrix, reps, sampled_layer,
            alpha, beta)

      new_batch = copy.deepcopy(batch)

      # Compute labels for the interpolated states:
      new_batch['label'] = tensor_util.convex_interpolate(
          batch['label'], batch['label'][jnp.argmax(matching_matrix, axis=-1)],
          sample_lambdas)

      # Compute weights for the interpolated states:
      if batch.get('weights') is not None:
        new_batch['weights'] = tensor_util.convex_interpolate(
            batch['weights'],
            batch['weights'][jnp.argmax(matching_matrix,
                                        axis=-1)], sample_lambdas)

    # Standard loss:
    loss = self.task.loss_function(logits, batch, flax_module.params)
    # Add the loss from interpolated states:
    loss += self.task.loss_function(new_logits, new_batch)

    return loss, (new_model_state, logits)

  def interpolate_and_predict(self, rng, flax_module, matching_matrix, reps,
                              sampled_layer, mixup_alpha, mixup_beta):
    """Gets model's logits for interpolated activations of the sampled layer.

    Args:
      rng: Jax PRNG key.
      flax_module: Flax model.
      matching_matrix: jnp array; 2d matrix specifying example pairs.
      reps: dict; Activations of all the layers (layer_name --> layer
        activations).
      sampled_layer: str; Name of the sampled layer.
      mixup_alpha: float; Parameter of the beta dist. from which lambda values
        are sampled.
      mixup_beta: float; Parameter of the beta dist. from which lambda values
        are sampled.

    Returns:
      logits for the interpolated states and the sampled lambdas values (used
        for computing the convex combination).
    """
    num_lambdas = self.hparams.get('num_of_lambda_samples_for_mixup', 1)
    new_reps, sample_lambdas = pipeline_utils.interpolate(
        rng, matching_matrix, reps[sampled_layer], reps[sampled_layer],
        num_lambdas, mixup_alpha, mixup_beta)

    # Get logits for the interpolated states:
    new_logits = flax_module(
        new_reps, train=True, input_layer_key=sampled_layer)

    return new_logits, sample_lambdas

  def train_step(self, train_state, batch, sampled_layer):
    """Runs a single step of training.

    Given the state of the training and a batch of data, computes
    the loss and updates the parameters of the model.

    Args:
      train_state: TrainState, the state of training including the current
        global_step, model_state, rng, and optimizer.
      batch: A single batch of data.
      sampled_layer: str; Name of the layer on which mixup will be applied.

    Returns:
      Updated state of training and calculated metrics.

    """
    max_grad_norm = self.hparams.get('max_grad_norm', None)
    new_rng, rng = jax.random.split(train_state.rng, 2)

    mixup_rng, dropout_rng = jax.random.split(rng, 2)
    # bind the rng to the host/device we are on.
    dropout_rng = pipeline_utils.bind_rng_to_host_device(
        dropout_rng, axis_name='batch', bind_to=['host', 'device'])
    mixup_rng = pipeline_utils.bind_rng_to_host_device(
        mixup_rng, axis_name='batch', bind_to=['host', 'device'])

    train_loss_fn = functools.partial(
        self.training_loss_fn,
        train_state=train_state,
        batch=batch,
        dropout_rng=dropout_rng,
        mixup_rng=mixup_rng,
        sampled_layer=sampled_layer)
    new_train_state, metrics = self.compute_grads_and_update(
        batch, max_grad_norm, new_rng, train_loss_fn, train_state)

    return new_train_state, metrics

  def train(self):
    """Training loop."""

    master = jax.host_id() == 0
    train_metrics = []
    train_summary, eval_summary = None, None
    tick = time.time()

    @jax.pmap
    def get_reps(train_state, flax_module, batch):
      with nn.stochastic(train_state.rng):
        with nn.stateful(train_state.model_state):
          _, reps, _ = flax_module(
              batch['inputs'], train=True, return_activations=True)

      return reps

    # Prepare arguments for layer sampling:
    sample_batch = self.get_next_batch(self.task.dataset.data_iters.train)
    reps = get_reps(self.train_state, self.train_state.optimizer.target,
                    sample_batch)
    layer_keys, mixup_layers = pipeline_utils.get_sample_layer_params(
        self.hparams, reps)

    # Train loop:
    for step in range(self.start_step + 1, self.total_steps + 1):
      train_batch = self.get_next_batch(self.task.dataset.data_iters.train)
      sampled_layer = pipeline_utils.sample_layer(
          layer_keys, mixup_layers=mixup_layers)

      self.train_state, t_metrics = self.pmapped_train_step(
          self.train_state, train_batch, sampled_layer)
      train_metrics.append(t_metrics)

      eval_summary, train_metrics, train_summary, tick = self.maybe_eval_and_log(
          eval_summary, master, step, tick, train_metrics, train_summary)

      # sync and save
      self.checkpoint(self.train_state, step)

    if master:
      # evaluate and log the results
      # sync model state across replicas
      self.train_state = pipeline_utils.sync_model_state_across_replicas(
          self.train_state)
      eval_summary, self.train_state = self.eval(step, self.train_state, 'test')

    # wait until computations are done before exiting (for timing!)
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

    # return the train and eval summary after last step for regresesion testing
    return train_summary, eval_summary
