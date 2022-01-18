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

"""training pipeline for GIFT."""

import functools
import time

from absl import logging
from flax import jax_utils
from flax.deprecated import nn
import jax
import jax.numpy as jnp

from gift.pipelines import multi_env_manifold_mixup
from gift.pipelines import pipeline_utils
from gift.tasks import style_transfer_ops
from gift.train_lib import optimizers
from gift.utils import tensor_util


class GradualDomainAdaptationWithMixup(
    multi_env_manifold_mixup.MultiEnvManifoldMixup):
  """Training pipeline for gradual adaptation with manifold mixup."""

  _INTERPOLATION_METHODS = {
      'plain_convex_combination': tensor_util.convex_interpolate,
      'wasserstein': style_transfer_ops.wasserstein,
      'wct': style_transfer_ops.wct
  }

  def __init__(self, model_cls, task, hparams, experiment_dir,
               tb_summary_writer, rng):

    super().__init__(model_cls, task, hparams, experiment_dir,
                     tb_summary_writer, rng)
    self.self_training_iterations = hparams.get('self_training_iterations', 1)
    self.iter_total_steps = self.total_steps // self.self_training_iterations
    self.hparams.keep_env_ckpts = self.hparams.get('keep_env_ckpts', False)
    self.includes_self_supervision = True

    logging.info('self_training_iterations %d', self.self_training_iterations)
    # Set train env (On this environment we use ground truth labels to train
    # the model).
    self.labeled_envs = hparams.get('labeled_environments',
                                    None) or task.dataset.train_environments
    self.unlabeled_envs = hparams.get('unlabeled_environments', [])

    interpolation_method = self.hparams.get('interpolation_method',
                                            'plain_convex_combination')
    intra_interpolation_method = self.hparams.get('intra_interpolation_method',
                                                  'plain_convex_combination')
    self.setup_interpolation_method(interpolation_method,
                                    intra_interpolation_method)

  def setup_interpolation_method(self, interpolation_method,
                                 intra_interpolation_method):
    """Define vmapped interpolation functions."""
    self.interpolate_fn = jax.vmap(
        functools.partial(
            pipeline_utils.interpolate,
            interpolation_method=self
            ._INTERPOLATION_METHODS[interpolation_method]),
        in_axes=(0, 0, 0, 0, None, None, None, None))

    self.intra_interpolate_fn = jax.vmap(
        functools.partial(
            pipeline_utils.interpolate,
            interpolation_method=self
            ._INTERPOLATION_METHODS[intra_interpolation_method]),
        in_axes=(0, 0, 0, 0, None, None, None, None))

  def setup_pmapped_tain_and_eval_steps(self):
    eval_env_ids = list(
        map(int, self.task.dataset.data_iters.validation.keys()))
    self.p_eval_step = functools.partial(
        self.eval_step, all_env_ids=eval_env_ids)

    self.pmapped_eval_step = jax.pmap(
        self.p_eval_step,
        axis_name='batch',
        in_axes=(0, 0),
        static_broadcasted_argnums=(2,),
        donate_argnums=(1))

    self.pmapped_forward_pass = jax.pmap(
        self.forward_pass, axis_name='batch', in_axes=(0, 0, 0, 0))

  def set_pseudo_label_generator(self):
    """Sets the pseudo label generator function."""
    logit_transformer = functools.partial(
        pipeline_utils.logit_transformer,
        temp=self.hparams.get('label_temp') or 1.0,
        confidence_quantile_threshold=self.hparams.get(
            'confidence_quantile_threshold', 0.1),
        self_supervised_label_transformation=self.hparams.get(
            'self_supervised_label_transformation', 'sharp'))
    pseudo_label_generator = functools.partial(
        pipeline_utils.pseudo_label_generator,
        pseudo_labels_transformer_fn=logit_transformer,
        train=self.hparams.get('pseudo_labels_train_mode') or False)
    self.pseudo_label_generator = jax.pmap(pseudo_label_generator)

  def maybe_reset_train_state(self):
    optimizer = jax_utils.unreplicate(self.train_state.optimizer)

    if self.hparams.get('reinitilize_params_at_each_step', False):
      del optimizer.target
      (flax_model, _, _) = pipeline_utils.create_flax_module(
          optimizer.target.module, self.task.dataset.meta_data['input_shape'],
          self.hparams, nn.make_rng(),
          self.task.dataset.meta_data.get('input_dtype', jnp.float32))
    else:
      flax_model = optimizer.target

    # Reset optimizer
    if self.hparams.get('reinitialize_optimizer_at_each_step', False):
      optimizer = optimizers.get_optimizer(self.hparams).create(flax_model)
    else:
      optimizer = optimizer.replace(target=flax_model)

    optimizer = jax_utils.replicate(optimizer)
    self.train_state = self.train_state.replace(optimizer=optimizer)

  def training_loss_fn(self, flax_model, train_state, teacher_train_state,
                       batch, unlabeled_batch, dropout_rng, env_ids,
                       unlabeled_env_ids, sampled_layer):
    """Runs forward pass and computes loss.

    Args:
      flax_model: A flax module.
      train_state: TrainState; The state of training including the current
        global_step, model_state, rng, and optimizer.
      teacher_train_state: TrainState; The state of training for the teacher
        (including the current global_step, model_state, rng, and optimizer).
      batch: list(dict); A batch of data for each environment in the labeld set.
      unlabeled_batch: list(dict); A batch of data for each environment in the
        unlabeld set.
      dropout_rng: FLAX PRNG key.
      env_ids: list(int); List of labeled training environments ids.
      unlabeled_env_ids: list(int); List of unlabeled environments ids.
      sampled_layer: str; Name of the layer on which mixup is applied.

    Returns:
      loss, new_module_state and computed logits for each batch.
    """

    dropout_rng, new_rng = jax.random.split(dropout_rng)
    with nn.stochastic(dropout_rng):
      # Run student forward pass on the labeled envs.
      (all_std_env_reps, std_env_logits, _,
       train_state) = self.stateful_forward_pass(flax_model, train_state, batch)

      # Run teacher forward pass on the labeled envs.
      (labeled_tchr_env_logits, _,
       _) = self.stateless_forward_pass(teacher_train_state.optimizer.target,
                                        teacher_train_state, batch)

      # Run teacher forward pass on the unlabeled envs.
      (unlabeled_tchr_env_logits, all_tchr_unlabeled_env_reps,
       _) = self.stateless_forward_pass(teacher_train_state.optimizer.target,
                                        teacher_train_state, unlabeled_batch)

      # Replace labels with predicted labels from the teacher model.
      for ub_id in range(len(unlabeled_env_ids)):
        unlabeled_batch[ub_id]['label'] = jnp.argmax(
            unlabeled_tchr_env_logits[ub_id], axis=-1)

    # Get sampled layer for interpolations:
    std_sampled_reps = all_std_env_reps[sampled_layer]
    sampled_unlabeled_reps = all_tchr_unlabeled_env_reps[sampled_layer]

    interpolation_rng, new_rng = jax.random.split(new_rng)
    with nn.stochastic(interpolation_rng):
      (interpolated_batches, interpolated_logits, _,
       train_state) = self.maybe_inter_env_interpolation(
           batch, env_ids, flax_model, self.intra_interpolate_fn, sampled_layer,
           std_sampled_reps, std_sampled_reps, train_state)

      (same_env_interpolated_batches, same_env_interpolated_logits, _,
       train_state) = self.maybe_intra_env_interpolation(
           batch, env_ids, flax_model, self.intra_interpolate_fn, sampled_layer,
           std_sampled_reps, train_state)

      (unlabeled_interpolated_batches, unlabeled_interpolated_logits,
       unlabeled_mixup_lambdas, unlabeled_mixup_alpha, unlabeled_mixup_beta,
       train_state) = self.maybe_gradual_interpolation(
           batch, unlabeled_batch, env_ids, unlabeled_env_ids, flax_model,
           self.interpolate_fn, sampled_layer, std_sampled_reps,
           sampled_unlabeled_reps, std_sampled_reps, sampled_unlabeled_reps,
           labeled_tchr_env_logits, unlabeled_tchr_env_logits, train_state,
           teacher_train_state)

      # Compute the total loss (inside nn.stochastic):
      # env_reps and env_ids are set to None to avoid computing a loss for
      # domain mapping (the mapping model is not trained and not used in
      # computing the loss).
      ground_truth_factor_params = pipeline_utils.get_weight_param(
          self.hparams, 'ground_truth_factor', 1.0)
      ground_truth_factor = pipeline_utils.scheduler(
          train_state.global_step, ground_truth_factor_params)

      ground_truth_loss = self.task.loss_function(std_env_logits, None, batch,
                                                  None, flax_model.params,
                                                  train_state.global_step)
      loss = ground_truth_loss * ground_truth_factor

      # Add the loss for cross environment interpolated states:
      if len(env_ids) > 1 and self.hparams.get('inter_env_interpolation', True):
        inter_mixup_factor_params = pipeline_utils.get_weight_param(
            self.hparams, 'inter_mixup_factor', 1.0)
        inter_mixup_factor = pipeline_utils.scheduler(
            train_state.global_step, inter_mixup_factor_params)
        loss += self.task.loss_function(
            interpolated_logits, None, interpolated_batches, None, None,
            train_state.global_step) * inter_mixup_factor

      # Add the loss for same environment interpolated states:
      if self.hparams.get('intra_env_interpolation', True):
        intra_mixup_factor_params = pipeline_utils.get_weight_param(
            self.hparams, 'intra_mixup_factor', 1.0)
        intra_mixup_factor = pipeline_utils.scheduler(
            train_state.global_step, intra_mixup_factor_params)

        loss += self.task.loss_function(
            same_env_interpolated_logits, None, same_env_interpolated_batches,
            None, None, train_state.global_step) * intra_mixup_factor

      # Add the loss for gradual environment interpolations toward unlabeled
      # target environment(s):
      unlabeled_mixup_factor = 0
      unlabeled_loss = 0
    if self.hparams.get('unlabeled_interpolation', True):
      unlabeled_mixup_factor_params = pipeline_utils.get_weight_param(
          self.hparams, 'unlabeled_mixup_factor', 1.0)
      unlabeled_mixup_factor = pipeline_utils.scheduler(
          train_state.global_step, unlabeled_mixup_factor_params)
      unlabeled_loss = self.task.loss_function(unlabeled_interpolated_logits,
                                               None,
                                               unlabeled_interpolated_batches,
                                               None, None,
                                               train_state.global_step)
      loss += unlabeled_loss * unlabeled_mixup_factor

    logs = {}
    logs['unlabeled_mixup_lambda'] = unlabeled_mixup_lambdas
    logs['unlabeled_mixup_alpha'] = unlabeled_mixup_alpha
    logs['unlabeled_mixup_beta'] = unlabeled_mixup_beta
    logs['unlabeled_mixup_factor'] = unlabeled_mixup_factor
    logs['train_loss'] = ground_truth_loss
    logs['unlabeled_loss'] = unlabeled_loss

    return loss, (train_state.model_state, std_env_logits, logs)

  def stateless_forward_pass(self,
                             flax_model,
                             train_state,
                             batch,
                             input_key='input'):
    (all_env_logits, all_env_reps,
     selected_env_reps, _) = self.forward_pass(flax_model, train_state, batch,
                                               nn.make_rng(), input_key)
    return all_env_logits, all_env_reps, selected_env_reps

  def train(self):
    """Training loop."""

    master = jax.host_id() == 0
    global_start_step = self.start_step

    # current_step keeps track of global (cumulative) number of steps the model
    # is trained on all of the environments so that we know the starting
    # step for the next environments.
    current_step = 0

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
    self.pmapped_train_step = jax.pmap(
        self.p_train_step,
        axis_name='batch',
        in_axes=(0, 0, 0, 0),
        static_broadcasted_argnums=(4),
        donate_argnums=(2, 3))
    # Prepare arguments for layer sampling:
    sample_batch = self.get_next_batch(labeled_iters)
    _, all_env_reps, _, _ = self.pmapped_forward_pass(
        self.train_state.optimizer.target, self.train_state, sample_batch,
        self.train_state.rng)
    layer_keys, mixup_layers = pipeline_utils.get_sample_layer_params(
        self.hparams, all_env_reps)

    self.teacher_train_state = self.train_state
    train_summary, eval_summary = None, None
    for _ in range(self.self_training_iterations):
      # Set start and end step for the current environment.
      iter_start_step = current_step
      iter_end_step = iter_start_step + self.iter_total_steps
      if global_start_step < iter_end_step:
        # Resume or start training on this environment if we haven't already
        # trained on it or stopped in the middle of it.

        # Update env_start_step if the preemption has occured in the middle of
        # training on this environments.
        iter_start_step += jnp.maximum(0, global_start_step - iter_start_step)
        train_summary, eval_summary = self._train_loop(
            eval_env_ids, iter_end_step, iter_start_step, labeled_env_ids,
            labeled_iters, layer_keys, master, mixup_layers, unlabeled_env_ids,
            unlabeled_iters)

      current_step += self.iter_total_steps

      # Sync and save
      if self.hparams.keep_env_ckpts:
        self.train_state = self.checkpoint(self.train_state)

      # Reset teacher to use the newly trained model.
      self.teacher_train_state = self.train_state
      self.maybe_reset_train_state()

    # wait until computations are done before exiting (for timing!)
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

    # return the train and eval summary after last step for regresesion testing
    return train_summary, eval_summary

  def _train_loop(self, eval_env_ids, iter_end_step, iter_start_step,
                  labeled_env_ids, labeled_iters, layer_keys, master,
                  mixup_layers, unlabeled_env_ids, unlabeled_iters):
    train_metrics = []
    train_summary, eval_summary = None, None
    tick = time.time()

    for step in range(iter_start_step + 1, iter_end_step + 1):
      labeled_batches = self.get_next_batch(labeled_iters)
      unlabeled_batches = self.get_next_batch(unlabeled_iters)
      sampled_layer = layer_keys[mixup_layers[step % len(mixup_layers)]]

      self.train_state, t_metrics = self.pmapped_train_step(
          self.train_state, self.teacher_train_state, labeled_batches,
          unlabeled_batches, sampled_layer)

      t_metrics = jax.tree_map(lambda x: x[0], t_metrics)
      train_metrics.append(t_metrics)

      (eval_summary, train_metrics, train_summary,
       tick) = self.maybe_eval_and_log(eval_env_ids, eval_summary, master, step,
                                       tick, train_metrics, train_summary)

      # Sync and save
      self.checkpoint(self.train_state, step)

    return eval_summary, train_summary

  def get_env_total_steps(self, labeled_env):
    env_n_exmpls = self.task.dataset.splits['train'][labeled_env].num_examples
    steps_per_epoch = env_n_exmpls // self.hparams.batch_size
    env_total_steps = (steps_per_epoch * self.hparams.num_training_epochs)
    return env_total_steps

  def maybe_gradual_interpolation(
      self, batch, unlabeled_batch, env_ids, unlabeled_env_ids, flax_model,
      interpolate_fn, sampled_layer, selected_env_reps,
      selected_unlabeled_env_reps, sampled_reps, sampled_unlabeled_reps, logits,
      unlabled_logits, train_state, teacher_train_state):

    # Compute alignment based on the selected reps.
    aligned_pairs = self.task.get_bipartite_env_aligned_pairs_idx(
        selected_env_reps, batch, env_ids, selected_unlabeled_env_reps,
        unlabeled_batch, unlabeled_env_ids)
    pair_keys, matching_matrix = zip(*aligned_pairs.items())
    matching_matrix = jnp.array(matching_matrix)

    # Convert pair keys to pair ids (indices in the env_ids list).
    pair_ids = [(env_ids.index(int(x[0])), unlabeled_env_ids.index(int(x[1])))
                for x in pair_keys]

    # Get sampled layer activations and group them similar to env pairs.
    paired_reps = jnp.array([(sampled_reps[envs[0]],
                              sampled_unlabeled_reps[envs[1]])
                             for envs in pair_ids])

    # Set alpha and beta for sampling lambda:
    beta_params = pipeline_utils.get_weight_param(self.hparams,
                                                  'unlabeled_beta', 1.0)
    alpha_params = pipeline_utils.get_weight_param(self.hparams,
                                                   'unlabeled_alpha', 1.0)
    step = train_state.global_step
    beta = pipeline_utils.scheduler(step, beta_params)
    alpha = pipeline_utils.scheduler(step, alpha_params)
    if self.hparams.get('unlabeled_lambda_params', None):
      lambda_params = pipeline_utils.get_weight_param(self.hparams,
                                                      'unlabeled_lambda', .0)
      lmbda = pipeline_utils.scheduler(step, lambda_params)
    else:
      lmbda = -1
    # Get interpolated reps for each en pair:
    inter_reps, sample_lambdas = interpolate_fn(
        jax.random.split(nn.make_rng(), len(paired_reps[:, 0])),
        matching_matrix, paired_reps[:, 0], paired_reps[:, 1],
        self.hparams.get('num_of_lambda_samples_for_inter_mixup',
                         1), alpha, beta, lmbda)

    # Get interpolated batches for each env pair:
    interpolated_batches = self.get_interpolated_batches(
        batch, inter_reps, pair_ids, sample_lambdas,
        self.hparams.get('interpolation_method', 'plain_convex_combination'))
    if self.hparams.get('stop_gradient_for_interpolations', False):
      interpolated_batches = jax.lax.stop_gradient(interpolated_batches)

    if self.hparams.get('interpolated_labels'):
      # Get logits for the interpolated states by interpoting pseudo labels on
      # source and target.
      if self.hparams.get('interpolation_method') == 'plain_convex_combination':
        teacher_interpolated_logits = jax.vmap(tensor_util.convex_interpolate)(
            logits, unlabled_logits, sample_lambdas)
      else:
        teacher_interpolated_logits = logits
    else:
      # Get logits for the interpolated states from the teacher.
      teacher_interpolated_logits, _, _, _ = self.forward_pass(
          teacher_train_state.optimizer.target, teacher_train_state,
          interpolated_batches, nn.make_rng(), sampled_layer)

    # Do we want to propagate the gradients  to the teacher?
    if self.hparams.get('stop_gradient_for_teacher', True):
      teacher_interpolated_logits = jax.lax.stop_gradient(
          teacher_interpolated_logits)

    for i in range(len(interpolated_batches)):
      (interpolated_batches[i]['label'],
       interpolated_batches[i]['weights']) = pipeline_utils.logit_transformer(
           logits=teacher_interpolated_logits[i],
           temp=self.hparams.get('label_temp') or 1.0,
           confidence_quantile_threshold=self.hparams.get(
               'confidence_quantile_threshold', 0.1),
           self_supervised_label_transformation=self.hparams.get(
               'self_supervised_label_transformation', 'sharp'),
           logit_indices=None)

    # Compute logits for the interpolated states:
    (_, interpolated_logits, _,
     train_state) = self.stateful_forward_pass(flax_model, train_state,
                                               interpolated_batches,
                                               sampled_layer)

    return (interpolated_batches, interpolated_logits, sample_lambdas, alpha,
            beta, train_state)

  def train_step(self, train_state, teacher_train_state, batch,
                 unlabeled_batches, sampled_layer, env_ids, unlabeled_env_ids):
    """Runs a single step of training.

    Given the state of the training and a batch of data, computes
    the loss and updates the parameters of the model.

    Args:
      train_state: TrainState; The state of training including the current
        global_step, model_state, rng, and optimizer.
      teacher_train_state: TrainState; The state of training for the teacher
        (including the current global_step, model_state, rng, and optimizer).
      batch: list(dict); A batch of data for each environment in the labeld set.
      unlabeled_batches: list(dict); A batch of data for each environment in the
        unlabeld set.
      sampled_layer: str; Name of the layer on which mixup is applied.
      env_ids: list(int); List of labeled training environments ids.
      unlabeled_env_ids: list(int); List of unlabeled environments ids.

    Returns:
      Updated state of training and calculated metrics.

    """
    max_grad_norm = self.hparams.get('max_grad_norm', None)
    new_rng, rng = jax.random.split(train_state.rng)

    # bind the rng to the host/device we are on.
    model_rng = pipeline_utils.bind_rng_to_host_device(
        rng, axis_name='batch', bind_to=['host', 'device'])

    train_loss_fn = functools.partial(
        self.training_loss_fn,
        train_state=train_state,
        teacher_train_state=teacher_train_state,
        batch=batch,
        unlabeled_batch=unlabeled_batches,
        dropout_rng=model_rng,
        env_ids=env_ids,
        unlabeled_env_ids=unlabeled_env_ids,
        sampled_layer=sampled_layer)

    new_train_state, metrics_dict = self.compute_grads_and_update(
        batch, env_ids, max_grad_norm, new_rng, train_loss_fn, train_state)

    return new_train_state, metrics_dict

  def get_learning_rate(self, step):
    if self.hparams.get('restart_learning_rate'):
      step = step % self.iter_total_steps
    lr = self.learning_rate_fn(step)

    return lr
