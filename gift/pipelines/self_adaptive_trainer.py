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

"""Self adaptive Trainer classes."""

import functools
import time

from absl import logging
from flax import jax_utils
from flax.deprecated import nn
from flax.training import common_utils
import jax
import jax.numpy as jnp

from gift.pipelines import multi_env_end2end
from gift.pipelines import pipeline_utils
from gift.tasks import metrics
from gift.train_lib import optimizers


class SelfAdaptiveGradualTrainer(multi_env_end2end.MultiEnvEnd2End):
  """Self adaptive gradual training pipeline.

  In this pipeline, a model is first trained on one or a set of labeled datasets
  with ground truth labels sequentially (the model trains on one environment at
  a time). Then the model is gradually (again one environment at a time) adapted
  to the unlabeled environments with self-training. We assume there is natural
  order in the environments provided to this pipeline with respect to the amount
  of distribution shift in the data.

  Reference:
  [Understanding Self-Training for Gradual Domain Adaptation]
  (https://arxiv.org/abs/2002.11361)

  """

  def __init__(self, model_cls, task, hparams, experiment_dir,
               tb_summary_writer, rng):

    super().__init__(model_cls, task, hparams, experiment_dir,
                     tb_summary_writer, rng)
    self.hparams.keep_env_ckpts = self.hparams.get('keep_env_ckpts', False)
    self.includes_self_supervision = True

    # Set train env (On this environment we use ground truth labels to train
    # the model).
    self.labeled_envs = hparams.get('labeled_environments',
                                    None) or task.dataset.train_environments
    self.unlabeled_envs = hparams.get('unlabeled_environments', [])

    self.set_pseudo_label_generator()
    self.lr_start_step = 0

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
      self.lr_start_step = jax_utils.unreplicate(self.train_state.global_step)
      optimizer = optimizers.get_optimizer(self.hparams).create(flax_model)
    else:
      optimizer = optimizer.replace(target=flax_model)

    optimizer = jax_utils.replicate(optimizer)
    self.train_state = self.train_state.replace(optimizer=optimizer)

  def train(self):
    """Training loop."""
    master = jax.host_id() == 0
    train_summary, eval_summary = None, None
    global_start_step = self.start_step

    # current_step keeps track of global (cumulative) number of steps the model
    # is trained on all of the environments so that we know the starting
    # step for the next environments.
    current_step = 0

    # Train on unlabled data.
    for labeled_env in self.labeled_envs:
      # TODO(samiraabnar): Encapsulate this in the Task class.
      labeled_env = str(self.task.dataset.env2id(labeled_env))
      labeled_env_dict = {
          labeled_env: self.task.dataset.data_iters.train[labeled_env]
      }

      # Set start and end step for the current environment.
      env_total_steps = self.get_env_total_steps(
          labeled_env, self.hparams.num_training_epochs)
      env_start_step = current_step
      env_end_step = env_start_step + env_total_steps
      if global_start_step < env_end_step:
        # Resume or start training on this environment if we haven't already
        # trained on it or stopped in the middle of it.

        # Update env_start_step if the preemption has occured in the middle of
        # training on this environments.
        env_start_step += jnp.maximum(0, global_start_step - env_start_step)
        eval_summary, train_summary = self._train_loop(labeled_env_dict,
                                                       env_start_step,
                                                       env_end_step, master)

      current_step += env_total_steps

    # Reset pseudo label generator to use the newly trained model.
    teacher_train_state = self.train_state.clone()
    pseudo_label_generator = functools.partial(
        self.pseudo_label_generator, train_state=teacher_train_state)
    self.task.dataset.reset_pseudo_label_generator(pseudo_label_generator)

    # Self supervised adaptation (Note the order of the self.unlabeled_envs is
    # important).
    if self.unlabeled_envs is None:
      self.unlabeled_envs = [
          env for env in self.task.dataset.train_environments
          if env not in self.labeled_envs
      ]
    for unlabeled_env in self.unlabeled_envs:
      self.maybe_reset_train_state()

      unlabeled_env = str(self.task.dataset.env2id(unlabeled_env))
      unlabeled_env_dict = {
          unlabeled_env: self.task.dataset.data_iters.train[unlabeled_env]
      }

      env_total_steps = self.get_env_total_steps(
          unlabeled_env, self.hparams.num_unlabeled_training_epochs)
      logging.info('Env total steps: %d', env_total_steps)
      env_start_step = current_step
      env_end_step = env_start_step + env_total_steps
      if global_start_step < env_end_step:
        # Resume or start training on this environment if we haven't already
        # trained on it or stopped in the middle of it.
        # Update env_start_step if the preemtion has occured in the middle of
        # training on this environments.
        env_start_step += jnp.maximum(0, global_start_step - env_start_step)
        logging.info('train on %s', unlabeled_env)
        eval_summary, train_summary = self._train_loop(unlabeled_env_dict,
                                                       env_start_step,
                                                       env_end_step, master)

      # Reset pseudo label generator to use the newly trained model.
      teacher_train_state = self.train_state.clone()
      pseudo_label_generator = functools.partial(
          self.pseudo_label_generator, train_state=teacher_train_state)
      self.task.dataset.reset_pseudo_label_generator(pseudo_label_generator)

      # Reset gift regularizer's init point.
      if self.hparams.get('gift_factor', None):
        self.task.regularisers = [
            functools.partial(
                metrics.parameter_distance,
                base_params=jax_utils.unreplicate(
                    teacher_train_state).optimizer.target.params,
                norm_factor=self.hparams.get('gift_factor'),
                mode='l2')
        ]

      current_step += env_total_steps

    # Wait until computations are done before exiting (for timing!).
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

    # return the train and eval summary after last step for regresesion testing
    return train_summary, eval_summary

  def get_env_total_steps(self, labeled_env, epochs):
    env_n_exmpls = self.task.dataset.splits['train'][labeled_env].num_examples
    steps_per_epoch = env_n_exmpls // self.hparams.batch_size + 1
    env_total_steps = (steps_per_epoch * epochs)
    return env_total_steps

  def _train_loop(self, environments, start_step, end_step, master):
    """Training loop.

    Trains the model on the given environment set for (end_step - start_step)
    number of steps.

    Args:
      environments: dict; A dictionary from environment name to environment data
        iterator.
      start_step: int; Staring step in the loop.
      end_step: int; End step in the loop.
      master: bool; Is this the host device? If yes, log and checkpoint.

    Returns:
      Evaluation summaries and metrics.
    """
    # Initialize return values.
    train_metrics = []
    train_summary, eval_summary = None, None
    tick = time.time()

    eval_env_ids = list(
        map(int, self.task.dataset.data_iters.validation.keys()))
    train_env_ids, train_iters = list(zip(*dict(environments).items()))
    train_env_ids = list(map(int, train_env_ids))

    for step in range(start_step + 1, end_step + 1):

      # Get next batch.
      train_batch = self.get_next_batch(train_iters)

      # Run train step and get the metrics and the new train state.
      self.train_state, t_metrics = self.pmapped_train_step(
          self.train_state, train_batch, train_env_ids)
      train_metrics.append(t_metrics)

      if (step % self.eval_frequency == 0) or (step == end_step):
        train_metrics = common_utils.get_metrics(train_metrics)
        train_summary = pipeline_utils.compute_global_mean_metrics(
            train_metrics)

        tock = time.time()
        steps_per_sec = self.eval_frequency / (tock - tick)
        tick = tock

        # Log train summary:
        if master:
          self.write_train_summary(
              step=step,
              metric_dict=train_metrics,
              summary=train_summary,
              steps_per_sec=steps_per_sec)

        # Reset metric accumulation for next evaluation cycle.
        train_metrics = []

        # Sync model state across replicas.
        self.train_state = pipeline_utils.sync_model_state_across_replicas(
            self.train_state)

        # Evaluate and log the results.
        eval_summary, self.train_state = self.eval(step, self.train_state,
                                                   eval_env_ids)

      # Sync and save.
      self.checkpoint(self.train_state, step)

    return eval_summary, train_summary

  def get_learning_rate(self, step):
    lr = self.learning_rate_fn(step - self.lr_start_step)

    return lr
