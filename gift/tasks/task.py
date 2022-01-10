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

"""Basic Task classes."""

import functools
import itertools

from absl import logging
from flax.deprecated import nn
from flax.training import common_utils
import jax
import jax.numpy as jnp
import scipy

from gift.data import all_datasets
from gift.tasks import all_metrics
from gift.tasks import domain_mapping_utils
from gift.tasks import metrics


class Task(object):
  """Base Task class.

    Task objects contain all the information about the objective of the
    training, evaluation metrics, and the dataset.
  """

  def __init__(self, task_params, num_shards, regularisers=None):
    """Init task objects.

    Args:
      task_params: ConfigDict; hyperparameters of the task.
      num_shards: int; Number of shards used for data parallelization (should
        normally be set to `jax.device_count()`).
      regularisers: list of functions; List of auxilary losses that get module
        parameters as input (L2 loss is handled seperately).
    """
    self.task_params = task_params
    self.dataset_name = task_params.get('dataset_name')

    self.regularisers = regularisers
    self.load_dataset(self.dataset_name, num_shards)
    self.task_params.output_dim = self.dataset.meta_data['num_classes']

  def load_dataset(self, dataset_name, num_shards):
    """Loads the dataset for the task.

    Args:
      dataset_name: str; Name of the dataset.
      num_shards: int; Number of shards used for data parallelization (should
        normally be set to `jax.device_count()`).
    """
    self.dataset = all_datasets.get_dataset(dataset_name)(
        batch_size=self.task_params.local_batch_size,
        eval_batch_size=self.task_params.eval_local_batch_size,
        resolution=self.task_params.get('resolution', None),
        data_augmentations=self.task_params.get('data_augmentations', None),
        teacher_data_augmentations=self.task_params.get(
            'teacher_data_augmentations', None),
        num_shards=num_shards)

  def loss_function(self, logits, batch, model_params=None, step=None):
    raise NotImplementedError

  def metrics_fn(self, logits, batch):
    raise NotImplementedError

  def get_l2_rate(self, step):
    del step
    return self.task_params.get('l2_decay_factor')


class ClassificationTask(Task):
  """Classification Task."""

  def __init__(self, task_params, num_shards):
    """Initializing Classification based Tasks.

    Args:
      task_params: configdict; Hyperparameters of the task.
      num_shards: int; Number of deviced that we shard the batch over.
    """
    super().__init__(task_params, num_shards)
    loss_fn_name = self.task_params.get('main_loss', None)

    if loss_fn_name is None:
      if self.dataset.meta_data['num_classes'] == 1:
        # Use the loss function for binary classification.
        loss_fn_name = 'sigmoid_cross_entropy'
      else:
        loss_fn_name = 'categorical_cross_entropy'

    self.main_loss_fn = functools.partial(metrics.weighted_loss,
                                          all_metrics.ALL_LOSSES[loss_fn_name])

  _METRICS = all_metrics.CLASSIFICATION_METRICS

  def metrics_fn(self, logits, batch):
    """Calculates metrics for the classification task.

    Args:
      logits: float array; Output of the model->[batch, length, num_classes].
      batch: dict; Batch of data that has 'label' and optionally 'weights'.

    Returns:
      a dict of metrics.
    """
    target_is_onehot = logits.shape == batch['label'].shape
    if target_is_onehot:
      one_hot_targets = batch['label']
    else:
      one_hot_targets = common_utils.onehot(batch['label'], logits.shape[-1])

    if self.dataset.meta_data['num_classes'] == 1:
      # If this is a binary classification task, make sure the shape of labels
      # is (bs, 1) and is the same as the shape of logits.
      one_hot_targets = jnp.reshape(one_hot_targets, logits.shape)

    if self.task_params.get('class_indices'):
      possible_labels_indices = self.task_params.get('class_indices')
      one_hot_targets = one_hot_targets[:, possible_labels_indices]
      logits = logits[:, possible_labels_indices]

    weights = batch.get('weights')  # weights might not be defined
    metrics_dic = {}
    for key in self._METRICS:
      metric_val, metric_normalizer = self._METRICS[key](logits,
                                                         one_hot_targets,
                                                         weights)
      metrics_dic[key] = (jax.lax.psum(metric_val, 'batch'),
                          jax.lax.psum(metric_normalizer, 'batch'))

    # Store dataset related factors.
    for key in batch:
      if 'factor' in key:
        factors = batch[key]
        if weights is not None:
          val = jnp.sum(metrics.apply_weights(factors, weights))
          norm = jnp.sum(weights)
        else:
          val = jnp.sum(factors)
          norm = len(factors)

        metrics_dic[key] = (jax.lax.psum(val,
                                         'batch'), jax.lax.psum(norm, 'batch'))

    return metrics_dic

  def loss_function(self, logits, batch, model_params=None, step=None):
    """Return cross entropy loss with an L2 penalty on the weights."""
    weights = batch.get('weights')

    if self.dataset.meta_data['num_classes'] == 1:
      # If this is a binary classification task, make sure the shape of labels
      # is (bs, 1) and is the same as the shape of logits.
      targets = jnp.reshape(batch['label'], logits.shape)
    elif batch['label'].shape[-1] == self.dataset.meta_data['num_classes']:
      # If the labels are already the shape of (bs, num_classes) use them as is.
      targets = batch['label']
    else:
      # Otherwise convert the labels to onehot labels.
      targets = common_utils.onehot(batch['label'], logits.shape[-1])

    loss_value, loss_normalizer = self.main_loss_fn(
        logits,
        targets,
        weights,
        label_smoothing=self.task_params.get('label_smoothing'))

    total_loss = loss_value / loss_normalizer

    if model_params:
      l2_decay_factor = self.get_l2_rate(step)
      if l2_decay_factor is not None:
        l2_loss = metrics.l2_regularization(
            model_params,
            include_bias_terms=self.task_params.get('l2_for_bias', False))
        total_loss = total_loss + 0.5 * l2_decay_factor * l2_loss

      if self.regularisers:
        for reg_fn in self.regularisers:
          total_loss += reg_fn(model_params)

    return total_loss


class MultiEnvClassificationTask(ClassificationTask):
  """Multi environment classification Task."""

  _METRICS = all_metrics.MULTI_ENV_CLASSIFICATION_METRICS

  def load_dataset(self, dataset_name, num_shards):
    """Loads the dataset for the task.

    Args:
      dataset_name: str; Name of the dataset.
      num_shards: int; Number of shards used for data parallelization (should
        normally be set to `jax.device_count()`).
    """
    self.dataset = all_datasets.get_dataset(dataset_name)(
        batch_size=self.task_params.local_batch_size,
        eval_batch_size=self.task_params.eval_local_batch_size,
        num_shards=num_shards,
        resolution=self.task_params.get('resolution', None),
        data_augmentations=self.task_params.get('data_augmentations', None),
        teacher_data_augmentations=self.task_params.get(
            'teacher_data_augmentations', None),
        train_environments=self.task_params.train_environments,
        eval_environments=self.task_params.eval_environments)

  def aggregate_envs_losses(self, env_losses):
    """Aggregate losses of all environments.

    Args:
      env_losses: list(float); list of losses of the environments.

    Returns:
        Average of the env losses.
    """
    return jnp.mean(jnp.array(env_losses))

  def environments_penalties(self, env_logits, env_batches):
    """Computes a penalty term based on inconsistencies between different env.

    Args:
      env_logits: list(dict); List of logits for examples from different
        environment (env_logits[0] is the logits for examples from env 0 which
        is a float array of shape `[batch, length, num_classes]`).
      env_batches: list(dict); List of batches of examples from different
        environment (env_batches[0] is a batch dict for examples from env 0 that
        has 'label' and optionally 'weights'.).

    Returns:
      Environments penalty term for the loss.
    """
    del env_logits
    del env_batches

    return 0

  def penalty_weight(self, step):
    """Return the weight of the environments penalty term in the loss.

    Args:
      step: int; Number of training steps passed so far.

    Returns:
      float; Weight of the environment penalty term.
    """
    del step

    return 0

  def metrics_fn(self, env_logits, env_batches, env_ids, params):
    """Calculates metrics for the classification task.

    Args:
      env_logits: list(dict); List of logits for examples from different
        environment (env_logits[0] is the logits for examples from env 0 which
        is a float array of shape `[batch, length, num_classes]`).
      env_batches: list(dict); List of batches of examples from different
        environment (env_batches[0] is a batch dict for examples from env 0 that
        has 'label' and optionally 'weights'.).
     env_ids: list(int); List of environment codes.
     params: pytree; parameters of the model.

    Returns:
      a dict of metrics.
    """
    metrics_dic = {}
    envs_metrics_dic = {}
    # Add all the keys to envs_metrics_dic, each key will point to a list of
    # values from the correspondig metric for each environment.

    # Task related metrics
    for key in self._METRICS:
      envs_metrics_dic[key] = []

    # Dataset related metrics (e.g., perturbation factors)
    for key in env_batches[0]:
      if 'factor' in key:
        envs_metrics_dic[key] = []

    for i in range(len(env_logits)):
      logits = env_logits[i]
      batch = env_batches[i]
      env_name = self.dataset.get_full_env_name(self.dataset.id2env(env_ids[i]))
      env_metric_dic = super().metrics_fn(logits, batch)
      for key in env_metric_dic:
        metrics_dic[env_name + '/' + key] = env_metric_dic[key]
        envs_metrics_dic[key].append(env_metric_dic[key])
    # Add overall metric values over all environments,
    for key in self._METRICS:
      metrics_dic[key] = (jnp.sum(
          jnp.array(jnp.array(envs_metrics_dic[key])[:, 0])),
                          jnp.sum(
                              jnp.array(jnp.array(envs_metrics_dic[key])[:,
                                                                         1])))
    if params:
      metrics_dic['l2'] = metrics.l2_regularization(
          params, include_bias_terms=self.task_params.get('l2_for_bias', False))

    return metrics_dic

  def get_env_losses(self, env_logits, env_batches):
    """Computes and return the loss on each environment.

    Args:
      env_logits: list(dict); List of logits for examples from different
        environment (env_logits[0] is the logits for examples from env 0).
      env_batches: list(dict); List of batches of examples from different
        environment (env_batches[0] is a batch dict for examples from env 0).

    Returns:
      List of loss values in all environments.
    """
    env_losses = []
    for i in range(len(env_logits)):
      logits = env_logits[i]
      batch = env_batches[i]
      ce_loss = super().loss_function(logits, batch)
      env_losses.append(ce_loss)

    return env_losses

  def loss_function(self, env_logits, env_batches, model_params=None, step=0):
    """Returns loss with an L2 penalty on the weights.

    Args:
      env_logits: list(dict); List of logits for examples from different
        environment (env_logits[0] is the logits for examples from env 0).
      env_batches: list(dict); List of batches of examples from different
        environment (env_batches[0] is a batch dict for examples from env 0).
      model_params: dict; Parameters of the model (used to commpute l2).
      step: int; Global training step.

    Returns:
      Total loss.
    """
    env_losses = self.get_env_losses(env_logits, env_batches)
    total_loss = self.aggregate_envs_losses(env_losses)
    p_weight = self.penalty_weight(step)
    total_loss += p_weight * self.environments_penalties(
        env_logits, env_batches)

    if model_params:
      l2_decay_rate = self.get_l2_rate(step)
      if l2_decay_rate is not None:
        l2_loss = metrics.l2_regularization(
            model_params,
            include_bias_terms=self.task_params.get('l2_for_bias', False))
        total_loss = total_loss + 0.5 * l2_decay_rate * l2_loss

      if self.regularisers:
        for reg_fn in self.regularisers:
          reg_value = reg_fn(model_params)
          total_loss += reg_value

    # If p_weights > 1:
    # Rescale the entire loss to keep gradients in a reasonable range.
    total_loss /= jnp.maximum(p_weight, 1)

    return total_loss


class MultiEnvIRMClassificationTask(MultiEnvClassificationTask):
  """Multi environment task with IRM loss.

  Reference:[Invariant Risk Minimization](https://arxiv.org/abs/1907.02893)
  """

  def penalty_weight(self, step):
    """Return the weight of the environments penalty term in the loss.

      This is a step-function (assuming step is an integer).

      Here is what we want to do in this method:
      def f(step):
        if step < self.task_params.penalty_anneal_iters:
           return 1
        else:
           return self.task_params.penalty_weight

      Because this method is called within a jax pmapped function, we cannot
      use if-statements that depend on the input arguments. Luckily, the
      functionality we need can be implemented as the sum of two step functions
      which we can implement with min max operations, conditioned on the step
      argument and task_params.penalty_anneal_iters being integers.

      standard step function: 1 if x > b else 0 --> max(0, min(1, x-b))


    Args:
      step: int; Number of training steps passed so far.

    Returns:
      float; Weight of the environment penalty term.
    """

    if step is None:
      step = 0

    # make sure penalty_anneal_iters is an integer.
    assert self.task_params.penalty_anneal_iters == int(
        self.task_params.penalty_anneal_iters), ('The penalty_anneal_iters '
                                                 'param is not an integer.')

    b = self.task_params.penalty_anneal_iters
    w1 = jnp.maximum(0.0, jnp.minimum(1, b - step))
    w2 = jnp.maximum(0.0, jnp.minimum(
        1, step - b)) * self.task_params.penalty_weight

    return w1 + w2

  def environments_penalties(self, env_logits, env_batches):
    """Computes the penalty part of the IRM loss.

    Args:
      env_logits: list(dict); List of logits for examples from different
        environment (env_logits[0] is the logits for examples from env 0).
      env_batches: list(dict); List of batches of examples from different
        environment (env_batches[0] is a batch dict for examples from env 0).

    Returns:
      Average of environment penalties (IRM penalty).
    """
    penalties = []
    for i in range(len(env_logits)):
      logits = env_logits[i]
      batch = env_batches[i]
      weights = batch.get('weights')

      if self.dataset.meta_data['num_classes'] == 1:
        # If this is a binary classification task, make sure the shape of labels
        # is (bs, 1) and is the same as the shape of logits.
        targets = jnp.reshape(batch['label'], logits.shape)
      elif batch['label'].shape[-1] == self.dataset.meta_data['num_classes']:
        # If the labels are already the shape of (bs, num_classes) use them as
        # they are.
        targets = batch['label']
      else:
        # Otherwise convert the labels to onehot labels.
        targets = common_utils.onehot(batch['label'], logits.shape[-1])

      penalties.append(
          metrics.irm_env_penalty(
              logits=logits,
              targets=targets,
              weights=weights,
              loss_fn=self.main_loss_fn)[0])

    return jnp.mean(jnp.array(penalties))


class MultiEnvVRexClassificationTask(MultiEnvClassificationTask):
  """Multi environment task with V-Rex loss.

  Reference:
  [Out-of-Distribution Generalization via Risk Extrapolation]
  (https://arxiv.org/pdf/2003.00688.pdf)
  """

  def environments_penalties(self, env_losses):
    """Computes the penalty part of the V-Rex loss.

    Equation 9 in
    [Out-of-Distribution Generalization via Risk Extrapolation]
    (https://arxiv.org/pdf/2003.00688.pdf)

    Args:
      env_losses: list(float): Loss value for all the environments.

    Returns:
      V-Rex penalty (which is the variance of all losess).
    """
    return jnp.var(jnp.array(env_losses))

  def penalty_weight(self, step):
    """Return the weight of the environments penalty term in the loss.

      This is a step-function (assuming step is an integer).

      Here is what we want to do in this method:
      def f(step):
        if step < self.task_params.penalty_anneal_iters:
           return 1
        else:
           return self.task_params.penalty_weight

      Because this method is called within a jax pmapped function, we cannot
      use if-statements that depend on the input arguments. Luckily, the
      functionality we need can be implemented as the sum of two step functions
      which we can implement with min max operations, conditioned on the step
      argument and task_params.penalty_anneal_iters being integers.

      standard step function: 1 if x > b else 0 --> max(0, min(1, x-b))


    Args:
      step: int; Number of training steps passed so far.

    Returns:
      float; Weight of the environment penalty term.
    """

    if step is None:
      step = 0

    # make sure penalty_anneal_iters is an integer.
    assert self.task_params.penalty_anneal_iters == int(
        self.task_params.penalty_anneal_iters), ('The penalty_anneal_iters '
                                                 'param is not an integer.')

    b = self.task_params.penalty_anneal_iters
    w1 = jnp.maximum(0.0, jnp.minimum(1, b - step))
    w2 = jnp.maximum(0.0, jnp.minimum(
        1, step - b)) * self.task_params.penalty_weight

    return w1 + w2

  def loss_function(self, env_logits, env_batches, model_params=None, step=0):
    """Returns loss with an L2 penalty on the weights.

    Args:
      env_logits: list(dict); List of logits for examples from different
        environment (env_logits[0] is the logits for examples from env 0).
      env_batches: list(dict); List of batches of examples from different
        environment (env_batches[0] is a batch dict for examples from env 0).
      model_params: dict; Parameters of the model (used to commpute l2).
      step: int; Global training step.

    Returns:
      Total loss.
    """
    env_losses = self.get_env_losses(env_logits, env_batches)

    total_loss = self.aggregate_envs_losses(env_losses)
    p_weight = self.penalty_weight(step)
    total_loss += self.penalty_weight(step) * self.environments_penalties(
        env_losses)

    l2_decay_rate = self.get_l2_rate(step)
    if l2_decay_rate is not None:
      l2_loss = metrics.l2_regularization(
          model_params,
          include_bias_terms=self.task_params.get('l2_for_bias', False))
      total_loss += 0.5 * l2_decay_rate * l2_loss

    # If p_weights > 1:
    # Rescale the entire loss to keep gradients in a reasonable range.
    total_loss /= jnp.maximum(p_weight, 1)

    return total_loss


class MultiEnvLinearDomainMappingClassification(MultiEnvClassificationTask):
  """Multi environment task with Domain Mapping.

  Domain mapping adds an auxiliary loss that encourages
  the model to have equivariant representations with respect to the environment.
  """

  def get_transformer_module(self, hidden_reps_dim):

    class Linear(nn.Module):

      def apply(self, x):
        x = nn.Dense(x, hidden_reps_dim, name='l1', bias=True)
        return x

    return Linear

  def setup_transformers(self, hidden_reps_dim):
    """Sets up linear transformers for the auxiliary loss.

    Args:
      hidden_reps_dim: int; Dimensionality of the representational space (size
        of the representations used for computing the domain mapping loss.
    """
    transformer_class = self.get_transformer_module(hidden_reps_dim)
    self.state_transformers = {}
    env_keys = list(map(int, self.dataset.splits.train.keys()))
    # Get list of all possible environment pairs (this includes
    # different permutations).
    env_pairs = list(itertools.permutations(env_keys, 2))

    rng = nn.make_rng()
    for env_pair in env_pairs:
      rng, params_rng = jax.random.split(rng)
      _, init_params = transformer_class.init_by_shape(
          params_rng, [((1, hidden_reps_dim), jnp.float32)])
      self.state_transformers[env_pair] = nn.Model(transformer_class,
                                                   init_params)

  def loss_function(self,
                    env_logits,
                    env_reps,
                    env_batches,
                    env_ids,
                    model_params=None,
                    step=0,
                    env_aligned_pairs_idx=None):
    """Returns loss with an L2 penalty on the weights.

    Args:
      env_logits: list(dict); List of logits for examples from different
        environment (env_logits[0] is the logits for examples from env 0).
      env_reps: list; hidden reps for different environments (similar to
        env_logits).
      env_batches: list(dict); List of batches of examples from different
        environment (env_batches[0] is a batch dict for examples from env 0).
      env_ids: list(int): List of environment codes.
      model_params: dict; Parameters of the model (used to compute l2).
      step: int; Global training step.
      env_aligned_pairs_idx: dict; Environment pair --> alignment (if None the
        alignment is computed).

    Returns:
      Total loss.
    """
    total_loss = super().loss_function(
        env_logits=env_logits,
        env_batches=env_batches,
        model_params=model_params,
        step=step)

    if env_ids:
      # If env_ids is None, we do not compute domain_mapping_loss:
      total_loss += self.domain_mapping_loss(env_reps, env_batches, env_ids,
                                             env_aligned_pairs_idx)

    return total_loss

  def get_env_aligned_pairs_idx(self, env_reps, env_batches, env_ids):
    """Computes and returns aligned pairs.

    Args:
      env_reps: list; List of different envs  representations.
      env_batches: list; List of different envs batches.
      env_ids: list(int): List of environment codes.

    Returns:
      Aligned pairs indices (aligned rows, aligned columns).
    """

    env_pairs = list(itertools.permutations(env_ids, 2))
    env_aligned_pairs_idx = {}
    for pair in env_pairs:
      e1, e2 = pair
      # We only have state_transformer for training envs.
      if pair not in self.state_transformers:
        logging.warn('Pair %s is not in the training pairs set.', str(pair))
      else:
        e1_index = env_ids.index(e1)
        e2_index = env_ids.index(e2)
        e1_labels = env_batches[e1_index]['label']
        e2_labels = env_batches[e2_index]['label']
        # Get representations for env1.
        e1_reps = env_reps[e1_index]
        # Get representations for env1.
        e2_reps = env_reps[e2_index]
        # Transform representations from env1.
        transformed_e1 = self.state_transformers[pair](e1_reps)

        env_aligned_pairs_idx[pair] = self.align_batches(
            transformed_e1, e2_reps, e1_labels, e2_labels)

    return env_aligned_pairs_idx

  def get_bipartite_env_aligned_pairs_idx(self, env_reps, env_batches, env_ids,
                                          env_reps2, env_batches2, env_ids2):
    """Computes and returns aligned pairs between two sets of environments.

    Args:
      env_reps: list; List of different envs  representations.
      env_batches: list; List of different envs batches.
      env_ids: list(int): List of environment codes.
      env_reps2: list; List of different envs  representations.
      env_batches2: list; List of different envs batches.
      env_ids2: list(int): List of environment codes.

    Returns:
      Aligned pairs indices (aligned rows, aligned columns).
    """

    env_pairs = list(itertools.product(env_ids, env_ids2))

    env_aligned_pairs_idx = {}

    for pair in env_pairs:
      e1, e2 = pair
      # We only have state_transformer for training envs.
      if pair not in self.state_transformers:
        logging.warn('Pair %s is not in the training pairs set.', str(pair))
      else:
        e1_index = env_ids.index(e1)
        e2_index = env_ids2.index(e2)
        e1_labels = env_batches[e1_index]['label']
        e2_labels = env_batches2[e2_index]['label']
        # Get representations for env1.
        e1_reps = env_reps[e1_index]
        # Get representations for env1.
        e2_reps = env_reps2[e2_index]
        # Transform representations from env1.
        transformed_e1 = self.state_transformers[pair](e1_reps)

        env_aligned_pairs_idx[pair] = self.align_batches(
            transformed_e1, e2_reps, e1_labels, e2_labels)

        # Convert alignments which is the array of aligned indices to match mat.
        alignments = jnp.asarray(env_aligned_pairs_idx[pair])

        batch_size = alignments.shape[1]
        matching_matrix = jnp.zeros(
            shape=(batch_size, batch_size), dtype=jnp.float32)
        env_aligned_pairs_idx[pair] = matching_matrix.at[alignments[0],
                                                         alignments[1]].set(1.0)

    return env_aligned_pairs_idx

  def domain_mapping_loss(self,
                          env_reps,
                          env_batches,
                          env_ids,
                          env_aligned_pairs_idx=None):
    """Compute Linear Transformation Constraint loss.

    Args:
      env_reps: list; List of different envs  representations.
      env_batches: list; List of different envs batches.
      env_ids: list(int): List of environment codes.
      env_aligned_pairs_idx: dict; Environment pair --> alignment. (if None the
        alignment is computed).

    Returns:
      Domain mapping loss (float).
    """
    mask_loss_diff_labels = self.task_params.get('mask_loss_diff_labels')

    # Get all possible environment pairs
    env_pairs = list(itertools.permutations(env_ids, 2))
    aux_losses = []
    l2s = []
    for pair in env_pairs:
      e1, e2 = pair
      # We only have state_transformer for training envs.
      if pair not in self.state_transformers:
        logging.warn('Pair %s is not in the training pairs set.', str(pair))
      else:
        e1_index = env_ids.index(e1)
        e2_index = env_ids.index(e2)
        e1_labels = env_batches[e1_index]['label']
        e2_labels = env_batches[e2_index]['label']
        # Get representations for env1.
        e1_reps = env_reps[e1_index]
        # Get representations for env1.
        e2_reps = env_reps[e2_index]
        # Transform representations from env1.
        transformed_e1 = self.state_transformers[pair](e1_reps)

        if env_aligned_pairs_idx is None:
          aligned_pairs_idx = self.align_batches(transformed_e1, e2_reps,
                                                 e1_labels, e2_labels)
        else:
          aligned_pairs_idx = env_aligned_pairs_idx[pair]

        if mask_loss_diff_labels:
          # Assign zero/one weights to each example pair based on the alignment
          # of their labels.
          pair_weights = jnp.float32(e1_labels[aligned_pairs_idx[0]] ==
                                     e2_labels[aligned_pairs_idx[1]])
        else:
          pair_weights = jnp.ones_like(e1_labels, dtype='float32')

        # Compute domain mapping loss for the environment pair:
        # Get representations for env1.
        transformed_e1 = transformed_e1[aligned_pairs_idx[0]]
        # Get corresponding representations for env2.
        e2_reps = env_reps[e2_index][aligned_pairs_idx[1]]

        # Minimize the distance between transformed reps from env1 and reps
        # from env2.
        aux_losses.append(
            jnp.mean(
                jnp.linalg.norm(transformed_e1 - e2_reps, axis=-1) *
                pair_weights))

        # Add l2 loss for the transformer weights (to make sure it is as minimal
        # as possible.
        l2s.append(
            metrics.l2_regularization(
                self.state_transformers[pair].params,
                include_bias_terms=self.task_params.get('l2_for_bias', False)))

    if not aux_losses:
      aux_losses = [0]
      l2s = [0]

    alpha = self.task_params.get('aux_weight', .0)
    beta = self.task_params.get('aux_l2', .0)

    # Average and return the final weighted value of the loss.
    return alpha * jnp.mean(jnp.array(aux_losses)) + beta * jnp.mean(
        jnp.array(l2s))

  def align_batches(self, x, y, x_labels, y_labels, supervised=True):
    """Computes alignment between two mini batches.

    In the MultiEnvDomainMappingClassification, this calls the random alignment
    (based on labels) function.

    Args:
      x: jnp array; Batch of representations with shape '[bs, feature_size]'.
      y: jnp array; Batch of representations with shape '[bs, feature_size]'.
      x_labels: jnp array; labels of x with shape '[bs, 1]'.
      y_labels: jnp array; labels of y with shape '[bs, 1]'.
      supervised: bool; If False we can not use y_labels and it defaults back to
        random alignment otherwise it does label based alignment (tries to align
        examples that have similar labels).

    Returns:
      aligned indexes of x, aligned indexes of y.
    """
    del y
    # Get aligned example pairs.
    if supervised:
      rng = nn.make_rng()
      new_rngs = jax.random.split(rng, len(x_labels))
      aligned_pairs_idx = domain_mapping_utils.align_examples(
          new_rngs, x_labels, jnp.arange(len(x_labels)), y_labels)
    else:
      number_of_examples = len(x)
      rng = nn.make_rng()
      matching_matrix = jnp.eye(number_of_examples)
      matching_matrix = jax.random.permutation(rng, matching_matrix)

      aligned_pairs_idx = jnp.arange(len(x)), jnp.argmax(
          matching_matrix, axis=-1)

    return aligned_pairs_idx


class MultiEnvNonLinearDomainMappingClassification(
    MultiEnvLinearDomainMappingClassification):
  """Non linear Domain Mapping."""

  def get_transformer_module(self, hidden_reps_dim):

    class NonLinear(nn.Module):

      def apply(self, x):
        x = nn.Dense(x, hidden_reps_dim, bias=True, name='l1')
        x = nn.relu(x)
        x = nn.Dense(x, hidden_reps_dim, bias=True, name='l2')

        return x

    return NonLinear


class MultiEnvHungarianDomainMappingClassification(
    MultiEnvLinearDomainMappingClassification):
  """Non linear Domain Mapping."""

  def align_batches(self, x, y, x_labels, y_labels):
    """Computes alignment between two mini batches.

    In the MultiEnvHungarianDomainMappingClassification, this calls the
     hungarian matching function.


    Args:
      x: jnp array; Batch of representations with shape '[bs, feature_size]'.
      y: jnp array; Batch of representations with shape '[bs, feature_size]'.
      x_labels: jnp array; labels of x with shape '[bs, 1]'.
      y_labels: jnp array; labels of y with shape '[bs, 1]'.

    Returns:
      aligned indexes of x, aligned indexes of y.

    """

    label_cost = self.task_params.get('ot_label_cost', 0.)

    cost = domain_mapping_utils.pairwise_l2(x, y)

    # Adjust cost such that representations with different labels
    # get assigned a very high cost.
    same_labels = domain_mapping_utils.pairwise_equality_1d(x_labels, y_labels)
    adjusted_cost = cost + (1 - same_labels) * label_cost

    # `linear_sum_assignment`  computes cheapest hard alignment.
    x_ind, y_ind = scipy.optimize.linear_sum_assignment(adjusted_cost)

    return x_ind, y_ind


class MultiEnvIdentityDomainMappingClassification(
    MultiEnvLinearDomainMappingClassification):
  """Multi environment task with Indentity Domain Mapping.

  Domain mapping adds an auxiliary loss that encourages
  the model to have equivariant representations with respect to the environment.

  Using domain mapping with identity mapping simply means that the domain
  mapping loss is the L2 distance between examples from different domains.
  """

  def get_transformer_module(self, hidden_reps_dim):
    """Return the domain mapping module."""

    # TODO(samiraabnar): Find a way to avoid defining these naive mapping
    #  models.
    class Idenity(nn.Module):
      """Does nothing but returns the input itself."""

      def apply(self, x):
        return x

    return Idenity


class MultiEnvSinkhornDomainMappingClassification(
    MultiEnvIdentityDomainMappingClassification):
  """Multi env CLS with Sinkhorn-based matching."""

  def get_bipartite_env_aligned_pairs_idx(self, env_reps, env_batches, env_ids,
                                          env_reps2, env_batches2, env_ids2):
    """Computes and returns aligned pairs between two sets of environments.

    Args:
      env_reps: list; List of different envs  representations.
      env_batches: list; List of different envs batches.
      env_ids: list(int): List of environment codes.
      env_reps2: list; List of different envs  representations.
      env_batches2: list; List of different envs batches.
      env_ids2: list(int): List of environment codes.

    Returns:
      Aligned pairs indices (aligned rows, aligned columns).
    """

    env_pairs = list(itertools.product(env_ids, env_ids2))

    env_aligned_pairs_idx = {}

    for pair in env_pairs:
      e1, e2 = pair
      # We only have state_transformer for training envs.
      if pair not in self.state_transformers:
        logging.warn('Pair %s is not in the training pairs set.', str(pair))
      else:
        e1_index = env_ids.index(e1)
        e2_index = env_ids2.index(e2)
        e1_labels = env_batches[e1_index]['label']
        e2_labels = env_batches2[e2_index]['label']
        # Get representations for env1.
        e1_reps = env_reps[e1_index]
        # Get representations for env2.
        e2_reps = env_reps2[e2_index]
        # Transform representations from env1.
        transformed_e1 = self.state_transformers[pair](e1_reps)

        env_aligned_pairs_idx[pair] = self.align_batches(
            transformed_e1, e2_reps, e1_labels, e2_labels)

    return env_aligned_pairs_idx

  def align_batches(self, x, y, x_labels, y_labels):
    """Computes optimal transport between two batches with Sinkhorn algorithm.

    This calls a sinkhorn solver in dual (log) space with a finite number
    of iterations and uses the dual unregularized transport cost as the OT cost.

    Args:
      x: jnp array; Batch of representations with shape '[bs, feature_size]'.
      y: jnp array; Batch of representations with shape '[bs, feature_size]'.
      x_labels: jnp array; labels of x with shape '[bs, 1]'.
      y_labels: jnp array; labels of y with shape '[bs, 1]'.

    Returns:
      ot_cost: scalar optimal transport loss.
    """

    epsilon = self.task_params.get('sinkhorn_eps', 0.1)
    num_iters = self.task_params.get('sinkhorn_iters', 50)
    label_weight = self.task_params.get('ot_label_cost', 0.)
    l2_weight = self.task_params.get('ot_l2_cost', 0.)
    noise_weight = self.task_params.get('ot_noise_cost', 1.0)
    x = x.reshape((x.shape[0], -1))
    y = y.reshape((x.shape[0], -1))

    # Solve sinkhorn in log space.
    num_x = x.shape[0]
    num_y = y.shape[0]

    x = x.reshape((num_x, -1))
    y = y.reshape((num_y, -1))

    # Marginal of rows (a) and columns (b)
    a = jnp.ones(shape=(num_x,), dtype=x.dtype)
    b = jnp.ones(shape=(num_y,), dtype=y.dtype)

    # TODO(samiraabnar): Check range of l2 cost?
    cost = domain_mapping_utils.pairwise_l2(x, y)

    # Adjust cost such that representations with different labels
    # get assigned a very high cost.
    same_labels = domain_mapping_utils.pairwise_equality_1d(x_labels, y_labels)
    adjusted_cost = (1 - same_labels) * label_weight + l2_weight * cost

    # Add noise to the cost.
    adjusted_cost += noise_weight * jax.random.uniform(
        nn.make_rng(), minval=0, maxval=1.0)
    _, matching, _ = domain_mapping_utils.sinkhorn_dual_solver(
        a, b, adjusted_cost, epsilon, num_iters)
    matching = domain_mapping_utils.round_coupling(matching, a, b)
    if self.task_params.get('interpolation_mode', 'hard') == 'hard':
      matching = domain_mapping_utils.sample_best_permutation(
          nn.make_rng(), coupling=matching, cost=adjusted_cost)

    return matching

  def ot_loss(self, x, y, x_labels, y_labels):
    """Computes optimal transport between two batches with Sinkhorn algorithm.

    This calls a sinkhorn solver in dual (log) space with a finite number
    of iterations and uses the dual unregularized transport cost as the OT cost.

    Args:
      x: jnp array; Batch of representations with shape '[bs, feature_size]'.
      y: jnp array; Batch of representations with shape '[bs, feature_size]'.
      x_labels: jnp array; labels of x with shape '[bs, 1]'.
      y_labels: jnp array; labels of y with shape '[bs, 1]'.

    Returns:
      ot_cost: scalar optimal transport loss.
    """

    epsilon = self.task_params.get('sinkhorn_eps', 0.1)
    num_iters = self.task_params.get('sinkhorn_iters', 100)
    label_cost = self.task_params.get('ot_label_cost', 0.)

    # Solve sinkhorn in log space.
    num_x = x.shape[0]
    num_y = y.shape[0]

    x = x.reshape((num_x, -1))
    y = y.reshape((num_y, -1))

    # Marginal of rows (a) and columns (b)
    a = jnp.ones(shape=(num_x,), dtype=x.dtype) / float(num_x)
    b = jnp.ones(shape=(num_y,), dtype=y.dtype) / float(num_y)
    cost = domain_mapping_utils.pairwise_l2(x, y)

    # Adjust cost such that representations with different labels
    # get assigned a very high cost.
    same_labels = domain_mapping_utils.pairwise_equality_1d(x_labels, y_labels)
    # adjusted_cost = same_labels * cost + (1 - same_labels) * (
    #     label_cost * jnp.max(cost))
    adjusted_cost = cost + (1 - same_labels) * label_cost
    ot_cost, _, _ = domain_mapping_utils.sinkhorn_dual_solver(
        a, b, adjusted_cost, epsilon, num_iters)

    return ot_cost

  def domain_mapping_loss(self,
                          env_reps,
                          env_batches,
                          env_ids,
                          env_aligned_pairs_idx=None):
    """Compute Linear Transformation Constraint loss.

    Args:
      env_reps: list; List of different envs  representations.
      env_batches: list; List of different envs batches.
      env_ids: list(int): List of environment codes.
      env_aligned_pairs_idx: Ignored. Is only here to ensure compatibility with
        the method "loss_function" which is defined in the parent class.

    Returns:
      domain mapping scalar loss (averaged over all environments).
    """
    del env_aligned_pairs_idx

    # Get all possible environment pairs
    env_pairs = list(itertools.permutations(env_ids, 2))
    aux_losses = []
    l2s = []
    for pair in env_pairs:
      e1, e2 = pair
      # We only have state_transformer for training envs.
      if pair not in self.state_transformers:
        logging.warn('Pair %s is not in the training pairs set.', str(pair))
      else:
        e1_index = env_ids.index(e1)
        e2_index = env_ids.index(e2)
        e1_labels = env_batches[e1_index]['label']
        e2_labels = env_batches[e2_index]['label']
        # Get representations for env1.
        e1_reps = env_reps[e1_index]
        # Get representations for env1.
        e2_reps = env_reps[e2_index]
        # Transform representations from env1.
        transformed_e1 = self.state_transformers[pair](e1_reps)

        ot_cost = self.ot_loss(transformed_e1, e2_reps, e1_labels, e2_labels)

        aux_losses.append(ot_cost)

        # Add l2 loss for the transformer weights (to make sure it is as minimal
        # as possible.
        l2s.append(
            metrics.l2_regularization(
                self.state_transformers[pair].params,
                include_bias_terms=self.task_params.get('l2_for_bias', False)))

    if not aux_losses:
      aux_losses = [0]
      l2s = [0]

    alpha = self.task_params.get('aux_weight', .0)
    beta = self.task_params.get('aux_l2', .0)

    # Average and return the final weighted value of the loss.
    return alpha * jnp.mean(jnp.array(aux_losses)) + beta * jnp.mean(
        jnp.array(l2s))


class MultiEnvDannClassification(MultiEnvVRexClassificationTask):
  """Task class for Domain Adverserial NNs."""

  def dann_loss(self, env_logits, env_labels, env_batches):
    """Compute DANN loss.

    Reference: https://jmlr.org/papers/volume17/15-239/15-239.pdf

    Args:
      env_logits: list; Domain logits for all labeled environments. This is the
        output of the domain discriminator module.
      env_labels: list; Domain Labels.
      env_batches: list(dict); List of batches of examples of all labeled
        environments.

    Returns:
      Dann loss.
    """
    # Domain CLS loss function:
    loss_fn = functools.partial(
        metrics.weighted_loss,
        all_metrics.ALL_LOSSES['categorical_cross_entropy'])

    # Agregate domain discriminator loss for all environments:
    env_dann_losses = []
    for i in range(len(env_logits)):
      batch = env_batches[i]
      loss_value, loss_normalizer = loss_fn(env_logits[i], env_labels[i],
                                            batch.get('weights'))
      loss = loss_value / loss_normalizer
      loss = jax.lax.cond(
          loss_normalizer > 0, lambda l: l, lambda _: 0.0, operand=loss)
      env_dann_losses.append(loss)

    total_dann_loss = jnp.mean(jnp.asarray(env_dann_losses))

    return total_dann_loss

  def loss_function(self,
                    env_logits,
                    env_batches,
                    all_env_logits,
                    all_env_labels,
                    all_env_batches,
                    dann_factor,
                    model_params=None,
                    step=0):
    """Returns loss with an L2 penalty on the weights.

    Args:
      env_logits: list(dict); List of logits for examples from different
        environment (env_logits[0] is the logits for examples from env_ids[0]).
      env_batches: list(dict); List of batches of examples from different
        environment (env_batches[0] is a batch dict for examples from
        env_ids[0]).
      all_env_logits: list; Domain logits for all environments.
      all_env_labels: list(dict); Domain labels for all environments.
      all_env_batches: list(int): List of all environment batches.
      dann_factor: float; DANN factor.
      model_params: dict; Parameters of the model (used to compute l2).
      step: int; Global training step.

    Returns:
      Total loss.
    """
    total_loss = super().loss_function(
        env_logits=env_logits,
        env_batches=env_batches,
        model_params=model_params,
        step=step)

    total_loss += self.dann_loss(all_env_logits, all_env_labels,
                                 all_env_batches) * dann_factor

    return total_loss
