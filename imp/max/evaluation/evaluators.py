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

"""Helper evaluation modules."""

import collections
import functools
from typing import Any, Callable, Iterator

from absl import logging
from flax.training import train_state as flax_train_state
import jax
from jax import numpy as jnp
from jax.experimental import multihost_utils
import numpy as np
import optax

from imp.max.core import constants
from imp.max.evaluation import config as eval_config
from imp.max.evaluation import metrics
from imp.max.execution import partitioning
from imp.max.modeling import heads
from imp.max.utils import sharding
from imp.max.utils import tree as mtu
from imp.max.utils import typing


DataFeatureType = constants.DataFeatureType
DataFeatureRoute = constants.DataFeatureRoute
DataFeatureName = constants.DataFeatureName
Modality = constants.Modality
OUTPUTS = DataFeatureType.OUTPUTS
TARGETS = DataFeatureType.TARGETS
COMMON_SPACE = DataFeatureRoute.COMMON_SPACE
ENCODER = DataFeatureRoute.ENCODER
LABEL_CLASSIFIER = DataFeatureRoute.LABEL_CLASSIFIER
TOKEN_RAW = DataFeatureName.TOKEN_RAW
TOKEN_ID = DataFeatureName.TOKEN_ID
FEATURES_AGG = DataFeatureName.FEATURES_AGG
LABEL = DataFeatureName.LABEL
LOGITS = DataFeatureName.LOGITS


def _get_modality_with_label(routed_targets,
                             dataset_name):
  """Checks if the targets contain one and only one label annotation."""
  modalities_with_label = []
  for modality in routed_targets:
    if LABEL in routed_targets[modality]:
      modalities_with_label.append(modality)
  if len(modalities_with_label) > 1:
    raise ValueError(
        'More than one modality has `label` annotation in dataset '
        f'`{dataset_name}`. By default it is assumed that only one modality '
        f'can contain `label` annotation, {modalities_with_label=}.')

  if not modalities_with_label:
    raise ValueError(
        f'Dataset `{dataset_name}` does not contain any modalities with `label`'
        ' annotation.')

  modality = modalities_with_label.pop()
  return modality


def _verify_single_instance(array):
  """Verifies if a given array has only one instance (in its second dim)."""
  if array.shape[1] != 1:
    raise ValueError(
        'Only single-instance labels are supported. Labels with shape '
        f'{array.shape} were provided.')


def _get_num_steps(num_samples, batch_size, total_epochs):
  """Calculates the number of steps given batch size and total epochs."""

  if total_epochs < 1:
    raise ValueError(
        f'The requested {total_epochs=} does not cover all train samples.')

  steps_per_epoch = np.ceil(num_samples / batch_size).astype(int)
  num_steps = total_epochs * steps_per_epoch

  if steps_per_epoch < 1:
    raise ValueError(
        f'Total {num_samples=} is smaller than the requested {batch_size=}.')

  return num_steps


def group_subsets(subsets):
  """Groups data subsets based on their splits (if any)."""

  trains = sorted([subset for subset in subsets if subset.startswith('train')])
  tests = sorted([subset for subset in subsets
                  if subset.startswith('test') or subset.startswith('valid')])
  if len(trains) != len(tests):
    raise AssertionError('Train and test subsets should have the same length, '
                         f'got {trains} and {tests}')

  grouped_subsets = [(train, test) for train, test in zip(trains, tests)]

  return grouped_subsets


def aggregate_multi_step_data(
    all_data,
    scalar_aggregate_fn = np.mean,
    batch_aggregate_fn = np.concatenate,
    batch_axis = 0,
):
  """Aggregate all data across batched steps.

  This function takes in a list of nested arrays (all with the same tree)
  structure. This list represents multiple steps in a full inference loop.
  It aggregates leaves across those steps and returns a same-structure tree
  whose leaves are the result of that aggregating.
  This is useful when iterating over a dataset and accumulating the resulting
  outputs in a single nested array.

  Args:
    all_data: A list of nested arrays. All nested arrays should have the same
      tree structure.
    scalar_aggregate_fn: A callable function that will be applied on any scalar
      value across multiple steps. This function should expect a list of arrays
      with rank=0.
    batch_aggregate_fn: A callable function that will be applied on any non-
      scalar array across multiple steps. This function should expect a lit of
      arrays whith rank > 0.
    batch_axis: The expected aggregation axis for the non-scalar arrays.

  Returns:
    A nested array with the same structure as those in the input list, whose
    leaves (arrays) are the result of aggregating the same leaves across
    multiple steps.
  """

  num_steps = len(all_data)

  # Flatten per-step tree.
  multi_step_leaves = [
      jax.tree.flatten(all_data[n])[0] for n in range(num_steps)
  ]
  tree_structure = jax.tree.structure(all_data[0])

  # Iterate over leaves and concatenate across steps.
  aggregated_leaves = []
  for n_leaf in range(tree_structure.num_leaves):
    # Fetch the values of the same leaf across all steps
    multi_step_leaf = [
        multi_step_leaves[n_step][n_leaf] for n_step in range(num_steps)
    ]

    # Check if the leaf-of-interest is scalar or non-scalar
    if multi_step_leaf[0].shape:
      leaf_aggregate_fn = functools.partial(batch_aggregate_fn, axis=batch_axis)
    else:
      leaf_aggregate_fn = scalar_aggregate_fn

    # Apply the proper aggregation function
    aggregated_leaf = leaf_aggregate_fn(multi_step_leaf)

    # Accumulate back the resulting aggregated leaf to reconstruct the tree
    aggregated_leaves.append(aggregated_leaf)

  # Reconstruct a nested array with the same tree structure as of those in the
  # 'all_data' list, but with the aggregated values as their leaves
  aggregated_data = jax.tree.unflatten(
      tree_structure, aggregated_leaves)
  return aggregated_data


class BaseLinearClassifier:
  """The base linear classifier providing certain methods."""

  def __init__(self,
               batch_size = 128,
               total_steps = None,
               total_epochs = None,
               learning_rate = 0.0001,
               regularization = 0.,
               input_noise_std = 0.,
               seed = 0,
               verbose = 0,
               host_local_data = True,
               name = 'linear_classifier'):
    """Instantiates a linear classifier head with train/eval parameters.

    Args:
      batch_size: A positive integer number indicating the total number of
        samples per train step.
      total_steps: An optional positive integer number indicating the total
       number of training steps. Either this number or `total_epochs` should be
       specified.
      total_epochs: An optional positive integer number indicating the total
       number of training epochs. Either this number or `total_steps` should be
       specified.
      learning_rate: A positive float OR an optax Schedule module that returns
        the learning rate of the training pipeline at a given step.
      regularization: A positive float number indicating the weight of the
        l2 regularization term in the total loss. If non-zero provided, it
        will add norm-2 of the linear weight (multiplied by this term) to
        the softmax cross-entropy loss value.
      input_noise_std: A positive float number indicating the standard
        deviation of an AGN (Additive Gaussian Noise) to the inputs.
      seed: A positive integer number indicating the seed of the pseudo-random
        generator (jax.random.key(seed)).
      verbose: A 0/1 integer number for controlling the logging level of this
        classifier. If 0 is provided, no logging will be generated. If 1 is
        provided, it will log the metrics every 50 train/eval steps.
      host_local_data: If True, inputs are expected to be host local and will
        be sharded to a global jax.Array.
      name: A string, indicating the name of the module.
    """
    if total_steps is None and total_epochs is None:
      raise ValueError('One of `total_steps` or `total_epochs` should be '
                       'specified for the classifier to operate. Both were '
                       'configured `None`')
    if total_steps and total_epochs:
      raise ValueError('Only one of `total_steps` or `total_epochs` should be '
                       'specified for the classifier to operate. Both were '
                       'specified with numbers.')
    if total_steps is not None and total_steps < 1:
      raise ValueError('`total_steps` should be a positive integer. '
                       f'Instead, received {total_steps}')
    if total_epochs is not None and total_epochs < 1:
      raise ValueError('`total_epochs` should be a positive integer. '
                       f'Instead, received {total_epochs}')

    self.batch_size = batch_size
    self.total_steps = total_steps
    self.total_epochs = total_epochs
    self.optimizer = optax.adam(learning_rate)
    self.loss = optax.softmax_cross_entropy
    self.l2_reg = regularization
    self.noise_std = input_noise_std
    self.seed = seed
    self.verbose = verbose
    self.host_local_data = host_local_data
    self.name = name
    self.state = None
    self.accuracy_fn = metrics.Accuracy(top=(1, 5)).enable_jax_mode()

    # Initialize the partitioner.
    self.partitioner = partitioning.Partitioner(
        num_partitions=1,
        model_parallel_submesh=None,
        params_on_devices=True,
    )

  def init_params(self, dim, num_classes,
                  rng):
    w_key, b_key = jax.random.split(rng)
    w = jax.nn.initializers.glorot_uniform()(w_key, (dim, num_classes))
    b = jax.nn.initializers.zeros(b_key, (num_classes,))

    return {'w': w, 'b': b}

  def init_state(
      self,
      inputs,
      labels):
    """Initializes train state and PRNGKey."""

    if inputs.shape[0] != labels.shape[0]:
      raise ValueError(
          'Inputs and Labels do not have the same number of samples.')

    dim = inputs.shape[-1]
    num_classes = labels.shape[-1]

    # Get prngkeys.
    rng = jax.random.key(self.seed)
    init_rng, rng = jax.random.split(rng)

    params = self.init_params(dim, num_classes, init_rng)
    predict_fn = self.create_predict_fn()
    state = flax_train_state.TrainState.create(
        apply_fn=predict_fn,
        params=params,
        tx=self.optimizer)
    return state, rng

  def create_predict_fn(self):
    """Returns the function to run model inference."""

    def predict(params, inputs):
      outputs = jnp.matmul(inputs, params['w'])
      outputs += params['b'][jnp.newaxis, jnp.newaxis, :]
      return outputs

    return predict

  def create_train_step(self):
    """Create the train step to run on each replica."""

    def train_step(state, inputs, labels,
                   rng):
      """Runs an optimization step with additive Gaussian noise and L2-Reg."""

      if self.noise_std > 0.:
        inputs += self.noise_std * jax.random.normal(rng, shape=inputs.shape)

      def loss_fn(params):
        logits = state.apply_fn(params, inputs)  # [B, N, C]
        if len(logits.shape) == 3:
          logits = logits.mean(axis=1, keepdims=True)  # [B, 1, C]
        loss_value = optax.softmax_cross_entropy(logits, labels).mean()
        step_metrics = self.accuracy_fn(pred=logits, true=labels)
        step_metrics['loss_value'] = loss_value

        # Add L2 regularization.
        if self.l2_reg > 0.:
          weight_l2 = sum([jnp.sum(x ** 2) for x in jax.tree.leaves(params)])
          loss_value += weight_l2 * self.l2_reg

        return loss_value, step_metrics

      grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
      (_, step_metrics), grads = grad_fn(state.params)
      state = state.apply_gradients(grads=grads)

      return state, step_metrics

    return train_step

  def create_evaluation_step(
      self, metrics_stack
  ):
    """Creates the evaluation step to run on each host."""

    def evaluation_step(state,
                        inputs,
                        labels,
                        mask):
      """Runs an evaluation step and return the per-host metrics.

      Args:
        state: The model state (based on Flax TrainState).
        inputs: The input features, which should have a shape of [B, N, D].
        labels: The corresponding labels, with a shape of [B, 1, C].
        mask: The batch mask to be applied on the metric calculation.
      Returns:
        A collection of metrics aggreaged across all hosts.
      """

      logits = state.apply_fn(state.params, inputs)  # [B, N, C]
      logits = logits.mean(axis=1, keepdims=True)  # [B, 1, C]
      if mask is not None:
        # The mask shape should be identical to the logits'.
        mask = mask[:, jnp.newaxis]  # [B, 1]

      step_metrics = metrics.construct_metrics_dictionary_from_metrics_stack(
          prediction=logits,
          target=labels,
          mask=mask,
          metrics_stack=metrics_stack)

      return step_metrics

    return evaluation_step

  def fit(self, data):
    """Receives a features/labels iterator and trains a network from scratch.

    Args:
      data: An iterator which yields an inputs/labels pair at each step.
    """

    if self.total_steps is None:
      raise ValueError(
          'If you are seeing this error, it is probable that you defined '
          '`total_epochs`, while trying to perform online classification. '
          'Epoch-based training is not supported with online classification. '
          'Please either configure the classifier with `total_steps`, or call '
          '`bulk_probing`.')

    data_specs = self.partitioner.get_data_specs()
    data_shds = sharding.tree_pspecs_to_named_shardings(
        pspecs_tree=data_specs, mesh=self.partitioner.mesh)
    replicate_shds = sharding.replication_named_shardings(
        self.partitioner.mesh)
    train_step = jax.jit(
        self.create_train_step(),
        in_shardings=(replicate_shds, data_shds, data_shds, replicate_shds),
        out_shardings=(replicate_shds, replicate_shds),
        donate_argnums=(0, 1, 2, 3),
    )

    state_initialized = False
    current_step = 0
    logging.info('Training on total_steps: %s', self.total_steps)
    for step_inputs, step_labels in data:
      if not state_initialized:
        state, rng = self.init_state(step_inputs, step_labels)
        state_initialized = True
      rng, current_rng = jax.random.split(rng)  # pylint: disable=undefined-variable
      if self.host_local_data:
        step_inputs = multihost_utils.host_local_array_to_global_array(
            local_inputs=step_inputs,
            global_mesh=self.partitioner.mesh,
            pspecs=data_specs,
        )
        step_labels = multihost_utils.host_local_array_to_global_array(
            local_inputs=step_labels,
            global_mesh=self.partitioner.mesh,
            pspecs=data_specs,
        )
      with self.partitioner.mesh:
        state, step_metrics = train_step(
            state, step_inputs, step_labels, current_rng)  # pylint: disable=undefined-variable

      if self.verbose == 1 and (current_step % 50 == 0):
        step_metrics = mtu.tree_convert_jax_float_to_float(step_metrics)
        logging.info('Classification train metrics at step %d: %s',
                     current_step, step_metrics)

      current_step += 1
      if current_step >= self.total_steps:
        break

    self.state = state  # pylint: disable=undefined-variable

  def evaluate(
      self, data,
      metrics_stack):
    """Receives a features/labels iterator and evaluates based on metrics.

    Args:
      data: An iterator which yields an inputs/labels pair at each step.
      metrics_stack: A sequence of callable metric functions.

    Returns:
      A nested dictionary containing the results of the call of each metric
      function inside 'metrics_stack'.
    """
    # make sure metric functions are in jax mode, to avoid overhead
    for metric_cls in metrics_stack:
      metric_cls.enable_jax_mode()  # type: ignore

    data_specs = self.partitioner.get_data_specs()
    data_shds = sharding.tree_pspecs_to_named_shardings(
        pspecs_tree=data_specs, mesh=self.partitioner.mesh)
    replicate_shds = sharding.replication_named_shardings(
        self.partitioner.mesh)
    eval_step = jax.jit(
        self.create_evaluation_step(metrics_stack),
        in_shardings=(replicate_shds, data_shds, data_shds, data_shds),
        out_shardings=replicate_shds,
        donate_argnums=(0, 1, 2, 3),
    )

    all_metrics = []
    current_step = 0
    for step_inputs, step_labels in data:
      # pad inputs along the batch dimension (if not shardable)
      inputs_dict = {'step_inputs': step_inputs, 'step_labels': step_labels}
      inputs_dict, batch_mask = self.partitioner.maybe_pad_batches(inputs_dict)

      if self.host_local_data:
        step_inputs = multihost_utils.host_local_array_to_global_array(
            local_inputs=inputs_dict['step_inputs'],
            global_mesh=self.partitioner.mesh,
            pspecs=data_specs,
        )
        step_labels = multihost_utils.host_local_array_to_global_array(
            local_inputs=inputs_dict['step_labels'],
            global_mesh=self.partitioner.mesh,
            pspecs=data_specs,
        )
        batch_mask = multihost_utils.host_local_array_to_global_array(
            local_inputs=batch_mask,
            global_mesh=self.partitioner.mesh,
            pspecs=data_specs,
        )
      with self.partitioner.mesh:
        step_metrics = eval_step(
            self.state, step_inputs, step_labels, batch_mask)

      if self.verbose == 1 and current_step % 50 == 0:
        logging.info('Classification eval metrics at step %d: %s',
                     current_step, jax.tree.map(float, step_metrics))
      current_step += 1
      all_metrics.append(step_metrics)

    # aggregate metrics on each host
    all_metrics = aggregate_multi_step_data(
        all_data=all_metrics,
        batch_aggregate_fn=np.mean,
        scalar_aggregate_fn=np.mean,
        batch_axis=None,
    )
    logging.info('Finished evaluation: %s', all_metrics)

    return all_metrics


class OnlineLinearClassifier(BaseLinearClassifier):
  """An online linear classifier based on model inference iterator.

  This classifier operates on a pair of data iterators which are assumed to be
  the model inference iterators. Since the main inference iteration supports
  multi-host topologies, this classifier can also operate in multi-host jobs
  at scale.
  """

  def __init__(self,
               batch_size = 128,
               total_steps = 50000,
               learning_rate = 0.0001,
               regularization = 0.,
               input_noise_std = 0.,
               seed = 0,
               verbose = 0,
               host_local_data = True,
               name = 'online_linear_classifier'):

    if total_steps < 1:
      raise ValueError('`total_steps` should be a positive integer. '
                       f'Instead, received {total_steps}')

    super().__init__(batch_size=batch_size,
                     total_steps=total_steps,
                     learning_rate=learning_rate,
                     regularization=regularization,
                     input_noise_std=input_noise_std,
                     seed=seed,
                     verbose=verbose,
                     host_local_data=host_local_data,
                     name=name)


class OfflineLinearClassifier(BaseLinearClassifier):
  """An offline linear classifier based on bulk input features.

  This classifier operates on bulk input features and assumes that the inputs
  are the same on all hosts (if under a multi-host job). Inputs are shuffled
  and sharded properly on each host before being distributed across data shards.
  """

  def __init__(self,
               batch_size = 128,
               total_epochs = 100,
               learning_rate = 0.0001,
               regularization = 0.,
               input_noise_std = 0.,
               seed = 0,
               verbose = 0,
               host_local_data = True,
               name = 'offline_linear_classifier'):

    if total_epochs < 1:
      raise ValueError('`total_epochs` should be a positive integer. '
                       f'Instead, received {total_epochs}')

    super().__init__(batch_size=batch_size,
                     total_epochs=total_epochs,
                     learning_rate=learning_rate,
                     regularization=regularization,
                     input_noise_std=input_noise_std,
                     seed=seed,
                     verbose=verbose,
                     host_local_data=host_local_data,
                     name=name)

  def create_data_generator(
      self,
      arrays,
      batch_size,
      num_epochs = 1,
      is_training = True,
  ):
    """Creates generator out of an unbatched data.

    Args:
      arrays: A sequence of NdArrays to be batched.
      batch_size: A positive integer number indicating the number of samples
        in each iteration step.
      num_epochs: An integer number indicating the number of data replication.
        This number should take positive values OR -1, in which case we iterate
        over the data infinitely.
      is_training: A bool, indicating whether to yield same-shape array in the
        final iteration over the data. If True, it will start over the beginning
        of the array to fill the remainder (instead of simply dropping them.
        This will yield an array with shape [batch_size, ...]. If False, it
        will yield the remainder which will have a shape [B, ...],
        in which B <= batch_size.

    Returns:
      An iterator that yields a sequence of batched arrays.
    """
    num_hosts = jax.process_count()
    host_id = jax.process_index()
    batch_size = int(batch_size / num_hosts)
    arrays = tuple([
        np.array_split(array, num_hosts, axis=0)[host_id] for array in arrays
    ])
    def generator(
        arrays
    ):
      """Constructs an iterator given a tuple of arrays."""

      num_samples = [array.shape[0] for array in arrays]
      if len(set(num_samples)) > 1:
        raise ValueError(
            'Inputs to data generator should contain the same number of '
            f'samples. Received {num_samples}')
      num_samples = set(num_samples).pop()
      if num_samples < batch_size:
        raise ValueError(
            f'Total number of samples ({num_samples}) is '
            f'smaller than the requested batch size ({batch_size}).')

      num_batched_samples = np.ceil(num_samples / batch_size).astype(int)
      if num_epochs > 0:
        total_steps = num_epochs * num_batched_samples
      else:
        total_steps = np.inf

      n = -1
      while n < total_steps:
        n += 1
        if is_training:
          indices = list(range(batch_size * n, batch_size * (n + 1)))
          indices = [i % num_samples for i in indices]
        else:
          begin = batch_size * (n % num_batched_samples)
          end = batch_size * (n % num_batched_samples + 1)
          if end > num_samples:
            end = num_samples
          indices = tuple(range(begin, end))

        batched_arrays = tuple([array[indices, :] for array in arrays])
        yield batched_arrays if len(batched_arrays) > 1 else batched_arrays[0]

    return generator(arrays)

  def bulk_probing(
      self,
      train_features,
      test_features,
      train_labels,
      test_labels,
      metrics_stack,
  ):
    """Performs linear probing on the train/test samples.

    Args:
      train_features: A tensor with shape [batch, instane, dim] representing
        test_features train split's semantic features.
      test_features: A tensor with shape [batch, instane, dim] representing
        the test split's semantic features.
      train_labels: A tensor with shape [batch, instane, num_classes]
        representing train split's one-hot labels.
      test_labels: A tensor with shape [batch, instane, num_classes]
        representing test split's one-hot labels.
      metrics_stack: A tuple of metric functions to be applied on pairs of
        (predictions, labels).

    Returns:
      A collection of (train, test) metrics based on the structure of
        `metrics_stack`.
    """

    # Verify the shape of train/test labels, since only single-instance
    # labels are supported.
    _verify_single_instance(train_labels)
    _verify_single_instance(test_labels)

    # Fetch the total number of steps
    num_train_samples = train_features.shape[0]
    self.total_steps = _get_num_steps(
        num_train_samples, self.batch_size, self.total_epochs)

    # train classifier
    train_iterator = self.create_data_generator(
        arrays=(train_features, train_labels),
        batch_size=self.batch_size,
        num_epochs=-1,
        is_training=True)
    self.fit(train_iterator)

    # Predict train logits.
    train_evaluation_iterator = self.create_data_generator(
        arrays=(train_features, train_labels),
        batch_size=self.batch_size,
        num_epochs=1,
        is_training=False)
    train_metrics = self.evaluate(train_evaluation_iterator, metrics_stack)

    # Predict test logits.
    test_evaluation_iterator = self.create_data_generator(
        arrays=(test_features, test_labels),
        batch_size=self.batch_size,
        num_epochs=1,
        is_training=False)
    test_metrics = self.evaluate(test_evaluation_iterator, metrics_stack)

    return train_metrics, test_metrics


class ZeroShotClassifier(object):
  """Performs zero-shot classification based on vision-text similarity."""

  def __init__(self,
               as_py = True,
               name = 'zero_shot_classifier'):
    self.np = np if as_py else jnp
    self.as_py = as_py
    self.name = name
    self.keys = None
    self.initialized = False

  def _set_ndarray_at_idx(self,
                          array,
                          idx,
                          value):
    """Multi-index array assignment for a given NP/JNP array.

    Args:
      array: a 2d array with size (N, D) in which we want to assign values
        to certain indices in the first dimension
      idx: an 1d-array with size (K,) containing the indices in which we
        want to assign values
      value: a 2d-array with size (K, D) containing the corresponding data
        for K samples

    Returns:
      array: a 2d array with size (N, D) whose K samples have been updated
        according to `idx` and `value`
    """

    if self.np is np:
      array[idx] = value
      return array
    elif self.np is jnp:
      return array.at[idx].set(value)
    else:
      raise ValueError('Backend not supported!')

  def _gather_unique_keys(self, features,
                          labels):
    """Fetches unique features from a collection of feature-label pairs.

    This function assumes that there are N vectors in `features` and their
    correspong annotations are given in `labels`. It further assumes that
    there are only C unique feature representations. The returning value
    is a (C, D) array containing those unique representations. If the
    mentioned assumption is not met for `features`, the behavior of this
    function is not predicted and changes based on the order in which
    vectors are stored in `features`.
    This can be used in Zero-Shot Classification tasks in which we have a
    set of reference features that we compare our queries against.

    Args:
      features: a rank-C matrix with size (N, D)
      labels: a 2d array with size (N, C) annotating the C unique vectors in
        `features` matrix

    Returns:
      keys: a full-rank matrix with size (C, D) containing the C unique vectors
        in `features` according to annotations in `labels`
    """

    d_features = features.shape[-1]
    n_classes = labels.shape[-1]
    keys = self.np.zeros((n_classes, d_features), dtype=self.np.float32)
    indices = self.np.argmax(labels, axis=1)
    keys = self._set_ndarray_at_idx(keys, indices, features)
    keys = self.l2_normalize(keys)
    return keys

  def set_keys(self, features, labels):
    """Fetches unique vectors in `features` and stores as a reference key."""

    self.keys = self._gather_unique_keys(features, labels)
    self.initialized = True

  def l2_normalize(self,
                   inputs,
                   axis = -1):
    """L2-normalized a given nd array."""

    return inputs / (
        self.np.linalg.norm(inputs, axis=axis, keepdims=True) + 1e-6)

  def predict(self, inputs):
    """Calculates similarities between input queries and the reference keys."""

    if not self.initialized:
      raise ValueError('Keys are not initialized!')

    inputs = self.l2_normalize(inputs)
    logits = self.np.einsum('bd,cd->bc', inputs, self.keys)
    return logits

  def bulk_probing(
      self,
      train_features,
      test_features,
      train_labels,
      test_labels,
      metrics_stack,
  ):
    """Performs zero-shot similarity-based classification."""

    # Verify the shape of train/test labels, since only single-instance
    # labels are supported.
    _verify_single_instance(train_labels)
    _verify_single_instance(test_labels)

    # Fetch label representations in the target modality space.
    train_features = train_features.mean(axis=1)  # (B, D)
    train_labels = train_labels[:, 0, :]

    # Fetch target modality features in the label space.
    num_samples, num_clips, dim = test_features.shape
    test_features = test_features.reshape((num_samples*num_clips, dim))
    test_labels = test_labels[:, 0, :]

    # Initialize the zero-shot classifier.
    self.set_keys(train_features, train_labels)
    test_logits = self.predict(test_features)

    # Average test logits over clips.
    test_logits = test_logits.reshape((num_samples, num_clips, -1)).mean(axis=1)

    # make sure metric functions are in the same mode as the evaluator
    for metric_cls in metrics_stack:
      if self.as_py:
        metric_cls.enable_numpy_mode()  # type: ignore
      else:
        metric_cls.enable_jax_mode()  # type: ignore

    # Calculate accuracy metrics.
    test_metrics = metrics.construct_metrics_dictionary_from_metrics_stack(
        prediction=test_logits,
        target=test_labels,
        metrics_stack=metrics_stack
        )

    return test_metrics


class OnlineMAPClassifier(OnlineLinearClassifier):
  """An online multi-head attention (MAP) classifier.

  Like OnlineLinearClassifier, but uses a MAP layer on a sequence of vectors
  to perform classification instead of a linear layer on a single vector.
  """

  def __init__(self,
               num_heads = 12,
               d_ff = 3072,
               use_bias = True,
               batch_size = 128,
               total_steps = 50000,
               learning_rate = 0.0001,
               regularization = 0.,
               input_noise_std = 0.,
               seed = 0,
               verbose = 0,
               host_local_data = True,
               name = 'map_classifier'):
    super().__init__(
        batch_size=batch_size,
        total_steps=total_steps,
        learning_rate=learning_rate,
        regularization=regularization,
        input_noise_std=input_noise_std,
        seed=seed,
        verbose=verbose,
        host_local_data=host_local_data,
        name=name,
    )
    self.num_heads = num_heads
    self.d_ff = d_ff
    self.use_bias = use_bias

    self.map_head = None

  def init_params(
      self,
      dim,
      num_classes,
      rng):
    self.map_head = heads.VitPostEncoderHead(
        aggregation_type=constants.AggregationType.MULTI_HEAD_ATTENTION_POOL,
        d_post_proj=None,
        post_proj_position=None,
        num_classes=num_classes,
        head_bias_init=0.,
        dtype=jnp.float32,
        num_heads=self.num_heads,
        d_ff=self.d_ff,
        dropout_rate=0.,
        use_bias=self.use_bias,
        name='map_head',
    )

    # The parameter shapes don't depend on these dims, so we set them to 1.
    batch_size, num_instances, seq_len = 1, 1, 1
    placeholder_input = jnp.zeros((batch_size, num_instances, seq_len, dim))
    return self.map_head.init(
        rng, placeholder_input, patched_shape=None, deterministic=True)

  def create_predict_fn(self):
    """Returns the function to run model inference."""

    def predict(params, inputs):
      if self.map_head is None:
        raise ValueError('Model must be initialized before running.')
      outputs = self.map_head.apply(
          params, inputs, patched_shape=None, deterministic=True)
      return outputs[constants.DataFeatureName.LOGITS]
    return predict


def bulk_test_predict_classification(
    bulk_data,
    metrics_stack):
  """Performs a classification prediction on model outputs on the test split.

  Args:
    bulk_data: A nested collection of arrays which contain the original inputs
      and their corresponding outputs for ALL samples in a dataset on which we
      want to perform the classification prediction. The expected shape for the
      leaf logits are [n_samples, n_instance, n_classes]. Similarly, it is
      assumed that labels are also included and have a shape of
      [n_samples, 1, n_classes].
    metrics_stack: A sequence of callable metric objects that operate on
      a pair of (pred, true) with shapes similar to (logits, labels) pairs.
  Returns:
    A nested dictionary that contains the corresponding metric values for
    all modalities in 'outputs'.
  """

  all_metrics = {}

  for dataset_name in bulk_data:
    test_key = list(bulk_data[dataset_name].keys())
    if len(test_key) > 1:
      raise ValueError(f'More than one test key in data of {dataset_name}!')
    test_key = test_key[0]
    test_data = bulk_data[dataset_name][test_key]
    test_outputs = test_data[OUTPUTS][LABEL_CLASSIFIER]
    test_targets = test_data[TARGETS][LABEL_CLASSIFIER]

    # Get target modality and assert there exists only one target modality
    modality = _get_modality_with_label(test_targets, dataset_name=dataset_name)

    for metric_fn in metrics_stack:
      cls_metrics = metric_fn(
          test_outputs[modality][TOKEN_RAW][LOGITS],
          test_targets[modality][LABEL])
      for key, value in cls_metrics.items():
        all_metrics[f'{dataset_name}/{test_key}/{key}'] = value

    logging.info('Bulk classification prediction metrics: %s', all_metrics)
  return all_metrics


def bulk_linear_classification(
    bulk_data,
    metrics_stack):
  """Performs a linear classification on model outputs.

  Args:
    bulk_data: A nested collection of arrays which contain the original inputs
      and their corresponding outputs for ALL samples in a dataset on which we
      want to perform the linear classification. The expected shape for the
      leaf features are [n_samples, n_instance, dim]. Similarly, it is assumed
      that labels are also included and have a shape of
      [n_samples, 1, n_classes].
    metrics_stack: A sequence of callable metric objects that operate on
      a pair of (pred, true) with shapes similar to (features, labels) pairs.
  Returns:
    A nested dictionary that contains the corresponding metric values for
    all modalities in 'outputs'.
  """

  all_metrics = {}
  for dataset_name in bulk_data:
    # group subsets based on their splits (if any):
    #                  (train_1, test_1), (train_2, test_2), ...
    subsets = group_subsets(list(bulk_data[dataset_name].keys()))
    for train_key, test_key in subsets:
      # Fetch train/test splits
      train_data = bulk_data[dataset_name][train_key]
      test_data = bulk_data[dataset_name][test_key]

      # Fetch the extracted features and their corresponding targets
      train_outputs = train_data[OUTPUTS][ENCODER]
      test_outputs = test_data[OUTPUTS][ENCODER]
      train_targets = train_data[TARGETS][LABEL_CLASSIFIER]
      test_targets = test_data[TARGETS][LABEL_CLASSIFIER]

      # Get target modality and assert there exists only one target modality
      modality = _get_modality_with_label(
          train_targets, dataset_name=dataset_name)

      # perform linear probing
      classifier_config = eval_config.OfflineLinearClassification().as_dict()
      linear_classifier = OfflineLinearClassifier(
          **classifier_config)
      train_metrics, test_metrics = linear_classifier.bulk_probing(
          train_features=train_outputs[modality][TOKEN_RAW][FEATURES_AGG],
          test_features=test_outputs[modality][TOKEN_RAW][FEATURES_AGG],
          train_labels=train_targets[modality][LABEL],
          test_labels=test_targets[modality][LABEL],
          metrics_stack=metrics_stack,
          )

      # accumulate all metrics
      for key, value in train_metrics.items():
        all_metrics[f'{dataset_name}/{train_key}/{key}'] = value

      for key, value in test_metrics.items():
        all_metrics[f'{dataset_name}/{test_key}/{key}'] = value

    logging.info('Bulk linear classification metrics: %s', all_metrics)
  return all_metrics


def bulk_zero_shot_classification(
    bulk_data,
    metrics_stack):
  """Performs a zero-shot classification on model outputs.

  Args:
    bulk_data: A nested collection of arrays which contain the original inputs
      and their corresponding outputs for ALL samples in a dataset on which we
      want to perform the zero-shot classification. The expected shape for the
      leaf features are [n_samples, n_instance, dim]. Similarly, it is assumed
      that labels are also included and have a shape of
      [n_samples, 1, n_classes]. It is also assumed that there is a 'text'
      modality in the outputs along with at least one other modality (e.g.
      'vision', 'audio'.)
    metrics_stack: A sequence of callable metric objects that operate on
      a pair of (pred, true) with shapes similar to (features, labels) pairs.
  Returns:
    A nested dictionary that contains the corresponding metric values for
    all modalities in 'outputs'.
  """

  all_metrics = {}

  for dataset_name in bulk_data:
    # group subsets based on their splits (if any):
    #                  (train_1, test_1), (train_2, test_2), ...
    subsets = group_subsets(list(bulk_data[dataset_name].keys()))
    for train_key, test_key in subsets:
      # Fetch train/test splits
      train_data = bulk_data[dataset_name][train_key]
      test_data = bulk_data[dataset_name][test_key]

      # Fetch the extracted features and their corresponding targets
      train_outputs = train_data[OUTPUTS][COMMON_SPACE]
      test_outputs = test_data[OUTPUTS][COMMON_SPACE]
      train_targets = train_data[TARGETS][LABEL_CLASSIFIER]
      test_targets = test_data[TARGETS][LABEL_CLASSIFIER]

      # Get target modality and assert there exists only one target modality
      modality = _get_modality_with_label(
          train_targets, dataset_name=dataset_name)

      # Text modality is the source modality in which we compare any modality to
      source_modality = Modality.TEXT

      # Extract the common space projection between text and the target modality
      train_features = train_outputs[source_modality][TOKEN_ID][modality]
      test_features = test_outputs[modality][TOKEN_RAW][source_modality]
      train_labels = train_targets[modality][LABEL]
      test_labels = test_targets[modality][LABEL]

      # perform zero-shot probing
      zero_shot_classifier = ZeroShotClassifier()
      zs_metrics = zero_shot_classifier.bulk_probing(
          train_features=train_features,
          test_features=test_features,
          train_labels=train_labels,
          test_labels=test_labels,
          metrics_stack=metrics_stack,
      )

      for key, value in zs_metrics.items():
        all_metrics[f'{dataset_name}/{test_key}/zs_{key}'] = value

    logging.info('Bulk zero-shot classification metrics: %s', all_metrics)
  return all_metrics


# TODO(b/239480971): merge this method with the full variant.
def bulk_test_zero_shot_classification(
    bulk_data,
    metrics_stack):
  """Performs a zero-shot classification on model outputs only using test split.

  Args:
     bulk_data: A nested collection of arrays which contain the original inputs
      and their corresponding outputs for ALL samples in a dataset on which we
      want to perform the zero-shot classification. The expected shape for the
      leaf features are [n_samples, n_instance, dim]. Similarly, it is assumed
      that labels are also included and have a shape of
      [n_samples, 1, n_classes]. It is also assumed that there is a 'text'
      modality in the outputs along with at least one other modality (e.g.
      'vision', 'audio'.)
    metrics_stack: A sequence of callable metric objects that operate on
      a pair of (pred, true) with shapes similar to (features, labels) pairs.
  Returns:
    A nested dictionary that contains the corresponding metric values for
    all modalities in 'outputs'.
  """

  all_metrics = {}

  for dataset_name in bulk_data:
    test_key = list(bulk_data[dataset_name].keys())
    if len(test_key) > 1:
      raise ValueError(f'More than one test key in data of {dataset_name}!')
    test_key = test_key[0]
    test_data = bulk_data[dataset_name][test_key]
    test_outputs = test_data[OUTPUTS][COMMON_SPACE]
    test_targets = test_data[TARGETS][LABEL_CLASSIFIER]

    # Get target modality and assert there exists only one target modality
    modality = _get_modality_with_label(test_targets, dataset_name=dataset_name)

    # Text modality is the source modality in which we compare any modality with
    source_modality = Modality.TEXT

    # Extract the common space projection between text and the target modality
    train_features = test_outputs[source_modality][TOKEN_ID][modality]
    test_features = test_outputs[modality][TOKEN_RAW][source_modality]
    test_labels = test_targets[modality][LABEL]

    logging.info('Performing zero-shot classification on %s modality from %s '
                 'with feature shapes %s_%s and labels %s',
                 modality,
                 dataset_name,
                 train_features.shape,
                 test_features.shape,
                 test_labels.shape)

    # perform zero-shot probing
    zero_shot_classifier = ZeroShotClassifier()
    zs_metrics = zero_shot_classifier.bulk_probing(
        train_features=train_features,
        test_features=test_features,
        train_labels=test_labels,  # Test labels are also train labels here
        test_labels=test_labels,
        metrics_stack=metrics_stack,
    )

    for key, value in zs_metrics.items():
      all_metrics[f'{dataset_name}/{test_key}/zs_{key}'] = value

    logging.info('Bulk zero-shot classification metrics: %s', all_metrics)
  return all_metrics


def bulk_zero_shot_retrieval(
    bulk_data,
    metrics_stack):
  """Performs a zero-shot retrieval between all modality pairs in model outputs.

  Args:
    bulk_data: A nested collection of arrays which contain the original inputs
      and their corresponding outputs for ALL samples in a dataset on which we
      want to perform the zero-shot retrieval. The expected shape for the leaf
      features are [n_samples, n_instance, dim]. It is also assumed that there
      are at least two modalities (other than 'label' in the outputs.
    metrics_stack: A sequence of callable metric objects that operate on
      a pair of (modality_1, modality_2) with a shape of
      [n_samples, n_instance, dim].
  Returns:
    A nested dictionary that contains the corresponding metric values for
    all modalities in 'outputs'.
  """
  all_metrics = {}
  for dataset_name in bulk_data:
    for subset in bulk_data[dataset_name]:
      data = bulk_data[dataset_name][subset]
      common_space_collection = data[OUTPUTS][COMMON_SPACE]
      pair_modalities = collections.defaultdict(list)
      for modality in common_space_collection:
        token_feature_names = set(common_space_collection[modality].keys())
        if len(token_feature_names) > 1:
          raise NotImplementedError
        token_feature_name = token_feature_names.pop()
        common_space = common_space_collection[modality][token_feature_name]
        for target_modality in common_space:
          pair_name = '_'.join(sorted([modality, target_modality]))
          pair_modalities[pair_name].append(
              (common_space[target_modality],
               '_to_'.join([modality, target_modality]))
              )

      for pair_name in pair_modalities:
        embeddings = pair_modalities[pair_name]
        if len(embeddings) != 2:
          raise ValueError(
              f'One modality is missing in {pair_name} common space!'
              )
        modality_1 = embeddings[0][0]
        modality_2 = embeddings[1][0]

        logging.info('Performing zero-shot retrieval on %s pair from %s with '
                     'feature shapes: %s_%s',
                     pair_name,
                     dataset_name,
                     modality_1.shape,
                     modality_2.shape)

        for metric_fn in metrics_stack:
          ret_metrics = metric_fn(modality_1, modality_2)
          for key, value in ret_metrics['m1_vs_m2'].items():
            all_metrics[f'{dataset_name}/{embeddings[0][1]}/{key}'] = value

          for key, value in ret_metrics['m2_vs_m1'].items():
            all_metrics[f'{dataset_name}/{embeddings[1][1]}/{key}'] = value

    logging.info('Bulk zero-shot retrieval metrics: %s', all_metrics)
  return all_metrics


def online_linear_classification(
    inference_train_iterator,
    inference_test_iterator,
    dataset_name,
    metrics_stack):
  """Performs an online linear classification by iterating on model outputs.

  Args:
    inference_train_iterator: An iterator that yields model outputs for a
      single inference step on the train split of a dataset. It is assumed
      that each step yields a nested array with two leaves: A 'features' with
      an expected shape [n_samples, n_instance, dim], and a 'labels' with
      an expected shape of [n_samples, 1, n_classes].
    inference_test_iterator: An iterator that yields model outputs for a
      single inference step on the test split of a dataset. It is assumed
      that each step yields a nested array with two leaves: A 'features' with
      an expected shape [n_samples, n_instance, dim], and a 'labels' with
      an expected shape of [n_samples, 1, n_classes].
    dataset_name: A string containing the name of the current dataset being
      iterated over.
    metrics_stack: A sequence of callable metric objects that operate on
      a pair of (pred, true) with shapes similar to (features, labels) pairs.
  Returns:
    A nested dictionary that contains the corresponding metric values for
    classification results.
  """

  def feature_iterator(inference_iterator):
    for _, _, data, _ in inference_iterator:
      # Fetch outputs and targets
      outputs = data[OUTPUTS][ENCODER]
      targets = data[TARGETS][ENCODER]

      # Fetch and yield features and labels
      modality = _get_modality_with_label(targets, dataset_name)
      features = outputs[modality][TOKEN_RAW][FEATURES_AGG]
      labels = targets[modality][LABEL]
      yield features, labels

  all_metrics = {}
  train_feature_iterator = feature_iterator(inference_train_iterator)
  test_feature_iterator = feature_iterator(inference_test_iterator)
  classifier_config = eval_config.OnlineLinearClassification(
      verbose=1).as_dict()
  linear_classifier = OnlineLinearClassifier(**classifier_config)
  linear_classifier.fit(train_feature_iterator)
  test_metrics = linear_classifier.evaluate(
      test_feature_iterator,
      metrics_stack=metrics_stack)
  # accumulate all metrics
  for key, value in test_metrics.items():
    all_metrics[f'{dataset_name}/test/{key}'] = value

  logging.info('Online linear classification metrics: %s', all_metrics)
  return all_metrics
