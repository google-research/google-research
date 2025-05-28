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

"""Base modules for executing train and evaluation."""
# TODO(b/233752479): add testing for base execution modules

import collections
from concurrent import futures
import functools
import os
import time
from typing import Any, Callable, Iterator, Type, TypeVar

from absl import logging
import fiddle as fdl
from flax import traverse_util
from flax.core import scope
import flax.linen as nn
from flax.training import checkpoints as flax_ckpt
from flax.training import train_state as flax_train_state
import jax
from jax.experimental import multihost_utils
import numpy as np

from imp.max.core import constants
from imp.max.core import probing
from imp.max.core import utils
from imp.max.data import processing
from imp.max.evaluation import evaluators
from imp.max.evaluation import metrics as metrics_lib
from imp.max.execution import checkpointing
from imp.max.execution import config as exec_config
from imp.max.execution import partitioning
from imp.max.modeling import module
from imp.max.optimization import gradients
from imp.max.optimization import objectives
from imp.max.optimization import optimizers
from imp.max.utils import sharding
from imp.max.utils import tree as mtu
from imp.max.utils import typing


ExperimentT = exec_config.ExperimentT
ExecutorT = Type[TypeVar('_ExecutorT', bound='BaseExecutor')]
Mode = constants.Mode
DenyList = scope.DenyList

# Execution constants
PARAMS = constants.FlaxCollection.PARAMS
PARAMS_AXES = constants.FlaxCollection.PARAMS_AXES
INTERMEDIATES = constants.FlaxCollection.INTERMEDIATES
PROBES = constants.FlaxCollection.PROBES
AUX_LOSS = constants.FlaxCollection.AUX_LOSS
_TRAIN_INIT_DENIED_COLLECTIONS = DenyList((INTERMEDIATES, PROBES, AUX_LOSS))
_TRAIN_APPLY_DENIED_COLLECTIONS = DenyList((PARAMS, PARAMS_AXES))
_EVAL_INIT_DENIED_COLLECTIONS = DenyList((PROBES, AUX_LOSS))
_EVAL_APPLY_DENIED_COLLECTIONS = DenyList((PARAMS, PARAMS_AXES, AUX_LOSS))

# I/O constants
DataFeatureType = constants.DataFeatureType
DataFeatureRoute = constants.DataFeatureRoute
DataFeatureName = constants.DataFeatureName
Modality = constants.Modality
INPUTS = DataFeatureType.INPUTS
OUTPUTS = DataFeatureType.OUTPUTS
TARGETS = DataFeatureType.TARGETS
HYPERPARAMS = DataFeatureType.HYPERPARAMS
METADATA = DataFeatureType.METADATA
COMMON_SPACE = DataFeatureRoute.COMMON_SPACE
FEATURES_AGG = DataFeatureName.FEATURES_AGG
LABEL = DataFeatureName.LABEL
LOGITS = DataFeatureName.LOGITS


# Mapping from each serving strategy to the respective evaluation category to
# set the appropriate metrics.
ServingStrategy = constants.ServingStrategy
_SERVING_SUPERSETS = {
    ServingStrategy.BULK_ZS_RETRIEVAL: ServingStrategy.RETRIEVAL,
    ServingStrategy.BULK_TEST_PREDICT_CLS: ServingStrategy.CLASSIFICATION,
    ServingStrategy.BULK_LINEAR_CLS: ServingStrategy.CLASSIFICATION,
    ServingStrategy.BULK_ZS_CLS: ServingStrategy.CLASSIFICATION,
    ServingStrategy.BULK_TEST_ZS_CLS: ServingStrategy.CLASSIFICATION,
    ServingStrategy.ONLINE_LINEAR_CLS: ServingStrategy.CLASSIFICATION,
}
# We create a mapping between each serving strategy and its corresponding
# method that performs that serving (evaluation).
_BULK_SERVINGS = {
    ServingStrategy.BULK_TEST_PREDICT_CLS: (
        evaluators.bulk_test_predict_classification
    ),
    ServingStrategy.BULK_LINEAR_CLS: evaluators.bulk_linear_classification,
    ServingStrategy.BULK_ZS_CLS: evaluators.bulk_zero_shot_classification,
    ServingStrategy.BULK_TEST_ZS_CLS: (
        evaluators.bulk_test_zero_shot_classification
    ),
    ServingStrategy.BULK_ZS_RETRIEVAL: evaluators.bulk_zero_shot_retrieval,
}
_ONLINE_SERVINGS = {
    ServingStrategy.ONLINE_LINEAR_CLS: evaluators.online_linear_classification
}
_SUPPORTED_BULK_SERVINGS = frozenset(_BULK_SERVINGS.keys())
_SUPPORTED_ONLINE_SERVINGS = frozenset(_ONLINE_SERVINGS.keys())
_SUPPORTED_SERVINGS = _SUPPORTED_BULK_SERVINGS | _SUPPORTED_ONLINE_SERVINGS


def set_hardware_rng_ops():
  jax.config.update('jax_default_prng_impl', 'unsafe_rbg')


def group_dataloaders(
    dataloaders
):
  """Groups dataloaders by dataset."""

  grouped_dataloaders = collections.defaultdict(tuple)
  for dataloader in dataloaders:
    grouped_dataloaders[dataloader['name']] += (dataloader,)

  return tuple(grouped_dataloaders.values())


def sort_grouped_dataloaders_by_subset(
    dataloaders
):
  """Sorts a tuple of dataloaders based on train->test/valid order."""

  if len(dataloaders) != 2:
    raise ValueError(
        'Expected to receive a tuple with length 2 when sorting grouped '
        f'data. Instead, received {dataloaders}')

  sorted_dataloaders = {}
  for dataloader in dataloaders:
    subset = dataloader['subset']
    if subset.startswith('train'):
      sorted_dataloaders['train'] = dataloader
    elif subset.startswith('test') or subset.startswith('valid'):
      sorted_dataloaders['test'] = dataloader
    else:
      raise ValueError(
          f'Subset {subset} not expected when sorting a grouped data.')

  if len(sorted_dataloaders) != 2:
    raise ValueError(
        'Expected subsets (`train`, `test`|`valid`), however received '
        f'{tuple(sorted_dataloaders.keys())} in the sorted variant. Please '
        'make sure to pass a Dataloader collection with distinct subsets.')

  return sorted_dataloaders


def get_rngs(keys, rng=None, rngs_per_key=1, add_params_rngs=False):
  """Util for getting/splitting PRNGKey for the given keys."""

  if add_params_rngs:
    keys += ('params',)

  if rng is None:
    rng = jax.random.key(0)

  rngs = jax.random.split(rng, len(keys) * rngs_per_key + 1)
  key_rngs = {
      key: rngs[n * rngs_per_key:(n + 1) * rngs_per_key]
      for n, key in enumerate(keys)
  }
  next_rng = rngs[-1]

  # Special case for param initialization.
  if rngs_per_key == 1:
    key_rngs = {k: v[0] for k, v in key_rngs.items()}

  return next_rng, key_rngs


def get_learning_rate(config, step):
  return float(optimizers.schedules.get_schedule(config)(step))


def prepend_data_scope(metrics, name):
  new_metrics = {}
  for metric in metrics:
    new_metrics[f'{name}/{metric}'] = metrics[metric]

  return new_metrics


def count_params(params):
  return sum(x.size for x in jax.tree.leaves(params))


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


def _flatten_collection_tree(tree):
  return traverse_util.flatten_dict(tree, sep='/')


def _unflatten_collection_tree(tree):
  return traverse_util.unflatten_dict(tree, sep='/')


def _check_nan_values(metrics_tree):
  if any(np.isnan(v) for v in _flatten_collection_tree(metrics_tree).values()):
    raise ValueError(f'Found NaN values in metrics: {metrics_tree}')


class Profiler:
  """A profiler to generate profiling plots and step metrics.

  After the first few steps, the profiler will run a profile trace, which will
  be added to the XManager UI. The profiler also periodically logs profiling
  information to TensorBoard, such as steps/sec, tokens/sec, etc.

  Attributes:
    total_steps: the total number of steps for the experiment.
    profile_seconds: the number of seconds to run profiling. Set to <=0
      to disable profiling.
    wait_for_steps: how many steps should be executed before running profiling
      trace.
    steps_profiled: the number of steps that have been profiled.
  """

  def __init__(
      self,
      path,
      total_steps = None,
      profile_seconds = 15.,
      wait_for_steps = 1000,
  ):
    """Initializes the profiler."""
    self._profiling_dir = path
    self.profile_seconds = profile_seconds
    self.wait_for_steps = wait_for_steps
    self.total_steps = total_steps
    self.steps_profiled = 0

    self._num_hosts = jax.process_count()
    self._num_tpus = jax.device_count()
    self._started_profiling_session = False
    self._last_profile_time = time.time()
    self._total_examples = 0
    self._total_tokens = 0

    self._local_step = 0

    self._dataset_examples = {}
    self._dataset_steps = collections.defaultdict(int)


  def maybe_start_profile_trace(self):
    """Launches a profiling session after the number of steps."""
    if (not self._started_profiling_session and
        self.wait_for_steps > 0 and
        self._local_step > self.wait_for_steps):
      jax.profiler.start_trace(self._profiling_dir)

      # Start a concurrent thread to wait and complete the profiling session.
      def _sleep_and_stop_profile_trace():
        """Finishes a profiling session after sleeping."""
        time.sleep(self.profile_seconds)
        jax.profiler.stop_trace()
        logging.info('Profiling trace finished after %s seconds.',
                     self.profile_seconds)

      futures.ThreadPoolExecutor(1, 'profile_trace').submit(
          _sleep_and_stop_profile_trace,
      )

      self._started_profiling_session = True

  def _count_input_examples(
      self,
      inputs):
    """Calculates the total number of examples and tokens from the input batch.

    Args:
      inputs: a collection of batched inputs for this host. It is assumed all
        hosts have the same batch shape.

    Returns:
      The total number of examples and total number of tokens in the batch
      across all hosts.
    """
    if not inputs:
      return 0, 0

    input_shapes = jax.tree.map(lambda array: array.shape, inputs)
    input_shapes = traverse_util.flatten_dict(input_shapes, sep='/')
    input_shapes = tuple(shape for name, shape in input_shapes.items()
                         if processing.has_token(name))

    # Calculate on the local batch size and multiply by the number of hosts to
    # get the global batch size.
    num_examples = sum(shape[0] for shape in input_shapes)
    num_tokens = sum(np.prod(shape[:2]) for shape in input_shapes)

    return num_examples * self._num_hosts, num_tokens * self._num_hosts

  def update(
      self,
      global_step,
      dataset_name,
      data):
    """Marks the start of a profiling step.

    Args:
      global_step: the global execution step independent of job runs.
      dataset_name: the name of the dataset for this executed step.
      data: a collection of batched data for this host. It is assumed all hosts
        have the same batch shape.

    Returns:
      A dict of metrics, if the current step coincides with the profile
      interval. Otherwise an empty dict.
    """

    inputs = data.get(constants.DataFeatureType.INPUTS, {})
    # Keep track of the number of examples and tokens across batches.
    if dataset_name not in self._dataset_examples:
      self._dataset_examples[dataset_name] = self._count_input_examples(inputs)

    num_examples, num_tokens = self._dataset_examples[dataset_name]

    self._total_examples += num_examples
    self._total_tokens += num_tokens
    self._dataset_steps[dataset_name] += 1
    self._last_global_step = global_step
    self._local_step += 1
    self.steps_profiled += 1

  def get_metrics(self):
    """Calculates profiler metrics for the current step.

    Returns:
      A dict of summary metrics containing timing information such
      as averaged step time, examples per sec, and the ratio of steps sampled.
    """
    metrics = {}

    if self.steps_profiled == 0:
      return metrics

    # Metrics based on the step.
    elapsed = time.time() - self._last_profile_time
    steps_per_sec = self.steps_profiled / elapsed
    sec_per_step = 1. / steps_per_sec

    metrics.update({
        'profile/steps_per_sec': steps_per_sec,
        'profile/sec_per_step': sec_per_step,
        'profile/uptime': self._local_step,
    })

    if self.total_steps:
      steps_remaining = self.total_steps - self._last_global_step
      metrics.update({
          'profile/estimated_hours_remaining': (
              steps_remaining / steps_per_sec / 3600)
      })

    # Metrics based on inputs.
    num_examples_per_step = self._total_examples / self.steps_profiled
    num_tokens_per_step = self._total_tokens / self.steps_profiled
    examples_per_sec = steps_per_sec * num_examples_per_step
    tokens_per_sec = steps_per_sec * num_tokens_per_step
    examples_per_sec_per_tpu = examples_per_sec / self._num_tpus
    tokens_per_sec_per_tpu = tokens_per_sec / self._num_tpus

    metrics.update({
        'profile/examples_per_sec': examples_per_sec,
        'profile/tokens_per_sec': tokens_per_sec,
        'profile/examples_per_sec_per_tpu': examples_per_sec_per_tpu,
        'profile/tokens_per_sec_per_tpu': tokens_per_sec_per_tpu,
    })

    # Metrics based on datasets.
    for dataset_name, num_steps in self._dataset_steps.items():
      percent_steps = num_steps / self.steps_profiled * 100
      metrics.update({
          f'{dataset_name}/percent_steps_sampled': percent_steps,
      })

    return metrics

  def reset(self):
    """Resets all metric statistics."""
    self.steps_profiled = 0
    self._total_examples = 0
    self._total_tokens = 0
    self._last_profile_time = time.time()
    self._dataset_steps.clear()


class BaseExecutor:
  """An executor containing the train/evaluation pipeline."""

  def __init__(self,
               model,
               dataloaders,
               config,
               metrics = None,
               init_override = None):
    # set model, data, and configs
    self.model = model
    self.dataloaders = dataloaders
    self.config = config
    self.metrics = metrics
    self.init_override = init_override

    # log total number of available TPUs
    logging.info('Number of available TPUs: %d.', jax.device_count())

    # choose a device as lead host for writing summary
    self.lead_host = (jax.process_index() == 0)

    # configure profiling settings
    profiling_path = os.path.join(config.path, config.mode.value)
    self.profiler = Profiler(
        profiling_path, total_steps=config.optimization.total_steps)

    # if lead_host, write summary
    if self.lead_host:
      self.summary_writer = probing.SummaryWriter(profiling_path)

    # configure the partitioner
    self.partitioner = partitioning.Partitioner(
        **config.execution.partitioning.as_dict()
    )

    self.ckpt_manager = checkpointing.CheckpointManager(
        workdir=config.path,
        partitioner=self.partitioner,
        keep=config.optimization.max_checkpoints,
        checkpoint_data=config.data.checkpointing,
        lazy_save=False,
        lazy_restore=False,
        parallel_restore=True,
        prefix='checkpoint_',
    )

  def prepare_data_iterators(self, dataloaders=None):
    dataloaders = dataloaders or self.dataloaders

    if not dataloaders:
      raise ValueError(
          f'No dataloaders were configured. Received {dataloaders}.')

    for dataloader in dataloaders:
      dataloader['iterator'] = iter(dataloader['loader'])

    return dataloaders

  def prepare_objective_functions(self):
    """Transforms objective functions to pjitted-friendly arg type."""

    dataloaders = self.dataloaders
    obj_fns = objectives.get_objective(self.config.optimization.loss)
    if len(obj_fns) > 1 and len(obj_fns) != len(dataloaders):
      raise ValueError(
          f'Number of objective functions, `{len(obj_fns)}`, does not match '
          f"datasets', `{len(dataloaders)}`",
      )

    # transform the obj_fn to a JAX type to be used as an arg to the pjitted
    # train_step later in the training loop
    # we make sure to cache the transformed obj_fns to avoid re-compiling
    # the smae object, hence avoiding re-tracing in train_step
    unique_transformed_obj_fns = {}
    transformed_obj_fns = ()
    for obj_fn in obj_fns:
      unique_obj_id = hash(obj_fn)
      if unique_obj_id not in unique_transformed_obj_fns:
        unique_transformed_obj_fns[unique_obj_id] = jax.tree_util.Partial(
            jax.jit(obj_fn)
            )
      transformed_obj_fns += (unique_transformed_obj_fns[unique_obj_id],)

    # if all datasets use one objective function, we replicate the same
    # instance to be static across different train_step calls
    if len(transformed_obj_fns) == 1:
      transformed_obj_fns *= len(dataloaders)

    return transformed_obj_fns

  def _maybe_unroll_probes(
      self,
      probes):
    """Unrolls the (possibly) scanned probes."""
    is_scanned = getattr(self.model, 'scanned_layers', False)
    if not is_scanned:
      return probes

    flattened_probes = _flatten_collection_tree(probes)
    unrolled_flattened_probes = {}
    for pname, probe in flattened_probes.items():
      try:
        # If `assert_data_rank` passes, the probe is not scanned
        unrolled_flattened_probes[pname] = probe.assert_data_rank()
      except ValueError:
        # Unroll the scanned probe and expand its naming correspondingly
        unrolled_probe = probe.unroll_scanned_data(0)
        for n, probe_n in enumerate(unrolled_probe):
          unrolled_flattened_probes[f'{pname}:layer_{n}'] = probe_n
    probes = _unflatten_collection_tree(unrolled_flattened_probes)
    return probes

  def initialize_states(
      self,
      mutable = _TRAIN_INIT_DENIED_COLLECTIONS,
  ):
    """Initializes all variables and partitions them accordingly."""
    set_hardware_rng_ops()
    rng, init_rngs = get_rngs(
        keys=self.model.get_rng_keys(),
        rng=None,
        rngs_per_key=1,
        add_params_rngs=True
    )

    if self.init_override:
      raise ValueError('init_override is deprecated.')

    # Fetch the optimizer
    # TODO(hassanak): make this modular and config-based-buildable
    optimizer = optimizers.get_optimizer(self.config.optimization.optimizer)

    @jax.jit
    def _run_init():
      # Initialize model variables
      variables = self.model.init(
          init_rngs,
          self.model.get_data_signature(),
          mutable=mutable,
      )
      # Separate model params and mutable variables
      params = {PARAMS: variables.pop(PARAMS)}
      mutables = variables

      # Construct the states
      boxed_state = flax_train_state.TrainState.create(
          apply_fn=self.model.apply,
          params=params,
          tx=optimizer,
      )
      return boxed_state, mutables

    with self.partitioner.mesh:
      # Run the jitted function to initialize
      boxed_state, mutables = _run_init()

      # Unbox all values
      unboxed_state = nn.unbox(boxed_state)

    # TODO(hassanak): Remove this upon removing the legacy T5X partitioner
    # Initialize partiontiner states based on the annotations in `state`
    self.partitioner.initialize_states(boxed_state, unboxed_state)

    return unboxed_state, mutables, rng

  def create_train_step(
      self
  ):
    """Creating a train_step to run optimization step."""
    model = self.model

    # WARNING: When data structure or the obj_fn changes, the graph is retraced.
    def train_step(
        state,
        mutables,
        data,
        rngs,
        obj_fn,
        microbatch_steps,
    ):
      """Performs forward and backward calls and update parameters."""
      params = dict(state.params)
      apply_and_grad_fn = gradients.apply_model_and_calculate_loss_and_grads
      (_, (mutables, probes, metrics)), grads = apply_and_grad_fn(
          params=params,
          mutables=mutables,
          model=model,
          data=data,
          rngs=rngs,
          obj_fn=obj_fn,
          mutable=_TRAIN_APPLY_DENIED_COLLECTIONS,
          microbatch_steps=microbatch_steps,
      )
      state = state.apply_gradients(grads=grads)

      return state, mutables, probes, metrics

    return train_step

  def create_evaluation_step(
      self,
      data_filter_fn = None,
  ):
    """Creating an evaluation_step to perform inference + metric calculation."""
    raise NotImplementedError(
        'This method should be implemented by the downstream application.')

  def create_inference_step(
      self,
      data_filter_fn = None,
  ):
    """Creates an inference_step to run evaluation step."""

    model = self.model

    def inference_step(
        params,
        mutables,
        data,
        ):
      """Performs a single forward call."""

      variables = {**params, **mutables}
      data, mutables = model.apply(
          variables=variables,
          data=data,
          deterministic=True,
          mutable=_EVAL_APPLY_DENIED_COLLECTIONS)

      # fetch probed information (if any)
      probes = mutables.pop(PROBES, {})

      # perform any arg filtering and/or tensor modification (if provided)
      if data_filter_fn is not None:
        data = data_filter_fn(data)

      return data, mutables, probes

    return inference_step

  def train(self):
    """Main train function."""

    # initialize model and train state
    logging.info('Initializing model...')
    state, mutables, rng = self.initialize_states(
        mutable=_TRAIN_INIT_DENIED_COLLECTIONS)

    # log total number of parameters
    logging.info(
        'Number of parameters in model: %f M.',
        count_params(state.params) / 1e6
    )

    # prepare data iterators
    dataloaders = self.prepare_data_iterators()

    # restore the latest checkpoint in experiment path
    # otherwise restore from restore_path (if provided)
    # otherwise continue with random initialization
    if flax_ckpt.latest_checkpoint(self.config.path):
      if self.config.optimization.restore_path:
        logging.warning(
            'A non-empty `restore_path` was provided while the current '
            'experiment path is not empty. `restore_path` will be ignored!')
      checkpoint_path = flax_ckpt.latest_checkpoint(self.config.path)
      state = self.ckpt_manager.restore(
          checkpoint_path, state, dataloaders, True)
      # Re-assure sharding
      state = sharding.shard_arrays_tree(
          state, self.partitioner.get_state_specs(),
          mesh=self.partitioner.mesh, enforce=True)
      logging.info('Checkpoint restored from the last saved checkpoint at step'
                   ' %d: %s', int(state.step), checkpoint_path)

    elif self.config.optimization.restore_path:
      checkpoint_path = self.config.optimization.restore_path
      state = self.ckpt_manager.restore(checkpoint_path, state, None, True)
      state = utils.replace_train_step(state, 0)
      # Re-assure sharding
      state = sharding.shard_arrays_tree(
          state, self.partitioner.get_state_specs(),
          mesh=self.partitioner.mesh, enforce=True)
      logging.info('Checkpoint restored from %s',
                   self.config.optimization.restore_path)
    else:
      logging.info('No checkpoint was found, skipping restoring step.')

    # transform objective functions to JAX type for jitted train step
    objective_fns = self.prepare_objective_functions()

    # Get metrics postprocessor
    metrics_postprocess_fn = self.config.optimization.metrics_postprocess_fn
    if metrics_postprocess_fn is not None:
      metrics_postprocess_fn = fdl.build(metrics_postprocess_fn)

    # Keep track of the value for each last-updated metric.
    # This is so that metrics can be aggregated even though each step only
    # emits a subset of metrics.
    recent_metrics = {}

    # get the current_step and total number of steps
    current_step = int(state.step)
    total_steps = self.config.optimization.total_steps

    # fetch and jit the train step
    train_step = jax.jit(
        self.create_train_step(),
        donate_argnums=(0, 1, 2, 3, 4),
        static_argnums=(5,),
    )

    # a helper function to read learning rate at a given step
    learning_rate = functools.partial(
        get_learning_rate,
        config=self.config.optimization.optimizer.learning_rate
    )

    # get all PRNGKeys for all training steps
    # this is necessary to avoid different seeds if an experiment restarts
    # it's also necessary to keep the train loop efficient and avoid overhead
    logging.info('Get all PRNGKeys...')
    _, all_rngs = get_rngs(
        self.model.get_rng_keys(),
        rng=rng,
        rngs_per_key=total_steps,
        add_params_rngs=False,
    )

    # start training
    data_iteration_counter = collections.Counter()
    logging.info('Train started...')
    while current_step < total_steps:
      with jax.profiler.StepTraceAnnotation('train', step_num=current_step):
        for dataloader, obj_fn in zip(dataloaders, objective_fns):
          dataset_name = dataloader['name']
          microbatch_steps = dataloader['config'].microbatch_splits

          # if outside optimization interval for this dataset, skip
          data_iteration_counter[dataset_name] += 1
          if data_iteration_counter[dataset_name] % dataloader['interval'] != 0:
            continue

          # add debug statement in case of errors on first step.
          if data_iteration_counter[dataset_name] == dataloader['interval']:
            logging.info('%s: initial step %d', dataset_name, current_step)

          # get next batch of data
          iterator = dataloader['iterator']
          data = next(iterator)

          # collect profile statistics of the current step
          self.profiler.update(current_step,
                               dataset_name=dataset_name,
                               data=data)
          if self.lead_host:
            self.profiler.maybe_start_profile_trace()

          # fetch rngs per key for the current step
          rngs = jax.tree.map(lambda x, step=current_step: x[step], all_rngs)

          # distribute data across processes
          data = multihost_utils.host_local_array_to_global_array(
              local_inputs=data,
              global_mesh=self.partitioner.mesh,
              pspecs=self.partitioner.get_data_specs(),
          )

          with self.partitioner.mesh:
            # perform train step
            state, mutables, probes, metrics = train_step(
                state, mutables, data, rngs, obj_fn, microbatch_steps)
          current_step += 1

          if (
              data_iteration_counter[dataset_name] % 50
              == dataloader['interval']
          ):
            # Fetch and replicate metrics across hosts/devices
            metrics = sharding.shard_arrays_tree(
                arrays_tree=metrics,
                shardings_tree=None,
                mesh=self.partitioner.mesh,
                enforce=True,
            )
            metrics = jax.tree.map(lambda v: v.addressable_data(0), metrics)
            metrics = mtu.tree_convert_jax_float_to_float(metrics)
            metrics = prepend_data_scope(metrics, dataset_name)
            _check_nan_values(metrics)
            metrics['learning_rate'] = learning_rate(step=current_step)

            if self.profiler.steps_profiled >= 1000:
              profile_metrics = self.profiler.get_metrics()
              metrics.update(profile_metrics)
              logging.info('Step profile %d: %s', current_step, profile_metrics)
              self.profiler.reset()

            recent_metrics.update(metrics)
            if metrics_postprocess_fn is not None:
              metrics.update(metrics_postprocess_fn(recent_metrics))

            logging.info('Train step %d: %s', current_step, metrics)

            if self.lead_host:
              if (data_iteration_counter[dataset_name] %
                  500 == dataloader['interval']):
                # Fetch and replicate probes across hosts/devices
                probes = sharding.shard_arrays_tree(
                    arrays_tree=probes,
                    shardings_tree=None,
                    mesh=self.partitioner.mesh,
                    enforce=True,
                )
                probes = jax.tree.map(lambda v: v.addressable_data(0), probes)
                probes = self._maybe_unroll_probes(probes)
                probes = mtu.tree_convert_jax_array_to_numpy(probes)
                probes = prepend_data_scope(probes, dataset_name)
              else:
                probes = {}
              # write metrics and probes after certain number of steps
              self.summary_writer(
                  metrics=metrics, probes=probes, step=current_step)

          # save checkpoint
          if (current_step % self.config.optimization.save_checkpoint_freq == 1
              or current_step == total_steps):
            logging.info('Saving checkpoint at training step %d.', current_step)
            self.ckpt_manager.save(state, dataloaders)

          if current_step == total_steps:
            # stop looping over additional data samples
            break

    # Block until complete on all hosts.
    multihost_utils.sync_global_devices('executor:training done.')

  def inference_iterator(
      self,
      model_params,
      mutables,
      dataloaders_override = None,
      data_filter_fn = None,
  ):
    """Performs inference on all datasets and yields the data.

    Args:
      model_params: A nested dictionary with model parameters as leaves.
      mutables: A nested dictionary with mutable collections' values as leaves.
      dataloaders_override: An optional Dataloader collection. If provided, the
        class-wide data will be ignored and the given dataloader will be
        used instead.
      data_filter_fn: An optional callable function that performs user-specific
        filtering on the data of the inference steps. If provided, it will
        manipulate the data before yielding them. This can be useful in cases
        where one does not need certain inputs/outputs/targets/etc. and wants
        to remove them before yielding them to save memory.
    Yields:
      A tuple of outputs that contains the following:
        (dataset_name, dataset_subset, data, outputs).
    """

    # get dataloaders
    dataloaders = dataloaders_override or self.dataloaders
    dataloaders = self.prepare_data_iterators(dataloaders)

    # fetch and jit the inference step
    inference_step = jax.jit(
        self.create_inference_step(data_filter_fn),
    )

    # start inference
    logging.info('Inference started...')
    init_mutables = mutables
    current_step = 0
    for dataloader in dataloaders:
      with jax.profiler.StepTraceAnnotation('infer', step_num=current_step):
        # reset mutables
        mutables = init_mutables

        # get next batch of data
        dataset_name = dataloader['name']
        dataset_subset = dataloader['subset']
        iterator = dataloader['iterator']
        loader_config = dataloader['config']

        logging.info('Running inference on %s_%s: %s',
                     dataset_name, dataset_subset, loader_config.as_dict())

        while True:
          try:
            # fetch data
            data = next(iterator)

            # pad data along the batch dimension (if not shardable)
            data, batch_mask = self.partitioner.maybe_pad_batches(
                data, int(loader_config.batch_size / jax.process_count()))

            # distribute data across processes
            data = multihost_utils.host_local_array_to_global_array(
                local_inputs=data,
                global_mesh=self.partitioner.mesh,
                pspecs=self.partitioner.get_data_specs(),
            )

            with self.partitioner.mesh:
              # perform inference step
              data, mutables, probes = inference_step(
                  model_params, mutables, data)

            # transfer data to local hosts
            data = multihost_utils.global_array_to_host_local_array(
                global_inputs=data,
                global_mesh=self.partitioner.mesh,
                pspecs=self.partitioner.get_data_specs(),
            )
            # Fetch and replicate probes across hosts/devices
            probes = sharding.shard_arrays_tree(
                arrays_tree=probes,
                shardings_tree=None,
                mesh=self.partitioner.mesh,
                enforce=True,
            )
            probes = jax.tree.map(lambda v: v.addressable_data(0), probes)

            # unpad data (if the original batches were padded)
            data = self.partitioner.maybe_unpad_batches(data, batch_mask)

            # increase step
            current_step += 1
            if current_step % 50 == 0:
              logging.info('Inference step %d on %s_%s',
                           current_step, dataset_name, dataset_subset)

            yield dataset_name, dataset_subset, data, probes

          except StopIteration:
            break

  # TODO(b/239240172): add efficient storing option
  def inference_loop(
      self,
      model_params,
      mutables,
      dataloaders_override = None,
      data_filter_fn = None,
      ):
    """Performs full inference on all datasets and returns all outputs.

    Args:
      model_params: A nested dictionary with model parameters as leaves
      mutables: A nested dictionary with mutable collections' values as leaves.
      dataloaders_override: An optional Dataloader collection. If provided, the
        class-wide data will be ignored and the given dataloader will be
        used instead.
      data_filter_fn: An optional callable function that performs user-specific
        filtering on the data of the inference steps. If provided, it will
        manipulate the data before yielding them. This can be useful in cases
        where one does not need certain inputs/outputs/targets/etc. and wants
        to remove them before yielding them to save memory.
    Returns:
      A nested array that contains the model's outputs for ALL samples in
      the dataloaders. The samples are concatenated along the batch axis (0).
      The expected structure is {dataset_name: {dataset_subset: all_outputs}}.
    """

    inference_iterator = self.inference_iterator(
        model_params, mutables, dataloaders_override, data_filter_fn)

    default_dict_fn = lambda: collections.defaultdict(list)
    all_data = collections.defaultdict(default_dict_fn)
    all_probes = collections.defaultdict(default_dict_fn)
    for ds_name, ds_subset, data, probes in inference_iterator:
      data = mtu.tree_convert_jax_array_to_numpy(data)
      probes = self._maybe_unroll_probes(probes)
      probes = mtu.tree_convert_jax_array_to_numpy(probes)
      all_data[ds_name][ds_subset].append(data)
      all_probes[ds_name][ds_subset].append(probes)

    # concatenate all steps
    for ds_name in all_data:
      for ds_subset in all_data[ds_name]:
        data_slices = all_data[ds_name][ds_subset]
        probes_slices = all_probes[ds_name][ds_subset]
        if jax.process_count() > 1:
          # If multi-host, all-gather across processes. Probes are already
          # replicated and do not need all-gather.
          data_slices = self.partitioner.all_gather_slices_across_processes(
              data_slices)

        # aggregate along the batch dimension
        aggregate_fn = functools.partial(
            aggregate_multi_step_data,
            scalar_aggregate_fn=np.mean,
            batch_aggregate_fn=np.concatenate,
            batch_axis=0,
        )
        all_data[ds_name][ds_subset] = aggregate_fn(data_slices)
        all_probes[ds_name][ds_subset] = aggregate_fn(probes_slices)

    return all_data, all_probes

  def evaluation_loop(
      self,
      model_params,
      mutables,
      data_filter_fn = None,
      metrics_postprocess_fn = None,
  ):
    """Performs a full evaluation on the entire datasets given a checkpoint."""

    # get dataloaders
    dataloaders = self.prepare_data_iterators(self.dataloaders)

    # fetch and jit the evaluation step
    evaluation_step = jax.jit(
        self.create_evaluation_step(data_filter_fn),
    )

    # start inference
    logging.info('Evaluation started...')
    init_mutables = mutables
    current_step = 0
    all_probes = []
    all_metrics = []
    for dataloader in dataloaders:
      with jax.profiler.StepTraceAnnotation('eval', step_num=current_step):
        # reset mutables
        mutables = init_mutables

        # get next batch of data
        dataset_name = dataloader['name']
        iterator = dataloader['iterator']

        while True:
          try:
            # fetch data
            data = next(iterator)

            # distribute data across processes
            data = multihost_utils.host_local_array_to_global_array(
                local_inputs=data,
                global_mesh=self.partitioner.mesh,
                pspecs=self.partitioner.get_data_specs(),
            )

            # perform inference step
            with self.partitioner.mesh:
              mutables, probes, metrics = evaluation_step(
                  model_params, mutables, data)

            # Convert probes to float and prepend proper context name
            probes = sharding.shard_arrays_tree(
                arrays_tree=probes,
                shardings_tree=None,
                mesh=self.partitioner.mesh,
                enforce=True,
            )
            probes = jax.tree.map(lambda v: v.addressable_data(0), probes)
            probes = self._maybe_unroll_probes(probes)
            probes = mtu.tree_convert_jax_array_to_numpy(probes)
            probes = prepend_data_scope(probes, dataset_name)
            all_probes.append(probes)

            # Convert metrics to float and prepend proper context name
            metrics = sharding.shard_arrays_tree(
                arrays_tree=metrics,
                shardings_tree=None,
                mesh=self.partitioner.mesh,
                enforce=True,
            )
            metrics = jax.tree.map(lambda v: v.addressable_data(0), metrics)
            metrics = mtu.tree_convert_jax_float_to_float(metrics)
            metrics = prepend_data_scope(metrics, dataset_name)
            all_metrics.append(metrics)

            # increase step
            current_step += 1
            if current_step % 50 == 0:
              logging.info('Evaluation step %d: %s', current_step, metrics)

          except StopIteration:
            break

    # Aggregate all probes and metrics across steps
    aggregate_fn = functools.partial(
        aggregate_multi_step_data,
        scalar_aggregate_fn=np.mean,
        batch_aggregate_fn=np.concatenate,
        batch_axis=0,
    )
    all_probes = aggregate_fn(all_probes)
    all_metrics = aggregate_fn(all_metrics)
    if metrics_postprocess_fn is not None:
      all_metrics.update(metrics_postprocess_fn(all_metrics))

    return all_probes, all_metrics

  def evaluate(self):
    """Iterates over checkpoints OR gets a ckpt path and evaluates the model."""

    # initialize model and train state
    logging.info('Initializing states...')
    state, mutables, _ = self.initialize_states(
        mutable=_EVAL_INIT_DENIED_COLLECTIONS)

    # log total number of parameters
    logging.info(
        'Number of parameters in model: %f M.',
        count_params(state.params) / 1e6
    )

    # if restore_path is provided inside the eval config, restore from that
    # path. otherwise, iterate over the checkpoints in the experiment path and
    # restore from the latest checkpoints and perform evaluation continuously
    # until it reaches the latest training step's saved checkpoint
    if self.config.evaluation.restore_path:
      checkpoint_path_iterator = [self.config.evaluation.restore_path]
      logging.info('Override checkpoint found. Restoring the model from the '
                   'checkpoint at %s.', self.config.evaluation.restore_path)

    else:
      # iterate over checkpoints as they appear
      checkpoint_path_iterator = checkpointing.checkpoints_iterator(
          self.config.path
          )
      logging.info('Override checkpoint not found. Iterating over the current '
                   'experiment directory at %s.', self.config.path)

    for checkpoint_path in checkpoint_path_iterator:
      state = self.ckpt_manager.restore(checkpoint_path, state, None, True)
      # Re-assure sharding
      state = sharding.shard_arrays_tree(
          state, self.partitioner.get_state_specs(),
          mesh=self.partitioner.mesh, enforce=True)
      logging.info('Model parameters restored from the saved checkpoint at '
                   'step %d: %s', int(state.step), checkpoint_path)

      # perform evaluation
      metrics_postprocess_fn = self.config.evaluation.metrics_postprocess_fn
      if metrics_postprocess_fn is not None:
        metrics_postprocess_fn = fdl.build(metrics_postprocess_fn)

      probes, metrics = self.evaluation_loop(
          model_params=state.params,
          mutables=mutables,
          metrics_postprocess_fn=metrics_postprocess_fn)

      logging.info('Evaluation metrics at step %d: %s', state.step, metrics)

      # write metrics to tensorboard
      if self.lead_host:
        self.summary_writer(probes=probes, metrics=metrics, step=state.step)

      # break the loop if in iterating mode and last checkpoint is restored
      if (not self.config.evaluation.restore_path
          and state.step == self.config.optimization.total_steps):
        logging.info('Reached total steps: %d, exitting...', state.step)
        break

    # Block until complete on all hosts.
    multihost_utils.sync_global_devices('executor:evaluation done.')

  def run(self, mode):
    """Executes the program based on the mode.

    Args:
      mode: the mode to run, train or eval

    Raises:
      ValueError: if the mode is unsupported.
    """

    if mode is Mode.TRAIN:
      self.train()
    elif mode is Mode.EVAL:
      self.evaluate()
    else:
      raise ValueError(f'Mode {mode} not supported!')


# TODO(b/276938664): Make feature_route/name selection automatic or configurable
class Executor(BaseExecutor):
  """Executor for train and evaluate on multiple datasets."""

  def __init__(self,
               model,
               dataloaders,
               config,
               init_override = None):
    super().__init__(model=model, dataloaders=dataloaders, config=config,
                     init_override=init_override)
    self.original_dataloaders = self.dataloaders

  def _create_inference_data_filter_fn(
      self,
      dataset_name
  ):
    """Constructs a function to filter the data in inference."""
    video_retrieval_datasets = {'msrvtt', 'msrvtt-1000', 'msvd', 'youcook2'}

    def filter_fn(data):
      all_feature_types = sorted(data.keys())
      for feature_type in all_feature_types:
        if feature_type in (INPUTS, HYPERPARAMS, METADATA):
          # Remove any of the input features, hyperparams, and metadata
          del data[feature_type]
          continue

        all_routes = sorted(data[feature_type].keys())
        for route in all_routes:

          all_modalities = sorted(data[feature_type][route].keys())
          for modality in all_modalities:
            all_feature_names = sorted(
                data[feature_type][route][modality].keys()
            )
            if feature_type == TARGETS:
              # We only keep the `label` annotation from the data to save memory
              if LABEL not in all_feature_names:
                # remove the entire modality if it does not have label
                del data[feature_type][route][modality]
              else:
                for feature_name in all_feature_names:
                  if feature_name != LABEL:
                    # remove the non-label feature names
                    del data[feature_type][route][modality][feature_name]

            if feature_type == OUTPUTS:
              # remove non-aggregated features to save memory
              for feature_name in all_feature_names:
                if route != COMMON_SPACE:
                  all_output_feature_names = sorted(
                      data[feature_type][route][modality][feature_name].keys()
                  )
                  for output_feature_name in all_output_feature_names:
                    if output_feature_name not in (LOGITS, FEATURES_AGG):
                      # remove the non-label feature names
                      del data[feature_type][route][modality][feature_name][
                          output_feature_name
                      ]
                else:
                  # Taking an average over test clips of video retrieval
                  # datasets because the retrieval metric function considers
                  # best instance while ranking the candidates - this
                  # 'best instance' selection is necessary for flickr30k, COCO,
                  # and any multi-caption datasets but not for video retrieval
                  # datasets that contain multiple video clips (not multiple
                  # captions)
                  if (
                      modality == Modality.VISION
                      and dataset_name in video_retrieval_datasets
                  ):
                    # fetch vision-to-common_space embeddings
                    vision_to_common = data[feature_type][route][modality][
                        feature_name
                    ]

                    # Take average over the instance dimension of vision-to-*
                    # embeddings
                    target_modalities = sorted(vision_to_common.keys())
                    for target_modality in target_modalities:
                      avg_embedding = vision_to_common[target_modality].mean(
                          axis=1, keepdims=True
                      )
                      data[feature_type][route][modality][feature_name][
                          target_modality] = avg_embedding

        if not data[feature_type]:
          # If this collection is fully wiped out, remove the key
          del data[feature_type]

      return data

    return filter_fn

  def evaluation_loop(
      self,
      model_params,
      mutables,
      data_filter_fn = None,
      metrics_postprocess_fn = None,
  ):
    # create metrics stack
    metrics_stack = metrics_lib.create_metric_stack(
        self.config.evaluation.metric)

    # group dataloaders based on dataset
    grouped_dataloaders = group_dataloaders(self.original_dataloaders)

    all_probes = {}
    all_metrics = {}
    # to avoid OOM, we manually iterate over datasets one by one
    for dataloaders in grouped_dataloaders:
      # fetch dataset name
      dataset_name = dataloaders[0]['name']

      # fetch serving strategy
      servings = dataloaders[0]['serving']
      servings = (servings,) if isinstance(servings, str) else servings
      servings = set(servings)

      # Construct the filter function. This function modifies the data
      # before aggregation for maximum efficiency.
      # See BaseExecutor.inference_iterator for details.
      inference_data_filter_fn = self._create_inference_data_filter_fn(
          dataloaders[0]['name'])

      bulk_servings = servings.intersection(_SUPPORTED_BULK_SERVINGS)
      online_servings = servings.intersection(_SUPPORTED_ONLINE_SERVINGS)
      unsupported_servings = servings.difference(_SUPPORTED_SERVINGS)

      if unsupported_servings:
        raise NotImplementedError(f'Serving strategy {unsupported_servings} '
                                  'not supported.')

      if bulk_servings:
        # calculate output features
        data, probes = self.inference_loop(
            model_params=model_params,
            mutables=mutables,
            dataloaders_override=dataloaders,
            data_filter_fn=inference_data_filter_fn)

        if not data:
          raise ValueError('Inference loop returned an empty data '
                           f'for dataset {dataset_name!r}.')

        probes = prepend_data_scope(probes, dataset_name)
        all_probes.update(probes)

        for serving in sorted(bulk_servings):
          bulk_eval_fn = _BULK_SERVINGS[serving]
          eval_superset = _SERVING_SUPERSETS[serving]
          eval_metrics = bulk_eval_fn(data, metrics_stack[eval_superset])

          # update the global eval metrics accordingly
          all_metrics.update(eval_metrics)

      if online_servings:
        # sort dataloaders based on train->test/valid order
        sorted_dataloaders = sort_grouped_dataloaders_by_subset(dataloaders)

        # set executor dataloaders to train split of this dataset and get an
        # inference iterator based on it
        train_iterator = self.inference_iterator(
            model_params=model_params, mutables=mutables,
            dataloaders_override=(sorted_dataloaders['train'],),
            data_filter_fn=inference_data_filter_fn)

        # set executor data to test split of this dataset and get an
        # inference iterator based on it
        test_iterator = self.inference_iterator(
            model_params=model_params, mutables=mutables,
            dataloaders_override=(sorted_dataloaders['test'],),
            data_filter_fn=inference_data_filter_fn)

        for serving in sorted(online_servings):
          online_eval_fn = _ONLINE_SERVINGS[serving]
          eval_superset = _SERVING_SUPERSETS[serving]
          eval_metrics = online_eval_fn(train_iterator,
                                        test_iterator,
                                        dataset_name,
                                        metrics_stack[eval_superset])

          # update the global eval metrics accordingly
          all_metrics.update(eval_metrics)

    if metrics_postprocess_fn is not None:
      all_metrics.update(metrics_postprocess_fn(all_metrics))

    return all_probes, all_metrics
