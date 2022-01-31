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

# Lint as: python3
"""Training logic for variable misuse task."""

import contextlib
import json
import os
import re
import time
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from absl import logging
import dataclasses

import flax
import gin
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow.io import gfile

from gfsa import automaton_builder
from gfsa import jax_util
from gfsa.datasets import data_loading
from gfsa.datasets import graph_bundle
from gfsa.datasets.var_misuse import example_definition
from gfsa.model import side_outputs
from gfsa.model import var_misuse_models
from gfsa.training import train_util


@gin.configurable
def loss_fn(
    model,
    padded_example_and_rng,
    static_metadata,
    regularization_weights = None,
    reinforce_weight = 1.0,
    baseline_weight = 0.001,
):
  """Loss function for multi-pointer task.

  Args:
    model: The model to evaluate.
    padded_example_and_rng: Padded example to evaluate on, with a PRNGKey.
    static_metadata: Padding configuration for the example, since this may vary
      for different examples.
    regularization_weights: Associates side output key regexes with
      regularization penalties.
    reinforce_weight: Weight to give to the reinforce term.
    baseline_weight: Weight to give to the baseline.

  Returns:
    Tuple of loss and metrics.
  """
  padded_example, rng = padded_example_and_rng

  # Run the model.
  with side_outputs.collect_side_outputs() as collected_side_outputs:
    with flax.deprecated.nn.stochastic(rng):
      joint_log_probs = model(padded_example, static_metadata)

  # Computing the loss:
  # Extract logits for the correct location.
  log_probs_at_bug = joint_log_probs[padded_example.bug_node_index, :]
  # Compute p(repair) = sum[ p(node) p(repair | node) ]
  # -> log p(repair) = logsumexp[ log p(node) + log p (repair | node) ]
  log_prob_joint = jax.scipy.special.logsumexp(
      log_probs_at_bug + jnp.log(padded_example.repair_node_mask))

  # Metrics:
  # Marginal log probabilities:
  log_prob_bug = jax.scipy.special.logsumexp(log_probs_at_bug)
  log_prob_repair = jax.scipy.special.logsumexp(
      jax.scipy.special.logsumexp(joint_log_probs, axis=0) +
      jnp.log(padded_example.repair_node_mask))

  # Conditional log probabilities:
  log_prob_repair_given_bug = log_prob_joint - log_prob_bug
  log_prob_bug_given_repair = log_prob_joint - log_prob_repair

  # Majority accuracy (1 if we assign the correct tuple > 50%):
  # (note that this is easier to compute, since we can't currently aggregate
  # probability separately for each candidate.)
  log_half = jnp.log(0.5)
  majority_acc_joint = log_prob_joint > log_half

  # Probabilities associated with each node.
  node_node_probs = jnp.exp(joint_log_probs)
  # Accumulate across unique candidates by identifier. This has the same shape,
  # but only the first few values will be populated.
  node_candidate_probs = padded_example.unique_candidate_operator.apply_add(
      in_array=node_node_probs,
      out_array=jnp.zeros_like(node_node_probs),
      in_dims=[1],
      out_dims=[1])

  # Classify: 50% decision boundary
  only_buggy_probs = node_candidate_probs.at[0, :].set(0).at[:, 0].set(0)
  p_buggy = jnp.sum(only_buggy_probs)
  pred_nobug = p_buggy <= 0.5

  # Localize/repair: take most likely bug position, conditioned on being buggy
  pred_bug_loc, pred_cand_id = jnp.unravel_index(
      jnp.argmax(only_buggy_probs), only_buggy_probs.shape)

  actual_nobug = jnp.array(padded_example.bug_node_index == 0)

  actual_bug = jnp.logical_not(actual_nobug)
  pred_bug = jnp.logical_not(pred_nobug)

  metrics = {
      'nll/joint':
          -log_prob_joint,
      'nll/marginal_bug':
          -log_prob_bug,
      'nll/marginal_repair':
          -log_prob_repair,
      'nll/repair_given_bug':
          -log_prob_repair_given_bug,
      'nll/bug_given_repair':
          -log_prob_bug_given_repair,
      'inaccuracy/legacy_overall':
          1 - majority_acc_joint,
      'inaccuracy/overall':
          (~((actual_nobug & pred_nobug) |
             (actual_bug & pred_bug &
              (pred_bug_loc == padded_example.bug_node_index) &
              (pred_cand_id == padded_example.repair_id)))),
      'inaccuracy/classification_overall': (actual_nobug != pred_nobug),
      'inaccuracy/classification_given_nobug':
          train_util.RatioMetric(
              numerator=(actual_nobug & ~pred_nobug), denominator=actual_nobug),
      'inaccuracy/classification_given_bug':
          train_util.RatioMetric(
              numerator=(actual_bug & ~pred_bug), denominator=actual_bug),
      'inaccuracy/localized_given_bug':
          train_util.RatioMetric(
              numerator=(actual_bug
                         & ~(pred_bug_loc == padded_example.bug_node_index)),
              denominator=actual_bug),
      'inaccuracy/repaired_given_bug':
          train_util.RatioMetric(
              numerator=(actual_bug
                         & ~(pred_cand_id == padded_example.repair_id)),
              denominator=actual_bug),
      'inaccuracy/localized_repaired_given_bug':
          train_util.RatioMetric(
              numerator=(actual_bug
                         & ~((pred_bug_loc == padded_example.bug_node_index) &
                             (pred_cand_id == padded_example.repair_id))),
              denominator=actual_bug),
      'inaccuracy/overall_given_bug':
          train_util.RatioMetric(
              numerator=(actual_bug
                         & ~(pred_bug &
                             (pred_bug_loc == padded_example.bug_node_index) &
                             (pred_cand_id == padded_example.repair_id))),
              denominator=actual_bug),
  }

  loss = -log_prob_joint

  for k, v in collected_side_outputs.items():
    # Flax collection keys will start with "/".
    if v.shape == ():  # pylint: disable=g-explicit-bool-comparison
      metrics['side' + k] = v

  if regularization_weights:
    total_regularization = 0
    for query, weight in regularization_weights.items():
      logging.info('Regularizing side outputs matching query %s', query)
      found = False
      for k, v in collected_side_outputs.items():
        if re.search(query, k):
          found = True
          logging.info('Regularizing %s with weight %f', k, weight)
          total_regularization += weight * v
      if not found:
        raise ValueError(
            f'Regularization query {query} did not match any side output. '
            f'Side outputs were {set(collected_side_outputs.keys())}')

    loss = loss + total_regularization

  is_single_sample = any(
      k.endswith('one_sample_log_prob_per_edge_per_node')
      for k in collected_side_outputs)
  if is_single_sample:
    log_prob, = [
        v for k, v in collected_side_outputs.items()
        if k.endswith('one_sample_log_prob_per_edge_per_node')
    ]
    baseline, = [
        v for k, v in collected_side_outputs.items()
        if k.endswith('one_sample_reward_baseline')
    ]

    num_real_nodes = padded_example.input_graph.bundle.graph_metadata.num_nodes
    valid_mask = (
        jnp.arange(static_metadata.bundle_padding.static_max_metadata.num_nodes)
        < num_real_nodes)
    log_prob = jnp.where(valid_mask[None, :], log_prob, 0)
    total_log_prob = jnp.sum(log_prob)

    reinforce_virtual_cost = (
        total_log_prob * jax.lax.stop_gradient(loss - baseline))
    baseline_penalty = jnp.square(loss - baseline)

    reinforce_virtual_cost_zeroed = reinforce_virtual_cost - jax.lax.stop_gradient(
        reinforce_virtual_cost)

    loss = (
        loss + reinforce_weight * reinforce_virtual_cost_zeroed +
        baseline_weight * baseline_penalty)
    metrics['reinforce_virtual_cost'] = reinforce_virtual_cost
    metrics['baseline_penalty'] = baseline_penalty
    metrics['baseline'] = baseline
    metrics['total_log_prob'] = total_log_prob

  metrics = jax.tree_map(lambda x: x.astype(jnp.float32), metrics)
  return loss, metrics


def build_padding_config(
    log2_num_nodes, log2_num_input_tagged_nodes,
    log2_max_initial_transitions, log2_max_in_tagged_transitions,
    log2_max_edges, log2_max_tokens
):
  """Builds a padding config with power-of-2 sizes."""
  return example_definition.GraphBundleWithTokensPaddingConfig(
      bundle_padding=graph_bundle.PaddingConfig(
          static_max_metadata=automaton_builder.EncodedGraphMetadata(
              num_nodes=2**log2_num_nodes,
              num_input_tagged_nodes=2**log2_num_input_tagged_nodes),
          max_initial_transitions=2**log2_max_initial_transitions,
          max_in_tagged_transitions=2**log2_max_in_tagged_transitions,
          max_edges=2**log2_max_edges),
      max_tokens=2**log2_max_tokens)


PaddingAndBatchSizes = (
    List[Tuple[example_definition.GraphBundleWithTokensPaddingConfig, int]])

TINY_PADDING_CONFIG = example_definition.GraphBundleWithTokensPaddingConfig(
    bundle_padding=graph_bundle.PaddingConfig(
        static_max_metadata=automaton_builder.EncodedGraphMetadata(
            num_nodes=1, num_input_tagged_nodes=1),
        max_initial_transitions=1,
        max_in_tagged_transitions=1,
        max_edges=1),
    max_tokens=1)


def pad_and_batch_with_rng(
    it, num_devices,
    padding_and_batch_sizes,
    base_rng):
  """Pad and batch according to a collection of sizes.

  Args:
    it: Iterable over individual examples.
    num_devices: Number of devices; determines constant leading batch dimension.
    padding_and_batch_sizes: List of pairs of padding config and per-device
      batch size. Padding configs will be tried in order until the example fits
      in one.
    base_rng: PRNGKey to use to generate RNG seeds.

  Yields:
    Batched tuples of padded examples and RNG keys. Each batch will contain
    examples of approximately the same shape, and the `static_metadata` field
    for each will be the padding config used. RNG keys are deterministically
    based on `base_rng` and the order of examples in `it` (i.e. the nth
    example from `it` will get a specific RNG value, regardless of padding and
    batch sizes).
  """

  # Assign each example to a bucket, and pad it appropriately
  def _find_buckets_and_pad():
    for example_number, ex in enumerate(it):
      padded_example = None
      for (current_bucket, (padding_config,
                            _)) in enumerate(padding_and_batch_sizes):
        padded_example = example_definition.pad_example(
            ex.example, padding_config, allow_failure=True)
        if padded_example:
          bucket = current_bucket
          break

      if padded_example:
        example_rng = jax.random.fold_in(base_rng, example_number)
        yield (bucket,
               dataclasses.replace(ex, example=(padded_example, example_rng)))
      else:
        logging.info('Dropping example %d (exceeded padding config)',
                     ex.example_id)

  # Batch within each bucket.
  batched = data_loading.batch_bucketed(
      _find_buckets_and_pad(),
      batch_dim_sizes={
          i: (num_devices, device_batch_size)
          for i, (_, device_batch_size) in enumerate(padding_and_batch_sizes)
      },
      remainder_behavior=data_loading.BatchRemainderBehavior.PAD_ZERO)

  # Move the bucket's padding config into the batch metadata.
  for bucket, ex in batched:
    yield dataclasses.replace(
        ex, static_metadata=padding_and_batch_sizes[bucket][0])


@gin.configurable
def train(
    runner,
    dataset_paths = gin.REQUIRED,
    padding_and_batch_sizes = gin.REQUIRED,
    prefetch = 8,
    validation_example_count = gin.REQUIRED,
    evaluate_only = False,
    evaluation_model_path = None,
    evaluation_save_path = None,
    parameter_seed = int(time.time() * 1000),
    profile_during_warmup = True,
    restore_from_path = None,
):
  """Training script entry point.

  Args:
    runner: Helper object that runs the experiment.
    dataset_paths: Dictionary of dataset paths, with keys:
      - "train_dataset": Path to training dataset files.
      - "train_metadata": Path to JSON file with training dataset metadata.
      - "eval_dataset": Path to validation/test dataset files.
      - "eval_metadata": Path to JSON file with eval dataset metadata.
    padding_and_batch_sizes: Padding configurations and batch sizes to use.
      Padding should be specified using the keyword arguments for the function
      `build_padding_config`. Batch sizes may be None, in which case we will try
      to find the maximum batch size for each padding config.
    prefetch: How many examples to prefetch.
    validation_example_count: How many examples to use when validating during
      training. If None, uses all examples.
    evaluate_only: If True, doesn't run any training; instead evaluates a
      trained model on the validation/evaluation set.
    evaluation_model_path: Where to load the model from.
    evaluation_save_path: Where to save the result JSON file.
    parameter_seed: Random seed to use when initializing parameters.
    profile_during_warmup: Whether to use XProf during warmup.
    restore_from_path: Optional path to restore parameters from; useful for
      warm-starting.

  Returns:
    Final optimizer.
  """

  num_devices = jax.local_device_count()
  logging.info('Found %d devices: %s', num_devices, jax.devices())

  padding_and_batch_sizes = [
      (build_padding_config(**config_kwargs), batch_size)
      for config_kwargs, batch_size in padding_and_batch_sizes
  ]

  with contextlib.ExitStack() as exit_stack:
    # Loading metadata and task info.
    with gfile.GFile(dataset_paths['train_metadata'], 'r') as fp:
      train_metadata = json.load(fp)
    with gfile.GFile(dataset_paths['eval_metadata'], 'r') as fp:
      valid_metadata = json.load(fp)

    assert train_metadata['spec_file'] == valid_metadata['spec_file']
    assert train_metadata['vocab_file'] == valid_metadata['vocab_file']
    assert train_metadata['edge_types'] == valid_metadata['edge_types']

    encoding_info = example_definition.ExampleEncodingInfo.from_files(
        train_metadata['spec_file'], train_metadata['vocab_file'])
    assert encoding_info.edge_types == train_metadata['edge_types']

    # Model setup.
    logging.info('Setting up model...')
    model_def = var_misuse_models.var_misuse_model.partial(
        encoding_info=encoding_info)

    # Set up a dummy stochastic scope for random perturbations.
    with flax.deprecated.nn.stochastic(jax.random.PRNGKey(0)):
      # Initialize parameters based on our seed.
      _, initial_params = model_def.init(
          jax.random.PRNGKey(parameter_seed),
          jax.tree_map(
              jnp.array,
              example_definition.zeros_like_padded_example(
                  TINY_PADDING_CONFIG)), TINY_PADDING_CONFIG)

    model = flax.deprecated.nn.Model(model_def, initial_params)
    del initial_params
    optimizer = flax.optim.Adam().create(model)

    if restore_from_path:
      optimizer, checkpoint_info = runner.load_from_checkpoint(
          optimizer, restore_from_path)
      logging.info('Warm starting from checkpoint with info: %s',
                   checkpoint_info)

    # Compute missing batch sizes.
    tmp_replicated_optimizer = train_util.device_broadcast(
        optimizer, num_devices)
    for i, (padding_config, batch_size) in enumerate(padding_and_batch_sizes):
      fake_example_and_rng = (
          example_definition.zeros_like_padded_example(padding_config),
          jax.random.PRNGKey(0))
      assert batch_size is not None
      logging.info(
          'Running a fake train step for batch size %d and padding config %d: %s',
          batch_size, i, padding_config)
      # pylint: disable=cell-var-from-loop
      fake_batch = jax.vmap(lambda _: fake_example_and_rng)(
          jnp.arange(batch_size))
      fake_device_batch = jax.vmap(lambda _: fake_batch)(
          jnp.arange(num_devices))
      # pylint: enable=cell-var-from-loop
      train_util.warmup_train_step(
          tmp_replicated_optimizer,
          fake_device_batch,
          padding_config,
          loss_fn,
          optimizer_is_replicated=True,
          profile=profile_during_warmup,
          runner=runner)

    del tmp_replicated_optimizer

    extra_artifacts = {
        'encoding_info.pickle': encoding_info,
    }

    # Dataset iterator setup.
    logging.info('Setting up datasets...')
    unbatched_train_iterator = runner.build_sampling_iterator(
        dataset_paths['train_dataset'],
        example_type=example_definition.VarMisuseExample)

    # Randomly generate the base RNG. (Since the iterator is already randomly
    # shuffling, and we might have restarted this job anyway, there's no point
    # in setting a seed here.)
    train_iterator = pad_and_batch_with_rng(
        unbatched_train_iterator,
        num_devices,
        padding_and_batch_sizes,
        base_rng=jax.random.PRNGKey(int(time.time() * 1000)))
    if prefetch:
      train_iterator = exit_stack.enter_context(
          data_loading.ThreadedPrefetcher(train_iterator, prefetch))

    unbatched_valid_iterator_factory = (
        runner.build_one_pass_iterator_factory(
            dataset_paths['eval_dataset'],
            example_type=example_definition.VarMisuseExample,
            truncate_at=validation_example_count))

    def logging_progress(it):
      maxct = validation_example_count or valid_metadata['num_examples']
      for i, val in enumerate(it):
        if i % 10000 == 0:
          logging.info('Validation progress: %d of %d', i, maxct)
        yield val

    def valid_iterator_factory():
      unbatched = unbatched_valid_iterator_factory()
      if evaluate_only:
        unbatched = logging_progress(unbatched)
      # Always use the same PRNGKey for the validation set.
      valid_iterator = pad_and_batch_with_rng(
          unbatched,
          num_devices,
          padding_and_batch_sizes,
          base_rng=jax.random.PRNGKey(0))
      if prefetch:
        with data_loading.ThreadedPrefetcher(valid_iterator, prefetch) as it:
          yield from it
      else:
        yield from valid_iterator

    validation_fn = train_util.build_averaging_validator(
        loss_fn,
        valid_iterator_factory,
        objective_metric_name='inaccuracy/overall',
        include_total_counts=evaluate_only)

    if evaluate_only:
      logging.warning('This job is running in evaluation mode!')

      optimizer, checkpoint_info = runner.load_from_checkpoint(
          optimizer, checkpoint_path=evaluation_model_path)
      model = train_util.device_broadcast(optimizer.target, num_devices)

      _, metrics = validation_fn(model)
      metrics['checkpoint_info'] = checkpoint_info
      metrics['model_path'] = evaluation_model_path
      metrics['dataset_path'] = dataset_paths['eval_dataset']
      metrics['example_count'] = validation_example_count

      array_types = (np.ndarray, jnp.ndarray)
      metrics = jax.tree_map(
          lambda x: x.tolist() if isinstance(x, array_types) else x, metrics)

      gfile.makedirs(os.path.dirname(evaluation_save_path))
      with gfile.GFile(evaluation_save_path, 'w') as fp:
        json.dump(metrics, fp, indent=2)

      logging.info('Computed evaluation metrics: %s', metrics)

    else:
      return runner.training_loop(
          optimizer=optimizer,
          train_iterator=train_iterator,
          loss_fn=loss_fn,
          validation_fn=validation_fn,
          extra_artifacts=extra_artifacts)
