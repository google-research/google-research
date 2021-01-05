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

# Lint as: python3
"""Training logic for edge supervision task.

This module contains the training and evaluation logic for edge-supervision
tasks on Python ASTs.

To launch a training job, use the executable script `train_edge_supervision.py`.
"""

import contextlib
import functools
import json
import operator
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from absl import logging
import dataclasses
import flax
import gin
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow.io import gfile

from gfsa import jax_util
from gfsa import py_ast_graphs
from gfsa.datasets import data_loading
from gfsa.datasets import graph_bundle
from gfsa.model import edge_supervision_models
from gfsa.model import model_util
from gfsa.model import side_outputs
from gfsa.training import train_util


@gin.configurable
def preprocess_targets(targets,
                       transitive=False,
                       reverse=False,
                       reflexive=False):
  """Preprocess targets.

  Used for basic sanity testing of models.

  Args:
    targets: Initial targets <bool[num_nodes, num_nodes]>
    transitive: Whether to compute the transitive closure of the target edge
      type (i.e. if initial targets represent parents, this would make it
      represent all ancestors).
    reverse: Whether to flip the direciton of the targets at the end.
    reflexive: Whether to add self-loops.

  Returns:
    Preprocessed targets <bool[num_nodes, num_nodes]>. If all arguments are
    at default values, this will be the same as `targets`.
  """
  if transitive:

    def _fixpt_cond(stuff):
      _, done = stuff
      return jnp.logical_not(done)

    def _fixpt(stuff):
      cur_targets, _ = stuff
      new_targets = jnp.logical_or(cur_targets, cur_targets @ cur_targets)
      return new_targets, jnp.all(new_targets == cur_targets)

    targets, _ = jax.lax.while_loop(_fixpt_cond, _fixpt, (targets, False))

  if reflexive:
    targets = jnp.logical_or(targets, jnp.eye(targets.shape[0], dtype=bool))

  if reverse:
    targets = targets.T

  return targets


def extract_outputs_and_targets(
    model,
    padded_example_and_rng,
    target_edge_index,
    num_edge_types,
):
  """Extract model outputs and targets for an example.

  Args:
    model: Model to run on the example.
    padded_example_and_rng: Example to extract targets from, with RNG.
    target_edge_index: Index of the target edge type.
    num_edge_types: How many edge types there are.

  Returns:
    Tuple (output_logits, targets, valid_mask, num_nodes, captured)
  """
  padded_example, rng = padded_example_and_rng
  # Run the model.
  with side_outputs.collect_side_outputs() as captured:
    with flax.nn.stochastic(rng):
      output_logits = model(padded_example)
  # Extract targets.
  targets = padded_example.edges.apply_add(
      in_array=(
          jnp.arange(num_edge_types) == target_edge_index).astype("int32"),
      out_array=jnp.zeros(output_logits.shape, dtype="int32")).astype("bool")
  targets = preprocess_targets(targets)
  # Compute valid mask for outputs and targets.
  max_num_nodes = output_logits.shape[0]
  num_nodes = padded_example.graph_metadata.num_nodes
  valid_nodes = jnp.arange(max_num_nodes) < num_nodes
  valid_nodes_float = valid_nodes.astype("float32")
  valid_mask = jnp.einsum("i,j->ij", valid_nodes_float, valid_nodes_float)
  return output_logits, targets, valid_mask, num_nodes, captured


@gin.configurable
def loss_fn(
    output_logits,
    targets,
    valid_mask,
    num_nodes,
    captured,
    negative_example_weight = 1,
    focal_loss_gamma = 0.0,
):
  """Compute loss and single-batch metrics for some outputs.

  Args:
    output_logits: Binary logits produced by the model.
    targets: Model targets.
    valid_mask: Mask determining which outputs are valid.
    num_nodes: How many nodes there are in each example.
    captured: Ignored
    negative_example_weight: Weight to assign to a negative example when
      computing the loss. Positive examples always get weight 1.
    focal_loss_gamma: Focusing parameter for the focal loss, as described in Lin
      et al. (2018). If zero, uses standard cross-entropy loss.

  Returns:
    Tuple (loss, metrics_dict).
  """
  del captured
  num_targets = jnp.count_nonzero(targets)
  # Compute cross entropy.
  unmasked_nll = model_util.binary_logit_cross_entropy(output_logits, targets)
  if focal_loss_gamma:
    # (1-p_correct)**gamma = (-(p-1))**gamma = (-expm1(log(p)))**gamma
    focus_term = jnp.power(-jnp.expm1(-unmasked_nll), focal_loss_gamma)
    unmasked_nll = unmasked_nll * focus_term
  # Mask the results so that they only count nodes that exist.
  masked_nll = unmasked_nll * valid_mask
  # Primary loss: Sum of nll over all nodes. We use sum because most of the
  # edges are easy negatives.
  positive_nll = jnp.sum(
      jnp.where(targets, masked_nll, jnp.zeros_like(masked_nll)))
  negative_nll = jnp.sum(
      jnp.where(targets, jnp.zeros_like(masked_nll), masked_nll))
  reweighted_nll = positive_nll + negative_example_weight * negative_nll
  binary_nll = jnp.sum(reweighted_nll)
  # Compute additional metrics to track learning progress.
  # Average NLL of target edges:
  avg_nll_per_target = positive_nll / num_targets
  # Average NLL of non-target edges:
  num_non_targets = num_nodes**2 - num_targets
  avg_nll_per_non_target = negative_nll / num_non_targets
  # Max error for any edge prediction:
  worst_nll = jnp.max(masked_nll)

  loss = binary_nll

  # Ratio of positive to negative targets. If this is equal to
  # negative_example_weight, the positive and negative examples will have the
  # same total weight.
  positive_per_negative = num_targets / num_non_targets
  # Precision and recall at 0.1 threshold
  thresholded_preds = output_logits > jax.scipy.special.logit(0.1)
  count_target_pred = jnp.count_nonzero(thresholded_preds & targets)
  count_pred = jnp.count_nonzero(thresholded_preds & valid_mask.astype(bool))
  precision = count_target_pred / count_pred
  recall = count_target_pred / num_targets
  return loss, {
      "avg_per_target":
          avg_nll_per_target,
      "avg_per_non_target":
          avg_nll_per_non_target,
      "worst":
          worst_nll,
      "positive_per_negative":
          positive_per_negative,
      "effective_p_model_given_target":
          jnp.exp(-avg_nll_per_target),
      "effective_p_model_given_nontarget":
          1 - jnp.exp(-avg_nll_per_non_target),
      "batch_clf_thresh_at_0.1/precision":
          precision,
      "batch_clf_thresh_at_0.1/recall":
          recall,
      "batch_clf_thresh_at_0.1/f1":
          2 * (precision * recall) / (precision + recall),
  }


@gin.configurable
def sample_loss_fn(
    model,
    padded_example_and_rng,
    target_edge_index,
    num_edge_types,
    baseline_weight = 0.001,
    num_rollouts = 1,
    leave_one_out_baseline = False,
):
  """Compute a single-sample version of the loss.

  Used for running sample-based baseline.

  Args:
    model: Model to run on the example.
    padded_example_and_rng: Example to extract targets from, with RNG.
    target_edge_index: Index of the target edge type.
    num_edge_types: How many edge types there are.
    baseline_weight: How much weight to give to learned baseline loss term.
    num_rollouts: How many rollouts to use.
    leave_one_out_baseline: Whether to use leave-one-out baseline instead of
      learned baseline.

  Returns:
    Tuple (output_logits, targets, valid_mask, loss, metrics_dict).
  """
  padded_example, rng = padded_example_and_rng

  @functools.partial(jax.vmap, out_axes=(0, None, None, None, 0, 0, None, None))
  def go(rng):
    (output_logits, targets, valid_mask, num_nodes,
     captured) = extract_outputs_and_targets(model, (padded_example, rng),
                                             target_edge_index, num_edge_types)

    valid_nodes = (jnp.arange(output_logits.shape[-1]) < num_nodes).astype(
        jnp.float32)
    log_prob, = [
        v for k, v in captured.items()
        if k.endswith("one_sample_log_prob_per_edge_per_node")
    ]
    log_prob = log_prob.squeeze(0) * valid_nodes
    learned_baseline, = [
        v for k, v in captured.items()
        if k.endswith("one_sample_reward_baseline")
    ]
    # Compute reward: +1 for doing the correct thing, 0 for doing anything else.
    output_probs = jax.nn.sigmoid(output_logits) * valid_mask
    num_targets_per_node = jnp.sum(targets.astype(jnp.int32), axis=-1)
    no_targets = (num_targets_per_node == 0)
    fail_prob = 1 - jnp.sum(output_probs, axis=-1)
    reward_per_node = (
        jnp.sum(output_probs * targets.astype(jnp.float32), axis=-1) +
        fail_prob * no_targets.astype(jnp.float32)) * valid_nodes

    return (output_logits, targets, valid_mask, num_nodes, reward_per_node,
            log_prob, learned_baseline, valid_nodes)

  (output_logits, targets, valid_mask, num_nodes, reward_per_node, log_prob,
   learned_baseline, valid_nodes) = go(jax.random.split(rng, num_rollouts))

  if leave_one_out_baseline:
    baseline = (jnp.sum(reward_per_node, axis=0, keepdims=True) -
                reward_per_node) / (
                    num_rollouts - 1)
  else:
    baseline = learned_baseline

  shifted_reward = (reward_per_node - baseline) * valid_nodes
  # REINFORCE: scale shifted reward by log probs
  reinforce_virtual_loss = jnp.sum(
      -log_prob * jax.lax.stop_gradient(shifted_reward)) / (
          num_nodes * num_rollouts)

  if not leave_one_out_baseline:
    # Penalty for baseline to bring it close to 0
    baseline_penalty = baseline_weight * jnp.sum(jnp.square(shifted_reward)) / (
        num_nodes * num_rollouts)
  else:
    baseline_penalty = jnp.zeros([])

  mean_logits = model_util.safe_logit(
      jnp.mean(jax.scipy.special.expit(output_logits), axis=0))

  virtual_loss = reinforce_virtual_loss + baseline_penalty
  return mean_logits, targets, valid_mask, virtual_loss, {
      "reward": jnp.sum(reward_per_node) / (num_nodes * num_rollouts),
      "shifted_reward": jnp.sum(shifted_reward) / (num_nodes * num_rollouts),
      "policy_log_prob": jnp.sum(log_prob) / (num_nodes * num_rollouts),
      "learned_baseline": learned_baseline,
      "baseline_penalty": baseline_penalty,
      "reinforce_term": reinforce_virtual_loss,
  }


def build_validation_fn(
    valid_iterator_factory,
    target_edge_index,
    num_edge_types,
    full_evaluation = False,
    use_sampling_model = False,
):
  """Computes validation performance of a model.

  Args:
    valid_iterator_factory: Function that builds an iterable over the validation
      set. Each call should reset the iterator to the start of the set.
    target_edge_index: Index of the target edge type.
    num_edge_types: How many edge types there are.
    full_evaluation: Whether we are doing a detailed evaluation pass (i.e. on
      the test set). If so, uses a wider set of candidate thresholds, and
      returns some additional information in the metrics dictionary, including
      detailed true/false positive/negative information.
    use_sampling_model: Whether to use sample-based version of the model.

  Returns:
    Validation function that takes a model and returns:
      objective_value: 1 - F1 score, at the best threshold out of a set of
        options.
      metrics: Dictionary of metrics, including the same metrics as in the
        training set along with precision/recall at an optimal F1 threshold.
  """
  # Combination of log-space and linear-space, so that we can capture both
  # very small thresholds and also very large thresholds.
  if full_evaluation:
    candidate_thresholds = np.concatenate(
        [10.**np.linspace(-8, -1, 64, endpoint=False),
         np.linspace(0.1, 1, 64)])
  else:
    candidate_thresholds = np.concatenate(
        [10.**np.arange(-8, -1),
         np.arange(0.05, 1, .05)])

  def example_helper(model, padded_example_and_rng):
    """Run the model on one example."""
    # Compute the same estimates as in training, for ease of comparison.
    # Instead of aggregating with nanmean per-batch, we aggregate over the full
    # validation set.
    if use_sampling_model:
      (output_logits, targets, valid_mask, loss,
       batch_metrics) = sample_loss_fn(model, padded_example_and_rng,
                                       target_edge_index, num_edge_types)
    else:
      (output_logits, targets, valid_mask, num_nodes,
       captured) = extract_outputs_and_targets(model, padded_example_and_rng,
                                               target_edge_index,
                                               num_edge_types)
      loss, batch_metrics = loss_fn(output_logits, targets, valid_mask,
                                    num_nodes, captured)
    batch_metrics_non_nan = jax.tree_map(jnp.nan_to_num, batch_metrics)
    batch_metrics_non_nan_counts = jax.tree_map(
        lambda x: jnp.count_nonzero(~jnp.isnan(x)), batch_metrics)
    # Compute additional metrics by counting how many predictions cross our
    # thresholds.
    output_probs = jax.scipy.special.expit(output_logits)
    preds = (output_probs[None, :, :] > candidate_thresholds[:, None, None])
    # Count true/false target/pred pairs.
    valid = valid_mask.astype(bool)
    count_t_target_t_pred = jnp.count_nonzero(
        valid & targets & preds, axis=(1, 2))
    count_t_target_f_pred = jnp.count_nonzero(
        valid & targets & (~preds), axis=(1, 2))
    count_f_target_t_pred = jnp.count_nonzero(
        valid & (~targets) & preds, axis=(1, 2))
    count_f_target_f_pred = jnp.count_nonzero(
        valid & (~targets) & (~preds), axis=(1, 2))
    counts = (count_t_target_t_pred, count_t_target_f_pred,
              count_f_target_t_pred, count_f_target_f_pred)
    return loss, 1, batch_metrics_non_nan, batch_metrics_non_nan_counts, counts

  @functools.partial(jax.pmap, axis_name="devices")
  def batch_helper(model, batched_examples, batch_mask):
    # Map our metrics over the examples.
    values = jax.vmap(example_helper, (None, 0))(model, batched_examples)
    values = jax.tree_map(
        lambda x: jax.vmap(jnp.where)(batch_mask, x, jnp.zeros_like(x)), values)
    # Sum everything together.
    values = jax.lax.psum(
        jax.tree_map(lambda x: jnp.sum(x, axis=0), values), "devices")
    return values

  def validation_fn(model):
    """Iterates over the full validation set and computes metrics."""
    valid_iterator = valid_iterator_factory()
    accumulator = None
    for batch in valid_iterator:
      new_values = flax.jax_utils.unreplicate(
          batch_helper(model, batch.example, batch.mask))
      if accumulator is None:
        accumulator = new_values
      else:
        accumulator = jax.tree_multimap(operator.add, accumulator, new_values)

    (
        loss_sum,
        example_count,
        batch_metrics_non_nan,
        batch_metrics_non_nan_counts,
        (
            count_t_target_t_pred,
            count_t_target_f_pred,
            count_f_target_t_pred,
            count_f_target_f_pred,
        ),
    ) = accumulator

    metrics = {}
    metrics["loss"] = float(loss_sum / example_count)
    for k in batch_metrics_non_nan:
      metrics[k] = float(batch_metrics_non_nan[k] /
                         batch_metrics_non_nan_counts[k])

    precision_at_thresholds = jnp.nan_to_num(
        count_t_target_t_pred / (count_t_target_t_pred + count_f_target_t_pred))
    recall_at_thresholds = jnp.nan_to_num(
        count_t_target_t_pred / (count_t_target_t_pred + count_t_target_f_pred))
    f1_at_thresholds = jnp.nan_to_num(
        2 * (precision_at_thresholds * recall_at_thresholds) /
        (precision_at_thresholds + recall_at_thresholds))

    best_threshold_index = jnp.argmax(f1_at_thresholds)
    logging.info("F1 score across thresholds: %s",
                 jnp.stack([candidate_thresholds, f1_at_thresholds]))
    threshold = candidate_thresholds[best_threshold_index]
    precision = precision_at_thresholds[best_threshold_index]
    recall = recall_at_thresholds[best_threshold_index]
    f1 = f1_at_thresholds[best_threshold_index]

    metrics["best_threshold"] = float(threshold)
    metrics["flipped_precision"] = float(1 - precision)
    metrics["flipped_recall"] = float(1 - recall)
    metrics["flipped_f1"] = float(1 - f1)

    if full_evaluation:
      # Add (possibly non-scalar) detailed metrics
      metrics["example_count"] = example_count
      metrics["threshold_curves"] = {
          "thresholds": candidate_thresholds,
          "count_t_target_t_pred": count_t_target_t_pred,
          "count_t_target_f_pred": count_t_target_f_pred,
          "count_f_target_t_pred": count_f_target_t_pred,
          "count_f_target_f_pred": count_f_target_f_pred,
          "precision_at_thresholds": precision_at_thresholds,
          "recall_at_thresholds": recall_at_thresholds,
          "f1_at_thresholds": f1_at_thresholds,
      }

    return metrics["flipped_f1"], metrics

  return validation_fn


def load_dataset_metadata(
    metadata_filename):
  """Helper function to load dataset metadata.

  Args:
    metadata_filename: Filename containing dataset metadata.

  Returns:
    Padding configuration and edge types for the dataset.
  """
  with gfile.GFile(metadata_filename, "r") as fp:
    metadata = json.load(fp)

  edge_types = metadata["edge_types"]
  padding_config = flax.serialization.from_state_dict(
      target=jax_util.synthesize_dataclass(graph_bundle.PaddingConfig),
      state=metadata["padding_config"])
  return padding_config, edge_types


def add_rng_to_examples(
    example_iter,
    base_rng):
  """Add an RNG to each example.

  Args:
    example_iter: Iterator over examples.
    base_rng: RNG to seed with.

  Yields:
    Examples that are tuples (orig_example, rng)
  """
  base_rng = jax.device_put(base_rng, jax.devices("cpu")[0])
  for i, item in enumerate(example_iter):
    rng = jax.random.fold_in(base_rng, i)
    yield dataclasses.replace(item, example=(item.example, rng))


@gin.configurable
def train(
    runner,
    dataset_paths = gin.REQUIRED,
    prefetch = 4,
    target_edge = gin.REQUIRED,
    batch_size_per_device = gin.REQUIRED,
    truncate_training_dataset_at = None,
    validation_example_skip = 0,
    validation_example_count = gin.REQUIRED,
    model_type = "automaton",
    evaluate_only = False,
    evaluation_model_path = None,
    evaluation_save_path = None,
    use_sampling_model = False,
):
  """Launch a training job for edge supervision.

  The dataset directories should be configured with gin.

  Args:
    runner: Helper object that runs the experiment.
    dataset_paths: Dictionary of dataset paths, with keys:
      - "metadata": Path to JSON file with dataset metadata.
      - "train_dataset": Path to training dataset files.
      - "eval_dataset": Path to validation/test dataset files.
    prefetch: Maximum number of examples to prefetch in a background thread.
      Note that we prefetch a maximum of 1 example for the validation set.
    target_edge: What edge to use as the training target.
    batch_size_per_device: Batch size for each device.
    truncate_training_dataset_at: Number of examples to truncate the training
      dataset to.
    validation_example_skip: Number of examples to skip when computing
      validation metrics.
    validation_example_count: How many examples to use when computing validation
      metrics.
    model_type: Either "automaton" or "baseline".
    evaluate_only: If True, doesn't run any training; instead evaluates a
      trained model on the validation/evaluation set. Make sure to change
      "eval_dataset" to the test dataset if using this to compute final metrics.
    evaluation_model_path: Path to the model checkpoint to evaluate.
    evaluation_save_path: Where to save the result JSON file.
    use_sampling_model: Whether to use sample-based version of the loss.

  Returns:
    Optimizer at the end of training (for interactive debugging).
  """

  logging.info("Hello from train_edge_supervision_lib!")
  num_devices = jax.local_device_count()
  logging.info("Found %d devices: %s", num_devices, jax.devices())
  logging.info("Setting up datasets...")
  with contextlib.ExitStack() as exit_stack:

    padding_config, edge_types = load_dataset_metadata(
        dataset_paths["metadata"])

    if evaluate_only:
      assert evaluation_model_path is not None
    else:
      unbatched_train_iterator = runner.build_sampling_iterator(
          dataset_paths["train_dataset"],
          example_type=graph_bundle.GraphBundle,
          truncate_at=truncate_training_dataset_at)
      unbatched_train_iterator = add_rng_to_examples(
          unbatched_train_iterator, jax.random.PRNGKey(int(time.time() * 1000)))
      train_iterator = data_loading.batch(unbatched_train_iterator,
                                          (num_devices, batch_size_per_device))

      if prefetch:
        train_iterator = exit_stack.enter_context(
            data_loading.ThreadedPrefetcher(train_iterator, prefetch))

    unbatched_valid_iterator_factory = runner.build_one_pass_iterator_factory(
        dataset_paths["eval_dataset"],
        example_type=graph_bundle.GraphBundle,
        truncate_at=validation_example_count,
        skip_first=validation_example_skip)

    def valid_iterator_factory():
      it = unbatched_valid_iterator_factory()
      # Fix validation randomness to smooth out noise.
      # (note: for final evaluation we should compute true marginals and not
      # do any sampling)
      it = add_rng_to_examples(it, jax.random.PRNGKey(0))
      return data_loading.batch(
          it, (num_devices, batch_size_per_device),
          remainder_behavior=data_loading.BatchRemainderBehavior.PAD_ZERO)

    num_edge_types = len(edge_types)
    edge_types_to_indices = {name: i for i, name in enumerate(edge_types)}

    logging.info("Setting up model...")
    if model_type == "automaton":
      model_def = edge_supervision_models.automaton_model
    elif model_type == "baseline":
      model_def = edge_supervision_models.BaselineModel
    else:
      raise ValueError(f"Unknown model type '{model_type}'")

    # Bind statically-known information from the dataset.
    model_def = model_def.partial(
        graph_metadata=padding_config.static_max_metadata,
        edge_types_to_indices=edge_types_to_indices)

    # Initialize parameters randomly.
    @jax.jit
    def _init(rng):
      # Set up a dummy stochastic scope for random perturbations.
      with flax.nn.stochastic(jax.random.PRNGKey(0)):
        ex = graph_bundle.zeros_like_padded_example(padding_config)
        ex = jax.tree_map(jnp.array, ex)
        _, initial_params = model_def.init(rng, ex)
      return initial_params

    initial_params = _init(jax.random.PRNGKey(int(time.time() * 1000)))

    model = flax.nn.Model(model_def, initial_params)
    optimizer = flax.optim.Adam().create(model)

    validation_fn = build_validation_fn(
        valid_iterator_factory=valid_iterator_factory,
        target_edge_index=edge_types_to_indices[target_edge],
        num_edge_types=num_edge_types,
        full_evaluation=evaluate_only,
        use_sampling_model=use_sampling_model)

    if evaluate_only:
      optimizer, checkpoint_info = runner.load_from_checkpoint(
          optimizer, checkpoint_path=evaluation_model_path)
      model = train_util.device_broadcast(optimizer.target, num_devices)

      _, metrics = validation_fn(model)
      metrics["checkpoint_info"] = checkpoint_info
      metrics["model_path"] = evaluation_model_path
      metrics["dataset_metadata_path"] = dataset_paths["metadata"]
      metrics["dataset_path"] = dataset_paths["eval_dataset"]
      metrics["example_skip"] = validation_example_skip
      metrics["example_count"] = validation_example_count

      array_types = (np.ndarray, jnp.ndarray)
      metrics = jax.tree_map(
          lambda x: x.tolist() if isinstance(x, array_types) else x, metrics)

      gfile.makedirs(os.path.dirname(evaluation_save_path))
      with gfile.GFile(evaluation_save_path, "w") as fp:
        json.dump(metrics, fp, indent=2)

      logging.info("Computed evaluation metrics: %s", metrics)

    else:

      def compute_loss_for_model(model, padded_example_and_rng,
                                 static_metadata):
        assert static_metadata is None
        if use_sampling_model:
          (_, _, _, loss, batch_metrics) = sample_loss_fn(
              model,
              padded_example_and_rng,
              target_edge_index=edge_types_to_indices[target_edge],
              num_edge_types=num_edge_types)
          return loss, batch_metrics
        else:
          return loss_fn(*extract_outputs_and_targets(
              model,
              padded_example_and_rng,
              target_edge_index=edge_types_to_indices[target_edge],
              num_edge_types=num_edge_types))

      extra_artifacts = {
          "builder.pickle": py_ast_graphs.BUILDER,
      }

      return runner.training_loop(
          optimizer=optimizer,
          train_iterator=train_iterator,
          loss_fn=compute_loss_for_model,
          validation_fn=validation_fn,
          extra_artifacts=extra_artifacts)
