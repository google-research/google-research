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

"""Training pipeline for DP-GNN."""

import functools
import os
from typing import Callable, Dict, Optional, Tuple

from absl import logging
import chex
from clu import checkpoint
from clu import metric_writers
from clu import parameter_overview
from clu import periodic_actions
from flax import linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import jraph
import ml_collections
import numpy as np
import optax

from differentially_private_gnns import input_pipeline
from differentially_private_gnns import models
from differentially_private_gnns import normalizations
from differentially_private_gnns import optimizers
from differentially_private_gnns import privacy_accountants

_SUBGRAPH_PADDING_VALUE = -1


@jax.jit
def compute_logits(state,
                   graph):
  """Computes unnormalized logits."""
  return state.apply_fn(state.params, graph).nodes


@jax.jit
def compute_loss(logits,
                 labels):
  """Computes the mean softmax cross-entropy loss."""
  assert labels.shape == logits.shape, f'Got incompatible shapes: logits as {logits.shape}, labels as {labels.shape}'

  loss = optax.softmax_cross_entropy(logits=logits, labels=labels)
  loss = jnp.mean(loss)
  return loss


def get_subgraphs(graph,
                  pad_to):
  """Creates an array of padded subgraphs."""
  num_nodes = jax.tree.leaves(graph.nodes)[0].shape[0]
  outgoing_edges = {u: [] for u in range(num_nodes)}
  for sender, receiver in zip(graph.senders, graph.receivers):
    if sender != receiver:
      outgoing_edges[sender].append(receiver)

  subgraphs = np.zeros((num_nodes, pad_to), dtype=np.int32)
  for node in outgoing_edges:
    subgraph_indices = [node] + outgoing_edges[node]
    subgraph_indices = np.asarray(subgraph_indices)
    subgraph_indices = subgraph_indices[np.sort(
        np.unique(subgraph_indices, return_index=True)[1])]
    subgraph_indices = subgraph_indices[:pad_to]
    subgraphs[node] = np.pad(
        subgraph_indices, (0, pad_to - len(subgraph_indices)),
        'constant',
        constant_values=_SUBGRAPH_PADDING_VALUE)
  return jnp.asarray(subgraphs)


@functools.partial(
    jax.jit, static_argnames=['add_reverse_edges', 'adjacency_normalization'])
def make_subgraph_from_indices(
    graph,
    subgraph_indices,
    add_reverse_edges,
    adjacency_normalization):
  """Constructs the subgraph corresponding to the given indices."""
  # Extract valid node indices.
  valid_mask = (subgraph_indices != _SUBGRAPH_PADDING_VALUE)

  # Node features.
  subgraph_nodes = graph.nodes[subgraph_indices]
  subgraph_nodes = jnp.where(valid_mask[:, jnp.newaxis], subgraph_nodes, 0.)

  # Add a dummy padding node.
  padding_node = len(subgraph_indices)
  subgraph_nodes = jnp.concatenate(
      (subgraph_nodes, jnp.zeros((1, subgraph_nodes.shape[1]))))
  subgraph_indices = jnp.where(valid_mask, subgraph_indices, padding_node)

  # Remap indices to within the subgraph.
  subgraph_senders = jnp.zeros_like(subgraph_indices, dtype=jnp.int32)
  subgraph_receivers = jnp.arange(len(subgraph_indices))

  # Handle padding.
  subgraph_senders = jnp.where(valid_mask, subgraph_senders, padding_node)
  subgraph_receivers = jnp.where(valid_mask, subgraph_receivers, padding_node)

  # Add reverse edges, ignoring self-loops.
  # This is generally not necessary, because we are only interested in the
  # predictions at the root node.
  if add_reverse_edges:
    subgraph_senders, subgraph_receivers = (
        jnp.concatenate((subgraph_senders, subgraph_receivers[1:])),
        jnp.concatenate((subgraph_receivers, subgraph_senders[1:])))

  # Build subgraph.
  subgraph_edges_unnormalized = jnp.ones_like(subgraph_senders)
  subgraph = graph._replace(
      nodes=subgraph_nodes,
      edges=subgraph_edges_unnormalized,
      senders=subgraph_senders,
      receivers=subgraph_receivers,
      n_node=jnp.expand_dims(subgraph_nodes.shape[0], axis=0),
      n_edge=jnp.expand_dims(subgraph_senders.shape[0], axis=0))

  # Normalize edge weights.
  return normalizations.normalize_edges_with_mask(subgraph, valid_mask,
                                                  adjacency_normalization)


@jax.jit
def compute_updates(state, graph,
                    labels,
                    node_indices):
  """Computes gradients for a single batch."""
  def loss_fn(params, graph,
              labels, node_indices):
    curr_state = state.replace(params=params)
    logits = compute_logits(curr_state, graph)
    logits = logits[node_indices]
    labels = labels[node_indices]
    return compute_loss(logits, labels)
  return jax.grad(loss_fn)(state.params, graph, labels, node_indices)


@jax.jit
def reshape_before_pmap(arr):
  """Reshapes an array to have leading dimensions == jax.local_device_count()."""
  return arr.reshape((jax.local_device_count(),
                      arr.shape[0] // jax.local_device_count(), *arr.shape[1:]))


@jax.jit
def reshape_after_pmap(arr):
  """Undoes reshape_before_pmap()."""
  return arr.reshape((arr.shape[0] * arr.shape[1], *arr.shape[2:]))


@functools.partial(jax.jit, static_argnames='adjacency_normalization')
def compute_updates_for_dp(state,
                           graph, labels,
                           subgraphs,
                           node_indices,
                           adjacency_normalization):
  """Computes gradients for a single batch for differentially private training."""

  def subgraph_loss(params, graph,
                    node_labels,
                    subgraph_indices):
    """Compute loss over this subgraph at the root node."""
    subgraph = make_subgraph_from_indices(
        graph,
        subgraph_indices,
        add_reverse_edges=False,
        adjacency_normalization=adjacency_normalization)
    subgraph_preds = state.apply_fn(params, subgraph).nodes
    node_preds = subgraph_preds[0, :]
    return compute_loss(node_preds, node_labels)

  # Reshape leading axes for multiple devices.
  node_labels = reshape_before_pmap(labels[node_indices])
  subgraph_indices = reshape_before_pmap(subgraphs[node_indices])

  # Compute per-example gradients.
  per_example_gradient_fn = jax.vmap(
      jax.grad(subgraph_loss), in_axes=(None, None, 0, 0))
  per_example_gradient_fn = jax.pmap(
      per_example_gradient_fn,
      axis_name='devices',
      in_axes=(None, None, 0, 0),
      devices=jax.local_devices())
  grads = per_example_gradient_fn(state.params, graph, node_labels,
                                  subgraph_indices)

  # Undo reshape.
  grads = jax.tree.map(reshape_after_pmap, grads)

  # Normalize gradients by batch size.
  return jax.tree.map(lambda grad: grad / grad.shape[0], grads)


@jax.jit
def update_model(state,
                 grads):
  """Applies updates to the parameters."""
  return state.apply_gradients(grads=grads)


@jax.jit
def evaluate_predictions(logits, labels,
                         mask):
  """Evaluates the model on the given dataset."""
  loss = optax.softmax_cross_entropy(logits, labels)
  loss = jnp.where(mask, loss, 0.)
  loss = jnp.sum(loss) / jnp.sum(mask)

  logits_match_labels = (jnp.argmax(logits, -1) == jnp.argmax(labels, -1))
  logits_match_labels = jnp.where(mask, logits_match_labels, 0.)
  accuracy = jnp.sum(logits_match_labels) / jnp.sum(mask)
  return loss, accuracy


@functools.partial(
    jax.jit, static_argnames=['apply_fn', 'adjacency_normalization'])
def estimate_clipping_thresholds(
    apply_fn,
    params,
    l2_norm_clip_percentile, graph,
    labels, subgraphs, estimation_indices,
    adjacency_normalization):
  """Estimates gradient clipping thresholds."""
  dummy_state = train_state.TrainState.create(
      apply_fn=apply_fn, params=params, tx=optax.identity())
  grads = compute_updates_for_dp(dummy_state, graph, labels, subgraphs,
                                 estimation_indices, adjacency_normalization)
  grad_norms = jax.tree.map(jax.vmap(jnp.linalg.norm), grads)
  get_percentile = lambda norms: jnp.percentile(norms, l2_norm_clip_percentile)
  l2_norms_threshold = jax.tree.map(get_percentile, grad_norms)
  return l2_norms_threshold


def create_model(config, graph,
                 rng):
  """Creates the model and initial parameters."""
  if config.model == 'mlp':
    model = models.GraphMultiLayerPerceptron(
        dimensions=([config.latent_size] * config.num_layers +
                    [config.num_classes]),
        activation=getattr(nn, config.activation_fn))
  elif config.model == 'gcn':
    model = models.GraphConvolutionalNetwork(
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        num_message_passing_steps=config.num_message_passing_steps,
        latent_size=config.latent_size,
        num_classes=config.num_classes,
        activation=getattr(nn, config.activation_fn))
  else:
    raise ValueError(f'Unsupported model: {config.model}.')
  params = model.init(rng, graph)
  return model, params


def compute_max_terms_per_node(config):
  """Returns the maximum number of gradient terms affected by a node.

  Args:
    config: The configuration dictionary.
      config.num_message_passing_steps must be one of 0, 1, or 2.
  """
  if config.model == 'mlp':
    return 1

  num_message_passing_steps = config.num_message_passing_steps
  max_node_degree = config.max_degree

  if num_message_passing_steps == 1:
    return max_node_degree + 1

  if num_message_passing_steps == 2:
    return max_node_degree ** 2 + max_node_degree + 1

  # We only support MLP and upto 2-layer GNNs.
  raise ValueError('Not supported for num_message_passing_steps > 2.')


def compute_base_sensitivity(config):
  """Returns the base sensitivity which is multiplied to the clipping threshold.

  Args:
    config: The configuration dictionary.
      config.num_message_passing_steps must be one of 0, 1, or 2.
  """
  if config.model == 'mlp':
    return 1.

  num_message_passing_steps = config.num_message_passing_steps
  max_node_degree = config.max_degree

  if num_message_passing_steps == 1:
    return float(2 * (max_node_degree + 1))

  if num_message_passing_steps == 2:
    return float(2 * (max_node_degree ** 2 + max_node_degree + 1))

  # We only support MLP and upto 2-layer GNNs.
  raise ValueError('Not supported for num_message_passing_steps > 2.')


def create_optimizer(
    apply_fn,
    params,
    config,
    graph,
    labels,
    subgraphs,
    estimation_indices,
    rng,
):
  """Creates the optimizer."""
  if config.differentially_private_training:
    l2_norms_threshold = estimate_clipping_thresholds(
        apply_fn, params, config.l2_norm_clip_percentile, graph, labels,
        subgraphs, estimation_indices, config.adjacency_normalization)
    base_sensitivity = compute_base_sensitivity(config)
    privacy_params = {
        'l2_norms_threshold': l2_norms_threshold,
        'init_rng': rng,
        'base_sensitivity': base_sensitivity,
        'noise_multiplier': config.training_noise_multiplier,
    }

  if config.optimizer == 'sgd':
    opt_params = {
        'learning_rate': config.learning_rate,
        'momentum': config.momentum,
        'nesterov': config.nesterov,
    }
    if config.differentially_private_training:
      return optimizers.dpsgd(**opt_params, **privacy_params)
    return optax.sgd(**opt_params)

  if config.optimizer == 'adam':
    opt_params = {
        'learning_rate': config.learning_rate,
    }
    if config.differentially_private_training:
      return optimizers.dpadam(**opt_params, **privacy_params)
    return optax.adam(**opt_params)

  raise ValueError(f'Unsupported optimizer: {config.optimizer}')


def create_train_state(
    rng, config,
    graph,
    labels,
    subgraphs,
    estimation_indices,
):
  """Creates initial `TrainState`."""
  model_rng, opt_rng = jax.random.split(rng)
  model, params = create_model(config, graph, model_rng)
  apply_fn = jax.jit(model.apply)
  tx = create_optimizer(apply_fn, params, config, graph, labels, subgraphs,
                        estimation_indices, opt_rng)
  return train_state.TrainState.create(
      apply_fn=apply_fn, params=params, tx=tx)


def get_max_training_epsilon(
    config):
  """Returns the privacy budget for DP training."""
  if not config.differentially_private_training:
    return None
  return config.max_training_epsilon


def compute_metrics(logits, labels,
                    masks):
  train_loss, train_accuracy = evaluate_predictions(logits, labels,
                                                    masks['train'])
  val_loss, val_accuracy = evaluate_predictions(logits, labels,
                                                masks['validation'])
  test_loss, test_accuracy = evaluate_predictions(logits, labels,
                                                  masks['test'])
  return {
      'train_loss': train_loss,
      'train_accuracy': train_accuracy,
      'val_loss': val_loss,
      'val_accuracy': val_accuracy,
      'test_loss': test_loss,
      'test_accuracy': test_accuracy,
    }


def log_metrics(step, metrics,
                summary_writer,
                postfix = ''):
  """Logs all metrics."""
  # Formatting for accuracy.
  for metric in metrics:
    if 'accuracy' in metric:
      metrics[metric] *= 100

  # Add postfix to all metric names.
  metrics = {
      metric + postfix: metric_val for metric, metric_val in metrics.items()
  }

  for metric, metric_val in metrics.items():
    logging.info('num_steps: % 3d, %s: %.4f', step, metric, metric_val)

  summary_writer.write_scalars(step, metrics)
  summary_writer.flush()


def get_estimation_indices(
    train_indices,
    config):
  """Returns node indices for estimating clipping thresholds."""
  if config.differentially_private_training:
    return train_indices[:config.num_estimation_samples]
  return None


def train_and_evaluate(config,
                       workdir):
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.

  Returns:
    The train state (which includes the `.params`).
  """
  # Seed for reproducibility.
  rng = jax.random.PRNGKey(config.rng_seed)

  # Set up logging.
  summary_writer = metric_writers.create_default_writer(workdir)
  summary_writer.write_hparams(dict(config))

  # Get datasets.
  rng, dataset_rng = jax.random.split(rng)
  dataset = input_pipeline.get_dataset(config, dataset_rng)
  graph, labels, masks = jax.tree.map(jnp.asarray, dataset)
  labels = jax.nn.one_hot(labels, config.num_classes)
  train_mask = masks['train']
  train_indices = jnp.where(train_mask)[0]
  train_labels = labels[train_indices]
  num_training_nodes = len(train_indices)

  # Get subgraphs.
  if config.differentially_private_training:
    graph = jax.tree.map(np.asarray, graph)
    subgraphs = get_subgraphs(
        graph, pad_to=config.pad_subgraphs_to)
    graph = jax.tree.map(jnp.asarray, graph)

    # We only need the subgraphs for training nodes.
    train_subgraphs = subgraphs[train_indices]
    del subgraphs
  else:
    train_subgraphs = None

  # Initialize privacy accountant.
  training_privacy_accountant = privacy_accountants.get_training_privacy_accountant(
      config, num_training_nodes, compute_max_terms_per_node(config))

  # Construct and initialize model.
  rng, init_rng = jax.random.split(rng)
  estimation_indices = get_estimation_indices(train_indices, config)
  state = create_train_state(init_rng, config, graph, train_labels,
                             train_subgraphs, estimation_indices)

  # Set up checkpointing of the model.
  checkpoint_dir = os.path.join(workdir, 'checkpoints')
  ckpt = checkpoint.Checkpoint(checkpoint_dir, max_to_keep=2)
  state = ckpt.restore_or_initialize(state)
  initial_step = int(state.step) + 1

  # Log overview of parameters.
  parameter_overview.log_parameter_overview(state.params)

  # Log metrics after initialization.
  logits = compute_logits(state, graph)
  metrics_after_init = compute_metrics(logits, labels, masks)
  metrics_after_init['epsilon'] = 0
  log_metrics(
      0, metrics_after_init, summary_writer, postfix='_after_init')

  # Train model.
  rng, train_rng = jax.random.split(rng)
  max_training_epsilon = get_max_training_epsilon(config)

  # Hooks called periodically during training.
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=config.num_training_steps, writer=summary_writer)
  profiler = periodic_actions.Profile(num_profile_steps=5, logdir=workdir)
  hooks = [report_progress, profiler]

  for step in range(initial_step, config.num_training_steps):

    # Perform one step of training.
    with jax.profiler.StepTraceAnnotation('train', step_num=step):
      # Sample batch.
      step_rng = jax.random.fold_in(train_rng, step)
      indices = jax.random.choice(step_rng, num_training_nodes,
                                  (config.batch_size,))

      # Compute gradients.
      if config.differentially_private_training:
        grads = compute_updates_for_dp(state, graph, train_labels,
                                       train_subgraphs, indices,
                                       config.adjacency_normalization)
      else:
        grads = compute_updates(state, graph, train_labels, indices)

      # Update parameters.
      state = update_model(state, grads)

    # Quick indication that training is happening.
    logging.log_first_n(logging.INFO, 'Finished training step %d.', 10, step)
    for hook in hooks:
      hook(step)

    # Evaluate, if required.
    is_last_step = (step == config.num_training_steps - 1)
    if step % config.evaluate_every_steps == 0 or is_last_step:
      with report_progress.timed('eval'):
        # Check if privacy budget exhausted.
        training_epsilon = training_privacy_accountant(step + 1)
        if max_training_epsilon is not None and training_epsilon >= max_training_epsilon:
          break

        # Compute metrics.
        logits = compute_logits(state, graph)
        metrics_during_training = compute_metrics(logits, labels, masks)
        metrics_during_training['epsilon'] = training_epsilon
        log_metrics(
            step,
            metrics_during_training,
            summary_writer)

    # Checkpoint, if required.
    if step % config.checkpoint_every_steps == 0 or is_last_step:
      with report_progress.timed('checkpoint'):
        ckpt.save(state)

  return state
