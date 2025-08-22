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

"""Model training utility functions."""

import collections
import os
import time
from typing import Any, NamedTuple, Optional, Tuple

from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import objax
import optax
import pandas as pd
import sklearn.metrics
import tensorflow as tf
import tensorflow_datasets as tfds

from logit_dp.logit_dp import encoders
from logit_dp.logit_dp import gradient_utils
from logit_dp.logit_dp import objax_utils


class OptimizationParams(NamedTuple):
  noise_multiplier: float
  l2_norm_clip: float
  temperature: float
  optax_params: collections.defaultdict[str, float]
  gradient_accumulation_steps: int = 1
  sequential_computation_steps: int = 1


class ModelConfig(NamedTuple):
  input_shape: Tuple[int, Ellipsis]
  seed: int
  opt_params: OptimizationParams


class TrainingState(NamedTuple):
  """Container for the training state."""

  # This should be hk.Params but typing is complaining across functions
  # TODO(mribero): write a specific type that works across functions.
  params: Any
  opt_state: optax.OptState
  rng: jax.Array
  acc_step: int
  step: int
  opt_params: OptimizationParams
  current_time: float
  start_time: float


def setup_summary_writing(output_dir):
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  with open(os.path.join(output_dir, 'train_loss.csv'), 'a') as f:
    f.write('step , train_loss \n')


def compute_neighbors_and_true_labels(
    embeddings,
    y_test,
    lookup_class_to_test_idxs,
    num_classes,
    num_samples_evaluate,
    nearest_neighbors_per_example,
):
  """Computes nearest neighbors and true labels for each class."""
  num_entries = (
      num_classes * num_samples_evaluate * nearest_neighbors_per_example
  )
  neighbors = np.zeros(num_entries)
  labels = np.zeros(num_entries)
  embeddings = jnp.reshape(embeddings, (embeddings.shape[0], -1))
  gram_matrix = jnp.einsum('ae,be->ab', embeddings, embeddings)
  nearest_neighbors = jnp.argsort(gram_matrix.T)[
      :, -(nearest_neighbors_per_example + 1) :
  ]
  for class_idx in range(num_classes):
    # Shuffle to avoid biases.
    example_idxs = np.random.choice(
        lookup_class_to_test_idxs[class_idx][:-1],
        num_samples_evaluate,
        replace=False,
    )
    for sample_idx, y_test_idx in enumerate(example_idxs):
      for nn_idx, test_idx in enumerate(nearest_neighbors[y_test_idx][:-1]):
        # 3-level hierarchical indexing.
        entry_idx = (
            class_idx * num_samples_evaluate * nearest_neighbors_per_example
            + sample_idx * nearest_neighbors_per_example
            + nn_idx
        )
        neighbors[entry_idx] = y_test[test_idx]
        labels[entry_idx] = class_idx
  return neighbors, labels


def contrastive_loss_fn(params, rng, batch_inputs, forward, temperature):
  """Computes the contrastive loss."""
  batch_size = batch_inputs.shape[0]
  x, y = batch_inputs[:, 0], batch_inputs[:, 1]
  v = jax.vmap(forward.apply, in_axes=(None, None, 0))(params, rng, x)
  u = jax.vmap(forward.apply, in_axes=(None, None, 0))(params, rng, y)

  v = jnp.squeeze(v)
  u = jnp.squeeze(u)

  similarities = jnp.matmul(v, u.T) / temperature
  labels = jax.nn.one_hot(jnp.arange(batch_size), similarities.shape[-1])

  return -jnp.sum(labels * jax.nn.log_softmax(similarities)) / labels.shape[0]


def get_initial_model_state_and_optimizer(
    model_config,
    forward_fn,
):
  """Obtains an initial model state from some configuration."""
  optimizer = optax.adamw(**model_config.opt_params.optax_params)
  if model_config.opt_params.gradient_accumulation_steps > 1:
    optimizer = optax.MultiSteps(
        optimizer,
        every_k_schedule=model_config.opt_params.gradient_accumulation_steps,
    )
  hk_key = hk.PRNGSequence(model_config.seed)
  initial_params = forward_fn.init(
      next(hk_key), jnp.zeros(model_config.input_shape)
  )
  initial_opt_state = optimizer.init(initial_params)
  jax_key = jax.random.PRNGKey(model_config.seed)
  initial_step = 0

  t0 = time.time()
  initial_state = TrainingState(
      params=initial_params,
      opt_state=initial_opt_state,
      rng=jax_key,
      acc_step=initial_step,
      step=initial_step,
      opt_params=model_config.opt_params,
      current_time=t0,
      start_time=t0,
  )

  return initial_state, optimizer


def compute_next_state(
    current_state,
    forward_fn,
    optimizer,
    batch_inputs,
):
  """Gets the next model state and metrics using the optimizer."""

  rng, new_rng = jax.random.split(current_state.rng)
  dp_grads = gradient_utils.compute_dp_gradients(
      key=new_rng,
      params=current_state.params,
      input_pairs=batch_inputs,
      forward_fn=forward_fn,
      l2_norm_clip=current_state.opt_params.l2_norm_clip,
      noise_multiplier=current_state.opt_params.noise_multiplier,
      temperature=current_state.opt_params.temperature,
      sequential_computation_steps=current_state.opt_params.sequential_computation_steps,
  )

  updates, new_opt_state = optimizer.update(
      dp_grads, current_state.opt_state, current_state.params
  )

  new_params = optax.apply_updates(current_state.params, updates)

  rng, new_rng = jax.random.split(rng)

  grad_acc_steps = current_state.opt_params.gradient_accumulation_steps
  next_acc_step = (current_state.acc_step + 1) % grad_acc_steps
  next_step = (
      current_state.step + 1 if next_acc_step == 0 else current_state.step
  )

  new_state = TrainingState(
      params=new_params,
      opt_state=new_opt_state,
      rng=rng,
      acc_step=next_acc_step,
      step=next_step,
      opt_params=current_state.opt_params,
      current_time=time.time(),
      start_time=current_state.start_time,
  )

  return new_state


def compute_eval_metrics(y_pred, y_true, num_classes):
  """Computes metrics evaluated from test data."""
  eval_metrics = {}
  confusion_matrix = sklearn.metrics.confusion_matrix(
      y_true, y_pred, labels=list(range(num_classes))
  )
  precision, recall, fbeta_score, _ = (
      sklearn.metrics.precision_recall_fscore_support(
          y_true, y_pred, labels=list(range(num_classes))
      )
  )
  eval_metrics['confusion_matrix'] = confusion_matrix
  eval_metrics['test_max_precision'] = np.max(precision)
  eval_metrics['test_max_recall'] = np.max(recall)
  eval_metrics['test_max_fbeta_score'] = np.max(fbeta_score)
  eval_metrics['test_accuracy'] = np.sum(np.diag(confusion_matrix)) / np.sum(
      confusion_matrix
  )
  return eval_metrics


def compute_metrics(
    state,
    forward_fn,
    train_inputs,
    test_inputs,
    lookup_class_to_test_idxs,
    num_eval_confusion_matrix,
    nearest_neighbors_per_example,
):
  """Compute train and eval metrics.

  Args:
    state: Training state.
    forward_fn: Embedding model.
    train_inputs: batch of train data.
    test_inputs: Evaluation dataset.
    lookup_class_to_test_idxs: dictionary where keys are classes on the training
      and values are the list of indeces corresponding to that class.
    num_eval_confusion_matrix: number of examples to use per class to compute
      the confusion matrix.
    nearest_neighbors_per_example: number of nearest neighbors to use to compute
      confusion matrix.

  Returns:
    Dictionary with metrics.
  """
  # TODO(mribero): add more metrics like accuracy.
  step_metrics = {}
  num_classes = len(lookup_class_to_test_idxs)
  step_metrics['runtime'] = state.current_time - state.start_time
  if train_inputs is not None:
    step_metrics['train_loss'] = contrastive_loss_fn(
        params=state.params,
        rng=state.rng,
        batch_inputs=train_inputs,
        forward=forward_fn,
        temperature=state.opt_params.temperature,
    )
  if test_inputs is not None:
    dummy_rng = jax.random.PRNGKey(0)
    eval_fn = jax.jit(jax.vmap(forward_fn.apply, in_axes=(None, None, 0)))
    x_test, y_test = test_inputs
    embeddings = eval_fn(state.params, dummy_rng, x_test)
    neighbors, labels = compute_neighbors_and_true_labels(
        embeddings,
        y_test,
        lookup_class_to_test_idxs,
        num_classes,
        num_samples_evaluate=num_eval_confusion_matrix,
        nearest_neighbors_per_example=nearest_neighbors_per_example,
    )
    eval_metrics = compute_eval_metrics(neighbors, labels, num_classes)
    step_metrics.update(eval_metrics)
  return step_metrics


def fit_model(
    encoder,
    train_inputs,
    test_inputs,
    test_lookup_class_to_idx,
    num_epochs,
    output_dir,
    model_config,
    steps_per_eval,
    num_eval_confusion_matrix,
    nearest_neighbors_per_example,
):
  """Fits the model on a set of training data."""

  setup_summary_writing(output_dir)
  forward_fn = encoders.get_forward_fn_from_module(encoder)
  state, optimizer = get_initial_model_state_and_optimizer(
      model_config, forward_fn
  )
  metrics = collections.defaultdict(list)

  train_inputs = train_inputs.repeat(num_epochs)
  train_inputs = tfds.as_numpy(train_inputs)

  if output_dir:
    tf_summary_writer = tf.summary.create_file_writer(output_dir).as_default()
  else:
    tf_summary_writer = None
  for batch_inputs in train_inputs:
    first_input = (state.step == 0) and (state.acc_step == 0)
    state = compute_next_state(
        state,
        forward_fn,
        optimizer,
        batch_inputs,
    )
    # Only output metrics at the end of an accumulation step (if present).
    if first_input or state.acc_step == 0:
      if state.step % steps_per_eval == 0:
        batch_metrics = compute_metrics(
            state,
            forward_fn,
            batch_inputs,
            test_inputs,
            test_lookup_class_to_idx,
            num_eval_confusion_matrix=num_eval_confusion_matrix,
            nearest_neighbors_per_example=nearest_neighbors_per_example,
        )
        logging.info(
            'Mean prediction c.matrix: %s',
            np.mean(np.diag(batch_metrics['confusion_matrix'])),
        )
      else:
        batch_metrics = compute_metrics(
            state,
            forward_fn,
            batch_inputs,
            test_inputs=None,
            lookup_class_to_test_idxs=test_lookup_class_to_idx,
            num_eval_confusion_matrix=num_eval_confusion_matrix,
            nearest_neighbors_per_example=nearest_neighbors_per_example,
        )
      for k, v in batch_metrics.items():
        metrics[k].append(v)

      if output_dir:
        batch_size = batch_inputs.shape[0]
        grad_acc_steps = state.opt_params.gradient_accumulation_steps
        num_processed = state.step * batch_size * grad_acc_steps
        if tf_summary_writer:
          save_results(
              num_processed, batch_metrics, output_dir, tf_summary_writer
          )
  return state, metrics


def save_results(
    step,
    metrics,
    output_dir,
    tf_summary_writer = None,
):
  """Saves the training results to some directory."""
  for k, v in metrics.items():
    # CSV writer.
    if 'confusion_matrix' in k:
      file_path = os.path.join(output_dir, k + str(step) + '.csv')
      with open(file_path, 'w') as f:
        pd.DataFrame(v).to_csv(f)
    else:
      file_path = os.path.join(output_dir, f'{k}.csv')
      with open(file_path, 'a') as f:
        f.write(str(step) + ',' + str(v) + '\n')
    # Tensorboard writer.
    if tf_summary_writer and 'confusion_matrix' not in k:
      with tf_summary_writer:
        tf.summary.scalar(k, v, step=step)


def save_objax_model(model_vars, file_path):
  with open(file_path, 'wb') as f:
    objax.io.save_var_collection(f, model_vars)
    logging.info('Saved model to %s', file_path)


def get_objax_params_from_haiku_params(haiku_embedding_params):
  """Creates objax variables with values given by haiku_embedding_params."""
  embedding_net = objax_utils.ObjaxEmbeddingNet()
  objax_vars = embedding_net.vars()

  objax_to_haiku = {
      '(ObjaxEmbeddingNet).convs(Sequential)[0](Conv2D).b': (
          haiku_embedding_params['small_embedding_net/~/conv2_d']['b']
      ),
      '(ObjaxEmbeddingNet).convs(Sequential)[0](Conv2D).w': (
          haiku_embedding_params['small_embedding_net/~/conv2_d']['w']
      ),
      '(ObjaxEmbeddingNet).convs(Sequential)[2](Conv2D).b': (
          haiku_embedding_params['small_embedding_net/~/conv2_d_1']['b']
      ),
      '(ObjaxEmbeddingNet).convs(Sequential)[2](Conv2D).w': (
          haiku_embedding_params['small_embedding_net/~/conv2_d_1']['w']
      ),
      '(ObjaxEmbeddingNet).convs(Sequential)[4](Conv2D).b': (
          haiku_embedding_params['small_embedding_net/~/conv2_d_2']['b']
      ),
      '(ObjaxEmbeddingNet).convs(Sequential)[4](Conv2D).w': (
          haiku_embedding_params['small_embedding_net/~/conv2_d_2']['w']
      ),
      '(ObjaxEmbeddingNet).embeddings(Linear).b': haiku_embedding_params[
          'small_embedding_net/~/linear'
      ]['b'],
      '(ObjaxEmbeddingNet).embeddings(Linear).w': haiku_embedding_params[
          'small_embedding_net/~/linear'
      ]['w'],
  }
  objax_vars.assign([objax_to_haiku[k] for k in objax_vars.keys()])
  return objax_vars


def load_embedding_net_into_finetuning_net(file_path, model):
  vars_freeze = objax.VarCollection(
      (k, v) for k, v in model.vars().items() if 'classification_layer' not in k
  )
  renamer = lambda x: x.replace('(ObjaxEmbeddingNet)', '(ObjaxFinetuningNet)')
  if not os.path.exists(file_path):
    raise ValueError(
        'There is not a pretrained checkpoint for the current finetuning'
        ' experiment.'
    )
  with open(file_path, 'rb') as f:
    objax.io.load_var_collection(f, vars_freeze, renamer=renamer)
