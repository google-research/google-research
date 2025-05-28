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

"""Factory methods for creating model trainers.

Callables with the arguments

* `model`
* `optimizer_cls`
* `learning_rate`
* `epochs`
* `steps_per_eval`
* `output_dir`

Each callable returns a dictionary of metrics obtained during training.
"""

from collections import defaultdict
import os
import time

from absl import logging
import jax.numpy as jnp
import numpy as np
import objax
import tensorflow as tf
import tensorflow_datasets as tfds

from logit_dp.logit_dp import objax_utils
from logit_dp.logit_dp import training_utils


def make_objax_trainer(
    train_input_iterator,
    test_inputs,
    test_class_to_idx,
    get_gradients_generator,
    num_classes,
    temperature,
    num_eval_confusion_matrix,
    nearest_neighbors_per_example,
    final_checkpoint_file_name,
    extended_batch_axis=False,
):
  """Creates a generic Objax trainer."""

  def objax_trainer(
      model,
      optimizer_cls,
      learning_rate,
      epochs,
      steps_per_eval,
      output_dir,
  ):
    # Set up JAX ops.
    model_optimizer = optimizer_cls(model.vars())
    loss = objax_utils.make_objax_loss_function(
        model, temperature, extended_batch_axis=extended_batch_axis
    )
    get_gradients = get_gradients_generator(loss, model.vars())

    @objax.Function.with_vars(
        model.vars() + model_optimizer.vars() + get_gradients.vars()
    )
    def train_op(batch):
      if extended_batch_axis:
        batch = jnp.expand_dims(batch, axis=0)
      gradients, loss_value = get_gradients(batch)
      model_optimizer(learning_rate, gradients)
      return loss_value

    if output_dir:
      tf_summary_writer = tf.summary.create_file_writer(output_dir).as_default()
    else:
      tf_summary_writer = None
    # Main train loop.
    train_inputs = tfds.as_numpy(train_input_iterator.repeat(epochs))
    start_time = time.time()
    metrics = defaultdict(list)
    for step, batch_inputs in enumerate(train_inputs):
      train_op(batch_inputs)
      batch_metrics = {}
      batch_metrics['train_loss'] = loss(batch_inputs)
      if step % steps_per_eval == 0:
        x_test, y_test = test_inputs
        embeddings = model(x_test, training=True)
        neighbors, labels = training_utils.compute_neighbors_and_true_labels(
            embeddings,
            y_test,
            test_class_to_idx,
            num_classes,
            num_eval_confusion_matrix,
            nearest_neighbors_per_example,
        )
        eval_metrics = training_utils.compute_eval_metrics(
            labels, neighbors, num_classes
        )
        batch_metrics.update(eval_metrics)
        batch_metrics['runtime'] = time.time() - start_time
      # Write metrics.
      for k, v in batch_metrics.items():
        metrics[k].append(v)
      batch_size = batch_inputs.shape[0]
      num_processed = step * batch_size
      if tf_summary_writer:
        training_utils.save_results(
            num_processed, batch_metrics, output_dir, tf_summary_writer
        )
    params_file_path = os.path.join(output_dir, final_checkpoint_file_name)
    training_utils.save_objax_model(model.vars(), params_file_path)
    return metrics

  return objax_trainer


def make_objax_finetuning_trainer(
    train_input_iterator,
    test_inputs,
    final_checkpoint_file_name,
    num_classes,
):
  """Creates an Objax trainer for finetuning on CIFAR100."""

  def objax_trainer(
      model,
      optimizer_cls,
      learning_rate,
      epochs,
      steps_per_eval,
      output_dir,
  ):
    loss = objax_utils.make_objax_finetuning_loss_function(model)
    vars_train = objax.VarCollection(
        (k, v) for k, v in model.vars().items() if 'classification_layer' in k
    )

    # Set up JAX ops.
    model_optimizer = optimizer_cls(vars_train)
    get_gradients_freeze = objax.GradValues(loss, vars_train)

    @objax.Function.with_vars(
        model.vars() + model_optimizer.vars() + get_gradients_freeze.vars()
    )
    def train_op(batch):
      gradients, loss_value = get_gradients_freeze(
          batch['image'], batch['label']
      )
      model_optimizer(learning_rate, gradients)
      return loss_value

    if output_dir:
      tf_summary_writer = tf.summary.create_file_writer(output_dir).as_default()
    else:
      tf_summary_writer = None
    # Main train loop.
    train_inputs = tfds.as_numpy(train_input_iterator.repeat(epochs))
    metrics = defaultdict(list)
    for step, batch_inputs in enumerate(train_inputs):
      train_op(batch_inputs)
      batch_metrics = {}
      batch_metrics['finetune_train_loss'] = loss(
          batch_inputs['image'], batch_inputs['label']
      )
      if step % steps_per_eval == 0:
        x_test, labels = test_inputs

        test_logits = model(x_test, training=False)
        predictions = np.argmax(objax.functional.softmax(test_logits), axis=1)
        eval_metrics = training_utils.compute_eval_metrics(
            predictions, labels, num_classes
        )
        finetune_eval_metrics = {}
        for key in eval_metrics:
          new_key = 'finetuning_' + key
          finetune_eval_metrics[new_key] = eval_metrics[key]

        batch_metrics.update(finetune_eval_metrics)

      # Write metrics.
      for k, v in batch_metrics.items():
        metrics[k].append(v)
      batch_size = batch_inputs['image'].shape[0]
      num_processed = step * batch_size
      if tf_summary_writer:
        training_utils.save_results(
            num_processed, batch_metrics, output_dir, tf_summary_writer
        )
    params_file_path = os.path.join(output_dir, final_checkpoint_file_name)
    training_utils.save_objax_model(model.vars(), params_file_path)
    return metrics

  return objax_trainer


def make_non_private_trainer(
    train_inputs,
    test_inputs,
    test_class_to_idx,
    num_classes,
    temperature,
    num_eval_confusion_matrix,
    nearest_neighbors_per_example,
    final_checkpoint_file_name,
):
  """Wrapper for making a non-private Objax trainer."""

  return make_objax_trainer(
      train_inputs,
      test_inputs,
      test_class_to_idx,
      objax.GradValues,
      num_classes,
      temperature,
      num_eval_confusion_matrix,
      nearest_neighbors_per_example,
      final_checkpoint_file_name,
  )


def make_naive_dp_trainer(
    train_input_iterator,
    test_inputs,
    test_class_to_idx,
    num_classes,
    temperature,
    l2_sensitivity,
    noise_multiplier,
    num_eval_confusion_matrix,
    nearest_neighbors_per_example,
    final_checkpoint_file_name,
):
  """Wrapper for making a naive DP Objax trainer."""

  def get_gradients_generator(loss, params):

    return objax.privacy.dpsgd.PrivateGradValues(
        f=loss,
        vc=params,
        noise_multiplier=noise_multiplier,
        l2_norm_clip=l2_sensitivity,
        microbatch=1,
        use_norm_accumulation=True,
    )

  return make_objax_trainer(
      train_input_iterator,
      test_inputs,
      test_class_to_idx,
      get_gradients_generator,
      num_classes,
      temperature,
      num_eval_confusion_matrix,
      nearest_neighbors_per_example,
      final_checkpoint_file_name,
      # Needed due to the microbatching logic in ContrastivePrivateGradValues().
      extended_batch_axis=True,
  )


def make_logit_dp_trainer(
    train_input_iterator,
    test_inputs_arr,
    test_class_to_idx,
    temperature,
    l2_sensitivity,
    noise_multiplier,
    num_grad_acc_steps,
    num_eval_confusion_matrix,
    nearest_neighbors_per_example,
    num_seq_comp_steps,
    weight_decay,
    final_checkpoint_file_name,
    model_name='',
):
  """Creates a logit-based DP trainer."""

  # For now, hardcode this for CIFAR10.
  if model_name == 'ResNet18':
    input_shape = (1, 3, 32, 32)
  else:
    input_shape = (3, 32, 32)

  def logit_dp_trainer(
      model, optimizer, learning_rate, epochs, steps_per_eval, output_dir
  ):
    logging.warning('Optimizer is ignored.')
    del optimizer  # Fixed to Adam for now.
    optax_params = defaultdict(list)
    optax_params['learning_rate'] = learning_rate
    optax_params['weight_decay'] = weight_decay
    opt_params = training_utils.OptimizationParams(
        noise_multiplier=noise_multiplier,
        l2_norm_clip=l2_sensitivity,
        temperature=temperature,
        gradient_accumulation_steps=(num_grad_acc_steps or 1),
        sequential_computation_steps=(num_seq_comp_steps or 1),
        optax_params=optax_params,
    )
    dummy_seed = 0
    model_config = training_utils.ModelConfig(
        input_shape=input_shape, seed=dummy_seed, opt_params=opt_params
    )
    final_state, metrics = training_utils.fit_model(
        encoder=model,
        train_inputs=train_input_iterator,
        test_inputs=test_inputs_arr,
        test_lookup_class_to_idx=test_class_to_idx,
        num_epochs=epochs,
        output_dir=output_dir,
        model_config=model_config,
        steps_per_eval=steps_per_eval,
        num_eval_confusion_matrix=num_eval_confusion_matrix,
        nearest_neighbors_per_example=nearest_neighbors_per_example,
    )
    params_file_path = os.path.join(output_dir, final_checkpoint_file_name)
    objax_params = training_utils.get_objax_params_from_haiku_params(
        final_state.params
    )
    training_utils.save_objax_model(objax_params, params_file_path)
    return metrics

  return logit_dp_trainer
