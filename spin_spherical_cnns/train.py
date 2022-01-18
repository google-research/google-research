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

"""Functions for training Spin-Weighted Spherical CNNs."""

import functools
import os
from typing import Any, Callable, Dict, Sequence, Tuple, Union

from absl import logging

from clu import checkpoint
from clu import metric_writers
from clu import metrics
from clu import parameter_overview
from clu import periodic_actions
import flax
import flax.jax_utils as flax_utils
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import tensorflow as tf

from spin_spherical_cnns import input_pipeline
from spin_spherical_cnns import models


_PMAP_AXIS_NAME = "batch"


@flax.struct.dataclass
class TrainState:
  step: int
  optimizer: flax.optim.Optimizer
  batch_stats: Any


def create_train_state(config, rng,
                       input_shape,
                       num_classes):
  """Create and initialize the model.

  Args:
    config: Configuration for model.
    rng: JAX PRNG Key.
    input_shape: Shape of the inputs fed into the model.
    num_classes: Number of classes in the output layer.

  Returns:
    The model and initialized TrainState with the optimizer.
  """
  # Use of getattr could simplify this but is discouraged by the style guide. We
  # should consider using it anyway if sequence of ifs grows too big.
  if config.model_name == "tiny_classifier":
    get_model = models.tiny_classifier
  elif config.model_name == "spin_classifier_6_layers":
    get_model = models.spin_classifier_6_layers
  elif config.model_name == "spherical_classifier_6_layers":
    get_model = models.spherical_classifier_6_layers
  elif config.model_name == "cnn_classifier_6_layers":
    get_model = models.cnn_classifier_6_layers
  else:
    raise ValueError(f"Model {config.model_name} not supported.")
  model = get_model(num_classes, axis_name=_PMAP_AXIS_NAME)

  variables = model.init(rng, jnp.ones(input_shape), train=False)
  params = variables["params"]
  batch_stats = variables.get("batch_stats", {})

  abs_if_complex = lambda x: jnp.abs(x) if x.dtype == jnp.complex64 else x
  parameter_overview.log_parameter_overview(
      jax.tree_util.tree_map(abs_if_complex, params))
  optimizer = flax.optim.Adam().create(params)
  return model, TrainState(step=0, optimizer=optimizer, batch_stats=batch_stats)


def cross_entropy_loss(*, logits, labels):
  logp = jax.nn.log_softmax(logits)
  loglik = jnp.take_along_axis(logp, labels[:, None], axis=1)
  return -loglik


@flax.struct.dataclass
class EvalMetrics(metrics.Collection):
  accuracy: metrics.Accuracy
  eval_loss: metrics.Average.from_output("loss")


@flax.struct.dataclass
class TrainMetrics(metrics.Collection):
  train_accuracy: metrics.Accuracy
  learning_rate: metrics.LastValue.from_output("learning_rate")
  loss: metrics.Average.from_output("loss")
  loss_std: metrics.Std.from_output("loss")


def cosine_decay(lr, step, total_steps):
  ratio = jnp.maximum(0., step / total_steps)
  mult = 0.5 * (1. + jnp.cos(jnp.pi * ratio))
  return mult * lr


def get_learning_rate(step,
                      *,
                      base_learning_rate,
                      steps_per_epoch,
                      num_epochs,
                      warmup_epochs = 5):
  """Cosine learning rate schedule."""
  logging.info(("get_learning_rate(step=%s, "
                "base_learning_rate=%s, "
                "steps_per_epoch=%s, "
                "num_epochs=%s)"),
               step, base_learning_rate, steps_per_epoch, num_epochs)

  if steps_per_epoch <= 0:
    raise ValueError(f"steps_per_epoch should be a positive integer but was "
                     f"{steps_per_epoch}.")
  if warmup_epochs >= num_epochs:
    raise ValueError(f"warmup_epochs should be smaller than num_epochs. "
                     f"Currently warmup_epochs is {warmup_epochs}, "
                     f"and num_epochs is {num_epochs}.")
  epoch = step / steps_per_epoch
  lr = cosine_decay(base_learning_rate, epoch - warmup_epochs,
                    num_epochs - warmup_epochs)
  warmup = jnp.minimum(1., epoch / warmup_epochs)
  return lr * warmup


def train_step(model, state,
               batch, learning_rate_fn,
               weight_decay):
  """Perform a single training step.

  Args:
    model: Flax module for the model. The apply method must take input images
      and a boolean argument indicating whether to use training or inference
      mode.
    state: State of the model (optimizer and state).
    batch: Training inputs for this step.
    learning_rate_fn: Function that computes the learning rate given the step
      number.
    weight_decay: Weighs L2 regularization term.

  Returns:
    The new model state and dictionary with metrics.
  """
  logging.info("train_step(batch=%s)", batch)

  step = state.step + 1
  lr = learning_rate_fn(step)

  def loss_fn(params):
    variables = {"params": params, "batch_stats": state.batch_stats}
    logits, new_variables = model.apply(
        variables, batch["input"], mutable=["batch_stats"], train=True)

    loss = jnp.mean(cross_entropy_loss(logits=logits, labels=batch["label"]))
    weight_penalty_params = jax.tree_leaves(variables["params"])
    # NOTE(machc): jnp.abs is needed for this to work with complex weights.
    weight_l2 = sum(
        [jnp.sum(jnp.abs(x)**2) for x in weight_penalty_params if x.ndim > 1])
    weight_penalty = weight_decay * 0.5 * weight_l2
    loss = loss + weight_penalty
    return loss, (new_variables["batch_stats"], logits)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, (new_batch_stats, logits)), grad = grad_fn(state.optimizer.target)

  # Compute average gradient across multiple workers.
  grad = jax.lax.pmean(grad, axis_name=_PMAP_AXIS_NAME)

  # NOTE(machc): JAX uses different conventions than TensorFlow for the
  # gradients of complex functions. They differ by a conjugate, so we conjugate
  # all gradients here in order to make gradient descent work seamlessly. This
  # is crucial if there are complex weights in the model, and makes no
  # difference for real weights. See https://github.com/google/jax/issues/4891.
  grad = jax.tree_map(jnp.conj, grad)

  new_optimizer = state.optimizer.apply_gradient(grad, learning_rate=lr)
  new_state = state.replace(  # pytype: disable=attribute-error
      step=step,
      optimizer=new_optimizer,
      batch_stats=new_batch_stats)

  metrics_update = TrainMetrics.gather_from_model_output(
      loss=loss, logits=logits, labels=batch["label"], learning_rate=lr)
  return new_state, metrics_update


@functools.partial(jax.pmap,
                   axis_name=_PMAP_AXIS_NAME,
                   static_broadcasted_argnums=0)
def eval_step(model, state,
              batch):
  """Compute the metrics for the given model in inference mode.

  The model is applied to the inputs with train=False using all devices on the
  host. Afterwards metrics are averaged across *all* devices (of all hosts).

  Args:
    model: Flax module for the model. The apply method must take input images
      and a boolean argument indicating whether to use training or inference
      mode.
    state: Replicate model state.
    batch: Inputs that should be evaluated.

  Returns:
    Dictionary of the replicated metrics.
  """
  logging.info("eval_step(batch=%s)", batch)
  variables = {
      "params": state.optimizer.target,
      "batch_stats": state.batch_stats
  }
  logits = model.apply(variables, batch["input"], mutable=False, train=False)
  loss = jnp.mean(cross_entropy_loss(logits=logits, labels=batch["label"]))
  return EvalMetrics.gather_from_model_output(
      logits=logits,
      labels=batch["label"],
      loss=loss,
      mask=batch.get("mask"),
  )


class StepTraceContextHelper:
  """Helper class to use jax.profiler.StepTraceContext."""

  def __init__(self, name, init_step_num):
    self.name = name
    self.step_num = init_step_num

  def __enter__(self):
    self.context = jax.profiler.StepTraceContext(
        self.name, step_num=self.step_num)
    self.step_num += 1
    self.context.__enter__()
    return self

  def __exit__(self, exc_type, exc_value, tb):
    self.context.__exit__(exc_type, exc_value, tb)
    self.context = None

  def next_step(self):
    self.context.__exit__(None, None, None)
    self.__enter__()


def evaluate(model,
             state,
             eval_ds,
             num_eval_steps = -1):
  """Evaluate the model on the given dataset."""
  logging.info("Starting evaluation.")
  eval_metrics = None
  with StepTraceContextHelper("eval", 0) as trace_context:
    for step, batch in enumerate(eval_ds):  # pytype: disable=wrong-arg-types
      batch = jax.tree_map(np.asarray, batch)
      metrics_update = flax_utils.unreplicate(
          eval_step(model, state, batch))
      eval_metrics = (
          metrics_update
          if eval_metrics is None else eval_metrics.merge(metrics_update))
      if num_eval_steps > 0 and step + 1 == num_eval_steps:
        break
      trace_context.next_step()
  return eval_metrics


def train_and_evaluate(config, workdir):
  """Runs a training and evaluation loop.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """
  tf.io.gfile.makedirs(workdir)

  rng = jax.random.PRNGKey(config.seed)

  # Build input pipeline.
  rng, data_rng = jax.random.split(rng)
  data_rng = jax.random.fold_in(data_rng, jax.host_id())
  splits = input_pipeline.create_datasets(config, data_rng)
  num_classes = splits.info.features["label"].num_classes
  train_iter = iter(splits.train)  # pytype: disable=wrong-arg-types

  # Learning rate schedule.
  num_train_steps = config.num_train_steps
  if num_train_steps == -1:
    num_train_steps = splits.train.cardinality().numpy()
  steps_per_epoch = num_train_steps // config.num_epochs
  logging.info("num_train_steps=%d, steps_per_epoch=%d", num_train_steps,
               steps_per_epoch)
  # We treat the learning rate in the config as the learning rate for batch size
  # 32 but scale it according to our batch size.
  global_batch_size = config.per_device_batch_size * jax.device_count()
  base_learning_rate = config.learning_rate * global_batch_size / 32.0
  learning_rate_fn = functools.partial(
      get_learning_rate,
      base_learning_rate=base_learning_rate,
      steps_per_epoch=steps_per_epoch,
      num_epochs=config.num_epochs,
      warmup_epochs=config.warmup_epochs)

  # Initialize model.
  rng, model_rng = jax.random.split(rng)
  model, state = create_train_state(
      config,
      model_rng,
      input_shape=splits.train.element_spec["input"].shape[1:],
      num_classes=num_classes)

  # Set up checkpointing of the model and the input pipeline.
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  ckpt = checkpoint.MultihostCheckpoint(
      checkpoint_dir, {"train_iter": train_iter}, max_to_keep=2)
  state = ckpt.restore_or_initialize(state)
  initial_step = int(state.step) + 1

  # Count number of trainable parameters. This must be done before replicating
  # the state to avoid double-counting replicated parameters.
  param_count = sum(p.size for p in jax.tree_leaves(state.optimizer.target))

  # Distribute training over local devices.
  state = flax_utils.replicate(state)

  p_train_step = jax.pmap(
      functools.partial(
          train_step,
          model=model,
          learning_rate_fn=learning_rate_fn,
          weight_decay=config.weight_decay),
      axis_name=_PMAP_AXIS_NAME)

  writer = metric_writers.create_default_writer(
      workdir, just_logging=jax.host_id() > 0)
  if initial_step == 1:
    writer.write_hparams(dict(config))
    # Log the number of trainable params.
    writer.write_scalars(initial_step, {"param_count": param_count})

  logging.info("Starting training loop at step %d.", initial_step)
  hooks = []
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=num_train_steps, writer=writer)
  if jax.host_id() == 0:
    hooks += [
        report_progress,
        periodic_actions.Profile(num_profile_steps=5, logdir=workdir)
    ]
  train_metrics = None
  with metric_writers.ensure_flushes(writer):
    for step in range(initial_step, num_train_steps + 1):
      # `step` is a Python integer. `state.step` is JAX integer on the GPU/TPU
      # devices.
      is_last_step = step == num_train_steps

      with jax.profiler.StepTraceContext("train", step_num=step):
        batch = jax.tree_map(np.asarray, next(train_iter))
        state, metrics_update = p_train_step(state=state, batch=batch)
        metric_update = flax_utils.unreplicate(metrics_update)
        train_metrics = (
            metric_update
            if train_metrics is None else train_metrics.merge(metric_update))

      # Quick indication that training is happening.
      logging.log_first_n(logging.INFO, "Finished training step %d.", 5, step)
      for h in hooks:
        h(step)

      if step % config.log_loss_every_steps == 0 or is_last_step:
        writer.write_scalars(step, train_metrics.compute())
        train_metrics = None

      # When combining train and eval, we do not evaluate while training.
      if ((step % config.eval_every_steps == 0 or is_last_step) and
          not config.combine_train_val_and_eval_on_test):
        with report_progress.timed("eval"):
          eval_metrics = evaluate(model, state,
                                  splits.validation, config.num_eval_steps)
        writer.write_scalars(step, eval_metrics.compute())

      if step % config.checkpoint_every_steps == 0 or is_last_step:
        with report_progress.timed("checkpoint"):
          ckpt.save(flax_utils.unreplicate(state))

      if is_last_step and config.combine_train_val_and_eval_on_test:
        # Evaluate a single time on the test set when requested.
        with report_progress.timed("test"):
          test_metrics = evaluate(model, state,
                                  splits.test, config.num_eval_steps)
        writer.write_scalars(step, test_metrics.compute())

  logging.info("Finishing training at step %d", num_train_steps)
