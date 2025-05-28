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

"""Methods for training ResNet-50 on ImageNet using JAX."""

import functools
import os
from typing import Any, Callable, Dict, Sequence, Tuple, Union, Optional

from absl import logging

from clu import checkpoint
from clu import metric_writers
from clu import metrics
from clu import parameter_overview
from clu import periodic_actions
import flax
import flax.jax_utils as flax_utils
import flax.linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import tensorflow as tf

from representation_clustering import input_pipeline_breeds
from representation_clustering import input_pipeline_celebA
from representation_clustering import resnet_v1


class TrainState(train_state.TrainState):
  batch_stats: Any
  ema_tx: optax.GradientTransformation = flax.struct.field(pytree_node=False)
  ema_stats: optax.OptState
  ema_params: Optional[flax.core.FrozenDict[str, Any]] = None


_pmap_bias_correction = jax.pmap(optax.bias_correction, "x",
                                 in_axes=(0, None, 0))
_cross_replica_mean = jax.pmap(lambda x: jax.lax.pmean(x, "x"), "x")


def merge_batch_stats(config,
                      replicated_state):
  """Merge model batch stats and populate ema_params."""
  if jax.tree.leaves(replicated_state.batch_stats):
    replicated_state = replicated_state.replace(
        batch_stats=_cross_replica_mean(replicated_state.batch_stats))
  if config.ema_decay:
    # Perform bias correction for EMA to get EMA parameters.
    replicated_state = replicated_state.replace(
        ema_params=_pmap_bias_correction(replicated_state.ema_stats.ema,
                                         config.ema_decay,
                                         replicated_state.ema_stats.count))
  return replicated_state


def create_train_state(
    config, rng,
    input_shape, num_classes,
    learning_rate_fn):
  """Create and initialize the model.

  Args:
    config: Configuration for model.
    rng: JAX PRNG Key.
    input_shape: Shape of the inputs fed into the model.
    num_classes: Number of classes in the output layer.
    learning_rate_fn: A function that takes the current step and returns the
      learning rate to use for training.

  Returns:
    The initialized TrainState with the optimizer.
  """
  if config.model_name == "resnet18":
    model_cls = resnet_v1.ResNet18
  elif config.model_name == "resnet50":
    model_cls = resnet_v1.ResNet50
  else:
    raise ValueError(f"Model {config.model_name} not supported.")
  model = model_cls(num_classes=num_classes,
                    batch_norm_decay=config.batch_norm_decay)
  variables = model.init(rng, jnp.ones(input_shape), train=False)
  params = variables["params"]
  batch_stats = variables["batch_stats"]
  parameter_overview.log_parameter_overview(params)
  tx = optax.sgd(
      learning_rate=learning_rate_fn,
      momentum=config.sgd_momentum
  )
  if config.ema_decay:
    ema_tx = optax.ema(config.ema_decay, debias=True)
  else:
    ema_tx = optax.identity()
  ema_stats = ema_tx.init(params)
  state = TrainState.create(
      apply_fn=model.apply,
      params=params,
      tx=tx,
      ema_tx=ema_tx,
      batch_stats=batch_stats,
      ema_stats=ema_stats)
  return model, state


def cross_entropy_loss(*, logits, labels):
  logp = jax.nn.log_softmax(logits)
  loglik = jnp.take_along_axis(logp, labels[:, None], axis=1)
  return -loglik


def squared_loss(*, logits, labels):
  k = 9
  M = 60  # pylint: disable=invalid-name
  n_class = logits.shape[-1]
  one_hot = jax.nn.one_hot(labels, n_class)
  # return jnp.sum(jnp.square(logits - one_hot), axis=1)
  return jnp.sum(
      jnp.multiply((k - 1) * one_hot + 1, jnp.square(M * one_hot - logits)),
      axis=1)


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
  logging.info(
      "get_learning_rate(step=%s, base_learning_rate=%s, steps_per_epoch=%s, num_epochs=%s",
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


def train_step(model, state, batch,
               learning_rate_fn, weight_decay,
               loss_fn):
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
    loss_fn: Choice of loss function.

  Returns:
    The new model state and dictionary with metrics.
  """
  lr = learning_rate_fn(state.step)

  def loss_function(params):
    variables = {"params": params, "batch_stats": state.batch_stats}
    logits, new_variables = model.apply(
        variables, batch["image"], mutable=["batch_stats"], train=True)
    if loss_fn == "cross_entropy":
      loss = jnp.mean(cross_entropy_loss(logits=logits, labels=batch["label"]))
    elif loss_fn == "squared":
      loss = jnp.mean(squared_loss(logits=logits, labels=batch["label"]))
      loss = loss / 10.0
    weight_penalty_params = jax.tree.leaves(variables["params"])
    weight_l2 = sum(
        [jnp.sum(x**2) for x in weight_penalty_params if x.ndim > 1])
    weight_penalty = weight_decay * 0.5 * weight_l2
    loss = loss + weight_penalty
    return loss, (new_variables["batch_stats"], logits)

  grad_fn = jax.value_and_grad(loss_function, has_aux=True)
  (loss, (new_batch_stats, logits)), grad = grad_fn(state.params)

  # Compute average gradient across multiple workers.
  grad = jax.lax.pmean(grad, axis_name="batch")
  new_state = state.apply_gradients(grads=grad, batch_stats=new_batch_stats)

  # Update EMA params.
  _, ema_stats = new_state.ema_tx.update(new_state.params, new_state.ema_stats)
  new_state = new_state.replace(ema_stats=ema_stats)

  metrics_update = TrainMetrics.gather_from_model_output(
      loss=loss, logits=logits, labels=batch["label"], learning_rate=lr)
  return new_state, metrics_update


@functools.partial(
    jax.pmap, axis_name="batch", static_broadcasted_argnums=(0, 3))
def eval_step(model, state, batch,
              loss_fn):
  """Compute the metrics for the given model in inference mode.

  The model is applied to the inputs with train=False using all devices on the
  host. Afterwards metrics are averaged across *all* devices (of all hosts).

  Args:
    model: Flax module for the model. The apply method must take input images
      and a boolean argument indicating whether to use training or inference
      mode.
    state: Replicate model state.
    batch: Inputs that should be evaluated.
    loss_fn: Choice of loss function.

  Returns:
    Dictionary of the replicated metrics.
  """
  variables = {
      "params": state.params if state.ema_params is None else state.ema_params,
      "batch_stats": state.batch_stats
  }
  logits = model.apply(variables, batch["image"], mutable=False, train=False)
  if loss_fn == "cross_entropy":
    loss = jnp.mean(cross_entropy_loss(logits=logits, labels=batch["label"]))
  elif loss_fn == "squared":
    loss = jnp.mean(squared_loss(logits=logits, labels=batch["label"]))
    loss = loss / 10.0
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
    self.context = jax.profiler.StepTraceAnnotation(
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
             num_eval_steps = -1,
             loss_fn = "cross_entropy"):
  """Evaluate the model on the given dataset."""
  logging.info("Starting evaluation.")
  eval_metrics = None
  with StepTraceContextHelper("eval", 0) as trace_context:
    for step, batch in enumerate(eval_ds):  # pytype: disable=wrong-arg-types
      logging.info("Eval step %d", step)
      batch = jax.tree.map(np.asarray, batch)
      metrics_update = flax_utils.unreplicate(
          eval_step(model, state, batch, loss_fn))
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
  data_rng = jax.random.fold_in(data_rng, jax.process_index())
  if config.dataset_name == "breeds":
    num_train_steps, num_eval_steps, num_classes, train_ds, eval_ds = input_pipeline_breeds.create_datasets(
        config)
  else:
    num_train_steps, num_eval_steps, num_classes, train_ds, eval_ds = input_pipeline_celebA.create_datasets(
        config)

  train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types

  # Learning rate schedule.
  if config.num_train_steps > 0:
    num_train_steps = config.num_train_steps
  steps_per_epoch = num_train_steps // config.num_epochs
  logging.info("num_train_steps=%d, steps_per_epoch=%d", num_train_steps,
               steps_per_epoch)
  logging.info("num_eval_steps=%d", num_eval_steps)

  # We treat the learning rate in the config as the learning rate for batch size
  # 256 but scale it according to our batch size.
  global_batch_size = config.per_device_batch_size * jax.device_count()
  if "breeds" in workdir:
    base_learning_rate = config.learning_rate * global_batch_size / 256.0
    learning_rate_fn = functools.partial(
        get_learning_rate,
        base_learning_rate=base_learning_rate,
        steps_per_epoch=steps_per_epoch,
        num_epochs=config.num_epochs,
        warmup_epochs=config.warmup_epochs)
  else:
    base_learning_rate = config.learning_rate * global_batch_size / 128.0
    learning_rate_fn = lambda x: base_learning_rate

  # Initialize model.
  rng, model_rng = jax.random.split(rng)
  model, state = create_train_state(
      config, model_rng, input_shape=(8, 224, 224, 3), num_classes=num_classes,
      learning_rate_fn=learning_rate_fn)

  # Set up checkpointing of the model and the input pipeline.
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  ckpt = checkpoint.MultihostCheckpoint(checkpoint_dir, max_to_keep=1000)
  state = ckpt.restore_or_initialize(state)
  initial_step = int(state.step) + 1

  # Distribute training.
  state = flax_utils.replicate(state)
  p_train_step = jax.pmap(
      functools.partial(
          train_step,
          model=model,
          learning_rate_fn=learning_rate_fn,
          weight_decay=config.weight_decay,
          loss_fn=config.loss_fn),
      axis_name="batch")

  writer = metric_writers.create_default_writer(
      workdir, just_logging=jax.process_index() > 0)
  if initial_step == 1:
    writer.write_hparams(dict(config))

  logging.info("Starting training loop at step %d.", initial_step)
  hooks = []
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=num_train_steps, writer=writer)
  if jax.process_index() == 0:
    hooks += [
        report_progress,
        # periodic_actions.Profile(num_profile_steps=5, logdir=workdir)
    ]
  train_metrics = None
  with metric_writers.ensure_flushes(writer):
    for step in range(initial_step, num_train_steps + 1):
      # `step` is a Python integer. `state.step` is JAX integer on the GPU/TPU
      # devices.
      is_last_step = step == num_train_steps

      with jax.profiler.StepTraceAnnotation("train", step_num=step):
        batch = jax.tree.map(np.asarray, next(train_iter))
        state, metrics_update = p_train_step(state=state, batch=batch)
        metric_update = flax_utils.unreplicate(metrics_update)
        train_metrics = (
            metric_update
            if train_metrics is None else train_metrics.merge(metric_update))

      # Quick indication that training is happening.
      # logging.log_first_n(logging.INFO, "Finished training step %d.", 5, step)
      for hook_idx, h in enumerate(hooks):
        if hook_idx == 0:
          logging.info("reporting progress at step %d", step)
        else:
          logging.info("profiling at step %d", step)
        h(step)

      if step % config.log_loss_every_steps == 0 or is_last_step:
        logging.info("Writing train metrics at step %d", step)
        writer.write_scalars(step, train_metrics.compute())
        train_metrics = None

      if step % config.eval_every_steps == 0 or is_last_step:
        logging.info("Writing eval metrics at step %d", step)
        with report_progress.timed("eval"):
          state = merge_batch_stats(config, state)
          eval_metrics = evaluate(model, state, eval_ds, config.num_eval_steps,
                                  config.loss_fn)
        eval_metrics_cpu = jax.tree.map(np.array, eval_metrics.compute())
        writer.write_scalars(step, eval_metrics_cpu)

      if step % (steps_per_epoch * 5) == 0 or is_last_step:
        logging.info("Writing checkpoint at step %d", step)
        with report_progress.timed("checkpoint"):
          state = merge_batch_stats(config, state)
          ckpt.save(flax_utils.unreplicate(state))

  logging.info("Finishing training at step %d", num_train_steps)
