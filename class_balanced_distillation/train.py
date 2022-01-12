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

"""Library for training a model."""

import copy
import functools
import math

from absl import logging
from class_balanced_distillation import input_pipeline
from class_balanced_distillation import resnet
from clu import metric_writers
from clu import parameter_overview
from clu import periodic_actions
import ml_collections
import sonnet as snt
import tensorflow as tf


class TeacherModel(tf.keras.Model):
  """Teacher model class."""

  def __init__(self,
               teacher_models,
               name=""):
    super(TeacherModel, self).__init__(name=name)
    self.teacher_models = teacher_models
    self.num_teacher_models = len(self.teacher_models)

  def __call__(self, image, label=None, training=False, return_features=True):
    teacher_features = [[]] * self.num_teacher_models
    teacher_logits = [[]] * self.num_teacher_models
    for i in range(self.num_teacher_models):
      teacher_features[i], teacher_logits[i] = self.teacher_models[i](
          image, label, training=False, return_features=True)  # type: ignore

    teacher_features = tf.concat(teacher_features, axis=1)
    teacher_logits = tf.math.add_n(teacher_logits) / self.num_teacher_models

    return teacher_features, teacher_logits


class MyMetrics(tf.Module):
  """Collections of metrics to track in your model."""

  def __init__(self, prefix):
    self.prefix = prefix
    # Update self.update_state() if you add new metrics here.
    self.loss = tf.keras.metrics.Mean()
    self.distill_loss = tf.keras.metrics.Mean()
    self.distill_feat_loss = tf.keras.metrics.Mean()
    self.accuracy = tf.keras.metrics.Accuracy()
    self.learning_rate = None

  def reset_states(self):
    for v in vars(self).values():
      if isinstance(v, tf.keras.metrics.Metric):
        v.reset_states()

  def reset_lr(self):
    self.learning_rate = None

  def result(self):
    out_dict = {}
    for k, v in vars(self).items():
      if isinstance(v, tf.keras.metrics.Metric):
        out_dict[self.prefix + k] = v.result()
      elif isinstance(v, tf.Tensor):
        out_dict[self.prefix + k] = v

    return out_dict

  def update_state(self, *, labels, logits, alpha=1.0):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels)
    loss = (1-alpha) * loss
    self.loss.update_state(loss)
    self.accuracy.update_state(
        y_true=labels, y_pred=tf.math.argmax(logits, axis=-1))

  def update_state_lr(self, lr):
    self.learning_rate = lr

  def update_distill_state(self, *, logits, features, teacher_logits,
                           teacher_feats, distill_params):

    feat_distill_loss = compute_feat_distillation(features,
                                                  teacher_feats)
    feat_distill_loss = (distill_params["alpha"] *
                         distill_params["beta"] * feat_distill_loss)

    self.distill_feat_loss.update_state(feat_distill_loss)


def compute_feat_distillation(features, teacher_features):
  features = tf.nn.l2_normalize(features, axis=1)
  teacher_features = tf.nn.l2_normalize(teacher_features, axis=1)

  cosine_loss = 1.0 - tf.reduce_sum(features * teacher_features, axis=-1)
  cosine_loss = tf.reduce_mean(cosine_loss)
  return cosine_loss


def load_teacher_models(file_list, num_classes, config, strategy):
  """Loads teacher models for distillation."""
  teacher_config = copy.deepcopy(config)
  teacher_config.proj_dim = -1
  num_teacher_models = len(file_list)
  teacher_models = [[]] * num_teacher_models
  for i in range(num_teacher_models):
    with strategy.scope():
      teacher_model = get_model(teacher_config, num_classes)
      teacher_filename = "%s/final_weights" % file_list[i]

      if not tf.io.gfile.exists(teacher_filename+".index"):
        logging.error("Teacher model is not trained: %s", file_list[i])
        raise ValueError(f"Teacher model {file_list[i]} does not exist.")

      teacher_model.load_weights(teacher_filename)
    teacher_model.trainable = False
    teacher_models[i] = teacher_model

  return teacher_models


def cosine_decay(lr, step, total_steps):
  """Cosine decay scheduler for the learning rate."""
  ratio = tf.math.maximum(0., tf.cast(step / total_steps, tf.float32))
  mult = 0.5 * (1. + tf.math.cos(math.pi * ratio))
  return mult * lr


def get_learning_rate(step,
                      *,
                      base_learning_rate,
                      steps_per_epoch,
                      num_epochs,
                      warmup_epochs = 5):
  """Cosine learning rate schedule."""
  if steps_per_epoch <= 0:
    raise ValueError(f"steps_per_epoch should be a positive integer but was "
                     f"{steps_per_epoch}.")
  if warmup_epochs >= num_epochs:
    raise ValueError(f"warmup_epochs should be smaller than num_epochs. "
                     f"Currently warmup_epochs is {warmup_epochs}, "
                     f"and num_epochs is {num_epochs}.")
  epoch = tf.cast(step / steps_per_epoch, tf.float32)
  lr = cosine_decay(base_learning_rate, epoch - warmup_epochs,
                    num_epochs - warmup_epochs)
  warmup = tf.math.minimum(1., epoch / warmup_epochs)
  return lr * warmup


def get_model(config,
              num_classes
              ):
  """Creates an instance of a ResNet model."""
  if config.model_name == "resnet18":
    model = resnet.resnet(
        num_layers=18,
        num_classes=num_classes,
        proj_dim=config.proj_dim,
        )
  elif config.model_name == "resnet50":
    model = resnet.resnet(
        num_layers=50,
        num_classes=num_classes,
        proj_dim=config.proj_dim,
        )
  elif config.model_name == "resnet101":
    model = resnet.resnet(
        num_layers=101,
        num_classes=num_classes,
        proj_dim=config.proj_dim,
        )
  elif config.model_name == "resnet152":
    model = resnet.resnet(
        num_layers=152,
        num_classes=num_classes,
        proj_dim=config.proj_dim,
        )
  else:
    raise ValueError(f"Model {config.model_name} not supported.")

  return model


def create_state(config,
                 *,
                 num_classes,
                 strategy
                 ):
  """Creates the model state.

  Args:
    config: Configuration for this model.
    num_classes: Number of classes for the network head.
    strategy: Distribution strategy.

  Returns:
    The state as `tf.train.Checkpoint`. This includes the `model` (network),
    the `optimizer` and the `global_step` variable.
  """
  with strategy.scope():
    model = get_model(config, num_classes)
    global_step = tf.Variable(
        0, trainable=False, name="global_step", dtype=tf.int64)
    optimizer = snt.optimizers.Momentum(
        config.learning_rate, momentum=config.sgd_momentum)
    return tf.train.Checkpoint(
        model=model,
        optimizer=optimizer,
        global_step=global_step,
        train_metrics=MyMetrics("train_"),
        val_metrics=MyMetrics("val_"),
        test_metrics=MyMetrics("test_"))


@tf.function
def train_step(state, train_inputs, weight_decay,
               learning_rate_fn, do_distill, distill_params,
               strategy):
  """Perform a single training step. Returns the loss."""

  # Set the learning rate
  step = state.global_step + 1
  state.optimizer.learning_rate = learning_rate_fn(step)

  def step_fn(image, label):  # pylint: disable=missing-docstring

    if do_distill:
      teacher_model = distill_params["teacher_model"]
      teacher_features, teacher_logits = teacher_model(image, label)

    with tf.GradientTape() as tape:
      features, logits = state.model(
          image, label, training=True, return_features=True)

      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=label)
      loss = tf.reduce_mean(loss)

      if do_distill:

        feat_distill_loss = compute_feat_distillation(features,
                                                      teacher_features)
        distill_loss = distill_params["beta"] * feat_distill_loss
        loss = (1 - distill_params["alpha"]) * loss + (
            distill_params["alpha"] * distill_loss)
      else:
        distill_loss = 0.0

      # Compute the L2 regularization loss
      model_variables = [
          tf.reshape(v, (-1,)) for v in state.model.trainable_variables
      ]
      weight_l2 = tf.nn.l2_loss(tf.concat(model_variables, axis=0))
      weight_penalty = weight_decay * 0.5 * weight_l2
      loss = loss + weight_penalty

      # We scale the local loss here so we can later sum the gradients.
      # This is cheaper than averaging all gradients.
      local_loss = loss / strategy.num_replicas_in_sync

    replica_ctx = tf.distribute.get_replica_context()
    grads = tape.gradient(local_loss, state.model.trainable_variables)
    grads = replica_ctx.all_reduce("sum", grads)
    state.optimizer.apply(grads, state.model.trainable_variables)
    state.global_step.assign_add(1)
    if do_distill:
      state.train_metrics.update_state(labels=label, logits=logits,
                                       alpha=distill_params["alpha"])
      state.train_metrics.update_distill_state(
          logits=logits,
          features=features,
          teacher_logits=teacher_logits,
          teacher_feats=teacher_features,
          distill_params=distill_params)
    else:
      state.train_metrics.update_state(labels=label, logits=logits)

  strategy.run(step_fn, kwargs=next(train_inputs))


def eval_step(state, image, label,
              mask, metrics):
  """Single non-distributed eval step."""

  image = tf.boolean_mask(image, mask, axis=0)
  label = tf.boolean_mask(label, mask, axis=0)

  _, logits = state.model(image, training=False, return_features=True)

  metrics.update_state(logits=logits, labels=label)


def evaluate(state, eval_ds, metrics,
             strategy):
  """Evaluates the model on the `eval_ds`."""
  metrics.reset_states()
  tf_eval_step = tf.function(functools.partial(eval_step, state=state,
                                               metrics=metrics))
  for batch in eval_ds:
    strategy.run(tf_eval_step, kwargs=batch)


def train_and_evaluate(config, workdir,
                       strategy):
  """Runs a training and evaluation loop.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
    strategy: Distribution strategy to use for distributing the model.
  """
  tf.io.gfile.makedirs(workdir)

  tf_rng, data_rng = tf.random.experimental.stateless_split((config.seed, 0), 2)
  tf.random.set_seed(tf_rng.numpy()[0])

  # Input pipeline.
  ds_info, train_ds, val_ds, test_ds = input_pipeline.create_datasets(
      config, data_rng, strategy=strategy)
  train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types

  # Learning rate schedule.
  num_train_steps = config.num_train_steps
  if num_train_steps == -1:
    num_train_steps = (ds_info.splits["train"].num_examples
                       // config.global_batch_size * config.num_epochs)
  steps_per_epoch = num_train_steps // config.num_epochs
  logging.info("num_train_steps=%d, steps_per_epoch=%d", num_train_steps,
               steps_per_epoch)

  # We treat the learning rate in the config as the learning rate for batch size
  # 256 but scale it according to our batch size.
  base_learning_rate = config.learning_rate * config.global_batch_size / 256.0

  # Initialize model.
  num_classes = ds_info.features["label"].num_classes

  if config.distill_teacher:
    do_distill = True
    teacher_file_list = (config.distill_teacher).split(",")
    teacher_models = load_teacher_models(teacher_file_list, num_classes, config,
                                         strategy)
    distill_params = {}
    distill_params["alpha"] = config.distill_alpha
    distill_params["beta"] = config.distill_fd_beta
    distill_params["teacher_model"] = TeacherModel(teacher_models,
                                                   name="teacher")
  else:
    do_distill = False
    distill_params = None

  state = create_state(
      config,
      num_classes=num_classes,
      strategy=strategy)

  ckpt_manager = tf.train.CheckpointManager(
      checkpoint=state, directory=workdir, max_to_keep=5)

  if ckpt_manager.latest_checkpoint:
    state.restore(ckpt_manager.latest_checkpoint)
    logging.info("Restored from %s", ckpt_manager.latest_checkpoint)
  else:
    logging.info("Initializing from scratch.")
  initial_step = state.global_step.numpy().item()

  learning_rate_fn = functools.partial(
      get_learning_rate,
      base_learning_rate=base_learning_rate,
      steps_per_epoch=steps_per_epoch,
      num_epochs=config.num_epochs,
      warmup_epochs=config.warmup_epochs)

  writer = metric_writers.create_default_writer(workdir)
  writer.write_hparams(dict(config))

  logging.info("Starting training loop at step %d.", initial_step)
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=num_train_steps, writer=writer)
  with metric_writers.ensure_flushes(writer):
    for step in range(initial_step, num_train_steps + 1):
      state.model.trainable = True

      # `step` is a Python integer. `global_step` is a TF variable on the
      # GPU/TPU devices.
      is_last_step = step == num_train_steps

      train_step(state, train_iter, config.weight_decay, learning_rate_fn,
                 do_distill, distill_params, strategy)

      state.train_metrics.update_state_lr(learning_rate_fn(
          state.global_step.numpy().item()))

      # Quick indication that training is happening.
      logging.log_first_n(logging.INFO, "Finished training step %d.", 5, step)
      report_progress(step)

      if step == initial_step:
        parameter_overview.log_parameter_overview(state.model)

      if step % config.log_loss_every_steps == 0 or is_last_step:
        writer.write_scalars(step, state.train_metrics.result())
        state.train_metrics.reset_states()
        state.train_metrics.reset_lr()

      if step % config.eval_every_steps == 0 or is_last_step:
        state.model.trainable = False
        if  config.dataset == "imagenet-lt":
          evaluate(state, val_ds, state.val_metrics, strategy)
          writer.write_scalars(step, state.val_metrics.result())
          logging.info("Num val images %d",
                       state.val_metrics.accuracy.count.numpy())

        evaluate(state, test_ds, state.test_metrics, strategy)
        writer.write_scalars(step, state.test_metrics.result())

        logging.info("Num test images %d",
                     state.test_metrics.accuracy.count.numpy())

      if step % config.checkpoint_every_steps == 0 or is_last_step:
        checkpoint_path = ckpt_manager.save(step)
        logging.info("Saved checkpoint %s", checkpoint_path)

  logging.info("Finishing training at step %d", step)
  logging.info("Saving the final weights")
  file_path = "%s/final_weights" % workdir
  state.model.save_weights(file_path, save_format="tf")
