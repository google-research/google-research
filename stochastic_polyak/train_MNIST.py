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

"""Train a CNN on MNIST using Stochastic Polyak variants."""

# pytype: disable=wrong-keyword-args

import itertools
from absl import app
from absl import flags
from absl import logging
from clu import platform

from flax import linen as nn
from flax.metrics import tensorboard
import jax
import jax.numpy as jnp
from jaxopt import loss
import ml_collections
from ml_collections import config_flags
import numpy as np
import tensorflow as tf


from stochastic_polyak.get_solver import get_solver
from stochastic_polyak.utils import create_dumpfile
from stochastic_polyak.utils import get_datasets


FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", "/tmp/stochastic_polyak/",
                    "Parent directory to store model data.")
config_flags.DEFINE_config_file(
    "config",
    "stochastic_polyak/configs/sps.py",
    "File path to the training hyperparameter configuration.",
    lock_config=True)
flags.DEFINE_integer("max_steps_per_epoch", -1,
                     "Maximum number of steps in an epoch.")
flags.DEFINE_float(
    "slack_lmbda", -1,
    "The lmbda regularization parameter of the slack formulation.")
flags.DEFINE_float("slack_delta", -1,
                   "The delta dampening parameter of the slack formulation.")
flags.DEFINE_float("momentum", 0.0, "The momentum parameter.")
flags.DEFINE_integer(
    "choose_update", 1,
    "What solver to use in the SSPS methods. Can take values [1, 2, 3, 4, 5] SSPS."
)


class CNN(nn.Module):
  """A simple CNN model."""

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)
    return x


logistic_loss = jax.vmap(loss.multiclass_logistic_loss)


def compute_metrics(logits, labels):
  loss_value = jnp.mean(logistic_loss(labels, logits))
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  metrics = {
      "loss": loss_value,
      "accuracy": accuracy,
  }
  return metrics


@jax.jit
def losses(params, data):
  logits = CNN().apply({"params": params}, data["image"])
  loss_values = logistic_loss(data["label"], logits)
  metrics = compute_metrics(logits=logits, labels=data["label"])
  return loss_values, metrics


@jax.jit
def loss_fun(params, data):
  logits = CNN().apply({"params": params}, data["image"])
  metrics = compute_metrics(logits=logits, labels=data["label"])
  return metrics["loss"], metrics


@jax.jit
def eval_step(params, data):
  logits = CNN().apply({"params": params}, data["image"])
  return compute_metrics(logits=logits, labels=data["label"])


def train_epoch(config, solver, params, state, train_ds, epoch, rng):
  """Train for a single epoch."""

  # Prepare batch.
  train_ds_size = len(train_ds["image"])
  steps_per_epoch = train_ds_size // config.batch_size
  perms = jax.random.permutation(rng, len(train_ds["image"]))
  perms = perms[:steps_per_epoch * config.batch_size]  # skip incomplete batch
  perms = perms.reshape((steps_per_epoch, config.batch_size))

  # Run one epoch.
  batch_metrics = []

  if FLAGS.max_steps_per_epoch == -1:
    max_steps = len(perms)
  else:
    max_steps = FLAGS.max_steps_per_epoch

  for perm in itertools.islice(perms, 0, max_steps):
    batch = {k: v[perm, Ellipsis] for k, v in train_ds.items()}
    if config.solver == "SPS" or config.solver == "SGD":
      params, state = solver.update(params=params, state=state, data=batch)
    else:
      params, state = solver.update(params=params, state=state, data=batch,
                                    epoch=epoch)
    batch_metrics.append(state.aux)

  # compute mean of metrics across each batch in epoch.
  batch_metrics_np = jax.device_get(batch_metrics)
  epoch_metrics_np = {
      k: np.mean([metrics[k] for metrics in batch_metrics_np
                 ]) for k in batch_metrics_np[0]
  }

  logging.info("train epoch: %d, loss: %.4f, accuracy: %.2f", epoch,
               epoch_metrics_np["loss"], epoch_metrics_np["accuracy"] * 100)

  return params, state, epoch_metrics_np


def eval_model(params, test_ds):
  metrics = eval_step(params, test_ds)
  metrics = jax.device_get(metrics)
  summary = jax.tree_map(lambda x: x.item(), metrics)
  return summary["loss"], summary["accuracy"]


def train_and_evaluate(config, workdir):
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.
  """
  train_ds, test_ds = get_datasets("mnist")
  # Get solver
  solver, solver_param_name = get_solver(FLAGS, config, loss_fun, losses)

  rng = jax.random.PRNGKey(0)
  rng, init_rng = jax.random.split(rng)

  init_params = CNN().init(init_rng, jnp.ones([1, 28, 28, 1]))["params"]
  params, state = solver.init(init_params)

  # Full path to dump resultss
  dumpath = create_dumpfile(config, solver_param_name, workdir, "mnist")

  summary_writer = tensorboard.SummaryWriter(dumpath)
  summary_writer.hparams(dict(config))

  # Run solver.
  for epoch in range(1, config.num_epochs + 1):
    rng, input_rng = jax.random.split(rng)

    params, state, train_metrics = train_epoch(config, solver, params, state,
                                               train_ds, epoch, input_rng)
    test_loss, test_accuracy = eval_model(params, test_ds)

    print("eval epoch: %d, loss: %.4f, accuracy: %.2f", epoch, test_loss,
          test_accuracy * 100)
    logging.info("eval epoch: %d, loss: %.4f, accuracy: %.2f", epoch, test_loss,
                 test_accuracy * 100)

    summary_writer.scalar("train_loss", train_metrics["loss"], epoch)
    summary_writer.scalar("train_accuracy", train_metrics["accuracy"], epoch)
    summary_writer.scalar("eval_loss", test_loss, epoch)
    summary_writer.scalar("eval_accuracy", test_accuracy, epoch)

  summary_writer.flush()


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  flags.mark_flags_as_required(["workdir"])

  # Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], "GPU")

  logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
  logging.info("JAX local devices: %r", jax.local_devices())

  # Add a note so that we can tell which task is which JAX host.
  # (Depending on the platform task 0 is not guaranteed to be host 0)
  platform.work_unit().set_task_status(f"process_index: {jax.process_index()}, "
                                       f"process_count: {jax.process_count()}")
  platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                       FLAGS.workdir, "workdir")

  train_and_evaluate(FLAGS.config, FLAGS.workdir)


if __name__ == "__main__":
  app.run(main)
