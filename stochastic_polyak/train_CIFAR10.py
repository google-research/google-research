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

"""Train a CNN on MNIST using Stochastic Polyak variants."""

# pytype: disable=wrong-keyword-args

import itertools
from absl import app
from absl import flags
from absl import logging

from flax.metrics import tensorboard
import jax
import jax.numpy as jnp
from jaxopt import loss
from ml_collections import config_flags
## Data loaders

## Resnet model definition built on flax
from stochastic_polyak import models
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


def cross_entropy_loss(preds, labels):
  one_hot_labels = jax.nn.one_hot(labels, num_classes=10)
  log_softmax_preds = jax.nn.log_softmax(preds)
  return -jnp.mean(jnp.sum(one_hot_labels * log_softmax_preds, axis=-1))


def compute_metrics(preds, labels):
  loss_value = cross_entropy_loss(
      preds, labels
  )
  accuracy = jnp.mean(jnp.argmax(preds, -1) == labels)
  metrics = {
      "loss": loss_value,
      "accuracy": accuracy,
  }
  return metrics


def eval_step(params, data):
  preds = models.ResNet18(num_classes=10).apply(
      params, data["image"], train=False, mutable=False)
  return compute_metrics(preds=preds, labels=data["label"])


def eval_model(params, data):
  metrics = eval_step(params, data)
  metrics = jax.device_get(metrics)
  summary = jax.tree_map(lambda x: x.item(), metrics)
  return summary["loss"], summary["accuracy"]


def loss_fun(params, data):
  preds, new_batch_stats = models.ResNet18(num_classes=10).apply(
      params, data["image"], mutable=["batch_stats"])
  metrics = compute_metrics(preds=preds, labels=data["label"])
  return metrics["loss"], (new_batch_stats, preds)


def train_epoch(config, solver, params, state, train_ds, rng):
  """Train the model for one pass over the dataset."""
  ## Prepare batches
  train_ds_size = len(train_ds["image"])
  steps_per_epoch = train_ds_size // config.batch_size
  perms = jax.random.permutation(rng, train_ds_size)
  perms = perms[: steps_per_epoch * config.batch_size]
  perms = perms.reshape((steps_per_epoch, config["batch_size"]))
  ## Run one epoch
  for perm in itertools.islice(perms, 0, len(perms)):
    batch = {k: v[perm, Ellipsis] for k, v in train_ds.items()}
    params, state = solver.update(params=params, state=state, data=batch)
  return params, state


def train_and_evaluate(config, workdir):
  """Execute model training and evaluation loop.

  Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Directory where the tensorboard summaries are written to.
  """
  ## get random seed
  rng = jax.random.PRNGKey(0)
  ## Get data
  train_ds, test_ds = get_datasets("cifar10")

  ## Initializing model and infering dimensions of layers from one example batch
  model = models.ResNet18(num_classes=10)
  init_params = model.init(
      rng, jnp.ones((1, 32, 32, 3))
  )  # figure this shape out automatically ?
  params = init_params

  solver, solver_param_name = get_solver(
      FLAGS, config, loss_fun, losses=loss_fun)  # losses is not defined yet!
  params, state = solver.init(params)

  ## Path to dump results
  dumpath = create_dumpfile(config, solver_param_name, workdir, "cifar10")

  summary_writer = tensorboard.SummaryWriter(dumpath)
  summary_writer.hparams(dict(config))

  for epoch in range(1, config.num_epochs + 1):
    rng, _ = jax.random.split(rng)
    params, state = train_epoch(
        config, solver, params, state, train_ds, rng
    )
    test_loss, test_accuracy = eval_model(params, test_ds)
    train_loss, train_accuracy = eval_model(params, train_ds)
    print("eval epoch: %d, loss: %.4f, accuracy: %.2f", epoch, test_loss,
          test_accuracy * 100)
    print("train epoch: %d, train_loss: %.4f, train_accuracy: %.2f", epoch,
          train_loss, train_accuracy * 100)
    logging.info("eval epoch: %d, loss: %.4f, accuracy: %.2f", epoch, test_loss,
                 test_accuracy * 100)
    summary_writer.scalar("train_loss", train_loss, epoch)
    summary_writer.scalar("test_loss", loss, epoch)
    summary_writer.scalar("train_accuracy", train_accuracy, epoch)
    summary_writer.scalar("test_accuracy", test_accuracy, epoch)

  summary_writer.flush()


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments!!")
  train_and_evaluate(FLAGS.config, FLAGS.workdir)


if __name__ == "__main__":
  app.run(main)
