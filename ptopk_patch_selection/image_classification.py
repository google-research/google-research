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
"""Main file for image classification."""

from absl import app
from absl import flags
from absl import logging
from clu import platform
from flax.deprecated import nn
import jax
import jax.numpy as jnp
from lib import data
from lib import models
from lib import utils
import lib.classification_utils as classification_lib
from lib.layers import sample_patches
import ml_collections
import ml_collections.config_flags as config_flags
import optax
import tensorflow as tf

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work unit directory.")


class ClassificationModule(nn.Module):
  """A module that does classification."""

  def apply(self, x, config,
            num_classes, train = True):
    """Creates a model definition."""

    if config.get("append_position_to_input", False):
      b, h, w, _ = x.shape
      coords = utils.create_grid([h, w], value_range=(0., 1.))
      x = jnp.concatenate([x, coords[jnp.newaxis, Ellipsis].repeat(b, axis=0)],
                          axis=-1)

    if config.model.lower() == "cnn":
      h = models.SimpleCNNImageClassifier(x)
      h = nn.relu(h)
      stats = None
    elif config.model.lower() == "resnet":
      smallinputs = config.get("resnet.small_inputs", False)
      blocks = config.get("resnet.blocks", [3, 4, 6, 3])
      h = models.ResNet(
          x, train=train, block_sizes=blocks, small_inputs=smallinputs)
      h = jnp.mean(h, axis=[1, 2])   # global average pool
      stats = None
    elif config.model.lower() == "resnet18":
      h = models.ResNet18(x, train=train)
      h = jnp.mean(h, axis=[1, 2])   # global average pool
      stats = None
    elif config.model.lower() == "resnet50":
      h = models.ResNet50(x, train=train)
      h = jnp.mean(h, axis=[1, 2])   # global average pool
      stats = None
    elif config.model.lower() == "ats-traffic":
      h = models.ATSFeatureNetwork(x, train=train)
      stats = None
    elif config.model.lower() == "patchnet":
      feature_network = {
          "resnet18": models.ResNet18,
          "resnet18-fourth": models.ResNet.partial(
              num_filters=16,
              block_sizes=(2, 2, 2, 2),
              block=models.BasicBlock),
          "resnet50": models.ResNet50,
          "ats-traffic": models.ATSFeatureNetwork,
      }[config.feature_network.lower()]

      selection_method = sample_patches.SelectionMethod(config.selection_method)
      selection_method_kwargs = {}
      if selection_method is sample_patches.SelectionMethod.SINKHORN_TOPK:
        selection_method_kwargs = config.sinkhorn_topk_kwargs
      if selection_method is sample_patches.SelectionMethod.PERTURBED_TOPK:
        selection_method_kwargs = config.perturbed_topk_kwargs

      h, stats = sample_patches.PatchNet(
          x,
          patch_size=config.patch_size,
          k=config.k,
          downscale=config.downscale,
          scorer_has_se=config.get("scorer_has_se", False),
          selection_method=config.selection_method,
          selection_method_kwargs=selection_method_kwargs,
          selection_method_inference=config.get("selection_method_inference",
                                                None),
          normalization_str=config.normalization_str,
          aggregation_method=config.aggregation_method,
          aggregation_method_kwargs=config.get("aggregation_method_kwargs", {}),
          append_position_to_input=config.get("append_position_to_input",
                                              False),
          feature_network=feature_network,
          use_iterative_extraction=config.use_iterative_extraction,
          hard_topk_probability=config.get("hard_topk_probability", 0.),
          random_patch_probability=config.get("random_patch_probability", 0.),
          train=train)
      stats["x"] = x
    else:
      raise RuntimeError(
          "Unknown classification model type: %s" % config.model.lower())
    out = nn.Dense(h, num_classes, name="final")
    return out, stats


def create_optimizer(config):
  """Creates the optimizer associated to a config."""
  ops = []

  # Gradient clipping either by norm `gradient_norm_clip` or by absolute value
  # `gradient_value_clip`.
  if "gradient_clip" in config:
    raise ValueError("'gradient_clip' is deprecated, please use "
                     "'gradient_norm_clip'.")
  assert not ("gradient_norm_clip" in config and
              "gradient_value_clip" in config), (
                  "Gradient clipping by norm and by value are exclusive.")

  if "gradient_norm_clip" in config:
    ops.append(optax.clip_by_global_norm(config.gradient_norm_clip))
  if "gradient_value_clip" in config:
    ops.append(optax.clip(config.gradient_value_clip))

  # Define the learning rate schedule.
  schedule_fn = utils.get_optax_schedule_fn(
      warmup_ratio=config.get("warmup_ratio", 0.),
      num_train_steps=config.num_train_steps,
      decay=config.get("learning_rate_step_decay", 1.0),
      decay_at_steps=config.get("learning_rate_decay_at_steps", []),
      cosine_decay_schedule=config.get("cosine_decay", False))

  schedule_ops = [optax.scale_by_schedule(schedule_fn)]

  # Scale some parameters matching a regex by a multiplier. Config field
  # `scaling_by_regex` is a list of pairs (regex: str, multiplier: float).
  scaling_by_regex = config.get("scaling_learning_rate_by_regex", [])
  for regex, multiplier in scaling_by_regex:
    logging.info("Learning rate is scaled by %f for parameters matching '%s'",
                 multiplier, regex)
    schedule_ops.append(utils.scale_selected_parameters(regex, multiplier))
  schedule_optimizer = optax.chain(*schedule_ops)

  if config.optimizer.lower() == "adam":
    optimizer = optax.adam(config.learning_rate)
    ops.append(optimizer)
    ops.append(schedule_optimizer)
  elif config.optimizer.lower() == "sgd":
    ops.append(schedule_optimizer)
    optimizer = optax.sgd(config.learning_rate, momentum=config.momentum)
    ops.append(optimizer)
  else:
    raise NotImplementedError("Invalid optimizer: {}".format(
        config.optimizer))

  if "weight_decay" in config and config.weight_decay > 0.:
    ops.append(utils.decoupled_weight_decay(
        decay=config.weight_decay, step_size_fn=schedule_fn))

  # Freeze parameters that match the given regexes (if any).
  freeze_weights_regexes = config.get("freeze_weights_regex", []) or []
  if isinstance(freeze_weights_regexes, str):
    freeze_weights_regexes = [freeze_weights_regexes]
  for reg in freeze_weights_regexes:
    ops.append(utils.freeze(reg))

  return optax.chain(*ops)


def train_and_evaluate(config, workdir):
  """Runs a training and evaluation loop.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint, training will be resumed from the latest checkpoint.

  Returns:
    Training state.
  """
  rng = jax.random.PRNGKey(config.seed)
  rng, data_rng = jax.random.split(rng)

  # Make sure config defines num_epochs and num_train_steps appropriately.
  utils.check_epochs_and_steps(config)

  train_preprocessing_fn, eval_preprocessing_fn = data.parse_preprocessing_strings(
      config.get("train_preprocess_str", ""),
      config.get("eval_preprocess_str", ""))

  assert config.batch_size % jax.local_device_count() == 0, (
      f"Batch size ({config.batch_size}) should be divisible by number of "
      f"devices ({jax.local_device_count()}).")

  per_device_batch_size = config.batch_size // jax.local_device_count()
  train_ds, eval_ds, num_classes = data.get_dataset(
      config.dataset,
      per_device_batch_size,
      data_rng,
      train_preprocessing_fn=train_preprocessing_fn,
      eval_preprocessing_fn=eval_preprocessing_fn,
      **config.get("data", {}))

  module = ClassificationModule.partial(config=config, num_classes=num_classes)

  optimizer = create_optimizer(config)

  # Enables relevant statistics aggregator.
  stats_aggregators = []

  train_metrics_dict = {
      "train_loss": classification_lib.cross_entropy,
      "train_accuracy": classification_lib.accuracy
  }
  eval_metrics_dict = {
      "eval_loss": classification_lib.cross_entropy,
      "eval_accuracy": classification_lib.accuracy
  }
  loss_fn = classification_lib.cross_entropy

  def loss_from_stats(field, multiplier):
    return lambda logits, labels, stats: multiplier * stats[field]

  # Add some regularizer to the loss if needed.
  if (config.model == "patchnet" and
      config.selection_method not in [sample_patches.SelectionMethod.HARD_TOPK,
                                      sample_patches.SelectionMethod.RANDOM]):
    entropy_regularizer = config.get("entropy_regularizer", 0.)
    entropy_before_normalization = config.get("entropy_before_normalization",
                                              False)

    stat_field = "entropy"
    if entropy_before_normalization:
      stat_field = "entropy_before_normalization"

    if entropy_regularizer != 0.:
      logging.info("Add entropy regularizer %s normalization to the loss %f.",
                   "before" if entropy_before_normalization else "after",
                   entropy_regularizer)
      loss_fn = [loss_fn, loss_from_stats(stat_field, entropy_regularizer)]

    def entropy_aggregator(stats):
      return {stat_field: stats[stat_field],}
    stats_aggregators.append(entropy_aggregator)

  def add_image_prefix(image_aggregator):
    def aggregator(stats):
      d = image_aggregator(stats)
      return {f"image_{k}": v for k, v in d.items()}
    return aggregator

  if config.model == "patchnet" and config.get("log_images", True):
    @add_image_prefix
    def plot_patches(stats):
      keys = ["extracted_patches", "x", "scores"]
      return {k: stats[k] for k in keys if k in stats}

    stats_aggregators.append(plot_patches)

  state = classification_lib.training_loop(
      module=module,
      rng=rng,
      train_ds=train_ds,
      eval_ds=eval_ds,
      loss_fn=loss_fn,
      optimizer=optimizer,
      train_metrics_dict=train_metrics_dict,
      eval_metrics_dict=eval_metrics_dict,
      stats_aggregators=stats_aggregators,
      config=config,
      workdir=workdir)
  return state


def main(argv):
  del argv

  # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], "GPU")


  logging.info("JAX host: %d / %d", jax.host_id(), jax.host_count())
  logging.info("JAX devices: %r", jax.devices())

  # Add a note so that we can tell which task is which JAX host.
  # (Borg task 0 is not guaranteed to be host 0)
  platform.work_unit().set_task_status(
      f"host_id: {jax.host_id()}, host_count: {jax.host_count()}")
  platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                       FLAGS.workdir, "workdir")

  state = train_and_evaluate(FLAGS.config, FLAGS.workdir)
  del state


if __name__ == "__main__":
  flags.mark_flags_as_required(["config", "workdir"])
  app.run(main)
