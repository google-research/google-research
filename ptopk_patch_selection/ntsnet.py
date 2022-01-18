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
"""NTS-Net adapted for perturbed top-k.

Based on the original PyTorch code
https://github.com/yangze0930/NTS-Net/blob/master/core/model.py
"""

import enum
import functools
import math
from typing import List, Tuple

from absl import app
from absl import flags
from absl import logging
import chex
from clu import platform
import einops
from flax.deprecated import nn
import jax
import jax.numpy as jnp
import ml_collections
import ml_collections.config_flags as config_flags
from off_the_grid.lib import data
from off_the_grid.lib import models
from off_the_grid.lib import utils
import off_the_grid.lib.classification_utils as classification_lib
from off_the_grid.lib.layers import sample_patches
from off_the_grid.lib.layers import transformer
import optax
import tensorflow as tf


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work unit directory.")
NUM_CLASSES = 200

ANCHORS_SETTINGS = (
    dict(
        layer="p3",
        stride=32,
        size=48,
        scale=[2**(1. / 3.), 2**(2. / 3.)],
        aspect_ratio=[0.667, 1, 1.5]),  # Anchors 0-5
    dict(
        layer="p4",
        stride=64,
        size=96,
        scale=[2**(1. / 3.), 2**(2. / 3.)],
        aspect_ratio=[0.667, 1, 1.5]),  # Anchors 6-11
    dict(
        layer="p5",
        stride=128,
        size=192,
        scale=[1, 2**(1. / 3.), 2**(2. / 3.)],
        aspect_ratio=[0.667, 1, 1.5]),  # Anchors 12-20
)


class Communication(str, enum.Enum):
  NONE = "none"
  SQUEEZE_EXCITE_D = "squeeze_excite_d"
  SQUEEZE_EXCITE_X = "squeeze_excite_x"
  TRANSFORMER = "transformer"


def zeroone(scores, x_min, x_max):
  """Normalize values to lie between [0, 1]."""
  return [(x - x_min) / (x_max - x_min + 1e-5) for x in scores]


class ProposalNet(nn.Module):
  """FPN inspired scorer module."""

  def apply(self, x,
            communication = Communication.NONE,
            train = True):
    """Forward pass."""
    batch_size = x.shape[0]

    if communication is Communication.SQUEEZE_EXCITE_X:
      x = sample_patches.SqueezeExciteLayer(x)
    # end if squeeze excite x

    d1 = nn.relu(nn.Conv(
        x, 128, kernel_size=(3, 3), strides=(1, 1), bias=True, name="down1"))
    d2 = nn.relu(nn.Conv(
        d1, 128, kernel_size=(3, 3), strides=(2, 2), bias=True, name="down2"))
    d3 = nn.relu(nn.Conv(
        d2, 128, kernel_size=(3, 3), strides=(2, 2), bias=True, name="down3"))

    if communication is Communication.SQUEEZE_EXCITE_D:
      d1_flatten = einops.rearrange(d1, "b h w c -> b (h w) c")
      d2_flatten = einops.rearrange(d2, "b h w c -> b (h w) c")
      d3_flatten = einops.rearrange(d3, "b h w c -> b (h w) c")

      nd1 = d1_flatten.shape[1]
      nd2 = d2_flatten.shape[1]

      d_together = jnp.concatenate([d1_flatten, d2_flatten, d3_flatten], axis=1)

      num_channels = d_together.shape[-1]
      y = d_together.mean(axis=1)
      y = nn.Dense(y, features=num_channels // 4, bias=False)
      y = nn.relu(y)
      y = nn.Dense(y, features=num_channels, bias=False)
      y = nn.sigmoid(y)

      d_together = d_together * y[:, None, :]

      # split and reshape
      d1 = d_together[:, :nd1].reshape(d1.shape)
      d2 = d_together[:, nd1:nd1+nd2].reshape(d2.shape)
      d3 = d_together[:, nd1+nd2:].reshape(d3.shape)

    elif communication is Communication.TRANSFORMER:
      d1_flatten = einops.rearrange(d1, "b h w c -> b (h w) c")
      d2_flatten = einops.rearrange(d2, "b h w c -> b (h w) c")
      d3_flatten = einops.rearrange(d3, "b h w c -> b (h w) c")

      nd1 = d1_flatten.shape[1]
      nd2 = d2_flatten.shape[1]

      d_together = jnp.concatenate([d1_flatten, d2_flatten, d3_flatten], axis=1)

      positional_encodings = self.param(
          "scale_ratio_position_encodings",
          shape=(1,) + d_together.shape[1:],
          initializer=jax.nn.initializers.normal(1. / d_together.shape[-1]))
      d_together = transformer.Transformer(
          d_together + positional_encodings,
          num_layers=2,
          num_heads=8,
          is_training=train)

      # split and reshape
      d1 = d_together[:, :nd1].reshape(d1.shape)
      d2 = d_together[:, nd1:nd1+nd2].reshape(d2.shape)
      d3 = d_together[:, nd1+nd2:].reshape(d3.shape)

    t1 = nn.Conv(
        d1, 6, kernel_size=(1, 1), strides=(1, 1), bias=True, name="tidy1")
    t2 = nn.Conv(
        d2, 6, kernel_size=(1, 1), strides=(1, 1), bias=True, name="tidy2")
    t3 = nn.Conv(
        d3, 9, kernel_size=(1, 1), strides=(1, 1), bias=True, name="tidy3")

    raw_scores = (jnp.split(t1, 6, axis=-1) +
                  jnp.split(t2, 6, axis=-1) +
                  jnp.split(t3, 9, axis=-1))

    # The following is for normalization.
    t = jnp.concatenate((jnp.reshape(t1, [batch_size, -1]),
                         jnp.reshape(t2, [batch_size, -1]),
                         jnp.reshape(t3, [batch_size, -1])), axis=1)
    t_min = jnp.reshape(jnp.min(t, axis=-1), [batch_size, 1, 1, 1])
    t_max = jnp.reshape(jnp.max(t, axis=-1), [batch_size, 1, 1, 1])
    normalized_scores = zeroone(raw_scores, t_min, t_max)

    stats = {
        "scores": normalized_scores,
        "raw_scores": t,
    }
    # removes the split dimension. scores are now b x h' x w' shaped
    normalized_scores = [s.squeeze(-1) for s in normalized_scores]

    return normalized_scores, stats


def extract_weighted_patches(x,
                             weights,
                             kernel,
                             stride,
                             padding):
  """Weighted average of patches using jax.lax.scan."""
  logging.info("recompiling for kernel=%s and stride=%s and padding=%s", kernel,
               stride, padding)
  x = jnp.pad(x, ((0, 0),
                  (padding[0], padding[0] + kernel[0]),
                  (padding[1], padding[1] + kernel[1]),
                  (0, 0)))
  batch_size, _, _, channels = x.shape
  _, k, weights_h, weights_w = weights.shape

  def accumulate_patches(acc, index_i_j):
    i, j = index_i_j
    patch = jax.lax.dynamic_slice(
        x,
        (0, i * stride[0], j * stride[1], 0),
        (batch_size, kernel[0], kernel[1], channels))
    weight = weights[:, :, i, j]

    weighted_patch = jnp.einsum("bk, bijc -> bkijc", weight, patch)
    acc += weighted_patch
    return acc, None

  indices = jnp.stack(
      jnp.meshgrid(jnp.arange(weights_h), jnp.arange(weights_w), indexing="ij"),
      axis=-1)
  indices = indices.reshape((-1, 2))

  init_patches = jnp.zeros((batch_size, k, kernel[0], kernel[1], channels))
  patches, _ = jax.lax.scan(accumulate_patches, init_patches, indices)

  return patches


def weighted_anchor_aggregator(x, weights):
  """Given a tensor of weights per anchor computes the weighted average."""
  counter = 0
  all_sub_aggregates = []

  for anchor_info in ANCHORS_SETTINGS:
    stride = anchor_info["stride"]
    size = anchor_info["size"]
    for scale in anchor_info["scale"]:
      for aspect_ratio in anchor_info["aspect_ratio"]:
        kernel_size = (
            int(size * scale / float(aspect_ratio) ** 0.5),
            int(size * scale * float(aspect_ratio) ** 0.5))
        padding = (
            math.ceil((kernel_size[0] - stride) / 2.),
            math.ceil((kernel_size[1] - stride) / 2.))
        aggregate = extract_weighted_patches(
            x, weights[counter], kernel_size, (stride, stride), padding)
        aggregate = jnp.reshape(aggregate,
                                [-1, kernel_size[0], kernel_size[1], 3])
        aggregate_224 = jax.image.resize(aggregate,
                                         [aggregate.shape[0], 224, 224, 3],
                                         "bilinear")
        all_sub_aggregates.append(aggregate_224)
        counter += 1

  return jnp.sum(jnp.stack(all_sub_aggregates, axis=0), axis=0)


class AttentionNet(nn.Module):
  """The complete NTS-Net model using perturbed top-k."""

  def apply(self,
            x,
            config,
            num_classes,
            train = True):
    """Creates a model definition."""
    b, c = x.shape[0], x.shape[3]
    k = config.k
    sigma = config.ptopk_sigma
    num_samples = config.ptopk_num_samples

    sigma *= self.state("sigma_mutiplier", shape=(),
                        initializer=nn.initializers.ones).value

    stats = {"x": x, "sigma": sigma}

    feature_extractor = models.ResNet50.shared(train=train, name="ResNet_0")

    rpn_feature = feature_extractor(x)
    rpn_scores, rpn_stats = ProposalNet(
        jax.lax.stop_gradient(rpn_feature),
        communication=Communication(config.communication),
        train=train)
    stats.update(rpn_stats)

    # rpn_scores are a list of score images. We keep track of the structure
    # because it is used in the aggregation step later-on.
    rpn_scores_shapes = [s.shape for s in rpn_scores]
    rpn_scores_flat = jnp.concatenate(
        [jnp.reshape(s, [b, -1]) for s in rpn_scores], axis=1)
    top_k_indicators = sample_patches.select_patches_perturbed_topk(
        rpn_scores_flat,
        k=k,
        sigma=sigma,
        num_samples=num_samples)
    top_k_indicators = jnp.transpose(top_k_indicators, [0, 2, 1])
    offset = 0
    weights = []
    for sh in rpn_scores_shapes:
      cur = top_k_indicators[:, :, offset:offset + sh[1] * sh[2]]
      cur = jnp.reshape(cur, [b, k, sh[1], sh[2]])
      weights.append(cur)
      offset += sh[1] * sh[2]
    chex.assert_equal(offset, top_k_indicators.shape[-1])

    part_imgs = weighted_anchor_aggregator(x, weights)
    chex.assert_shape(part_imgs, (b * k, 224, 224, c))
    stats["part_imgs"] = jnp.reshape(part_imgs, [b, k*224, 224, c])

    part_features = feature_extractor(part_imgs)
    part_features = jnp.mean(part_features, axis=[1, 2])  # GAP the spatial dims

    part_features = nn.dropout(  # features from parts
        jnp.reshape(part_features, [b * k, 2048]),
        0.5,
        deterministic=not train,
        rng=nn.make_rng())
    features = nn.dropout(  # features from whole image
        jnp.reshape(jnp.mean(rpn_feature, axis=[1, 2]), [b, -1]),
        0.5,
        deterministic=not train,
        rng=nn.make_rng())

    # Mean pool all part features, add it to features and predict logits.
    concat_out = jnp.mean(jnp.reshape(part_features, [b, k, 2048]),
                          axis=1) + features
    concat_logits = nn.Dense(concat_out, num_classes)
    raw_logits = nn.Dense(features, num_classes)
    part_logits = jnp.reshape(nn.Dense(part_features, num_classes), [b, k, -1])

    all_logits = {
        "raw_logits": raw_logits,
        "concat_logits": concat_logits,
        "part_logits": part_logits,
    }
    # add entropy into it for entropy regularization.
    stats["rpn_scores_entropy"] = jax.scipy.special.entr(
        jax.nn.softmax(stats["raw_scores"])).sum(axis=1).mean(axis=0)
    return all_logits, stats


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

  if "weight_decay_coupled" in config and config.weight_decay_coupled > 0.:
    # it calls decoupled weight decay before applying optimizer which is
    # coupled weight decay. :D
    ops.append(utils.decoupled_weight_decay(
        decay=config.weight_decay_coupled,
        step_size_fn=lambda x: jnp.ones([], dtype=jnp.float32)))

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


def cross_entropy(logits, labels):
  """Basic corss entropy loss."""
  logp = jax.nn.log_softmax(logits)
  loglik = jnp.take_along_axis(logp, labels[:, None], axis=1)
  return -jnp.mean(loglik)


def ntsnet_loss(logits_dict, labels, stats, config):
  """Customized cross entropy loss for dictionary of logits."""
  raw_logits = logits_dict["raw_logits"]
  concat_logits = logits_dict["concat_logits"]
  part_logits = logits_dict["part_logits"]

  raw_loss = cross_entropy(raw_logits, labels)
  concat_loss = cross_entropy(concat_logits, labels)

  k = part_logits.shape[1]
  num_classes = part_logits.shape[2]
  labels_per_part = jnp.tile(jnp.expand_dims(labels, axis=1), [1, k])
  part_loss = cross_entropy(
      jnp.reshape(part_logits, [-1, num_classes]),
      jnp.reshape(labels_per_part, [-1,]))

  reg = config.entropy_regularizer * rpn_scores_entropy(
      logits_dict, labels, stats)

  return raw_loss + concat_loss + part_loss + reg


def accuracy(logits_dict, labels, stats):
  """Customized accuracy metric for dictionary of logits."""
  del stats
  logits = logits_dict["concat_logits"]
  predictions = jnp.argmax(logits, axis=-1)
  return jnp.mean(predictions == labels)


def cross_entropy_raw_logits(logits_dict, labels, stats):
  """Customized cross entropy loss for dictionary of logits."""
  del stats
  return cross_entropy(logits_dict["raw_logits"], labels)


def cross_entropy_concat_logits(logits_dict, labels, stats):
  """Customized cross entropy loss for dictionary of logits."""
  del stats
  return cross_entropy(logits_dict["concat_logits"], labels)


def cross_entropy_part_logits(logits_dict, labels, stats):
  """Customized cross entropy loss for dictionary of logits."""
  del stats
  part_logits = logits_dict["part_logits"]
  k = part_logits.shape[1]
  num_classes = part_logits.shape[2]
  labels_per_part = jnp.tile(jnp.expand_dims(labels, axis=1), [1, k])
  part_loss = cross_entropy(
      jnp.reshape(part_logits, [-1, num_classes]),
      jnp.reshape(labels_per_part, [-1,]))
  return part_loss


def rpn_scores_entropy(logits_dict, labels, stats):
  """Entropy."""
  del logits_dict
  del labels
  return stats["rpn_scores_entropy"]


def train_and_evaluate(config, workdir):
  """Runs a training and evaluation loop.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.

  Returns:
    Training state.
  """
  rng = jax.random.PRNGKey(config.seed)
  rng, data_rng = jax.random.split(rng)

  # Make sure config defines num_epochs and num_train_steps appropriately.
  utils.check_epochs_and_steps(config)

  # Check that perturbed-topk is selection method.
  assert config.selection_method == "perturbed-topk", (
      "ntsnet only supports perturbed-topk as selection method. Got: {}".format(
          config.selection_method))

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

  module = AttentionNet.partial(config=config, num_classes=num_classes)

  optimizer = create_optimizer(config)

  loss_fn = functools.partial(ntsnet_loss, config=config)
  train_metrics_dict = {
      "train_loss": loss_fn,
      "train_loss_raw": cross_entropy_raw_logits,
      "train_loss_concat": cross_entropy_concat_logits,
      "train_loss_part": cross_entropy_part_logits,
      "train_accuracy": accuracy,
      "train_rpn_scores_entropy": rpn_scores_entropy,
  }
  eval_metrics_dict = {
      "eval_loss": loss_fn,
      "eval_loss_raw": cross_entropy_raw_logits,
      "eval_loss_concat": cross_entropy_concat_logits,
      "eval_loss_part": cross_entropy_part_logits,
      "eval_accuracy": accuracy,
      "eval_rpn_scores_entropy": rpn_scores_entropy,
  }

  # Enables relevant statistics aggregator.
  stats_aggregators = []

  def add_image_prefix(image_aggregator):
    def aggregator(stats):
      d = image_aggregator(stats)
      return {f"image_{k}": v for k, v in d.items()}
    return aggregator

  if config.get("log_images", True):
    @add_image_prefix
    def plot_patches(stats):
      d = {
          "part_imgs": (stats["part_imgs"] + 1.0) / 2.0,
          "x": (stats["x"] + 1.0) / 2.0
      }
      for i, sc in enumerate(stats["scores"]):
        d[f"scores_{i}"] = sc
      return d

    stats_aggregators.append(plot_patches)

  stats_aggregators.append(lambda x: {"sigma": x["sigma"]})

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

  # Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], "GPU")

  state = train_and_evaluate(FLAGS.config, FLAGS.workdir)
  del state


if __name__ == "__main__":
  flags.mark_flags_as_required(["config", "workdir"])
  app.run(main)
