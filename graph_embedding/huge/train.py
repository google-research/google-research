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

"""Train Huge Unsupervised Graph Embedding Model using (optionally) TPUs.

Implementation of
[HUGE: Huge Unsupervised Graph Embeddings with TPUs]
(https://arxiv.org/pdf/2307.14490.pdf).

Main training program. To use with TPUs, will need a single VM TPU machine or
address of a TPU VM.
"""
import functools
import os
from typing import Dict, List, Union

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from graph_embedding.huge import io as huge_io
from graph_embedding.huge import model as huge


_POSITIVE_SAMPLES = flags.DEFINE_string(
    "positive_samples",
    None,
    "Path spec or glob to positive examples tfrecords.",
    required=True,
)

_WALK_LENGTH = flags.DEFINE_integer(
    "walk_length", None, "Walk length.", required=True
)

_NUM_NODES = flags.DEFINE_integer(
    "num_nodes", None, "Number of nodes.", required=True
)

_EMBEDDING_DIM = flags.DEFINE_integer(
    "embedding_dim", 128, "Embedding dimension."
)

_POSITIVE_BATCH_SIZE = flags.DEFINE_integer(
    "positive_batch_size", 512, "Positive batch size."
)

_NUM_NEGS_PER_POS = flags.DEFINE_integer(
    "num_negs_per_pos", 16, "Number of negs per positive."
)

_COSINE_ADJUSTMENT = flags.DEFINE_float(
    "cosine_adjustment", None, "Cosine adjustment parameter."
)

_FEATURE_WEIGHTS = flags.DEFINE_list(
    "feature_weights", None, "Feature weights."
)

_EDGE_SCORE_NORM = flags.DEFINE_float(
    "edge_score_norm", None, "Edge score normalization."
)

_PIPELINE_EXECUTION_WITH_TENSOR_CORE = flags.DEFINE_bool(
    "pipeline_execution_with_tensor_core",
    False,
    "Option to pipeline SparseCore execution with TensorCore. "
    "This option will cause the host machine to pre-fetch "
    "embedding vectors for the next time step while tensorcore "
    "is processing the current step. This may result in stale "
    "embedding vectors being used on the next step. Enabling "
    "this feature can lead to significan performance improvments "
    "with respect to step time but may or may not degrade "
    "performance with respect to embedding quality.",
)

_MODEL_DIR = flags.DEFINE_string(
    "model_dir", None, "Model directory.", required=True
)

_TPU = flags.DEFINE_string(
    "tpu_address",
    "",
    "TPU address. If not provided, will attempt to connect to local TPU.",
)

_FORCE_CPU = flags.DEFINE_bool(
    "force_cpu",
    False,
    "Force CPU training. `--tpu_address` will be ignored if set to True.",
)

_EPOCHS = flags.DEFINE_integer("epochs", 5, "Number of epochs to train.")

_TRAIN_STEPS = flags.DEFINE_integer(
    "train_steps",
    10,
    "The number of steps to train. Must be a multiple of `--num_host_steps`.",
)

_NUM_HOST_STEPS = flags.DEFINE_integer(
    "num_host_steps", 5, "Number of steps to take per host loop."
)

_CHECKPOINT_INTERVAL = flags.DEFINE_integer(
    "checkpoint_interval",
    100,
    "Only save a checkpoint if there have been more than "
    "`checkpoint_interval` training steps since the last checkpoint save. ",
)

_MAX_CHECKPOINTS_TO_KEEP = flags.DEFINE_integer(
    "max_checkpoints_to_keep",
    2,
    (
        "Number of checkpoints to keep. Oldest will be deleted from active set."
        " Set to a negative number to keep all checkpoints. Defaults to 2."
    ),
)

_TF_DATA_SERVICE_ADDRESS = flags.DEFINE_string(
    "tf_data_service_address", None, "Address of a remote tf.data service."
)

_TF_DATA_SERVICE_SHARDING_POLICY = flags.DEFINE_enum_class(
    "tf_data_service_sharding_policy",
    tf.data.experimental.service.ShardingPolicy.OFF,
    tf.data.experimental.service.ShardingPolicy,
    (
        "Sharding policy for TF data service: "
        "https://www.tensorflow.org/api_docs/python/tf/data/experimental/service"
    ),
)

_OPTIMIZER = flags.DEFINE_string(
    "optimizer",
    "sgd",
    "Optimizer name. Currently supporting `sgd` and `warmup_with_poly_decay`.",
)

_OPTIMIZER_KWARGS = flags.DEFINE_list(
    "optimizer_kwargs",
    ["learning_rate:0.01"],
    "Additional key-value pairs for optimizer. Key-value pairs should be split"
    " by the ':' character and delimited by ','. Example 'learning_rate:0.01'",
)

# When using Vertex Tensorboard, the tensorboard will be present as a
# environment variable.
_LOG_DIR = os.environ.get("AIP_TENSORBOARD_LOG_DIR", "")


def _get_opt_kwargs(args):
  """Parse optimizer kwargs list flag."""
  kwargs = {}
  float_kwargs = set([
      "learning_rate",
      "warmup_power",
      "warmup_end_lr",
      "warmup_decay_power",
      "warmup_decay_end_lr",
  ])
  int_kwargs = set(["warmup_steps", "warmup_decay_steps"])
  for arg in args:
    k, v = arg.split(":")
    if k in float_kwargs:
      kwargs[k] = float(v)
    elif k in int_kwargs:
      kwargs[k] = float(v)
    else:
      raise ValueError(f"Unknown optimizer kwarg: {arg}, {k}, {v}")

  return kwargs


def _get_feature_weights():
  if _FEATURE_WEIGHTS.value is None:
    return None

  return [float(v) for v in _FEATURE_WEIGHTS.value]


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  positive_samples_files = tf.io.gfile.glob(_POSITIVE_SAMPLES.value)

  if not positive_samples_files:
    raise ValueError(
        "Did not find positive sample files with glob:"
        f" {_POSITIVE_SAMPLES.value}",
    )

  logging.info("Found positive_sample_files: %s", positive_samples_files)

  if not tf.io.gfile.exists(_MODEL_DIR.value):
    logging.info("Creating model dir: %s", _MODEL_DIR.value)
    tf.io.gfile.mkdir(_MODEL_DIR.value)
  else:
    logging.info("Using model dir: %s", _MODEL_DIR.value)

  if _FORCE_CPU.value:
    logging.info("Forcing CPU training.")
    strategy = huge.initialize_cpu()
  else:
    logging.info("Using TPU: %s", _TPU.value)
    strategy = huge.initialize_tpu(_TPU.value)

  logging.info("Creating dataset.")
  feature_weights = _get_feature_weights()
  logging.info("Feature weights: %s", feature_weights)
  input_fn = functools.partial(
      huge_io.deepwalk_input_fn,
      filenames=positive_samples_files,
      num_nodes=_NUM_NODES.value,
      walk_length=_WALK_LENGTH.value,
      positive_batch_size=_POSITIVE_BATCH_SIZE.value,
      num_negs_per_pos=_NUM_NEGS_PER_POS.value,
      feature_weights=feature_weights,
      edge_score_norm=_EDGE_SCORE_NORM.value,
      tf_data_service_address=_TF_DATA_SERVICE_ADDRESS.value,
      tf_data_service_sharding_policy=_TF_DATA_SERVICE_SHARDING_POLICY.value,
  )

  ds_itr = huge_io.create_distributed_dataset_iterator(strategy, input_fn)

  logging.info("Creating optimizer.")
  optimizer = huge.create_optimizer(
      _OPTIMIZER.value, strategy, **_get_opt_kwargs(_OPTIMIZER_KWARGS.value)
  )

  total_batch_size = huge.compute_total_batch_size(
      _POSITIVE_BATCH_SIZE.value, _NUM_NEGS_PER_POS.value
  )

  logging.info("Creating model.")
  model = huge.huge_model(
      num_nodes=_NUM_NODES.value,
      embedding_dim=_EMBEDDING_DIM.value,
      total_batch_size=total_batch_size,
      strategy=strategy,
      optimizer=optimizer,
      cosine_adjustment=_COSINE_ADJUSTMENT.value,
      pipeline_execution_with_tensor_core=_PIPELINE_EXECUTION_WITH_TENSOR_CORE.value,
  )

  huge.train(
      model,
      optimizer=optimizer,
      strategy=strategy,
      ds_iter=ds_itr,
      model_dir=_MODEL_DIR.value,
      epochs=_EPOCHS.value,
      train_steps=_TRAIN_STEPS.value,
      nhost_steps=_NUM_HOST_STEPS.value,
      positive_batch_size=_POSITIVE_BATCH_SIZE.value,
      num_negs_per_pos=_NUM_NEGS_PER_POS.value,
      logs_dir=_LOG_DIR,
  )


if __name__ == "__main__":
  app.run(main)
