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

# Copyright 2024 Google LLC
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

"""Main model training module."""

import os
import random
from typing import List

from absl import app
from absl import flags
from absl import logging
from clu import metric_writers
import data as data_lib
import eval as eval_lib
import jax
import jax.numpy as jnp
import model as model_lib
import tensorflow as tf


_SEED = flags.DEFINE_integer("seed", 21, "Random seed")
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 256, "Batch size")
_NUM_STEPS = flags.DEFINE_integer("num_steps", 500, "Number of training steps")
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
_DECAY_STEPS = flags.DEFINE_integer(
    "decay_steps", 1000, "Number of decay steps for the optimizer"
)
_DATASET_NAME = flags.DEFINE_string(
    "dataset_name",
    "yelp_polarity_reviews",
    "Dataset name for train and test.",
)
_TARGET_KEY = flags.DEFINE_string(
    "target_key", "label", "Name of the prediction target."
)
_NUM_CLASSES = flags.DEFINE_integer("num_classes", 2, "Number of classes")
_TEXT_KEY = flags.DEFINE_string("text_key", "text", "Name of the text feature.")
_RANKS = flags.DEFINE_spaceseplist(
    "ranks",
    "2,3,4,3,3,5,4,4,4,5,6,5",
    "List of ranks of the low-rank layers. If specifying ranks "
    "of the low-rank layers for all modules. Must be in order "
    '"query", "key", "value", "dense". E.g.: 1,3,3,3,4,5,5,6,7,7,7,7 '
    "5,6,5,5,6,6,5,6,6,6,6,6 2,6,11,14,15,14,13,14,13,16,16,13 "
    "7,8,8,8,7,7,6,8,8,7,8,8. Separate each module by a space.",
)
_TPU = flags.DEFINE_string(
    "tpu",
    None,
    "Name of TPU to use. Passed by XManager.",
)
_REPLACED_MODULE = flags.DEFINE_string(
    "replaced_module",
    "query",
    "Name of the module to be low-rank adapted, can be key, query, value, "
    "dense, or all.",
)
_MODEL_NAME = flags.DEFINE_string(
    "model_name",
    "bert",
    "Name of the transformer model to use. E.g. bert, roberta",
)
_WORK_DIR = flags.DEFINE_string(
    "work_dir",
    None,
    "Working directory for logging and saving artifacts.",
)
_TASK_TYPE = flags.DEFINE_string(
    "task_type",
    "classification",
    "Task type can be classification or regression.",
)

# TODO(yihed): add learning rate decay etc.
_LOGGING_STEPS = 100
_NUM_LAYERS = 12
_DEFAULT_RANK = 4
_HIDDEN_DIMENSION = 768
_SEQUENCE_LEN = 512
_ALL_MODULES = ["query", "key", "value", "dense"]

os.environ["TF_DETERMINISTIC_OPS"] = "1"


def _validate_flags():
  """Validates user input flags."""
  local_device_count = jax.local_device_count()
  logging.info("Local Device count: %d", local_device_count)
  if _BATCH_SIZE.value % local_device_count != 0:
    new_batch_size = local_device_count * jnp.ceil(
        _BATCH_SIZE.value / local_device_count
    )
    flags.FLAGS.batch_size = new_batch_size
    logging.info(
        "Batch size must be divisible by local device count. Increasing batch"
        " size to %d.",
        new_batch_size,
    )


def replace_model_layers_with_adaptive_low_rank(
    model,
    ranks,
    module_to_replace = "query",
):
  """Replaces model layers with adaptive low-rank layers.

  `ranks` will include the ranks of all replaceable modules if
  `module_to_replace` is `all`.
  """
  if module_to_replace == "all":
    for i, module in enumerate(_ALL_MODULES):
      replace_model_layers_with_adaptive_low_rank_for_module(
          model, ranks[i], module
      )
  else:
    replace_model_layers_with_adaptive_low_rank_for_module(
        model, ranks, module_to_replace
    )


def replace_model_layers_with_adaptive_low_rank_for_module(
    model,
    ranks,
    module_to_replace = "query",
):
  """Replaces model layers with adaptive low-rank layers."""
  if len(ranks) == 1:
    ranks = [_DEFAULT_RANK] * _NUM_LAYERS
  layers_to_replace = []
  # TODO(yihed): make num_layers adaptive to model.
  for i in range(_NUM_LAYERS):
    layers_to_replace.append(
        model_lib.get_model_layer(model, i, _MODEL_NAME.value)
    )

  replace_layer_fn = model_lib.get_bert_replace_layer_fn(module_to_replace)
  model_lib.replace_model_dense_layers(
      layers_to_replace,
      replace_layer_fn,
      ranks,
      module_to_replace=module_to_replace,
  )


def _parse_ranks():
  """Parses ranks from user input."""
  if _REPLACED_MODULE.value == "all":
    ranks = []
    for module_ranks in _RANKS.value:
      try:
        ranks.append([int(r) for r in module_ranks.split(",")])
      except ValueError as ve:
        raise ValueError("Ranks must be comma-separated integers.") from ve
    return ranks

  try:
    ranks = [int(r) for r in _RANKS.value[0].split(",")]
  except ValueError as ve:
    raise ValueError("Ranks must be comma-separated integers.") from ve
  return ranks


def train(strategy):
  """Trains model."""
  tokenizer = model_lib.get_pretrained_tokenizer(
      model_lib.get_model_tokenizer_path_from_name(
          _MODEL_NAME.value,
      )
  )
  ranks = _parse_ranks()
  with strategy.scope():
    model = model_lib.get_pretrained_model(
        model_lib.get_model_tokenizer_path_from_name(
            _MODEL_NAME.value,
        )
    )
    replace_model_layers_with_adaptive_low_rank(
        model, ranks, module_to_replace=_REPLACED_MODULE.value
    )
    prediction_model = model_lib.get_prediction_model(
        model,
        seq_len=_SEQUENCE_LEN,
        hidden_dimension=_HIDDEN_DIMENSION,
        num_classes=_NUM_CLASSES.value,
    )
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        _LEARNING_RATE.value,
        decay_steps=_DECAY_STEPS.value,
        decay_rate=0.8,
        staircase=True,
    )
    optimizer = tf.keras.optimizers.Adam(lr_schedule)

    trainable_weights = prediction_model.trainable_variables
    logging.info(
        "***Number of Model trainable weights***: %d",
        len(trainable_weights),
    )

    @tf.function
    def get_train_ds(input_context):
      return data_lib.get_train_dataset(
          _DATASET_NAME.value,
          tokenizer,
          _BATCH_SIZE.value,
          _TEXT_KEY.value,
          _TARGET_KEY.value,
      )

    test_ds, validation_ds = data_lib.get_test_and_validation_dataset(
        _DATASET_NAME.value,
        tokenizer,
        _BATCH_SIZE.value,
        _TEXT_KEY.value,
        _TARGET_KEY.value,
    )

    train_ds = strategy.distribute_datasets_from_function(get_train_ds)
    if _TASK_TYPE.value == "classification":
      loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
          from_logits=True, reduction=tf.keras.losses.Reduction.SUM
      )
    else:
      loss_fn = tf.keras.losses.MeanAbsoluteError(
          reduction=tf.keras.losses.Reduction.SUM
      )

  do_training(
      train_ds,
      test_ds,
      validation_ds,
      prediction_model,
      loss_fn,
      optimizer,
      strategy,
  )


def do_training(
    train_ds,
    test_ds,
    validation_ds,
    model,
    loss_fn,
    optimizer,
    strategy,
):
  """Performs training.

  Args:
    train_ds: dataset for training.
    model: model for training.
    loss_fn: Loss function for training.
    optimizer: Optimizer for training.
    num_feature_scaler: Object used for gradually scaling number of features.
  """
  logging.info("Starting training!")
  writer = metric_writers.create_default_writer(
      _WORK_DIR.value,
  )
  train_iter = iter(train_ds)

  @tf.function
  def step_fn(step_data):
    x = step_data[data_lib.X_KEY]
    labels = step_data[data_lib.TARGET_KEY]

    with tf.GradientTape() as tape:

      logits = model(x, training=True)
      loss = loss_fn(labels, logits)
      grads = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss

  for step, step_data in enumerate(train_iter):
    if step == _NUM_STEPS.value:
      break
    loss = strategy.run(step_fn, args=(step_data,))
    if step % _LOGGING_STEPS == 0 and jax.process_index() == 0:
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None)
      logging.info("step %d loss %s", step, str(loss))
      writer.write_scalars(step, {"loss": loss})
      eval_lib.evaluate_model_tf(
          model,
          test_ds=test_ds,
          validation_ds=validation_ds,
          task_type=_TASK_TYPE.value,
          writer=writer,
          step=step,
      )

  logging.info("Done training!")


def main_tpu(argv):
  del argv
  _validate_flags()
  random.seed(_SEED.value)
  tf.random.set_seed(_SEED.value)
  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=_TPU.value)
  tf.config.experimental_connect_to_cluster(resolver)
  tf.tpu.experimental.initialize_tpu_system(resolver)
  strategy = tf.distribute.TPUStrategy(resolver)
  train(strategy)


def main_gpu(argv):
  del argv
  _validate_flags()
  random.seed(_SEED.value)
  tf.random.set_seed(_SEED.value)
  strategy = tf.distribute.MirroredStrategy()
  train(strategy)


if __name__ == "__main__":
  app.run(main_gpu)
