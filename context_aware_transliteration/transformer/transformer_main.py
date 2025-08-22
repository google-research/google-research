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

# pylint: disable=unused-variable
# pylint: disable=unused-argument
# pylint: disable=missing-function-docstring
# pylint: disable=protected-access
# pylint: disable=logging-format-interpolation
"""Train and evaluate the Transformer model."""

import os

from absl import app
from absl import flags
from absl import logging
import data_pipeline
import misc
from official.common import distribute_utils
from official.modeling import performance
from official.utils.flags import core as flags_core
from official.utils.misc import keras_utils
import optimizer
import tensorflow as tf
import transformer
import translate
from utils import char_tokenizer
from utils import fullstring_tokenizer

flags.DEFINE_string("infer_string", "", "A single string to test the model.")
flags.DEFINE_string(
    "infer_file", "", "A file of inputs to evaluate, one per line."
)
flags.DEFINE_string(
    "preds_file", "", "A path indicating where to write inference predictions."
)
flags.DEFINE_integer(
    "beam_size_override", -1, "Override beam size set in model params."
)
flags.DEFINE_float("label_smoothing", None, "Amount of label smoothing to use.")
flags.DEFINE_float("learning_rate", None, "Learning rate.")
flags.DEFINE_integer("learning_rate_warmup_steps", None, "Warmup steps.")
flags.DEFINE_bool("wordpiece", False, "Use fullstring wordpiece tokenization.")

# For TPU
flags.DEFINE_string("master", "", "The BNS address of the first TPU worker.")


class TransformerTask(object):
  """Main entry of Transformer model."""

  def __init__(self, flags_obj):
    """Init function of TransformerMain.

    Args:
      flags_obj: Object containing parsed flag values, i.e., FLAGS.

    Raises:
      ValueError: if not using static batch for input data on TPU.
    """
    self.flags_obj = flags_obj
    self.predict_model = None

    # Add flag-defined parameters to params object
    num_gpus = flags_core.get_num_gpus(flags_obj)
    self.params = params = misc.get_model_params(flags_obj.param_set, num_gpus)

    # Override beam size if needed.
    if flags_obj.beam_size_override > 0:
      params["beam_size"] = flags_obj.beam_size_override
    # Override label smoothing if needed.
    if flags_obj.label_smoothing is not None:
      params["label_smoothing"] = flags_obj.label_smoothing
    if flags_obj.learning_rate is not None:
      params["learning_rate"] = flags_obj.learning_rate
    if flags_obj.learning_rate_warmup_steps is not None:
      params["learning_rate_warmup_steps"] = (
          flags_obj.learning_rate_warmup_steps
      )
    params["num_gpus"] = num_gpus
    params["use_ctl"] = flags_obj.use_ctl
    params["data_dir"] = flags_obj.data_dir
    params["model_dir"] = flags_obj.model_dir
    params["static_batch"] = flags_obj.static_batch
    params["max_length"] = flags_obj.max_length
    params["decode_batch_size"] = flags_obj.decode_batch_size
    params["decode_max_length"] = flags_obj.decode_max_length
    params["padded_decode"] = flags_obj.padded_decode
    params["max_io_parallelism"] = (
        flags_obj.num_parallel_calls or tf.data.experimental.AUTOTUNE
    )

    print(
        "IO_PARALLEL:",
        params["max_io_parallelism"],
        flags_obj.num_parallel_calls,
    )

    params["use_synthetic_data"] = flags_obj.use_synthetic_data
    params["batch_size"] = flags_obj.batch_size or params["default_batch_size"]
    params["repeat_dataset"] = None
    params["dtype"] = flags_core.get_tf_dtype(flags_obj)
    params["enable_tensorboard"] = flags_obj.enable_tensorboard
    params["enable_metrics_in_training"] = flags_obj.enable_metrics_in_training
    params["steps_between_evals"] = flags_obj.steps_between_evals
    params["enable_checkpointing"] = flags_obj.enable_checkpointing
    params["save_weights_only"] = flags_obj.save_weights_only

    self.distribution_strategy = distribute_utils.get_distribution_strategy(
        distribution_strategy=flags_obj.distribution_strategy,
        num_gpus=num_gpus,
        all_reduce_alg=flags_obj.all_reduce_alg,
        num_packs=flags_obj.num_packs,
        tpu_address=flags_obj.master or "",
    )
    if self.use_tpu:
      params["num_replicas"] = self.distribution_strategy.num_replicas_in_sync
    else:
      logging.info("Running transformer with num_gpus = %d", num_gpus)

    if self.distribution_strategy:
      logging.info(
          "For training, using distribution strategy: %s",
          self.distribution_strategy,
      )
    else:
      logging.info("Not using any distribution strategy.")

    performance.set_mixed_precision_policy(params["dtype"])

  @property
  def use_tpu(self):
    if self.distribution_strategy:
      return isinstance(self.distribution_strategy, tf.distribute.TPUStrategy)
    return False

  def _parse_example(self, serialized_example):
    """Return inputs and targets Tensors from a serialized tf.Example."""
    data_fields = {
        "inputs": tf.io.VarLenFeature(tf.int64),
        "targets": tf.io.VarLenFeature(tf.int64),
    }
    parsed = tf.io.parse_single_example(serialized_example, data_fields)
    inputs = tf.sparse.to_dense(parsed["inputs"])
    targets = tf.sparse.to_dense(parsed["targets"])
    return inputs, targets

  def _filter_max_length(self, example, max_length=256):
    """Indicates whether the example's length is lower than the maximum length."""
    return tf.logical_and(
        tf.size(example[0]) <= max_length, tf.size(example[1]) <= max_length
    )

  def train(self):
    """Trains the model."""
    params = self.params
    flags_obj = self.flags_obj
    # Sets config options.
    keras_utils.set_session_config(enable_xla=flags_obj.enable_xla)

    _ensure_dir(flags_obj.model_dir)
    with distribute_utils.get_strategy_scope(self.distribution_strategy):
      model = transformer.create_model(params, is_train=True)
      opt = self._create_optimizer()

      current_step = 0
      checkpoint = tf.train.Checkpoint(model=model, optimizer=opt)
      latest_checkpoint = tf.train.latest_checkpoint(flags_obj.model_dir)
      if latest_checkpoint:
        checkpoint.restore(latest_checkpoint)
        logging.info("Loaded checkpoint %s", latest_checkpoint)
        current_step = opt.iterations.numpy()

      model.compile(opt)

    model.summary()

    # Grab training data in uniform way.
    file_pattern = os.path.join(params["data_dir"] or "", "*train*")
    fset = tf.data.Dataset.list_files(file_pattern, shuffle=True)
    fset = list(fset.as_numpy_iterator())[0]
    dataset = tf.data.TFRecordDataset(fset, buffer_size=8 * 1000 * 1000)
    dataset = dataset.map(
        self._parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.filter(
        lambda x, y: self._filter_max_length((x, y), params["max_length"])
    )
    dataset = dataset.repeat(params["repeat_dataset"])
    if params["static_batch"]:
      dataset = dataset.padded_batch(
          # First calculate batch size (token number) per worker, then divide it
          # into sentences, and finally expand to a global batch. It could prove
          # the global batch divisble for distribution strategy.
          params["batch_size"],
          ([params["max_length"]], [params["max_length"]]),
          drop_remainder=True,
      )
    else:
      dataset = data_pipeline._batch_examples(
          dataset, params["batch_size"], params["max_length"]
      )
    map_data_fn = data_pipeline.map_data_for_transformer_fn
    dataset = dataset.map(
        map_data_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA
    )
    dataset = dataset.with_options(options)

    train_ds = self.distribution_strategy.experimental_distribute_dataset(
        dataset
    )

    callbacks = self._create_callbacks(flags_obj.model_dir, params)

    while current_step < flags_obj.train_steps:
      remaining_steps = flags_obj.train_steps - current_step
      train_steps_per_eval = (
          remaining_steps
          if remaining_steps < flags_obj.steps_between_evals
          else flags_obj.steps_between_evals
      )
      current_iteration = current_step // flags_obj.steps_between_evals

      logging.info(
          "Start train iteration at global step:{}".format(current_step)
      )
      history = None

      history = model.fit(
          train_ds,
          initial_epoch=current_iteration,
          epochs=current_iteration + 1,
          steps_per_epoch=train_steps_per_eval,
          callbacks=callbacks,
          # If TimeHistory is enabled, progress bar would be messy. Increase
          # the verbose level to get rid of it.
          verbose=(2 if flags_obj.enable_time_history else 1),
      )
      current_step += train_steps_per_eval
      logging.info("Train history: {}".format(history.history))

      logging.info("End train iteration at global step:{}".format(current_step))

  def predict_topk(self):
    """Predicts result from the model."""
    params = self.params
    flags_obj = self.flags_obj
    probe = flags_obj.infer_string
    infer_file = flags_obj.infer_file
    preds_file = flags_obj.preds_file

    if self.flags_obj.model_dir.endswith(".ckpt"):
      checkpoint = self.flags_obj.model_dir
    else:
      checkpoint = tf.train.latest_checkpoint(self.flags_obj.model_dir)

    with tf.name_scope("model"):
      model = transformer.create_model(params, is_train=False)
      self._load_weights_if_possible(model, checkpoint)

    if flags_obj.wordpiece:
      subtokenizer = fullstring_tokenizer.Subtokenizer(flags_obj.vocab_file)
    else:
      subtokenizer = char_tokenizer.Subtokenizer(flags_obj.vocab_file)

    with tf.io.gfile.GFile(preds_file, "w") as ofile:
      if probe:
        translate.tlit_from_text(model, subtokenizer, probe, None)
      if infer_file:
        with tf.io.gfile.GFile(infer_file) as ifile:
          for line in ifile:
            translate.tlit_from_text(model, subtokenizer, line.strip(), ofile)

  def _create_callbacks(self, cur_log_dir, params):
    """Creates a list of callbacks."""
    callbacks = misc.get_callbacks()
    if params["enable_checkpointing"]:
      ckpt_full_path = os.path.join(cur_log_dir, "cp-{epoch:04d}.ckpt")
      callbacks.append(
          tf.keras.callbacks.ModelCheckpoint(
              ckpt_full_path, save_weights_only=params["save_weights_only"]
          )
      )
    return callbacks

  def _load_weights_if_possible(self, model, init_weight_path=None):
    """Loads model weights when it is provided."""
    if init_weight_path:
      logging.info("Load weights: {}".format(init_weight_path))
      model.load_weights(init_weight_path)
    else:
      logging.info("Weights not loaded from path:{}".format(init_weight_path))

  def _create_optimizer(self):
    """Creates optimizer."""
    params = self.params
    lr_schedule = optimizer.LearningRateSchedule(
        params["learning_rate"],
        params["hidden_size"],
        params["learning_rate_warmup_steps"],
    )
    opt = tf.keras.optimizers.Adam(
        lr_schedule,
        params["optimizer_adam_beta1"],
        params["optimizer_adam_beta2"],
        epsilon=params["optimizer_adam_epsilon"],
    )

    opt = performance.configure_optimizer(
        opt,
        use_float16=params["dtype"] == tf.float16,
        loss_scale=flags_core.get_loss_scale(
            self.flags_obj, default_for_fp16="dynamic"
        ),
    )

    return opt


def _ensure_dir(log_dir):
  """Makes log dir if not existed."""
  if not tf.io.gfile.exists(log_dir):
    tf.io.gfile.makedirs(log_dir)


def main(_):
  flags_obj = flags.FLAGS
  task = TransformerTask(flags_obj)

  if flags_obj.mode == "train":
    task.train()
  elif flags_obj.mode == "predict_topk":
    task.predict_topk()
  else:
    raise ValueError("Invalid mode {}".format(flags_obj.mode))


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  misc.define_transformer_flags()
  app.run(main)
