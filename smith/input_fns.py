# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Input functions used in dual encoder SMITH model."""


from absl import flags
import tensorflow.compat.v1 as tf  # tf
from smith import constants
FLAGS = flags.FLAGS


def input_fn_builder(input_files,
                     is_training,
                     drop_remainder,
                     max_seq_length=32,
                     max_predictions_per_seq=5,
                     num_cpu_threads=4,
                     batch_size=16,
                     is_prediction=False):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  def input_fn(params):  # pylint: disable=unused-argument
    """The actual input function."""
    name_to_features = {
        "input_ids_1": tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask_1": tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_ids_2": tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask_2": tf.FixedLenFeature([max_seq_length], tf.int64),
        "documents_match_labels": tf.FixedLenFeature([1], tf.float32, 0)
    }
    if (FLAGS.train_mode == constants.TRAIN_MODE_PRETRAIN or
        FLAGS.train_mode == constants.TRAIN_MODE_JOINT_TRAIN):
      # Add some features related to word masked LM losses.
      name_to_features["masked_lm_positions_1"] = tf.FixedLenFeature(
          [max_predictions_per_seq], tf.int64)
      name_to_features["masked_lm_ids_1"] = tf.FixedLenFeature(
          [max_predictions_per_seq], tf.int64)
      name_to_features["masked_lm_weights_1"] = tf.FixedLenFeature(
          [max_predictions_per_seq], tf.float32)
      name_to_features["masked_lm_positions_2"] = tf.FixedLenFeature(
          [max_predictions_per_seq], tf.int64)
      name_to_features["masked_lm_ids_2"] = tf.FixedLenFeature(
          [max_predictions_per_seq], tf.int64)
      name_to_features["masked_lm_weights_2"] = tf.FixedLenFeature(
          [max_predictions_per_seq], tf.float32)

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
      file_list = tf.data.Dataset.list_files(tf.constant(input_files))
      file_list = file_list.shuffle(buffer_size=len(input_files))
      # `cycle_length` is the number of parallel files that get read.
      cycle_length = min(num_cpu_threads, len(input_files))
      # `sloppy` mode means that the interleaving is not exact. This adds
      # even more randomness to the training pipeline.
      d = file_list.apply(
          tf.data.experimental.parallel_interleave(
              tf.data.TFRecordDataset,
              sloppy=is_training,
              cycle_length=cycle_length))
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    else:
      d = tf.data.TFRecordDataset(tf.constant(input_files))
      # In contrast to TPU training/evaluation, the input_fn for prediction
      # should raise an end-of-input exception (OutOfRangeError or
      # StopIteration), which serves as the stopping signal to TPUEstimator.
      # Thus during model prediction, the data can not be repeated forever.
      # Refer to
      # https://www.tensorflow.org/api_docs/python/tf/compat/v1/estimator/tpu/TPUEstimator#predict
      if not is_prediction:
        # Since we evaluate for a fixed number of steps we don't want to
        # encounter out-of-range exceptions.
        d = d.repeat()

    # We must `drop_remainder` on training because the TPU requires fixed
    # size dimensions. For eval, we assume we are evaling on the CPU or GPU
    # and we *don"t* want to drop the remainder, otherwise we won't cover
    # every sample.
    d = d.apply(
        tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=drop_remainder))
    return d

  return input_fn


def _decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.parse_single_example(record, name_to_features)
  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  example["input_ids_1"] = tf.cast(example["input_ids_1"], tf.int32)
  example["input_ids_2"] = tf.cast(example["input_ids_2"], tf.int32)
  example["documents_match_labels"] = tf.cast(example["documents_match_labels"],
                                              tf.float32)
  example["input_mask_1"] = tf.cast(example["input_mask_1"], tf.int32)
  example["input_mask_2"] = tf.cast(example["input_mask_2"], tf.int32)
  if (FLAGS.train_mode == constants.TRAIN_MODE_PRETRAIN or
      FLAGS.train_mode == constants.TRAIN_MODE_JOINT_TRAIN):
    example["masked_lm_ids_1"] = tf.cast(example["masked_lm_ids_1"], tf.int32)
    example["masked_lm_ids_2"] = tf.cast(example["masked_lm_ids_2"], tf.int32)
    example["masked_lm_weights_1"] = tf.cast(example["masked_lm_weights_1"],
                                             tf.float32)
    example["masked_lm_weights_2"] = tf.cast(example["masked_lm_weights_2"],
                                             tf.float32)
    example["masked_lm_positions_1"] = tf.cast(example["masked_lm_positions_1"],
                                               tf.int32)
    example["masked_lm_positions_2"] = tf.cast(example["masked_lm_positions_2"],
                                               tf.int32)
  return example


def make_serving_input_example_fn(max_seq_length=32, max_predictions_per_seq=5):
  """Returns an Estimator input_fn for serving the model.

  Args:
    max_seq_length: The max input sequence length.
    max_predictions_per_seq: The max number of masked words per sequence.

  Returns:
    An Estimator input_fn for serving the model.
  """

  def _serving_input_fn():
    """An input_fn that expects a serialized tf.Example."""

    serialized_example = tf.placeholder(
        dtype=tf.string, shape=[None], name="examples")
    receiver_tensors = {"examples": serialized_example}
    name_to_features = {
        "input_ids_1": tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask_1": tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_ids_2": tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask_2": tf.FixedLenFeature([max_seq_length], tf.int64),
        "documents_match_labels": tf.FixedLenFeature([1], tf.float32, 0)
    }
    if (FLAGS.train_mode == constants.TRAIN_MODE_PRETRAIN or
        FLAGS.train_mode == constants.TRAIN_MODE_JOINT_TRAIN):
      # This is to support model export during model pretraining or
      # joint-training process.
      name_to_features["masked_lm_positions_1"] = tf.FixedLenFeature(
          [max_predictions_per_seq], tf.int64)
      name_to_features["masked_lm_ids_1"] = tf.FixedLenFeature(
          [max_predictions_per_seq], tf.int64)
      name_to_features["masked_lm_weights_1"] = tf.FixedLenFeature(
          [max_predictions_per_seq], tf.float32)
      name_to_features["masked_lm_positions_2"] = tf.FixedLenFeature(
          [max_predictions_per_seq], tf.int64)
      name_to_features["masked_lm_ids_2"] = tf.FixedLenFeature(
          [max_predictions_per_seq], tf.int64)
      name_to_features["masked_lm_weights_2"] = tf.FixedLenFeature(
          [max_predictions_per_seq], tf.float32)

    parsed_features = tf.parse_example(serialized_example, name_to_features)
    # As tf.Example only supports tf.int64, but the TPU only supports
    # tf.int32, we need to cast all int64 to int32.
    parsed_features["input_ids_1"] = tf.cast(parsed_features["input_ids_1"],
                                             tf.int32)
    parsed_features["input_ids_2"] = tf.cast(parsed_features["input_ids_2"],
                                             tf.int32)
    parsed_features["documents_match_labels"] = tf.cast(
        parsed_features["documents_match_labels"], tf.float32)
    parsed_features["input_mask_1"] = tf.cast(parsed_features["input_mask_1"],
                                              tf.int32)
    parsed_features["input_mask_2"] = tf.cast(parsed_features["input_mask_2"],
                                              tf.int32)
    if (FLAGS.train_mode == constants.TRAIN_MODE_PRETRAIN or
        FLAGS.train_mode == constants.TRAIN_MODE_JOINT_TRAIN):
      parsed_features["masked_lm_ids_1"] = tf.cast(
          parsed_features["masked_lm_ids_1"], tf.int32)
      parsed_features["masked_lm_ids_2"] = tf.cast(
          parsed_features["masked_lm_ids_2"], tf.int32)
      parsed_features["masked_lm_weights_1"] = tf.cast(
          parsed_features["masked_lm_weights_1"], tf.float32)
      parsed_features["masked_lm_weights_2"] = tf.cast(
          parsed_features["masked_lm_weights_2"], tf.float32)
      parsed_features["masked_lm_positions_1"] = tf.cast(
          parsed_features["masked_lm_positions_1"], tf.int32)
      parsed_features["masked_lm_positions_2"] = tf.cast(
          parsed_features["masked_lm_positions_2"], tf.int32)
    return tf.estimator.export.ServingInputReceiver(
        features=parsed_features, receiver_tensors=receiver_tensors)

  return _serving_input_fn
