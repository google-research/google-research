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

"""Helper functions for dual encoder SMITH model."""

import collections
from typing import Any, Text

import tensorflow.compat.v1 as tf

from google.protobuf import text_format
from smith import constants


def transfer_2d_array_to_str(array):
  """Transfer a 2D float32 array to a string."""
  str_list = []
  for r in array:
    str_list.append(",".join([str(e) for e in r]))
  return " ".join(str_list)


def get_actual_max_seq_len(model_name, max_doc_length_by_sentence,
                           max_sent_length_by_word, max_predictions_per_seq):
  """Get the actual maximum sequence length.

  Args:
    model_name: The name of the model.
    max_doc_length_by_sentence: The maximum document length by the number of
      sentences.
    max_sent_length_by_word: The maximum sentence length by the number of words.
    max_predictions_per_seq: The maxinum number of predicted masked tokens in
      sequence, which can be useful for the masked LM prediction task.

  Returns:
    The actual maximum sequence length and maximum number of masked LM
    predictions per sequence. For SMITH model, we need to consider the
    maximum number of sentence blocks in a document to compute these
    statistics.

  Raises:
    ValueError: if the arguments are not usable.

  """
  if model_name == constants.MODEL_NAME_SMITH_DUAL_ENCODER:
    max_seq_length_actual = \
        max_doc_length_by_sentence * max_sent_length_by_word
    max_predictions_per_seq_actual = \
        max_doc_length_by_sentence * max_predictions_per_seq
  else:
    raise ValueError("Only the SMITH model is supported: %s" % model_name)
  return (max_seq_length_actual, max_predictions_per_seq_actual)


def get_export_outputs_prediction_dict_smith_de(
    seq_embed_1, seq_embed_2, predicted_score, predicted_class,
    documents_match_labels, input_sent_embed_1, input_sent_embed_2,
    output_sent_embed_1, output_sent_embed_2):
  """Generates export and prediction dict for dual encoder SMITH model."""
  export_outputs = {
      tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
          tf.estimator.export.PredictOutput(predicted_score),
      "seq_embed_1":
          tf.estimator.export.PredictOutput(seq_embed_1),
      "seq_embed_2":
          tf.estimator.export.PredictOutput(seq_embed_2),
      "input_sent_embed_1":
          tf.estimator.export.PredictOutput(input_sent_embed_1),
      "input_sent_embed_2":
          tf.estimator.export.PredictOutput(input_sent_embed_2),
      "output_sent_embed_1":
          tf.estimator.export.PredictOutput(output_sent_embed_1),
      "output_sent_embed_2":
          tf.estimator.export.PredictOutput(output_sent_embed_2),
      "predicted_class":
          tf.estimator.export.PredictOutput(predicted_class),
      "documents_match_labels":
          tf.estimator.export.PredictOutput(documents_match_labels)
  }

  prediction_dict = {
      "predicted_score": predicted_score,
      "predicted_class": predicted_class,
      "documents_match_labels": documents_match_labels,
      "seq_embed_1": seq_embed_1,
      "seq_embed_2": seq_embed_2,
      "input_sent_embed_1": input_sent_embed_1,
      "input_sent_embed_2": input_sent_embed_2,
      "output_sent_embed_1": output_sent_embed_1,
      "output_sent_embed_2": output_sent_embed_2
  }
  return (export_outputs, prediction_dict)


def get_pred_res_list_item_smith_de(result):
  """Update the prediction results list for the dual encoder SMITH model."""
  pred_item_dict = {}
  pred_item_dict["predicted_score"] = str(result["predicted_score"])
  pred_item_dict["predicted_class"] = str(result["predicted_class"])
  pred_item_dict["documents_match_labels"] = str(
      result["documents_match_labels"][0])
  pred_item_dict["seq_embed_1"] = ",".join(
      [str(e) for e in result["seq_embed_1"]])
  pred_item_dict["seq_embed_2"] = ",".join(
      [str(e) for e in result["seq_embed_2"]])
  pred_item_dict["input_sent_embed_1"] = transfer_2d_array_to_str(
      result["input_sent_embed_1"])
  pred_item_dict["input_sent_embed_2"] = transfer_2d_array_to_str(
      result["input_sent_embed_2"])
  pred_item_dict["output_sent_embed_1"] = transfer_2d_array_to_str(
      result["output_sent_embed_1"])
  pred_item_dict["output_sent_embed_2"] = transfer_2d_array_to_str(
      result["output_sent_embed_2"])
  return pred_item_dict


def load_config_from_file(config_file, protobuf):
  """Return the config proto loaded from config_file.

  Args:
    config_file: a string to the path of a pbtxt file.
    protobuf: an instance of a proto.

  Returns:
    An parsed of proto with the same type of protobuf.

  Raises:
    IOError: if config_file does not exist.
    ParseError: if a wrong protobuf is given.
  """
  if not tf.io.gfile.exists(config_file):
    raise IOError("{} does not exist!".format(config_file))
  with tf.gfile.Open(config_file, "r") as reader:
    proto = text_format.Parse(reader.read(), protobuf)
  return proto


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
  """Creates the predictions for the masked LM objective in preprocessing."""

  cand_indexes = []
  for (i, token) in enumerate(tokens):
    if token == "[CLS]" or token == "[SEP]":
      continue
    cand_indexes.append(i)

  rng.shuffle(cand_indexes)

  output_tokens = list(tokens)

  num_to_predict = min(max_predictions_per_seq,
                       max(1, int(round(len(tokens) * masked_lm_prob))))

  masked_lms = []
  covered_indexes = set()
  for index in cand_indexes:
    if len(masked_lms) >= num_to_predict:
      break
    if index in covered_indexes:
      continue
    covered_indexes.add(index)

    masked_token = None
    # 80% of the time, replace with [MASK]
    if rng.random() < 0.8:
      masked_token = "[MASK]"
    else:
      # 10% of the time, keep original
      if rng.random() < 0.5:
        masked_token = tokens[index]
      # 10% of the time, replace with random word
      else:
        masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

    output_tokens[index] = masked_token

    masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

  masked_lms = sorted(masked_lms, key=lambda x: x.index)

  masked_lm_positions = []
  masked_lm_labels = []
  for p in masked_lms:
    masked_lm_positions.append(p.index)
    masked_lm_labels.append(p.label)

  return (output_tokens, masked_lm_positions, masked_lm_labels)


def get_majority_vote(rating_scores):
  """Compute majority vote rating given a list.

  Args:
    rating_scores: a list of rating scores.

  Returns:
    The majority voting rating.

  """
  return collections.Counter(rating_scores).most_common()[0][0]


def get_mean_score(rating_scores):
  """Compute the mean rating score given a list.

  Args:
    rating_scores: a list of rating scores.

  Returns:
    The mean rating.

  """
  return sum(rating_scores) / len(rating_scores)


def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature


def create_bytes_feature(values):
  feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=list(values)))
  return feature
