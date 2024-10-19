# coding=utf-8
# Copyright 2024 The Google Research Authors.
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
"""Script based on the BERT finetuning runner, modified for performing emotion prediction.

Main changes:
- Updated DataProcessor
- Included multilabel classification
- Included various evaluation metrics
- Included evaluation as part of training
- Included sentiment-based regularization (manually defined)
- Included correlation-based regularization (based on training data)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import os

from goemotions.bert import modeling
from goemotions.bert import optimization
from goemotions.bert import tokenization
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import estimator as tf_estimator

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string("emotion_file", "goemotions/data/emotions.txt",
                    "File containing a list of emotions.")

flags.DEFINE_string(
    "data_dir", "goemotions/data",
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("test_fname", "test.tsv", "The name of the test file.")
flags.DEFINE_string("train_fname", "train.tsv",
                    "The name of the training file.")
flags.DEFINE_string("dev_fname", "dev.tsv", "The name of the dev file.")

flags.DEFINE_boolean("multilabel", False,
                     "Whether to perform multilabel classification.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", False,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 50,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", True,
                  "Whether to run training & evaluation on the dev set.")

flags.DEFINE_bool(
    "calculate_metrics", True,
    "Whether to calculate performance metrics on the test set "
    "(FLAGS.test_fname must have labels).")

flags.DEFINE_bool(
    "do_predict", True,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_bool(
    "do_export", False,
    "Whether to export the model to SavedModel format.")

flags.DEFINE_integer("train_batch_size", 16, "Total batch size for training.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 4.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_integer("keep_checkpoint_max", 10,
                     "Maximum number of checkpoints to store.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_float("pred_cutoff", 0.05,
                   "Cutoff probability for showing top emotions.")

flags.DEFINE_float(
    "eval_prob_threshold", 0.3,
    "Cutoff probability determine which labels are 1 vs 0, when calculating "
    "certain evaluation metrics.")

flags.DEFINE_string(
    "eval_thresholds", "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99",
    "Thresholds for evaluating precision, recall and F-1 scores.")

flags.DEFINE_float("sentiment", 0,
                   "Regularization parameter for sentiment relations.")

flags.DEFINE_float("correlation", 0,
                   "Regularization parameter for emotion correlations.")

flags.DEFINE_integer("save_checkpoints_steps", 500,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("save_summary_steps", 100,
                     "How often to save model summaries.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("eval_steps", None,
                     "How many steps to take to go over the eval set.")

flags.DEFINE_string("sentiment_file", "goemotions/data/sentiment_dict.json",
                    "Dictionary of sentiment categories.")

flags.DEFINE_string(
    "emotion_correlations", None,
    "Dataframe containing emotion correlation values "
    "(if FLAGS.correlation != 0)."
)

flags.DEFINE_bool(
    "transfer_learning", False,
    "Whether to perform transfer learning (i.e. replace output layer).")

flags.DEFINE_bool("freeze_layers", False, "Whether to freeze BERT layers.")

flags.DEFINE_bool(
    "add_neutral", False,
    "Whether to add a neutral label in addition to the other labels "
    "(necessary when neutral is not part of the emotion file).")


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text, labels=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text: string. The untokenized text of the input sequence.
      labels: (Optional) string. The labels of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text = text
    self.labels = labels


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, input_ids, input_mask, segment_ids, label_ids):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_ids = label_ids


class DataProcessor(object):
  """Class for preprocessing the corpus emotion dataset."""

  def __init__(self, num_labels, data_dir):
    self.num_labels = num_labels
    self.data_dir = data_dir

  def get_examples(self, data_type, fname):
    """Gets a collection of `InputExample`s for the train/dev/test set."""
    input_fname = os.path.join(self.data_dir, fname)
    return self._create_examples(
        self._read_df(input_fname, data_type), data_type)

  @classmethod
  def _read_df(cls, input_file, data_type):
    """Reads a tab separated value file."""
    sep = None
    if input_file.endswith("tsv"):
      sep = "\t"
    elif input_file.endswith("csv"):
      sep = ","
    elif data_type == "test":
      sep = "\t"
    else:
      print("Filetype not supported for %s" % input_file)
      return None

    if data_type == "test":
      names = ["text"]
    else:
      names = ["text", "labels"]

    # Load file
    print(input_file)
    return pd.read_csv(
        input_file,
        sep=sep,
        encoding="utf-8",
        header=None,
        names=names,
        usecols=names,
        dtype={"text": str})

  def _create_examples(self, df, data_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, row) in df.iterrows():
      guid = "%s-%s" % (data_type, i)
      labels = [0] * self.num_labels
      if isinstance(row["text"], float):
        text = ""  # This accounts for rare encoding errors
      else:
        text = tokenization.convert_to_unicode(row["text"])
      if data_type != "test":
        label_ids = str(row["labels"]).split(",")
        for idx in label_ids:
          labels[int(idx)] = 1
      examples.append(InputExample(guid=guid, text=text, labels=labels))
    return examples


def convert_single_example(ex_index, example, max_seq_length, tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  tokens = tokenizer.tokenize(example.text)
  # Account for [CLS] and [SEP] with "- 2"
  if len(tokens) > max_seq_length - 2:
    tokens = tokens[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = ["[CLS]"] + tokens + ["[SEP]"]
  segment_ids = [0] * len(tokens)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" %
                    " ".join([tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("labels: %s" % " ".join([str(x) for x in example.labels]))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_ids=example.labels)
  return feature


def file_based_convert_examples_to_features(examples, max_seq_length, tokenizer,
                                            output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, max_seq_length,
                                     tokenizer)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature(feature.label_ids)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder, num_labels):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([num_labels], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, multilabel, sent_rels, sentiment,
                 corr_rels, correlation):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids)

  # Here, we are doing a classification task on the entire segment. For
  # token-level output, use model.get_sequece_output() instead.
  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)

    # Labels both for single and multilabel classification
    labels = tf.cast(labels, tf.float32)

    if multilabel:
      probabilities = tf.nn.sigmoid(logits)
      tf.logging.info("num_labels:{};logits:{};labels:{}".format(
          num_labels, logits, labels))
      per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels, logits=logits)
    else:
      probabilities = tf.nn.softmax(logits, axis=-1)
      per_example_loss = tf.nn.softmax_cross_entropy_with_logits(
          labels=labels, logits=logits)
    loss = tf.reduce_mean(per_example_loss)

    # Add regularization based on label relations prior
    probs_exp = tf.expand_dims(probabilities, 1)
    m = tf.tile(probs_exp, [1, num_labels, 1])
    probs_exp_t = tf.transpose(probs_exp, perm=[0, 2, 1])

    # Subtract each prediction from all others:
    # Example (with batch size=1):
    #     tiled predictions: [0.1] [0.1] [0.1]
    #                        [0.2] [0.2] [0.2]
    #                        [0.3] [0.3] [0.3]
    #     subtract [0.1, 0.2, 0.3] row-wise
    #     result:   [0.0] [-.1] [-.2] --> row represents difference between
    #                                     emotion 1 and all other emotions
    #               [0.1] [0.0] [-.1]
    #               [0.2] [0.1] [0.0]
    dists = tf.square(tf.subtract(m, probs_exp_t))  # square distances
    dists = tf.transpose(dists, perm=[0, 2, 1])

    # Sentiment-based regularization
    sent_reg = tf.multiply(
        tf.constant(sentiment),
        tf.reduce_mean(
            tf.multiply(dists, tf.constant(sent_rels, dtype=tf.float32))))
    tf.summary.scalar("sentiment_regularization", sent_reg)
    loss += sent_reg

    # Correlation-based regularization
    corr_reg = tf.multiply(
        tf.constant(correlation),
        tf.reduce_mean(
            tf.multiply(dists, tf.constant(corr_rels, dtype=tf.float32))))
    tf.summary.scalar("correlation_regularization", corr_reg)
    loss += corr_reg

    tf.summary.scalar("loss", loss)

    return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, multilabel, sent_rels,
                     sentiment, corr_rels, correlation, idx2emotion,
                     sentiment_groups):
  """Returns `model_fn` closure for Estimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for Estimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]

    is_training = (mode == tf_estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, multilabel, sent_rels, sentiment, corr_rels, correlation)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint,
                                                      FLAGS.transfer_learning)
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Initialized Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf_estimator.ModeKeys.TRAIN:

      freeze_layer_fn = (None
                         if not FLAGS.freeze_layers else lambda x: "bert" in x)
      train_op = optimization.create_optimizer(
          total_loss,
          learning_rate,
          num_train_steps,
          num_warmup_steps,
          use_tpu=False,
          freeze_layer_fn=freeze_layer_fn)

      output_spec = tf_estimator.EstimatorSpec(
          mode=mode, loss=total_loss, train_op=train_op, scaffold=scaffold_fn)

    elif mode == tf_estimator.ModeKeys.EVAL:

      # Create dictionary for evaluation metrics
      eval_dict = {}

      def metric_fn_single(per_example_loss, label_ids, logits):
        """Compute accuracy for the single-label case."""
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        true_labels = tf.argmax(
            label_ids, axis=-1,
            output_type=tf.int32)  # Get ids from one hot labels
        accuracy = tf.metrics.accuracy(
            labels=true_labels, predictions=predictions)
        loss = tf.metrics.mean(values=per_example_loss)
        eval_dict["eval_accuracy"] = accuracy
        eval_dict["eval_loss"] = loss

      def get_f1(precision, recall):
        """Calculate F1 score based on precision and recall."""
        return (2 * precision[0] * recall[0] /
                (precision[0] + recall[0] + 1e-5),
                tf.group(precision[1], recall[1]))

      def get_threshold_based_scores(y_true, y_pred):
        """Compute precision, recall and F1 at thresholds."""
        thresholds = [float(v) for v in FLAGS.eval_thresholds.split(",")]
        (prec_t, prec_t_op) = tf.metrics.precision_at_thresholds(
            y_true, y_pred, thresholds=thresholds)
        (rec_t, rec_t_op) = tf.metrics.recall_at_thresholds(
            y_true, y_pred, thresholds=thresholds)
        for i, v in enumerate(thresholds):
          eval_dict["precision_at_threshold_%.2f" % v] = (prec_t[i], prec_t_op)
          eval_dict["recall_at_threshold_%.2f" % v] = (rec_t[i], rec_t_op)
          eval_dict["F1_at_threshold_%.2f" % v] = get_f1((prec_t[i], prec_t_op),
                                                         (rec_t[i], rec_t_op))

      def get_relation_based_scores(y_true, y_pred, relations, name):
        """Measure performance based on label relations."""

        def expand_labels(labels):
          """Expand the set of labels based on label relations."""

          def check_relations(rels):
            """Check whether a relation applies to a particular label set."""
            is_in_category = tf.reduce_any((labels + rels) > 1)
            return tf.cond(is_in_category, lambda: labels + rels,
                           lambda: labels)

          new_labels = tf.reduce_sum(
              tf.map_fn(check_relations, relations), axis=0)
          return tf.cast(new_labels >= 1, tf.int64)

        pred = tf.map_fn(expand_labels, y_pred)
        true = tf.map_fn(expand_labels, y_true)
        precision = tf.metrics.precision(true, pred)
        recall = tf.metrics.recall(true, pred)
        eval_dict[name + "_precision"] = precision
        eval_dict[name + "_recall"] = recall
        eval_dict[name + "_f1"] = get_f1(precision, recall)
        eval_dict[name + "_accuracy"] = tf.metrics.accuracy(true, pred)

      def metric_fn_multi(per_example_loss, label_ids, probabilities):
        """Compute class-level accuracies for the multi-label case."""
        label_ids = tf.cast(label_ids, tf.int64)
        logits_split = tf.split(probabilities, num_labels, axis=-1)
        label_ids_split = tf.split(label_ids, num_labels, axis=-1)
        pred_ind = tf.cast(probabilities >= FLAGS.eval_prob_threshold, tf.int64)
        pred_ind_split = tf.split(pred_ind, num_labels, axis=-1)
        weights = tf.reduce_sum(label_ids, axis=0)

        eval_dict["per_example_eval_loss"] = tf.metrics.mean(
            values=per_example_loss)

        # Calculate accuracy, precision and recall
        get_threshold_based_scores(label_ids, probabilities)

        # Calculate values at the emotion level
        auc_vals = []
        accuracies = []
        for j, logits in enumerate(logits_split):
          current_auc, update_op_auc = tf.metrics.auc(label_ids_split[j],
                                                      logits)
          eval_dict[idx2emotion[j] + "_auc"] = (current_auc, update_op_auc)
          current_acc, update_op_acc = tf.metrics.accuracy(
              label_ids_split[j], pred_ind_split[j])
          eval_dict[idx2emotion[j] + "_accuracy"] = (current_acc, update_op_acc)
          eval_dict[idx2emotion[j] + "_precision"] = tf.metrics.precision(
              label_ids_split[j], pred_ind_split[j])
          eval_dict[idx2emotion[j] + "_recall"] = tf.metrics.recall(
              label_ids_split[j], pred_ind_split[j])
          auc_vals.append(current_auc)
          accuracies.append(current_auc)
        auc_vals = tf.convert_to_tensor(auc_vals, dtype=tf.float32)
        accuracies = tf.convert_to_tensor(accuracies, dtype=tf.float32)
        eval_dict["auc"] = tf.metrics.mean(values=auc_vals)
        eval_dict["auc_weighted"] = tf.metrics.mean(
            values=auc_vals, weights=weights)
        eval_dict["accuracy"] = tf.metrics.mean(values=accuracies)
        eval_dict["accuracy_weighted"] = tf.metrics.mean(
            values=accuracies, weights=weights)

        # Calculate sentiment-based performance
        get_relation_based_scores(label_ids, pred_ind,
                                  tf.constant(sentiment_groups, dtype=tf.int64),
                                  "sentiment")

      if multilabel:
        metric_fn_multi(per_example_loss, label_ids, probabilities)
      else:
        metric_fn_single(per_example_loss, label_ids, logits)

      output_spec = tf_estimator.EstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metric_ops=eval_dict,
          scaffold=scaffold_fn)
    else:
      print("mode:", mode, "probabilities:", probabilities)
      output_spec = tf_estimator.EstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities},
          scaffold=scaffold_fn)
    return output_spec

  return model_fn


def get_sent_rels(emotions):
  """Get emotion distance matrix for sentiment regularization."""
  with open(FLAGS.sentiment_file) as f:
    sent_dict = json.loads(f.read())

  emotion2sentiment = {}
  for k, v in sent_dict.items():
    for e in v:
      # no emotion should be in two categories
      assert e not in emotion2sentiment
      emotion2sentiment[e] = k
  rels = []
  for e1 in emotions:
    e1_rels = []
    for e2 in emotions:
      if e1 not in emotion2sentiment or e2 not in emotion2sentiment:
        e1_rels.append(0)
      elif emotion2sentiment[e1] != emotion2sentiment[e2]:
        e1_rels.append(-1)
      elif emotion2sentiment[e1] == emotion2sentiment[e2]:
        e1_rels.append(1)
    rels.append(e1_rels)
  return rels


def get_correlations(emotions):
  """Get correlations between emotions based training data."""
  corrs = pd.read_csv(FLAGS.emotion_correlations, index_col=0, sep="\t")
  rels = []
  for e1 in emotions:
    if e1 == "neutral":
      rels.append([0] * len(emotions))
    else:
      e1_rels = []
      for e2 in emotions:
        if e2 == "neutral":
          e1_rels.append(0)
        else:
          e1_rels.append(corrs.loc[e1, e2])
      rels.append(e1_rels)
  return rels


def get_sentiment_groups(emotions):
  """Get sentiment groups for evaluating sentiment-based performance."""
  with open(FLAGS.sentiment_file) as f:
    sent_dict = json.loads(f.read())
  rels = []
  for _, v in sent_dict.items():
    grouped_labels = []
    for e in emotions:
      if e in v:
        grouped_labels.append(1)
      else:
        grouped_labels.append(0)
    rels.append(grouped_labels)
  return rels


def main(_):
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  # Load emotion categories
  with open(FLAGS.emotion_file, "r") as f:
    all_emotions = f.read().splitlines()
    if FLAGS.add_neutral:
      all_emotions = all_emotions + ["neutral"]
    idx2emotion = {i: e for i, e in enumerate(all_emotions)}
  num_labels = len(all_emotions)
  print("%d labels" % num_labels)
  print("Multilabel: %r" % FLAGS.multilabel)

  sentiment = FLAGS.sentiment
  correlation = FLAGS.correlation

  # Create emotion distance matrix
  # If the regularization parameter is set to 0, don't load matrix.
  print("Getting distance matrix...")
  empty_rels = [[0] * num_labels] * num_labels
  if sentiment == 0:
    sent_rels = empty_rels
  else:
    sent_rels = get_sent_rels(all_emotions)
  sent_groups = get_sentiment_groups(all_emotions)
  print(sent_rels)
  if correlation == 0:
    corr_rels = empty_rels
  else:
    corr_rels = get_correlations(all_emotions)
  print(corr_rels)

  tf.logging.set_verbosity(tf.logging.INFO)

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  if not FLAGS.do_train and not FLAGS.do_predict and not FLAGS.do_export:
    raise ValueError(
        "At least one of `do_train`, `do_predict', or `do_export` must be True."
    )

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  processor = DataProcessor(num_labels, FLAGS.data_dir)  # set up preprocessor

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  run_config = tf_estimator.RunConfig(
      model_dir=FLAGS.output_dir,
      save_summary_steps=FLAGS.save_summary_steps,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max)

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None

  if FLAGS.do_train:
    train_examples = processor.get_examples("train", FLAGS.train_fname)
    eval_examples = processor.get_examples("dev", FLAGS.dev_fname)
    num_eval_examples = len(eval_examples)
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    params = {
        "num_labels": num_labels,
        "learning_rate": FLAGS.learning_rate,
        "num_train_epochs": FLAGS.num_train_epochs,
        "warmup_proportion": FLAGS.warmup_proportion,
        "sentiment": FLAGS.sentiment,
        "correlations": FLAGS.correlation,
        "batch_size": FLAGS.train_batch_size,
        "num_train_examples": len(train_examples),
        "num_eval_examples": num_eval_examples,
        "data_dir": FLAGS.data_dir,
        "output_dir": FLAGS.output_dir,
        "train_fname": FLAGS.train_fname,
        "dev_fname": FLAGS.dev_fname,
        "test_fname": FLAGS.test_fname
    }
    with open(os.path.join(FLAGS.output_dir, "config.json"), "w") as f:
      json.dump(params, f)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=num_labels,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      multilabel=FLAGS.multilabel,
      sent_rels=sent_rels,
      sentiment=sentiment,
      corr_rels=corr_rels,
      correlation=correlation,
      idx2emotion=idx2emotion,
      sentiment_groups=sent_groups)

  estimator = tf_estimator.Estimator(
      model_fn=model_fn,
      config=run_config,
      params={"batch_size": FLAGS.train_batch_size})

  if FLAGS.do_train:
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    file_based_convert_examples_to_features(train_examples,
                                            FLAGS.max_seq_length, tokenizer,
                                            train_file)
    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    file_based_convert_examples_to_features(eval_examples, FLAGS.max_seq_length,
                                            tokenizer, eval_file)

    tf.logging.info("***** Running training and evaluation *****")
    tf.logging.info("  Num train examples = %d", len(train_examples))
    tf.logging.info("  Num eval examples = %d", num_eval_examples)
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num training steps = %d", num_train_steps)
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True,
        num_labels=num_labels)
    train_spec = tf_estimator.TrainSpec(
        input_fn=train_input_fn, max_steps=num_train_steps)
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False,
        num_labels=num_labels)
    eval_spec = tf_estimator.EvalSpec(
        input_fn=eval_input_fn,
        steps=FLAGS.eval_steps,
        start_delay_secs=0,
        throttle_secs=1000)

    tf_estimator.train_and_evaluate(
        estimator, train_spec=train_spec, eval_spec=eval_spec)

  if FLAGS.calculate_metrics:

    # Setting the parameter to "dev" ensures that we get labels for the examples
    eval_examples = processor.get_examples("dev", FLAGS.test_fname)

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num eval examples = %d", len(eval_examples))
    eval_file = os.path.join(FLAGS.output_dir, FLAGS.test_fname + ".tf_record")
    file_based_convert_examples_to_features(eval_examples, FLAGS.max_seq_length,
                                            tokenizer, eval_file)
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False,
        num_labels=num_labels)

    result = estimator.evaluate(input_fn=eval_input_fn, steps=None)
    output_eval_file = os.path.join(FLAGS.output_dir,
                                    FLAGS.test_fname + ".eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))

  if FLAGS.do_predict:
    predict_examples = processor.get_examples("test", FLAGS.test_fname)
    num_actual_predict_examples = len(predict_examples)

    predict_file = os.path.join(FLAGS.output_dir,
                                FLAGS.test_fname + ".tf_record")
    file_based_convert_examples_to_features(predict_examples,
                                            FLAGS.max_seq_length, tokenizer,
                                            predict_file)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(predict_examples), num_actual_predict_examples,
                    len(predict_examples) - num_actual_predict_examples)
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)

    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False,
        num_labels=num_labels)

    result = estimator.predict(input_fn=predict_input_fn)

    output_predict_file = os.path.join(FLAGS.output_dir,
                                       FLAGS.test_fname + ".predictions.tsv")
    output_labels = os.path.join(FLAGS.output_dir,
                                 FLAGS.test_fname + ".label_predictions.tsv")

    with tf.gfile.GFile(output_predict_file, "w") as writer:
      with tf.gfile.GFile(output_labels, "w") as writer2:
        writer.write("\t".join(all_emotions) + "\n")
        writer2.write("\t".join([
            "text", "emotion_1", "prob_1", "emotion_2", "prob_2", "emotion_3",
            "prob_3"
        ]) + "\n")
        tf.logging.info("***** Predict results *****")
        num_written_lines = 0
        for (i, prediction) in enumerate(result):
          probabilities = prediction["probabilities"]
          if i >= num_actual_predict_examples:
            break
          output_line = "\t".join(
              str(class_probability)
              for class_probability in probabilities) + "\n"
          sorted_idx = np.argsort(-probabilities)
          top_3_emotion = [idx2emotion[idx] for idx in sorted_idx[:3]]
          top_3_prob = [probabilities[idx] for idx in sorted_idx[:3]]
          pred_line = []
          for emotion, prob in zip(top_3_emotion, top_3_prob):
            if prob >= FLAGS.pred_cutoff:
              pred_line.extend([emotion, "%.4f" % prob])
            else:
              pred_line.extend(["", ""])
          writer.write(output_line)
          writer2.write(predict_examples[i].text + "\t" + "\t".join(pred_line) +
                        "\n")
          num_written_lines += 1
    assert num_written_lines == num_actual_predict_examples

  if FLAGS.do_export:
    tf.logging.info("***** Exporting to SavedModel *****")
    feature_spec = {
        "input_ids": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([num_labels], tf.int64)
    }
    input_receiver = (
        tf_estimator.export.build_parsing_serving_input_receiver_fn(
            feature_spec))
    estimator.export_saved_model(FLAGS.output_dir, input_receiver)


if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
