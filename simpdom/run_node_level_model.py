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

"""Node level classifier model for filed classification."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import os
import pickle
import sys

from absl import app
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf

from simpdom import constants
from simpdom import model_util
from simpdom import models
from tensorflow_estimator.python.estimator import early_stopping


FLAGS = flags.FLAGS
tf.set_random_seed(42)

flags.DEFINE_boolean(
    "add_goldmine", False,
    "If to use node-level goldmine embedding to enrich node representations.")
flags.DEFINE_boolean(
    "add_leaf_types", False,
    "If to use node-level leaf type embedding to enrich node representations.")
flags.DEFINE_boolean(
    "cross_vertical", False,
    "If to apply the cross vertical training mode.")
flags.DEFINE_boolean(
    "extract_node_emb", True,
    "If to extract the node embeddings while writing predictions.")
flags.DEFINE_boolean(
    "match_keywords", False,
    "If to match keywords in partners and labels while writing predictions.")
flags.DEFINE_boolean(
    "use_crf", False,
    "If to use the CRF layer for decoding the sequencial predictions.")
flags.DEFINE_boolean("use_friends_cnn", False,
                     "If to use the CNN for encoding friend node feature.")
flags.DEFINE_boolean(
    "use_friends_discrete_feature", False,
    "If to use the feature that checks whether friends contain an attribute.")
flags.DEFINE_boolean(
    "use_friend_semantic", False,
    "If to encode the semantic similarity between friend texts and labels.")
flags.DEFINE_boolean("use_prev_text_lstm", False,
                     "If to use the LSTM for encoding prev_text feature.")
flags.DEFINE_boolean(
    "use_position_embedding", False,
    "If to use the relative position in the DOM tree as a node feature.")
flags.DEFINE_boolean(
    "use_uniform_embedding", False,
    "If to use same embedding initialization for all the verticals.")
flags.DEFINE_boolean(
    "use_uniform_label", False,
    "If to use same label embedding for all the verticals.")
flags.DEFINE_boolean("use_xpath_lstm", False,
                     "If to use lstm for encoding xpath sequences.")
flags.DEFINE_enum("circle_features", None, ["partner", "friends", "all"],
                  "Choose the set of circle features to encode.")
flags.DEFINE_enum("friend_encoder", None,
                  ["average", "max", "cnn", "attention", "self-attention"],
                  "Choose the method to encode friend features.")
flags.DEFINE_enum(
    "node_encoder", None, ["cnn", "lstm", "transformer"],
    "Choose the node-level model to further encode the node sequences.")
flags.DEFINE_enum(
    "objective", None, ["classification", "semantic_scorer", "binary_scorer"],
    "Choose the computing method for logits. The classification "
    "mode gets the logits from a dense layer on node embeddings, "
    "the semantic scorer mode directly uses the semantic "
    "similarities as the logits, while binary scorer concatenates "
    "node and label embeddings for binary classification.")
flags.DEFINE_enum("semantic_encoder", None,
                  ["inner_prod", "cos_sim", "cos_sim_randomized"],
                  "Choose the method to encode semantic similarities.")
flags.DEFINE_integer("batch_size", 16, "Batch size.")
flags.DEFINE_integer("dim_chars", 100, "Dim of Char vectors.")
flags.DEFINE_integer("dim_label_embedding", 32, "Dim of label embeddings.")
flags.DEFINE_integer("dim_word_embedding", 100, "Dim of word vectors.")
flags.DEFINE_integer("dim_xpath_units", 30, "Dim of Xpath Unit embedding.")
flags.DEFINE_integer("epochs", 10, "Epoch number.")
flags.DEFINE_integer("friend_hidden_size", 25,
                     "Dim of friend hidden embeddings.")
flags.DEFINE_integer("last_hidden_layer_size", 10,
                     "Dim of the dense layer before classification.")
flags.DEFINE_integer("max_len_friend_nodes", 10,
                     "Maximum size of the friends for each node.")
flags.DEFINE_integer("max_len_prev_text", 5,
                     "Maximum length of the tokens in prev_text feature.")
flags.DEFINE_integer("max_len_text", 10,
                     "Maximum length of the tokens in node text.")
flags.DEFINE_integer("max_steps_no_increase", 50,
                     "Num of max steps without increase for early stop.")
flags.DEFINE_integer("node_lstm_size", 100,
                     "Number of the hidden units of LSTM.")
flags.DEFINE_integer(
    "none_cutoff", 30000,
    "Controlling the random sampling rate for nodes with label as None. Scope"
    " is 0~100000, the lower, the more None nodes will be sampled.")
# The transformer_hidden_unit, transformer_head, transformer_hidden_layer
# should be ignored unless use_node_transformer is true.
flags.DEFINE_integer("transformer_hidden_unit", 128,
                     "Num of hidden units in transformer.")
flags.DEFINE_integer("transformer_head", 4, "Num of heads in transformer.")
flags.DEFINE_integer("transformer_hidden_layer", 2,
                     "Num of hidden layers in transformer.")
flags.DEFINE_integer("xpath_lstm_size", 10, "Dim of Xpath-level LSTM states.")
# The checkpoint_path, checkpoint_vertical, checkpoint_websites
# are only used when FLAGS_cross_vertical is true.
flags.DEFINE_string("checkpoint_path", "", "The path to restore checkpoints.")
flags.DEFINE_string("checkpoint_vertical", "", "The vertical of checkpoints.")
flags.DEFINE_string("checkpoint_websites", "", "The websites of checkpoints.")
flags.DEFINE_string(
    "domtree_data_path", "",
    "The path to the folder of all tree data formatted of SWDE dataset.")
flags.DEFINE_string(
    "goldmine_data_path", "",
    "The path to the folder of all the node-level goldmine features.")
flags.DEFINE_string("result_path", "",
                    "The path to the folder saving the results.")
flags.DEFINE_string("run", "train",
                    "The mode of running the model, either train or test")
flags.DEFINE_string("source_website", "",
                    "The source website names for training.")
flags.DEFINE_string("target_website", "",
                    "The target website name for testing.")
flags.DEFINE_string("vertical", "auto", "Vertical name.")


def get_data_path(vertical, website, dev=False, goldmine=False):
  """Gets the file path for the required data."""
  assert vertical
  assert website
  file_path = os.path.join(FLAGS.domtree_data_path,
                           "{}-{}.json".format(vertical, website))
  if dev and tf.gfile.Exists(file_path.replace(".json", ".dev.json")):
    file_path = file_path.replace(".json", ".dev.json")
  if goldmine:
    file_path = file_path.replace(FLAGS.domtree_data_path,
                                  FLAGS.goldmine_data_path)
    file_path = file_path.replace(".json", ".feat.json")
  assert tf.gfile.Exists(file_path)
  return file_path


def normalize_text(w):
  """Normalize the text in a node."""
  return str(w, "utf-8").lower().replace("-", "")


def write_predictions(estimator, vertical, source_website, target_website):
  """Writes the model prediction to a tsv file for further analysis."""
  score_dir_path = os.path.join(
      FLAGS.result_path, "{}/{}-results/score".format(vertical, source_website))

  tf.gfile.MakeDirs(score_dir_path)
  pred_filename = os.path.join(
      FLAGS.result_path,
      "{}/{}-results/score/{}.preds.txt".format(vertical, source_website,
                                                target_website))
  node_emb_filename = os.path.join(
      FLAGS.result_path,
      "{}/{}-results/score/{}.node_emb.npz".format(vertical, source_website,
                                                   target_website))
  print("Writing predictions to file: %s" % pred_filename, file=sys.stderr)
  golds_gen = model_util.joint_generator_fn(
      get_data_path(
          vertical=vertical, website=target_website, dev=False, goldmine=False),
      get_data_path(
          vertical=vertical, website=target_website, dev=False, goldmine=True),
      vertical,
      mode="all")
  transfer_eval_input_function = functools.partial(
      model_util.joint_input_fn,
      get_data_path(
          vertical=vertical, website=target_website, dev=False, goldmine=False),
      get_data_path(
          vertical=vertical, website=target_website, dev=False, goldmine=True),
      vertical,
      mode="all")
  preds_gen = estimator.predict(transfer_eval_input_function)
  prediction_str = ""
  if FLAGS.extract_node_emb:
    node_embs = []
  for gold, pred in zip(golds_gen, preds_gen):
    if FLAGS.circle_features:
      ((nnodes), (_), (words_list, words_len), (_, _), (_, _),
       (partner_words, _), (friend_words, _), (_, _), (_, _),
       (html_path, xpath_list), (_, _), (_, _), (_)), tags = gold

      for index in range(nnodes):
        normalized_partner = []
        for w in partner_words[index]:
          normalized_partner.append(normalize_text(w))

        if FLAGS.match_keywords:
          normalized_word = [
              normalize_text(w)
              for w in words_list[index][:words_len[index]]
          ]
          candicate_labels = constants.ATTRIBUTES[vertical]
          print("Partner: %s, Words: %s, Pred: %s" %
                (" ".join(normalized_partner), " ".join(normalized_word),
                 pred["tags"][index]))
          normalized_partner = " ".join(normalized_partner)
          for i, l in enumerate(candicate_labels):
            l = str(l).lower().replace("tor", "t").split("_")
            status = all([x in normalized_partner for x in l])
            if status:
              print("OLD:", pred["tags"][index])
              print("NEW:", candicate_labels[i].encode())
              pred["tags"][index] = candicate_labels[i].encode()

        if FLAGS.friend_encoder:
          normalized_friend = []
          for w in friend_words[index]:
            normalized_friend.append(normalize_text(w))
          print(normalized_friend)
          print(pred["friends_embs"][index])

    else:
      ((nnodes), (words_list, words_len), (_, _), (_, _), (_, _),
       (html_path, xpath_list), (_, _), (_), (_)), tags = gold
    assert nnodes == len(words_list) == len(tags)
    for index in range(nnodes):
      s = "\t".join([
          str(html_path, "utf-8"),
          str(xpath_list[index], "utf-8"),
          " ".join([
              str(w, "utf-8") for w in words_list[index][:int(words_len[index])]
          ]),
          str(tags[index], "utf-8"),
          str(pred["tags"][index], "utf-8"),
          ",".join([str(score) for score in pred["raw_scores"][index]]),
      ]) + "\n"
      prediction_str += s
      if FLAGS.extract_node_emb:
        node_embs.append([float(i) for i in pred["node_embs"][index]])

  with tf.gfile.Open(pred_filename, "w") as f:
    f.write(prediction_str)

  node_embs = np.array(node_embs)
  # Save np.array to file.
  with tf.gfile.Open(node_emb_filename, "wb") as gfo:
    print("Writing node emb pickle: %s" % node_emb_filename, file=sys.stderr)
    pickle.dump(node_embs, gfo)
    print("Node Representation Save- done.", file=sys.stderr)


def main(_):

  # Modify the paths to save results when tuning hyperparameters.
  if FLAGS.node_encoder == "lstm":
    FLAGS.result_path = os.path.join(FLAGS.result_path,
                                     str(FLAGS.node_lstm_size))
  if FLAGS.node_encoder == "transformer":
    FLAGS.result_path = os.path.join(
        FLAGS.result_path, "max_steps_" + str(FLAGS.max_steps_no_increase))
    FLAGS.result_path = os.path.join(
        FLAGS.result_path, "hidden_unit_" + str(FLAGS.transformer_hidden_unit))
  if FLAGS.cross_vertical:
    FLAGS.result_path = os.path.join(
        FLAGS.result_path, "CKP-{0}/{1}/".format(FLAGS.checkpoint_vertical,
                                                 FLAGS.checkpoint_websites))
    FLAGS.checkpoint_path = os.path.join(
        FLAGS.checkpoint_path,
        "{0}/{1}-results/".format(FLAGS.checkpoint_vertical,
                                  FLAGS.checkpoint_websites))

  tf.gfile.MakeDirs(
      os.path.join(
          FLAGS.result_path, "{0}/{1}-results/".format(FLAGS.vertical,
                                                       FLAGS.source_website)))
  tf.logging.set_verbosity(tf.logging.INFO)

  if FLAGS.use_uniform_embedding:
    vocab_vertical = "all"
  else:
    vocab_vertical = FLAGS.vertical

  # Hyper-parameters.
  params = {
      "add_goldmine":
          FLAGS.add_goldmine,
      "add_leaf_types":
          FLAGS.add_leaf_types,
      "batch_size":
          FLAGS.batch_size,
      "buffer":
          1000,  # Buffer for shuffling. No need to care about.
      "chars":
          os.path.join(FLAGS.domtree_data_path,
                       "%s.vocab.chars.txt" % vocab_vertical),
      "circle_features":
          FLAGS.circle_features,
      "dim_word_embedding":
          FLAGS.dim_word_embedding,
      "dim_chars":
          FLAGS.dim_chars,
      "dim_label_embedding":
          FLAGS.dim_label_embedding,
      "dim_goldmine":
          30,
      "dim_leaf_type":
          20,
      "dim_positions":
          30,
      "dim_xpath_units":
          FLAGS.dim_xpath_units,
      "dropout":
          0.3,
      "epochs":
          FLAGS.epochs,
      "extract_node_emb":
          FLAGS.extract_node_emb,
      "filters":
          50,  # The dimension of char-level word representations.
      "friend_encoder":
          FLAGS.friend_encoder,
      "use_friend_semantic":
          FLAGS.use_friend_semantic,
      "goldmine_features":
          os.path.join(FLAGS.domtree_data_path, "vocab.goldmine_features.txt"),
      "glove":
          os.path.join(
              FLAGS.domtree_data_path,
              "%s.%d.emb.npz" % (vocab_vertical, FLAGS.dim_word_embedding)),
      "friend_hidden_size":
          FLAGS.friend_hidden_size,
      "kernel_size":
          3,  # CNN window size to embed char sequences.
      "last_hidden_layer_size":
          FLAGS.last_hidden_layer_size,
      "leaf_types":
          os.path.join(FLAGS.domtree_data_path,
                       "%s.vocab.leaf_types.txt" % vocab_vertical),
      "lstm_size":
          100,
      "max_steps_no_increase":
          FLAGS.max_steps_no_increase,
      "node_encoder":
          FLAGS.node_encoder,
      "node_filters":
          100,
      "node_kernel_size":
          5,
      "node_lstm_size":
          FLAGS.node_lstm_size,
      "num_oov_buckets":
          1,
      "objective":
          FLAGS.objective,
      "positions":
          os.path.join(FLAGS.domtree_data_path, "vocab.positions.txt"),
      "running_mode":
          FLAGS.run,
      "semantic_encoder":
          FLAGS.semantic_encoder,
      "source_website":
          FLAGS.source_website,
      "tags":
          os.path.join(FLAGS.domtree_data_path,
                       "%s.vocab.tags.txt" % (FLAGS.vertical)),
      "tags-all":
          os.path.join(FLAGS.domtree_data_path, "all.vocab.tags.txt"),
      "target_website":
          FLAGS.target_website,
      "transformer_hidden_unit":
          FLAGS.transformer_hidden_unit,
      "transformer_head":
          FLAGS.transformer_head,
      "transformer_hidden_layer":
          FLAGS.transformer_hidden_layer,
      "use_crf":
          FLAGS.use_crf,
      "use_friends_cnn":
          FLAGS.use_friends_cnn,
      "use_friends_discrete_feature":
          FLAGS.use_friends_discrete_feature,
      "use_prev_text_lstm":
          FLAGS.use_prev_text_lstm,
      "use_xpath_lstm":
          FLAGS.use_xpath_lstm,
      "use_uniform_label":
          FLAGS.use_uniform_label,
      "use_position_embedding":
          FLAGS.use_position_embedding,
      "words":
          os.path.join(FLAGS.domtree_data_path,
                       "%s.vocab.words.txt" % vocab_vertical),
      "xpath_lstm_size":
          100,
      "xpath_units":
          os.path.join(FLAGS.domtree_data_path,
                       "%s.vocab.xpath_units.txt" % vocab_vertical),
  }
  with tf.gfile.Open(
      os.path.join(
          FLAGS.result_path,
          "{0}/{1}-results/params.json".format(FLAGS.vertical,
                                               FLAGS.source_website)),
      "w") as f:
    json.dump(params, f, indent=4, sort_keys=True)
  # Build estimator, train and evaluate.
  train_input_function = functools.partial(
      model_util.joint_input_fn,
      get_data_path(
          vertical=FLAGS.vertical,
          website=FLAGS.source_website,
          dev=False,
          goldmine=False),
      get_data_path(
          vertical=FLAGS.vertical,
          website=FLAGS.source_website,
          dev=False,
          goldmine=True),
      FLAGS.vertical,
      params,
      shuffle_and_repeat=True,
      mode="train")

  cfg = tf.estimator.RunConfig(
      save_checkpoints_steps=300, save_summary_steps=300, tf_random_seed=42)
  # Set up the checkpoint to load.
  if FLAGS.checkpoint_path:
    # The best model was always saved in "cpkt-601".
    checkpoint_file = FLAGS.checkpoint_path + "/model/model.ckpt-601"
    # Do not load parameters whose names contain the "label_dense".
    # These parameters are ought to be learned from scratch.
    ws = tf.estimator.WarmStartSettings(
        ckpt_to_initialize_from=checkpoint_file,
        vars_to_warm_start="^((?!label_dense).)*$")
    estimator = tf.estimator.Estimator(
        models.joint_extraction_model_fn,
        os.path.join(
            FLAGS.result_path,
            "{0}/{1}-results/model".format(FLAGS.vertical,
                                           FLAGS.source_website)),
        cfg,
        params,
        warm_start_from=ws)
  else:
    estimator = tf.estimator.Estimator(
        models.joint_extraction_model_fn,
        os.path.join(
            FLAGS.result_path,
            "{0}/{1}-results/model".format(FLAGS.vertical,
                                           FLAGS.source_website)), cfg, params)

  tf.gfile.MakeDirs(estimator.eval_dir())

  hook = early_stopping.stop_if_no_increase_hook(
      estimator,
      metric_name="f1",
      max_steps_without_increase=FLAGS.max_steps_no_increase,
      min_steps=300,
      run_every_steps=100,
      run_every_secs=None)
  train_spec = tf.estimator.TrainSpec(
      input_fn=train_input_function, hooks=[hook])

  if FLAGS.run == "train":
    eval_input_function = functools.partial(
        model_util.joint_input_fn,
        get_data_path(
            vertical=FLAGS.vertical,
            website=FLAGS.source_website,
            dev=True,
            goldmine=False),
        get_data_path(
            vertical=FLAGS.vertical,
            website=FLAGS.source_website,
            dev=True,
            goldmine=True),
        FLAGS.vertical,
        mode="all")
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_function, steps=300, throttle_secs=1)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

  target_websites = FLAGS.target_website.split("_")
  if FLAGS.source_website not in target_websites:
    target_websites = [FLAGS.source_website] + target_websites
  for target_website in target_websites:
    write_predictions(
        estimator=estimator,
        vertical=FLAGS.vertical,
        source_website=FLAGS.source_website,
        target_website=target_website)
    model_util.page_hits_level_metric(
        result_path=FLAGS.result_path,
        vertical=FLAGS.vertical,
        source_website=FLAGS.source_website,
        target_website=target_website)
    model_util.site_level_voting(
        result_path=FLAGS.result_path,
        vertical=FLAGS.vertical,
        source_website=FLAGS.source_website,
        target_website=target_website)
    model_util.page_level_constraint(
        domtree_data_path=FLAGS.domtree_data_path,
        result_path=FLAGS.result_path,
        vertical=FLAGS.vertical,
        source_website=FLAGS.source_website,
        target_website=target_website)


if __name__ == "__main__":
  app.run(main)
