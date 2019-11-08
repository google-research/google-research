# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Training related functions and trainer.
"""
import datetime
import os
import random
import time
import numpy as np
import tensorflow as tf
from property_linking.src import util
from tensorflow.contrib import memory_stats as contrib_memory_stats

# Uses these global flags:
#  learning_rate
#  batch_size
#  num_epochs
#  max_properties
#  log_frequency

FLAGS = tf.flags.FLAGS


class Trainer(object):
  """Trainer class to handle both distant training and evaluation.
  """

  def __init__(self,
               builder,
               context,
               model,
               train_examples,
               test_examples,
               loss_type,
               save_ckpt_dir=None,
               restore_ckpt_dir=None):
    """Create a trainer object for both train and eval.

    Args:
      builder: builder to map symbols back to names (for logging)
      context: KB/context object
      model: Model to be trained
      train_examples: training examples
      test_examples: test examples
      loss_type: whether to use a direct learning objective
      save_ckpt_dir: optional path to save a checkpoint
      restore_ckpt_dir: optional path to restore a checkpoint
    """
    self.context = context
    self.builder = builder
    self.model = model
    self.train_examples = train_examples
    self.test_examples = test_examples
    self.lr = FLAGS.learning_rate
    self.batch_size = FLAGS.batch_size
    self.restore_ckpt_dir = restore_ckpt_dir
    self.ckpt_dir = save_ckpt_dir
    with tf.variable_scope("trainer", reuse=tf.AUTO_REUSE):
      if loss_type == "distant":
        loss = model.loss
      elif loss_type == "direct":
        loss = model.property_loss
      elif loss_type == "mixed":
        loss = model.loss + model.property_loss
      else:
        raise ValueError("{} is not a valid loss type".format(loss_type))
      self.train_step = tf.train.AdagradOptimizer(self.lr).minimize(
          loss,
          colocate_gradients_with_ops=True
      )
    self.saver = tf.train.Saver()
    if not self.test_examples:
      tf.logging.info("No test examples: dev evaluation will return nan.")

  def restore_model(self, sess):
    if self.restore_ckpt_dir is not None:
      self.saver.restore(sess, "{}/model.ckpt".format(self.restore_ckpt_dir))
      tf.logging.info("Restored from {}".format(self.restore_ckpt_dir))
    else:
      tf.logging.info("Nowhere to restore from")

  def train(self, sess):
    """Main training function/loop.

    Args:
      sess: a tf session object
    """
    # For debugging/pushing limits of model
    gpu_mb = tf.constant(1024*1024, dtype=tf.int64)
    gpus = tf.config.experimental.list_logical_devices("GPU")
    memory_footprints = []
    for gpu in gpus:
      with tf.device(gpu.name):
        memory_footprint = tf.Print(
            tf.constant(0), [
                contrib_memory_stats.BytesLimit() / gpu_mb,
                contrib_memory_stats.MaxBytesInUse() / gpu_mb
            ],
            message=gpu.name)
      memory_footprints.append(memory_footprint)

    epochs = FLAGS.num_epochs
    prints = FLAGS.log_frequency

    training_start_time = time.time()
    epochs_start_time = time.time()

    num_batches = max(int(len(self.train_examples)/self.batch_size), 1)
    tf.logging.info("Num batches per epoch: {}".format(num_batches))

    # Additional logging
    losses = np.zeros((epochs * num_batches))
    accuracies = np.zeros((epochs * num_batches))

    for epoch in range(epochs):
      random.shuffle(self.train_examples)
      for batch in range(num_batches):
        batch_no = epoch * num_batches + batch
        should_sample = (batch_no % prints == 0)

        train_ops_to_run = {
            "train_step": self.train_step,
            "loss": self.model.loss,
            "accuracy": self.model.accuracy,
            "accuracy_per_example": self.model.accuracy_per_ex,
            "output_relations": self.model.log_decoded_relations,
        }
        if should_sample:
          train_ops_to_run["props"] = self.model.property_loss
          train_ops_to_run["regularization"] = self.model.regularization
          for i, memory_footprint in enumerate(memory_footprints):
            train_ops_to_run["memory_footprint_{}".format(i)] = memory_footprint

        batch_examples = self.train_examples[batch:
                                             batch + self.batch_size]
        feed_dict = self._compute_feed_dict(batch_examples)
        train_output = sess.run(train_ops_to_run, feed_dict)
        losses[batch_no] = train_output["loss"]
        accuracies[batch_no] = train_output["accuracy"]

        if should_sample:
          # Timing info
          epochs_end_time = time.time()
          epochs_time_str = str(datetime.timedelta(
              seconds=epochs_end_time - epochs_start_time))
          epochs_start_time = epochs_end_time
          precision, recall = self._evaluate_sample(sess,
                                                    train_output,
                                                    feed_dict,
                                                    batch_examples,
                                                    full_log=True)
          if precision and recall:
            pr_string = "\tPrecision: {:.3f}\tRecall {:.3f}".format(
                np.mean(precision), np.mean(recall))
          else:
            pr_string = ""
          tf.logging.info(
              ("[{}] Epoch: {}.{}\tLoss: {:.3f}|{:.3f}|{:.3f}\t" +
               "Accuracy: {:.3f}{}\n").format(
                   epochs_time_str,
                   epoch, batch,
                   train_output["loss"],
                   train_output["props"],
                   train_output["regularization"],
                   train_output["accuracy"],
                   pr_string))

          # Do a dev run, it doesn't take that long
          self.evaluate(sess, full=False)

    training_end_time = time.time()
    tf.logging.info("Training took: %s" % str(datetime.timedelta(
        seconds=training_end_time - training_start_time)))
    if self.ckpt_dir is not None:
      save_path = self.saver.save(sess,
                                  os.path.join(self.ckpt_dir, "model.ckpt"))
      tf.logging.info("Saved model at {}".format(save_path))

  def evaluate(self, sess, full=True):
    """Eval function/loop.

    Args:
      sess: a tf session object
      full: whether this is the final eval
    """
    eval_ops_to_run = {
        "accuracy_per_example": self.model.accuracy_per_ex,
        "output_relations": self.model.log_decoded_relations,
    }
    accuracies = []
    precision = []
    recall = []
    for example_number, example in enumerate(self.test_examples):
      feed_dict = self._compute_feed_dict([example], is_training=False)
      eval_output = sess.run(eval_ops_to_run, feed_dict)
      accuracies.append(eval_output["accuracy_per_example"])
      # print a lot but don't overwhelm log
      if full:
        full_log = example_number % 10 == 0
        if full_log:
          tf.logging.info("Example: {}".format(example_number))
        p, r = self._evaluate_sample(sess,
                                     eval_output,
                                     feed_dict,
                                     [example],
                                     full_log=full_log)
        precision.extend(p)
        recall.extend(r)
    if precision and recall:
      tf.logging.info(
          "Dev accuracy: {:.3f} p: {:.3f} r: {:.3f} f1: {:.3f}".format(
              np.mean(accuracies),
              np.mean(precision),
              np.mean(recall),
              2.0 / (1/np.mean(precision) + 1/np.mean(recall)),
          ))
    else:
      tf.logging.info("Dev accuracy: {:.3f}".format(np.mean(accuracies)))

  def _compute_feed_dict(self, batch_examples, is_training=True):
    """Preprocessing function to compute a feed_dict given examples.

    Args:
      batch_examples: list of (query, labels, query_text, prior_starts)
                      tuples
      is_training: bool to compute training vs test feed dict

    More info:
      query: ((max_seq_len, emb_size), # encoding
              (max_seq_len)            # binary mask
      labels: (num_core_entities)  # binary representation of set
      query_text: string
      prior_starts: sparse [1 * num_noncore_entities] matrix
      properties: gold (value, relation) ids if they exist, None otherwise

    Returns:
      feed_dict to be used with self.model
    """
    (queries, labels, _, prior_starts, properties) = zip(*batch_examples)
    batch_k_hot = np.asarray([util.k_hot_numpy_array(self.context, y, "id_t")
                              for y in labels])
    targets = np.squeeze(batch_k_hot, axis=1)
    oh_inputs, oh_len_masks = zip(*queries)
    prior_indices, prior_values = util.sparse_to_sparse_batch(prior_starts)
    output_size = len(self.context.get_symbols("val_g"))
    feed_dict = {
        self.model.input_ph: oh_inputs,
        self.model.mask_ph: oh_len_masks,
        self.model.correct_set_ph.name: targets,
        self.model.prior_start: (prior_indices, prior_values,
                                 np.array([len(prior_starts),
                                           output_size])),
        self.model.is_training: is_training,
    }
    if any(properties):
      batch_vals_k_hot = np.asarray(
          [util.k_hot_numpy_array(
              self.context, [prop[1] for prop in props[0]], "val_g")
           for props in properties])
      batch_rels_k_hot = np.asarray(
          [util.k_hot_numpy_array(
              self.context, [prop[0] for prop in props[0]], "rel_g")
           for props in properties])
      feed_dict[self.model.correct_vals] = np.squeeze(batch_vals_k_hot, axis=1)
      feed_dict[self.model.correct_rels] = np.squeeze(batch_rels_k_hot, axis=1)
    return feed_dict

  def _get_relation_ids(self, relations, full_name=True):
    """Get entity id of relation from context.

    Uses name_dict instead of nql kb because name_dict is faster.
    Args:
      relations: list of relation indices
      full_name: flag indicating whether to return full name or just the id

    Returns:
      list of name stings corresponding to each of the relation indices
    """
    # Get entity id for each relation index
    relation_seq = [self.context.get_entity_name(relation, "rel_g")
                    for relation in relations]
    if full_name:
      return self.builder.get_names(relation_seq)
    else:
      return relation_seq

  def _evaluate_sample(self, sess, tf_output,
                       feed_dict, examples, full_log=False):
    """Sample from model and print it to stdout.

    The tf_output and feed_dict are passed to minimize calling sess.run.
    Args:
      sess: tf session
      tf_output: output of sess.run
      feed_dict: feed_dict used in sess.run
      examples: examples used in sess.run
      full_log: optional - whether to print evaluation of sample

    Returns:
      precision and recall for each example
    """

    (_, labels, query_text, _, properties) = zip(*examples)

    # Set up internal formatting function that evals nql expressions
    def eval_and_format_nql(nql_expr):
      nql_expr_eval = nql_expr.eval(sess,
                                    feed_dict=feed_dict,
                                    as_top=10,
                                    simplify_unitsize_minibatch=False)
      kb_ids, weights = zip(*nql_expr_eval[0])
      return zip(self.builder.get_names(list(kb_ids)), weights)

    # Evaluate NQL expressions
    pred_set_eval = eval_and_format_nql(self.model.log_nql_pred_set)
    start_values = [eval_and_format_nql(values)
                    for values in self.model.log_start_values]
    start_values_cmps = [[eval_and_format_nql(values)
                          for values in timestep]
                         for timestep in self.model.log_start_cmps]
    label_names = self.builder.get_names(labels[0])
    output_relations = tf_output["output_relations"]
    relation_seq = [self._get_relation_ids(relations_at_time)
                    for relations_at_time in output_relations.indices[0]]

    relation_seq_per_step = [zip(relations, val) for relations, val in
                             zip(relation_seq, output_relations.values[0])]
    relation_pairs = ["{}".format(util.set_to_string(relations))
                      for relations in relation_seq_per_step]
    if full_log:
      tf.logging.info("properties: {}".format(properties[0]))
      tf.logging.info("Input (prior, sim): {}\n".format(query_text[0]))
      for t in range(FLAGS.max_properties):
        tf.logging.info(
            "Starts at timestep {}:\n {}\n{}\n{}\n Relation: {}\n".format(
                t,
                util.set_to_string(start_values[t]),
                util.set_to_string(start_values_cmps[t][0]),
                util.set_to_string(start_values_cmps[t][1]),
                relation_pairs[t]))
      tf.logging.info("Pred: {}\nGold: [{:.3f}]: {}\n".format(
          util.set_to_string(pred_set_eval),
          tf_output["accuracy_per_example"][0],
          sorted(label_names)))

    # Calculate and return precision and recall. We recompute nql expressions
    # because we need to do it for all examples in a batch and the indexing
    # is easier to reason about separately.

    # (timesteps * batch_size * 1), each element is a (name, weight) tuple
    output_start_values = [start_value.eval(
        sess, feed_dict=feed_dict, as_top=1, simplify_unitsize_minibatch=False)
                           for start_value in self.model.log_start_values]
    # (batch_size * timesteps * 1)
    output_relations = [[self._get_relation_ids(relations, full_name=False)
                         for relations in example_relations]
                        for example_relations in output_relations.indices]
    precision = []
    recall = []
    properties = [props[0] for props in properties]
    if any(properties):
      for i, gold in enumerate(properties):
        if properties is not None and gold:
          predicted = set()
          for t in range(FLAGS.max_properties):
            predicted_value = output_start_values[t][i][0][0]
            predicted_relation = output_relations[i][t][0]
            predicted.add((predicted_relation, predicted_value))
            intersection = predicted.intersection(gold)
          precision.append(float(len(intersection)) / len(predicted))
          recall.append(float(len(intersection)) / len(gold))
    return (precision, recall)
