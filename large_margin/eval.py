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

"""Evaluate model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

from large_margin import margin_loss
from large_margin.mnist import data_provider as mnist
from large_margin.mnist import mnist_config
from large_margin.mnist import mnist_model

flags.DEFINE_string("checkpoint_dir", "/tmp/large_margin/train/",
                    "Results directory.")
flags.DEFINE_string("eval_dir", "/tmp/large_margin/eval/",
                    "Evaluation directory.")
flags.DEFINE_string("subset", "test", "Data subset.")
flags.DEFINE_integer("batch_size", 10, "Training batch size.")
flags.DEFINE_enum("experiment_type", "mnist", ["mnist"], "Experiment type.")
flags.DEFINE_string("data_dir", "", "Data directory.")
flags.DEFINE_boolean("run_once", True,
                     "Whether to run eval only once or multiple times.")
flags.DEFINE_integer("eval_interval_secs", 2 * 60,
                     "The frequency with which evaluation is run.")

FLAGS = flags.FLAGS


def _eval_once(session_creator, ops_dict, summary_writer, merged_summary,
               global_step, num_examples):
  """Runs evaluation on the full data and saves results.

  Args:
    session_creator: session creator.
    ops_dict: dictionary with operations to evaluate.
    summary_writer: summary writer.
    merged_summary: merged summaries.
    global_step: global step.
    num_examples: number of examples in the data.
  """
  num_batches = int(num_examples / float(FLAGS.batch_size))

  list_ops = []
  list_phs = []
  list_keys = []
  for key, value in ops_dict.iteritems():
    if value[0] is not None and value[1] is not None:
      list_keys.append(key)
      list_ops.append(value[1])
      list_phs.append(value[0])

  with tf.train.MonitoredSession(session_creator=session_creator) as sess:
    list_results = []
    count = 0.
    total_correct = 0
    for _ in xrange(num_batches):
      res, top1 = sess.run((list_ops, ops_dict["top1"][1]))
      number_correct = np.sum(top1)
      total_correct += number_correct
      count += FLAGS.batch_size
      list_results.append(res)

    accuracy = total_correct / count
    mean_results = np.mean(np.array(list_results), axis=0)
    feed_dict = {ph: v for ph, v in zip(list_phs, list(mean_results))}
    feed_dict[ops_dict["top1_accuracy"][0]] = accuracy
    summary, g_step = sess.run((merged_summary, global_step), feed_dict)
    tf.logging.info("\n\n\n\n\n\n\n\n"
                    "Global step: %d \n"
                    "top1 accuracy on %s set is %.6f"
                    "\n\n\n\n\n\n\n\n" % (g_step, FLAGS.subset, accuracy))
    tf.logging.info(num_examples)
    tf.logging.info(count)
    summary_writer.add_summary(summary, global_step=g_step)


def evaluate():
  """Evaluating function."""
  g = tf.Graph()
  ops_dict = {}
  with g.as_default():
    # Data set.
    if FLAGS.experiment_type == "mnist":
      config = mnist_config.ConfigDict()
      dataset = mnist.MNIST(
          data_dir=FLAGS.data_dir,
          subset=FLAGS.subset,
          batch_size=FLAGS.batch_size,
          is_training=False)
      model = mnist_model.MNISTNetwork(config)

    images, labels, num_examples, num_classes = (dataset.images, dataset.labels,
                                                 dataset.num_examples,
                                                 dataset.num_classes)

    logits, _ = model(images, is_training=False)

    top1_op = tf.nn.in_top_k(logits, labels, 1)

    top1_op = tf.cast(top1_op, dtype=tf.float32)
    ops_dict["top1"] = (None, top1_op)
    accuracy_ph = tf.placeholder(tf.float32, None)
    ops_dict["top1_accuracy"] = (accuracy_ph, None)
    tf.summary.scalar("top1_accuracy", accuracy_ph)

    with tf.name_scope("optimizer"):
      global_step = tf.train.get_or_create_global_step()

    # Define losses.
    l2_loss_wt = config.l2_loss_wt
    xent_loss_wt = config.xent_loss_wt
    margin_loss_wt = config.margin_loss_wt
    gamma = config.gamma
    alpha = config.alpha
    top_k = config.top_k
    dist_norm = config.dist_norm
    with tf.name_scope("losses"):
      xent_loss = tf.reduce_mean(
          tf.nn.sparse_softmax_cross_entropy_with_logits(
              logits=logits, labels=labels))
      margin = margin_loss.large_margin(
          logits=logits,
          one_hot_labels=tf.one_hot(labels, num_classes),
          layers_list=[images],
          gamma=gamma,
          alpha_factor=alpha,
          top_k=top_k,
          dist_norm=dist_norm)
      l2_loss = 0.
      for v in tf.trainable_variables():
        tf.logging.info(v)
        l2_loss += tf.nn.l2_loss(v)

      total_loss = 0
      if xent_loss_wt > 0:
        total_loss += xent_loss_wt * xent_loss
      if margin_loss_wt > 0:
        total_loss += margin_loss_wt * margin
      if l2_loss_wt > 0:
        total_loss += l2_loss_wt * l2_loss

      xent_loss_ph = tf.placeholder(tf.float32, None)
      margin_loss_ph = tf.placeholder(tf.float32, None)
      l2_loss_ph = tf.placeholder(tf.float32, None)
      total_loss_ph = tf.placeholder(tf.float32, None)
      tf.summary.scalar("xent_loss", xent_loss_ph)
      tf.summary.scalar("margin_loss", margin_loss_ph)
      tf.summary.scalar("l2_loss", l2_loss_ph)
      tf.summary.scalar("total_loss", total_loss_ph)

      ops_dict["losses/xent_loss"] = (xent_loss_ph, xent_loss)
      ops_dict["losses/margin_loss"] = (margin_loss_ph, margin)
      ops_dict["losses/l2_loss"] = (l2_loss_ph, l2_loss)
      ops_dict["losses/total_loss"] = (total_loss_ph, total_loss)

    # Prepare evaluation session.
    merged_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir,
                                           tf.get_default_graph())
    vars_to_save = tf.global_variables()
    saver = tf.train.Saver(var_list=vars_to_save)
    scaffold = tf.train.Scaffold(saver=saver)
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=scaffold, checkpoint_dir=FLAGS.checkpoint_dir)
    while True:
      _eval_once(session_creator, ops_dict, summary_writer, merged_summary,
                 global_step, num_examples)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv):
  del argv  # Unused.
  evaluate()


if __name__ == "__main__":
  app.run(main)
