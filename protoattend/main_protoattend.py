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

"""Demo code to run training, testing and analysis of attention-based prototypical learning for Fashion-MNIST dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import time

import input_data
import model
import numpy as np
from options import FLAGS
import tensorflow.compat.v1 as tf
import utils

# GPU options

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# File names

model_name = os.path.basename(__file__).split(".")[0]
checkpoint_name = "./checkpoints/" + model_name + ".ckpt"
export_name = os.path.join("exports", time.strftime("%Y%m%d-%H%M%S"))

# Set random seed
tf.set_random_seed(FLAGS.random_seed)
np.random.seed(FLAGS.random_seed)


def inference(input_image, m_encoded_test_cand_keys, m_encoded_test_cand_values,
              m_label_test_cand):
  """Constructs inference graph."""
  processed_input_image = input_data.parse_function_test(input_image)
  _, encoded_query, _ = model.cnn_encoder(
      processed_input_image, reuse=False, is_training=False)
  weighted_encoded_test, weight_coefs_test = model.relational_attention(
      encoded_query,
      tf.constant(m_encoded_test_cand_keys),
      tf.constant(m_encoded_test_cand_values),
      reuse=False)
  _, prediction_weighted_test = model.classify(
      weighted_encoded_test, reuse=False)
  predicted_class = tf.argmax(prediction_weighted_test, axis=1)
  expl_per_class = tf.py_func(
      utils.class_explainability,
      (tf.constant(m_label_test_cand), weight_coefs_test), tf.float32)
  confidence = tf.reduce_max(expl_per_class, axis=1)

  return predicted_class, confidence, weight_coefs_test


def main(unused_argv):
  """Main function."""

  # Load training and eval data - this portion can be modified if the data is
  # imported from other sources.
  (m_train_data, m_train_labels), (m_eval_data, m_eval_labels) = \
    tf.keras.datasets.fashion_mnist.load_data()
  train_dataset = tf.data.Dataset.from_tensor_slices(
      (m_train_data, m_train_labels))
  eval_dataset = tf.data.Dataset.from_tensor_slices(
      (m_eval_data, m_eval_labels))

  train_dataset = train_dataset.map(input_data.parse_function_train)
  eval_dataset = eval_dataset.map(input_data.parse_function_eval)
  eval_batch_size = int(
      math.floor(len(m_eval_data) / FLAGS.batch_size) * FLAGS.batch_size)

  train_batch = train_dataset.repeat().batch(FLAGS.batch_size)
  train_cand = train_dataset.repeat().batch(FLAGS.example_cand_size)
  eval_cand = train_dataset.repeat().batch(FLAGS.eval_cand_size)
  eval_batch = eval_dataset.repeat().batch(eval_batch_size)

  iter_train = train_batch.make_initializable_iterator()
  iter_train_cand = train_cand.make_initializable_iterator()
  iter_eval_cand = eval_cand.make_initializable_iterator()
  iter_eval = eval_batch.make_initializable_iterator()

  image_batch, _, label_batch = iter_train.get_next()
  image_train_cand, _, _ = iter_train_cand.get_next()
  image_eval_cand, orig_image_eval_cand, label_eval_cand = iter_eval_cand.get_next(
  )
  eval_batch, orig_eval_batch, eval_labels = iter_eval.get_next()

  # Model and loss definitions
  _, encoded_batch_queries, encoded_batch_values = model.cnn_encoder(
      image_batch, reuse=False, is_training=True)
  encoded_cand_keys, _, encoded_cand_values = model.cnn_encoder(
      image_train_cand, reuse=True, is_training=True)

  weighted_encoded_batch, weight_coefs_batch = model.relational_attention(
      encoded_batch_queries,
      encoded_cand_keys,
      encoded_cand_values,
      normalization=FLAGS.normalization)

  tf.summary.scalar("Average max. coef. train",
                    tf.reduce_mean(tf.reduce_max(weight_coefs_batch, axis=1)))

  # Sparsity regularization
  entropy_weights = tf.reduce_sum(
      -weight_coefs_batch * tf.log(FLAGS.epsilon_sparsity + weight_coefs_batch),
      axis=1)
  sparsity_loss = tf.reduce_mean(entropy_weights) - tf.log(
      FLAGS.epsilon_sparsity +
      tf.constant(FLAGS.example_cand_size, dtype=tf.float32))
  tf.summary.scalar("Sparsity entropy loss", sparsity_loss)

  # Intermediate loss
  joint_encoded_batch = (1 - FLAGS.alpha_intermediate) * encoded_batch_values \
    + FLAGS.alpha_intermediate * weighted_encoded_batch

  logits_joint_batch, _ = model.classify(joint_encoded_batch, reuse=False)
  softmax_joint_op = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits_joint_batch, labels=label_batch))

  # Self loss
  logits_orig_batch, _ = model.classify(encoded_batch_values, reuse=True)
  softmax_orig_key_op = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits_orig_batch, labels=label_batch))

  # Prototype combination loss
  logits_weighted_batch, _ = model.classify(weighted_encoded_batch, reuse=True)
  softmax_weighted_op = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits_weighted_batch, labels=label_batch))

  train_loss_op = softmax_orig_key_op + softmax_weighted_op + \
    softmax_joint_op + FLAGS.sparsity_weight * sparsity_loss
  tf.summary.scalar("Total loss", train_loss_op)

  global_step = tf.train.get_or_create_global_step()
  learning_rate = tf.train.exponential_decay(
      FLAGS.init_learning_rate,
      global_step=global_step,
      decay_steps=FLAGS.decay_every,
      decay_rate=FLAGS.decay_rate)
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  tf.summary.scalar("Learning rate", learning_rate)
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    gvs = optimizer.compute_gradients(train_loss_op)
    capped_gvs = [(tf.clip_by_value(grad, -FLAGS.gradient_thresh,
                                    FLAGS.gradient_thresh), var)
                  for grad, var in gvs]
    train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

  # Evaluate model

  # Process sequentially to avoid out-of-memory.
  i = tf.constant(0)
  encoded_cand_keys_val = tf.zeros([0, FLAGS.attention_dim])
  encoded_cand_queries_val = tf.zeros([0, FLAGS.attention_dim])
  encoded_cand_values_val = tf.zeros([0, FLAGS.val_dim])

  def cond(i, unused_l1, unused_l2, unused_l3):
    return i < int(math.ceil(FLAGS.eval_cand_size / FLAGS.example_cand_size))

  def body(i, encoded_cand_keys_val, encoded_cand_queries_val,
           encoded_cand_values_val):
    """Loop body."""
    temp = image_eval_cand[i * FLAGS.example_cand_size:(i + 1) *
                           FLAGS.example_cand_size, :, :, :]
    temp_keys, temp_queries, temp_values = model.cnn_encoder(
        temp, reuse=True, is_training=False)
    encoded_cand_keys_val = tf.concat([encoded_cand_keys_val, temp_keys], 0)
    encoded_cand_queries_val = tf.concat(
        [encoded_cand_queries_val, temp_queries], 0)
    encoded_cand_values_val = tf.concat([encoded_cand_values_val, temp_values],
                                        0)
    return i+1, encoded_cand_keys_val, encoded_cand_queries_val, \
        encoded_cand_values_val

  _, encoded_cand_keys_val, encoded_cand_queries_val, \
      encoded_cand_values_val, = tf.while_loop(
          cond, body, [i, encoded_cand_keys_val, encoded_cand_queries_val,
                       encoded_cand_values_val],
          shape_invariants=[
              i.get_shape(), tf.TensorShape([None, FLAGS.attention_dim]),
              tf.TensorShape([None, FLAGS.attention_dim]),
              tf.TensorShape([None, FLAGS.val_dim])])

  j = tf.constant(0)
  encoded_val_keys = tf.zeros([0, FLAGS.attention_dim])
  encoded_val_queries = tf.zeros([0, FLAGS.attention_dim])
  encoded_val_values = tf.zeros([0, FLAGS.val_dim])

  def cond2(j, unused_j1, unused_j2, unused_j3):
    return j < int(math.ceil(eval_batch_size / FLAGS.batch_size))

  def body2(j, encoded_val_keys, encoded_val_queries, encoded_val_values):
    """Loop body."""
    temp = eval_batch[j * FLAGS.batch_size:(j + 1) * FLAGS.batch_size, :, :, :]
    temp_keys, temp_queries, temp_values = model.cnn_encoder(
        temp, reuse=True, is_training=False)
    encoded_val_keys = tf.concat([encoded_val_keys, temp_keys], 0)
    encoded_val_queries = tf.concat([encoded_val_queries, temp_queries], 0)
    encoded_val_values = tf.concat([encoded_val_values, temp_values], 0)
    return j + 1, encoded_val_keys, encoded_val_queries, encoded_val_values

  _, encoded_val_keys, encoded_val_queries, \
      encoded_val_values = tf.while_loop(
          cond2, body2, [
              j, encoded_val_keys, encoded_val_queries, encoded_val_values],
          shape_invariants=[
              j.get_shape(), tf.TensorShape([None, FLAGS.attention_dim]),
              tf.TensorShape([None, FLAGS.attention_dim]),
              tf.TensorShape([None, FLAGS.val_dim])])

  weighted_encoded_val, weight_coefs_val = model.relational_attention(
      encoded_val_queries,
      encoded_cand_keys_val,
      encoded_cand_values_val,
      normalization=FLAGS.normalization)

  # Coefficient distribution
  tf.summary.scalar("Average max. coefficient val",
                    tf.reduce_mean(tf.reduce_max(weight_coefs_val, axis=1)))

  # Analysis of median number of prototypes above a certain
  # confidence threshold.
  sorted_weights = tf.contrib.framework.sort(
      weight_coefs_val, direction="DESCENDING")
  cum_sorted_weights = tf.cumsum(sorted_weights, axis=1)
  for threshold in [0.5, 0.9, 0.95]:
    num_examples_thresh = tf.shape(sorted_weights)[1] + 1 - tf.reduce_sum(
        tf.cast(cum_sorted_weights > threshold, tf.int32), axis=1)
    tf.summary.histogram(
        "Number of samples for explainability above " + str(threshold),
        num_examples_thresh)
    tf.summary.scalar(
        "Median number of samples for explainability above " + str(threshold),
        tf.contrib.distributions.percentile(num_examples_thresh, q=50))

  expl_per_class = tf.py_func(utils.class_explainability,
                              (label_eval_cand, weight_coefs_val), tf.float32)
  max_expl = tf.reduce_max(expl_per_class, axis=1)
  tf.summary.histogram("Maximum per-class explainability", max_expl)

  _, prediction_val = model.classify(encoded_val_values, reuse=True)
  _, prediction_weighted_val = model.classify(weighted_encoded_val, reuse=True)

  val_eq_op = tf.equal(
      tf.cast(tf.argmax(prediction_val, 1), dtype=tf.int32), eval_labels)
  val_acc_op = tf.reduce_mean(tf.cast(val_eq_op, dtype=tf.float32))
  tf.summary.scalar("Val accuracy input query", val_acc_op)

  val_weighted_eq_op = tf.equal(
      tf.cast(tf.argmax(prediction_weighted_val, 1), dtype=tf.int32),
      eval_labels)
  val_weighted_acc_op = tf.reduce_mean(
      tf.cast(val_weighted_eq_op, dtype=tf.float32))
  tf.summary.scalar("Val accuracy weighted prototypes", val_weighted_acc_op)

  conf_wrong = tf.reduce_mean(
      (1 - tf.cast(val_weighted_eq_op, tf.float32)) * max_expl)
  tf.summary.scalar("Val average confidence of wrong decisions", conf_wrong)

  conf_right = tf.reduce_mean(
      tf.cast(val_weighted_eq_op, tf.float32) * max_expl)
  tf.summary.scalar("Val average confidence of right decisions", conf_right)

  # Confidence-controlled prediction
  for ti in [0.5, 0.8, 0.9, 0.95, 0.99, 0.999]:
    mask = tf.cast(tf.greater(max_expl, ti), tf.float32)
    acc_tot = tf.reduce_sum(tf.cast(val_weighted_eq_op, tf.float32) * mask)
    conf_tot = tf.reduce_sum(mask)

    tf.summary.scalar("Val accurate ratio for confidence above " + str(ti),
                      acc_tot / conf_tot)
    tf.summary.scalar("Val total ratio for confidence above " + str(ti),
                      conf_tot / eval_batch_size)

  # Visualization of example images and corresponding prototypes
  for image_ind in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    tf.summary.image("Input image " + str(image_ind),
                     tf.expand_dims(orig_eval_batch[image_ind, :, :, :], 0))
    mask = tf.greater(weight_coefs_val[image_ind, :], 0.05)
    mask = tf.squeeze(mask)
    mask.set_shape([None])
    relational_attention_images = tf.boolean_mask(
        orig_image_eval_cand, mask, axis=0)
    relational_attention_weight_coefs = tf.boolean_mask(
        tf.squeeze(weight_coefs_val[image_ind, :]), mask, axis=0)
    annotated_images = utils.tf_put_text(relational_attention_images,
                                         relational_attention_weight_coefs)
    tf.summary.image("Prototype images for image " + str(image_ind),
                     annotated_images)

  # Training setup
  init = (tf.global_variables_initializer(), tf.local_variables_initializer())
  saver_all = tf.train.Saver()
  summaries = tf.summary.merge_all()

  with tf.Session() as sess:

    summary_writer = tf.summary.FileWriter("./tflog/" + model_name, sess.graph)

    sess.run(init)
    sess.run(iter_train.initializer)
    sess.run(iter_train_cand.initializer)
    sess.run(iter_eval_cand.initializer)
    sess.run(iter_eval.initializer)

    for step in range(1, FLAGS.num_steps):
      if step % FLAGS.display_step == 0:
        _, train_loss = sess.run([train_op, train_loss_op])
        print("Step " + str(step) + " , Training loss = " +
              "{:.4f}".format(train_loss))
      else:
        sess.run(train_op)

      if step % FLAGS.val_step == 0:
        val_acc, merged_summary = sess.run([val_weighted_acc_op, summaries])
        print("Step " + str(step) + " , Val Accuracy = " +
              "{:.4f}".format(val_acc))
        summary_writer.add_summary(merged_summary, step)

      if step % FLAGS.save_step == 0:
        saver_all.save(sess, checkpoint_name)

if __name__ == "__main__":
  tf.app.run()
