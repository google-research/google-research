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

"""Training and evaluation using Distance Based Learning from Errors (DBLE)."""

from __future__ import print_function

import argparse
import logging
import os
import sys
import time

import numpy as np
import pathlib
import tensorflow.compat.v1 as tf
from tqdm import trange

sys.path.insert(0, '..')
# pylint: disable=g-import-not-at-top
from dble import data_loader
from dble import mlp
from dble import resnet
from dble import utils
from dble import vgg
from tensorflow.contrib import slim as contrib_slim

tf.logging.set_verbosity(tf.logging.INFO)
logging.basicConfig(level=logging.INFO)

feature_dim = 512


class Namespace(object):

  def __init__(self, adict):
    self.__dict__.update(adict)


def get_arguments():
  """Processes all parameters."""
  parser = argparse.ArgumentParser()

  # Dataset parameters
  parser.add_argument(
      '--data_dir',
      type=str,
      default='',
      help='Path to the data, only for Tiny-ImageNet.')
  parser.add_argument(
      '--val_data_dir',
      type=str,
      default='',
      help='Path to the validation data, only for Tiny-ImageNet.')

  # Training parameters
  parser.add_argument(
      '--number_of_steps',
      type=int,
      default=int(120000),
      help='Number of training steps.')
  parser.add_argument(
      '--number_of_steps_to_early_stop',
      type=int,
      default=int(75000),
      help='Number of training steps after half way to early stop.')
  parser.add_argument(
      '--log_dir',
      type=str,
      default='/tmp/cifar_10/',
      help='experiment directory.')
  parser.add_argument(
      '--num_tasks_per_batch',
      type=int,
      default=2,
      help='Number of few shot episodes per batch.')
  parser.add_argument(
      '--init_learning_rate',
      type=float,
      default=0.1,
      help='Initial learning rate.')
  parser.add_argument(
      '--save_summaries_secs',
      type=int,
      default=300,
      help='Time between saving summaries')
  parser.add_argument(
      '--save_interval_secs',
      type=int,
      default=300,
      help='Time between saving models.')
  parser.add_argument(
      '--optimizer', type=str, default='sgd', choices=['sgd', 'adam'])

  # Optimization parameters
  parser.add_argument(
      '--lr_anneal',
      type=str,
      default='pwc',
      choices=['const', 'pwc', 'cos', 'exp'])
  parser.add_argument('--n_lr_decay', type=int, default=3)
  parser.add_argument('--lr_decay_rate', type=float, default=10.0)
  parser.add_argument(
      '--num_steps_decay_pwc',
      type=int,
      default=10000,
      help='Decay learning rate every num_steps_decay_pwc')
  parser.add_argument(
      '--clip_gradient_norm',
      type=float,
      default=1.0,
      help='gradient clip norm.')
  parser.add_argument(
      '--weights_initializer_factor',
      type=float,
      default=0.1,
      help='multiplier in the variance of the initialization noise.')

  # Evaluation parameters
  parser.add_argument(
      '--eval_interval_secs',
      type=int,
      default=0,
      help='Time between evaluating model.')
  parser.add_argument(
      '--eval_interval_steps',
      type=int,
      default=2000,
      help='Number of train steps between evaluating model in training loop.')
  parser.add_argument(
      '--eval_interval_fine_steps',
      type=int,
      default=1000,
      help='Number of train steps between evaluating model in the final phase.')

  # Architecture parameters
  parser.add_argument('--conv_keepdim', type=float, default=0.5)
  parser.add_argument('--neck', type=bool, default=False, help='')
  parser.add_argument('--num_forward', type=int, default=10, help='')
  parser.add_argument('--weight_decay', type=float, default=0.0005)
  parser.add_argument('--num_cases_train', type=int, default=50000)
  parser.add_argument('--num_cases_test', type=int, default=10000)
  parser.add_argument('--model_name', type=str, default='vgg')
  parser.add_argument('--dataset', type=str, default='cifar10')
  parser.add_argument(
      '--num_samples_per_class', type=int, default=5000, help='')
  parser.add_argument(
      '--num_classes_total',
      type=int,
      default=10,
      help='Number of classes in total of the data set.')
  parser.add_argument(
      '--num_classes_test',
      type=int,
      default=10,
      help='Number of classes in the test phase. ')
  parser.add_argument(
      '--num_classes_train',
      type=int,
      default=10,
      help='Number of classes in a protoypical episode.')
  parser.add_argument(
      '--num_shots_train',
      type=int,
      default=10,
      help='Number of shots (support samples) in a prototypical episode.')
  parser.add_argument(
      '--train_batch_size',
      type=int,
      default=100,
      help='The size of the query batch in a prototypical episode.')

  args, _ = parser.parse_known_args()
  print(args)
  return args


def build_feature_extractor_graph(inputs,
                                  flags,
                                  is_variance,
                                  is_training=False,
                                  model=None):
  """Calculates the representations and variances for inputs.

  Args:
    inputs: The input batch with shape (batch_size, height, width,
      num_channels). The batch_size of a support batch is num_classes_per_task
      *num_supports_per_class*num_tasks. The batch_size of a query batch is
      query_batch_size_per_task*num_tasks.
    flags: The hyperparameter dictionary.
    is_variance: The bool value of whether to calculate variances for every
      training sample. For support samples, calculating variaces is not
      required.
    is_training: The bool value of whether to use training mode.
    model: The representation model defined in function train(flags).

  Returns:
    h: The representations of the input batch with shape
    (batch_size, feature_dim).
    variance: The variances of the input batch with shape
    (batch_size, feature_dim).
  """
  variance = None
  with tf.variable_scope('feature_extractor', reuse=tf.AUTO_REUSE):
    h = model.encoder(inputs, training=is_training)
    if is_variance:
      variance = model.confidence_model(h, training=is_training)
    embedding_shape = h.get_shape().as_list()
    if is_training:
      h = tf.reshape(
          h,
          shape=(flags.num_tasks_per_batch,
                 embedding_shape[0] // flags.num_tasks_per_batch, -1),
          name='reshape_to_multi_task_format')
      if is_variance:
        variance = tf.reshape(
            variance,
            shape=(flags.num_tasks_per_batch,
                   embedding_shape[0] // flags.num_tasks_per_batch, -1),
            name='reshape_to_multi_task_format')
    else:
      h = tf.reshape(
          h,
          shape=(1, embedding_shape[0], -1),
          name='reshape_to_multi_task_format')
      if is_variance:
        variance = tf.reshape(
            variance,
            shape=(1, embedding_shape[0], -1),
            name='reshape_to_multi_task_format')

    return h, variance


def calculate_class_center(support_embeddings,
                           flags,
                           is_training,
                           scope='class_center_calculator'):
  """Calculates the class centers for every episode given support embeddings.

  Args:
    support_embeddings: The support embeddings with shape
      (num_classes_per_task*num_supports_per_class*num_tasks, height, width,
      num_channels).
    flags: The hyperparameter dictionary.
    is_training: The bool value of whether to use training mode.
    scope: The name of the variable scope.

  Returns:
    class_center: The representations of the class centers with shape
    (num_supports_per_class*num_tasks, feature_dim).
  """
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    class_center = support_embeddings
    if is_training:
      class_center = tf.reshape(
          class_center,
          shape=(flags.num_tasks_per_batch, flags.num_classes_train,
                 flags.num_shots_train, -1),
          name='reshape_to_multi_task_format')
    else:
      class_center = tf.reshape(
          class_center,
          shape=(1, flags.num_classes_test, flags.num_shots_train, -1),
          name='reshape_to_multi_task_format')
    class_center = tf.reduce_mean(class_center, axis=2, keep_dims=False)

    return class_center


def build_euclidean_calculator(query_representation,
                               class_center,
                               flags,
                               scope='prototypical_head'):
  """Calculates the negative Euclidean distance of queries to class centers.

  Args:
    query_representation: The query embeddings with shape (num_tasks,
      query_batch_size_per_task, feature_dim).
    class_center: The representations of class centers with shape (num_tasks,
      num_training_class, feature_dim).
    flags: The hyperparameter dictionary.
    scope: The name of the variable scope.

  Returns:
    negative_euclidean: The negative euclidean distance of queries to the class
     centers in their episodes. The shape of negative_euclidean is (num_tasks,
     query_batch_size_per_task, num_training_class).
  """
  with tf.variable_scope(scope):
    if len(query_representation.get_shape().as_list()) == 2:
      query_representation = tf.expand_dims(query_representation, axis=0)
    if len(class_center.get_shape().as_list()) == 2:
      class_center = tf.expand_dims(class_center, axis=0)

    # j is the number of classes in each episode
    # i is the number of queries in each episode
    j = class_center.get_shape().as_list()[1]
    i = query_representation.get_shape().as_list()[1]
    print('task_encoding_shape:' + str(j))

    # tile to be able to produce weight matrix alpha in (i,j) space
    query_representation = tf.expand_dims(query_representation, axis=2)
    class_center = tf.expand_dims(class_center, axis=1)
    # features_generic changes over i and is constant over j
    # task_encoding changes over j and is constant over i
    class_center_tile = tf.tile(class_center, (1, i, 1, 1))
    query_representation_tile = tf.tile(query_representation, (1, 1, j, 1))
    negative_euclidean = -tf.norm(
        (class_center_tile - query_representation_tile),
        name='neg_euclidean_distance',
        axis=-1)
    negative_euclidean = tf.reshape(
        negative_euclidean, shape=(flags.num_tasks_per_batch * i, -1))

    return negative_euclidean


def build_proto_train_graph(images_query, images_support, flags, is_training,
                            model):
  """Builds the tf graph of dble's prototypical training.

  Args:
    images_query: The processed query batch with shape
      (query_batch_size_per_task*num_tasks, height, width, num_channels).
    images_support: The processed support batch with shape
      (num_classes_per_task*num_supports_per_class*num_tasks, height, width,
      num_channels).
    flags: The hyperparameter dictionary.
    is_training: The bool value of whether to use training mode.
    model: The model defined in the main train function.

  Returns:
    logits: The logits before softmax (negative Euclidean) of the batch
    calculated with the original representations (mu in the paper) of queries.
    logits_z: The logits before softmax (negative Euclidean) of the batch
    calculated with the sampled representations (z in the paper) of queries.
  """

  with tf.variable_scope('Proto_training'):
    support_representation, _ = build_feature_extractor_graph(
        inputs=images_support,
        flags=flags,
        is_variance=False,
        is_training=is_training,
        model=model)
    class_center = calculate_class_center(
        support_embeddings=support_representation,
        flags=flags,
        is_training=is_training)
    query_representation, query_variance = build_feature_extractor_graph(
        inputs=images_query,
        flags=flags,
        is_variance=True,
        is_training=is_training,
        model=model)

    logits = build_euclidean_calculator(query_representation, class_center,
                                        flags)
    eps = tf.random.normal(shape=query_representation.shape)
    z = eps * tf.exp((query_variance) * .5) + query_representation
    logits_z = build_euclidean_calculator(z, class_center, flags)

  return logits, logits_z


def placeholder_inputs(batch_size, image_size, scope):
  """Builds the placeholders for the training images and labels."""
  with tf.variable_scope(scope):
    if image_size != 28:  # not mnist:
      images_placeholder = tf.placeholder(
          tf.float32,
          shape=(batch_size, image_size, image_size, 3),
          name='images')
    else:
      images_placeholder = tf.placeholder(
          tf.float32, shape=(batch_size, 784), name='images')
    labels_placeholder = tf.placeholder(
        tf.int32, shape=(batch_size), name='labels')
    return images_placeholder, labels_placeholder


def build_episode_placeholder(flags):
  """Builds the placeholders for the support and query input batches."""
  image_size = data_loader.get_image_size(flags.dataset)
  images_query_pl, labels_query_pl = placeholder_inputs(
      batch_size=flags.num_tasks_per_batch * flags.train_batch_size,
      image_size=image_size,
      scope='inputs/query')
  images_support_pl, labels_support_pl = placeholder_inputs(
      batch_size=flags.num_tasks_per_batch * flags.num_classes_train *
      flags.num_shots_train,
      image_size=image_size,
      scope='inputs/support')

  return images_query_pl, labels_query_pl, images_support_pl, labels_support_pl


def build_model(flags):
  """Builds model according to flags.

  For image data types, we considered ResNet and VGG models. One can use DBLE
  with other data types, by choosing a model with appropriate inductive bias
  for feature extraction, e.g. WaveNet for speech or BERT for text.
  Args:
    flags: The hyperparameter dictionary.

  Returns:
    mlp_model: the mlp model instance.
    vgg_model: the vgg model instance.
    resnet_model: the resnet model instance.
  """
  if flags.model_name == 'vgg':
    # Primary task operations
    vgg_model = vgg.vgg11(
        keep_prob=flags.conv_keepdim,
        wd=flags.weight_decay,
        neck=flags.neck,
        feature_dim=feature_dim)
    return vgg_model
  elif flags.model_name == 'mlp':
    mlp_model = mlp(
        keep_prob=flags.conv_keepdim,
        feature_dim=feature_dim,
        wd=flags.weight_decay)
    return mlp_model
  elif flags.model_name == 'resnet':
    if flags.dataset == 'cifar10' or flags.dataset == 'cifar100':
      resnet_model = resnet.Model(
          wd=flags.weight_decay,
          resnet_size=50,
          bottleneck=True,
          num_classes=flags.num_classes_train,
          num_filters=16,
          kernel_size=3,
          conv_stride=1,
          first_pool_size=None,
          first_pool_stride=None,
          block_sizes=[8, 8, 8],
          block_strides=[1, 2, 2],
          data_format='channels_last',
          feature_dim=feature_dim)
    else:
      resnet_model = resnet.Model(
          wd=flags.weight_decay,
          resnet_size=50,
          bottleneck=True,
          num_classes=flags.num_classes_train,
          num_filters=16,
          kernel_size=3,
          conv_stride=1,
          first_pool_size=3,
          first_pool_stride=1,
          block_sizes=[3, 4, 6, 3],
          block_strides=[1, 2, 2, 2],
          data_format='channels_last',
          feature_dim=feature_dim)
    return resnet_model


def train(flags):
  """Training entry point."""
  log_dir = flags.log_dir
  flags.pretrained_model_dir = log_dir
  log_dir = os.path.join(log_dir, 'train')
  flags.eval_interval_secs = 0
  with tf.Graph().as_default():
    global_step = tf.Variable(
        0, trainable=False, name='global_step', dtype=tf.int64)
    global_step_confidence = tf.Variable(
        0, trainable=False, name='global_step_confidence', dtype=tf.int64)

    model = build_model(flags)
    images_query_pl, labels_query_pl, \
    images_support_pl, labels_support_pl = \
      build_episode_placeholder(flags)

    # Augments the input.
    if flags.dataset == 'cifar10' or flags.dataset == 'cifar100':
      images_query_pl_aug = data_loader.augment_cifar(
          images_query_pl, is_training=True)
      images_support_pl_aug = data_loader.augment_cifar(
          images_support_pl, is_training=True)
    elif flags.dataset == 'tinyimagenet':
      images_query_pl_aug = data_loader.augment_tinyimagenet(
          images_query_pl, is_training=True)
      images_support_pl_aug = data_loader.augment_tinyimagenet(
          images_support_pl, is_training=True)

    logits, logits_z = build_proto_train_graph(
        images_query=images_query_pl_aug,
        images_support=images_support_pl_aug,
        flags=flags,
        is_training=True,
        model=model)
    # Losses and optimizer
    ## Classification loss
    loss_classification = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=logits,
            labels=tf.one_hot(labels_query_pl, flags.num_classes_train)))

    # Confidence loss
    _, top_k_indices = tf.nn.top_k(logits, k=1)
    pred = tf.squeeze(top_k_indices)
    incorrect_mask = tf.math.logical_not(tf.math.equal(pred, labels_query_pl))
    incorrect_logits_z = tf.boolean_mask(logits_z, incorrect_mask)
    incorrect_labels_z = tf.boolean_mask(labels_query_pl, incorrect_mask)
    signal_variance = tf.math.reduce_sum(tf.cast(incorrect_mask, tf.int32))
    loss_variance_incorrect = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=incorrect_logits_z,
            labels=tf.one_hot(incorrect_labels_z, flags.num_classes_train)))
    loss_variance_zero = 0.0
    loss_confidence = tf.cond(
        tf.greater(signal_variance, 0), lambda: loss_variance_incorrect,
        lambda: loss_variance_zero)

    regu_losses = tf.losses.get_regularization_losses()
    loss = tf.add_n([loss_classification] + regu_losses)

    # Learning rate
    if flags.lr_anneal == 'const':
      learning_rate = flags.init_learning_rate
    elif flags.lr_anneal == 'pwc':
      learning_rate = get_pwc_learning_rate(global_step, flags)
    elif flags.lr_anneal == 'exp':
      lr_decay_step = flags.number_of_steps // flags.n_lr_decay
      learning_rate = tf.train.exponential_decay(
          flags.init_learning_rate,
          global_step,
          lr_decay_step,
          1.0 / flags.lr_decay_rate,
          staircase=True)
    else:
      raise Exception('Not implemented')

    # Optimizer
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate, momentum=0.9)
    optimizer_confidence = tf.train.MomentumOptimizer(
        learning_rate=learning_rate, momentum=0.9)

    train_op = contrib_slim.learning.create_train_op(
        total_loss=loss,
        optimizer=optimizer,
        global_step=global_step,
        clip_gradient_norm=flags.clip_gradient_norm)
    variable_variance = []
    for v in tf.trainable_variables():
      if 'fc_variance' in v.name:
        variable_variance.append(v)
    train_op_confidence = contrib_slim.learning.create_train_op(
        total_loss=loss_confidence,
        optimizer=optimizer_confidence,
        global_step=global_step_confidence,
        clip_gradient_norm=flags.clip_gradient_norm,
        variables_to_train=variable_variance)

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('loss_classification', loss_classification)
    tf.summary.scalar('loss_variance', loss_confidence)
    tf.summary.scalar('regu_loss', tf.add_n(regu_losses))
    tf.summary.scalar('learning_rate', learning_rate)
    # Merges all summaries except for pretrain
    summary = tf.summary.merge(
        tf.get_collection('summaries', scope='(?!pretrain).*'))

    # Gets datasets
    few_shot_data_train, test_dataset, train_dataset = get_train_datasets(flags)
    # Defines session and logging
    summary_writer_train = tf.summary.FileWriter(log_dir, flush_secs=1)
    saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)
    print(saver.saver_def.filename_tensor_name)
    print(saver.saver_def.restore_op_name)
    # pylint: disable=unused-variable
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    supervisor = tf.train.Supervisor(
        logdir=log_dir,
        init_feed_dict=None,
        summary_op=None,
        init_op=tf.global_variables_initializer(),
        summary_writer=summary_writer_train,
        saver=saver,
        global_step=global_step,
        save_summaries_secs=flags.save_summaries_secs,
        save_model_secs=0)

    with supervisor.managed_session() as sess:
      checkpoint_step = sess.run(global_step)
      if checkpoint_step > 0:
        checkpoint_step += 1
      eval_interval_steps = flags.eval_interval_steps
      for step in range(checkpoint_step, flags.number_of_steps):
        # Computes the classification loss using a batch of data.
        images_query, labels_query,\
        images_support, labels_support = \
          few_shot_data_train.next_few_shot_batch(
              query_batch_size_per_task=flags.train_batch_size,
              num_classes_per_task=flags.num_classes_train,
              num_supports_per_class=flags.num_shots_train,
              num_tasks=flags.num_tasks_per_batch)

        feed_dict = {
            images_query_pl: images_query.astype(dtype=np.float32),
            labels_query_pl: labels_query,
            images_support_pl: images_support.astype(dtype=np.float32),
            labels_support_pl: labels_support
        }

        t_batch = time.time()
        dt_batch = time.time() - t_batch

        t_train = time.time()
        loss, loss_confidence = sess.run([train_op, train_op_confidence],
                                         feed_dict=feed_dict)
        dt_train = time.time() - t_train

        if step % 100 == 0:
          summary_str = sess.run(summary, feed_dict=feed_dict)
          summary_writer_train.add_summary(summary_str, step)
          summary_writer_train.flush()
          logging.info('step %d, loss : %.4g, dt: %.3gs, dt_batch: %.3gs', step,
                       loss, dt_train, dt_batch)

        if float(step) / flags.number_of_steps > 0.5:
          eval_interval_steps = flags.eval_interval_fine_steps

        if eval_interval_steps > 0 and step % eval_interval_steps == 0:
          saver.save(sess, os.path.join(log_dir, 'model'), global_step=step)
          eval(
              flags=flags,
              train_dataset=train_dataset,
              test_dataset=test_dataset)

        if float(
            step
        ) > 0.5 * flags.number_of_steps + flags.number_of_steps_to_early_stop:
          break


def get_class_center_for_evaluation(train_bs, test_bs, num_classes):
  """The tf graph of calculating class centers at eval given training data."""
  x_train = tf.placeholder(shape=[train_bs, feature_dim], dtype=tf.float32)
  x_test = tf.placeholder(shape=[test_bs, feature_dim], dtype=tf.float32)
  y_train = tf.placeholder(
      shape=[
          train_bs,
      ], dtype=tf.int32)
  y_test = tf.placeholder(
      shape=[
          test_bs,
      ], dtype=tf.int32)

  # Finds the class centers for the training data. class label should be 0-N
  ind_c = tf.squeeze(tf.where(tf.equal(y_train, 0)))
  train_input_c = tf.gather(x_train, ind_c)
  train_input_c = tf.expand_dims(train_input_c, 0)
  centroid = tf.reduce_sum(train_input_c, 1)

  for i in range(1, num_classes):
    ind_c = tf.squeeze(tf.where(tf.equal(y_train, i)))
    tmp_input_c = tf.gather(x_train, ind_c)
    tmp_input_c = tf.expand_dims(tmp_input_c, 0)
    tmp_centroid = tf.reduce_sum(tmp_input_c, 1)
    centroid = tf.concat([centroid, tmp_centroid], 0)

  return x_train, x_test, y_train, y_test, centroid


def make_predictions_for_evaluation(centroid, x_test, y_test, flags):
  """The tf graph for making predictions given class centers and test data."""
  # Calculates the centroid pair-wise distance
  centroid_expand = tf.expand_dims(centroid, axis=0)
  # Calculates the test sample - centroid distance
  i = x_test.get_shape().as_list()[0]
  # Tiles to be able to produce weight matrix alpha in (i,j) space
  x_data_test = tf.expand_dims(x_test, axis=1)
  x_data_test = tf.tile(x_data_test, (1, flags.num_classes_total, 1))
  centroid_expand_test = tf.tile(centroid_expand, (i, 1, 1))
  euclidean = tf.norm(
      x_data_test - centroid_expand_test, name='euclidean_distance', axis=-1)
  # Prediction based on nearest neighbors
  _, top_k_indices = tf.nn.top_k(-euclidean, k=1)
  pred = tf.squeeze(top_k_indices)
  correct_mask = tf.cast(tf.math.equal(pred, y_test), tf.float32)
  correct = tf.reduce_sum(correct_mask, axis=0)

  return pred, correct


def calculate_nll(y_test, softmaxed_logits, num_classes):
  y_test = tf.one_hot(y_test, depth=num_classes)
  nll = tf.reduce_sum(-tf.log(tf.reduce_sum(y_test * softmaxed_logits, axis=1)))

  return nll


def confidence_estimation_and_evaluation(centroid, x_test, x_test_variance,
                                         y_test, flags):
  """The tf graph for confidence estimation and NLL for a batch of test data."""
  centroid_expand = tf.expand_dims(centroid, axis=0)
  i = x_test.get_shape().as_list()[0]
  # Tiles to be able to produce weight matrix alpha in (i,j) space
  x_test_tile = tf.expand_dims(x_test, axis=1)
  x_test_tile = tf.tile(x_test_tile, (1, flags.num_classes_total, 1))
  centroid_expand_tile = tf.tile(centroid_expand, (i, 1, 1))
  distance = tf.norm(
      x_test_tile - centroid_expand_tile, name='euclidean_distance', axis=-1)

  softmaxed_distance = tf.math.softmax(-distance)
  _, top_k_indices = tf.nn.top_k(softmaxed_distance, k=1)
  pred = tf.squeeze(top_k_indices)

  # Calculates the distance with the correct label
  row = tf.constant(np.arange(pred.get_shape().as_list()[0]))
  row = tf.cast(row, tf.int32)

  eps = tf.random.normal(shape=x_test.shape)
  z = eps * tf.exp((x_test_variance) * .5) + x_test
  z_tile = tf.expand_dims(z, axis=1)
  z_tile = tf.tile(z_tile, (1, flags.num_classes_total, 1))
  distance = tf.norm(
      z_tile - centroid_expand_tile, name='euclidean_distance', axis=-1)
  softmaxed_distance = tf.math.softmax(-distance)

  for _ in range(1, flags.num_forward):
    eps = tf.random.normal(shape=x_test.shape)
    z = eps * tf.exp((x_test_variance) * .5) + x_test
    z_tile = tf.expand_dims(z, axis=1)
    z_tile = tf.tile(z_tile, (1, flags.num_classes_total, 1))
    distance = tf.norm(
        z_tile - centroid_expand_tile, name='euclidean_distance', axis=-1)
    softmaxed_distance += tf.math.softmax(-distance)
  softmaxed_distance = softmaxed_distance / flags.num_forward
  nll_sum = calculate_nll(y_test, softmaxed_distance, flags.num_classes_total)
  ind_tensor = tf.transpose(tf.stack([row, pred]))
  confidence = tf.gather_nd(softmaxed_distance, ind_tensor)

  return nll_sum, confidence


def get_train_datasets(flags):
  if flags.dataset == 'cifar10':
    train_set, test_set = data_loader.load_cifar10()
  elif flags.dataset == 'cifar100':
    train_set, test_set = data_loader.load_cifar100()
  elif flags.dataset == 'tinyimagenet':
    train_set, test_set = data_loader.load_tiny_imagenet(
        flags.data_dir, flags.val_data_dir)
  episodic_train = data_loader.Dataset(train_set)
  return episodic_train, test_set, train_set


def get_pwc_learning_rate(global_step, flags):
  learning_rate = tf.train.piecewise_constant(global_step, [
      np.int64(flags.number_of_steps / 2),
      np.int64(flags.number_of_steps / 2 + flags.num_steps_decay_pwc),
      np.int64(flags.number_of_steps / 2 + 2 * flags.num_steps_decay_pwc)
  ], [
      flags.init_learning_rate, flags.init_learning_rate * 0.1,
      flags.init_learning_rate * 0.01, flags.init_learning_rate * 0.001
  ])
  return learning_rate


class ModelLoader:
  """The class definition for the evaluation module."""

  def __init__(self, model_path, batch_size, train_dataset, test_dataset):
    self.train_batch_size = batch_size
    self.test_batch_size = batch_size
    self.test_dataset = test_dataset
    self.train_dataset = train_dataset

    latest_checkpoint = tf.train.latest_checkpoint(
        checkpoint_dir=os.path.join(model_path, 'train'))
    print(latest_checkpoint)
    step = int(os.path.basename(latest_checkpoint).split('-')[1])
    flags = Namespace(
        utils.load_and_save_params(default_params=dict(), exp_dir=model_path))
    image_size = data_loader.get_image_size(flags.dataset)
    self.flags = flags

    with tf.Graph().as_default():
      self.tensor_images, self.tensor_labels = placeholder_inputs(
          batch_size=self.train_batch_size,
          image_size=image_size,
          scope='inputs')
      if flags.dataset == 'cifar10' or flags.dataset == 'cifar100':
        tensor_images_aug = data_loader.augment_cifar(
            self.tensor_images, is_training=False)
      else:
        tensor_images_aug = data_loader.augment_tinyimagenet(
            self.tensor_images, is_training=False)
      model = build_model(flags)
      with tf.variable_scope('Proto_training'):
        self.representation, self.variance = build_feature_extractor_graph(
            inputs=tensor_images_aug,
            flags=flags,
            is_variance=True,
            is_training=False,
            model=model)
      self.tensor_train_rep, self.tensor_test_rep, \
      self.tensor_train_rep_label, self.tensor_test_rep_label,\
      self.center = get_class_center_for_evaluation(
          self.train_batch_size, self.test_batch_size, flags.num_classes_total)

      self.prediction, self.acc \
        = make_predictions_for_evaluation(self.center,
                                          self.tensor_test_rep,
                                          self.tensor_test_rep_label,
                                          self.flags)
      self.tensor_test_variance = tf.placeholder(
          shape=[self.test_batch_size, feature_dim], dtype=tf.float32)
      self.nll, self.confidence = confidence_estimation_and_evaluation(
          self.center, self.tensor_test_rep, self.tensor_test_variance,
          self.tensor_test_rep_label, flags)

      config = tf.ConfigProto(allow_soft_placement=True)
      config.gpu_options.allow_growth = True
      self.sess = tf.Session(config=config)
      # Runs init before loading the weights
      self.sess.run(tf.global_variables_initializer())
      # Loads weights
      saver = tf.train.Saver()
      saver.restore(self.sess, latest_checkpoint)
      self.flags = flags
      self.step = step
      log_dir = flags.log_dir
      graphpb_txt = str(tf.get_default_graph().as_graph_def())
      with open(os.path.join(log_dir, 'eval', 'graph.pbtxt'), 'w') as f:
        f.write(graphpb_txt)

  def eval_ece(self, pred_logits_np, pred_np, label_np, num_bins):
    """Calculates ECE.

    Args:
      pred_logits_np: the softmax output at the dimension of the predicted
        labels of test samples.
      pred_np:  the numpy array of the predicted labels of test samples.
      label_np:  the numpy array of the ground-truth labels of test samples.
      num_bins: the number of bins to partition all samples. we set it as 15.

    Returns:
      ece: the calculated ECE value.
    """
    acc_tab = np.zeros(num_bins)  # Empirical (true) confidence
    mean_conf = np.zeros(num_bins)  # Predicted confidence
    nb_items_bin = np.zeros(num_bins)  # Number of items in the bins
    tau_tab = np.linspace(
        min(pred_logits_np), max(pred_logits_np),
        num_bins + 1)  # Confidence bins
    tau_tab = np.linspace(0, 1, num_bins + 1)  # Confidence bins
    for i in np.arange(num_bins):  # Iterates over the bins
      # Selects the items where the predicted max probability falls in the bin
      # [tau_tab[i], tau_tab[i + 1)]
      sec = (tau_tab[i + 1] > pred_logits_np) & (pred_logits_np >= tau_tab[i])
      nb_items_bin[i] = np.sum(sec)  # Number of items in the bin
      # Selects the predicted classes, and the true classes
      class_pred_sec, y_sec = pred_np[sec], label_np[sec]
      # Averages of the predicted max probabilities
      mean_conf[i] = np.mean(
          pred_logits_np[sec]) if nb_items_bin[i] > 0 else np.nan
      # Computes the empirical confidence
      acc_tab[i] = np.mean(
          class_pred_sec == y_sec) if nb_items_bin[i] > 0 else np.nan
    # Cleaning
    mean_conf = mean_conf[nb_items_bin > 0]
    acc_tab = acc_tab[nb_items_bin > 0]
    nb_items_bin = nb_items_bin[nb_items_bin > 0]
    if sum(nb_items_bin) != 0:
      ece = np.average(
          np.absolute(mean_conf - acc_tab),
          weights=nb_items_bin.astype(np.float) / np.sum(nb_items_bin))
    else:
      ece = 0.0
    return ece

  def eval_acc_nll_ece(self, num_cases_train, num_cases_test):
    """Returns evaluation metrics.

    Args:
      num_cases_train: the total number of training samples.
      num_cases_test:  the total number of test samples.

    Returns:
      num_correct / num_cases_test: the accuracy of the evaluation.
      nll: the calculated NLL value.
      ece: the calculated ECE value.
    """
    num_batches_train = num_cases_train // self.train_batch_size
    num_batches_test = num_cases_test // self.test_batch_size
    num_correct = 0.0
    features_train_np = []
    features_test_np = []
    variance_test_np = []
    for i in trange(num_batches_train):
      images_train = self.train_dataset[0][(
          i * self.train_batch_size):((i + 1) * self.train_batch_size)]
      feed_dict = {self.tensor_images: images_train.astype(dtype=np.float32)}
      [features_train_batch] = self.sess.run([self.representation], feed_dict)
      features_train_np.extend(features_train_batch)
    features_train_np = np.concatenate(features_train_np, axis=0)
    for i in trange(num_batches_test):
      images_test = self.test_dataset[0][(
          i * self.test_batch_size):((i + 1) * self.test_batch_size)]
      feed_dict = {self.tensor_images: images_test.astype(dtype=np.float32)}
      [features_test_batch, variances_test_batch
      ] = self.sess.run([self.representation, self.variance], feed_dict)
      features_test_np.extend(features_test_batch)
      variance_test_np.extend(variances_test_batch)
    features_test_np = np.concatenate(features_test_np, axis=0)
    variance_test_np = np.concatenate(variance_test_np, axis=0)

    # Computes class centers.
    features_train_batch = features_train_np[:self.train_batch_size]
    feed_dict = {
        self.tensor_train_rep:
            features_train_batch,
        self.tensor_train_rep_label:
            self.train_dataset[1][:self.train_batch_size]
    }
    [centroid] = self.sess.run([self.center], feed_dict)
    for i in trange(1, num_batches_train):
      features_train_batch = features_train_np[(
          i * self.train_batch_size):((i + 1) * self.train_batch_size)]
      feed_dict = {
          self.tensor_train_rep:
              features_train_batch,
          self.tensor_train_rep_label:
              self.train_dataset[1]
              [(i * self.train_batch_size):((i + 1) * self.train_batch_size)]
      }
      [centroid_batch] = self.sess.run([self.center], feed_dict)
      centroid = centroid + centroid_batch

    centroid = centroid / self.flags.num_samples_per_class
    pred_list = []
    confidence_list = []
    nll_sum = 0
    for i in trange(num_batches_test):
      features_test_batch = features_test_np[(
          i * self.test_batch_size):((i + 1) * self.test_batch_size)]
      variance_test_batch = variance_test_np[(
          i * self.test_batch_size):((i + 1) * self.test_batch_size)]
      feed_dict = {
          self.center:
              centroid,
          self.tensor_test_rep:
              features_test_batch,
          self.tensor_test_variance:
              variance_test_batch,
          self.tensor_test_rep_label:
              self.test_dataset[1]
              [(i * self.test_batch_size):((i + 1) * self.test_batch_size)]
      }
      [prediction,
       num_correct_per_batch] = self.sess.run([self.prediction, self.acc],
                                              feed_dict)
      num_correct += num_correct_per_batch
      pred_list.append(prediction)

      [nll, confidence] = self.sess.run([self.nll, self.confidence], feed_dict)
      confidence_list.append(confidence)
      nll_sum += nll
    pred_np = np.concatenate(pred_list, axis=0)
    confidence_np = np.concatenate(confidence_list, axis=0)

    # The definition of NLL can be found at "On calibration of modern neural
    # networks." Guo, Chuan, et al. Proceedings of the 34th International
    # Conference on Machine Learning-Volume 70. JMLR. org, 2017. NLL averages
    # the negative log-likelihood of all test samples.

    nll = nll_sum / num_cases_test

    # The definition of ECE can be found at "On calibration of modern neural
    # networks." Guo, Chuan, et al. Proceedings of the 34th International
    # Conference on Machine Learning-Volume 70. JMLR. org, 2017. ECE
    # approximates the expectation of the difference between accuracy and
    # confidence. It partitions the confidence estimations (the likelihood of
    # the predicted label) of all test samples into L equally-spaced bins and
    # calculates the average confidence and accuracy of test samples lying in
    # each bin.

    ece = self.eval_ece(confidence_np, pred_np, self.test_dataset[1], 15)
    print('acc: ' + str(num_correct / num_cases_test))
    print('nll: ')
    print(nll)
    print('ece: ')
    print(ece)

    return num_correct / num_cases_test, nll, ece


def eval(flags, train_dataset, test_dataset):
  """Evaluation entry point."""
  # pylint: disable=redefined-builtin
  log_dir = flags.log_dir
  eval_writer = utils.summary_writer(log_dir + '/eval')
  results = {}
  model = ModelLoader(
      model_path=flags.pretrained_model_dir,
      batch_size=10000,
      train_dataset=train_dataset,
      test_dataset=test_dataset)
  acc_tst, nll, ece \
      = model.eval_acc_nll_ece(flags.num_cases_train, flags.num_cases_test)

  results['accuracy_target_tst'] = acc_tst
  results['nll'] = nll
  results['ece'] = ece
  eval_writer(model.step, **results)
  logging.info('accuracy_%s: %.3g.', 'target_tst', acc_tst)


def main(argv=None):
  # pylint: disable=unused-argument
  # pylint: disable=unused-variable
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.per_process_gpu_memory_fraction = 1.0
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)

  # Gets parameters.
  default_params = get_arguments()

  # Creates the experiment directory.
  log_dir = default_params.log_dir
  ad = pathlib.Path(log_dir)
  if not ad.exists():
    ad.mkdir(parents=True)

  # Main function for training and evaluation.
  flags = Namespace(utils.load_and_save_params(vars(default_params),
                                               log_dir, ignore_existing=True))
  train(flags=flags)


if __name__ == '__main__':
  tf.app.run()
