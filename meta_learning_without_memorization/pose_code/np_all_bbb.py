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

"""Add KL, beta-BBB, just on encoder_w and include a version of vanilla NP."""
from __future__ import print_function
import functools
import os
import pickle
import time

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import MaxPooling2D
import tensorflow_probability as tfp
from tensorflow_probability.python.layers import util as tfp_layers_util


tf.compat.v1.enable_v2_tensorshape()
FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_string('logdir', '/tmp/data',
                    'directory for summaries and checkpoints.')
flags.DEFINE_string('data_dir', None,
                    'Directory of data files.')
get_data_dir = lambda: FLAGS.data_dir
flags.DEFINE_list('data', ['train_data_ins.pkl', 'val_data_ins.pkl'],
                  'data name')
flags.DEFINE_integer('update_batch_size', 15, 'number of context/target')
flags.DEFINE_integer('meta_batch_size', 10, 'number of tasks')
flags.DEFINE_integer('dim_im', 128, 'image size')
flags.DEFINE_integer('dim_y', 1, 'dimension of y')

## Training options
flags.DEFINE_list('n_hidden_units_g', [100, 100],
                  'number of tasks sampled per meta-update')
flags.DEFINE_list('n_hidden_units_r', [100, 100],
                  'number of inner gradient updates during test.')
flags.DEFINE_integer('dim_z', 200, 'dimension of z')
flags.DEFINE_integer('dim_r', 200, 'dimension of r for aggregating')
flags.DEFINE_float('update_lr', 5e-4, 'lr')
flags.DEFINE_integer('num_updates', 100000, 'num_updates')
flags.DEFINE_integer('trial', 1, 'trial number')
flags.DEFINE_integer(
    'num_classes', 1,
    'number of classes used in classification (e.g. 5-way classification).')
flags.DEFINE_bool('deterministic', True, 'deterministic encoder')

flags.DEFINE_float('beta', 0.001, 'beta for IB')
flags.DEFINE_float('var', -3.0, 'var initial')
flags.DEFINE_integer('dim_w', 20, 'dimension of w')
flags.DEFINE_float('facto', 1.0, 'zero out z to memorize or not')


def get_batch(x, y):
  """Get data batch."""
  xs, ys, xq, yq = [], [], [], []
  for _ in range(FLAGS.meta_batch_size):
    # sample WAY classes
    classes = np.random.choice(
        range(np.shape(x)[0]), size=FLAGS.num_classes, replace=False)

    support_set = []
    query_set = []
    support_sety = []
    query_sety = []
    for k in list(classes):
      # sample SHOT and QUERY instances
      idx = np.random.choice(
          range(np.shape(x)[1]),
          size=FLAGS.update_batch_size + FLAGS.update_batch_size,
          replace=False)
      x_k = x[k][idx]
      y_k = y[k][idx]

      support_set.append(x_k[:FLAGS.update_batch_size])
      query_set.append(x_k[FLAGS.update_batch_size:])
      support_sety.append(y_k[:FLAGS.update_batch_size])
      query_sety.append(y_k[FLAGS.update_batch_size:])

    xs_k = np.concatenate(support_set, 0)
    xq_k = np.concatenate(query_set, 0)
    ys_k = np.concatenate(support_sety, 0)
    yq_k = np.concatenate(query_sety, 0)

    xs.append(xs_k)
    xq.append(xq_k)
    ys.append(ys_k)
    yq.append(yq_k)

  xs, ys = np.stack(xs, 0), np.stack(ys, 0)
  xq, yq = np.stack(xq, 0), np.stack(yq, 0)

  xs = np.reshape(
      xs,
      [FLAGS.meta_batch_size, FLAGS.update_batch_size * FLAGS.num_classes, -1])
  xq = np.reshape(
      xq,
      [FLAGS.meta_batch_size, FLAGS.update_batch_size * FLAGS.num_classes, -1])
  xs = xs.astype(np.float32) / 255.0
  xq = xq.astype(np.float32) / 255.0
  ys = ys.astype(np.float32) * 10.0
  yq = yq.astype(np.float32) * 10.0
  return xs, ys, xq, yq


def gen(x, y):
  while True:
    yield get_batch(np.array(x), np.array(y))


def sampling(output):
  mu, logstd = tf.split(output, num_or_size_splits=2, axis=-1)
  sigma = tf.nn.softplus(logstd)
  ws = mu + tf.random_normal(tf.shape(mu)) * sigma
  return ws, mu, sigma


def mse(pred, label):
  pred = tf.reshape(pred, [-1])
  label = tf.reshape(label, [-1])
  return tf.reduce_mean(tf.square(pred - label))


def encoder_w(xs, encoder_w0):
  """xs is [n_task, n_im, dim_x]; return [n_task, n_im, dim_w]."""
  n_task = tf.shape(xs)[0]
  n_im = tf.shape(xs)[1]
  xs = tf.reshape(xs, [-1, 128, 128, 1])

  ws = encoder_w0(xs)
  ws = tf.reshape(ws, [n_task, n_im, FLAGS.dim_w])
  return ws


def xy_to_z(xs, ys, encoder_w0, encoder_r):
  r"""ws = T0(xs), rs = T1(ws, ys), r = mean(rs), z \sim N(mu(r), sigma(r))."""
  with tf.variable_scope(''):
    ws = encoder_w(xs, encoder_w0)  # (n_task * n_im_per_task) * dim_w

  util1 = tfp.layers.DenseReparameterization(FLAGS.dim_w // 4)
  transformed_ys = util1(ys)
  wys = tf.concat([ws, transformed_ys],
                  axis=-1)  # n_task *  n_im_per_task * (dim_w+dim_transy)

  rs = encoder_r(wys)  # n_task *  n_im_per_task * dim_r

  r = tf.reduce_mean(rs, axis=1, keepdims=True)  # n_task * 1 * dim_r

  # z_sample = tf.layers.dense(r, FLAGS.dim_z, name='r2z',
  #                     reuse=tf.AUTO_REUSE, kernel_initializer='normal')
  util2 = tfp.layers.DenseReparameterization(FLAGS.dim_z)
  z_sample = util2(r)
  return tf.tile(z_sample,
                 [1, FLAGS.update_batch_size, 1]), util1.losses, util2.losses


def construct_model(input_tensors,
                    encoder_w0,
                    decoder0,
                    encoder_r,
                    prefix=None):
  """Build model."""
  facto = tf.placeholder_with_default(1.0, ())
  context_xs = input_tensors['inputa']
  context_ys = input_tensors['labela']
  target_xs = input_tensors['inputb']
  target_ys = input_tensors['labelb']

  # sample ws ~ w|(x_all,a), rs = T(ws, ys), r = mean(rs), z = T(r)
  # x_all = tf.concat([context_xs, target_xs], axis=1) #n_task * 20 * (128*128)
  # y_all = tf.concat([context_ys, target_ys], axis=1)

  x_all = context_xs
  y_all = context_ys

  # n_task * [n_im] * d_z
  if 'train' in prefix:
    z_samples, util_loss1, util_loss2 = xy_to_z(x_all, y_all, encoder_w0,
                                                encoder_r)
  else:
    z_samples, util_loss1, util_loss2 = xy_to_z(context_xs, context_ys,
                                                encoder_w0, encoder_r)

  target_ws = encoder_w(target_xs, encoder_w0)
  input_zxs = tf.concat([z_samples, target_ws], axis=-1)

  # sample y_hat ~  y|(w,z)
  with tf.variable_scope('decoder'):
    target_yhat_mu = decoder0(input_zxs)  # n_task * n_im * dim_y

  # when var of  p(y | x,z) is fixed, neg-loglik <=> MSE
  mse_loss = mse(target_yhat_mu, target_ys)

  tf.summary.scalar(prefix + 'mse', mse_loss)
  optimizer1 = tf.train.AdamOptimizer(FLAGS.update_lr)

  if 'train' in prefix:
    all_var = tf.trainable_variables()

    kl_loss = sum(encoder_w0.losses) + sum(decoder0.losses) + sum(
        encoder_r.losses) + sum(util_loss1) + sum(util_loss2)
    loss = mse_loss + FLAGS.beta * kl_loss

    gvs_theta = optimizer1.compute_gradients(loss, all_var)
    train_op = optimizer1.apply_gradients(gvs_theta)

    tf.summary.scalar(prefix + 'full_loss', loss)
    tf.summary.scalar(prefix + 'regularizer', FLAGS.beta * kl_loss)
    return mse_loss, train_op, facto
  else:
    return mse_loss


def main(_):
  kernel_posterior_fn = tfp_layers_util.default_mean_field_normal_fn(
      untransformed_scale_initializer=tf.compat.v1.initializers.random_normal(
          mean=FLAGS.var, stddev=0.1))
  encoder_w0 = tf.keras.Sequential([
      tfp.layers.Convolution2DReparameterization(
          filters=32,
          kernel_size=3,
          strides=(2, 2),
          activation='relu',
          padding='SAME',
          kernel_posterior_fn=kernel_posterior_fn),
      tfp.layers.Convolution2DReparameterization(
          filters=48,
          kernel_size=3,
          strides=(2, 2),
          activation='relu',
          padding='SAME',
          kernel_posterior_fn=kernel_posterior_fn),
      MaxPooling2D(pool_size=(2, 2)),
      tfp.layers.Convolution2DReparameterization(
          filters=64,
          kernel_size=3,
          strides=(2, 2),
          activation='relu',
          padding='SAME',
          kernel_posterior_fn=kernel_posterior_fn),
      tf.keras.layers.Flatten(),
      tfp.layers.DenseReparameterization(
          FLAGS.dim_w, kernel_posterior_fn=kernel_posterior_fn),
  ])

  decoder0 = tf.keras.Sequential([
      tfp.layers.DenseReparameterization(100, activation=tf.nn.relu),
      tfp.layers.DenseReparameterization(100, activation=tf.nn.relu),
      tfp.layers.DenseReparameterization(FLAGS.dim_y),
  ])

  encoder_r = tf.keras.Sequential([
      tfp.layers.DenseReparameterization(100, activation=tf.nn.relu),
      tfp.layers.DenseReparameterization(100, activation=tf.nn.relu),
      tfp.layers.DenseReparameterization(FLAGS.dim_r),
  ])

  dim_output = FLAGS.dim_y
  dim_input = FLAGS.dim_im * FLAGS.dim_im * 1

  exp_name = '%s.beta-%g.update_lr-%g.trial-%d' % ('np_bbb_all', FLAGS.beta,
                                                   FLAGS.update_lr, FLAGS.trial)
  checkpoint_dir = os.path.join(FLAGS.logdir, exp_name)

  x_train, y_train = pickle.load(
      tf.io.gfile.GFile(os.path.join(get_data_dir(), FLAGS.data[0]), 'rb'))
  x_val, y_val = pickle.load(
      tf.io.gfile.GFile(os.path.join(get_data_dir(), FLAGS.data[1]), 'rb'))

  x_train, y_train = np.array(x_train), np.array(y_train)
  y_train = y_train[:, :, -1, None]
  x_val, y_val = np.array(x_val), np.array(y_val)
  y_val = y_val[:, :, -1, None]

  ds_train = tf.data.Dataset.from_generator(
      functools.partial(gen, x_train, y_train),
      (tf.float32, tf.float32, tf.float32, tf.float32),
      (tf.TensorShape(
          [None, FLAGS.update_batch_size * FLAGS.num_classes, dim_input]),
       tf.TensorShape(
           [None, FLAGS.update_batch_size * FLAGS.num_classes, dim_output]),
       tf.TensorShape(
           [None, FLAGS.update_batch_size * FLAGS.num_classes, dim_input]),
       tf.TensorShape(
           [None, FLAGS.update_batch_size * FLAGS.num_classes, dim_output])))

  ds_val = tf.data.Dataset.from_generator(
      functools.partial(gen, x_val, y_val),
      (tf.float32, tf.float32, tf.float32, tf.float32),
      (tf.TensorShape(
          [None, FLAGS.update_batch_size * FLAGS.num_classes, dim_input]),
       tf.TensorShape(
           [None, FLAGS.update_batch_size * FLAGS.num_classes, dim_output]),
       tf.TensorShape(
           [None, FLAGS.update_batch_size * FLAGS.num_classes, dim_input]),
       tf.TensorShape(
           [None, FLAGS.update_batch_size * FLAGS.num_classes, dim_output])))

  inputa, labela, inputb, labelb = ds_train.make_one_shot_iterator().get_next()

  input_tensors = {'inputa': inputa,\
                   'inputb': inputb,\
                   'labela': labela, 'labelb': labelb}

  inputa_val, labela_val, inputb_val, labelb_val = ds_val.make_one_shot_iterator(
  ).get_next()

  metaval_input_tensors = {'inputa': inputa_val,\
                           'inputb': inputb_val,\
                           'labela': labela_val, 'labelb': labelb_val}

  loss, train_op, facto = construct_model(
      input_tensors, encoder_w0, decoder0, encoder_r, prefix='metatrain_')
  loss_val = construct_model(
      metaval_input_tensors, encoder_w0, decoder0, encoder_r, prefix='metaval_')

  ###########

  summ_op = tf.summary.merge_all()
  sess = tf.InteractiveSession()
  summary_writer = tf.summary.FileWriter(checkpoint_dir, sess.graph)
  tf.global_variables_initializer().run()

  PRINT_INTERVAL = 50  # pylint: disable=invalid-name
  SUMMARY_INTERVAL = 5  # pylint: disable=invalid-name
  prelosses, prelosses_val = [], []
  old_time = time.time()
  for itr in range(FLAGS.num_updates):

    feed_dict = {facto: FLAGS.facto}

    if itr % SUMMARY_INTERVAL == 0:
      summary, cost, cost_val = sess.run([summ_op, loss, loss_val], feed_dict)
      summary_writer.add_summary(summary, itr)
      prelosses.append(cost)  # 0 step loss on training set
      prelosses_val.append(cost_val)  # 0 step loss on meta_val training set

    sess.run(train_op, feed_dict)

    if (itr != 0) and itr % PRINT_INTERVAL == 0:
      print('Iteration ' + str(itr) + ': ' + str(np.mean(prelosses)), 'time =',
            time.time() - old_time)
      prelosses = []
      old_time = time.time()
      print('Validation results: ' + str(np.mean(prelosses_val)))
      prelosses_val = []


if __name__ == '__main__':
  app.run(main)
