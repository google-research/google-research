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

"""NP w/ IB."""
from __future__ import print_function
import functools
import os
import pickle
import time

from absl import app
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras.layers import Conv2D
from tensorflow.compat.v1.keras.layers import MaxPooling2D


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
flags.DEFINE_integer('dim_z', 64, 'dimension of z')
flags.DEFINE_integer('dim_r', 100, 'dimension of r for aggregating')
flags.DEFINE_float('update_lr', 0.0005, 'lr')
flags.DEFINE_integer('num_updates', 120000, 'num_updates')
flags.DEFINE_integer('trial', 1, 'trial number')
flags.DEFINE_integer(
    'num_classes', 1,
    'number of classes used in classification (e.g. 5-way classification).')
flags.DEFINE_bool('deterministic', True, 'deterministic encoder')

## IB options
flags.DEFINE_float('beta', 0.01, 'beta for IB')
flags.DEFINE_integer('dim_w', 64, 'dimension of w')


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


def mse(pred, label):
  pred = tf.reshape(pred, [-1])
  label = tf.reshape(label, [-1])
  return tf.reduce_mean(tf.square(pred - label))


def encoder_r(xys):
  """Define encoder."""
  with tf.variable_scope('encoder_r', reuse=tf.AUTO_REUSE):
    hidden_layer = xys
    # First layers are relu
    for i, n_hidden_units in enumerate(FLAGS.n_hidden_units_r):
      hidden_layer = tf.layers.dense(
          hidden_layer,
          n_hidden_units,
          activation=tf.nn.relu,
          name='encoder_r_{}'.format(i),
          reuse=tf.AUTO_REUSE,
          kernel_initializer='normal')

    # Last layer is simple linear
    i = len(FLAGS.n_hidden_units_r)
    r = tf.layers.dense(
        hidden_layer,
        FLAGS.dim_r,
        name='encoder_r_{}'.format(i),
        reuse=tf.AUTO_REUSE,
        kernel_initializer='normal')
  return r


def kl_qp_gaussian(mu_q, sigma_q, mu_p, sigma_p):
  r"""KL(N(mu_q), Diag(sigma_q^2) || N(mu_p), Diag(sigma_p^2))."""
  sigma2_q = tf.square(sigma_q) + 1e-16
  sigma2_p = tf.square(sigma_p) + 1e-16
  temp = (tf.log(sigma2_p) - tf.log(sigma2_q) - 1.0
          + sigma2_q / sigma2_p
          + tf.square(mu_q - mu_p) / sigma2_p)  # n_target * d_w
  kl = 0.5 * tf.reduce_sum(temp, axis=-1)  # use reduce_sum for dim_w
  return tf.reduce_mean(kl)


def sampling(output):
  mu, logstd = tf.split(output, num_or_size_splits=2, axis=-1)
  sigma = tf.nn.softplus(logstd)
  ws = mu + tf.random_normal(tf.shape(mu)) * sigma
  return ws, mu, sigma


def encoder_w(xs, encoder_w0):
  """xs is [n_task, n_im, dim_x]; return [n_task, n_im, dim_w]."""
  n_task = tf.shape(xs)[0]
  n_im = tf.shape(xs)[1]
  xs = tf.reshape(xs, [-1, 128, 128, 1])

  output = encoder_w0(xs)

  output = tf.reshape(output, [n_task, n_im, FLAGS.dim_w + FLAGS.dim_w])
  ws, mu_w, sigma_w = sampling(output)
  return ws, mu_w, sigma_w


def xy_to_z(xs, ys, encoder_w0):
  r"""ws = T0(xs), rs = T1(ws, ys), r = mean(rs), z \sim N(mu(r), sigma(r))."""
  with tf.variable_scope('encoder_w', reuse=tf.AUTO_REUSE):
    # (n_task * n_im_per_task) * dim_w
    ws, mu_w, sigma_w = encoder_w(xs, encoder_w0)

  transformed_ys = tf.layers.dense(
      ys,
      FLAGS.dim_w // 2,
      name='lift_y',
      reuse=tf.AUTO_REUSE,
      kernel_initializer='normal')
  xys = tf.concat([ws, transformed_ys],
                  axis=-1)  # n_task *  n_im_per_task * (dim_w+dim_transy)

  rs = encoder_r(xys)  # n_task *  n_im_per_task * dim_r

  r = tf.reduce_mean(rs, axis=1, keepdims=True)  # n_task * 1 * dim_r

  if FLAGS.deterministic:
    z_sample = tf.layers.dense(
        r,
        FLAGS.dim_z,
        name='r2z',
        reuse=tf.AUTO_REUSE,
        kernel_initializer='normal')
  else:
    z = tf.layers.dense(
        r,
        FLAGS.dim_z + FLAGS.dim_z,
        name='r2z',
        reuse=tf.AUTO_REUSE,
        kernel_initializer='normal')
    z_sample, _, _ = sampling(z)

  # tile n_targets
  z_sample = tf.tile(z_sample, [1, FLAGS.update_batch_size, 1])
  return z_sample, mu_w, sigma_w


def decoder_g(zxs):
  """Define decoder."""
  with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
    hidden_layer = zxs
    for i, n_hidden_units in enumerate(FLAGS.n_hidden_units_g):
      hidden_layer = tf.layers.dense(
          hidden_layer,
          n_hidden_units,
          activation=tf.nn.relu,
          name='decoder_{}'.format(i),
          reuse=tf.AUTO_REUSE,
          kernel_initializer='normal')
    i = len(FLAGS.n_hidden_units_g)
    y_hat = tf.layers.dense(
        hidden_layer,
        FLAGS.dim_y,
        name='decoder_{}'.format(i),
        reuse=tf.AUTO_REUSE,
        kernel_initializer='normal')
  return y_hat


def construct_model(input_tensors, encoder_w0, prefix=None):
  """Construct model."""
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
    z_samples, mu_w_all, sigma_w_all = xy_to_z(x_all, y_all, encoder_w0)
    z_samples = z_samples * facto
  else:
    z_samples, _, _ = xy_to_z(context_xs, context_ys, encoder_w0)
    z_samples = z_samples * facto

  target_ws, _, _ = encoder_w(target_xs, encoder_w0)
  input_zxs = tf.concat([z_samples, target_ws], axis=-1)

  # sample y_hat ~  y|(w,z)
  target_yhat_mu = decoder_g(input_zxs)  # n_task * n_im * dim_y

  # when var of  p(y | x,z) is fixed, neg-loglik <=> MSE
  mse_loss = mse(target_yhat_mu, target_ys)

  tf.summary.scalar(prefix + 'mse', mse_loss)
  optimizer1 = tf.train.AdamOptimizer(FLAGS.update_lr)
  optimizer2 = tf.train.AdamOptimizer(0.001)

  if 'train' in prefix:
    # mu_w_all is n_task * n_im * dim_w
    # target_yhat_mu is n_task * n_im * dim_w
    kl_ib = kl_qp_gaussian(mu_w_all, sigma_w_all, tf.zeros(tf.shape(mu_w_all)),
                           tf.ones(tf.shape(mu_w_all)))

    THETA = (  # pylint: disable=invalid-name
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder')
        + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder_w'))
    all_var = tf.trainable_variables()
    PHI = [v for v in all_var if v not in THETA]  # pylint: disable=invalid-name

    loss = mse_loss + FLAGS.beta * kl_ib

    gvs_theta = optimizer1.compute_gradients(loss, THETA)
    train_theta_op = optimizer1.apply_gradients(gvs_theta)

    gvs_phi = optimizer2.compute_gradients(loss, PHI)
    train_phi_op = optimizer2.apply_gradients(gvs_phi)

    with tf.control_dependencies([train_theta_op, train_phi_op]):
      train_op = tf.no_op()

    tf.summary.scalar(prefix + 'kl', kl_ib)
    tf.summary.scalar(prefix + 'full_loss', loss)
    return mse_loss, kl_ib, train_op, facto
  else:
    return mse_loss


def main(_):
  encoder_w0 = tf.keras.Sequential([
      Conv2D(
          filters=32,
          kernel_size=3,
          strides=(2, 2),
          activation='relu',
          padding='same'),
      Conv2D(
          filters=48,
          kernel_size=3,
          strides=(2, 2),
          activation='relu',
          padding='same'),
      MaxPooling2D(pool_size=(2, 2)),
      Conv2D(
          filters=64,
          kernel_size=3,
          strides=(2, 2),
          activation='relu',
          padding='same'),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(FLAGS.dim_w + FLAGS.dim_w),
  ])

  dim_output = FLAGS.dim_y
  dim_input = FLAGS.dim_im * FLAGS.dim_im * 1

  exp_name = '%s.beta-%g.update_lr-%g.trial-%d' % ('np_ib', FLAGS.beta,
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

  loss, _, train_op, _ = construct_model(
      input_tensors, encoder_w0, prefix='metatrain_')
  loss_val = construct_model(
      metaval_input_tensors, encoder_w0, prefix='metaval_')

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
    # print('###############################')
    # print(itr)
    # print('###############################')
    feed_dict = {}
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
