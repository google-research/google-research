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

# pylint: skip-file
"""BBB for MAML, on encoder_w.

Based on code by Chelsea Finn (https://github.com/cbfinn/maml).
and Mingzhang Yin (https://github.com/google-research/google-research/tree/master/meta_learning_without_memorization)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import pickle
import time

from absl import app
from absl import flags
import numpy as np
from six.moves import range
import tensorflow as tf
from tensorflow.keras.layers import MaxPooling2D
import tensorflow_probability as tfp
from tensorflow_probability.python.layers import util as tfp_layers_util

from maml_bbb_2 import MAML
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#%%

FLAGS = flags.FLAGS

flags.DEFINE_integer('dim_w', 196, 'dimension of w')
flags.DEFINE_integer('dim_im', 128, 'dimension of image')
flags.DEFINE_integer('dim_y', 1, 'dimension of w')
flags.DEFINE_string('data_dir', None,
                    'Directory of data files.')
get_data_dir = lambda: FLAGS.data_dir
flags.DEFINE_list('data', ['train_data_ins.pkl', 'val_data_ins.pkl'],
                  'data name')

## Training options
flags.DEFINE_float('beta', 1e-3, 'beta for IB')
flags.DEFINE_integer(
    'num_classes', 1,
    'number of classes used in classification (e.g. 5-way classification).')
flags.DEFINE_integer(
    'update_batch_size', 15,
    'number of examples used for inner gradient update (K for K-shot learning).'
)
flags.DEFINE_integer('num_updates', 5,
                     'number of inner gradient updates during training.')
flags.DEFINE_integer('meta_batch_size', 10,
                     'number of tasks sampled per meta-update')
flags.DEFINE_integer('test_num_updates', 20,
                     'number of inner gradient updates during test.')

flags.DEFINE_float('meta_lr', 0.002, 'the base learning rate of the generator')
flags.DEFINE_float(
    'update_lr', 0.002,
    'step size alpha for inner gradient update.')  # 0.1 for omniglot
flags.DEFINE_integer(
    'metatrain_iterations', 30000,
    'number of metatraining iterations.')  # 15k for omniglot, 50k for sinusoid

flags.DEFINE_float('noise_scale', 0., 'scale of nosei to add')
flags.DEFINE_integer('num_noise', 0, 'number of discrete noises to sample from.')
## Model options
flags.DEFINE_integer(
    'num_filters', 64,
    'number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
flags.DEFINE_bool(
    'conv', True,
    'whether or not to use a convolutional network, only applicable in some cases'
)
flags.DEFINE_bool(
    'max_pool', False,
    'Whether or not to use max pooling rather than strided convolutions')
flags.DEFINE_bool(
    'stop_grad', False,
    'if True, do not use second derivatives in meta-optimization (for speed)')
flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')

## Logging, saving, and testing options
flags.DEFINE_string('logdir', '/tmp/data',
                    'directory for summaries and checkpoints.')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('trial', 1, 'trial_num')
flags.DEFINE_float('var', -20.0, 'var initial')

def train(model, sess, checkpoint_dir):
    directory = os.path.join(FLAGS.logdir,checkpoint_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)


    print('Done initializing, start training.')
    #old_time = time.time()
    SUMMARY_INTERVAL = 5
    PRINT_INTERVAL = 50
    EXPERIMENT = 'pose'+ str(FLAGS.update_batch_size)+'shot'+str(FLAGS.beta)

    summary_writer = tf.summary.FileWriter(checkpoint_dir, sess.graph)
    prelosses, postlosses = [], []
    prelosses_val, postlosses_val = [], []
    iter_r = []; pre_train_r = []; post_train_r = [];
    pre_val_r = []; post_val_r = []

    for itr in range( FLAGS.metatrain_iterations):
        input_tensors = [model.metatrain_op]

        if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
            input_tensors.extend([model.total_loss1, model.total_losses2[-1]])
            input_tensors_val = [
                  model.metaval_total_loss1, model.metaval_total_losses2[-1]
              ]
        result = sess.run(input_tensors, feed_dict={})

        if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
            summary, result_val = sess.run([model.summ_op, input_tensors_val], feed_dict={})
            summary_writer.add_summary(summary, itr)
            prelosses.append(result[-2])
            postlosses.append(result[-1])
            prelosses_val.append(result_val[-2]) #0 step loss on meta_val training set
            postlosses_val.append(result_val[-1]) #K step loss on meta_val validation set

        if (itr!=0) and itr % PRINT_INTERVAL == 0:
            print('###############################')
            print(' Iteration ' + str(itr) + ':')
            print('Training: ', 'pre -->', np.mean(prelosses), 'post-->', np.mean(postlosses))
            print('Validation: ', 'pre-->', np.mean(prelosses_val), 'post-->', np.mean(postlosses_val))
            #print('time =', time.time() - old_time)
            print('###############################')

            iter_r.append(itr)
            pre_train_r.append(np.mean(prelosses)); post_train_r.append(np.mean(postlosses))
            pre_val_r.append(np.mean(prelosses_val)); post_val_r.append(np.mean(postlosses_val))
            all_ = [iter_r, pre_train_r,post_train_r, pre_val_r, post_val_r]
            pickle.dump(all_, open(os.path.join(directory,EXPERIMENT), 'wb'))

            prelosses, postlosses = [], []
            prelosses_val, postlosses_val = [], []


def get_batch(x, y):
  """Get data batch."""
  xs, ys, xq, yq = [], [], [], []
  for _ in range(FLAGS.meta_batch_size):
    # sample WAY classes
    classes = np.random.choice(
        list(range(np.shape(x)[0])), size=FLAGS.num_classes, replace=False)

    support_set = []
    query_set = []
    support_sety = []
    query_sety = []
    for k in list(classes):
      # sample SHOT and QUERY instances
      idx = np.random.choice(
          list(range(np.shape(x)[1])),
          size=FLAGS.update_batch_size + FLAGS.update_batch_size,
          replace=False)
      x_k = x[k][idx]
      y_k = y[k][idx].copy()
      if FLAGS.noise_scale:
        u = np.random.uniform(-FLAGS.noise_scale, FLAGS.noise_scale)
        y_k = (y_k + u) % 1.0
      elif FLAGS.num_noise:
        noise_values = np.linspace(0, 1, FLAGS.num_noise+1)[:-1]
        u = np.random.choice(noise_values)
        y_k = (y_k + u) % 1.0
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


def main(_):
  dim_output = FLAGS.dim_y
  dim_input = FLAGS.dim_im * FLAGS.dim_im * 1

  exp_name = '%s.num_noise-%g.noise-%g.beta-%g.meta_lr-%g.update_lr-%g.trial-%d' % (
      'maml_bbb',FLAGS.num_noise, FLAGS.noise_scale, FLAGS.beta, FLAGS.meta_lr, FLAGS.update_lr, FLAGS.trial)
  checkpoint_dir = os.path.join(FLAGS.logdir, exp_name)

  x_train, y_train = pickle.load(open(os.path.join(FLAGS.data_dir,FLAGS.data[0]),'rb'))
  x_val, y_val = pickle.load(open(os.path.join(FLAGS.data_dir,FLAGS.data[1]),'rb'))

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

  kernel_posterior_fn = tfp_layers_util.default_mean_field_normal_fn(
      untransformed_scale_initializer=tf.compat.v1.initializers.random_normal(
          mean=FLAGS.var, stddev=0.1))

  encoder_w = tf.keras.Sequential([
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

  xa, labela, xb, labelb = ds_train.make_one_shot_iterator().get_next()
  xa = tf.reshape(xa, [-1, 128, 128, 1])
  xb = tf.reshape(xb, [-1, 128, 128, 1])
  with tf.variable_scope('encoder'):
    inputa = encoder_w(xa)
  inputa = tf.reshape(
      inputa, [-1, FLAGS.update_batch_size * FLAGS.num_classes, FLAGS.dim_w])
  inputb = encoder_w(xb)
  inputb = tf.reshape(
      inputb, [-1, FLAGS.update_batch_size * FLAGS.num_classes, FLAGS.dim_w])

  input_tensors = {'inputa': inputa,\
                   'inputb': inputb, \
                   'labela': labela, 'labelb': labelb}
  # n_task * n_im_per_task * dim_w
  xa_val, labela_val, xb_val, labelb_val = ds_val.make_one_shot_iterator(
  ).get_next()
  xa_val = tf.reshape(xa_val, [-1, 128, 128, 1])
  xb_val = tf.reshape(xb_val, [-1, 128, 128, 1])

  inputa_val = encoder_w(xa_val)
  inputa_val = tf.reshape(
      inputa_val,
      [-1, FLAGS.update_batch_size * FLAGS.num_classes, FLAGS.dim_w])

  inputb_val = encoder_w(xb_val)
  inputb_val = tf.reshape(
      inputb_val,
      [-1, FLAGS.update_batch_size * FLAGS.num_classes, FLAGS.dim_w])

  metaval_input_tensors = {'inputa': inputa_val,\
                           'inputb': inputb_val, \
                           'labela': labela_val, 'labelb': labelb_val}

  # num_updates = max(self.test_num_updates, FLAGS.num_updates)
  model = MAML(encoder_w, FLAGS.dim_w, dim_output)
  model.construct_model(input_tensors=input_tensors, prefix='metatrain_')
  model.construct_model(
      input_tensors=metaval_input_tensors,
      prefix='metaval_',
      test_num_updates=FLAGS.test_num_updates)

  # model.construct_model(input_tensors=input_tensors, prefix='metaval_')
  model.summ_op = tf.summary.merge_all()
  sess = tf.InteractiveSession()

  tf.global_variables_initializer().run()

  if FLAGS.train:
    train(model, sess, checkpoint_dir)


if __name__ == '__main__':
  app.run(main)
