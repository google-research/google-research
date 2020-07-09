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

# pylint: skip-file
"""MAML w/o meta regularization.

Based on code by Chelsea Finn (https://github.com/cbfinn/maml).
and Mingzhang Yin (https://github.com/google-research/google-research/tree/master/meta_learning_without_memorization)
"""
from __future__ import print_function
import functools
import os
import pickle
import random
import time
from absl import app
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf
#from meta_learning_without_memorization.pose_code.maml_vanilla_2 import MAML
from maml_vanilla_2 import MAML

FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_string('datasource', 'pose',
                    'sinusoid or omniglot or miniimagenet')
flags.DEFINE_integer('dim_w', 196, 'dimension of w')
flags.DEFINE_integer('dim_im', 128, 'dimension of image')
flags.DEFINE_integer('dim_y', 1, 'dimension of w')
flags.DEFINE_string('data_dir', None,
                    'Directory of data files.')
get_data_dir = lambda: FLAGS.data_dir
flags.DEFINE_list('data', ['train_data_ins.pkl', 'val_data_ins.pkl'],
                  'data name')

## Training options
flags.DEFINE_float('beta', 0.001, 'the beta for weight decay')
flags.DEFINE_bool('weight_decay', True, 'whether or not to use weight decay')
flags.DEFINE_integer('num_noise', 0, 'Discrete noise augmentation.')
flags.DEFINE_float('noise_scale', 0, 'Add noise to canonical pose')
flags.DEFINE_bool('testing', True, 'whether to split train set into val or not.')

flags.DEFINE_integer(
    'num_classes', 1,
    'number of classes used in classification (e.g. 5-way classification).')
flags.DEFINE_integer(
    'update_batch_size', 10,
    'number of examples used for inner gradient update (K for K-shot learning).'
)
flags.DEFINE_integer('num_updates', 5,
                     'number of inner gradient updates during training.')
flags.DEFINE_integer('meta_batch_size', 10,
                     'number of tasks sampled per meta-update')

flags.DEFINE_float('meta_lr', 0.0005, 'the base learning rate of the generator')
flags.DEFINE_float(
    'update_lr', 0.002,
    'step size alpha for inner gradient update.')  # 0.1 for omniglot
flags.DEFINE_integer(
    'metatrain_iterations',10000,
    'number of metatraining iterations.')  # 15k for omniglot, 50k for sinusoid

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
flags.DEFINE_string('extra', 'exp', 'extra info')
flags.DEFINE_integer('trial', 1, 'trial_num')

def train(model, sess, checkpoint_dir, _):
  """Train model."""
  print('Done initializing, starting training.')

  old_time = time.time()
  SUMMARY_INTERVAL = 50  # pylint: disable=invalid-name
  PRINT_INTERVAL = 50  # pylint: disable=invalid-name
  TEST_PRINT_INTERVAL = 50  # pylint: disable=invalid-name
  prelosses, postlosses = [], []
  prelosses_val, postlosses_val = [], []

  train_0 = []
  train_k = []
  train_step = []
  val_0 = []
  val_k = []
  val_step = []

  # Merge all the summaries and write them out to

  summary_writer = tf.summary.FileWriter(checkpoint_dir, sess.graph)

  tf.global_variables_initializer().run()

  for itr in range(FLAGS.metatrain_iterations):
    #print('###############################')
    #print(itr)
    #print('###############################')
    feed_dict = {}
    input_tensors = [model.metatrain_op]

    if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
      input_tensors.extend([model.total_loss1, model.total_losses2[-1]])
      input_tensors_val = [
          model.metaval_total_loss1, model.metaval_total_losses2[-1]
      ]
      feed_dict = {}
      summary, result_val = sess.run([model.summ_op, input_tensors_val],
                                     feed_dict)

      summary_writer.add_summary(summary, itr)

    result = sess.run(input_tensors, feed_dict)

    if itr % SUMMARY_INTERVAL == 0:
      prelosses.append(result[-2])  # 0 step loss on training set
      postlosses.append(result[-1])  # K step loss on validation set

      prelosses_val.append(
          result_val[-2])  # 0 step loss on meta_val training set
      postlosses_val.append(
          result_val[-1])  # K step loss on meta_val validation set

    if (itr != 0) and itr % PRINT_INTERVAL == 0:
      print_str = 'Iteration ' + str(itr)
      print_str += ': ' + str(np.mean(prelosses)) + ', ' + str(
          np.mean(postlosses))
      print(print_str, 'time =', time.time() - old_time)

      train_0.append(np.mean(prelosses))
      train_k.append(np.mean(postlosses))
      train_step.append(itr)
      prelosses, postlosses = [], []
      old_time = time.time()

    if (itr != 0) and itr % TEST_PRINT_INTERVAL == 0:
      print('Validation results: ' + str(np.mean(prelosses_val)) + ', ' +
            str(np.mean(postlosses_val)))
      val_0.append(np.mean(prelosses_val))
      val_k.append(np.mean(postlosses_val))
      val_step.append(itr)
      all_ = (val_step, train_0, train_k, val_0, val_k)
      pickle.dump(all_, open(os.path.join(checkpoint_dir, 'results.p'), 'wb'))
      prelosses_val, postlosses_val = [], []


def test(model, sess, checkpoint_dir):
  """Test model."""
  np.random.seed(1)
  random.seed(1)
  NUM_TEST_POINTS = 600  # pylint: disable=invalid-name
  metaval_accuracies = []

  for _ in range(NUM_TEST_POINTS):
    feed_dict = {}
    feed_dict = {model.meta_lr: 0.0}
    result = sess.run([model.metaval_total_loss1] + model.metaval_total_losses2,
                      feed_dict)
    metaval_accuracies.append(result)

  metaval_accuracies = np.array(metaval_accuracies)
  means = np.mean(metaval_accuracies, 0)
  stds = np.std(metaval_accuracies, 0)
  ci95 = 1.96 * stds / np.sqrt(NUM_TEST_POINTS)

  print('Mean validation accuracy/loss, stddev, and confidence intervals')
  print((means, stds, ci95))
  with open(os.path.join(checkpoint_dir, 'results.p'), 'wb') as f:
      pickle.dump([means, stds, ci95],f)



def get_batch(x, y, is_training):
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
      y_k = y[k][idx].copy()
      if FLAGS.num_noise and is_training:
        # Sample a random canonical pose.
        noise_values = np.linspace(0, 1, FLAGS.num_noise+1)[:-1]
        noise = np.random.choice(noise_values)
        y_k = (y_k + noise) % 1.0
      elif FLAGS.noise_scale and is_training:
        low, high = -FLAGS.noise_scale, FLAGS.noise_scale
        noise = np.random.uniform(low, high)
        # For some reason, compared to np_vanilla.py, y_k is pre-scaled
        # add same noise variate to both support and query batch
        y_k = (y_k + noise) % 1.0

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


def gen(x, y, is_training):
  while True:
      yield get_batch(np.array(x), np.array(y), is_training)


def main(_):
  dim_output = FLAGS.dim_y
  dim_input = FLAGS.dim_im * FLAGS.dim_im * 1

  base_exp_name = 'maml_vanilla'
  if FLAGS.noise_scale:
      base_exp_name = 'maml_vanilla_noise_scale' + str(FLAGS.noise_scale)
  elif FLAGS.num_noise > 0:
      base_exp_name = 'maml_vanilla_noise' + str(FLAGS.num_noise)
  if FLAGS.weight_decay:
    exp_name = '%s.meta_lr-%g.update_lr-%g.beta-%g.trial-%d' % (
        base_exp_name, FLAGS.meta_lr, FLAGS.update_lr, FLAGS.beta, FLAGS.trial)
  else:
    exp_name = '%s.meta_lr-%g.update_lr-%g.trial-%d' % (
        base_exp_name, FLAGS.meta_lr, FLAGS.update_lr, FLAGS.trial)

  if FLAGS.testing:
      exp_name += '-test'
  checkpoint_dir = os.path.join(FLAGS.logdir, exp_name)

  x_train, y_train = pickle.load(
      tf.io.gfile.GFile(os.path.join(get_data_dir(), FLAGS.data[0]), 'rb'))
  x_val, y_val = pickle.load(
      tf.io.gfile.GFile(os.path.join(get_data_dir(), FLAGS.data[1]), 'rb'))

  if not FLAGS.testing:
      x_train, y_train = x_train[:-5], y_train[:-5]
      x_val, y_val = x_val[-5:], y_val[-5:]


  x_train, y_train = np.array(x_train), np.array(y_train)
  # Y VALUES RANGE BETWEEN 0 and 10
  y_train = y_train[:, :, -1, None]
  x_val, y_val = np.array(x_val), np.array(y_val)
  y_val = y_val[:, :, -1, None]

  ds_train = tf.data.Dataset.from_generator(
      functools.partial(gen, x_train, y_train, True),
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
      functools.partial(gen, x_val, y_val, False),
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

  # num_updates = max(self.test_num_updates, FLAGS.num_updates)
  model = MAML(dim_input, dim_output)
  model.construct_model(input_tensors=input_tensors, prefix='metatrain_')
  model.construct_model(
      input_tensors=metaval_input_tensors,
      prefix='metaval_',
      test_num_updates=20)

  # model.construct_model(input_tensors=input_tensors, prefix='metaval_')
  model.summ_op = tf.summary.merge_all()
  sess = tf.InteractiveSession()

  tf.global_variables_initializer().run()

  if FLAGS.train:
    train(model, sess, checkpoint_dir, exp_name)
  #if FLAGS.testing:
  #  test(model, sess, checkpoint_dir)


if __name__ == '__main__':
  app.run(main)
