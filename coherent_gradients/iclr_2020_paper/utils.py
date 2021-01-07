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

"""Utils for constructing and training models."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections

from absl import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

################################################################################
# policies for combining gradients from examples
################################################################################


def policy_sum(t):
  return t.sum(axis=-1)


def policy_sign_sum(t):
  return 0.001 * np.sign(policy_sum(t))


def policy_sum_sign(t):
  return 0.001 * np.sign(t).sum(axis=-1)


def np_winsorize(t, c):
  assert c >= 0 and c <= 100
  # l = np.percentile(t, c,     axis=-1, interpolation='higher', keepdims=True)
  # h = np.percentile(t, 100-c, axis=-1, interpolation='lower',  keepdims=True)
  [l, h] = np.percentile(
      t, [c, 100 - c], axis=-1, interpolation='nearest', keepdims=True)
  u = np.clip(t, l, h)
  return u

# def policy_winsorize_sum(t):
#   c = FLAGS.winsorize_pct
#   return policy_sum(np_winsorize(t, c))


def get_policy(config):
  """Returns gradient aggregation policy."""
  policy = config['policy']
  if policy == 'baseline':
    return None
  if policy == 'sum':
    return policy_sum
  if policy == 'sign_sum':
    return policy_sign_sum
  if policy == 'sum_sign':
    return policy_sum_sign
  assert policy == 'winsorize_sum'
  c = config['winsorize_pct']
  return lambda t: policy_sum(np_winsorize(t, c))


################################################################################
# load data
################################################################################


def randomize_labels(t, p):
  assert len(t.shape) == 1
  assert 0 <= p and p <= 1
  o = np.array(t)
  c = np.random.binomial(1, p, size=o.shape[0])
  a = o[c == 1]
  b = np.random.permutation(a)
  o[c == 1] = b
  return o


def load_from_tfds(dataset='cifar10'):
  tz = tfds.as_numpy(tfds.load(dataset, split=tfds.Split.TRAIN, batch_size=-1))
  vz = tfds.as_numpy(tfds.load(dataset, split=tfds.Split.TEST, batch_size=-1))
  tx, ty = tz['image'], tz['label']
  vx, vy = vz['image'], vz['label']
  return (tx, ty), (vx, vy)


def load_data(dataset, randomize_labels_pct=0):
  """Loads data and randomizes labels."""
  (tx, ty), (vx, vy) = load_from_tfds(dataset=dataset)

  oy = ty
  ty = randomize_labels(ty, randomize_labels_pct / 100.)

  tx = tx.reshape(tx.shape[0], -1) / 255.
  vx = vx.reshape(vx.shape[0], -1) / 255.

  logging.info('tx.shape = %s', tx.shape)
  logging.info('vx.shape = %s', vx.shape)

  ty = tf.keras.utils.to_categorical(ty, num_classes=10)
  vy = tf.keras.utils.to_categorical(vy, num_classes=10)
  return (tx, ty), (vx, vy), oy

################################################################################
# data feeder
################################################################################


class Feeder(object):
  """Feeder class."""

  def __init__(self, dataset, mb_size):
    (tx, ty,), (vx, vy), oy = dataset
    assert len(tx.shape) == 2
    assert len(ty.shape) == 2
    assert tx.shape[0] == ty.shape[0]
    assert len(vx.shape) == 2
    assert len(vy.shape) == 2
    assert vx.shape[0] == vy.shape[0]
    self.tx = tx
    self.ty = ty
    self.vx = vx
    self.vy = vy
    self.oy = oy
    self.x = tf.compat.v1.placeholder(tf.float32, shape=(None, tx.shape[1]))
    self.o = self.x
    self.y = tf.compat.v1.placeholder(tf.float32, shape=(None, ty.shape[1]))
    self.mb = mb_size
    self.count = self.tx.shape[0]  # trigger a shuffle upon starting

    assert tx.shape[0] % mb_size == 0, 'want all mini batches to be same size'

  def add(self, loss):
    pass

  def read(self, mode, feed_dict, _):
    """Populates feed directory."""
    if mode == 'train':
      if self.count >= self.tx.shape[0]:
        self.count = 0
        p = np.random.permutation(self.tx.shape[0])
        self.val_x = self.tx[p]
        self.val_y = self.ty[p]

      feed_dict[self.x] = self.val_x[self.count : self.count + self.mb]
      feed_dict[self.y] = self.val_y[self.count : self.count + self.mb]
      self.count += self.mb

    elif mode == 'eval_train':

      feed_dict[self.x] = self.tx
      feed_dict[self.y] = self.ty

    elif mode == 'eval_val':

      feed_dict[self.x] = self.vx
      feed_dict[self.y] = self.vy

    else:

      assert False, 'bad mode: {}'.format(mode)

  def write(self, mode, out_dict, step):
    pass


################################################################################
# linear layer
################################################################################


class Linear(object):
  """Linear layer."""

  def __init__(self, x, width, activation, layer_id, config):
    assert activation in ['relu', None]
    assert len(x.shape) == 2
    self.x = x
    self.nin = x.get_shape()[1].value
    self.width = width
    self.config = config
    self.lr = config['learning_rate']
    self.policy = get_policy(config)
    self.weights = tf.compat.v1.placeholder(
        tf.float32, shape=(self.x.shape[1], self.width))
    self.bias = tf.compat.v1.placeholder(tf.float32, shape=(self.width,))
    self.preact = tf.add(tf.matmul(self.x, self.weights), self.bias)
    self.o = tf.nn.relu(self.preact) if activation == 'relu' else self.preact

    self.val_weights = (
        np.random.normal(0, 1, size=(self.nin, self.width)) * 2.0 /
        np.sqrt(self.nin + self.width))
    self.val_bias = np.zeros(shape=width)
    self.layer_id = layer_id

    ntrack_weights = min(300, self.nin * self.width)
    self.track_weights = np.random.RandomState(0).choice(
        self.nin * self.width, ntrack_weights, replace=False)
    # we don't know how many example we'll get till later
    self.track_examples = None

  def add(self, loss):
    self.dl_weights, self.dl_bias, self.dl_preact = tf.gradients(
        loss, [self.weights, self.bias, self.preact])

  def read(self, _, feed_dict, targets):
    feed_dict[self.weights] = self.val_weights
    feed_dict[self.bias] = self.val_bias

    targets.append(self.dl_weights)
    targets.append(self.dl_bias)
    targets.append(self.dl_preact)
    targets.append(self.x)

  def write(self, mode, out_dict, step):
    """Updates weights."""
    val_dl_weights = out_dict[self.dl_weights]
    val_dl_bias = out_dict[self.dl_bias]
    val_dl_preact = out_dict[self.dl_preact]
    val_x = out_dict[self.x]

    if self.policy is not None:

      if mode == 'train':

        val_dl_bias_per_example = np.einsum('mo->om', val_dl_preact)
        val_dl_weights_per_example = np.einsum('mo,mi->iom', val_dl_preact,
                                               val_x)

        self.val_weights -= self.lr * self.policy(val_dl_weights_per_example)
        self.val_bias -= self.lr * self.policy(val_dl_bias_per_example)

      elif mode == 'eval_train':

        if self.config['log_gradients']:

          if self.track_examples is None:
            ntrack_examples = 400
            self.track_examples = np.random.RandomState(1).choice(
                val_x.shape[0], ntrack_examples, replace=False)

          # redo the computation as for training but limited only to examples we
          # are tracking
          t = np.einsum('mo,mi->iom', val_dl_preact[self.track_examples],
                        val_x[self.track_examples]).reshape(
                            -1,
                            self.track_examples.shape[0])[self.track_weights]

          with open(
              '{}/gradients-{}.csv'.format(self.config['output_dir'],
                                           self.config['idx']), 'a+') as f:
            for i_w, w in enumerate(self.track_weights):
              for i_e, e in enumerate(self.track_examples):
                f.write('{},{},{},{},{}\n'.format(self.layer_id, step, w, e,
                                                  t[i_w, i_e]))

    else:  # policy is None, i.e., (performance) baseline

      if mode == 'train':
        self.val_weights -= self.lr * val_dl_weights
        self.val_bias -= self.lr * val_dl_bias


################################################################################
# loss layer
################################################################################


class LossLayer(object):
  """Loss layer."""

  def __init__(self, yh, y, oy, config):
    self.config = config
    self.yh = yh
    self.y = y
    self.oy = oy  # original training labels (when labels are permuted)
    self.loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=yh, labels=y))
    self.acc = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(yh, 1), tf.argmax(y, 1)), tf.float32))

    inf = np.iinfo(np.int32).max
    self.learnt_first = np.full_like(oy, inf, dtype=np.int32)
    self.learnt_last = np.full_like(oy, inf, dtype=np.int32)

  def add(self, loss):
    pass

  def read(self, mode, _, targets):
    """Populates targets."""
    targets.append(self.loss)
    targets.append(self.acc)

    if mode == 'eval_train':
      targets.append(self.yh)
      targets.append(self.y)

  def write(self, mode, out_dict, step):
    """Writes output to a csv file."""
    if mode != 'eval_train':
      return

    val_yh = out_dict[self.yh]
    val_y = out_dict[self.y]

    syh = np.argmax(val_yh, axis=1)
    sy = np.argmax(val_y, axis=1)

    assert syh.shape == sy.shape

    inf = np.iinfo(np.int32).max

    self.learnt_first[syh == sy] = np.minimum(step,
                                              self.learnt_first[syh == sy])
    self.learnt_last[syh == sy] = np.minimum(step, self.learnt_last[syh == sy])
    self.learnt_last[syh != sy] = inf

    df = pd.DataFrame(
        collections.OrderedDict([('learnt_first', self.learnt_first),
                                 ('learnt_last', self.learnt_last),
                                 ('current', step), ('pred', syh), ('expt', sy),
                                 ('orig', self.oy)]))

    with open(
        '{}/when-learnt-{}.csv'.format(self.config['output_dir'],
                                       self.config['idx']), 'w') as f:
      df.to_csv(f, index_label='training_example')


################################################################################
# overall network
################################################################################

SPECIAL_STEPS = frozenset(
    list(np.arange(10)) + list(np.arange(10) * 10) + list(np.arange(10) * 100) +
    list(np.arange(10) * 1000))


def should_log(step):
  return (step in SPECIAL_STEPS) or (step % 10000 == 0)


class Network(object):
  """Represents neural network model."""

  def __init__(self, dataset, config):
    (tx, _), (_, _), oy = dataset
    tf.compat.v1.reset_default_graph()
    layers = []
    self.config = config
    self.max_steps = int(1.0 * tx.shape[0] * config['max_epochs'] /
                         config['mb_size'])
    self.log_every = self.config['log_every']

    device = '/gpu:0' if config['use_gpu'] else '/cpu:0'

    logging.info('using tf.device %s', device)

    with tf.device(device):
      layers.append(Feeder(dataset, config['mb_size']))
      for i in range(config['hidden']):
        layers.append(
            Linear(layers[-1].o, config['width'], 'relu',
                   'layer_{}'.format(i + 1), config))
      layers.append(Linear(layers[-1].o, 10, None, 'layer_out', config))
      layers.append(LossLayer(layers[-1].o, layers[0].y, oy, config))

    self.feeder = layers[0]
    self.loss_layer = layers[-1]

    self.loss = self.loss_layer.loss
    self.acc = self.loss_layer.acc

    for layer in layers:
      layer.add(self.loss)

    self.layers = layers
    self.session = tf.compat.v1.Session()
    init = tf.compat.v1.global_variables_initializer()
    self.session.run(init)

  def train(self):
    """Train the network."""
    for step in range(self.max_steps):

      # don't train in step zero so we know how things start off
      if step == 0:
        out_batch = {self.acc: 0, self.loss: 0}
      else:
        out_batch = self.run_step('train', step)

      if (should_log(step) or step == self.max_steps - 1 or step == 0):

        out_train = self.run_step('eval_train', step)
        out_val = self.run_step('eval_val', step)

        logging.info(
            'step = {:6d} (of {:6d}), ta = {:6.3f}, va = {:6.3f}, tl = {:6.3f}, vl = {:6.3f}, ba = {:6.3f}, bl = {:6.3f}'
            .format(step, self.max_steps, out_train[self.acc],
                    out_val[self.acc], out_train[self.loss], out_val[self.loss],
                    out_batch[self.acc], out_batch[self.loss]))

        with open(
            '{}/scalars-{}.csv'.format(self.config['output_dir'],
                                       self.config['idx']), 'a+') as f:
          f.write(
              '{:6d}, {:6.3f}, {:6.3f}, {:6.3f}, {:6.3f}, {:6.3f}, {:6.3f}\n'
              .format(step, out_train[self.acc], out_val[self.acc],
                      out_train[self.loss], out_val[self.loss],
                      out_batch[self.acc], out_batch[self.loss]))

  def run_step(self, mode, step):
    """Make one pass through the network."""

    assert mode in ['train', 'eval_train', 'eval_val']

    feed_dict = {}
    targets = [self.loss, self.acc]
    for layer in self.layers:
      layer.read(mode, feed_dict, targets)
    values = self.session.run(targets, feed_dict=feed_dict)
    out_dict = dict(zip(targets, values))
    for layer in self.layers:
      layer.write(mode, out_dict, step)

    return out_dict
