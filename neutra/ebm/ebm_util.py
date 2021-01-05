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

# python3
"""Utilities for the EBM model."""

import io
import logging
import math
import os
import pickle
import random
import sys

from absl import flags
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import numpy as np
import PIL
import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS

flags.DEFINE_boolean('run_tf_functions_eagerly', False,
                     'Run TF functions eagerly for debugging.')
flags.DEFINE_string('mnist_path', None, 'Path to the `mnist.npz`')
flags.DEFINE_string('celeba_path', None, 'Path to the `celeba_40000_32.pickle`')

IMAGE_BINS = 256


def mnist_dataset(num_channels=1):
  """Loads the MNIST dataset.

  This loads the dataset from `mnist_path` flag. The dataset is padded to be
  32x32. Optionally, this also duplicates the grayscale channel `num_channels`
  times.

  Args:
    num_channels: How many channels the output should have.

  Returns:
    x_train: A numpy array with shape `[10000, 32, 32, num_channels]`.
  """
  with tf.io.gfile.GFile(FLAGS.mnist_path, mode='rb') as f:
    with np.load(f, allow_pickle=True) as npf:
      x_train = npf['x_train']

  x_train = np.lib.pad(x_train, ((0, 0), (2, 2), (2, 2)), 'minimum')
  x_train = x_train[Ellipsis, tf.newaxis]
  x_train = np.repeat(x_train, num_channels, axis=3)

  return x_train


def celeba_dataset():
  """Loads the CELEB-A dataset.

  This loads the dataset from `celeba_path` flag.uld have.

  Returns:
    x_train: A numpy array with shape `[10000, 32, 32, 3]`.
  """
  with tf.io.gfile.GFile(FLAGS.celeba_path, mode='rb') as f:
    x_train = pickle.load(f)
  return x_train


def init_tf2():
  """Initializes TF2."""
  tf.enable_v2_behavior()
  tf.config.set_soft_device_placement(True)
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    for gpu in gpus:
      # To help debug OOMs at the expense of speed.
      tf.config.experimental.set_memory_growth(gpu, True)

  tf.config.experimental_run_functions_eagerly(FLAGS.run_tf_functions_eagerly)


class LambdaLr(tf.optimizers.schedules.LearningRateSchedule):
  """A learning rate schedule created from a lambda."""

  def __init__(self, f, name='LambdaLr'):
    super(LambdaLr, self).__init__()
    self.f = f
    self.name = name

  def __call__(self, step):
    return self.f(step)

  def get_config(self):
    return {}


def set_seed(seed):
  """Set the random seed for everything."""
  assert seed
  random.seed(seed)
  np.random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  tf.random.set_seed(seed)


def setup_logging(name, f, console=True):
  """Set up a logging system."""
  log_format = logging.Formatter('%(asctime)s : %(message)s')
  logger = logging.getLogger(name)
  logger.handlers = []
  file_handler = logging.StreamHandler(f)
  file_handler.setFormatter(log_format)
  logger.addHandler(file_handler)
  if console:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
  logger.setLevel(logging.INFO)
  return logger


def data_preprocess(x):
  """Preprocess the data before feeding into the model."""
  x = tf.cast(x, tf.float32)
  x = x / IMAGE_BINS - .5
  return x


def data_postprocess(x):
  """Postprocess the samples from the model before plotting."""
  return tf.cast(
      tf.clip_by_value(tf.floor((x + .5) * IMAGE_BINS), 0, IMAGE_BINS - 1),
      tf.uint8)


def data_discrete_noise(x):
  """Add noise to mimic discretization as far as information content goes."""
  return x + tf.random.uniform(tf.shape(x), 0, 1. / IMAGE_BINS)


def nearby_difference(x):
  """Compute L2 norms for nearby entries in a batch."""
  # This is a very rough measure of diversity.
  with tf.device('cpu'):
    x1 = tf.reshape(x, shape=[int(x.shape[0]), -1])
    x2 = tf.roll(x1, shift=1, axis=0)
    return tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x1, x2))))


def _to_grid(image_batch, size):
  """Stacks images into a grid."""
  h, w = image_batch.shape[1], image_batch.shape[2]
  c = image_batch.shape[3]
  img = np.zeros((int(h * size[0]), w * size[1], c))
  for idx, im in enumerate(image_batch):
    i = idx % size[1]
    j = idx // size[1]
    img[j * h:j * h + h, i * w:i * w + w, :] = im
  return img


def plot(x, path):
  """Plots `x` and saves figure to `path`."""
  with tf.io.gfile.GFile(path, mode='w') as f:
    n_batch = int(x.shape[0])
    if int(math.sqrt(n_batch))**2 != n_batch:
      raise ValueError('This function only accepts square batches.')
    PIL.Image.fromarray(
        np.squeeze(
            _to_grid(x, [int(math.sqrt(n_batch)),
                         int(math.sqrt(n_batch))]).astype(np.uint8))).save(f)


def plot_n_by_m(x, path, n, m):
  """Plots `x` and saves figure to `path`."""
  with tf.io.gfile.GFile(path, mode='w') as f:
    PIL.Image.fromarray(
        np.squeeze(_to_grid(x, [int(n), int(m)]).astype(np.uint8))).save(f)


def plot_stat(stat_keys, stats, stats_i, output_dir):
  """Plots statistics."""
  p_n = len(stats)
  fig = plt.figure(figsize=(20, p_n * 5))

  p_i = 1
  for k in stat_keys:
    plt.subplot(p_n, 1, p_i)
    plt.plot(stats_i, stats[k])
    plt.ylabel(k)
    p_i += 1

  buf = io.BytesIO()
  plt.savefig(buf, format='pdf', bbox_inches='tight')
  plt.close(fig)
  buf.seek(0)

  with tf.io.gfile.GFile(os.path.join(output_dir, 'stats.pdf'), mode='wb') as f:
    f.write(buf.read(-1))


class SpectralNormalization(tf.keras.layers.Wrapper):
  """Spectral normalization wrapper.

  From https://arxiv.org/abs/1802.05957.
  """

  def build(self, input_shape):
    if not self.layer.built:
      self.layer.build(input_shape)

      assert hasattr(self.layer, 'kernel')

      self.w = self.layer.kernel
      self.w_shape = self.w.shape.as_list()
      self.u = self.add_variable(
          shape=tuple([1, self.w_shape[-1]]),
          initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
          name='sn_u',
          trainable=False,
          dtype=tf.float32)

    super(SpectralNormalization, self).build()

  @tf.function
  def call(self, inputs):
    self._compute_weights()
    output = self.layer(inputs)
    return output

  def _compute_weights(self, eps=1e-12):
    w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])
    u = tf.identity(self.u)
    v = tf.matmul(u, w_reshaped, transpose_b=True)
    v = v / tf.maximum(tf.linalg.norm(v), eps)
    u = tf.matmul(v, w_reshaped)
    u = u / tf.maximum(tf.linalg.norm(u), eps)

    self.u.assign(u)
    sigma = tf.matmul(tf.matmul(v, w_reshaped), u, transpose_b=True)

    self.layer.kernel = self.w / sigma

  def compute_output_shape(self, input_shape):
    return tf.TensorShape(
        self.layer.compute_output_shape(input_shape).as_list())


def lipswish(x):
  """Activation function from https://arxiv.org/abs/1906.02735."""
  return tf.nn.swish(x) / 1.1
