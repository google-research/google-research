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

""" Utility functions. """
import numpy as np
import os
import random
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.python.platform import flags
from tensorflow.contrib.layers.python import layers as contrib_layers_python_layers

FLAGS = flags.FLAGS

## Image helper
def get_images(paths, labels, nb_samples=None, shuffle=True, train=True):
  if nb_samples is not None:
    if FLAGS.expt_number == '8' or FLAGS.expt_number == '8a' or FLAGS.expt_number == '8c':
      sampler = lambda x: random.sample(x[:(len(x)//4)], nb_samples)
    elif (FLAGS.expt_number == '11a1' or FLAGS.expt_number == '11b1' or FLAGS.expt_number == '11c1') and train:
      sampler = lambda x: random.sample(x[:(len(x)//4)], nb_samples)
    elif (FLAGS.expt_number == '11a2' or FLAGS.expt_number == '11b2' or FLAGS.expt_number == '11c2') and train:
      sampler = lambda x: random.sample(x[:(len(x)//2)], nb_samples)
    elif (FLAGS.expt_number == '11a3' or FLAGS.expt_number == '11b3' or FLAGS.expt_number == '11c3') and train:
      sampler = lambda x: random.sample(x[:((3*len(x))//4)], nb_samples)
    else:
      sampler = lambda x: random.sample(x, nb_samples)
  else:
    sampler = lambda x: x
  images = [(i, os.path.join(path, image)) \
      for i, path in zip(labels, paths) \
      for image in sampler(os.listdir(path))]
  if shuffle:
    random.shuffle(images)
  return images

## Network helpers
def conv_block(inp, cweight, bweight, reuse, scope, activation=tf.nn.relu, max_pool_pad='VALID', residual=False):
  """ Perform, conv, batch norm, nonlinearity, and max pool """
  stride, no_stride = [1,2,2,1], [1,1,1,1]

  if FLAGS.max_pool:
    conv_output = tf.nn.conv2d(inp, cweight, no_stride, 'SAME') + bweight
  else:
    conv_output = tf.nn.conv2d(inp, cweight, stride, 'SAME') + bweight
  normed = normalize(conv_output, activation, reuse, scope)
  if FLAGS.max_pool:
    normed = tf.nn.max_pool(normed, stride, stride, max_pool_pad)
  return normed

def normalize(inp, activation, reuse, scope):
  if FLAGS.norm == 'batch_norm':
    return contrib_layers_python_layers.batch_norm(
        inp, activation_fn=activation, reuse=reuse, scope=scope)
  elif FLAGS.norm == 'layer_norm':
    return contrib_layers_python_layers.layer_norm(
        inp, activation_fn=activation, reuse=reuse, scope=scope)
  elif FLAGS.norm == 'None':
    if activation is not None:
      return activation(inp)
    else:
      return inp

## Loss functions
def mse(pred, label):
  pred = tf.reshape(pred, [-1])
  label = tf.reshape(label, [-1])
  return tf.reduce_mean(tf.square(pred-label))

def xent(pred, label):
  # Note - with tf version <=0.12, this loss has incorrect 2nd derivatives
  return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label) / FLAGS.update_batch_size
