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

"""Functions for loading and encoding sequences.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
import time
import tensorflow as tf
import yaml
from tensorflow.contrib import training as contrib_training


def parse_single_tfexample(_, serialized_example):
  """Parsing serialized pb2 example."""
  # read data from serialized examples
  features = tf.parse_single_example(
      serialized_example,
      features={
          'x': tf.FixedLenFeature([], tf.string),
          'y': tf.FixedLenFeature([], tf.int64),
          # z is for sequence origins,
          # i.e. which genome and which position the seq is from
          # 'z': tf.VarLenFeature(tf.string)
      })
  seq_str = features['x']

  x_str = tf.string_split([seq_str], delimiter=' ').values
  features['x'] = tf.string_to_number(x_str, out_type=tf.int32)
  features['y'] = tf.cast(features['y'], dtype=tf.int32)

  return features


def mutate_x(x, mutation_rate, seq_len):
  """Randomly and independently mutate a sequence based on a mutation rate."""
  # generate mutations for all positions,
  # in order to be different than itselves, the mutations have to be >= 1
  # mute the untargeted positions by multiple mask (1 for targeted)
  # adding the mutations to the original, mod 4 if necessary
  tf.set_random_seed(time.time())

  def true_fn():
    """no mutations."""
    return x

  def false_fn():
    """add mutations."""
    mask = tf.cast(
        tf.multinomial(tf.log([[1 - mutation_rate, mutation_rate]]), seq_len),
        tf.int32)[0]
    possible_mutations = tf.random_uniform([seq_len],
                                           minval=1,
                                           maxval=4,
                                           dtype=tf.int32)
    x_new = tf.mod(x + mask * possible_mutations, 4)
    return x_new

  return tf.cond(tf.equal(mutation_rate, 0), true_fn, false_fn)


def parse_single_tfexample_addmutations(_, serialized_example, mutation_rate,
                                        seq_len):
  """Parsing serialized pb2 example and add mutations."""
  # read data from serialized examples
  features = tf.parse_single_example(
      serialized_example,
      features={
          'x': tf.FixedLenFeature([], tf.string),
          'y': tf.FixedLenFeature([], tf.int64),
          # z is for sequence origins,
          # i.e. which genome and which position the seq is from
          # 'z': tf.VarLenFeature(tf.string)
      })
  seq_str = features['x']

  x_str0 = tf.string_split([seq_str], delimiter=' ').values
  x = tf.string_to_number(x_str0, out_type=tf.int32)

  x_new = mutate_x(x, mutation_rate, seq_len)

  features['x'] = x_new
  features['y'] = tf.cast(features['y'], dtype=tf.int32)

  return features


def compute_label_weights_using_sample_size(label_dict, label_sample_size):
  """Compute weights for each class according to their sample sizes.

  Args:
    label_dict: a dictionary with class labels as keys (strings) and encoded
      label index as values (ints).
    label_sample_size: a dictionary with class labels as keys (strings) and
      sample size as values (ints).

  Returns:
    label_weights: weights for labels.
  """

  # keys: encoded class labels, values: sample size
  label_code_sample_size = {
      label_dict[label]: label_sample_size[label]
      for label in label_sample_size.keys()
  }
  print('label_code_sample_size={}'.format(label_code_sample_size))
  # create label weights = 1/label_sample_size
  label_weights = [
      1 / float(label_code_sample_size[idx])
      for idx in range(len(label_code_sample_size))
  ]
  # label_weights = [
  #     x / float(sum(label_weights0)) * len(label_weights0) / float(
  #         params.batch_size) for x in label_weights0
  # ]
  return label_weights


def get_latest_ckpt(tr_model_dir):
  """find previous ckpt and return the filename of the latest ckpt."""
  tf.logging.info('model dir={}'.format(
      os.path.join(tr_model_dir, '*.ckpt.index')))
  list_of_ckpt = tf.gfile.Glob(os.path.join(
      tr_model_dir,
      '*.ckpt.index'))  # * means all if need specific format then *.csv
  # tf.logging.info('list_of_ckpt={}'.format(list_of_ckpt))
  if list_of_ckpt:
    steps = [
        int(os.path.basename(x).split('model_')[1].split('.ckpt')[0])
        for x in list_of_ckpt
    ]
    prev_steps, latest_file0 = [
        (x, y) for x, y in sorted(zip(steps, list_of_ckpt), reverse=True)
    ][0]
    latest_file = latest_file0.replace('.index', '')
    #     latest_file = max(list_of_ckpt, key=os.path.getctime) does not work
    tf.logging.info('previous model exists: {}'.format(latest_file))
    # prev_steps = int(latest_file.split('.')[0].split('_')[1])
    return prev_steps, latest_file.replace('.meta', '')
  else:
    prev_steps = 0
    return 0, None


def get_ckpt_at_step(tr_model_dir, step):
  """find previous ckpt and return the filename of the latest ckpt."""
  tf.logging.info('model dir={}'.format(
      os.path.join(tr_model_dir, '*.ckpt.index')))
  ckpt_file_pattern = os.path.join(tr_model_dir, '*_{}.ckpt.index'.format(step))
  ckpt_file = tf.gfile.Glob(
      ckpt_file_pattern)  # * means all if need specific format then *.csv
  if ckpt_file:
    return step, ckpt_file[0].replace('.index', '')
  else:
    tf.logging.info('Cannot find the ckpt file at step {}'.format(step))
    return step, None


def clean_last_slash_if_any(path):
  return path[:-1] if path.endswith('/') else path


def generate_hparams(params_yaml_file):
  """Create tf.HParams object based on params loaded from yaml file."""
  with tf.gfile.Open(params_yaml_file, mode='rb') as f:
    params_json = yaml.safe_load(f)
    params_dict = json.loads(params_json)
    params = contrib_training.HParams()
    for key, value in params_dict.items():
      params.add_hparam(key, value)
    params.master = ''  # should be 'local' or ''
    params.dropout_rate = 0.0  # turn off dropout for eval

  return params
