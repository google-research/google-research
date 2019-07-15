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

# Lint as: python2, python3
"""Library for building tf.data.Datasets for Criteo experiments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
from absl import flags
from absl import logging

import attr
import six
from six.moves import range
import tensorflow.compat.v2 as tf
keras = tf.keras

NUM_INT_FEATURES = 13  # Number of Criteo integer features.
NUM_CAT_FEATURES = 26  # Number of Criteo categorical features.
NUM_TOTAL_FEATURES = NUM_INT_FEATURES + NUM_CAT_FEATURES
INT_FEATURE_INDICES = list(range(1, NUM_INT_FEATURES + 1))
CAT_FEATURE_INDICES = list(range(NUM_INT_FEATURES + 1, NUM_TOTAL_FEATURES + 1))

INT_KEY_TMPL = 'int-feature-%d'
CAT_KEY_TMPL = 'categorical-feature-%d'

NUM_TRAIN_EXAMPLES = int(37e6)

_CAT_UNIQ_CTS = [1472, 577, int(8.2e6), int(1.8e6), 305, 23, 11724, 633, 3,
                 90903, 5918, int(6.4e6), 3207, 27, 15501, int(4.4e6), 10,
                 5485, 2161, 3, int(5.6e6), 17, 15, 273605, 104, 129345]
assert len(_CAT_UNIQ_CTS) == NUM_CAT_FEATURES

flags.DEFINE_string('criteo_train_glob', None,
                    'Glob pattern for Criteo training data.')
flags.DEFINE_string('criteo_valid_glob', None,
                    'Glob pattern for Criteo validation data.')
flags.DEFINE_string('criteo_test_glob', None,
                    'Glob pattern for Criteo testing data.')
flags.DEFINE_string('criteo_dummy_path_for_test', None,
                    'Path to dummy Criteo data for testing.')



@attr.s
class DataConfig(object):
  """Config options for Criteo CTR prediction dataset."""
  split = attr.ib()
  randomize_prob = attr.ib(0.)
  fake_data = attr.ib(False)

  @staticmethod
  def from_name(name, fake_data=False):
    split, randomize_prob = name.split('-')
    return DataConfig(split=split,
                      randomize_prob=float(randomize_prob) / 100,
                      fake_data=fake_data)


def _get_data_glob(split, fake_data):
  if fake_data:
    return flags.FLAGS.criteo_dummy_path_for_test
  return {'train': flags.FLAGS.criteo_train_glob,
          'test': flags.FLAGS.criteo_test_glob,
          'valid': flags.FLAGS.criteo_valid_glob}[split]


def feature_name(idx):
  assert 0 < idx <= NUM_TOTAL_FEATURES
  return INT_KEY_TMPL % idx if idx <= NUM_INT_FEATURES else CAT_KEY_TMPL % idx


def get_categorical_num_unique(index):
  assert NUM_INT_FEATURES < index <= NUM_TOTAL_FEATURES
  return _CAT_UNIQ_CTS[index - NUM_INT_FEATURES - 1]


def _make_features_spec():
  features = {'clicked': tf.io.FixedLenFeature([1], tf.float32)}
  for idx in INT_FEATURE_INDICES:
    features[feature_name(idx)] = tf.io.FixedLenFeature([1], tf.float32, -1)
  for idx in CAT_FEATURE_INDICES:
    features[feature_name(idx)] = tf.io.FixedLenFeature([1], tf.string, '')
  return features


def parse_example(serialized):
  features_spec = _make_features_spec()
  features = tf.io.parse_example(serialized, features_spec)
  return {k: tf.squeeze(v, axis=1) for k, v in six.iteritems(features)}


def _parse_fn(serialized):
  """Parse a dictionary of features and label from a serialized tf.Example."""
  features = parse_example(serialized)
  label = features.pop('clicked')
  return features, label


def apply_randomization(features, label, randomize_prob):
  """Randomize each categorical feature with some probability."""
  rnd_tok = lambda: tf.as_string(tf.random.uniform([], 0, 99999999, tf.int32))

  for idx in CAT_FEATURE_INDICES:
    key = feature_name(idx)
    # Ignore lint since tf.cond should evaluate lambda immediately.
    features[key] = tf.cond(tf.random.uniform([]) < randomize_prob,
                            rnd_tok,
                            lambda: features[key])  # pylint: disable=cell-var-from-loop
  return features, label


def build_dataset(config, batch_size, is_training=False, fake_training=False,
                  repeat=True):
  """Builds a tf.data.Dataset."""
  glob_path = _get_data_glob(config.split, config.fake_data)
  logging.info('Building dataset from glob_path=%s', glob_path)
  cycle_len = 10 if is_training and not fake_training else 1
  out = (tf.data.Dataset.list_files(glob_path, shuffle=is_training)
         .interleave(tf.data.TFRecordDataset, cycle_length=cycle_len))
  if repeat:
    out = out.repeat()
  if is_training:
    out = out.shuffle(20 * batch_size)
  out = out.batch(batch_size).map(_parse_fn)
  if not config.randomize_prob:
    return out
  return out.map(lambda x, y: apply_randomization(x, y, config.randomize_prob))
