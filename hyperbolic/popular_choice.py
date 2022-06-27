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

"""Collaborative filtering predictions based on popular choice."""
import os

from absl import app
from absl import flags
import tensorflow.compat.v2 as tf
from hyperbolic.datasets.datasets import DatasetClass
from hyperbolic.models.popchoice import PopChoice
import hyperbolic.utils.train as train_utils

tf.enable_v2_behavior()

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'data_dir',
    default='data/',
    help='Path to dataset dir')
flags.DEFINE_string(
    'dataset',
    default='meetup',
    help='Which dataset to use')
flags.DEFINE_boolean(
    'debug',
    default=False,
    help='whether or not to use debug mode')


def main(_):
  # load data
  dataset_path = os.path.join(FLAGS.data_dir, FLAGS.dataset)
  dataset = DatasetClass(dataset_path, FLAGS.debug)
  train_np_examples = dataset.data['train']
  valid_examples = dataset.get_examples('valid')
  test_examples = dataset.get_examples('test')
  filters = dataset.get_filters()
  n_items = dataset.n_items

  # create model
  model = PopChoice(n_items, train_np_examples)

  # eval on valid and test
  valid = train_utils.metric_dict_full_and_random(
      *model.random_eval(valid_examples, filters))
  print(train_utils.format_metrics(valid, split='valid'))
  test = train_utils.metric_dict_full_and_random(
      *model.random_eval(test_examples, filters))
  print(train_utils.format_metrics(test, split='test'))


if __name__ == '__main__':
  app.run(main)
