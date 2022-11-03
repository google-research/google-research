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

"""Tests for WT5 tasks."""

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import t5
import tensorflow.compat.v1 as tf

import wt5.wt5.mixtures  # pylint:disable=unused-import
import wt5.wt5.tasks  # pylint:disable=unused-import

tf.disable_v2_behavior()
tf.enable_eager_execution()

MixtureRegistry = t5.data.MixtureRegistry
TaskRegistry = t5.data.TaskRegistry
_SEQUENCE_LENGTH = {'inputs': 2048, 'targets': 512}

_TASKS = [
    'esnli_v010',
    'esnli_v010_0_expln',
    'esnli_explanations_take100_v010',
    'esnli_labels_skip100_v010',
    'mnli_v002',
    'cos_e_v001',
    'cos_e_v001_0_expln_like_esnli',
    'cos_e_explanations_take100_v001',
    'cos_e_labels_skip100_v001',
    'movie_rationales_v010',
    'movie_rationales_v010_no_expl',
    'imdb_reviews_v100',
    'amazon_reviews_books_v1_00_v010',
]

_MIXTURES = [
    'cos_e_100_explanations',
    'esnli_100_explanations',
    'esnli_mnli_all_explanations',
    'imdb_reviews_movie_rationales',
    'esnli_cos_e_transfer',
    'movie_rationales_100_explanations',
    'amazon_books_movies_equal',
]


class TasksTest(parameterized.TestCase):

  @parameterized.parameters(((name,) for name in _TASKS))
  def test_task(self, name):
    task = TaskRegistry.get(name)
    logging.info('task=%s', name)
    ds = task.get_dataset(_SEQUENCE_LENGTH, 'train')
    for d in ds:
      logging.info(d)
      break

  @parameterized.parameters(((name,) for name in _MIXTURES))
  def test_mixture(self, name):
    mixture = MixtureRegistry.get(name)
    logging.info('mixture=%s', name)
    ds = mixture.get_dataset(_SEQUENCE_LENGTH, 'train')
    for d in ds:
      logging.info(d)
      break


if __name__ == '__main__':
  absltest.main()
