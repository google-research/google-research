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

"""Tests for T5 CBQA tasks."""

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import t5
import tensorflow.compat.v1 as tf

import t5_closed_book_qa.t5_cbqa.tasks  # pylint:disable=unused-import

tf.disable_v2_behavior()
tf.enable_eager_execution()

MixtureRegistry = t5.data.MixtureRegistry
TaskRegistry = t5.data.TaskRegistry
_SEQUENCE_LENGTH = {'inputs': 512, 'targets': 114}

_TASKS = [
    'natural_questions_nocontext', 'natural_questions_nocontext_test',
    'natural_questions_open', 'natural_questions_open_test',
    'natural_questions_open_randanswer',
    'web_questions_open', 'web_questions_open_test',
    'trivia_qa_open', 'trivia_qa_open_test',
]

_MIXTURES = ['closed_book_qa', 'closed_book_qa_test']


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
