# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Tests for task_set.tasks.language_model."""
from task_set.tasks import family_test_utils
from task_set.tasks import language_model
import tensorflow.compat.v1 as tf


class CharLanguageModelTest(family_test_utils.TaskFamilyTestCase):

  def __init__(self, *args, **kwargs):
    super(CharLanguageModelTest, self).__init__(
        language_model.sample_char_rnn_language_model_family_cfg,
        language_model.get_char_language_model_family, *args, **kwargs)


class WordLanguageModelTest(family_test_utils.TaskFamilyTestCase):

  def __init__(self, *args, **kwargs):
    super(WordLanguageModelTest,
          self).__init__(language_model.sample_word_language_model_family_cfg,
                         language_model.get_word_language_model_family, *args,
                         **kwargs)


if __name__ == "__main__":
  tf.test.main()
