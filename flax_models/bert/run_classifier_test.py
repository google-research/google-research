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

"""Tests for flax_models.bert.run_classifier."""

from absl.testing import absltest
import tensorflow as tf
from flax_models.bert import run_classifier
from flax_models.bert.configs import classification as default_config


class RunClassifierTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Make sure tf does not allocate gpu memory.
    tf.config.experimental.set_visible_devices([], 'GPU')

  def test_create_model(self):
    """Tests creating model."""
    run_classifier.create_model(default_config.get_config())

  def test_train_and_evaluate(self):
    # TODO(marcvanzee): Write a test for this.
    # We have to refactor the code somewhat in order to test this, since it
    # currently is too slow to run on CPU, and if we are using the py3_test
    # build rule then we cannot read the vocab and checkpoint from CNS.
    pass


if __name__ == '__main__':
  absltest.main()
