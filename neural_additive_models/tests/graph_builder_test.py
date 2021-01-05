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

# Lint as: python3
"""Tests functionality of building tensorflow graph."""

from absl.testing import absltest
from absl.testing import parameterized

import tensorflow.compat.v1 as tf
from neural_additive_models import data_utils
from neural_additive_models import graph_builder


class GraphBuilderTest(parameterized.TestCase):
  """Tests whether neural net models can be run without error."""

  @parameterized.named_parameters(('classification', 'BreastCancer', False),
                                  ('regression', 'Housing', True))
  def test_build_graph(self, dataset_name, regression):
    """Test whether build_graph works as expected."""
    data_x, data_y, _ = data_utils.load_dataset(dataset_name)
    data_gen = data_utils.split_training_dataset(
        data_x, data_y, n_splits=5, stratified=not regression)
    (x_train, y_train), (x_validation, y_validation) = next(data_gen)
    sess = tf.InteractiveSession()
    graph_tensors_and_ops, metric_scores = graph_builder.build_graph(
        x_train=x_train,
        y_train=y_train,
        x_test=x_validation,
        y_test=y_validation,
        activation='exu',
        learning_rate=1e-3,
        batch_size=256,
        shallow=True,
        regression=regression,
        output_regularization=0.1,
        dropout=0.1,
        decay_rate=0.999,
        name_scope='model',
        l2_regularization=0.1)
    # Run initializer ops
    sess.run(tf.global_variables_initializer())
    sess.run([
        graph_tensors_and_ops['iterator_initializer'],
        graph_tensors_and_ops['running_vars_initializer']
    ])
    for _ in range(2):
      sess.run(graph_tensors_and_ops['train_op'])
    self.assertIsInstance(metric_scores['train'](sess), float)
    sess.close()


if __name__ == '__main__':
  absltest.main()
