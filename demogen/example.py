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

"""Example usage of the library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
import demogen.data_util as data_util
import demogen.model_config as mc


def load_and_run(model_config, root_dir):
  """An example usage of loading and running a model from the dataset.

  Args:
    model_config: A ModelConfig object that contains relevant hyperparameters of
      a model.
    root_dir: Directory containing the dataset
  """
  model_path = model_config.get_checkpoint_path(root_dir)
  model_fn = model_config.get_model_fn()
  with tf.Session() as sess:
    input_fn = data_util.get_input(
        data=model_config.dataset, data_format=model_config.data_format)
    image, _ = input_fn()
    logits = model_fn(image, is_training=False)
    sess.run(tf.global_variables_initializer())
    model_config.load_parameters(model_path, sess)
    sess.run(logits)


def evaluate_model(model_config, root_dir):
  """Example for evalutate a model."""
  model_path = model_config.get_checkpoint_path(root_dir)
  model_fn = model_config.get_model_fn()
  with tf.Session() as sess:
    input_fn = data_util.get_input(
        batch_size=500,
        data=model_config.dataset,
        data_format=model_config.data_format,
        mode=tf.estimator.ModeKeys.EVAL)
    images, labels = input_fn()
    logits = model_fn(images, is_training=False)
    predictions = tf.argmax(logits, axis=-1)
    true_labels = tf.argmax(labels, axis=-1)
    sess.run(tf.global_variables_initializer())
    model_config.load_parameters(model_path, sess)
    correct_prediction = 0
    for _ in range(20):
      batch_prediction, batch_label = sess.run([predictions, true_labels])
      correct_prediction += np.sum(
          np.int32(np.equal(batch_prediction, batch_label)))
  return correct_prediction/10000.


def main(_):
  # Please make sure that a root dir is specified before running this script!
  root_dir = None

  model_config = mc.ModelConfig(
      model_type='nin', dataset='cifar10', root_dir=root_dir)
  load_and_run(model_config, root_dir)
  print('Loaded a NIN_CIFAR10 model.')
  print('Evaluating the NIN_CIFAR10 model.')
  eval_result = evaluate_model(model_config, root_dir)
  print('Test Accuracy: {}'.format(eval_result))
  print('Stored Test Accuracy: {}'.format(model_config.test_stats()))
  print('Stored Train Accuracy: {}'.format(model_config.training_stats()))
  print('==========================================')
  # example for resnet cifar100
  model_config = mc.ModelConfig(model_type='resnet', dataset='cifar100')
  load_and_run(model_config, root_dir)
  print('Loaded a RESNET_CIFAR100 model.')


if __name__ == '__main__':
  tf.app.run(main)
