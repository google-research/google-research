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

"""Example usage of the library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
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


def main(_):
  # Please make sure that a root dir is specified before running this script!
  root_dir = None

  model_config = mc.ModelConfig(model_type='nin', dataset='cifar10')
  load_and_run(model_config, root_dir)
  print('Loaded a NIN_CIFAR10 model.')

  # example for resnet cifar100
  model_config = mc.ModelConfig(
      model_type='resnet', dataset='cifar100')
  load_and_run(model_config, root_dir)
  print('Loaded a RESNET_CIFAR100 model.')


if __name__ == '__main__':
  tf.app.run(main)
