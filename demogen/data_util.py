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

"""Getting a input function that will give input and label tensors."""

from tensor2tensor import problems
import tensorflow as tf


def get_input(
    batch_size=50,
    augmented=False,
    data='cifar10',
    mode=tf.estimator.ModeKeys.TRAIN,
    repeat_num=None,
    class_num=100,
    data_format='HWC'):
  """Returns a input function for the estimator framework.

  Args:
    batch_size: batch size for training or testing
    augmented:  whether data augmentation is used
    data:       a string that specifies the dataset, must be cifar10
                  or cifar100
    mode:       indicates whether the input is for training or testing,
                  needs to be a member of tf.estimator.ModeKeys
    repeat_num: how many times the dataset is repeated
    class_num:  number of classes
    data_format: order of the data's axis

  Returns:
    an input function
  """
  assert data == 'cifar10' or data == 'cifar100'
  data = 'image_' + data

  if mode != tf.estimator.ModeKeys.TRAIN:
    repeat_num = 1

  problem_name = data
  if data == 'image_cifar10' and not augmented:
    problem_name = 'image_cifar10_plain'

  def standardization(example):
    """Perform per image standardization on a single image."""
    image = example['inputs']
    image.set_shape([32, 32, 3])
    example['inputs'] = tf.image.per_image_standardization(image)
    return example

  def input_data():
    """Input function to be returned."""
    prob = problems.problem(problem_name)
    if data == 'image_cifar100':
      dataset = prob.dataset(mode, preprocess=augmented)
      if not augmented: dataset = dataset.map(map_func=standardization)
    else:
      dataset = prob.dataset(mode)

    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(repeat_num)
    dataset = dataset.make_one_shot_iterator().get_next()
    if data_format == 'CHW':
      dataset['inputs'] = tf.transpose(dataset['inputs'], (0, 3, 1, 2))
    return dataset['inputs'], tf.squeeze(tf.one_hot(dataset['targets'],
                                                    class_num))
  return input_data
