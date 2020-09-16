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

"""Experiment using 4 variants of the MNIST dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

from learning_parameter_allocation import data
from learning_parameter_allocation import models
from learning_parameter_allocation import utils

from learning_parameter_allocation.pathnet import components as pn_components
from learning_parameter_allocation.pathnet import pathnet_lib as pn
from learning_parameter_allocation.pathnet.utils import create_uniform_layer

import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'logdir', '/tmp/summary_dir/',
    'Path to the directory to save logs and summaries.')
flags.DEFINE_string(
    'method', 'gumbel_matrix',
    'Approach to use to determine which tasks gets which components, '
    'one of "shared_bottom", "no_sharing", "gumbel_matrix".')
flags.DEFINE_float(
    'budget', 1.0,
    'What fraction of the network do we want each task to use.')
flags.DEFINE_float(
    'budget_penalty', 0.0,
    'Penalty for exceeding the budget.')
flags.DEFINE_float(
    'entropy_penalty', 0.0,
    'Penalty that lowers the allocation entropy.')
flags.DEFINE_float(
    'entropy_penalty_alpha', 1.0,
    'Exponent to control how sharply the entropy penalty it increases.')


def loss_fn(labels, logits):
  return tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)


def construct_pathnet_and_run_mnist_experiment(
    task_names, task_data, num_classes_for_tasks, router_fn):
  """Runs the MNIST experiment.

  Args:
    task_names: (list of strings) names of tasks.
    task_data: (list of dicts) list of dictionaries, one per task.
      Each dictionary should map strings 'train' and 'test' into
      `tf.data.Dataset`s for training and testing, respectively.
    num_classes_for_tasks: (list of ints) number of classes for each task.
    router_fn: function that, given a single argument `num_components`, returns
      a router (see routers in `pathnet/pathnet_lib.py`) for a layer containing
      `num_components` components.

  """
  num_tasks = len(task_names)

  input_data_shape = [28, 28, 1]
  batch_size = 16

  for task_id in range(num_tasks):
    task_data[task_id] = data.batch_all(task_data[task_id], batch_size)

  # Train each task for 10 epochs
  n_epochs = 10

  training_hparams = tf.contrib.training.HParams(
      num_steps=n_epochs * 60000 // batch_size,
      batch_size=batch_size,
      learning_rate=0.005
  )

  routers = []

  def get_router(num_components):
    routers.append(router_fn(num_components))
    return routers[-1]

  # PathNet layers

  keras_layers = models.get_keras_layers_for_mnist_experiment(
      num_components=num_tasks)

  pathnet_layers = models.build_model_from_keras_layers(
      input_data_shape, num_tasks, keras_layers, get_router)

  # Task-specific linear heads

  pathnet_layers.append(
      utils.create_layer_with_task_specific_linear_heads(num_classes_for_tasks))

  # Output components to compute task loss

  auxiliary_loss_fn = utils.create_auxiliary_loss_function(
      routers=routers,
      num_total_components=12,
      num_total_steps=training_hparams.num_steps * num_tasks,
      budget=FLAGS.budget,
      budget_penalty=FLAGS.budget_penalty,
      entropy_penalty=FLAGS.entropy_penalty,
      entropy_penalty_alpha=FLAGS.entropy_penalty_alpha)

  def component_fn():
    return pn_components.ModelHeadComponent(
        loss_fn=loss_fn, auxiliary_loss_fn=auxiliary_loss_fn)

  pathnet_layers.append(create_uniform_layer(
      num_components=num_tasks,
      component_fn=component_fn,
      combiner_fn=pn.SelectCombiner,
      router_fn=lambda: None))

  utils.run_pathnet_training_and_evaluation(
      task_names=task_names,
      task_data=task_data,
      input_data_shape=input_data_shape,
      training_hparams=training_hparams,
      components_layers=pathnet_layers,
      evaluate_on=['train', 'test'],
      summary_dir=FLAGS.logdir)


def main(_):
  mnist, mnist_classes = data.get_mnist()
  mnist_2, mnist_classes_2 = data.get_mnist()
  mnist_rot, mnist_rot_classes = data.get_rotated_mnist()
  mnist_rot_2, mnist_rot_classes_2 = data.get_rotated_mnist()

  task_names = ['MNIST', 'MNIST-2', 'MNIST-Rot', 'MNIST-Rot-2']
  task_data = [mnist, mnist_2, mnist_rot, mnist_rot_2]
  num_classes_for_tasks = [
      mnist_classes, mnist_classes_2, mnist_rot_classes, mnist_rot_classes_2
  ]

  num_tasks = len(task_names)
  router_fn = utils.get_router_fn_by_name(num_tasks, FLAGS.method)

  construct_pathnet_and_run_mnist_experiment(
      task_names, task_data, num_classes_for_tasks, router_fn)


if __name__ == '__main__':
  app.run(main)
