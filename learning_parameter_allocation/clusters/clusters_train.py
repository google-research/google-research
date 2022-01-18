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

# Lint as: python3

"""Experiment using three task clusters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from absl import app
from absl import flags

from learning_parameter_allocation import data
from learning_parameter_allocation import models
from learning_parameter_allocation import utils

from learning_parameter_allocation.pathnet import components as pn_components
from learning_parameter_allocation.pathnet import pathnet_lib as pn
from learning_parameter_allocation.pathnet.utils import create_uniform_layer

import tensorflow.compat.v1 as tf  # pylint: disable=g-explicit-tensorflow-version-import


_INPUT_SHAPE = [32, 32, 3]


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'logdir', '/tmp/summary_dir/',
    'Path to the directory to save logs and summaries.')
flags.DEFINE_string(
    'method', 'gumbel_matrix',
    'Approach to use to determine which tasks gets which components, '
    'either "shared_bottom" or "gumbel_matrix".')

# Learning rates for the model and router variables.
flags.DEFINE_float(
    'learning_rate', 3e-4,
    'Learning rate for the model weights.')
flags.DEFINE_float(
    'router_learning_rate', 6e-3,
    'Learning rate for the router weights.')

# Parameters specific to the GumbelMatrixRouter.
flags.DEFINE_float(
    'probs_init', 0.5,
    'Initial connection probabilities for the GumbelMatrixRouter.')
flags.DEFINE_float(
    'temperature_init', 50.0,
    'Initial temperature for the GumbelMatrixRouter.')
flags.DEFINE_float(
    'temperature_min', 0.5,
    'Minimum temperature for the GumbelMatrixRouter.')

# Budget penalty (auxiliary loss).
flags.DEFINE_float(
    'budget', 0.5,
    'What fraction of the network do we want each task to use.')
flags.DEFINE_float(
    'budget_penalty', 1.0,
    'Penalty for exceeding the budget.')


def loss_fn(labels, logits):
  return tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)


def construct_pathnet(
    num_steps_per_task, task_names, num_classes_for_tasks, router_fn):
  """Runs the three task clusters routing experiment.

  Args:
    num_steps_per_task: (int) number of training steps that will be performed
      per task. This function does not run any training; the number of steps
      is used to determine how some auxiliary losses are annealed over time.
    task_names: (list of strings) names of tasks.
    num_classes_for_tasks: (list of ints) number of classes for each task.
    router_fn: function that, given a single argument `num_components`, returns
      a router (see routers in `pathnet/pathnet_lib.py`) for a layer containing
      `num_components` components.

  Returns:
    A list of `pn.ComponentsLayer`s - layers that make up the PathNet model.
  """
  num_tasks = len(task_names)

  routers = []
  num_total_components = 0

  def get_router(num_components):
    routers.append(router_fn(num_components))

    nonlocal num_total_components
    num_total_components += num_components

    return routers[-1]

  # PathNet layers

  keras_layers = models.get_keras_layers_for_task_clusters_experiment()

  pathnet_layers = models.build_model_from_keras_layers(
      _INPUT_SHAPE, num_tasks, keras_layers, get_router)

  # Task-specific linear heads

  pathnet_layers.append(
      utils.create_layer_with_task_specific_linear_heads(num_classes_for_tasks))

  # Output components to compute task loss

  auxiliary_loss_fn = utils.create_auxiliary_loss_function(
      routers=routers,
      num_total_components=num_total_components,
      num_total_steps=num_steps_per_task * num_tasks,
      budget=FLAGS.budget,
      budget_penalty=FLAGS.budget_penalty)

  def component_fn():
    return pn_components.ModelHeadComponent(
        loss_fn=loss_fn, auxiliary_loss_fn=auxiliary_loss_fn)

  pathnet_layers.append(create_uniform_layer(
      num_components=num_tasks,
      component_fn=component_fn,
      combiner_fn=pn.SelectCombiner,
      router_fn=lambda: None))

  return pathnet_layers


# pylint: disable=dangerous-default-value
def run_routing_experiment(
    pathnet_layers,
    training_hparams,
    task_names,
    task_data,
    resume_checkpoint_dir=None,
    intermediate_eval_steps=[]):
  """Runs the three task clusters routing experiment.

  Args:
    pathnet_layers: (list of `pn.ComponentsLayer`s) layers that make up
      the PathNet model.
    training_hparams: (tf.contrib.training.HParams) training hyperparameters.
    task_names: (list of strings) names of tasks.
    task_data: (list of dicts) list of dictionaries, one per task.
      Each dictionary should map strings 'train', 'validation' and 'test' into
      `tf.data.Dataset`s for training, validation, and testing, respectively.
    resume_checkpoint_dir: (string or None) directory for the checkpoint
      to reload, or None if should start from scratch.
    intermediate_eval_steps: (list of ints) training step numbers at which
      accuracy should be evaluated.
  """
  for dataset in task_data:
    dataset['train_clean'] = dataset['train']
    dataset['train'] = dataset['train'].map(data.augment_with_random_crop)

  for task_id in range(len(task_data)):
    task_data[task_id] = data.batch_all(
        task_data[task_id], training_hparams.batch_size)

  utils.run_pathnet_training_and_evaluation(
      task_names=task_names,
      task_data=task_data,
      input_data_shape=_INPUT_SHAPE,
      training_hparams=training_hparams,
      components_layers=pathnet_layers,
      evaluate_on=['train_clean', 'test'],
      resume_checkpoint_dir=resume_checkpoint_dir,
      summary_dir=FLAGS.logdir,
      intermediate_eval_steps=intermediate_eval_steps,
      save_checkpoint_every_n_steps=sys.maxsize)


def get_all_tasks(num_cifar, num_mnist, num_fashion_mnist):
  """Loads a given number of tasks from each of the three task clusters.

  Args:
    num_cifar: (int) number of cifar coarse label tasks to use, must be between
      1 and 20 inclusive.
    num_mnist: (int) number of MNIST leave-one-out tasks to use,
      must be between 1 and 10 inclusive.
    num_fashion_mnist: (int) number of Fashion-MNIST leave-one-out tasks to use,
      must be between 1 and 10 inclusive.

  Returns:
    A tuple of three lists (`task_names`, `task_data`, `num_classes_for_tasks`),
    each containing one element per task. These lists respectively contain
    task name, task input data and number of classification classes.
  """
  assert 1 <= num_cifar <= 20
  assert 1 <= num_mnist <= 10
  assert 1 <= num_fashion_mnist <= 10

  task_names = []
  task_names += ['CIFAR-100-%d' % task_id for task_id in range(num_cifar)]
  task_names += ['MNIST-%d' % task_id for task_id in range(num_mnist)]
  task_names += ['Fashion-%d' % task_id for task_id in range(num_fashion_mnist)]

  raw_mnist = data.get_mnist_in_cifar_format()
  raw_fashion_mnist = data.get_fashion_mnist_in_cifar_format()

  tasks = []

  tasks += [
      data.get_cifar100(coarse_label_id) for coarse_label_id in range(num_cifar)
  ]

  tasks += [
      data.get_leave_one_out_classification(*raw_mnist, leave_out_class)
      for leave_out_class in range(num_mnist)
  ]

  tasks += [
      data.get_leave_one_out_classification(*raw_fashion_mnist, leave_out_class)
      for leave_out_class in range(num_fashion_mnist)
  ]

  # Convert a list of pairs into a pair of lists
  task_data, num_classes_for_tasks = [list(tup) for tup in zip(*tasks)]

  return task_names, task_data, num_classes_for_tasks


def main(_):
  task_names, task_data, num_classes_for_tasks = get_all_tasks(
      num_cifar=20, num_mnist=10, num_fashion_mnist=10)

  router_kwargs = {}

  if FLAGS.method == 'gumbel_matrix':
    router_kwargs = {
        'probs_init': FLAGS.probs_init,
        'temperature_init': FLAGS.temperature_init,
        'temperature_min': FLAGS.temperature_min
    }

  router_fn = utils.get_router_fn_by_name(
      len(task_names), FLAGS.method, **router_kwargs)

  num_steps = 20000
  eval_frequency = 1000
  batch_size = 32

  intermediate_eval_steps = list(range(0, num_steps + 1, eval_frequency))

  training_hparams = tf.contrib.training.HParams(
      num_steps=num_steps,
      batch_size=batch_size,
      learning_rate=FLAGS.learning_rate,
      router_learning_rate=FLAGS.router_learning_rate)

  pathnet_layers = construct_pathnet(
      training_hparams.num_steps, task_names, num_classes_for_tasks, router_fn)

  run_routing_experiment(
      pathnet_layers,
      training_hparams,
      task_names,
      task_data,
      None,
      intermediate_eval_steps)


if __name__ == '__main__':
  app.run(main)
