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

# Lint as: python3

"""Experiment using 20 alphabets from the Omniglot dataset."""

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

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import


_OMNIGLOT_INPUT_SHAPE = [105, 105, 1]


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'logdir', '/tmp/summary_dir/',
    'Path to the directory to save logs and summaries.')
flags.DEFINE_string(
    'method', 'gumbel_matrix',
    'Approach to use to determine which tasks gets which components, '
    'one of "shared_bottom", "no_sharing", "gumbel_matrix".')

# Learning rates for the model and router variables.
flags.DEFINE_float(
    'learning_rate', 1e-4,
    'Learning rate for the model weights.')
flags.DEFINE_float(
    'router_learning_rate', 2e-3,
    'Learning rate for the router weights.')

# Parameters specific to the GumbelMatrixRouter.
flags.DEFINE_float(
    'probs_init', 0.97,
    'Initial connection probabilities for the GumbelMatrixRouter.')
flags.DEFINE_float(
    'temperature_init', 50.0,
    'Initial temperature for the GumbelMatrixRouter.')
flags.DEFINE_float(
    'temperature_min', 0.5,
    'Minimum temperature for the GumbelMatrixRouter.')

# L2 penalty.
flags.DEFINE_float(
    'l2_penalty', 3e-4,
    'L2 penalty on the model weights.')

# Budget penalty (auxiliary loss).
flags.DEFINE_float(
    'budget', 1.0,
    'What fraction of the network do we want each task to use.')
flags.DEFINE_float(
    'budget_penalty', 0.0,
    'Penalty for exceeding the budget.')

# Entropy penalty (auxiliary loss).
flags.DEFINE_float(
    'entropy_penalty', 0.0,
    'Penalty that lowers the allocation entropy.')
flags.DEFINE_float(
    'entropy_penalty_alpha', 1.0,
    'Exponent to control how sharply the entropy penalty it increases.')


def loss_fn(labels, logits):
  return tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)


def construct_pathnet(
    num_steps_per_task, task_names, num_classes_for_tasks, router_fn):
  """Runs the Omniglot experiment.

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
    nonlocal num_total_components

    routers.append(router_fn(num_components))
    num_total_components += num_components

    return routers[-1]

  # PathNet layers

  keras_layers = models.get_keras_layers_for_omniglot_experiment()

  pathnet_layers = models.build_model_from_keras_layers(
      _OMNIGLOT_INPUT_SHAPE, num_tasks, keras_layers, get_router)

  # Task-specific linear heads

  pathnet_layers.append(
      utils.create_layer_with_task_specific_linear_heads(num_classes_for_tasks))

  # Output components to compute task loss

  auxiliary_loss_fn = utils.create_auxiliary_loss_function(
      routers=routers,
      num_total_components=num_total_components,
      num_total_steps=num_steps_per_task * num_tasks,
      l2_penalty=FLAGS.l2_penalty,
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

  return pathnet_layers


def run_omniglot_experiment(
    pathnet_layers,
    training_hparams,
    task_names,
    task_data,
    resume_checkpoint_dir=None):
  """Runs the Omniglot experiment.

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
  """
  for task_id in range(len(task_data)):
    task_data[task_id] = data.batch_all(
        task_data[task_id], training_hparams.batch_size)

  utils.run_pathnet_training_and_evaluation(
      task_names=task_names,
      task_data=task_data,
      input_data_shape=_OMNIGLOT_INPUT_SHAPE,
      training_hparams=training_hparams,
      components_layers=pathnet_layers,
      evaluate_on=['train', 'validation', 'test'],
      resume_checkpoint_dir=resume_checkpoint_dir,
      summary_dir=FLAGS.logdir)


def main(_):
  # If available, resume from the latest checkpoint.
  checkpoint_dir = tf.train.latest_checkpoint(FLAGS.logdir)

  num_alphabets = 20
  task_names = ['Omniglot-%d' % task_id for task_id in range(num_alphabets)]

  task_data, num_classes_for_tasks = data.get_data_for_multitask_omniglot_setup(
      num_alphabets)

  router_kwargs = {}

  if FLAGS.method == 'gumbel_matrix':
    router_kwargs = {
        'probs_init': FLAGS.probs_init,
        'temperature_init': FLAGS.temperature_init,
        'temperature_min': FLAGS.temperature_min
    }

  router_fn = utils.get_router_fn_by_name(
      num_alphabets, FLAGS.method, **router_kwargs)

  training_hparams = tf.contrib.training.HParams(
      num_steps=8_000,
      batch_size=16,
      learning_rate=FLAGS.learning_rate,
      router_learning_rate=FLAGS.router_learning_rate
  )

  pathnet_layers = construct_pathnet(
      training_hparams.num_steps, task_names, num_classes_for_tasks, router_fn)

  run_omniglot_experiment(
      pathnet_layers, training_hparams, task_names, task_data, checkpoint_dir)


if __name__ == '__main__':
  app.run(main)
