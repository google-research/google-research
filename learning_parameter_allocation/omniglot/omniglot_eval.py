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

"""Evaluation job for the Omniglot experiments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from absl import app
from absl import flags

from learning_parameter_allocation import data
from learning_parameter_allocation import models
from learning_parameter_allocation import utils

from learning_parameter_allocation.pathnet import components as pn_components
from learning_parameter_allocation.pathnet import pathnet_lib as pn
from learning_parameter_allocation.pathnet.utils import create_uniform_layer

import tensorflow.compat.v1 as tf


_OMNIGLOT_INPUT_SHAPE = [105, 105, 1]


# Delay in seconds to wait before rechecking if there are new checkpoints.
_CHECK_FOR_CHECKPOINTS_FREQUENCY = 15

# If there are no checkpoints for this number of seconds give up and finish.
_MAX_WAIT_FOR_NEW_CHECKPOINTS = 3 * 60 * 60


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'logdir', '/tmp/summary_dir/',
    'Path to the directory to save logs and summaries.')
flags.DEFINE_string(
    'method', 'gumbel_matrix',
    'Approach to use to determine which tasks gets which components, '
    'one of "shared_bottom", "no_sharing", "gumbel_matrix".')


def loss_fn(labels, logits):
  return tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)


def build_pathnet_eval_graph(
    task_names, batch_size, num_classes_for_tasks, router_fn):
  """Constructs the PathNet eval graph.

  Args:
    task_names: (list of strings) names of tasks.
    batch_size: (int) batch size to use.
    num_classes_for_tasks: (list of ints) number of classes for each task.
    router_fn: function that, given a single argument `num_components`, returns
      a router (see routers in `pathnet/pathnet_lib.py`) for a layer containing
      `num_components` components.

  Returns:
    A tuple of (`p_inputs`, `p_task_id`, `out_logits`). `p_inputs` and
    `p_task_id` are placeholders for input image and scalar task id,
    respectively. `out_logits` are the final network output (classification
    logits).
  """
  num_tasks = len(task_names)

  # PathNet layers

  keras_layers = models.get_keras_layers_for_omniglot_experiment()

  pathnet_layers = models.build_model_from_keras_layers(
      _OMNIGLOT_INPUT_SHAPE, num_tasks, keras_layers, router_fn)

  # Task-specific linear heads

  pathnet_layers.append(
      utils.create_layer_with_task_specific_linear_heads(num_classes_for_tasks))

  # Output components

  pathnet_layers.append(create_uniform_layer(
      num_components=num_tasks,
      component_fn=lambda: pn_components.ModelHeadComponent(loss_fn=loss_fn),
      combiner_fn=pn.SelectCombiner,
      router_fn=lambda: None))

  pathnet = pn.PathNet(
      pathnet_layers, tf.contrib.training.HParams(batch_size=batch_size))

  p_inputs, _, p_task_id, _, out_logits = utils.build_pathnet_graph(
      pathnet, _OMNIGLOT_INPUT_SHAPE, training=False)

  return p_inputs, p_task_id, out_logits


def main(_):
  num_alphabets = 20
  task_names = ['Omniglot-%d' % task_id for task_id in range(num_alphabets)]

  task_data, num_classes = data.get_data_for_multitask_omniglot_setup(
      num_alphabets)

  batch_size = 16
  for task_id in range(num_alphabets):
    task_data[task_id] = data.batch_all(task_data[task_id], batch_size)

  router_fn = utils.get_router_fn_by_name(num_alphabets, FLAGS.method)

  session = tf.Session(graph=tf.get_default_graph())

  tf.train.get_or_create_global_step()

  summary_writer = tf.contrib.summary.create_file_writer(FLAGS.logdir)
  summary_writer.set_as_default()

  tf.contrib.summary.initialize(session=session)

  p_inputs, p_task_id, out_logits = build_pathnet_eval_graph(
      task_names, batch_size, num_classes, router_fn)

  evaluate_on = ['train', 'validation', 'test']

  p_task_accuracies = {}
  accuracy_summary_op = {}

  for data_split in evaluate_on:
    (p_task_accuracies[data_split], accuracy_summary_op[data_split]) =\
        utils.create_accuracy_summary_ops(
            task_names, summary_name_prefix='eval_%s' % data_split)

  # This `Saver` is not used to save variables, only to restore them from
  # the checkpoints.
  saver = tf.train.Saver(tf.global_variables())

  previous_checkpoint_path = ''
  time_waited_for_checkpoints = 0

  while time_waited_for_checkpoints < _MAX_WAIT_FOR_NEW_CHECKPOINTS:
    latest_checkpoint_path = tf.train.latest_checkpoint(FLAGS.logdir)

    if latest_checkpoint_path in [None, previous_checkpoint_path]:
      print('Found no new checkpoints')

      time_waited_for_checkpoints += _CHECK_FOR_CHECKPOINTS_FREQUENCY
      time.sleep(_CHECK_FOR_CHECKPOINTS_FREQUENCY)

      continue
    else:
      time_waited_for_checkpoints = 0

    print('Reloading checkpoint: %s' % latest_checkpoint_path)
    previous_checkpoint_path = latest_checkpoint_path

    saver.restore(session, latest_checkpoint_path)

    for data_split in evaluate_on:
      eval_data = [
          dataset[data_split].make_one_shot_iterator().get_next()
          for dataset in task_data
      ]

      print('Evaluating on: %s' % data_split)

      task_accuracies = utils.run_pathnet_evaluation(
          session, p_inputs, p_task_id, out_logits, task_names, eval_data)

      utils.run_accuracy_summary_ops(
          session,
          p_task_accuracies[data_split],
          task_accuracies,
          accuracy_summary_op[data_split])


if __name__ == '__main__':
  app.run(main)
