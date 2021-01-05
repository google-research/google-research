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

"""Utility functions for building models with learned parameter allocations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

from learning_parameter_allocation.pathnet import components as pn_components
from learning_parameter_allocation.pathnet import pathnet_lib as pn
from learning_parameter_allocation.pathnet.utils import \
    create_wrapped_routed_layer

import numpy as np
import tensorflow.compat.v1 as tf

from tqdm import tqdm


def run_pathnet_training_step(
    session, p_inputs, p_labels, p_task_id, train_step_op, train_data):
  """Trains a PathNet multitask image classification model for a single step.

  Args:
    session: (tf.Session) session to use.
    p_inputs: (tf.placeholder) placeholder for the input image.
    p_labels: (tf.placeholder) placeholder for the target labels.
    p_task_id: (tf.placeholder) placeholder for the task id.
    train_step_op: (tf.Operation) training op.
    train_data: (list of tf.data.Datasets) training data for each of the tasks.
  """
  for task_id, dataset in train_data:
    data = session.run(dataset)
    inputs, labels = data['image'], data['label']

    feed_dict = {
        p_inputs: inputs,
        p_labels: labels,
        p_task_id: np.array(task_id)
    }

    summary_ops = tf.contrib.summary.all_summary_ops()
    summary_ops = [op for op in summary_ops if not op.name.startswith('final')]

    fetch = [train_step_op, summary_ops]
    session.run(fetch, feed_dict=feed_dict)


def run_pathnet_evaluation(
    session, p_inputs, p_task_id, out_logits, task_names, eval_data):
  """Evaluates a PathNet multitask image classification model.

  Args:
    session: (tf.Session) session to use.
    p_inputs: (tf.placeholder) placeholder for the input image.
    p_task_id: (tf.placeholder) placeholder for the task id.
    out_logits: (tf.Operation) PathNet output (classification logits).
    task_names: (list of strings) names of tasks.
    eval_data: (list of tf.data.Datasets) eval data for each task, should have
      the same length as `task_names`. The evaluation will go through each
      dataset in `eval_data` until it stops returning new batches.

  Returns:
    A list of floats, containing evaluation accuracy for each of the tasks.
  """
  task_accuracies = []

  for task_id in range(len(task_names)):
    # Number of correctly classified eval samples.
    count_correct = 0

    # Total number of eval samples.
    count_all = 0

    data_iterator, num_batches = eval_data[task_id]

    for _ in range(num_batches):
      data = session.run(data_iterator)
      inputs, labels = data['image'], data['label']

      feed_dict = {p_inputs: inputs, p_task_id: np.array(task_id)}

      logits = session.run([out_logits], feed_dict=feed_dict)
      answers = np.argmax(logits, axis=-1)

      count_correct += np.sum(answers == labels)
      count_all += labels.shape[0]

    accuracy = count_correct / count_all
    print(task_names[task_id], accuracy, '(%d/%d)' % (count_correct, count_all))

    task_accuracies.append(accuracy)

  return task_accuracies


def create_accuracy_summary_ops(task_names, summary_name_prefix):
  """Creates ops for writing accuracy summaries.

  Args:
    task_names: (list of strings) names of tasks.
    summary_name_prefix: (string) prefix to prepend to summary names.

  Returns:
    A pair of (`p_task_accuracies`, `summary_op`), where `p_task_accuracies`
    is a list of tf.placeholder's for all tasks' accuracies, and `summary_op`
    is a tf.Operation that writes these summaries.
  """
  p_task_accuracies = []
  summary_ops = []

  with tf.contrib.summary.always_record_summaries():
    for task_id in range(len(task_names)):
      task_name = task_names[task_id]
      p_task_accuracy = tf.placeholder(tf.float32, shape=())

      summary_ops.append(tf.contrib.summary.scalar(
          '%s/%s' % (summary_name_prefix, task_name), p_task_accuracy))

      p_task_accuracies.append(p_task_accuracy)

    average_task_accuracy = tf.reduce_mean(p_task_accuracies)

    summary_ops.append(tf.contrib.summary.scalar(
        '%s/Average' % summary_name_prefix, average_task_accuracy))

  return p_task_accuracies, tf.group(*summary_ops)


def run_accuracy_summary_ops(
    session,
    p_task_accuracies,
    task_accuracies,
    accuracy_summary_op):
  """Runs ops that write accuracy summaries.

  Args:
    session: (tf.Session) session to use.
    p_task_accuracies: (list of tf.placeholder's) placeholders for all tasks'
      accuracies, as returned from `create_accuracy_summary_ops`.
    task_accuracies: (list of floats) all tasks' accuracies, they will be fed
      into `p_task_accuracies`.
    accuracy_summary_op: (tf.Operation) an op that writes accuracy summaries,
      as returned from `create_accuracy_summary_ops`.
  """
  feed_dict = {
      p_task_accuracy: task_accuracy
      for (p_task_accuracy, task_accuracy) in zip(
          p_task_accuracies, task_accuracies)
  }

  session.run(accuracy_summary_op, feed_dict=feed_dict)


def build_pathnet_graph(p_inputs, p_labels, p_task_id, pathnet, training):
  """Builds a PathNet graph, returns input placeholders and output tensors.

  Args:
    p_inputs: (tf.placeholder) placeholder for the input image.
    p_labels: (tf.placeholder) placeholder for the target labels.
    p_task_id: (tf.placeholder) placeholder for the task id.
    pathnet: (pn.PathNet) PathNet model to use.
    training: (bool) whether the graph is being created for training.

  Returns:
    A pair of (`train_step_op`, `out_logits`), where `train_step_op` is
    the training op, and `out_logits` is the final network output
    (classification logits).
  """
  end_state = pathnet({
      'in_tensor': p_inputs,
      'labels': p_labels,
      'task_id': p_task_id,
      'training': training
  })

  # The train op is the same for all tasks. Note that the last layer in
  # PathNet models contains heads that compute the task losses, with one head
  # per task. The head is chosen depending on `task_id`, so the loss for
  # the current task is always under `task_loss` key in `end_state`.
  train_step_op = end_state['train_op']
  out_logits = end_state['in_tensor']

  return train_step_op, out_logits


def count_batches(session, dataset):
  """Count the number of batches in a dataset."""
  num_batches = 0

  try:
    while True:
      session.run(dataset)
      num_batches += 1
  except tf.errors.OutOfRangeError:
    pass

  return num_batches


# pylint: disable=dangerous-default-value
def run_pathnet_training_and_evaluation(
    task_names,
    task_data,
    input_data_shape,
    training_hparams,
    components_layers,
    evaluate_on,
    summary_dir,
    resume_checkpoint_dir=None,
    save_checkpoint_every_n_steps=250,
    intermediate_eval_steps=[]):
  """Trains and evaluates a PathNet multitask image classification model.

  Args:
    task_names: (list of strings) names of tasks.
    task_data: (list of dicts) list of dictionaries, one per task.
      Each dictionary should map strings into `tf.data.Dataset`s.
      The `i`-th dictionary should contain all dataset splits (such as 'train',
      'test', 'eval', etc) for the `i`-th task. The splits can be arbitrary,
      but to run the training, every dataset should contain a 'train' split.
    input_data_shape: (sequence of ints) expected shape of input images
      (excluding batch dimension). For example, for the MNIST dataset
      `input_data_shape=[28, 28, 1]`.
    training_hparams: (tf.contrib.training.HParams) training hyperparameters.
    components_layers: (list of `pn.ComponentsLayer`s) layers that make up
      the PathNet model.
    evaluate_on: (list of strings) dataset splits on which the trained PathNet
      should be evaluated. These keys should be present in every dictionary
      in `task_data`.
    summary_dir: (string) directory for the summary writer.
    resume_checkpoint_dir: (string or None) directory for the checkpoint
      to reload, or None if should start from scratch.
    save_checkpoint_every_n_steps: (int) frequency for saving model checkpoints.
    intermediate_eval_steps: (list of ints) training step numbers at which
      accuracy should be evaluated. An evaluation after the last step is
      always performed.
  """
  session = tf.Session(graph=tf.get_default_graph())

  summary_writer = tf.contrib.summary.create_file_writer(summary_dir)
  summary_writer.set_as_default()

  num_tasks = len(task_names)

  # Every `num_tasks` subsequent steps contain exactly one step for each task,
  # and always in the order as they appear in `task_data`. Setting the logging
  # frequency to `num_tasks + 1` (or any other number coprime with `num_tasks`)
  # guarantees that each task will get to record summaries with the same
  # frequency.
  with tf.contrib.summary.record_summaries_every_n_global_steps(num_tasks + 1):
    pathnet = pn.PathNet(components_layers, training_hparams)
    num_steps = training_hparams.num_steps

    eval_steps = intermediate_eval_steps + [num_steps]

    # Loop each training dataset forever.
    train_data = [
        dataset['train'].repeat().make_one_shot_iterator().get_next()
        for dataset in task_data
    ]

    # Attach the task id to each dataset.
    train_data = list(enumerate(train_data))

    p_inputs = tf.placeholder(tf.float32, shape=[None] + input_data_shape)
    p_labels = tf.placeholder(tf.int64, shape=[None])
    p_task_id = tf.placeholder(tf.int32, shape=[], name='task_id')

    train_step_op, _ = build_pathnet_graph(
        p_inputs, p_labels, p_task_id, pathnet, training=True)

  with tf.variable_scope(tf.get_variable_scope(), reuse=True):
    _, out_logits_eval = build_pathnet_graph(
        p_inputs, p_labels, p_task_id, pathnet, training=False)

  session.run(tf.global_variables_initializer())
  tf.contrib.summary.initialize(session=session)

  saver = tf.train.Saver(tf.global_variables())

  start_step = 0

  p_task_accuracies = {}
  accuracy_summary_op = {}

  for data_split in evaluate_on:
    p_task_accuracies[data_split], accuracy_summary_op[data_split] = \
      create_accuracy_summary_ops(
          task_names, summary_name_prefix='final_eval_%s' % data_split)

  if resume_checkpoint_dir is not None:
    print('Resuming from checkpoint: %s' % resume_checkpoint_dir)

    last_global_step = int(resume_checkpoint_dir.split('-')[-1])

    assert last_global_step % num_tasks == 0
    start_step = last_global_step // num_tasks

    saver.restore(session, resume_checkpoint_dir)

  for dataset in task_data:
    for data_split in evaluate_on:
      num_batches = count_batches(
          session, dataset[data_split].make_one_shot_iterator().get_next())

      dataset[data_split] = dataset[data_split].repeat()
      dataset[data_split] = (
          dataset[data_split].make_one_shot_iterator().get_next())

      dataset[data_split] = (dataset[data_split], num_batches)

  for step in tqdm(range(start_step, num_steps)):
    random.shuffle(train_data)

    run_pathnet_training_step(
        session, p_inputs, p_labels, p_task_id, train_step_op, train_data)

    if step + 1 in eval_steps:
      for data_split in evaluate_on:
        eval_data = [dataset[data_split] for dataset in task_data]

        print('Running evaluation on: %s' % data_split)

        task_accuracies = run_pathnet_evaluation(
            session=session,
            p_inputs=p_inputs,
            p_task_id=p_task_id,
            out_logits=out_logits_eval,
            task_names=task_names,
            eval_data=eval_data)

        run_accuracy_summary_ops(
            session,
            p_task_accuracies[data_split],
            task_accuracies,
            accuracy_summary_op[data_split])

    if (step + 1) % save_checkpoint_every_n_steps == 0:
      path = summary_dir + '/chkpt'
      saver.save(
          session, path, global_step=tf.train.get_or_create_global_step())


def compute_output_shape_and_create_routed_layer(
    keras_components, in_shape, router_fn, router_out):
  """A utility to wrap a list of keras layers into a routed layer.

  Args:
    keras_components: (list of keras.layers.Layer) components to wrap.
    in_shape: (sequence of ints) expected input shape to this layer (excluding
      batch dimension).
    router_fn: function that, given a single argument `num_components`, returns
      a router (see routers in `pathnet/pathnet_lib.py`) for a layer containing
      `num_components` components.
    router_out: a router (see routers in `pathnet/pathnet_lib.py`) to use to
      route into the next layer.

  Raises:
    Exception: if components in `keras_components` would produce different
    output shapes for the given input shape `in_shape`.

  Returns:
    A pair of (`layers`, `out_shape`), where `layers` is a list of PathNet
    layers (returned from `create_wrapped_routed_layer`) and `out_shape` is
    the output shape (excluding batch dimension), computed based on
    `keras_components` and `in_shape`.
  """
  num_components = len(keras_components)
  assert num_components >= 1

  # Compute output shapes by using `compute_output_shape`. Note that we
  # temporarily add the batch dimension of 1, and then remove it.
  out_shapes = [
      component.compute_output_shape([1] + in_shape)[1:]
      for component in keras_components
  ]

  out_shape = out_shapes[0]

  # Check if all outputs are the same.
  for shape in out_shapes:
    if shape != out_shape:
      raise Exception(
          'Output shapes for keras components do not match:'
          ' got %s and %s' % (str(out_shape), str(shape)))

  components = [pn_components.KerasComponent(
      'KerasComponent', component, out_shape) for component in keras_components]

  return create_wrapped_routed_layer(
      components=components,
      router=router_fn(num_components),
      router_out=router_out,
      combiner=pn.WeightedSumCombiner(),
      in_shape=in_shape,
      out_shape=out_shape,
      sparse=False
  ), out_shape


def get_router_fn_by_name(num_tasks, routing_method_name, **kwargs):
  """Returns a `router_fn` based on its name.

  Args:
    num_tasks: (int) number of tasks.
    routing_method_name: (string) name of the allocation pattern, one of:
      'no_sharing' - the `i`-th task is routed only to the `i`-th component.
        This method can be only used for layer with `num_tasks` components.
      'shared_bottom' - every task is routed to all components.
      'gumbel_matrix' - uses a `pathnet_lib.GumbelMatrixRouter`.
    **kwargs: additional keyword arguments that will be passed as constructor
      arguments when creating the router.

  Raises:
    Exception: if `routing_method_name` is outside of the allowed set.

  Returns:
    A `router_fn` function for the given name. This function, when given
    a single argument `num_components`, returns a router (see routers in
    `pathnet/pathnet_lib.py`) for a layer containing `num_components`
    components. The returned router assumes the number of tasks is `num_tasks`.
  """
  def no_sharing_router_fn(num_components):
    if num_components != num_tasks:
      raise Exception(
          'Got %d components, and %d tasks. The `no_sharing` routing method '
          'was used, which expects these numbers to match.' % (
              num_components, num_tasks))

    return pn.IndependentTaskBasedRouter(
        num_tasks=num_tasks,
        record_summaries=True,
        **kwargs)

  def shared_bottom_router_fn(num_components):
    return pn.FixedRouter(
        tf.ones((num_tasks, num_components)),
        record_summaries=True,
        **kwargs)

  def gumbel_router_fn(num_components):
    return pn.GumbelMatrixRouter(
        num_tasks=num_tasks,
        num_out_paths=num_components,
        record_summaries=True,
        **kwargs)

  if routing_method_name == 'no_sharing':
    return no_sharing_router_fn
  elif routing_method_name == 'shared_bottom':
    return shared_bottom_router_fn
  elif routing_method_name == 'gumbel_matrix':
    return gumbel_router_fn
  else:
    raise Exception('Unrecognized method: %s' % routing_method_name)


def create_layer_with_task_specific_linear_heads(num_classes_for_tasks):
  """Returns a `pathnet_lib.ComponentsLayer` with linear task specific layers.

  This is a small helper function to create a layer of fully connected
  components for multiple classification tasks (with possibly different
  numbers of classses). The constructed layer assumes that the next layer
  contains task heads to compute the task loss, and uses a
  `pathnet_lib.TaskBasedRouter` to route into the next layer.

  Args:
    num_classes_for_tasks: (list of ints) number of classes for each task.

  Returns:
    A `pathnet_lib.ComponentsLayer` containing one FC layer per task.
  """
  num_tasks = len(num_classes_for_tasks)

  components = []
  for num_classes in num_classes_for_tasks:
    components.append(pn.RoutedComponent(
        pn_components.FCLComponent(numbers_of_units=[num_classes]),
        pn.IndependentTaskBasedRouter(num_tasks=num_tasks)))

  return pn.ComponentsLayer(components=components, combiner=pn.SelectCombiner())


def create_auxiliary_loss_function(routers, **kwargs):
  """Returns an auxiliary loss function that can be used with PathNet.

  This function can take any subset of the following arguments, which correspond
  to adding various auxiliary losses. All kinds of losses are listed below,
  along with parameters that have to be supplied to enable them.

    Disconnect penalty:
    - `disconnect_penalty`: a coefficient to multiply by the probability, that
      some task selects zero components in some layer.

    Connection penalty:
    - `connection_penalty`: a penalty per every active component.

    Budget exceeded penalty:
    - `budget`: a value in the [0.0, 1.0] range, denoting the target fraction
      of the network that we want each task to use. The network will
      be penalized only if `budget` is exceeded.
    - `num_total_components`: total number of components in the network (only in
      routed layers).
    - `budget_penalty`: if `budget` is exceeded, the value by which
      it is exceeded is multiplied by this coefficient.

    Entropy penalty:
    - `num_total_steps`: number of steps the traning will take (i.e. the maximum
      value that will ever be attained by `global_step`).
    - `entropy_penalty`: a target value for entropy penalty applied at the end
      of training.
    - `entropy_penalty_alpha`: a hyperparameter to control the speed at which
      the entropy penalty increases from 0 to `entropy_penalty`. The actual
      entropy penalty at time `t` will be:
        `entropy_penalty * (t / num_total_steps)^entropy_penalty_alpha`

  Args:
    routers: routers in all routed layers which should be taken into account
      for the auxiliary losses.
    **kwargs: parameters for selected auxiliary losses.

  Returns:
    A function that computes the total auxiliary loss from the current state.
    The returned function can be directly passed into `ModelHeadComponent`
    under the `auxiliary_loss_fn` argument.
  """
  def auxiliary_loss_fn(state):
    """Computes an auxiliary loss.

    Args:
      state: (dict) PathNet state as a dict containing a 'task_id' entry with
        scalar task id.

    Raises:
      Exception: if some parameter names are not recognized.

    Returns:
      The total auxiliary loss for the given task and at the given timestep.
    """
    task_id = state['task_id']
    loss = tf.constant(0.0)

    for key, value in kwargs.items():
      if key == 'disconnect_penalty':
        disconnect_penalty = value
        if disconnect_penalty > 0.0:
          loss += disconnect_penalty * tf.reduce_sum([
              router.get_probability_of_having_no_connections(task_id)
              for router in routers
          ])
      elif key == 'connection_penalty':
        connection_penalty = value
        if connection_penalty > 0.0:
          loss += connection_penalty * tf.reduce_sum([
              router.get_expected_number_of_connections_for_task(task_id)
              for router in routers
          ])
      elif key == 'budget_penalty':
        budget_penalty = value
        if budget_penalty > 0.0:
          budget = kwargs['budget']
          num_total_components = kwargs['num_total_components']

          expected_number_of_connections = tf.reduce_sum([
              router.get_expected_number_of_connections_for_task(task_id)
              for router in routers
          ])

          expected_fraction_of_connections = (
              expected_number_of_connections / num_total_components)

          loss += budget_penalty * tf.math.maximum(
              tf.constant(0.0), expected_fraction_of_connections - budget)
      elif key in ['budget', 'num_total_components']:
        pass
      elif key == 'entropy_penalty':
        entropy_penalty = value
        if entropy_penalty > 0.0:
          entropy_penalty_alpha = kwargs['entropy_penalty_alpha']
          num_total_steps = kwargs['num_total_steps']

          global_step = tf.train.get_or_create_global_step()

          current_entropy_penalty = entropy_penalty * tf.math.pow(
              global_step / num_total_steps, entropy_penalty_alpha)
          current_entropy_penalty = tf.dtypes.cast(
              current_entropy_penalty, dtype=tf.float32)

          loss += current_entropy_penalty * tf.reduce_sum([
              router.get_entropy(task_id) for router in routers
          ])
      elif key in ['entropy_penalty_alpha', 'num_total_steps']:
        pass
      elif key == 'l2_penalty':
        l2_penalty = value
        if l2_penalty > 0.0:
          # Penalize all trainable variables apart from biases and
          # allocation logits.
          l2_penalty_vars = [
              var for var in tf.trainable_variables()
              if 'bias' not in var.name and 'router_dist' not in var.name
          ]

          loss += tf.add_n([
              tf.nn.l2_loss(var) for var in l2_penalty_vars
          ]) * l2_penalty
      else:
        raise Exception('Unrecognized parameter for auxiliary losses: %s' % key)

    return loss

  return auxiliary_loss_fn
