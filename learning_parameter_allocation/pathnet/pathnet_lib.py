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

"""A framework for building routed models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class PathNet(object):
  """Builds a Path Net TF graph."""

  def __init__(self, componenets_layers, hparams):
    self.componenets_layers = componenets_layers
    self.hparams = hparams

  def __call__(self, state):
    hparams = self.hparams
    c_layers = self.componenets_layers
    default_params = {
        'learning_rate': 0.001,
        'gradient_clip_value': 0.0
    }

    if 'task_id' not in state:
      state['task_id'] = tf.placeholder(tf.int32, shape=[], name='task_id')

    task_id = state['task_id']

    assert len(c_layers[0].components) == len(c_layers[-1].components)
    task_id_one_hot = tf.cast(tf.one_hot(task_id, len(c_layers[0].components)),
                              dtype=tf.float32)

    # Total allocation entropy.
    entropy = tf.zeros([])

    layer_id = 0
    out_paths_weights = task_id_one_hot

    # Layers execution.
    while layer_id < len(c_layers):
      if layer_id == len(c_layers)-1:  # Last iteration
        out_paths_weights = task_id_one_hot
        num_out_paths = 1
      else:
        num_out_paths = len(c_layers[layer_id+1].components)

      components_layer = c_layers[layer_id]
      (outputs, out_paths_weights, curr_entropy) = components_layer(
          layer_id=layer_id,
          state=state,
          active_paths_weights=out_paths_weights,
          num_out_paths=num_out_paths)
      state.update(outputs)

      entropy += curr_entropy
      layer_id += 1

    # Param util.
    def get_optimizer_params(hparams, defaults, prefix=''):
      rtn = {}
      for key in defaults:
        rtn[key] = getattr(hparams, prefix+key, defaults[key])
      return rtn

    # Task optimizer.
    task_loss = state['task_loss']
    task_optimizer_params = get_optimizer_params(hparams, default_params)
    with tf.variable_scope('global/task_optimizer'):
      all_vars = tf.trainable_variables()

      router_vars = [var for var in all_vars if 'router_dist' in var.name]
      model_vars = [var for var in all_vars if 'router_dist' not in var.name]

      all_vars = model_vars + router_vars

      model_optimizer_params = task_optimizer_params
      model_opt = tf.compat.v1.train.AdamOptimizer(
          model_optimizer_params['learning_rate'])

      tf.logging.info(
          'building model optimizer with params: %s', model_optimizer_params)

      # To optimize the router variables, we use the same hyperparams as
      # for the model variables, unless a corresponding hyperparam with
      # a 'router_' prefix was supplied.
      router_optimizer_params = get_optimizer_params(
          hparams, model_optimizer_params, 'router_')
      router_opt = tf.compat.v1.train.AdamOptimizer(
          router_optimizer_params['learning_rate'])

      tf.logging.info(
          'building router optimizer with params: %s', router_optimizer_params)

      all_gradients = list(clip_gradients(
          zip(tf.gradients(task_loss, all_vars), all_vars),
          task_optimizer_params['gradient_clip_value']))

      model_gradients = all_gradients[:len(model_vars)]
      router_gradients = all_gradients[len(model_vars):]

      model_train_op = model_opt.apply_gradients(
          model_gradients, global_step=tf.train.get_or_create_global_step())

      if router_vars:
        router_train_op = router_opt.apply_gradients(router_gradients)
        train_op = tf.group(model_train_op, router_train_op)
      else:
        train_op = model_train_op

    state['train_op'] = train_op

    # Summaries.
    tf.contrib.summary.scalar('global/entropy', entropy)
    tf.contrib.summary.scalar('global/task_loss', task_loss)

    return state


def clip_gradients(gradients, gradient_clip_value):
  # Clip the gradients.
  if gradient_clip_value > 0.0:
    g, v = zip(*gradients)
    g, norm = tf.clip_by_global_norm(g, gradient_clip_value)
    tf.contrib.summary.scalar('global_norm', norm)
    gradients = zip(g, v)

  return gradients


class ComponentsLayer(object):
  """A layer of components for a PathNet."""

  def __init__(self, components, combiner, sparse=True):
    """Constructor.

    Args:
      components: a list of `RoutedComponent`.
      combiner: a combiner to use to aggregate the outputs of the components.
      sparse: (bool) if set to False, every component in this layer will
        be evaluated, even if the router assigns it with a weight of 0.
        This behavior is necessary for approaches based on the Gumbel trick.
        By default, `sparse` is set to True, which means components with
        a weight of 0 are not evaluated.
    """
    self.components = components
    self.combiner = combiner
    self.sparse = sparse

  def __call__(self, layer_id, state, active_paths_weights, num_out_paths):
    components = self.components
    component_zero_output = self.zero_output(state, num_out_paths)
    combiner = self.combiner

    (layer_output, _) = component_zero_output
    out_paths_weights = tf.zeros([num_out_paths], dtype=tf.float32)
    entropy = tf.zeros([])

    with tf.variable_scope('layer%s' % layer_id):
      for i, component in enumerate(components):
        with tf.variable_scope('component%s' % i):
          one_hot_encoding = tf.one_hot(i, len(self.components),
                                        dtype=tf.float32)
          inner_product = tf.tensordot(active_paths_weights,
                                       one_hot_encoding, 1)
          if self.sparse:
            is_component_active = tf.not_equal(0.0, inner_product)
          else:
            is_component_active = tf.constant(True)

          (curr_component_output, curr_router_output) = tf.cond(
              is_component_active,
              lambda c=component: c(state, num_out_paths),
              lambda: component_zero_output
              )
          layer_output = combiner(layer_output, curr_component_output,
                                  is_component_active, active_paths_weights[i])
          (curr_out_paths_weights, curr_entropy) = curr_router_output
          out_paths_weights = tf.reduce_sum([
              out_paths_weights, curr_out_paths_weights], 0)
          entropy += curr_entropy
    return (layer_output, out_paths_weights, entropy)

  def zero_output(self, state, num_out_paths):
    return self.components[0].zero_output(state, num_out_paths)


class RoutedComponent(object):
  """Component with a router."""

  def __init__(self, component, router=None, record_summaries=True):
    self.component = component
    self.router = router
    self.record_summaries = record_summaries

  def __call__(self, state, num_out_paths):
    if self.record_summaries:
      tf.contrib.summary.scalar('active', 1)
    if self.router:
      router_output = self.router(state, num_out_paths)
    else:
      router_output = get_router_zero_output(num_out_paths)
    return (self.component(state), router_output)

  def zero_output(self, state, num_out_paths):
    return (self.component.zero_output(state),
            get_router_zero_output(num_out_paths))


class WeightedSumCombiner(object):
  """Combiner that creates a weighted sums of all inputs."""

  def __call__(self, prev, last, last_active, weight):
    del last_active
    if isinstance(prev, dict):
      assert isinstance(last, dict)
      rtn = {}
      for key in prev.keys():
        rtn[key] = prev[key] + last[key] * weight
      return rtn
    return prev + last * weight


class SumCombiner(object):
  """Combiner that sums all inputs."""

  def __call__(self, prev, last, last_active, weight):
    del last_active
    del weight
    if isinstance(prev, dict):
      assert isinstance(last, dict)
      rtn = {}
      for key in prev.keys():
        rtn[key] = prev[key] + last[key]
      return rtn
    return prev + last


class SelectCombiner(object):
  """Combiner that selects among the inputs."""

  def __call__(self, prev, last, last_active, weight):
    del weight
    return tf.cond(last_active, lambda: last, lambda: prev)


class FixedRouter(object):
  """Router with a fixed connection pattern conditioned only on `task_id`."""

  def __init__(self, connection_matrix, record_summaries=False):
    """Constructor.

    Args:
      connection_matrix: a tf.Tensor of shape [`num_tasks`, `num_out_paths`],
        which for each task and outgoing path describes the connection weight.
      record_summaries: (bool) whether to record summaries, such as
        path selection probabilities.
    """

    self.connection_matrix = connection_matrix
    self.record_summaries = record_summaries

  def __call__(self, state, num_out_paths):
    """Calls the router to get the connection weights.

    Args:
      state: a dictionary containing a key 'task_id' with the id of the current
        task.
      num_out_paths: number of modules to route through. Should match the second
        dimension of `self.connection_matrix`.

    Returns:
      A pair, containing the connection weights as the first element, and
      the allocation entropy as the second (which here is always 0).
    """
    assert self.connection_matrix.shape[1] == num_out_paths

    task_id = state['task_id']
    paths = self.connection_matrix[task_id]

    for j in range(num_out_paths):
      # The component will always be used unless the weight is 0.0
      prob = tf.cast(tf.not_equal(paths[j], 0.0), dtype=tf.float32)

      if self.record_summaries:
        tf.contrib.summary.scalar('prob_path_%s' % j, prob)

    # The return value is a tuple of (connection_weights, entropy)
    # Note that the entropy here is 0, as the allocation is fixed.
    return (paths, tf.constant(0.0))


class IndependentTaskBasedRouter(FixedRouter):
  """Router that assumes the next layer has exactly one component per task."""

  def __init__(self, num_tasks, record_summaries=False):
    super(IndependentTaskBasedRouter, self).__init__(
        tf.eye(num_tasks), record_summaries=record_summaries)


class SinglePathRouter(object):
  """Dummy router for the case when `num_out_paths` is 1."""

  def __init__(self, record_summaries=False):
    """Constructor.

    Args:
      record_summaries: (bool) whether to record summaries, such as
        path selection probabilities.
    """

    self.record_summaries = record_summaries

  def __call__(self, state, num_out_paths):
    assert num_out_paths == 1

    if self.record_summaries:
      tf.contrib.summary.scalar('prob_path_0', tf.ones(1))

    return (tf.ones([1]), tf.constant(0.0))


class GumbelMatrixRouter(object):
  """Router based on a matrix of binary Gumbel variables.

  All connections are modelled as a binary matrix of shape
  [`num_tasks`, `num_out_paths`]. This router allows for learning an arbitrary
  matrix of this form by using the Gumbel softmax.
  """

  def __init__(
      self,
      num_tasks,
      num_out_paths,
      probs_init=0.5,
      temperature_init=50,
      temperature_min=0.5,
      record_summaries=True):
    """Constructor.

    Args:
      num_tasks: (int) number of tasks.
      num_out_paths: (int) number of modules to route through.
      probs_init: initial probabilities for all connections.
        Can either be
          - a float, which is assumed to be the initial probability for all
            connections.
          - a numpy array of floats of shape [`num_tasks`, `num_out_paths`]
            containing the probabilities. Each connection probability can be
            independently chosen in the [0, 1] range.
      temperature_init: (float) initial value for the Gumbel softmax
        temperature. The temperature is decayed inversely to the square root of
        the number of steps.
      temperature_min: (float) minimum value for the Gumbel softmax temperature.
        Once the temperature reaches that value, it does not decay any longer.
      record_summaries: (bool) whether to record summaries, such as
        path selection probabilities.
    """
    self.num_out_paths = num_out_paths
    self.temperature_init = temperature_init
    self.temperature_min = temperature_min
    self.record_summaries = record_summaries

    shape = (num_tasks, num_out_paths)

    if isinstance(probs_init, float):
      probs_init = np.full(shape, fill_value=probs_init, dtype=np.float32)

    assert probs_init.shape == shape

    self.logits_init = np.log(np.stack([probs_init, 1.0 - probs_init], axis=-1))

  def __call__(self, state, num_out_paths):
    """Calls the router to get the connection weights.

    Args:
      state: a dictionary containing a key 'task_id' with the id of the current
        task.
      num_out_paths: number of modules to route through. Should match
        `num_out_paths` passed in the constructor.

    Returns:
      A pair, containing the connection weights as the first element,
      and router entropy as the second element.
    """
    assert self.num_out_paths == num_out_paths

    task_id = state['task_id']
    training = state['training']

    global_step = tf.cast(tf.train.get_or_create_global_step(), tf.float32)
    temperature = tf.maximum(
        self.temperature_min,
        self.temperature_init / tf.sqrt(global_step + 1.0))

    if self.record_summaries:
      tf.contrib.summary.scalar('temperature', temperature)

    paths = []

    with tf.variable_scope('router_dist'):
      self.logits = tf.get_variable(
          'logits', initializer=lambda: tf.constant(self.logits_init))

    for j in range(num_out_paths):
      logits = self.logits[task_id][j]

      sample, y_soft = self.binary_gumbel_softmax(
          logits, temperature, greedy=not training)
      prob = tf.nn.softmax(logits)[0]

      paths.append(sample)

      if self.record_summaries:
        tf.contrib.summary.scalar('prob_path_%s' % j, prob)
        tf.contrib.summary.scalar('soft_backprop_logit_%s' % j, y_soft)

    paths = tf.stack(paths)

    # Normalize so that the path weights sum up to 1 (except in the special case
    # when there are no connections). This is useful for combining the routed
    # outputs: by using a `WeightedSumCombiner`, the outputs from active
    # components are averaged, irrespective of how many components were chosen.
    paths /= tf.maximum(tf.reduce_sum(paths), 1.0)

    return (paths, self.get_entropy(task_id))

  def sample_gumbel(self, shape, eps=1e-20):
    """Draws independent samples from the Gumbel distribution.

    Args:
      shape: shape for the output tensor.
      eps: epsilon added under logarithm for numerical stability.

    Returns:
      A tf.Tensor of shape `shape`, where each entry has been independently
      drawn from the Gumbel distribution.
    """
    u = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(u + eps) + eps)

  def binary_gumbel_softmax(self, logits, temperature, greedy=False):
    """Draws a binary sample from a Gumbel softmax.

    Args:
      logits: tf.Tensor of shape [2] containing the logits of the probability
        distribution to sample from.
      temperature: temperature for the softmax that is used to approximate
        argmax in the backwards pass.
      greedy: (bool) if set to True, no Gumbel noise will be added, so the
        returned hard sample will be deterministic, and will correspond to the
        option with the highest probability.

    Returns:
      A pair of scalars `y` and `y_soft`, where `y` is the binary sample,
      and `y_soft` is a soft approximation of `y` that will be used in the
      backwards pass. Note that `y_soft` is returned only for logging purposes;
      this function already ensures that the gradients can flow from `y`
      (the binary sample) to the logits. It is possible because in the backwards
      pass, `y` is replaced by `y_soft`."
    """
    if greedy:
      gumbel_softmax_sample = logits
    else:
      gumbel_softmax_sample = logits + self.sample_gumbel(tf.shape(logits))

    y_soft = tf.nn.softmax(gumbel_softmax_sample / temperature)
    y_hard = tf.cast(tf.equal(y_soft, tf.reduce_max(
        y_soft, 0, keep_dims=True)), y_soft.dtype)

    y = tf.stop_gradient(y_hard - y_soft) + y_soft
    return y[0], y_soft[0]

  def get_connection_probs(self):
    """Gets probabilities for each of the router connections to be active.

    Returns:
      A tf.Tensor of floats with shape [`num_tasks`, `num_out_paths`], where
      each entry contains the probability of a particular connection to be
      active.
    """
    return tf.nn.softmax(self.logits)[:, :, 0]

  def get_expected_number_of_connections_for_task(self, task_id):
    """Gets the expected number of active connections for a task.

    Args:
      task_id: (int scalar) id of the task.

    Returns:
      A float scalar equal to the expected number of active connections for
      a given task.
    """
    probs = self.get_connection_probs()
    return tf.reduce_sum(probs[task_id])

  def get_probability_of_having_no_connections(self, task_id):
    """Gets the probability that a given task will have zero active connections.

    Args:
      task_id: (int scalar) id of the task.

    Returns:
      A float scalar in the [0.0, 1.0] range equal to the requested probability.
    """
    probs = self.get_connection_probs()
    return tf.reduce_prod(1.0 - probs[task_id])

  def get_entropy(self, task_id):
    """Gets the total entropy of decisions associated with a given task.

    Args:
      task_id: (int scalar) id of the task.

    Returns:
      A float scalar, equal to the sum of entropies over all binary
      allocation variables associated with a task `task_id`.
    """
    logits = self.logits[task_id]

    entropy = tf.constant(0.0)
    for i in range(logits.shape[0]):
      entropy += tfp.distributions.Categorical(logits=logits[i]).entropy()

    return entropy


def get_router_zero_output(num_paths):
  """All routers need to produce output with this shape."""
  return (tf.zeros([num_paths], dtype=tf.float32), tf.zeros([]))
