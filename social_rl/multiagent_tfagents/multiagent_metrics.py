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

"""TF metrics that work in the multi-agent case."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from absl import logging

import gin
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.drivers import tf_driver
from tf_agents.metrics import py_metrics
from tf_agents.metrics import tf_metric
from tf_agents.metrics.tf_metrics import TFDeque
from tf_agents.utils import common
from tf_agents.utils import numpy_storage

from social_rl.multiagent_tfagents.joint_attention import drivers


def zero_out_new_episodes(trajectory, return_accumulator):
  return tf.where(trajectory.is_first(), tf.zeros_like(return_accumulator),
                  return_accumulator)


@gin.configurable
class AverageReturnMetric(tf_metric.TFStepMetric):
  """Metric for the average collective return and individual agent returns."""

  def __init__(self,
               n_agents,
               name='MultiagentAverageReturn',
               prefix='Metrics',
               dtype=tf.float32,
               batch_size=1,
               buffer_size=10):
    super(AverageReturnMetric, self).__init__(name=name, prefix=prefix)
    self.n_agents = n_agents
    self._dtype = dtype

    # Accumulator and buffer for the average return of all agents
    self._collective_return_accumulator = common.create_variable(
        initial_value=0, dtype=dtype, shape=(batch_size,), name='Accumulator')
    self._collective_buffer = TFDeque(buffer_size, dtype)

    # Accumulators for each agent's independent reward
    self._agent_return_accumulators = []
    for a in range(n_agents):
      self._agent_return_accumulators.append(common.create_variable(
          initial_value=0, dtype=dtype, shape=(batch_size,),
          name='Accumulator' + str(a)))

    # Buffers for each agent's independent reward
    self._agent_buffers = []
    for a in range(n_agents):
      self._agent_buffers.append(TFDeque(buffer_size, dtype))

  @common.function(autograph=True)
  def call(self, trajectory):
    # Zero out batch indices where a new episode is starting.
    self._collective_return_accumulator.assign(
        zero_out_new_episodes(trajectory, self._collective_return_accumulator))
    for a in range(self.n_agents):
      self._agent_return_accumulators[a].assign(
          zero_out_new_episodes(trajectory, self._agent_return_accumulators[a]))

    # Note that trajectory.reward has shape (batch, n_agents)

    # Update accumulator with sum of received rewards.
    self._collective_return_accumulator.assign_add(
        tf.reduce_mean(trajectory.reward, axis=1))

    # Pull out data for each agent and assign
    for a in range(self.n_agents):
      self._agent_return_accumulators[a].assign_add(trajectory.reward[:, a])

    # Add final returns to buffer.
    last_episode_indices = tf.squeeze(tf.where(trajectory.is_last()), axis=-1)
    for indx in last_episode_indices:
      self._collective_buffer.add(self._collective_return_accumulator[indx])

      # Agent buffers that use the global done
      for a in range(self.n_agents):
        self._agent_buffers[a].add(self._agent_return_accumulators[a][indx])

    return trajectory

  def result(self):
    return self._collective_buffer.mean()

  def result_for_agent(self, agent_id):
    return self._agent_buffers[agent_id].mean()

  @common.function
  def reset(self):
    self._collective_buffer.clear()
    self._collective_return_accumulator.assign(
        tf.zeros_like(self._collective_return_accumulator))

    for a in range(self.n_agents):
      self._agent_buffers[a].clear()
      self._agent_return_accumulators[a].assign(
          tf.zeros_like(self._agent_return_accumulators[a]))

  def tf_summaries(self, train_step=None, step_metrics=()):
    """Generates summaries for all agents & collective summary against steps.

    Args:
      train_step: (Optional) Step counter for training iterations. If None, no
        metric is generated against the global step.
      step_metrics: (Optional) Iterable of step metrics to generate summaries
        against.

    Returns:
      A list of summaries.
    """
    summaries = super(AverageReturnMetric, self).tf_summaries(
        train_step=train_step, step_metrics=step_metrics)

    for a in range(self.n_agents):
      summaries.extend(self.single_agent_summary(
          a, train_step, step_metrics))

    return summaries

  def single_agent_summary(self, agent_id, train_step=None, step_metrics=()):
    summaries = []
    prefix = self._prefix
    name = self.name + '_agent' + str(agent_id)
    tag = common.join_scope(prefix, name)

    result = self.result_for_agent(agent_id)

    if train_step is not None:
      summaries.append(
          tf.compat.v2.summary.scalar(name=tag, data=result, step=train_step))
    if prefix:
      prefix += '_'
    for step_metric in step_metrics:
      # Skip plotting the metrics against itself.
      if self.name == step_metric.name:
        continue
      step_tag = '{}vs_{}/{}'.format(prefix, step_metric.name, name)
      # Summaries expect the step value to be an int64.
      step = tf.cast(step_metric.result(), tf.int64)
      summaries.append(tf.compat.v2.summary.scalar(
          name=step_tag,
          data=result,
          step=step))
    return summaries


@gin.configurable
class MultiagentScalar(tf_metric.TFStepMetric):
  """Metric to compute average of simple scalars like number of obstacles."""

  def __init__(self,
               n_agents,
               name,
               prefix='Metrics',
               dtype=tf.float32,
               buffer_size=10):
    super(MultiagentScalar, self).__init__(name=name, prefix=prefix)
    self._buffers = [TFDeque(buffer_size, dtype) for _ in range(n_agents)]
    self._n_agents = n_agents
    self._dtype = dtype

  @common.function(autograph=True)
  def call(self, new_scalar_vals, agent_id):
    self._buffers[agent_id].add(tf.reduce_mean(new_scalar_vals))
    return new_scalar_vals

  def result(self):
    return tf.reduce_mean([buffer.mean() for buffer in self._buffers])

  def result_for_agent(self, agent_id):
    return self._buffers[agent_id].mean()

  @common.function
  def reset(self):
    for buffer in self._buffers:
      buffer.clear()


def log_metrics(metrics, prefix=''):
  log = []
  for m in metrics:
    log.append('{0} = {1}'.format(m.name, m.result()))
    if 'Multiagent' in m.name:
      log += ['{0} = {1}'.format(
          m.name + '_agent' + str(a),
          m.result_for_agent(a)) for a in range(m.n_agents)]
  logging.info('%s \n\t\t %s', prefix, '\n\t\t '.join(log))


@gin.configurable
def eager_compute(metrics,
                  environment,
                  policy,
                  num_episodes=1,
                  train_step=None,
                  summary_writer=None,
                  summary_prefix='',
                  use_function=True,
                  use_attention_networks=False):
  """Compute metrics using `policy` on the `environment`.

  *NOTE*: Because placeholders are not compatible with Eager mode we can not use
  python policies. Because we use tf_policies we need the environment time_steps
  to be tensors making it easier to use a tf_env for evaluations. Otherwise this
  method mirrors `compute` directly.

  Args:
    metrics: List of metrics to compute.
    environment: tf_environment instance.
    policy: tf_policy instance used to step the environment.
    num_episodes: Number of episodes to compute the metrics over.
    train_step: An optional step to write summaries against.
    summary_writer: An optional writer for generating metric summaries.
    summary_prefix: An optional prefix scope for metric summaries.
    use_function: Option to enable use of `tf.function` when collecting the
      metrics.
    use_attention_networks: Option to use attention network architecture in the
    agent. This architecture requires observations from the previous time step.
  Returns:
    A dictionary of results {metric_name: metric_value}
  """
  for metric in metrics:
    metric.reset()

  multiagent_metrics = [m for m in metrics if 'Multiagent' in m.name]

  if use_attention_networks:
    driver = drivers.StateTFDriver(
        environment,
        policy,
        observers=metrics,
        max_episodes=num_episodes,
        disable_tf_function=not use_function,
    )
  else:
    driver = tf_driver.TFDriver(
        environment,
        policy,
        observers=metrics,
        max_episodes=num_episodes,
        disable_tf_function=not use_function)

  def run_driver():
    time_step = environment.reset()
    policy_state = policy.get_initial_state(environment.batch_size)
    if use_attention_networks:
      time_step.observation['policy_state'] = (
          policy_state['actor_network_state'][0],
          policy_state['actor_network_state'][1])
    driver.run(time_step, policy_state)

  if use_function:
    common.function(run_driver)()
  else:
    run_driver()

  results = [(metric.name, metric.result()) for metric in metrics]
  for m in multiagent_metrics:
    for a in range(m.n_agents):
      results.append((m.name + '_agent' + str(a), m.result_for_agent(a)))

  # TODO(b/120301678) remove the summaries and merge with compute
  if train_step and summary_writer:
    with summary_writer.as_default():
      for m in metrics:
        tag = common.join_scope(summary_prefix, m.name)
        tf.compat.v2.summary.scalar(name=tag, data=m.result(), step=train_step)
        if 'Multiagent' in m.name:
          for a in range(m.n_agents):
            tf.compat.v2.summary.scalar(name=tag + '_agent' + str(a),
                                        data=m.result_for_agent(a),
                                        step=train_step)
  # TODO(b/130249101): Add an option to log metrics.
  return collections.OrderedDict(results)


class MultiagentMetricsGroup(tf.Module):
  """Group a list of Metrics into a container."""

  def __init__(self, metrics, name=None):
    super(MultiagentMetricsGroup, self).__init__(name=name)
    self.metrics = metrics
    self.multiagent_metrics = [m for m in metrics if 'Multiagent' in m.name]

  def results(self):
    results = [(metric.name, metric.result()) for metric in self.metrics]

    for m in self.multiagent_metrics:
      for a in range(m.n_agents):
        results.append((m.name + '_agent' + str(a), m.result_for_agent(a)))
    return collections.OrderedDict(results)


@gin.configurable
class AverageReturnPyMetric(py_metrics.StreamingMetric):
  """Computes the average undiscounted reward."""

  def __init__(self,
               n_agents,
               name='MultiagentAverageReturn',
               buffer_size=10,
               batch_size=None):
    """Creates an AverageReturnPyMetric."""
    self.n_agents = n_agents
    self._np_state = numpy_storage.NumpyState()
    # Set a dummy value on self._np_state.episode_return so it gets included in
    # the first checkpoint (before metric is first called).
    self._np_state.episode_return = np.float64(0)
    self._agent_metrics = [
        py_metrics.AverageReturnMetric(
            'AverageReturn%i' % i, buffer_size=buffer_size)
        for i in range(n_agents)
    ]
    super(AverageReturnPyMetric, self).__init__(name, buffer_size=buffer_size,
                                                batch_size=batch_size)

  def result_for_agent(self, agent_id):
    return self._agent_metrics[agent_id].result()

  # We want to reuse methods for the sub-metrics
  # pylint: disable=protected-access
  def _reset(self, batch_size):
    """Resets stat gathering variables."""
    self._np_state.episode_return = np.zeros(
        shape=(batch_size,), dtype=np.float64)
    for metric in self._agent_metrics:
      metric._reset(batch_size)

  def _batched_call(self, trajectory):
    """Processes the trajectory to update the metric.

    Args:
      trajectory: a tf_agents.trajectory.Trajectory.
    """
    episode_return = self._np_state.episode_return
    agent_episode_returns = [
        metric._np_state.episode_return for metric in self._agent_metrics
    ]

    is_first = np.where(trajectory.is_first())
    episode_return[is_first] = 0
    for r in agent_episode_returns:
      r[is_first] = 0

    for i in range(self.n_agents):
      agent_episode_returns[i] += trajectory.reward[:, i]
    episode_return += np.mean(trajectory.reward, axis=-1)

    is_last = np.where(trajectory.is_last())
    self.add_to_buffer(episode_return[is_last])
    for metric in self._agent_metrics:
      metric.add_to_buffer(agent_episode_returns[i][is_last])
