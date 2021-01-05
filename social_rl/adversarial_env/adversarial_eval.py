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

"""TF-metrics for evaluating adversarial environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from absl import logging

import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.metrics import tf_metric
from tf_agents.metrics.tf_metrics import TFDeque
from tf_agents.utils import common


@gin.configurable
class AdversarialEnvironmentScalar(tf_metric.TFStepMetric):
  """Metric to compute average of simple scalars like number of obstacles."""

  def __init__(self,
               name,
               prefix='Metrics',
               dtype=tf.float32,
               batch_size=1,
               buffer_size=10):
    super(AdversarialEnvironmentScalar, self).__init__(name=name, prefix=prefix)
    self._buffer = TFDeque(buffer_size, dtype)
    self._dtype = dtype

  @common.function(autograph=True)
  def call(self, new_scalar_vals):
    for v in new_scalar_vals:
      self._buffer.add(v)
    return new_scalar_vals

  def result(self):
    return self._buffer.mean()

  @common.function
  def reset(self):
    self._buffer.clear()


def log_metrics(agents, env_metrics, prefix=''):
  """Log metrics for adversary, protagonist, and antagonist agents."""
  log_agents = [agents[k] for k in agents.keys() if agents[k] is not None]
  for agent_list in log_agents:
    for i, agent in enumerate(agent_list):
      log = []
      for m in agent.eval_metrics:
        log.append('{0} {1} = {2}'.format(m.name, i, m.result()))
      logging.info('%s \n\t\t %s', prefix, '\n\t\t '.join(log))

  log = []
  for m in env_metrics:
    log.append('{0} = {1}'.format(m.name, m.result()))
  logging.info('%s \n\t\t %s', prefix, '\n\t\t '.join(log))


@gin.configurable
def eager_compute(driver,
                  agents,
                  env_metrics=None,
                  train_step=None,
                  summary_writer=None,
                  summary_prefix=''):
  """Run adversary, protagonist, and antagonist agents using adversarial driver.

  Args:
    driver: An instance of adversarial_driver in eval mode.
    agents: A Dictionary of TrainAgentPackage instances, which each contain
      metrics and eval policies.
    env_metrics: Global environment metrics to log, such as path lengths.
    train_step: An optional step to write summaries against.
    summary_writer: An optional writer for generating metric summaries.
    summary_prefix: An optional prefix scope for metric summaries.
  Returns:
    A dictionary of results {metric_name: metric_value}
  """
  log_agents = [agents[k] for k in agents.keys() if agents[k] is not None]

  for agent_list in log_agents:
    for agent in agent_list:
      for metric in agent.eval_metrics:
        metric.reset()
      if agent.is_environment:
        agent.env_eval_metric.reset()

  for metric in env_metrics:
    metric.reset()

  train_idxs = driver.run()

  results = []
  for name, agent_list in agents.items():
    # Train the agent selected by the driver this training run
    for agent_idx in train_idxs[name]:
      agent = agent_list[agent_idx]
      if agent.is_environment:
        agent_metrics = agent.eval_metrics + [agent.env_eval_metric]
      else:
        agent_metrics = agent.eval_metrics
      results.extend(
          [(metric.name, metric.result()) for metric in agent_metrics])

    record_metrics(agent_metrics, train_step, summary_writer, summary_prefix)

  results.extend([(metric.name, metric.result()) for metric in env_metrics])
  record_metrics(env_metrics, train_step, summary_writer, summary_prefix)

  # TODO(b/130249101): Add an option to log metrics.
  return collections.OrderedDict(results)


def record_metrics(metrics, train_step, summary_writer, summary_prefix):
  if train_step and summary_writer:
    with summary_writer.as_default():
      for m in metrics:
        tag = common.join_scope(summary_prefix, m.name)
        tf.compat.v2.summary.scalar(
            name=tag, data=m.result(), step=train_step)

