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

"""Utilities for testing task families."""

import json
from absl import logging
from absl.testing import parameterized

from task_set import datasets
import tensorflow.compat.v1 as tf


class SingleTaskTestCase(parameterized.TestCase, tf.test.TestCase):

  def task_test(self, task):
    """Smoke test tasks to ensure they can produce gradients."""
    params = task.initial_params()
    loss = task.call_split(params, datasets.Split.TRAIN)
    grads = task.gradients(loss, params)
    with self.cached_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(grads)


class TaskFamilyTestCase(parameterized.TestCase, tf.test.TestCase):
  """Base TestCase used for testing sampled task families."""

  def __init__(self, sampler, getter, *args, **kwargs):
    self.sampler = sampler
    self.getter = getter
    super(TaskFamilyTestCase, self).__init__(*args, **kwargs)

  @parameterized.parameters(range(20))
  def test_configs_are_unique_and_consistent(self, seed):
    """Test that samplers produce the same configs for the same seeds."""
    cfg1 = self.sampler(seed)
    cfg2 = self.sampler(seed)
    self.assertEqual(cfg1, cfg2)

    cfg3 = self.sampler(seed + 10)
    self.assertNotEqual(cfg1, cfg3)

  @parameterized.parameters(range(20))
  def test_serialize_configs(self, seed):
    """Test that configs are serializable."""
    cfg = self.sampler(seed)
    try:
      _ = json.dumps(cfg)
    except ValueError:
      self.fail("Failed to serialize config to json!")

  @parameterized.parameters(range(2))
  def test_run_task_graph(self, seed):
    """Test that a graph can be constructed, and gradients can be computed."""
    cfg = self.sampler(seed)
    logging.info("Checking cfg: %s", cfg)
    task = self.getter(cfg)
    params = task.initial_params()
    loss = task.call_split(params, datasets.Split.TRAIN)
    grads = task.gradients(loss, params)
    with self.cached_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(grads)

  @parameterized.parameters(range(2, 10))
  def test_build_task_graph(self, seed):
    """Test that a graph can be constructed.

    This is faster than constructing and running, thus we can run more seeds.
    Args:
      seed: seed to call the sampler with.
    """
    cfg = self.sampler(seed)
    logging.info("Checking cfg: %s", cfg)
    tf.reset_default_graph()
    self.getter(cfg)
