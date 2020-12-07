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

r"""Implementation of the d_{\Delta} metric.

Defined as: d_{\Delta}(s, t) = \max_{\pi\in\Pi, a\in A} |Q*(s, a) - Q*(t, a)|

It's infeasible to maximize over all policies, but we use adversarial value
functions (AVFs) as a proxy for this. AVFs were defined in
Bellemare et al., 2019:
  "A Geometric Perspective on Optimal Representations for Reinforcement
   Learning"
  https://arxiv.org/pdf/1901.11530.pdf is the reference paper.
"""
import time
import gin
import numpy as np
import tensorflow.compat.v1 as tf
from rl_metrics_aaai2021 import metric


@gin.configurable
def avf_values(env, num_optimization_steps=1000, num_tasks=50,
               tf_device='/cpu:*'):
  """This method computes a set of AVFs.

  Args:
    env: an environment.
    num_optimization_steps: int, number of steps to optimize policy over.
    num_tasks: int, number of tasks.
    tf_device: str, name of TF device to use (CPU or GPU).

  Returns:
    vf_output
  """
  num_states = env.num_states
  num_actions = env.num_actions
  # Construct compatible P matrix, with action at the end, and assuming
  # deterministic transitions.
  p = np.transpose(env.transition_probs, [0, 2, 1])
  r = env.rewards
  sess = tf.Session()
  # deltas: probe vector, num_tasks x num_states-dimensional.
  deltas = np.random.random((num_tasks, num_states)) * 2.0 - 1.0
  with tf.device(tf_device):
    dummy = tf.constant(np.array([[1.0]]), dtype=tf.float32)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(num_tasks * num_states * num_actions))
    logits = model(dummy)
    logits = tf.reshape(logits, [num_tasks, num_states, num_actions])
    policy = tf.nn.softmax(logits)
    expanded_policy = tf.expand_dims(policy, axis=-1)
    tiled_p = np.tile(p, [num_tasks, 1, 1, 1])
    tiled_p *= np.float64(env.gamma)  # Precompute P as much as possible.
    expanded_policy = tf.cast(expanded_policy, tf.float64)
    ppi = tf.matmul(tiled_p, expanded_policy)
    ppi = tf.reshape(ppi, (num_tasks, num_states, num_states))
    tiled_eye = np.tile(np.eye(num_states), [num_tasks, 1, 1])
    # tiled_eye - Ppi is k x n x n, the last two dimensions are inverted.
    # Note discount factor has been incorporated above.
    resolvent = tf.linalg.inv(tiled_eye - ppi)
    r = np.expand_dims(r, axis=1)  # Make rewards row vectors.
    r = np.tile(r, [num_tasks, 1, 1, 1])
    rpi = tf.matmul(r, expanded_policy)  # Expected reward.
    rpi = tf.reshape(rpi, (num_tasks, num_states, 1))  # Drop last dimension.
    value_func = tf.matmul(resolvent, rpi)
    # Deltas should be k x n, make k x 1 x n (row vectors).
    deltas = np.expand_dims(deltas, axis=1)
    deltas = tf.constant(deltas, dtype=tf.float64)
    loss = -tf.matmul(deltas, value_func)
    optimizer = tf.train.AdamOptimizer(
        learning_rate=0.1, epsilon=0.0003125)
    loss = tf.reduce_sum(loss)
    train_op = optimizer.minimize(loss)
  sess.run(tf.initializers.global_variables())
  for _ in range(num_optimization_steps):
    sess.run([loss, train_op])
  vf_output = sess.run(value_func)
  return vf_output


@gin.configurable
class DDelta(metric.Metric):
  r"""Implementation of the d_{\Delta} metric."""

  def __init__(self,
               name,
               label,
               env,
               base_dir,
               gamma=0.9,
               normalize=False,
               num_tasks=50):
    self.num_tasks = num_tasks
    super().__init__(name, label, env, base_dir, gamma=0.9, normalize=False)

  def _compute(self, tolerance, verbose=False):
    start_time = time.time()
    num_optimization_steps = 1000
    value_functions = avf_values(
        self.env,
        num_optimization_steps=num_optimization_steps,
        num_tasks=self.num_tasks)
    value_functions = np.squeeze(value_functions)
    value_functions = np.atleast_2d(value_functions)
    self.metric = np.zeros((self.num_states, self.num_states))
    for s in range(self.num_states):
      # We take advantage of symmetry for faster computation.
      for t in range(s + 1, self.num_states):
        max_difference = 0.0
        for policy in range(value_functions.shape[0]):
          for a in range(self.num_actions):
            q1 = (self.env.rewards[s, a] +
                  self.env.gamma * np.matmul(self.env.transition_probs[s, a, :],
                                             value_functions[policy, :]))
            q2 = (self.env.rewards[t, a] +
                  self.env.gamma * np.matmul(self.env.transition_probs[t, a, :],
                                             value_functions[policy, :]))
            action_diff = abs(q1 - q2)
            if action_diff > max_difference:
              max_difference = action_diff
        self.metric[s, t] = max_difference
        self.metric[t, s] = max_difference
    # We don't really have a sampled versiion of this.
    total_time = time.time() - start_time
    self.statistics = metric.Statistics(
        0., total_time, num_optimization_steps, 0.)


# Using classes to define d_Delta metrics with different subsamples of AVF
# to be used in the 'METRIC' dic
@gin.configurable
class DDelta50(DDelta):
  pass


@gin.configurable
class DDelta1(DDelta):
  r"""Implementation of the d_{\Delta} metric with 100 AVFs."""

  def __init__(self,
               name,
               label,
               env,
               base_dir,
               gamma=0.9,
               normalize=False,
               num_tasks=1):
    super().__init__(name, label, env, base_dir, gamma, normalize, num_tasks)


@gin.configurable
class DDelta5(DDelta):
  r"""Implementation of the d_{\Delta} metric with 100 AVFs."""

  def __init__(self,
               name,
               label,
               env,
               base_dir,
               gamma=0.9,
               normalize=False,
               num_tasks=5):
    super().__init__(name, label, env, base_dir, gamma, normalize, num_tasks)


@gin.configurable
class DDelta10(DDelta):
  r"""Implementation of the d_{\Delta} metric with 100 AVFs."""

  def __init__(self,
               name,
               label,
               env,
               base_dir,
               gamma=0.9,
               normalize=False,
               num_tasks=10):
    super().__init__(name, label, env, base_dir, gamma, normalize, num_tasks)


@gin.configurable
class DDelta15(DDelta):
  r"""Implementation of the d_{\Delta} metric with 100 AVFs."""

  def __init__(self,
               name,
               label,
               env,
               base_dir,
               gamma=0.9,
               normalize=False,
               num_tasks=15):
    super().__init__(name, label, env, base_dir, gamma, normalize, num_tasks)


@gin.configurable
class DDelta20(DDelta):
  r"""Implementation of the d_{\Delta} metric with 100 AVFs."""

  def __init__(self,
               name,
               label,
               env,
               base_dir,
               gamma=0.9,
               normalize=False,
               num_tasks=15):
    super().__init__(name, label, env, base_dir, gamma, normalize, num_tasks)


@gin.configurable
class DDelta100(DDelta):
  r"""Implementation of the d_{\Delta} metric with 100 AVFs."""

  def __init__(self,
               name,
               label,
               env,
               base_dir,
               gamma=0.9,
               normalize=False,
               num_tasks=100):
    super().__init__(name, label, env, base_dir, gamma, normalize, num_tasks)


@gin.configurable
class DDelta500(DDelta):
  r"""Implementation of the d_{\Delta} metric with 500 AVFs."""

  def __init__(self,
               name,
               label,
               env,
               base_dir,
               gamma=0.9,
               normalize=False,
               num_tasks=500):
    super().__init__(name, label, env, base_dir, gamma, normalize, num_tasks)


@gin.configurable
class DDelta1000(DDelta):
  r"""Implementation of the d_{\Delta} metric with 1000 AVFs."""

  def __init__(self,
               name,
               label,
               env,
               base_dir,
               gamma=0.9,
               normalize=False,
               num_tasks=1000):
    super().__init__(name, label, env, base_dir, gamma, normalize, num_tasks)


@gin.configurable
class DDelta5000(DDelta):
  r"""Implementation of the d_{\Delta} metric with 5000 AVFs."""

  def __init__(self,
               name,
               label,
               env,
               base_dir,
               gamma=0.9,
               normalize=False,
               num_tasks=5000):
    super().__init__(name, label, env, base_dir, gamma, normalize, num_tasks)
