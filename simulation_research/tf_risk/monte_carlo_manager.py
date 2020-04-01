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

"""Time discretized Monte Carlo simulation engine for stochastic processes.

Useful to run simulations of trajectories of stochastic systems.
Provides wrappers to sample trajectories until convergence according to a
tolerance criterion.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
from absl import logging

import numpy as np
import tensorflow.compat.v1 as tf

from typing import Callable, MutableMapping, List, Optional, Tuple, Union
from simulation_research.tf_risk import util

NumpyArrayOrFloat = Union[np.ndarray, float]
TensorOrFloat = Union[tf.Tensor, float]
TensorOrNumpyArray = Union[tf.Tensor, np.ndarray]
TensorOrListOfTensors = Union[tf.Tensor, List[tf.Tensor]]
DynamicsOp = Callable[[tf.Tensor, TensorOrFloat, TensorOrFloat], tf.Tensor]
RunFunction = Callable[[DynamicsOp, tf.Tensor, TensorOrFloat, TensorOrFloat],
                       TensorOrListOfTensors]


def _max(x):
  """Returns max(x) for an array or max(x)=x for a scalar."""
  if isinstance(x, np.ndarray):
    return np.max(x)
  else:
    return x


def _min(x):
  """Returns min(x) for an array or min(x)=x for a scalar."""
  if isinstance(x, np.ndarray):
    return np.min(x)
  else:
    return x


def _maybe_tile_tensor_for_mc(tensor, num_samples):
  """Maybe tile state tensor to generate multiple scenarios in Monte Carlo."""
  original_rank = tensor.get_shape().rank
  if not original_rank:
    # If initial_state is a scalar, duplicate to [num_samples] shape.
    tensor = tf.expand_dims(tensor, axis=0)
    return tf.tile(tensor, [num_samples])
  elif original_rank == 1:
    # If initial_state is a vector, expand it to [num_samples, num_dims] shape.
    tensor = tf.expand_dims(tensor, axis=0)
    return tf.tile(tensor, [num_samples, 1])
  else:
    return tensor


def _reshape_initial_state_for_mc(initial_state,
                                  num_samples):
  """Tile initial conditions to generate multiple scenarios."""
  if (not isinstance(initial_state, tf.Tensor)) and isinstance(
      initial_state, collections.Iterable):
    return [_maybe_tile_tensor_for_mc(s, num_samples) for s in initial_state]
  else:
    return _maybe_tile_tensor_for_mc(initial_state, num_samples)


def multistep_runner_unrolled_tf(return_trajectories = False
                                ):
  """Explicit discretization scheme runner using an unrolled tensorflow loop.

  Returns a function simulating a dynamical process (potentially stochastic).

  Args:
    return_trajectories: whether to have run(...) return entire trajectories
      (states for all time steps) instead of just the final state.

  Returns:
    A run function taking 4 arguments
      1) dynamics_op: a function turning a triplet (state, time,
        discretization step) into the next state.
      2) states: typically a [num_samples, num_dims] tensor of scalars entailing
        the initial state values.
      3) dt: a scalar, the value of the discretization step;
      4) duration: a scalar, the amount of simulated time. Floor(duration / dt)
        steps will be run in all simulations;
    returning a (resp. a list of Floor(duration / dt) + 1)
    [num_samples, num_dims] tensor(s) of scalars if return_trajectories is False
    (resp. True).
  """

  def run(dynamics_op, states, dt, duration):
    t = 0.0
    trajectories = [states]
    while t < duration:
      trajectories.append(dynamics_op(trajectories[-1], t, dt))
      t += dt
    if return_trajectories:
      return trajectories
    return trajectories[-1]

  return run  # pytype: disable=bad-return-type


def multistep_runner_tf_while(maximum_iterations = None,
                              back_prop = True):
  """Explicit discretization scheme runner using a dynamic tensorflow loop.

  Returns a function simulating a dynamical process (potentially stochastic).

  Args:
    maximum_iterations: the maximum number of time steps dynamics will be
      simulated for.
    back_prop: whether to enable back-propagation through the while loop.
      Disabling back-propagation will reduce the amont of memory needed by a
      factor close to maximum_iterations factor.

  Returns:
    A run function (see multistep_runner_unrolled_tf's docstring for
    a description) returning a [num_samples, num_dims] tensor of scalars.
  """
  if not back_prop:
    logging.warning("Disabling back propagation in runner_tf_while.")

  def run(dynamics_op, states, dt, duration):
    t = tf.constant(0.0)
    _, end_states = tf.while_loop(
        cond=lambda t, s: tf.less(t, duration),
        body=lambda t, s: [t + dt, dynamics_op(s, t, dt)],
        loop_vars=[t, states],
        shape_invariants=[t.get_shape(), states.get_shape()],
        maximum_iterations=maximum_iterations,
        back_prop=back_prop)
    return end_states

  return run  # pytype: disable=bad-return-type


def non_callable_price_mc(
    initial_state,
    dynamics_op,
    payoff_fn,
    maturity,
    num_samples,
    dt,
    multistep_runner = multistep_runner_tf_while()):
  """Monte Carlo simulation to price a non callable derivative.

  Wrapper around run_multistep_dynamics to price European options.

  Args:
    initial_state: a scalar or a [num_samples, ...] tensor.
    dynamics_op: a function turning a triplet (state, time, discretization step)
      into the next state.
    payoff_fn: a function transforming the final state [num_samples, ...] of the
      simulation into a [num_samples] tensor of payoff samples.
    maturity: a scalar, the amount of simulated time. Floor(duration / dt) steps
      will be run.
    num_samples: the number of samples in the Monte-Carlo simulation.
    dt: a scalar, the value of the discretization step.
    multistep_runner: a run function (see multistep_runner_unrolled_tf's
      docstring for a description).

  Returns:
    a tensorflow scalar estimate of the mean of outcomes, a tensorflow scalar
      estimate of the mean of square outcomes, a [num_samples] tensor of
      outcomes.
  """
  if dt <= 0.0:
    raise ValueError("dt should be >0.0 but is %.f." % dt)

  initial_states = _reshape_initial_state_for_mc(initial_state, num_samples)

  terminal_states = multistep_runner(dynamics_op, initial_states, dt, maturity)

  outcomes = payoff_fn(terminal_states)

  mean_outcome = tf.reduce_mean(outcomes)
  mean_sq_outcome = tf.reduce_mean(outcomes**2)
  return mean_outcome, mean_sq_outcome, outcomes  # pytype: disable=bad-return-type


def sensitivity_autodiff(price, diff_target):
  """Compute the first order derivative of price estimate w.r.t. diff_target."""
  return tf.gradients([price], [diff_target])[0]


def mc_estimator(mean_est,
                 mean_sq_est,
                 batch_size,
                 key_placeholder,
                 feed_dict,
                 tol = 1e-3,
                 confidence = 0.95,
                 max_num_steps = 10000,
                 tf_session = None,
                 tol_is_relative = True
                ):
  """Run Monte-Carlo until convergence.

  Args:
    mean_est: [num_dims] tensor of scalars for the estimate of the mean of the
      outcome.
    mean_sq_est: [num_dims] tensor of scalars for the estimate of the mean of
      the squared outcome.
    batch_size: size of the mini-batch.
    key_placeholder: the placeholder entailing the key for the sub-stream of
      pseudo or quasi random numbers.
    feed_dict: (optional) feed_dict of placeholders for the parameters of the
      estimator (e.g. drift, volatility).
    tol: tolerance for the relative with of the confidence interval.
    confidence: level of confidence of the confidence interval. 0.95 means that
      we are confident at 95% that the actual mean is in the interval.
    max_num_steps: maximum number of mini-batches computed.
    tf_session: (optional) tensorflow session.
    tol_is_relative: (optional) whether to consider numerical precision in
      relative or absolute terms.

  Returns:
    the estimated mean ([num_dims] numpy array), the estimated (relative if
      tol_is_relative) half confidence interval given by the
      central limit theorem ([num_dims] numpy array), a boolean stating
      whether the method has converged or not.
  """
  if max_num_steps <= 0:
    raise ValueError("max_num_steps must be > 0 but is %d" % max_num_steps)

  if tf_session is None:
    config = tf.ConfigProto(isolate_session_state=True)
    with tf.Session(config=config) as tf_session:
      return mc_estimator(
          mean_est=mean_est,
          mean_sq_est=mean_sq_est,
          batch_size=batch_size,
          key_placeholder=key_placeholder,
          feed_dict=feed_dict,
          tol=tol,
          confidence=confidence,
          max_num_steps=max_num_steps,
          tf_session=tf_session,
          tol_is_relative=tol_is_relative)

  mean_est_eval = np.zeros(mean_est.get_shape().as_list())
  mean_sq_est_eval = np.zeros(mean_sq_est.get_shape().as_list())
  num_samples = 0
  num_steps = 0

  def _run_mc_step(
  ):
    """Run one mini-batch step of the Monte-Carlo method."""
    feed_dict.update({key_placeholder: num_steps})
    batch_mean_est, batch_mean_sq_est = tf_session.run((mean_est, mean_sq_est),
                                                       feed_dict=feed_dict)

    new_mean_est_eval = util.running_mean_estimate(mean_est_eval,
                                                   batch_mean_est, num_samples,
                                                   batch_size)
    new_mean_sq_est_eval = util.running_mean_estimate(mean_sq_est_eval,
                                                      batch_mean_sq_est,
                                                      num_samples, batch_size)
    std_est_eval = util.stddev_est(new_mean_est_eval, new_mean_sq_est_eval)

    half_conf_interval = util.half_clt_conf_interval(confidence,
                                                     num_samples + batch_size,
                                                     std_est_eval)

    return new_mean_est_eval, new_mean_sq_est_eval, half_conf_interval

  mean_est_eval, mean_sq_est_eval, half_conf_interval = _run_mc_step()

  converged = True
  effective_tol = (
      tol if not tol_is_relative else tol * _min(np.abs(mean_est_eval)))
  if effective_tol == 0.0:
    logging.warning("Zero effective tolerance. The run may not converge.")
  while _max(half_conf_interval) > effective_tol:
    logging.info("Monte Carlo estimation step %d", num_steps)
    logging.info("Half confidence interval %s", half_conf_interval)
    mean_est_eval, mean_sq_est_eval, half_conf_interval = _run_mc_step()
    effective_tol = (
        tol if not tol_is_relative else tol * _min(np.abs(mean_est_eval)))
    if effective_tol == 0.0:
      logging.warning("Zero effective tolerance. The run may not converge.")

    num_steps += 1
    num_samples += batch_size

    if num_steps >= max_num_steps:
      converged = False
      break

  logging.info("Monte Carlo estimation step %d", num_steps)
  logging.info("Half confidence interval %s", half_conf_interval)

  return mean_est_eval, half_conf_interval, converged
