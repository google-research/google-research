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

"""Transition functions for explicitly discretized stochastic dynamics.

See http://www.cmap.polytechnique.fr/~touzi/Poly-MAP552.pdf page 99 and 111
for more information on Geometric Brownian motions and
https://books.google.com/books?id=zOJmDwAAQBAJ for more on discretization
schemes for Stochastic Partial Differential Equations.

Pseudo-Random-Numbers are generated using keyed Pseudo-Random-Number-Generators
(PRNG). See
tensorflow/compiler/xla/client/lib/prng.cc
for implementation and
https://www.thesalmons.org/john/random123/papers/random123sc11.pdf
for a general introduction on keyed PRNGs.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf

from typing import Callable, List, Optional, Tuple, Union

TensorOrInt = Union[tf.Tensor, int]
TensorOrFloat = Union[tf.Tensor, float]
TensorOrNumpyArray = Union[tf.Tensor, np.ndarray]


def _convert_to_tensor_and_check_fully_known_shape(tensor
                                                  ):
  """Assert that tensor's shape is fully known."""
  tensor = tf.convert_to_tensor(value=tensor)
  if not tensor.shape.is_fully_defined():
    raise ValueError("Shape of %s should be fully known." % tensor.name)
  return tensor


################################################################################
#
#                         RANDOM NUMBER GENERATION
#
################################################################################


def _prng_key(i, key):
  """Generate a key for PRNG for counter value i and key value key."""
  return tf.stack([tf.cast(key, dtype=tf.int32),
                   tf.cast(i, dtype=tf.int32)],
                  axis=0)


def random_uniform(shape, i = 0,
                   key = 0):
  """Uniform (0, 1) sampling with counter value i and key value key.

  A keyed PRNG is employed to fill a tensor with the desired shape with
  samples distributed uniformly in (0, 1). For a given value of the counter i
  and the key the process is fully deterministic. Different keys will generate
  uncorrelated streams of samples.

  Args:
    shape: the desired shape of the samples (list of python integers).
    i: the counter value of the PRNG at generation (e.g. placeholder/tensor).
    key: the key value of the PRNG at generation (e.g. placeholder/tensor).

  Returns:
    a tensor of scalars whose shape is the shape argument.
  """
  return tf.random.stateless_uniform(shape=shape, seed=_prng_key(i, key))


def random_antithetic_uniform(shape,
                              i = 0,
                              key = 0):
  """Uniform (0, 1) antithetic sampling with counter value i and key value key.

  A keyed PRNG is employed to fill a tensor with the desired shape with
  samples distributed uniformly in (0, 1). For a given value of the counter i
  and the key the process is fully deterministic. Different keys will generate
  uncorrelated streams of samples.
  Antithetic sampling is employed to reduce the variance of averages computed
  from the sample. For each sample x in the first half of the output tensor,
  1 - x is present in the second half of the output tensor. The two mirrored
  samples have the same distribution and are anti-correlated.

  Args:
    shape: the desired shape of the samples (list of python integers).
    i: the counter value of the PRNG at generation (e.g. placeholder/tensor).
    key: the key value of the PRNG at generation (e.g. placeholder/tensor).

  Returns:
    a tensor of scalars whose shape is the shape argument. Sample mirroring
    occurs along axis 0 with a 50/50 split. Therefore it is expected that
    the first dimension of shape is an even number.

  Raises:
    ValueError whenever shape[0] is odd.
  """
  if shape[0] % 2 != 0:
    raise ValueError("Expects even number of samples for antithetic sampling.")
  shape = [shape[0] // 2] + shape[1:]
  sample = random_uniform(shape, i, key)
  return tf.concat([sample, 1.0 - sample], axis=0)


def random_normal(shape, i = 0,
                  key = 0):
  """Normal (0, I) antithetic sampling with counter value i and key value key.

  A keyed PRNG is employed to fill a tensor with the desired shape with
  samples distributed according to N(0, I). For a given value of the counter i
  and the key the process is fully deterministic. Different keys will generate
  uncorrelated streams of samples.

  Args:
    shape: the desired shape of the samples (list of python integers).
    i: the counter value of the PRNG at generation.
    key: the key value of the PRNG at generation.

  Returns:
    a tensor of scalars whose shape is the shape argument.
  """

  return tf.random.stateless_normal(shape=shape, seed=_prng_key(i, key))


def random_antithetic_normal(shape,
                             i = 0,
                             key = 0):
  """Normal (0, I) sampling with counter value i and key value key.

  A keyed PRNG is employed to fill a tensor with the desired shape with
  samples distributed according to N(0, I). For a given value of the counter i
  and the key the process is fully deterministic. Different keys will generate
  uncorrelated streams of samples.
  Antithetic sampling is employed to reduce the variance of averages computed
  from the sample. For each sample x in the first half of the output tensor,
  - x is present in the second half of the output tensor. The two mirrored
  samples have the same distribution and are anti-correlated.

  Args:
    shape: the desired shape of the samples (list of python integers).
    i: the counter value of the PRNG at generation.
    key: the key value of the PRNG at generation.

  Returns:
    a tensor of scalars whose shape is the shape argument. Sample mirroring
    occurs along axis 0 with a 50/50 split. Therefore it is expected that
    the first dimension of shape is an even number.

  Raises:
    ValueError whenever shape[0] is odd.
  """
  if shape[0] % 2 != 0:
    raise ValueError("Expects even number of samples for antithetic sampling.")
  shape = [shape[0] // 2] + shape[1:]
  sample = random_normal(shape, i, key)
  return tf.concat([sample, -sample], axis=0)


################################################################################
#
#                  1D GEOMETRIC BROWNIAN PROCESSES AND RELATED
#
################################################################################


def gbm_euler_step(states,
                   drift,
                   vol,
                   t,
                   dt,
                   key = 0,
                   random_normal_op = None
                  ):
  """Euler discretization step for the Geometric Brownian Motion.

  Whenever random_normal_op is not specified, Threefry is used as
  a back-end with a counter initialized to 0 and [key, int(t / dt)]
  as a block-cipher key.
  Will only use vector operations on TPU, tf.float32 is default precision.

  Args:
    states: a [num_samples] tensor of scalars, the simulated processes' states.
    drift: a scalar, the Geometric Brownian Motion drift.
    vol: a scalar, the Geometric Brownian Motion volatility.
    t: a scalar, the current time value.
    dt: a scalar, the temporal discretization resolution.
    key: an int32, the key representing the id for the random number stream.
      2^32 different keys can be used here, keyed streams have a period of 2^32.
    random_normal_op: function taking no arguments and returning a [num_samples]
      tensor of normal pseudo-random-numbers to override the default normal
      pseudo-random-number generator (for testing and debugging).

  Returns:
    a [num_samples] tensor of scalars, the simulated states at
      the next time step.

  Raises:
    ValueError if shape of states is not fully known.
  """

  states = _convert_to_tensor_and_check_fully_known_shape(states)

  if random_normal_op:
    normals = random_normal_op()
  else:
    num_samples = states.get_shape().as_list()[0]
    normals = random_normal([num_samples], t / dt, key)

  dw_t = normals * tf.sqrt(dt)

  return states * (
      tf.constant(1.0, dtype=states.dtype) + drift * dt + vol * dw_t)


def gbm_euler_step_running_max(
    states_and_max,
    drift,
    vol,
    t,
    dt,
    simulate_bridge = True,
    key = 0,
    random_normal_op = None,
    random_uniform_op = None):
  """Euler discretization step for Geometric Brownian Motion and its maximum.

  Uses gbm_euler_step to generate a Geometric Brownian Motion.
  If simulate_bridge is True, the Brownian bridge maximum simulation method
  (https://www.springer.com/us/book/9783319902746 p372 or
  https://people.maths.ox.ac.uk/gilesm/mc/module_4/module_4_1.pdf slide 34)
  is employed to reduce the discretization bias affecting the running maximum.
  To simulate the bridge, a generator of uniform (0, 1) samples is needed,
  these uniform samples have to be uncorrelated from the normal samples.
  If random_normal_op is None and random_uniform_op is None then steps are
  automatically taken to guarantee the uniform samples are uncorrelated to
  the normal samples employed to generate the Geometric Brownian Motion itself.

  Args:
    states_and_max: a pair of [num_samples] tensors of scalars, the simulated
      processes' states and running maxima.
    drift: a scalar, the Geometric Brownian Motion drift.
    vol: a scalar, the Geometric Brownian Motion volatility.
    t: a scalar, the current time value.
    dt: a scalar, the temporal discretization resolution.
    simulate_bridge: whether to simulate the running maximum between discretized
      time steps with the Brownian bridge maximum method.
    key: an int32, the key representing the id for the random number stream.
      2^32 different keys can be used here, keyed streams have a period of 2^32.
    random_normal_op: function taking no arguments and returning a [num_samples]
      tensor of normal pseudo-random-numbers to override the default normal
      pseudo-random-number generator (for testing and debugging).
    random_uniform_op: function taking no arguments and returning a
      [num_samples] tensor of uniform (0, 1) pseudo-random-numbers to override
      the default normal pseudo-random-number generator (for testing and
      debugging).

  Returns:
    a pair of [num_samples] tensors of scalars, the simulated states and running
      maxima at the next time step.

  Raises:
    ValueError if shapes of tensors in states_and_max are not fully known.
  """
  (states, running_max) = states_and_max

  states = _convert_to_tensor_and_check_fully_known_shape(states)
  running_max = _convert_to_tensor_and_check_fully_known_shape(running_max)

  if simulate_bridge:
    # Use a stride for the key to guarantee the normal and uniform samples are
    # not correlated.
    key *= 2

  next_states = gbm_euler_step(states, drift, vol, t, dt, key, random_normal_op)

  if simulate_bridge:
    # Use a stride for the key to guarantee the normal and uniform samples are
    # not correlated.
    if random_uniform_op:
      uniforms = random_uniform_op()
    else:
      num_samples = states.get_shape().as_list()[0]
      uniforms = random_uniform([num_samples], t / dt, key + 1)

    bridge_max = tf.constant(
        0.5, dtype=states.dtype) * (
            states + next_states +
            tf.sqrt((states - next_states)**2 -
                    tf.constant(2.0, dtype=states.dtype) * dt *
                    ((vol * states)**2) * tf.math.log(uniforms)))
    running_max = tf.maximum(running_max, bridge_max)

  running_max = tf.maximum(running_max, next_states)

  return (next_states, running_max)  # pytype: disable=bad-return-type


def gbm_euler_step_running_sum(
    states_and_sums,
    drift,
    vol,
    t,
    dt,
    key = 0,
    random_normal_op = None):
  """Run Euler discretization step and keep track of a running sum.

  Uses gbm_euler_step to generate a Geometric Brownian Motion.

  Args:
    states_and_sums: a pair of [num_samples] tensors of scalars, the simulated
      processes' states and running sums.
    drift: a scalar, the Geometric Brownian Motion drift.
    vol: a scalar, the Geometric Brownian Motion volatility.
    t: a scalar, the current time value.
    dt: a scalar, the temporal discretization resolution.
    key: an int32, the key representing the id for the random number stream.
      2^32 different keys can be used here, keyed streams have a period of 2^32.
    random_normal_op: function taking no arguments and returning a [num_samples]
      tensor of normal pseudo-random-numbers to override the default normal
      pseudo-random-number generator (for testing and debugging).

  Returns:
    a pair of [num_samples] tensors of scalars, the simulated states and running
      sums at the next time step.

  Raises:
    ValueError if shapes of tensors in states_and_sums are not fully known.
  """
  (states, running_sums) = states_and_sums

  states = _convert_to_tensor_and_check_fully_known_shape(states)
  running_sums = _convert_to_tensor_and_check_fully_known_shape(running_sums)

  states = gbm_euler_step(states, drift, vol, t, dt, key, random_normal_op)

  running_sums = running_sums + states

  return (states, running_sums)  # pytype: disable=bad-return-type


################################################################################
#
#                  ND GEOMETRIC BROWNIAN PROCESSES AND RELATED
#
################################################################################


def gbm_euler_step_nd(states,
                      drift,
                      vol,
                      t,
                      dt,
                      key = 0,
                      random_normal_op = None
                     ):
  """Euler discretization step of multi-dim Geometric Brownian Motion.

  Whenever dw_t is not specified, Threefry is used as a back-end with
  a counter initialized to 0 and [key, int(t / dt)] as a block-cipher key.

  Args:
    states: a [num_samples, num_dims] tensor of scalars, the simulated
      processes' states.
    drift: a [num_dims] tensor of scalars, the Geometric Brownian Motion drift.
    vol: a [num_dims, num_dims] tensor of scalars, the Geometric Brownian Motion
      volatility matrix.
    t: a scalar, the current time value.
    dt: a scalar, the temporal discretization resolution.
    key: an int32, the key representing the id for the random number stream.
      2^32 different keys can be used here, keyed streams have a period of 2^32.
    random_normal_op: function taking no arguments and returning a [num_samples,
      num_dims] tensor of normal pseudo-random-numbers to override the default
      normal pseudo-random-number generator (for testing and debugging).

  Returns:
    a [num_samples, num_dims] tensor of scalars, the simulated states at
      the next time step.

  Raises:
    ValueError if shape of states is not fully known.
  """

  states = _convert_to_tensor_and_check_fully_known_shape(states)

  if random_normal_op:
    normals = random_normal_op()
  if random_normal_op is None:
    sample_shape = states.get_shape().as_list()
    normals = random_normal(sample_shape, t / dt, key)

  dw_t = normals * tf.sqrt(dt)

  return states * (
      tf.constant(1.0, dtype=states.dtype) + drift * dt +
      tf.matmul(dw_t, vol, transpose_b=True))


def gbm_log_euler_step(
    log_states,
    drift,
    vol,
    t,
    dt,
    key = 0,
    random_normal_op = None):
  """Log-scale euler discretization step for the Geometric Brownian Motion.

  Whenever dw_t is not specified, Threefry is used as a back-end with
  a counter initialized to 0 and [key, int(t / dt)] as a block-cipher key.
  Will only use vector operations on TPU, tf.float32 is default precision.

  Args:
    log_states: a [num_samples] tensor of scalars, the simulated processes'
      states in log-scale.
    drift: a scalar, the Geometric Brownian Motion drift.
    vol: a scalar, the Geometric Brownian Motion volatility.
    t: a scalar, the current time value.
    dt: a scalar, the temporal discretization resolution.
    key: an int32, the key representing the id for the random number stream.
      2^32 different keys can be used here, keyed streams have a period of 2^32.
    random_normal_op: function taking no arguments and returning a [num_samples]
      tensor of normal pseudo-random-numbers to override the default normal
      pseudo-random-number generator (for testing and debugging).

  Returns:
    a [num_samples] tensor of scalars, the simulated states at the next time
      step in log-scale.

  Raises:
    ValueError if shape of log_states is not fully known.
  """

  log_states = _convert_to_tensor_and_check_fully_known_shape(log_states)

  if random_normal_op:
    normals = random_normal_op()
  else:
    num_samples = log_states.get_shape().as_list()[0]
    normals = random_normal([num_samples], t / dt, key)

  dw_t = normals * tf.sqrt(dt)

  return (log_states + (drift - tf.constant(0.5, dtype=log_states.dtype) *
                        (vol**2)) * dt + vol * dw_t)


def gbm_log_euler_step_nd(
    log_states,
    drift,
    vol,
    t,
    dt,
    key = 0,
    random_normal_op = None):
  """Log-scale Euler discretization step of multi-dim Geometric Brownian Motion.

  Whenever dw_t is not specified, Threefry is used as a back-end with
  a counter initialized to 0 and [key, int(t / dt)] as a block-cipher key.

  Args:
    log_states: a [num_samples, num_dims] tensor of scalars, the simulated
      processes' states in log-scale.
    drift: a [num_dims] tensor of scalars, the Geometric Brownian Motion drift.
    vol: a [num_dims, num_dims] tensor of scalars, the Geometric Brownian Motion
      volatility matrix.
    t: a scalar, the current time value.
    dt: a scalar, the temporal discretization resolution.
    key: an int32, the key representing the id for the random number stream.
      2^32 different keys can be used here, keyed streams have a period of 2^32.
    random_normal_op: function taking no arguments and returning a [num_samples,
      num_dims] tensor of normal pseudo-random-numbers to override the default
      normal pseudo-random-number generator (for testing and debugging).

  Returns:
    a [num_samples, num_dims] tensor of scalars, the simulated states at
      the next time step in log-scale.

  Raises:
    ValueError if shape of log_states is not fully known.
  """

  log_states = _convert_to_tensor_and_check_fully_known_shape(log_states)

  if random_normal_op:
    normals = random_normal_op()
  else:
    sample_shape = log_states.get_shape().as_list()
    normals = random_normal(sample_shape, t / dt, key)

  dw_t = normals * tf.sqrt(dt)

  return (log_states + (drift - tf.constant(0.5, dtype=log_states.dtype) *
                        tf.reduce_sum(input_tensor=vol**2, axis=0)) * dt +
          tf.matmul(dw_t, vol, transpose_b=True))
