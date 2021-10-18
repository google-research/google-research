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

"""Contains computations for transition matrices for the depth upscaling model.
"""
import jax.numpy as jnp
import numpy as np

Array = jnp.ndarray


def augment_text8(num_classes):
  """Computes transition matrices for text8 augmentation.

  Computes transition matrices for _augmented_ text8. This means that there is
  a specific consonant and vowel token that the problem has as intermediate
  variables.

  Args:
    num_classes: The number of classes that the problem has.

  Returns:
    A 3-tuple containing the transition matrices, the cumulative product of
    the transition matrices, and the final absorbing state (an integer).
  """
  text8_upscale_map = {
      0: 0, 1: 1, 2: 28, 3: 29, 4: 29, 5: 29, 6: 28, 7: 29, 8: 29, 9: 29,
      10: 28, 11: 29, 12: 29, 13: 29, 14: 29, 15: 29, 16: 28, 17: 29, 18: 29,
      19: 29, 20: 29, 21: 29, 22: 28, 23: 29, 24: 29, 25: 29, 26: 29, 27: 29}

  assert num_classes == 28
  num_stages = 2
  absorbing_state = 30
  limit_size = 31

  transition_matrices = np.zeros(
      (num_stages, limit_size, limit_size))

  inputs = set(range(num_classes))

  # Stage 1
  t = 1
  outputs = set()
  for in_class in inputs:
    out_class = text8_upscale_map[in_class]
    outputs.add(out_class)
    transition_matrices[t-1, out_class, in_class] = 1.

  # Stage 2
  t = 2
  inputs = outputs
  for in_class in inputs:
    transition_matrices[t-1, absorbing_state, in_class] = 1.

  cum_matmul_transition_matrices = np.zeros(
      (num_stages + 1, limit_size, limit_size))
  cum_matmul_transition_matrices[0, :num_classes, :num_classes] = np.eye(
      num_classes)

  for t in range(1, num_stages + 1):
    cum_matmul_transition_matrices[t, :, :] = transition_matrices[
        t - 1, :, :] @ cum_matmul_transition_matrices[t - 1, :, :]

  transition_matrices = jnp.array(transition_matrices, dtype=jnp.float32)
  cum_matmul_transition_matrices = jnp.array(
      cum_matmul_transition_matrices, dtype=jnp.float32)

  return transition_matrices, cum_matmul_transition_matrices, absorbing_state


def augment_least_significant_bit(num_classes,
                                  branch_factor):
  """Computes transition matrices, removing least significant bits each step.

  Computes transition matrices for _augmented_ bit reduction. In contrast with
  the standard bit flooring approach. Here, instead of simply flooring the least
  significant bits, they are redirected to a new augmented class.

  Args:
    num_classes: The number of classes that the problem has.
    branch_factor: The base in which least-significant values are removed.

  Returns:
    A 3-tuple containing the transition matrices, the cumulative product of
    the transition matrices, and the final absorbing state (an integer).
  """
  assert branch_factor >= 1
  num_stages = int(np.ceil(np.log(num_classes) / np.log(branch_factor)))

  # The transition matrices are initialized slightly larger, will be downsized
  # later.
  limit_size = branch_factor**(num_stages+1)
  transition_matrices = np.zeros((num_stages, limit_size, limit_size))
  inputs = set(range(num_classes))

  in_offset = 0
  out_offset = num_classes

  for t in range(1, num_stages + 1):
    outputs = set()
    for in_class in inputs:
      out_class = in_class // branch_factor  # Remove least signifant bits.
      outputs.add(out_class)

      # Offsets are used to augment the spaces.
      transition_matrices[
          t-1, out_class + out_offset, in_class + in_offset] = 1.

    # Outputs from previous round are new inputs
    new_out_offset = max(outputs) + out_offset + 1
    inputs = outputs
    in_offset = out_offset
    out_offset = new_out_offset

  assert len(outputs) == 1
  absorbing_state = outputs.pop() + in_offset  # The previous out_offset.

  # Recompute number of total classes.
  num_classes = absorbing_state + 1

  # Slice the relevant size of the transition matrices.
  transition_matrices = transition_matrices[:, :num_classes, :num_classes]

  cum_matmul_transition_matrices = np.zeros(
      (num_stages + 1, num_classes, num_classes))
  cum_matmul_transition_matrices[0, :, :] = np.eye(num_classes)

  for t in range(1, num_stages + 1):
    cum_matmul_transition_matrices[t, :, :] = transition_matrices[
        t - 1, :, :] @ cum_matmul_transition_matrices[t - 1, :, :]

  transition_matrices = jnp.array(transition_matrices, dtype=jnp.float32)
  cum_matmul_transition_matrices = jnp.array(
      cum_matmul_transition_matrices, dtype=jnp.float32)

  return transition_matrices, cum_matmul_transition_matrices, absorbing_state


def zero_least_significant_bit(num_classes,
                               branch_factor):
  """Computes transition matrices, removing least significant bits each step."""
  assert branch_factor >= 1
  num_stages = int(np.ceil(np.log(num_classes) / np.log(branch_factor)))

  transition_matrices = np.zeros((num_stages, num_classes, num_classes))
  inputs = set(range(num_classes))
  for t in range(1, num_stages + 1):
    outputs = set()
    for in_class in inputs:
      # Remove t least signifant digits in base `branch_factor`.
      out_class = (in_class // branch_factor**t) * branch_factor**t
      outputs.add(out_class)

      transition_matrices[t-1, out_class, in_class] = 1.

    # Outputs from previous round are new inputs
    inputs = outputs

  assert len(outputs) == 1
  absorbing_state = outputs.pop()

  cum_matmul_transition_matrices = np.zeros(
      (num_stages + 1, num_classes, num_classes))
  cum_matmul_transition_matrices[0, :, :] = np.eye(num_classes)

  for t in range(1, num_stages + 1):
    cum_matmul_transition_matrices[t, :, :] = transition_matrices[
        t - 1, :, :] @ cum_matmul_transition_matrices[t - 1, :, :]

  transition_matrices = jnp.array(transition_matrices, dtype=jnp.float32)
  cum_matmul_transition_matrices = jnp.array(
      cum_matmul_transition_matrices, dtype=jnp.float32)

  return transition_matrices, cum_matmul_transition_matrices, absorbing_state
