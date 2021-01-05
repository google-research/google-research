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

"""Uses stochastic_descent to find local optima of bang-bang protocols.

Starting from a given (likely random) protocol, iterate through all of the
protocols that are Hamming distance at most k away and evaluate their
performance. The iteration proceeds in a random order, and when a better
protocol is found then jump to that one and restart the search from the new
origin. The algorithm terminates when none of the neighbors have better
performance, and this defines a local optimum.
"""

import itertools

from absl import logging
import numpy as np

from bangbang_qaoa import circuit_lib


def get_all_protocols(num_chunks):
  """Returns all possible bang-bang protocols of a given number of chunks.

  Args:
    num_chunks: Positive integer, the length of the bang-bang protocol.

  Returns:
    A generator containing all possible bang-bang protocols of a particular
    length.

  Raises:
    ValueError: If num_chunks is not positive.
  """
  if num_chunks <= 0:
    raise ValueError('num_chunks should be positive, not %d' % num_chunks)
  return itertools.product([circuit_lib.HamiltonianType.X,
                            circuit_lib.HamiltonianType.CONSTRAINT],
                           repeat=num_chunks)


def get_random_protocol(num_chunks, random_state=None):
  """Gets a uniformly random protocol of a given number of chunks.

  Args:
    num_chunks: Positive integer, the length of the random bang-bang protocol.
    random_state: np.random.RandomState, default None, RandomState(seed=None).

  Returns:
    A uniformly random bang-bang protocol with a certain number of chunks.

  Raises:
    ValueError: If num_chunks is not positive.
  """
  if num_chunks <= 0:
    raise ValueError('num_chunks should be positive, not %d' % num_chunks)
  if random_state is None:
    random_state = np.random.RandomState(None)
  return [
      # NOTE(leeley): random.randint(a, b) returns a random integer [a, b]
      # but np.random.randint(a, b) returns a random integer [a, b).
      # So we need to use np.random.randint(0, 2) to generate numbers in {0, 1}.
      circuit_lib.HamiltonianType(random_state.randint(0, 2))
      for _ in range(num_chunks)]


def get_random_adiabatic_protocol(
    num_chunks, ascending=True, random_state=None):
  """Gets a random adabatic looking protocol of a given number of chunks.

  Takes a linear interpolation from 0 to 1 and then uses that as the probability
  of applying circuit_lib.HamiltonianType.CONSTRAINT.

  Args:
    num_chunks: Positive integer, the length of the random bang-bang protocol.
    ascending: Boolean, whether the probability of binomial sampling increases
        with chunk index. Default True, adiabatic approximation.
    random_state: np.random.RandomState, default None, RandomState(seed=None).

  Returns:
    A random bang-bang protocol with a certain number of chunks.

  Raises:
    ValueError: If num_chunks is not positive.
  """
  if num_chunks <= 0:
    raise ValueError('num_chunks should be positive, not %d' % num_chunks)
  if random_state is None:
    random_state = np.random.RandomState(None)
  if ascending:
    probabilities = np.linspace(0, 1, num_chunks)
  else:
    probabilities = np.linspace(1, 0, num_chunks)
  return [
      circuit_lib.HamiltonianType(random_state.binomial(1, probability))
      for probability in probabilities]


def _apply_changes_to_protocol(bangbang_protocol, changes):
  """Apply changes to a bang-bang protocol.

  Args:
    bangbang_protocol: List of circuit_lib.HamiltonianType describing the
        protocol.
    changes: List of positive integers. Each integer represents the index
        of the protocol that will be changed.

  Returns:
    A copy of the original bang-bang protocol with the corresponding
    circuit_lib.HamiltonianType changed at each index described in changes.

  Raises:
    IndexError: If value is not in the interval [0, len(bangbang_protocol))
  """
  protocol_copy = list(bangbang_protocol)
  protocol_len = len(protocol_copy)
  for index in changes:
    if index < 0 or index >= protocol_len:
      raise IndexError('Each index should be between 0 and %d, not %d'
                       % (protocol_len - 1, index))
    protocol_copy[index] = circuit_lib.switch_hamiltonian_type(
        bangbang_protocol[index])
  return protocol_copy


def _get_all_changes(num_chunks, max_num_flips):
  """Get all changes of Hamming distance up to max_num_flips.

  Args:
    num_chunks: Positive integer, the total number of chunks that can be
        flipped.
    max_num_flips: Positive integer, the maximum number of indices of the
        bang-bang protocol that can be changed (k).

  Returns:
    A generator over all possible changes of Hamming distance k, each described
    by a list of indices of length at most k, where each index corresponds
    to a change.

  Raises:
    ValueError: If 0 < max_num_flips <= num_chunks is not true.
  """
  if max_num_flips <= 0:
    raise ValueError('max_num_flips should be positive, not %d' % max_num_flips)
  if num_chunks < max_num_flips:
    raise ValueError('num_chunks should be at least max_num_flips')
  return itertools.chain.from_iterable(
      itertools.combinations(range(num_chunks), i)
      for i in range(1, max_num_flips + 1)
  )


def get_all_new_protocols(bangbang_protocol, max_num_flips):
  """Gets all new protocols within max_num_flips flips of the current protocol.

  Returns a shuffled generator of all neighbors within max_num_flips flip of
  the bang-bang protocol.

  Args:
    bangbang_protocol: List of circuit_lib.HamiltonianType describing the
        protocol.
    max_num_flips: Positive integer, the maximum number of indices of the
        bang-bang protocol that can be changed.

  Returns:
    A shuffled generator of all neighbours within max_num_flips flips of the
    bang-bang protocol.

  Raises:
    ValueError: If max_num_flips is greater than the number of chunks of the
        protocol.
  """
  if max_num_flips > len(bangbang_protocol):
    raise ValueError(
        'max_num_flips should be less than len(bangbang_protocol), not %d'
        % max_num_flips)
  changes = list(_get_all_changes(len(bangbang_protocol), max_num_flips))
  np.random.shuffle(changes)
  return (_apply_changes_to_protocol(bangbang_protocol, change)
          for change in changes)


def _more_optimal(minimize, new_val, old_val):
  """Returns if the new value is more optimal than the old value.

  Args:
    minimize: Bool, True if we are trying to minimize.
    new_val: Float, the new value that we we want to see if more optimal.
    old_val: Float, the old value that we will compare to new_val.

  Returns:
    Boolean that is true if new_val is more optimal than old_val.
  """
  return new_val < old_val if minimize else new_val > old_val


def _stochastic_descent_epoch(
    circuit, bangbang_protocol, max_num_flips, previous_eval, minimize):
  """One epoch of the stochastic descent process.

  Randomly goes through all neighbors within max_num_flips flips of the
  bang-bang protocol and evaluates them. If one of them performs better, move to
  that protocol, otherwise return the same protocol, signalling a local optimum.

  Args:
    circuit: circuit_lib.BangBangProtocolCircuit, object contains method to
        evaluate the bang-bang protocols.
    bangbang_protocol: The current bang-bang protocol.
    max_num_flips: Positive integer, the maximum number of indices of the
        bang-bang protocol that can be changed.
    previous_eval: Float, the evaluation of the current bang-bang protocol.
    minimize: Bool, True if we want to minimize the expectation.

  Returns:
    current_optimal_protocol: circuit_lib.HamiltonianType list, if
        bangbang_protocol is a max_num_flips local optimum, returns
        bangbang_protocol. Otherwise, returns a new bang-bang protocol from the
        neighbors that performs more optimally.
    current_optimal_eval: Float, the evaluation of the best_protocol.

  Raises:
    ValueError: If max_num_flips is not positive.
  """
  if max_num_flips <= 0:
    raise ValueError('max_num_flips should be positive, not %d' % max_num_flips)

  current_optimal_protocol = bangbang_protocol
  current_optimal_eval = previous_eval

  for new_protocol in get_all_new_protocols(bangbang_protocol, max_num_flips):
    new_eval = circuit.get_constraint_expectation(
        circuit.get_wavefunction(new_protocol))
    if _more_optimal(minimize, new_eval, current_optimal_eval):
      logging.info(
          '%s in the neighbors performs more optimally than %s, %f vs %f',
          circuit_lib.protocol_to_string(new_protocol),
          circuit_lib.protocol_to_string(current_optimal_protocol),
          new_eval,
          current_optimal_eval)
      current_optimal_eval = new_eval
      current_optimal_protocol = new_protocol
      break
  else:
    logging.info(
        'bangbang_protocol %s (%f) is a max_num_flips local optimum.',
        circuit_lib.protocol_to_string(current_optimal_protocol),
        current_optimal_eval)
  return current_optimal_protocol, current_optimal_eval


def stochastic_descent(
    circuit,
    max_num_flips,
    initial_protocol,
    minimize,
    skip_search=False):
  """Finds a locally optimal bang-bang protocol using stochastic descent.

  Iteratively through protocols with up to max_num_flip flips, and moves to a
  neighbor if better protocol is found. When no such neighbor is found, the
  process stops and returns the local optimum.

  Args:
    circuit: circuit_lib.BangBangProtocolCircuit, object contains method to
        evaluate the bang-bang protocols.
    max_num_flips: Positive integer, the maximum number of indices of the
        bang-bang protocol that can be changed.
    initial_protocol: A list of circuit_lib.HamiltonianType, the initial guess
       of bang-bang protocol for stochastic descent.
    minimize: Bool, True if we want to minimize the exepectation.
    skip_search: Bool, whether to skip the stochastic descent search. If True,
      only the initial protocol are evaluated. This is used as a baseline.

  Returns:
    current_optimal_protocol: circuit_lib.HamiltonianType list, a
        locally optimal bang-bang protocol according to stochastic descent.
    current_optimal_eval: Float, the evaluation of the protocol.
    num_epoch: Integer, the number of epoch from initial protocol to optimal
        protocol.

  Raises:
    ValueError: If max_num_flips is not positive.
  """
  if max_num_flips <= 0:
    raise ValueError('max_num_flips should be positive, not %d' % max_num_flips)

  current_optimal_protocol = initial_protocol
  current_optimal_eval = circuit.get_constraint_expectation(
      circuit.get_wavefunction(current_optimal_protocol))
  if skip_search:
    return current_optimal_protocol, current_optimal_eval, 0

  depth = 1
  new_protocol, new_eval = _stochastic_descent_epoch(
      circuit=circuit,
      bangbang_protocol=current_optimal_protocol,
      max_num_flips=max_num_flips,
      previous_eval=current_optimal_eval,
      minimize=minimize)

  while _more_optimal(minimize, new_eval, current_optimal_eval):
    current_optimal_eval = new_eval
    current_optimal_protocol = new_protocol
    depth += 1
    new_protocol, new_eval = _stochastic_descent_epoch(
        circuit=circuit,
        bangbang_protocol=current_optimal_protocol,
        max_num_flips=max_num_flips,
        previous_eval=current_optimal_eval,
        minimize=minimize)
    logging.log_every_n(
        logging.INFO, 'Stochastic Descent Depth %d.', 10, depth)

  return current_optimal_protocol, current_optimal_eval, depth - 1
