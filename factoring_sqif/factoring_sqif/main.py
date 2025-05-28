# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

# pylint: skip-file
import argparse
import logging
import math
import time
from typing import FrozenSet, Set, Tuple

import attrs
import cirq
from factoring_sqif import closest_vector_problem as cvp
from factoring_sqif import hamiltonian, number_theory, qaoa, schnorr
import numpy as np

parser = argparse.ArgumentParser(
    prog="Integer factoring using Schnorr's algorithm + quantum optimization.",
    description=(
        'Implementation of integer factoring algorithm described in '
        'https://arxiv.org/abs/2212.12372'
    ),
)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument(
    '-N', '--number_to_factor', type=int, help='Integer to factor.'
)
group.add_argument(
    '-b', '--bitsize', type=int, help='Bitsize of number to be factored.'
)

parser.add_argument(
    '-l',
    '--lattice_parameter',
    type=int,
    default=1.5,
    help='Lattice parameter.',
)
parser.add_argument(
    '-c',
    '--precision_parameter',
    type=int,
    default=4,
    help='Precision parameter.',
)
parser.add_argument(
    '-s',
    '--seed',
    type=int,
    default=99,
    help='Seed for random number generation.',
)
parser.add_argument(
    '-m',
    '--method',
    choices=['qaoa', 'bruteforce'],
    default='bruteforce',
    help='Method to use for finding ground states of the hamiltonian.',
)
parser.add_argument(
    '-p', '--qaoa_depth', type=int, default=2, help='Depth of qaoa circuit.'
)
parser.add_argument(
    '-NS',
    '--num_samples',
    type=int,
    default=2**15,
    help='Number of low energy states to sample',
)


def qaoa_main():
  num_qubits = 10
  H = hamiltonian.get_example_problem_hamiltonian(num_qubits)
  circuit = qaoa.generate_qaoa_circuit(H, p=1)

  grid_size = 50
  gamma_pts, beta_pts, exp_energy = qaoa.expected_energy_landscape(
      circuit, H, grid_size=grid_size
  )
  qaoa.plot_energy_landspace(gamma_pts, beta_pts, exp_energy)

  index = np.unravel_index(np.argmin(exp_energy, axis=None), exp_energy.shape)
  gamma, beta = gamma_pts[index[0]], beta_pts[index[1]]
  print(f'{gamma=}', f'{beta=}', f'{exp_energy[index]=}')

  param_resolver = cirq.ParamResolver({'gamma_0': gamma, 'beta_0': beta})
  hist = qaoa.sample_bitstrings_for_fixed_params(circuit, param_resolver, H)

  bitstrings, frequencies = zip(*hist.most_common(10))
  print(f'{bitstrings=}', f'{frequencies=}', sep='\n')
  energies = hamiltonian.energy_from_integer_states(H, bitstrings)
  print(f'{energies=}')


@attrs.frozen
class SchnorrFactoringConfig:
  N: int
  lattice_dimension: int
  precision_parameter: int
  linear_eq_system_dimension: int
  num_sr_pairs_to_sample: int
  max_lattice_iterations: int
  seed: int

  @classmethod
  def from_arxiv_paper_defaults(
      cls, N: int, c: int, seed: int, lattice_parameter: int = 1
  ):
    n = number_theory.integer_to_lattice_dimension(N, c=lattice_parameter)
    linear_eq_system_dimension = 1 + 2 * n**2
    num_sr_pairs_to_sample = linear_eq_system_dimension + 1
    max_lattice_iterations = num_sr_pairs_to_sample
    return SchnorrFactoringConfig(
        N=N,
        lattice_dimension=n,
        precision_parameter=c,
        linear_eq_system_dimension=linear_eq_system_dimension,
        num_sr_pairs_to_sample=num_sr_pairs_to_sample,
        max_lattice_iterations=max_lattice_iterations,
        seed=seed,
    )


@attrs.frozen
class SchnorrFactoringResult:
  sr_pairs: FrozenSet[Tuple[int, int]]
  factors: FrozenSet[int]


def solve(
    config: SchnorrFactoringConfig,
    *,
    method: str = 'bruteforce',
    qaoa_depth: int = 1,
    num_samples: int = 1 << 15,
) -> SchnorrFactoringResult:
  logging.info(f'{config}')
  num_samples = min(num_samples, 2**config.lattice_dimension)
  # Algorithm
  pairs: Set[Tuple[int, int]] = set()
  smooth_bound = config.linear_eq_system_dimension - 1
  rs = np.random.RandomState(config.seed)
  curr_iter = 0
  while (
      len(pairs) < config.num_sr_pairs_to_sample
      and curr_iter < config.max_lattice_iterations
  ):
    curr_iter = curr_iter + 1
    # 1. Construct a lattice `B` of dimension nxn+1 and a target vector `t`.
    B, t = cvp.sample(
        config.N, config.lattice_dimension, c=config.precision_parameter, rs=rs
    )
    logging.debug(f'B={str(B)}\n{t=}')

    # 2. Run Babai's Algorithm to find an approximate closest vector to target `t`.
    babai_result = cvp.babai_algorithm(B, t)
    logging.debug(babai_result)
    logging.debug(f'b_op = {np.array(t) - babai_result.residual_vector}')
    logging.debug(
        f'|residual vector|^2 = {np.linalg.norm(babai_result.residual_vector)}'
    )

    # 3. Construct the problem Hamiltonian to find better approximate solutions for cvp.
    qs = cirq.LineQubit.range(config.lattice_dimension)
    H = hamiltonian.hamiltonian_from_babai_result(qs, babai_result)
    logging.debug(f'H={str(H)}')

    # 4. Sample low energy states of the hamiltonian and try to find at least one new SR pair.
    if method == 'bruteforce':
      states = hamiltonian.brute_force_lowest_energy_states(H, num_samples)
    elif method == 'qaoa':
      states = qaoa.sample_low_energy_states(H, p=qaoa_depth)
    else:
      raise ValueError(f'Unknown method: {method}')

    ct = time.time_ns()
    lattice_vectors = hamiltonian.integer_states_to_lattice_vectors(
        states, babai_result
    )
    uv_pairs = hamiltonian.u_v_pairs_from_lattice_vectors(
        lattice_vectors, babai_result.prime_basis
    )
    del lattice_vectors
    sr_pairs = hamiltonian.sr_pairs_from_uv_pairs(
        uv_pairs, config.N, smooth_bound
    )
    elapsed = time.time_ns() - ct

    pairs.update(sr_pairs)
    logging.info(
        f'Filtered {len(sr_pairs)} SR pairs out of {len(uv_pairs)} UV pairs'
        f' using smooth bound {smooth_bound}.(Total'
        f' {len(pairs)}/{config.num_sr_pairs_to_sample} in'
        f' {curr_iter}/{config.max_lattice_iterations} iterations and'
        f' {elapsed*1e-6}ms)'
    )
  logging.info(
      f'Found {len(pairs)}/{config.num_sr_pairs_to_sample} SR pairs in'
      f' {curr_iter} iterations.'
  )

  if len(pairs) < config.num_sr_pairs_to_sample:
    logging.info(
        "Couldn't find enough SR pairs:\n"
        f'Found:{len(pairs)}\n'
        f'Needed: {config.num_sr_pairs_to_sample}\n'
        f'Lattice Dimension: {config.lattice_dimension}\n'
        f'Smooth Bound: {smooth_bound}\n'
        f'N: {config.N}'
    )

  # 5. Use the sr pairs to build the system of equations and generate the candidate factors.
  differences = [
      schnorr.sr_pair_to_differences(u, v, config.N, smooth_bound)
      for (u, v) in pairs
  ]
  differences = np.matrix(differences).T
  logging.info('Explore null space of E')
  factors = set()
  for exponents in schnorr.differences_to_exponents(differences):
    for p in schnorr.exponents_to_candidates(exponents):
      g = math.gcd(p, config.N)
      if g == 1 or g == config.N:
        continue
      if g not in factors:
        logging.info(f'found proper factors {g} and {config.N / g}.')
        assert (
            config.N % g == 0
        ), f'Found proper factor {g} does not divide {config.N}.'
        factors.add(g)
        factors.add(config.N // g)
        return SchnorrFactoringResult(
            sr_pairs=frozenset(pairs), factors=frozenset(factors)
        )
  raise ValueError(
      f"Couldn't find a proper factor for {config.N} using {len(pairs)} SR"
      f' pairs with smooth bound {smooth_bound}. The found SR pairs'
      f' are:\n{pairs}.'
  )


def n_bit_integer(bits: int, seed: int) -> int:
  from sage.arith import misc
  from sage.misc.randstate import set_random_seed

  set_random_seed(seed)
  p = misc.random_prime(2 ** (bits // 2), False, 2 ** (bits // 2 - 1))
  q = misc.random_prime(2 ** (bits // 2), False, 2 ** (bits // 2 - 1))
  return p * q


def run_arxiv_examples():
  N_to_factor = [1961, 48567227, 261980999226229]
  lattice_parameters = [1, 1.5, 1.5]
  cs = [1.5, 4, 4]
  seed = 99
  results = []
  for N, lattice_parameter, c in zip(N_to_factor, lattice_parameters, cs):
    try:
      config = SchnorrFactoringConfig.from_arxiv_paper_defaults(
          N=N, c=c, seed=seed, lattice_parameter=lattice_parameter
      )
      result = solve(config)
      results.append(
          (config.N, tuple(result.factors), config.lattice_dimension)
      )
    except ValueError as e:
      print(f"Couldn't factor {N}. Raised exception:\n{e}")

  for N, factors, num_qubits in results:
    print(f'Found factors {factors} for {N} using {num_qubits} qubits.')


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  args = parser.parse_args()
  N = (
      args.number_to_factor
      if args.number_to_factor
      else n_bit_integer(args.bitsize, args.seed)
  )
  config = SchnorrFactoringConfig.from_arxiv_paper_defaults(
      N=N,
      c=args.precision_parameter,
      seed=args.seed,
      lattice_parameter=args.lattice_parameter,
  )
  try:
    result = solve(
        config,
        method=args.method,
        qaoa_depth=args.qaoa_depth,
        num_samples=args.num_samples,
    )
    print(
        f'Found factors {tuple(result.factors)} for {N} using'
        f' {config.lattice_dimension} qubits via {args.method}.'
    )
  except ValueError as e:
    print(f"Couldn't factor {N}. Raised exception:\n{e}")
