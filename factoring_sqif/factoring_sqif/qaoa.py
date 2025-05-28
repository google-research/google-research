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
import collections
from typing import Iterable, Optional, Tuple

import cirq
import matplotlib.pyplot as plt
import numpy as np
import qsimcirq
from recirq.optimize import minimize
import sympy


def get_gamma_layer(hamiltonian, index):
  """Get the gamma layer in QAOA which executes the unitary exp(-i gamma H).

  Since H is of the form $\sum_{i} c_{i} * Z_{i} + \sum_{i, j} w_{ij} Z_{i}
  Z_{j}$,
  `exp(-i gamma H)` can be implemented as product of the individual single qubit
  `Z_{i} ** {gamma * c_{i}}` terms and two qubit `ZZ_{ij} ** {w_{ij} * gamma}`
  terms,
  all of which commute with each other.
  """
  gamma = sympy.Symbol(f"gamma_{index}")
  dense_zz = cirq.DensePauliString("ZZ")
  dense_z = cirq.DensePauliString("Z")
  dense_identity = cirq.DensePauliString("")
  for ps in hamiltonian:
    coeff = ps.coefficient
    dense_ps = ps.with_coefficient(1).gate
    if dense_ps == dense_zz:
      yield cirq.ZZ(*ps.qubits) ** (gamma * coeff)
    elif dense_ps == dense_z:
      yield cirq.Z(*ps.qubits) ** (gamma * coeff)
    elif dense_ps == dense_identity:
      yield []
    else:
      raise ValueError(
          f"Pauli string term {ps} in the hamiltonian should be either "
          "single qubit Z or two qubit ZZ interactions."
      )


def get_beta_layer(qubits, index):
  beta = sympy.Symbol(f"beta_{index}")
  return [cirq.X(q) ** beta for q in qubits]


def generate_qaoa_circuit(
    hamiltonian, p = 1
):
  qubits = hamiltonian.qubits
  qaoa_circuit = cirq.Circuit(
      cirq.H.on_each(*qubits),
      [
          (
              get_gamma_layer(hamiltonian, i),
              cirq.Moment(get_beta_layer(qubits, i)),
          )
          for i in range(p)
      ],
  )
  return qaoa_circuit


def expected_energy_landscape(
    circuit,
    hamiltonian,
    gamma_range = (0, np.pi),
    beta_range = (0, np.pi),
    grid_size = 10,
):
  gamma_pts, beta_pts = np.linspace(*gamma_range, grid_size), np.linspace(
      *beta_range, grid_size
  )
  gamma_sweep = cirq.Points(sympy.Symbol(f"gamma_0"), gamma_pts)
  beta_sweep = cirq.Points(sympy.Symbol(f"beta_0"), beta_pts)
  pauli_obs, coefficients = zip(
      *[(ps.with_coefficient(1), ps.coefficient) for ps in hamiltonian]
  )
  sim = qsimcirq.QSimSimulator(qsimcirq.QSimOptions(cpu_threads=8, verbosity=0))
  samples = sim.simulate_expectation_values_sweep(
      circuit, observables=list(pauli_obs), params=gamma_sweep * beta_sweep
  )
  ret = np.einsum("ij,j->i", samples, coefficients).reshape(-1, grid_size)
  np.testing.assert_array_almost_equal(np.imag(ret), 0)
  return gamma_pts, beta_pts, np.real(ret)


def plot_energy_landspace(
    gamma_pts,
    beta_pts,
    energy,
    ax = None,
):
  show_fig = not ax
  if not ax:
    fig, ax = plt.subplots(1, 1)
  else:
    fig = None
  grid_size = len(gamma_pts)
  assert len(gamma_pts) == len(beta_pts) == energy.shape[0] == energy.shape[1]
  plt.title("Heatmap of QAOA Cost Function Value")
  ax.set_xlabel(r"$\gamma$")
  ax.set_ylabel(r"$\beta$")
  im = ax.imshow(energy)
  tick_values = [*range(0, grid_size, 2)]
  ax.set_xticks(tick_values)
  ax.set_xticklabels(gamma_pts[tick_values].astype("a4").astype("str"))
  ax.set_yticks(tick_values)
  ax.set_yticklabels(beta_pts[tick_values].astype("a4").astype("str"))
  if show_fig:
    fig.colorbar(im, ax=ax)
    fig.show()
  return fig, ax


def sample_bitstrings_for_fixed_params(
    circuit,
    param_resolver,
    h,
    repetitions = 10_000,
):
  sim = qsimcirq.QSimSimulator(qsimcirq.QSimOptions(cpu_threads=8, verbosity=0))
  measurement_circuit = circuit + cirq.Circuit(cirq.measure(h.qubits, key="m"))
  result = sim.run(
      measurement_circuit,
      param_resolver=param_resolver,
      repetitions=repetitions,
  )
  return result.histogram(key="m")


def find_optimal_parameters(
    circuit, hamiltonian
):
  param_names = sorted(cirq.parameter_names(circuit))
  observables = [ps.with_coefficient(1) for ps in hamiltonian]

  def func_to_minimize(x):
    param_resolver = cirq.ParamResolver({k: v for k, v in zip(param_names, x)})
    sim = qsimcirq.QSimSimulator(
        qsimcirq.QSimOptions(cpu_threads=8, verbosity=0)
    )
    result = sim.simulate_expectation_values(
        program=circuit, observables=observables, param_resolver=param_resolver
    )
    assert len(hamiltonian) == len(result)
    ret = sum(ps.coefficient * val for ps, val in zip(hamiltonian, result))
    assert abs(ret.imag) < 1e-5, ret.imag
    return ret.real

  x0 = np.asarray([0.0] * len(param_names))
  # TODO: The paper uses model gradient descent (method="mgd") but it doesn't converge to the
  # optimal parameters , and we need to figure out why.
  result = minimize(func_to_minimize, x0, method="BFGS")
  return cirq.ParamResolver({k: v for k, v in zip(param_names, result.x)})


def sample_low_energy_states(
    hamiltonian, *, p = 1, repetitions = 10_000
):
  circuit = generate_qaoa_circuit(hamiltonian, p=p)
  params = find_optimal_parameters(circuit, hamiltonian)
  states_hist = sample_bitstrings_for_fixed_params(
      circuit, params, hamiltonian, repetitions=repetitions
  )
  bitstrings, frequencies = zip(*states_hist.most_common(len(states_hist)))
  return bitstrings
