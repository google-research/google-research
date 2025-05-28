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

"""Code to run the circuit."""


import abc
import argparse
import collections
from typing import Generator, List, Sequence, Tuple, Union, cast

import cirq
import cirq_google
import numpy as np

from qsp_quantum_metrology import qsp_types


Fsim_angle = argparse.Namespace(
    theta=1e-2,
    varphi=0.1,
    chi=0.5,
)


class CircuitRunner(abc.ABC):

  @abc.abstractmethod
  def run(self, circuit, repetitions):
    pass


def build_circuit(
    deg,
    omega,
    circuit_type,
    qubits,
    measure = False,
):
  """Builds quantum circuit for measuring p_x or p_y."""
  qr = qubits
  # initial state preparation, "z" = |00>, "x" = |01>+|10>, "y" = |01>+1j|10>
  if circuit_type != "z":
    yield [cirq.ops.H(qri) for qri in qr]
    if circuit_type == "y":
      yield cirq.ops.S(qr[0])
    yield cirq.ops.Z(qr[1])
    yield cirq.ops.CZ(*qr)
    yield cirq.ops.H(qr[1])
  # periodic application of Fsim and Rz
  for ii in range(deg):
    angles = [Fsim_angle.theta, Fsim_angle.varphi, Fsim_angle.chi]
    yield cirq.ops.PhasedFSimGate(*angles).on(*qr).with_tags(
        cirq_google.PhysicalZTag()
    )
    yield cirq.ops.Rz(rads=-2 * omega).on(qr[0])
  if measure:
    for ii, qri in enumerate(qr):
      yield cirq.measure(qri, key=f"q{ii}")


def get_counts(result):
  """A helper function for counting the frequency."""
  # ii[1:] to skip the prefix 'q'
  qubit_order = [int(ii[1:]) for ii in list(result.columns)]
  res = result.values[:, np.argsort(qubit_order)]
  res_str = ["".join(str(jj) for jj in ii) for ii in res]
  return collections.Counter(res_str)


def run_and_count_prob(
    qc,
    meas_shots,
    circuit_runner,
):
  res = circuit_runner.run(qc, repetitions=meas_shots)
  counts = get_counts(res.data)
  qbit_label_list = ["00", "01", "10", "11"]
  prob_list = np.zeros((len(qbit_label_list), 1), dtype=float)
  for ii, labelq in enumerate(qbit_label_list):
    if labelq in counts.keys():
      prob_list[ii] = counts[labelq] / meas_shots
  # prob = measure 10, qubit order has been sorted from top to bottom
  return prob_list


def unpack_batch_result(
    results,
):
  # Given List[cirq.Circuit], sampler.run_batch returns a nested list
  # res = List[List[cirq.Result]] len(res) = n_qcs, len(res[0]) = 1
  qbit_label_list = ["00", "01", "10", "11"]
  n_res = len(results)
  probs = np.zeros(
      (len(qbit_label_list), n_res), dtype=float
  )  # each column is a prob vector
  for ii, res in enumerate(results):
    counts = get_counts(res[0].data)
    meas_shots = sum(counts.values())
    for jj, labelq in enumerate(qbit_label_list):
      if labelq in counts.keys():
        probs[jj, ii] = counts[labelq] / meas_shots
  return probs


def experimental_circuit_samples_omega(
    omega,
    meas_shots,
    deg,
    qubits,
    circuit_runner,
    correct_confusion_matrix = False,
):
  """Measures circuit with a certain omega."""
  qc_x = build_circuit(deg, omega, "x", qubits, measure=True)
  qc_y = build_circuit(deg, omega, "y", qubits, measure=True)
  # turn qc_generators to circuits
  qc_x = cirq.Circuit(qc_x)
  qc_y = cirq.Circuit(qc_y)
  px_list = run_and_count_prob(qc_x, meas_shots, circuit_runner)
  py_list = run_and_count_prob(qc_y, meas_shots, circuit_runner)
  px = px_list[2]
  py = py_list[2]
  recon_h_from_mat = px - 0.5 + 1j * (py - 0.5)
  if correct_confusion_matrix:
    confusion_matrix = confusion_matrix_circuit_samples(
        qubits, circuit_runner, meas_shots
    )
    # (conf_mat)^T @ (prob_orig) = prob_meas
    confusion_matrix_T = confusion_matrix.transpose()  # pylint:disable=invalid-name
    px_list = np.linalg.solve(confusion_matrix_T, px_list)
    py_list = np.linalg.solve(confusion_matrix_T, py_list)
    qx = px_list[2]
    qy = py_list[2]
    recon_h_from_mat_q = qx - 0.5 + 1j * (qy - 0.5)
    return recon_h_from_mat, recon_h_from_mat_q
  else:
    return recon_h_from_mat


def experimental_circuit_samples_get_probs(
    meas_shots,
    deg,
    qubits,
    sampler,
):
  """Samples on equally spaced omegas and returns probability vectors."""
  omegas = np.pi / (2 * deg - 1) * np.array(range(2 * deg - 1))
  # list of circuits x, y, x, y, ...
  qcs = []
  for omega in omegas:
    for qc_type in ["x", "y"]:
      qc = build_circuit(deg, omega, qc_type, qubits, measure=True)
      qc = cirq.Circuit(qc)
      qcs.append(qc)
  results = sampler.run_batch(programs=qcs, repetitions=meas_shots)
  probs = unpack_batch_result(results)
  return probs


def get_correct_confusion_matrix(
    probs,
    confusion_matrix,
):
  """Corrects the confusion matrix."""
  confusion_matrix_T = confusion_matrix.transpose()  # pylint:disable=invalid-name
  probs_q = np.linalg.solve(confusion_matrix_T, probs)
  return probs_q


def recon_h_from_mat_with_probs(
    probs,
):
  px = probs[2, ::2]
  py = probs[2, 1::2]
  recon_h_from_mat = np.zeros(probs.shape[1] // 2, dtype=complex)
  recon_h_from_mat = px - 0.5 + 1j * (py - 0.5)
  return recon_h_from_mat


def experimental_circuit_samples(
    meas_shots,
    deg,
    qubits,
    sampler,
):
  """Sample on equally spaced omegas."""
  probs = experimental_circuit_samples_get_probs(
      meas_shots, deg, qubits, sampler
  )
  confusion_matrix = confusion_matrix_circuit_samples(
      qubits, sampler, meas_shots
  )
  probs_q = get_correct_confusion_matrix(probs, confusion_matrix)
  recon_h_from_mat = recon_h_from_mat_with_probs(probs_q)
  recon_h_from_mat_wo_correction = recon_h_from_mat_with_probs(probs)
  return (
      recon_h_from_mat,
      recon_h_from_mat_wo_correction,
      probs,
      confusion_matrix,
  )


def build_confusion_matrix_circuits(
    qubits
):
  """Makes circuits to measure a pair confusion matrix.

  Preps qubits in the lexicographic order: 00, 01, 10, 11

  Args:
    qubits: The pair of qubits for which to build the confusion matrix circuits.

  Returns:
    The list of circuits.
  """
  q0, q1 = qubits
  prep_moments = [
      cirq.Moment(cirq.X(q) for q in qs) for qs in [(), (q1,), (q0,), (q0, q1)]
  ]
  return [
      cirq.Circuit(
          prep_moment,
          cirq.measure(*qubits),
      )
      for prep_moment in prep_moments
  ]


def get_joint_counts(
    result, qubits
):
  meas_gate = cast(cirq.MeasurementGate, cirq.measure(*qubits).gate)
  meas_key = meas_gate.key
  meas_array = result.measurements[meas_key]
  num_meas_qubits = len(qubits)
  place_values = np.power(2, np.arange(num_meas_qubits - 1, -1, -1))
  int_meas_array = np.sum(meas_array * place_values[np.newaxis, :], axis=1)
  num_outcomes = 2**num_meas_qubits
  return tuple(np.bincount(int_meas_array, minlength=num_outcomes))


def _normalize_axis_sum(a, axis):
  """Normalizes the array so that the sum over an axis equals 1.

  Do the normalization by dividing elements by a quantity that only depends
  on the position along the specified axis.

  Args:
    a: The array to normalize.
    axis: The axis along which the elements are summed,

  Returns:
    The normalized array.
  """
  return a / np.expand_dims(a.sum(axis=axis), axis=axis)


def confusion_matrix_circuit_samples(
    qubits,
    sampler,
    meas_shots,
):
  """Measures the confusion matrix for a pair of qubits.

  The confusion matrix is the matrix:

      ⎡ Pr(00|00) Pr(01|00) Pr(10|00) Pr(11|00) ⎤
      ⎢ Pr(00|01) Pr(01|01) Pr(10|01) Pr(11|01) ⎥
      ⎢ Pr(00|10) Pr(01|10) Pr(10|10) Pr(11|10) ⎥
      ⎣ Pr(00|11) Pr(01|11) Pr(10|11) Pr(11|11) ⎦

  Args:
    qubits: The qubits to use for measurement.
    sampler: The circuit sampler to use.
    meas_shots: The number of repeated measurements.

  Returns:
    The measurements from the confusion matrix.
  """
  confusion_matrix_circuits = build_confusion_matrix_circuits(qubits)
  batch_results = sampler.run_batch(
      programs=confusion_matrix_circuits,
      repetitions=meas_shots,
  )
  counts = [
      get_joint_counts(sweep_results[0], qubits)
      for sweep_results in batch_results
  ]
  confusion_matrix = np.array(counts)
  return _normalize_axis_sum(confusion_matrix, axis=-1)
