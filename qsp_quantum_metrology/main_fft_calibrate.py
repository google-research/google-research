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

"""Build the QSP circuit and run it on an example."""

from typing import Sequence

from absl import app
from absl import flags
import cirq

from qsp_quantum_metrology import circuit_generation_test
from qsp_quantum_metrology import fft_analysis


_DEG = flags.DEFINE_integer("deg", 10, "Number of degrees of freedom.")
_MEAS_SHOTS = flags.DEFINE_integer(
    "meas_shots", 10_000, "Number of measurements."
)


def get_ma_length(deg):
  # a hyperparameter
  return (deg - 1) // 3 if deg < 25 else 8


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # parameters
  deg = _DEG.value
  meas_shots = _MEAS_SHOTS.value

  qubits = cirq.LineQubit.range(2)
  sampler = cirq.Simulator()

  samples_outputs = circuit_generation_test.experimental_circuit_samples(
      meas_shots, deg, qubits, sampler
  )
  recon_h_from_exp, recon_h_from_exp_wo_correction, probs, confusion_matrix = (
      samples_outputs
  )
  print(recon_h_from_exp)
  print(recon_h_from_exp_wo_correction)
  print(probs)
  print(confusion_matrix)
  moving_average_length = get_ma_length(deg)
  unwrapped_output = fft_analysis.angle_unwrap_wpa(
      deg, recon_h_from_exp, moving_average_length=moving_average_length
  )
  theta_estimate, varphi_estimate, (alpha_fidelity, discard_zero_freq_mode) = (
      unwrapped_output
  )
  del discard_zero_freq_mode  # Unused here, but preserved for readability.
  print(
      (
          f"orig\ttheta = {circuit_generation_test.Fsim_angle.theta}"
          f"\t\t\tvarphi = {circuit_generation_test.Fsim_angle.varphi}"
          "\t\t\talpha = 1"
      )
  )
  print(
      f"w corr\ttheta = {theta_estimate}\tvarphi = {varphi_estimate}\talpha ="
      f" {alpha_fidelity}"
  )


if __name__ == "__main__":
  app.run(main)
