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

"""Library of supporting functions."""


from typing import Callable, Tuple
import numpy as np
import scipy.linalg as la
from qsp_quantum_metrology import qsp_types


def angle_unwrap_wpa(
    deg,
    recon_h_from_exp,
    moving_average_length = 1,
    threshold = 0.975,
):
  """Unwraps angles by using weighted phase average (WPA) estimate.

  moving_average_length == 1:
      Kay's WPA estimate, Ref. Steven Kay. A fast and accurate single frequency
      estimator.
      IEEE Transactions on Acoustics, Speech, and Signal Processing,
      37(12):1987-1990, 1989.
  moving_average_length > 1:
      FFT-based WPA (FWPA) estimate, Ref. Zhou Shen and Rongfang Liu. Efficient
      and accurate frequency
      estimator under low SNR by phase unwrapping. Mathematical Problems in
      Engineering, 2019, 2019.

  Args:
    deg: Number of degrees of freedom.
    recon_h_from_exp:
    moving_average_length:
    threshold: The value of fidelity below which a correction is applied.

  Returns:
    Tuple containing the estimates of the unwrapped angles.
  """
  # coarse search
  max_idx = np.argmax(np.abs(recon_h_from_exp))
  omega_max = np.pi / (2 * deg - 1) * max_idx

  recon_c = np.fft.fft(recon_h_from_exp)
  recon_c_pos_k = 1 / (2 * deg - 1) * recon_c[:deg]

  mod_recon_c = np.abs(recon_c_pos_k)
  alpha_fidelity = 1 - 2 * np.sqrt(2) * (
      mod_recon_c[0] - np.mean(mod_recon_c[1:])
  )

  # fidelity is too small, not coherent and impose correction
  if alpha_fidelity < threshold:
    discard_zero_freq_mode = True
    mod_recon_c = mod_recon_c[1:] / alpha_fidelity
  else:
    discard_zero_freq_mode = False
  theta_estimate = np.mean(mod_recon_c)

  revised_recon_c = np.zeros(deg, dtype=complex)
  for ii, cc in enumerate(recon_c_pos_k):
    revised_recon_c[ii] = np.exp(1j * (2 * ii + 1) * omega_max) * cc

  if discard_zero_freq_mode:
    init_mode = 1
    revised_recon_c = revised_recon_c[1:]
  else:
    init_mode = 0
  num_iter = (deg - init_mode) % moving_average_length + 1
  num_ma_res = (deg - init_mode) // moving_average_length
  varphi_res_iter = np.zeros(num_iter, dtype=float)

  wt_covariance_pha_wo_prec = la.toeplitz([2, -1] + [0] * (num_ma_res - 3))
  one_arr = np.ones(num_ma_res - 1)
  covariance_arr = np.linalg.solve(wt_covariance_pha_wo_prec, one_arr)
  for ii in range(num_iter):
    ma_revised_c = np.zeros(num_ma_res, dtype=complex)
    pha_ma_revised_c = np.zeros(num_ma_res - 1, dtype=float)
    # moving average filter
    for jj in range(num_ma_res):
      ma_revised_c[jj] = np.sum(
          revised_recon_c[
              jj * moving_average_length
              + ii : (jj + 1) * moving_average_length
              + ii
          ]
      )
    # phase difference computation
    for jj in range(num_ma_res - 1):
      pha_ma_revised_c[jj] = np.angle(
          ma_revised_c[jj] * np.conjugate(ma_revised_c[jj + 1])
      )
    varphi_res_iter[ii] = (
        1
        / (2 * moving_average_length)
        * covariance_arr
        @ pha_ma_revised_c
        / (covariance_arr @ one_arr)
    )

  varphi_estimate = omega_max + np.mean(varphi_res_iter)

  return (
      theta_estimate,
      varphi_estimate,
      (alpha_fidelity, discard_zero_freq_mode),
  )


def refine_theta_parabolic_peak(
    request_recon_h,
    varphi_estimate,
    deg,
    n_pts = 3,
    alpha_fidelity = 1,
):
  """Refines the estimate to theta by fitting the peak with a parabola."""
  if n_pts < 3:
    raise ValueError(
        f"The number of sample points {n_pts} is not enough to fit a parabola"
    )
  half_window_width = np.pi / (2 * deg)
  angles = np.linspace(
      varphi_estimate - half_window_width,
      varphi_estimate + half_window_width,
      n_pts,
  )
  vals = np.array([request_recon_h(omega) for omega in angles], dtype=complex)
  # impose correction when the circuit error is not coherent
  vals = (vals + (1 - alpha_fidelity) * (1 + 1j) / 4) / alpha_fidelity
  p = np.polyfit(angles, np.abs(vals), 2)
  if p[0] >= 0:
    theta_estimate_mle = -np.inf
    varphi_estimate_mle = -np.inf
    return (theta_estimate_mle, varphi_estimate_mle), (angles, vals)
  else:
    peak_loc = -p[1] / (2 * p[0])
    peak_val = p[2] - p[1] ** 2 / (4 * p[0])
    theta_estimate_mle = peak_val / deg
    varphi_estimate_mle = peak_loc
    return (theta_estimate_mle, varphi_estimate_mle), (angles, vals)
