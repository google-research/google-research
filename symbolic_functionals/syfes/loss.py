# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Loss functions for training density functionals."""
import jax.numpy as jnp
import numpy as onp


def weighted_root_mean_square_deviation(y_pred, y_true, weights, use_jax=True):
  """Computes weighted root mean square deviation (WRMSD) between two arrays.

  WRMSD = sqrt{1 / num_samples * sum_i [weights_i * (y_pred_i - y_true_i)^2]}
  See Eq. 33 of 10.1063/1.4952647

  Args:
    y_pred: Float numpy array, the predicted values.
    y_true: Float numpy array, the true values.
    weights: Float numpy array, the sample weights for taking average.
    use_jax: Boolean, if True, use jax.numpy for calculations, otherwise use
      numpy.

  Returns:
    Float, the WRMSD between y_pred and y_true.
  """
  np = jnp if use_jax else onp
  return np.sqrt(np.average(weights * (y_pred - y_true)**2))


def combine_wrmsd(coeff_1, coeff_2, wrmsd_1, wrmsd_2):
  """Combines two WRMSD value with certain coefficients.

  combined_wrmsd = sqrt(coeff_1 * wrmsd_1^2 + coeff_2 * wrmsd_2^2)

  Args:
    coeff_1: Float, the coefficient for the first WRMSD value.
    coeff_2: Float, the coefficient for the second WRMSD value.
    wrmsd_1: Float, the first WRMSD value to be combined.
    wrmsd_2: Float, the second WRMSD value to be combined.

  Returns:
    Float, the combined WRMSD value.
  """
  return onp.sqrt(coeff_1 * wrmsd_1**2 + coeff_2 * wrmsd_2**2)
