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

"""Utilities for XC functionals."""

EPSILON = 1e-50  # an auxiliary small number to avoid numerical problems


def function_sum(*functions):
  """Generates a function that returns the sum of function outputs.

  Args:
    *functions: List of functions.

  Returns:
    Callable.
  """
  def output_function(*args, **kwargs):
    return sum(func(*args, **kwargs) for func in functions)

  return output_function


def get_hybrid_rsh_params(xc_name):
  """Queries libxc for hybrid coefficients and RSH parameters for given XC.

  Args:
    xc_name: String, name of the XC functional.

  Returns:
    hybrid_coeff: Float, the fraction of exact exchange for custom global
      hybrid functional.
    rsh_params: Tuple of (float, float, float), RSH parameters for custom
      range-separated hybrid functional.
  """
  # NOTE(htm): importing pyscf in the beginning causes issues for colab
  # adhoc import
  from pyscf import dft  # pylint: disable=g-import-not-at-top,import-outside-toplevel

  hybrid_coeff = dft.libxc.hybrid_coeff(xc_name)
  rsh_params = dft.libxc.rsh_coeff(xc_name)
  return hybrid_coeff, rsh_params
