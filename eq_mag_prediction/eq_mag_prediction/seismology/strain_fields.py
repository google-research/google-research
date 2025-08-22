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

"""A package containing strain solution for an earthquake.

The derivation of the solution implemented here is based on "Foundation of
modern global seismology by Ammon et. al. 2021, doi:
https://doi.org/10.1016/C2017-0-03756-4"
"""

from typing import Sequence, Tuple, Optional
import numpy as np

# Typical properties of southern california depth ~5km (Shearer Intro. to Seis.)
_C_S = 3500  # m/s shear wave velocity
_C_P = 6000  # m/s pressure wave velocity
_RHO = 2700  # kg/m^3 ground density
# the Lamé coefficients (https://en.wikipedia.org/wiki/Lam%C3%A9_parameters)
_ELASTIC_LAMBDA_DEFAULT = _RHO * (_C_P**2 - 2 * _C_S**2)
_ELASTIC_MU_DEFAULT = _RHO * (_C_S**2)


def youngs_modulus():
  """Return the Youngs modulus as cimputed from Lame parameters."""

  # Calculation follows conversion table at bottom of the page:
  # https://en.wikipedia.org/wiki/Elastic_modulus
  return (
      _ELASTIC_MU_DEFAULT
      * (3 * _ELASTIC_LAMBDA_DEFAULT + 2 * _ELASTIC_MU_DEFAULT)
      / (_ELASTIC_LAMBDA_DEFAULT + _ELASTIC_MU_DEFAULT)
  )


def elastic_gamma(elastic_l, elastic_m):
  """Returns capital Gamma as in Ammon et al. eq. 17.14."""
  return (elastic_l + elastic_m) / (elastic_l + 2 * elastic_m)


def _elastic_functions_preparation(
    coordinates,
    elastic_mu = None,
    elastic_l = None,
):
  """Prepare a prefactor that repeats multiple times."""
  (x1, x2, x3) = coordinates
  r = np.sqrt(x1**2 + x2**2 + x3**2)

  mu = _ELASTIC_MU_DEFAULT if elastic_mu is None else elastic_mu
  l = _ELASTIC_LAMBDA_DEFAULT if elastic_l is None else elastic_l
  # pylint: disable=invalid-name
  # Following 17.1.3:
  # naming following conventions in book referenced above
  G = elastic_gamma(l, mu)
  # pylint: enable=invalid-name

  return x1, x2, x3, r, mu, l, G


def double_couple_displacement_field(
    coordinates,
    moment,
    elastic_mu = None,
    elastic_l = None,
):
  """Returns the static displacement fields of a double couple source."""

  # pylint: disable=invalid-name, for the moment.
  # following convention of capital Gamma
  x1, x2, x3, r, mu, _, G = _elastic_functions_preparation(
      coordinates, elastic_mu, elastic_l
  )
  prefactor_M_over_4pimu = moment / (4 * np.pi * mu)
  # pylint: enable=invalid-name

  u1 = (
      prefactor_M_over_4pimu
      * (x2 / r**3)
      * (-1 - G * (1 - 3 * x1**2 / r**2))
  )
  u2 = (
      prefactor_M_over_4pimu
      * (x1 / r**3)
      * (1 - G * (1 - 3 * x2**2 / r**2))
  )
  u3 = prefactor_M_over_4pimu * (1 / r**5) * (3 * G * (x1 * x2 * x3))

  return u1, u2, u3


def _double_couple_u11(
    coordinates,
    moment,
    elastic_mu = None,
    elastic_l = None,
):
  """Returns the u11 element of the strain tensor of double couple source."""
  # g is conventionally referred to as capital Gamma
  x1, x2, x3, r, mu, _, g = _elastic_functions_preparation(
      coordinates, elastic_mu, elastic_l
  )

  prefactor = -(3 * moment) / (4 * np.pi * mu)
  numerator = (x1 * x2) * (
      ((1 + 2 * g) * x2**2 - (3 * g - 1) * (x2**2 + x3**2))
  )
  return prefactor * numerator / (r**7)


def _double_couple_u12(
    coordinates,
    moment,
    elastic_mu = None,
    elastic_l = None,
):
  """Returns the u12 element of the strain tensor of double couple source."""
  # g is conventionally referred to as capital Gamma
  x1, x2, x3, r, mu, _, g = _elastic_functions_preparation(
      coordinates, elastic_mu, elastic_l
  )

  prefactor = moment / (4 * np.pi * mu)
  x1_quartic_prefactor = 1 + 2 * g
  x1_cubic_prefactor = -(1 + 11 * g) * x2**2 + (2 + g) * x3**2
  numerator = (
      x1_quartic_prefactor * x1**4
      + x1_cubic_prefactor * x1**2
      + (g - 1) * (2 * x2**4 + (x2**2) * (x3**2) - x3**4)
  )
  return prefactor * numerator / (r**7)


def _double_couple_u13(
    coordinates,
    moment,
    elastic_mu = None,
    elastic_l = None,
):
  """Returns the u13 element of the strain tensor of double couple source."""
  # g is conventionally referred to as capital Gamma
  x1, x2, x3, r, mu, _, g = _elastic_functions_preparation(
      coordinates, elastic_mu, elastic_l
  )

  prefactor = 3 * moment / (4 * np.pi * mu)
  numerator = x2 * x3 * (-(1 + 4 * g) * x1**2 + (g - 1) * (x2**2 + x3**2))
  return prefactor * numerator / (r**7)


def _double_couple_u21(
    coordinates,
    moment,
    elastic_mu = None,
    elastic_l = None,
):
  """Returns the u21 element of the strain tensor of double couple source."""
  # g is conventionally referred to as capital Gamma
  x1, x2, x3, r, mu, _, g = _elastic_functions_preparation(
      coordinates, elastic_mu, elastic_l
  )

  prefactor = moment / (4 * np.pi * mu)
  x1_quartic_prefactor = 2 * (g - 1)
  x2x3_polynom_prefactor = (1 + 2 * g) * x2**2 - (g - 1) * x3**2
  x1_quadratic_prefactor = (1 + 11 * g) * x2**2 - (g - 1) * x3**2

  numerator = (
      x1_quartic_prefactor * x1**4
      + x2x3_polynom_prefactor * (x2**2 + x3**2)
      - x1_quadratic_prefactor * x1**2
  )
  return prefactor * numerator / (r**7)


def _double_couple_u22(
    coordinates,
    moment,
    elastic_mu = None,
    elastic_l = None,
):
  """Returns the u22 element of the strain tensor of double couple source."""
  # g is conventionally referred to as capital Gamma
  x1, x2, x3, r, mu, _, g = _elastic_functions_preparation(
      coordinates, elastic_mu, elastic_l
  )

  prefactor = -3 * moment / (4 * np.pi * mu)
  numerator = (
      x1
      * x2
      * ((1 - 3 * g) * x1**2 + (1 + 2 * g) * x2**2 + (1 - 3 * g) * x3**2)
  )
  return prefactor * numerator / (r**7)


def _double_couple_u23(
    coordinates,
    moment,
    elastic_mu = None,
    elastic_l = None,
):
  """Returns the u23 element of the strain tensor of double couple source."""
  # g is conventionally referred to as capital Gamma
  x1, x2, x3, r, mu, _, g = _elastic_functions_preparation(
      coordinates, elastic_mu, elastic_l
  )

  prefactor = 3 * moment / (4 * np.pi * mu)
  numerator = (
      x1 * x3 * ((g - 1) * x1**2 - (1 + 4 * g) * x2**2 + (g - 1) * x3**2)
  )
  return prefactor * numerator / (r**7)


def _double_couple_u31(
    coordinates,
    moment,
    elastic_mu = None,
    elastic_l = None,
):
  """Returns the u31 element of the strain tensor of double couple source."""
  # g is conventionally referred to as capital Gamma
  x1, x2, x3, r, mu, _, g = _elastic_functions_preparation(
      coordinates, elastic_mu, elastic_l
  )

  prefactor = 3 * moment / (4 * np.pi * mu)
  numerator = g * x2 * x3 * (-4 * x1**2 + x2**2 + x3**2)
  return prefactor * numerator / (r**7)


def _double_couple_u32(
    coordinates,
    moment,
    elastic_mu = None,
    elastic_l = None,
):
  """Returns the u32 element of the strain tensor of double couple source."""
  # g is conventionally referred to as capital Gamma
  x1, x2, x3, r, mu, _, g = _elastic_functions_preparation(
      coordinates, elastic_mu, elastic_l
  )

  prefactor = 3 * moment / (4 * np.pi * mu)
  numerator = g * x1 * x3 * (x1**2 - 4 * x2**2 - x3**2)
  return prefactor * numerator / (r**7)


def _double_couple_u33(
    coordinates,
    moment,
    elastic_mu = None,
    elastic_l = None,
):
  """Returns the u32 element of the strain tensor of double couple source."""
  # g is conventionally referred to as capital Gamma
  x1, x2, x3, r, mu, _, g = _elastic_functions_preparation(
      coordinates, elastic_mu, elastic_l
  )

  prefactor = 3 * moment / (4 * np.pi * mu)
  numerator = g * x1 * x2 * (x1**2 + x2**2 - 4 * x3**2)
  return prefactor * numerator / (r**7)


def double_couple_strain_tensor(
    coordinates,
    moment,
    elastic_mu = None,
    elastic_l = None,
):
  """Returns  double couple strain tensor in its eigenspace frame of reference.

  Args:
    coordinates: A sequence of length 3 (x,y,z) of ndarrays  indicating the
      coordinates requested.
    moment: moment of the requested earthquake.
    elastic_mu: the shear modulus (1st Lame coefficient).
    elastic_l:  the 2nd Lame coefficient.

  Returns:
  The strain tensor is an ndarray in the shape (3,3,n,...,m). First 2 dimensions
  ((3,3)) are the 3D strain tensor's elements. last n,...m dimensions are the
  values of the strain per coordinate and have the shape of the coordinates
  given to function as input.
  """
  u11 = _double_couple_u11(coordinates, moment, elastic_mu, elastic_l)
  u12 = _double_couple_u12(coordinates, moment, elastic_mu, elastic_l)
  u13 = _double_couple_u13(coordinates, moment, elastic_mu, elastic_l)
  u21 = _double_couple_u21(coordinates, moment, elastic_mu, elastic_l)
  u22 = _double_couple_u22(coordinates, moment, elastic_mu, elastic_l)
  u23 = _double_couple_u23(coordinates, moment, elastic_mu, elastic_l)
  u31 = _double_couple_u31(coordinates, moment, elastic_mu, elastic_l)
  u32 = _double_couple_u32(coordinates, moment, elastic_mu, elastic_l)
  u33 = _double_couple_u33(coordinates, moment, elastic_mu, elastic_l)

  strain_tensor = np.array([
      [u11, (u12 + u21) / 2, (u31 + u13) / 2],
      [(u12 + u21) / 2, u22, (u32 + u23) / 2],
      [(u31 + u13) / 2, (u32 + u23) / 2, u33],
  ])

  return strain_tensor
