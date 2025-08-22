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

"""A package containing and computing seimological properties.

Derivation of equations for the double couple strain tensor.

Glossary:
DC - Double couple
NED - North-East-Down frame of reference
x1x2x3 - 3D space of the DC where x1 is the rake direction, x3 is fault normal.

The main usage of this package will be through the two functions:
1)  double_couple_strain_tensor_in_ned - returns the strain difference tensor of
  an earthquake at the requested coordinates.
2)  strain_solution_rotated_to_ned - returns the strain difference tensor of
  an earthquake at the requested coordinates, just rotated to the earthquake's
  orientation, not transformed to the correct frame of reference. This is more
  easily interpreted though not physically correct.
An example of usage:
  coordinates = np.array(np.meshgrid(np.linspace(-20, 20, 100),
  np.linspace(-20, 20,100), np.linspace(-20, 20, 100)))
  # coordinates.shape = (3,100,100,100)
  # coordinates should have the shape of (3,...) where coordinates[0] contains
  # the x coordinates, etc.
  strain_tensor = earthquake_properties.strain_solution_rotated_to_ned(strike,
    rake, dip, magnitude, coordinates)
  # strain_tensor.shape = (3,3,100,100,100)

The u_ij (e.g. u_23) element of the strain tensor is a scalar in every
coordinate, we can access it by strain_tensor[i-1, j-1]
(e.g. strain_tensor[1,2]) # strain_tensor[i-1, j-1].shape = (100,100,100)

A usage implementation example:
eq_mag_prediction/seismology/double_couple_strain_tensor_solution.ipynb
"""

import functools
from typing import Optional, Sequence

import numpy as np

from eq_mag_prediction.seismology import strain_fields
from eq_mag_prediction.utilities import geometry

_DYNE_CM_TO_NM = 1e-7  # Moment units converting factor from dyne*cm to N*m


def double_couple_strain_tensor_in_ned(
    strike,
    rake,
    dip,
    magnitude,
    coordinates,
    magnitude_type = None,
):
  """Returns the  strain field of an earthquake.


  Will return a ndarray in the shape (3,3,n,...,m) of the strain field. The
  function requires the earthquake's (EQ's) directionality (3 angles, described
  in ref.) and its magnitude.
  First 2 dimensions ((3,3)) are the 3D strain tensor's elements. last n,...m
  dimensions are the values of the strain elements and have the shape of the
  coordinates given to function.

  Args:
    strike: strike angle of the earthquake. Described in ref.
    rake: rake angle of the earthquake. Described in ref.
    dip: dip angle of the earthquake. Described in ref.
    magnitude: magnitude of the given earthquake.
    coordinates:  ndarray in the shape of (3,n...,m). the first dimension
      represents the x,y,z coordinates respectively.
    magnitude_type: magnitude type. Used for conversion to moment.
  """
  rotation_x1x2x3_to_ned = _rotation_from_x1x2x3_to_ned(strike, rake, dip)
  strain_tensor_rotated_to_ned = strain_solution_rotated_to_ned(
      strike, rake, dip, magnitude, coordinates, magnitude_type
  )
  strain_tensor_ned = _rotate_tensor_frame_of_reference(
      strain_tensor_rotated_to_ned, rotation_x1x2x3_to_ned
  )
  return strain_tensor_ned


def strain_solution_rotated_to_ned(
    strike,
    rake,
    dip,
    magnitude,
    coordinates,
    magnitude_type = None,
):
  """Returns the strain field of an earthquake in NED, tensor itself in x1x2x3.

  Will return a ndarray in the shape (3,3,n,...,m) of the strain field rotated
  to the NED frame of reference. The tensor itself is in the x1x2x3 frame of
  reference, and so is easily interpreted.
  This is not a valid solution, and tensor should be transformed using the
  function rotate_tensor_frame_of_reference().

  Args:
    strike: strike angle of the earthquake. Described in ref.
    rake: rake angle of the earthquake. Described in ref.
    dip: dip angle of the earthquake. Described in ref.
    magnitude: magnitude of the given earthquake.
    coordinates:  ndarray in the shape of (3,n...,m). the first dimension
      represents the x,y,z coordinates respectively.
    magnitude_type: magnitude type. Used for conversion to moment.
  """
  rotation_ned_to_x1x2x3 = _rotation_from_ned_to_x1x2x3(strike, rake, dip)
  coordinates_ned_flattened = _flatten_coordinates(coordinates)
  # -- rotate coordinates to the x1x2x3 frame of reference, eq 19 in ref.
  coors_x1x2x3_flattened = np.matmul(
      rotation_ned_to_x1x2x3, coordinates_ned_flattened
  )
  coordinates_x1x2x3 = _unflatten_coordinates(
      coors_x1x2x3_flattened, coordinates
  )
  strain_tensor_ned = strain_fields.double_couple_strain_tensor(
      coordinates_x1x2x3, _magnitude_to_moment(magnitude, magnitude_type)
  )
  return strain_tensor_ned


def moment_vectors_from_angles(
    strike, rake, dip
):
  """Returns fault normal, strike and rake vectors in NED given the DC angles."""
  z_strike_rotation = geometry.rotation_matrix_about_axis(
      -strike, np.array([0, 0, 1])
  )
  # Eq 11 in ref:
  strike_vector = np.matmul(z_strike_rotation, np.array([1, 0, 0]))

  k_dip_rotation = e3_to_fault_normal_rotation(dip, strike_vector)
  # Eq 12 in ref:
  normal_vector = np.matmul(k_dip_rotation, np.array([0, 0, -1]))

  n_rake_rotation = _strike_to_rake_rotation(rake, normal_vector)
  # Eq 13 in ref:
  rake_vector = np.matmul(n_rake_rotation, strike_vector)

  return normal_vector, strike_vector, rake_vector


def e3_to_fault_normal_rotation(
    dip, strike_vector
):
  """Implements the rotation matrix from eq.

  12 in ref.

  Returns the rotation matrix which aligns the third normal base vector
  e3=[0,0,1] to the fault normal vector.

  Args:
    dip: the dip angle of a fault
    strike_vector: the strike vector of the fault

  Returns:
    A 3x3 np.ndarray of the rotation matrix.
  """
  align_normal_rotation_matrix = geometry.rotation_matrix_about_axis(
      dip, strike_vector
  )
  return align_normal_rotation_matrix


def _strike_to_rake_rotation(
    rake, normal_vector
):
  """Implements the roation matrix from eq. 13 in ref."""
  n_rake_rotation = geometry.rotation_matrix_about_axis(rake, normal_vector)
  return n_rake_rotation


def _align_on_fault_plane(
    e1_on_plane, rake_vector
):
  return geometry.rotation_matrix_between_vectors_in_3d(
      e1_on_plane, rake_vector
  )


def _rotate_tensor_frame_of_reference(
    tensor, rotation
):
  """Rotates a tensor using a given rotation matrix."""
  rotation_inv = np.linalg.inv(rotation)
  tensor_elements_in_back = _moveaxis_of_stacked_tensor(tensor)
  #  Equation 18 in ref:
  rotated_tensor_elements_in_back = np.matmul(
      np.matmul(rotation_inv, tensor_elements_in_back), rotation
  )
  rotated_tensor = _re_moveaxis_of_stacked_tensor(
      rotated_tensor_elements_in_back
  )
  return rotated_tensor


@functools.lru_cache(maxsize=None)
def _rotation_from_ned_to_x1x2x3(
    strike, rake, dip
):
  """Calculates the rotation matrix from NED to x1x2x3 given DC's angles."""
  return np.linalg.inv(_rotation_from_x1x2x3_to_ned(strike, rake, dip))


@functools.lru_cache(maxsize=None)
def _rotation_from_x1x2x3_to_ned(
    strike, rake, dip
):
  """Calculates the rotation matrix from x1x2x3 to NED given DC's angles."""
  _, strike_vector, rake_vector = moment_vectors_from_angles(strike, rake, dip)
  # "V" in equation 14 in ref:
  align_normal_rotation_matrix = e3_to_fault_normal_rotation(dip, strike_vector)
  # rotate to align on the fault plane:
  e1_on_plane = np.matmul(align_normal_rotation_matrix, np.array([1, 0, 0]))
  # "W" ; equation 16 in ref:
  rotation_upon_plane = _align_on_fault_plane(e1_on_plane, rake_vector)
  # "U=WV" ; equation 17 in ref:
  rotation_x1x2x3_to_ned = np.matmul(
      rotation_upon_plane, align_normal_rotation_matrix
  )
  return rotation_x1x2x3_to_ned


def _magnitude_to_moment(
    magnitude, magnitude_type = None
):
  """Computes the moment of an earthquake given its magnitude size and type."""
  magnitude_type = 'MW' if magnitude_type is None else magnitude_type
  if magnitude_type == 'MW':
    # Eq 7.24 in Ammon et. al. https://doi.org/10.1016/C2017-0-03756-4:
    return np.exp(1.5 * magnitude + 16.1) * _DYNE_CM_TO_NM
  else:
    assert (
        False
    ), 'INVALID MAGNITUDE TYPE. Currently, the supported types are MW.'


def _moveaxis_of_stacked_tensor(tensor):
  """Rearranges axis order to be used in matmul.

  As np.matmul applies matrix multiplications on the last two dimensions, and
  the code here is set so first two dimensions indicate the tensor's elements,
  this function will move the axes of the tensor to adapt it for matrix
  multiplication; i.e.: the dimensions of the input tensor (0,1,2...,N) where
  0 and 1 are both of size 3 indicating the 9 tensor elements, the resulting
  tensor will be ordered (2,...,N,0,1).
  e.g.: if tensor elements are in first two: (3,3,150,150,150) move them to last
  two resulting in: (150,150,150,3,3)

  Args:
    tensor: ndarray containing tensor elements. Should be shaped (3,3,n,...,m)

  Returns:
    A strain tensor with rearanges axes, ready to be np.matmul-ed.
  """
  moved_axis_tensor = np.moveaxis(np.moveaxis(tensor, 1, -1), 0, -2)
  return moved_axis_tensor


def _re_moveaxis_of_stacked_tensor(tensor):
  """Undos _moveaxis_of_stacked_tensor to reorder the tensor's axes."""

  organized_tensor = np.moveaxis(np.moveaxis(tensor, -1, 0), -1, 0)
  return organized_tensor


def _flatten_coordinates(coordinates):
  return coordinates.reshape((3, -1))


def _unflatten_coordinates(flattened_coordinates, original_coordinates):
  return flattened_coordinates.reshape(original_coordinates.shape)
