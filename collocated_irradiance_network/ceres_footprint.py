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

"""Code for working with CERES footprints.

Full documentation:
https://eospso.gsfc.nasa.gov/sites/default/files/atbd/atbd-cer-09.pdf
"""

from typing import Union, Tuple

import numpy as np

EARTH_RADIUS_KM = 6367
SATELLITE_HEIGHT_KM = 705
OPTICAL_FOV_CONSTANT_A = 0.65  # See Figure 4.4-1
CENTROID_OFFSET = 0.96  # See bottom of page 19 and Figure 4.4-3


def normalize(
    latitude_degrees, longitude_degrees
):
  """Clamps latitude to [-90, 90] degrees, and shifts longitude to [-180, 180]."""
  normalized_lat_degrees = np.maximum(-90., np.minimum(90., latitude_degrees))

  # Numpy implementation of http://www.cplusplus.com/reference/cmath/remainder/
  def remainder(numer, denom):
    rquot = np.around(numer / denom)
    return numer - rquot * denom

  normalized_lng_degrees = remainder(longitude_degrees, 360)
  return normalized_lat_degrees, normalized_lng_degrees


class Vector3(object):
  """Represents a 3d point."""

  def __init__(self, latitude_degrees,
               longitude_degrees):
    latitude_degrees, longitude_degrees = normalize(
        np.array(latitude_degrees), np.array(longitude_degrees))
    phi = latitude_degrees * (np.pi / 180)
    theta = longitude_degrees * (np.pi / 180)
    cos_phi = np.cos(phi)
    self.x = np.cos(theta) * cos_phi
    self.y = np.sin(theta) * cos_phi
    self.z = np.sin(phi)

  def to_array(self):
    return np.stack([self.x, self.y, self.z], axis=-1).astype(np.float64)


def dot(array1, array2):
  """higher-rank dot product."""
  # np.dot() method is overloaded depending on what type of
  # inputs are passed, so just being clear this is what we're doing.
  return np.sum(array1 * array2, axis=-1, keepdims=True)


def satellite_centered_unit_vectors(goes_lats,
                                    goes_lngs, subsat_lat,
                                    subsat_lng, centroid_lat,
                                    centroid_lng):
  """Returns Satellite-centric unit vectors to a footprint and points in it.

  Args:
    goes_lats: np.array of GOES pixel latitudes.
    goes_lngs: np.array of GOES pixel longitudes.
    subsat_lat: latitude of the point directly beneath the satellite.
    subsat_lng: longitude of the point directly beneath the satellite.
    centroid_lat: latitude of the CERES footprint centroid.
    centroid_lng: longitude of the CERES footprint centroid.

  Returns:
    y_hat_prime, z_hat_prime, y_hat_prime_goes
  """
  x_hat_sat = Vector3(subsat_lat, subsat_lng).to_array()
  x_hat_cen = Vector3(centroid_lat, centroid_lng).to_array()
  x_hat_goes = Vector3(goes_lats, goes_lngs).to_array()
  cos_gamma_goes = dot(x_hat_sat, x_hat_goes)
  cos_gamma_cen = dot(x_hat_sat, x_hat_cen)

  rho_goes = np.sqrt(
      np.square(EARTH_RADIUS_KM + SATELLITE_HEIGHT_KM) +
      np.square(EARTH_RADIUS_KM) - 2 * (EARTH_RADIUS_KM + SATELLITE_HEIGHT_KM) *
      EARTH_RADIUS_KM * cos_gamma_goes)
  rho_cen = np.sqrt(
      np.square(EARTH_RADIUS_KM + SATELLITE_HEIGHT_KM) +
      np.square(EARTH_RADIUS_KM) - 2 *
      (EARTH_RADIUS_KM + SATELLITE_HEIGHT_KM) * EARTH_RADIUS_KM * cos_gamma_cen)

  y_hat_prime = ((EARTH_RADIUS_KM * x_hat_cen) -
                 (EARTH_RADIUS_KM + SATELLITE_HEIGHT_KM) * x_hat_sat) / rho_cen
  y_hat_prime_goes = (
      (EARTH_RADIUS_KM * x_hat_goes) -
      (EARTH_RADIUS_KM + SATELLITE_HEIGHT_KM) * x_hat_sat) / rho_goes

  y_cross_x_hat = np.cross(y_hat_prime, x_hat_sat)

  x_hat_prime = y_cross_x_hat / np.linalg.norm(
      y_cross_x_hat, axis=-1, keepdims=True)

  z_hat_prime = np.cross(x_hat_prime, y_hat_prime)

  return y_hat_prime, z_hat_prime, y_hat_prime_goes


def footprint_internal_coords(y_hat_prime, z_hat_prime, y_hat_prime_goes):
  """Returns delta, beta axis values for given satellite-centric unit vectors.

  As documented in Equation 4.4-6, 4.4-7 and 4.4-8.

  Args:
    y_hat_prime: unit vector pointing from satellite to footprint centroid.
    z_hat_prime: along-scan (delta) axis.
    y_hat_prime_goes: unit vectors pointing from satellite to GOES pixels.

  Returns:
    delta, beta np.arrays (in degrees)
  """

  delta = np.arcsin(dot(y_hat_prime_goes, z_hat_prime))
  z_cross_y = np.cross(z_hat_prime, y_hat_prime_goes)
  z_cross_y_unit = z_cross_y / np.linalg.norm(z_cross_y, axis=-1, keepdims=True)
  beta = np.arcsin(-dot(z_cross_y_unit, y_hat_prime))

  delta = np.rad2deg(delta)
  beta = np.rad2deg(beta)

  # We return the absolute value of beta, because the documentation
  # says the PSF is symmetrical about the along-scan (delta) axis.
  beta = np.abs(beta)

  return delta, beta


def delta_prime_f(beta):
  """Returns the forward boundary of the along-scan optical field of view.

  As in left side of Figure 4.4-1.

  Args:
    beta: cross-scan angle.
  """
  beta = np.abs(beta)
  return np.where(
      beta < OPTICAL_FOV_CONSTANT_A, -OPTICAL_FOV_CONSTANT_A,
      np.where((OPTICAL_FOV_CONSTANT_A <= beta) &
               (beta <= 2 * OPTICAL_FOV_CONSTANT_A),
               -2 * OPTICAL_FOV_CONSTANT_A + beta, np.nan))


def delta_prime_b(beta):
  """Returns the backward boundary of the along-scan optical field of view.

  As in left side of Figure 4.4-1.

  Args:
    beta: cross-scan angle.
  """
  beta = np.abs(beta)
  return np.where(
      beta < OPTICAL_FOV_CONSTANT_A, OPTICAL_FOV_CONSTANT_A,
      np.where((OPTICAL_FOV_CONSTANT_A <= beta) &
               (beta <= 2 * OPTICAL_FOV_CONSTANT_A),
               2 * OPTICAL_FOV_CONSTANT_A - beta, np.nan))


def big_f(ksi):
  """Implements F(ksi) as in equation 4.4-2."""
  a1 = 1.84205
  a2 = -0.22502
  b1 = 1.47034
  b2 = 0.45904
  c1 = 1.98412
  return (1 - (1 + a1 + a2) * np.exp(-c1 * ksi) + np.exp(-6.35465 * ksi) *
          (a1 * np.cos(1.90282 * ksi) + b1 * np.sin(1.90282 * ksi)) +
          np.exp(-4.61598 * ksi) *
          (a2 * np.cos(5.83072 * ksi) + b2 * np.sin(5.83072 * ksi)))


def point_spread_function(delta_prime,
                          beta):
  """Implements the PSF as in equation 4.4-1.

  See summation_weights() and the unittest for a demo of
  how to prepare the inputs for this method,
  and normalize its outputs for a desired power cutoff.

  Important: Note the distinction between delta and delta_prime, this method
  expects delta_prime and not delta.

  Args:
    delta_prime: np.array of along-scan (delta) angles (in degrees), offset by
      CENTROID_OFFSET.
    beta: np.array of cross-scan (beta) angles (in degrees).

  Returns:
    Values for the point spread function at the given input degrees.
  """
  assert len(delta_prime.shape) == 2
  assert len(beta.shape) == 2

  return np.where(
      np.abs(beta) > 2 * OPTICAL_FOV_CONSTANT_A, 0,
      np.where(
          delta_prime < delta_prime_f(beta), 0,
          np.where((delta_prime_f(beta) <= delta_prime) &
                   (delta_prime < delta_prime_b(beta)),
                   big_f(delta_prime - delta_prime_f(beta)),
                   big_f(delta_prime - delta_prime_f(beta)) -
                   big_f(delta_prime - delta_prime_b(beta)))))


def summation_weights(goes_lats, goes_lngs,
                      subsat_lat, subsat_lng, centroid_lat,
                      centroid_lng,
                      vz_rate_of_change):
  """Returns point spread function weights (normalized) for GOES lat/lngs.

  Args:
    goes_lats: np.array of GOES pixel latitudes.
    goes_lngs: np.array of GOES pixel longitudes.
    subsat_lat: latitude of the point directly beneath the satellite.
    subsat_lng: longitude of the point directly beneath the satellite.
    centroid_lat: latitude of the CERES footprint centroid.
    centroid_lng: longitude of the CERES footprint centroid.
    vz_rate_of_change: positive if the scanner field of view is moving away from
      nadir, negative if it's moving towards nadir.
  """
  delta, beta = footprint_internal_coords(*satellite_centered_unit_vectors(
      goes_lats, goes_lngs, subsat_lat, subsat_lng, centroid_lat, centroid_lng))

  if vz_rate_of_change > 0:
    # The documentation says the point spread function given in the document
    # is for when the scanner is moving towards nadir, and it's reversed
    # when moving away from nadir.
    delta *= -1

  delta_prime = delta + CENTROID_OFFSET

  weights = point_spread_function(delta_prime, beta)

  # We normalize assuming the requested set of goes_lats/lngs contains
  # 100% of the desired power cutoff. See the unittest for examples of
  # normalizing to different power cutoff thresholds.
  return np.squeeze(weights / np.sum(weights))
