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

"""Tests for ceres_footprint."""

import enum

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from collocated_irradiance_network import ceres_footprint


class PowerCutoff(enum.Enum):
  """Angles at which X% of the power is contained, from Table 4.4.2."""
  HALF = (-0.88, 0.52, 1.08)
  NINETY_FIVE = (-1.25, 1.35, 1.27)

  def __init__(self, delta_min, delta_max, beta_max):
    self.delta_min = delta_min
    self.delta_max = delta_max
    self.beta_max = beta_max


def distance_between(lat1_degrees, lng1_degrees,
                     lat2_degrees,
                     lng2_degrees):
  """Returns array of element-wise earth-central angle (radians) b/w points.

  Args:
    lat1_degrees: Latitudes of first points.
    lng1_degrees: Longitudes of first points.
    lat2_degrees: Latitudes of second points.
    lng2_degrees: Longitudes of second points.
  """
  lat1_degrees, lng1_degrees = ceres_footprint.normalize(
      lat1_degrees, lng1_degrees)
  lat2_degrees, lng2_degrees = ceres_footprint.normalize(
      lat2_degrees, lng2_degrees)

  lats1 = np.deg2rad(lat1_degrees)
  lngs1 = np.deg2rad(lng1_degrees)
  lats2 = np.deg2rad(lat2_degrees)
  lngs2 = np.deg2rad(lng2_degrees)
  dlats = np.sin(0.5 * (lats2 - lats1))
  dlngs = np.sin(0.5 * (lngs2 - lngs1))
  x = dlats * dlats + dlngs * dlngs * np.cos(lats1) * np.cos(lats2)
  return 2 * np.arcsin(np.sqrt(np.minimum(1.0, x)))


def distance_km(lat1_degrees, lng1_degrees, lat2_degrees, lng2_degrees):
  return distance_between(lat1_degrees, lng1_degrees, lat2_degrees,
                          lng2_degrees) * ceres_footprint.EARTH_RADIUS_KM


class CeresFootprintTest(parameterized.TestCase):

  def _assert_almost_equal_km(self, expected_km, actual_km):
    # Discretization artifacts can introduce a small amount of error
    # in the viewing geometry (that should not matter to models
    # since the edges of the footprint are severely downweighted anyway)
    # so here we just check we're within a threshold of a few km of the
    # values from table 4.4-2.
    diff_km = np.abs(expected_km - actual_km)
    self.assertLess(diff_km, 4,
                    '%f and %f are not close enough' % (expected_km, actual_km))

  @parameterized.parameters(
      (0.01, PowerCutoff.HALF, 0.0, 17, 27),  # Row group 2
      (0.01, PowerCutoff.NINETY_FIVE, 0.0, 32, 31),  # Row group 3
      (12.22, PowerCutoff.NINETY_FIVE, 1357.7, 212, 71),  # Row group 4
      (14.58, PowerCutoff.NINETY_FIVE, 1621.1, 328, 82),  # Row group 5
  )
  def test_table442(self, gamma_degrees, power_cutoff, centroid_ell,
                    d_ell_delta, d_ell_beta):
    """Confirms viewing geometry of various footprints.

    Args:
      gamma_degrees: Earth-central angle between footprint and satellite.
      power_cutoff: 50% or 95% power cutoff thresholds.
      centroid_ell: Italicized 'l' (4th column in table 4.4-2): earth-surface
        distance (km) between footprint and sub-satellite point.
      d_ell_delta: delta 'l' (5th column in table 4.4-2): earth-surface length
        (km) of the footprint over the along-scan (delta) axis.
      d_ell_beta: delta 'l' (5th column in table 4.4-2): earth-surface length
        (km) of the footprint over the cross-scan (beta) axis.
    """
    num_bins = 801
    edge_degrees = d_ell_delta / 50
    centroid_lat, centroid_lng = 0., gamma_degrees
    sub_lat, sub_lng = 0., 0.

    # It's nice to have your latitude linspace start from +edge_degrees
    # and move towards -edge_degrees so that north is up in plots you make.
    plot_lats = np.linspace(centroid_lat + edge_degrees,
                            centroid_lat - edge_degrees, num_bins)
    plot_lngs = np.linspace(centroid_lng - edge_degrees,
                            centroid_lng + edge_degrees, num_bins)
    # The order here is important,
    # np.meshgrid expects as input (x, y) not (row, col) convention.
    mesh_lngs, mesh_lats = np.meshgrid(plot_lngs, plot_lats)

    delta, beta = ceres_footprint.footprint_internal_coords(
        *ceres_footprint.satellite_centered_unit_vectors(
            mesh_lats, mesh_lngs, sub_lat, sub_lng, centroid_lat, centroid_lng))

    zero_beta_index = np.argmin(np.abs(beta), axis=0)[num_bins // 2][0]
    self.assertEqual(zero_beta_index, num_bins // 2)
    zero_delta_index = np.argmin(np.abs(delta), axis=1)[zero_beta_index][0]
    self.assertEqual(zero_delta_index, num_bins // 2)

    # Check that ell is correct in the center of the footprint (row 2).
    # We're assuming if we got this right we probably would have gotten
    # ell elsewhere in the footprint correct also.
    self._assert_almost_equal_km(
        centroid_ell, distance_km(sub_lat, sub_lng, centroid_lat, centroid_lng))

    center_delta = np.squeeze(delta[zero_beta_index, :])
    inside_cutoff = ((center_delta < power_cutoff.delta_max) &
                     (center_delta > power_cutoff.delta_min))
    cols = inside_cutoff.nonzero()
    min_col, max_col = np.min(cols), np.max(cols)
    self._assert_almost_equal_km(
        d_ell_delta,
        distance_km(plot_lats[zero_beta_index], plot_lngs[min_col],
                    plot_lats[zero_beta_index], plot_lngs[max_col]))

    center_beta = np.squeeze(beta[:, zero_delta_index])
    inside_cutoff = (center_beta < power_cutoff.beta_max)
    rows = inside_cutoff.nonzero()
    min_row, max_row = np.min(rows), np.max(rows)
    self._assert_almost_equal_km(
        d_ell_beta,
        distance_km(plot_lats[min_row], plot_lngs[zero_delta_index],
                    plot_lats[max_row], plot_lngs[zero_delta_index]))

  def test_figure449(self):
    """Confirms point spread function formula was implemented correctly.

    The expected_weight values below are the w_{ij} values from Figure 4.4-9.
    """
    # For whatever reason this figure uses these cutoffs which are slightly
    # different than the table 4.4-2 power cutoffs.
    cutoff_deg = 1.32
    cutoff_percentage = 0.9634
    boundaries = np.linspace(-cutoff_deg, cutoff_deg, 9)
    bin_edges = list(zip(boundaries[:-1], boundaries[1:]))

    # Sampling a single point inside an angular bin will probably
    # give sufficiently accurate weights for machine learning purposes.
    # To precisely match the weights in Figure 4.4-9, here we do a numerical
    # integration matching equation 4.4-18.
    weights = np.zeros((8, 8))
    for col, (delta_bin_start, delta_bin_end) in enumerate(bin_edges):
      for row, (beta_bin_start, beta_bin_end) in enumerate(bin_edges):
        bin_delta = np.linspace(delta_bin_start, delta_bin_end, 100)
        bin_beta = np.linspace(beta_bin_start, beta_bin_end, 100)
        bin_delta, bin_beta = np.meshgrid(bin_delta, bin_beta)
        bin_beta = np.abs(bin_beta)
        bin_delta_prime = bin_delta + ceres_footprint.CENTROID_OFFSET

        # The cosine(delta) term is always ~= 1 at these angles so is omitted.
        weights[row, col] = np.sum(
            ceres_footprint.point_spread_function(bin_delta_prime, bin_beta))
    weights /= (np.sum(weights) / cutoff_percentage)

    row0 = [0.0000, 0.0018, 0.0091, 0.0116, 0.0074, 0.0038, 0.0020, 0.0010]
    row1 = [0.0019, 0.0116, 0.0248, 0.0310, 0.0248, 0.0142, 0.0073, 0.0038]
    row2 = [0.0055, 0.0191, 0.0304, 0.0362, 0.0334, 0.0213, 0.0111, 0.0058]
    row3 = [0.0055, 0.0191, 0.0304, 0.0362, 0.0334, 0.0213, 0.0111, 0.0058]
    expected_weights = np.array(
        [row0, row1, row2, row3, row3, row2, row1, row0])

    np.testing.assert_allclose(weights, expected_weights, atol=0.00007)


if __name__ == '__main__':
  absltest.main()
