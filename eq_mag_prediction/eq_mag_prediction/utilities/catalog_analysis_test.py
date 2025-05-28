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

"""Tests for catalog_analysis."""

import numpy as np
import pandas as pd
from scipy import stats
import tensorflow_probability as tfp
from absl.testing import absltest
from absl.testing import parameterized
from eq_mag_prediction.utilities import catalog_analysis
from eq_mag_prediction.utilities import geometry


_ABS_NUMERICAL_ERROR = 1e-10  # A threshold for an absolute numerical error
_REL_NUMERICAL_ERROR = 1e-4  # A threshold for a relative numerical error


class CatalogAnalysisTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('mle', 'MLE', 0.01), ('b_positive', 'BPOS', 0.02)
  )
  # atol based on expected variance from https://doi.org/10.1029/2020JB021027
  def test_estimate_beta(self, method, atol):
    rs = np.random.RandomState(42)
    for _ in range(10):
      beta, mc = np.abs(rs.randn()), np.abs(rs.randn())
      magnitudes = rs.exponential(1 / beta, 10000) + mc
      np.testing.assert_allclose(
          catalog_analysis.estimate_beta(magnitudes, mc, method=method),
          beta,
          rtol=0.01,
          atol=atol,
      )

  def test_estimate_completeness_by_b_stability(self):
    rs = np.random.RandomState(42)
    good_estimates = 0
    reasonable_estimates = 0
    attempts = 20
    for _ in range(attempts):
      beta, mc = rs.randint(8, 15) / 10, rs.randint(25, 35) / 10
      magnitudes = rs.exponential(1 / beta, 10000)
      not_all_magnitudes = []
      for m in magnitudes:
        # We filter out some magnitudes below the completeness threshold, to
        # simulate a real catalog. The distribution was chosen empirically, in
        # order to make it so that lower magnitudes have a lower probability to
        # stay.
        if m >= mc:
          not_all_magnitudes.append(m)
        else:
          if rs.logistic(0, 0.5) > 2 / m:
            not_all_magnitudes.append(m)

      estimate = catalog_analysis.estimate_completeness_by_b_stability(
          not_all_magnitudes, min_mc=1, max_mc=5
      )
      if abs(estimate - mc) < 0.05:
        good_estimates += 1
      if abs(estimate - mc) < 0.15:
        reasonable_estimates += 1

    self.assertGreaterEqual(good_estimates, attempts * 0.7)
    self.assertGreaterEqual(reasonable_estimates, attempts * 0.8)

  @parameterized.named_parameters(
      ('parameter_set_1', 4, 2), ('parameter_set_2', 2, 1.2)
  )
  def test_estimate_completeness_and_beta_by_maximal_curvature(
      self, alpha, beta
  ):
    rs = tfp.random.sanitize_seed(1905)
    gamma_random_variable = tfp.distributions.Gamma(alpha, beta)
    n_samples = 100_000
    mc_addition_factor = 0.2
    bin_width = 0.2
    magnitudes = np.array(gamma_random_variable.sample(n_samples, seed=rs))
    estimated_mc = catalog_analysis.estimate_completeness_by_maximal_curvature(
        magnitudes,
        mc_addition_factor=mc_addition_factor,
        bin_width=bin_width,
    )
    true_mc = gamma_random_variable.mode() + mc_addition_factor
    np.testing.assert_allclose(
        estimated_mc,
        true_mc,
        rtol=0,
        atol=1.5 * bin_width,
    )

  def test_correct_beta_return_in_joint_m_b_function(self):
    rs = tfp.random.sanitize_seed(1905)
    expected_beta = 1.4
    exponent_random_variable = tfp.distributions.Exponential(expected_beta)
    for _ in range(10):
      magnitudes = np.array(exponent_random_variable.sample(10000, seed=rs)) + 1
      best_mc, best_beta = (
          catalog_analysis.estimate_completeness_and_beta_by_b_stability(
              magnitudes, min_mc=1, max_mc=2
          )
      )
      np.testing.assert_allclose(
          expected_beta,
          best_beta,
          rtol=0.01,
          atol=0.01,
      )
      np.testing.assert_allclose(
          catalog_analysis.estimate_beta(magnitudes, best_mc),
          best_beta,
          rtol=0.01,
          atol=0.01,
      )

  def test_compute_property_in_time_and_space(self):
    time = [10, 20, 30]
    xs = [-1, 1]
    ys = [-1, 1]
    magnitudes = [1, 2, 3]
    space_time_magnitude_event_counts = np.zeros((
        len(time),
        len(xs),
        len(magnitudes),
    ))

    mock_catalog = []
    timestamps = [25, 45]
    examples = {k: [] for k in timestamps}
    number_of_events_in_cell = 0
    for i_idx, i in enumerate(xs):
      for j_idx, j in enumerate(ys):
        for k in timestamps:
          examples[k].append([geometry.Point(i, j)])
        for t_idx, t in enumerate(time):
          for m_idx, m in enumerate(magnitudes):
            for _ in range(number_of_events_in_cell):
              mock_catalog.append({
                  'longitude': i + np.random.rand() * 0.2 - 0.1,
                  'latitude': j + np.random.rand() * 0.2 - 0.2,
                  'magnitude': m + np.random.rand() * 0.2 - 0.1,
                  'time': t + np.random.rand() * 0.2 - 0.1,
              })
              space_time_magnitude_event_counts[t_idx, i_idx, j_idx, m_idx] += 1
        number_of_events_in_cell += 1
    mock_catalog = pd.DataFrame(mock_catalog).sort_values('time')

    lookback_seconds = [10, 20]
    lookback_seconds = np.array(sorted(lookback_seconds, reverse=True)).astype(
        'float64'
    )
    threshold_magnitudes = [0, 2.5, 4]
    threshold_magnitudes = sorted(threshold_magnitudes, reverse=True)
    grid_side_deg = 1
    features = catalog_analysis.compute_property_in_time_and_space(
        catalog=mock_catalog,
        property_function=catalog_analysis.counts_in_square,
        examples=examples,
        grid_side_deg=grid_side_deg,
        lookback_seconds=lookback_seconds,
        magnitudes=threshold_magnitudes,
    )
    cells_shape = np.array(examples[list(examples.keys())[0]]).shape

    self.assertEqual(
        features.shape,
        (
            len(examples),
            *cells_shape,
            len(lookback_seconds),
            len(magnitudes),
            1,
        ),
    )

    for t_i, t in enumerate(list(examples.keys())):
      cell = np.array(examples[t])
      cell_shape = cell.shape
      for row_i in range(cell_shape[0]):
        for col_i in range(cell_shape[1]):
          for lookback_i, lookback in enumerate(lookback_seconds):
            for mag_i, mag in enumerate(threshold_magnitudes):
              time_logical = (mock_catalog.time.values >= (t - lookback)) & (
                  mock_catalog.time.values < t
              )
              magnitude_logical = mock_catalog.magnitude.values >= mag
              location_logical = (
                  (
                      mock_catalog.longitude.values
                      >= cell[row_i, col_i].lng - grid_side_deg / 2
                  )
                  & (
                      mock_catalog.longitude.values
                      <= cell[row_i, col_i].lng + grid_side_deg / 2
                  )
                  & (
                      mock_catalog.latitude.values
                      >= cell[row_i, col_i].lat - grid_side_deg / 2
                  )
                  & (
                      mock_catalog.latitude.values
                      <= cell[row_i, col_i].lat + grid_side_deg / 2
                  )
              )
              total_logical = (
                  time_logical & magnitude_logical & location_logical
              )
              self.assertEqual(
                  total_logical.sum(),
                  features[t_i, row_i, col_i, lookback_i, mag_i, 0],
              )


  def test_return_properties_for_timestamp(self):
    np.random.seed(1905)
    catalog_length = 1000
    catalog = pd.DataFrame({
        'magnitude': np.random.uniform(0, 8, catalog_length),
        'time': np.arange(catalog_length),
    })
    mock_mc = -100
    mc_and_counter = lambda magnitudes: (mock_mc, magnitudes.size)
    window_length = 100
    counter_list = []
    for t in catalog.time.values:
      _, counter = catalog_analysis._return_properties_for_timestamp(
          timestamp=t,
          catalog=catalog,
          n_events=window_length,
          m_minimal=mock_mc,
          n_above_complete=1,
          weight_on_past=0.5,
          property_function=mc_and_counter,
      )
      counter_list.append(counter)
    self.assertTrue(
        np.all(
            np.array(counter_list[window_length : -(window_length - 1)])
            == window_length
        )
    )


class GrMovingWindowTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    rs = tfp.random.sanitize_seed(1905)

    # Set early time distribution
    self.early_beta = 1
    n_sample_early = 1000
    self.mc_early = 0.5
    pdf_early = tfp.distributions.Gamma(1.1, self.early_beta)
    magnitudes_early = (
        np.array(pdf_early.sample(n_sample_early, seed=rs)) + self.mc_early
    )
    times_early = np.linspace(-10, -1, n_sample_early)

    # Set late time distribution
    self.late_beta = 2
    n_sample_late = 500
    self.mc_late = 2
    pdf_late = tfp.distributions.Gamma(1.1, self.late_beta)
    magnitudes_late = (
        np.array(pdf_late.sample(n_sample_late, seed=rs)) + self.mc_late
    )
    times_late = np.linspace(1, 10, n_sample_late)

    all_magnitudes = np.concatenate((magnitudes_early, magnitudes_late))
    all_times = np.concatenate((times_early, times_late))
    self.n_events = 500
    self.time_window = 10
    self.test_catalog = (
        pd.DataFrame({'magnitude': all_magnitudes, 'time': all_times})
        .sort_values('time')
        .reset_index(drop=True)
    )
    self.estimation_times = np.array([-11, 0, 11])

  def test_window_entirely_in_past(self):
    weight_on_past = 1
    expected_mc = np.array([np.nan, self.mc_early, self.mc_late])
    expected_beta = np.array([np.nan, self.early_beta, self.late_beta])
    mc_result, beta_result = self._gr_moving_window_n_events(
        weight_on_past,
        catalog_analysis.estimate_completeness_and_beta_by_maximal_curvature,
    )
    np.testing.assert_allclose(expected_mc, mc_result, rtol=0, atol=0.5)
    np.testing.assert_allclose(expected_beta, beta_result, rtol=0, atol=0.5)

    mc_result, beta_result = self._gr_moving_window_constant_time(
        weight_on_past,
        catalog_analysis.estimate_completeness_by_b_stability,
    )
    np.testing.assert_allclose(expected_mc, mc_result, rtol=0, atol=0.5)
    np.testing.assert_allclose(expected_beta, beta_result, rtol=0, atol=0.5)

  def test_window_entirely_in_future(self):
    weight_on_past = 0
    expected_mc = np.array([self.mc_early, self.mc_late, np.nan])
    expected_beta = np.array([self.early_beta, self.late_beta, np.nan])

    mc_result, beta_result = self._gr_moving_window_n_events(
        weight_on_past,
        catalog_analysis.estimate_completeness_and_beta_by_maximal_curvature,
    )
    np.testing.assert_allclose(expected_mc, mc_result, rtol=0, atol=0.5)
    np.testing.assert_allclose(expected_beta, beta_result, rtol=0, atol=0.5)

    mc_result, beta_result = self._gr_moving_window_constant_time(
        weight_on_past,
        catalog_analysis.estimate_completeness_by_b_stability,
    )
    np.testing.assert_allclose(expected_mc, mc_result, rtol=0, atol=0.5)
    np.testing.assert_allclose(expected_beta, beta_result, rtol=0, atol=0.5)

  def test_window_equally_in_past_and_future(self):
    weight_on_past = 0.5
    expected_mc = np.array(
        [self.mc_early, np.maximum(self.mc_early, self.mc_late), self.mc_late]
    )
    expected_beta = np.array([
        self.early_beta,
        np.mean([self.early_beta, self.late_beta]),
        self.late_beta,
    ])

    mc_result, beta_result = self._gr_moving_window_n_events(
        weight_on_past,
        catalog_analysis.estimate_completeness_and_beta_by_b_stability,
    )
    np.testing.assert_allclose(expected_mc, mc_result, rtol=0, atol=0.5)
    np.testing.assert_allclose(expected_beta, beta_result, rtol=0, atol=0.5)

    mc_result, beta_result = self._gr_moving_window_constant_time(
        weight_on_past,
        catalog_analysis.estimate_completeness_by_b_stability,
    )
    np.testing.assert_allclose(expected_mc, mc_result, rtol=0, atol=0.5)
    np.testing.assert_allclose(expected_beta, beta_result, rtol=0, atol=0.5)

  @parameterized.named_parameters(
      ('only_in_past_100events', 1, 100),
      ('only_in_future_100events', 0, 100),
      ('quarter_in_past_100events', 0.25, 100),
      ('quarter_in_past_300events', 0.25, 300),
  )
  def test_estimate_beta_given_mc(self, weight_on_past, n_events):
    test_timestamps = np.linspace(-5, 10, 30)
    completeness_vec, base_beta = catalog_analysis.gr_moving_window_n_events(
        estimate_times=test_timestamps,
        catalog=self.test_catalog,
        n_events=n_events,
        weight_on_past=weight_on_past,
        completeness_and_beta_calculator=catalog_analysis.estimate_completeness_and_beta_by_maximal_curvature,
    )

    test_beta = catalog_analysis.estimate_beta_given_mc(
        timestamps=test_timestamps,
        mc=completeness_vec,
        catalog=self.test_catalog,
        n_events=n_events,
        weight_on_past=weight_on_past,
    )
    np.testing.assert_allclose(
        base_beta,
        test_beta,
        rtol=_REL_NUMERICAL_ERROR,
        atol=_ABS_NUMERICAL_ERROR,
    )

  def _gr_moving_window_n_events(
      self, weight_on_past, completeness_and_beta_calculator
  ):
    return catalog_analysis.gr_moving_window_n_events(
        estimate_times=self.estimation_times,
        catalog=self.test_catalog,
        n_events=self.n_events,
        weight_on_past=weight_on_past,
        completeness_and_beta_calculator=completeness_and_beta_calculator,
    )

  def _gr_moving_window_constant_time(
      self, weight_on_past, completeness_calculator
  ):
    return catalog_analysis.gr_moving_window_constant_time(
        estimate_times=self.estimation_times,
        catalog=self.test_catalog,
        window_time=self.time_window,
        weight_on_past=weight_on_past,
        completeness_calculator=completeness_calculator,
    )


class SpatialBetaCalculatorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    rng = np.random.default_rng(seed=1905)

    self.n_many = 10000
    self.n_few = 250

    self.left_location = (-10, 0)
    self.center_location = (0, 0)
    self.right_location = (10, 0)
    self.left_beta = 1
    self.center_beta = 1
    self.right_beta = 2

    left_cluster = self._generate_part_of_catalog(
        n_earthquakes=self.n_many,
        center_coordinates=self.left_location,
        square_side=2,
        beta=self.left_beta,
        mc=0.5,
        random_seed=rng,
    )
    center_cluster = self._generate_part_of_catalog(
        n_earthquakes=self.n_few,
        center_coordinates=self.center_location,
        square_side=2,
        beta=self.center_beta,
        mc=0.5,
        random_seed=rng,
    )
    right_cluster = self._generate_part_of_catalog(
        n_earthquakes=self.n_many,
        center_coordinates=self.right_location,
        square_side=2,
        beta=self.right_beta,
        mc=0.5,
        random_seed=rng,
    )
    catalog = pd.concat(
        [left_cluster, center_cluster, right_cluster], ignore_index=True
    )
    # Randomize the row order
    catalog = catalog.sample(frac=1).reset_index(drop=True)
    # Add a sequential 'time' column
    catalog['time'] = np.arange(len(catalog))
    self.test_catalog = catalog

  @parameterized.named_parameters(
      ('enforced_completeness', 0.5),
      ('deduced_completeness', None),
  )
  def test_spatial_beta_calculator(self, mc):
    beta_results, spatial_beta_calculator, test_longitudes, test_latitudes = (
        self._beta_and_vars_for_test(mc)
    )
    # Center has few samples thus beta measurements is unreliable
    expected_beta = np.array([self.left_beta, self.right_beta])
    np.testing.assert_allclose(
        expected_beta, np.delete(beta_results, 1), rtol=0, atol=0.3
    )

    expected_beta = np.array([self.left_beta, np.nan, self.right_beta])
    beta_results = spatial_beta_calculator(
        test_longitudes,
        test_latitudes,
        discard_few_event_locations=self.n_few + 1,
    )
    np.testing.assert_allclose(expected_beta, beta_results, rtol=0, atol=0.3)

  def test_estimate_beta_at_location_by_vicinity_to_sample(self):
    min_correlation = 0.7
    beta_results, spatial_beta_calculator, test_longitudes, test_latitudes = (
        self._beta_and_vars_for_test()
    )
    beta_estimates = (
        spatial_beta_calculator.estimate_beta_at_location_by_vicinity_to_sample(
            test_longitudes,
            test_latitudes,
            discard_few_event_locations=None,
        )
    )
    finite_logical = np.isfinite(beta_results) & np.isfinite(beta_estimates)
    pc = stats.pearsonr(
        beta_results[finite_logical], beta_estimates[finite_logical]
    ).statistic
    self.assertGreaterEqual(pc, min_correlation)

  def _beta_and_vars_for_test(self, mc=None):
    spatial_beta_calculator = catalog_analysis.SpatialBetaCalculator(
        catalog=self.test_catalog,
        completeness_magnitude=mc,
        grid_spacing=1.1,
        smoothing_distance=1,
    )
    test_coordinates = np.array(
        (self.left_location, self.center_location, self.right_location)
    )
    test_longitudes = test_coordinates[:, 0]
    test_latitudes = test_coordinates[:, 1]
    beta_results = spatial_beta_calculator(
        test_longitudes,
        test_latitudes,
        discard_few_event_locations=None,
    )
    return (
        beta_results,
        spatial_beta_calculator,
        test_longitudes,
        test_latitudes,
    )

  def _generate_part_of_catalog(
      self,
      n_earthquakes,
      center_coordinates,
      square_side,
      beta,
      mc,
      random_seed,
  ):
    magnitudes = random_seed.exponential(1 / beta, n_earthquakes) + mc
    longitudes = random_seed.uniform(
        center_coordinates[0] - square_side / 2,
        center_coordinates[0] + square_side / 2,
        n_earthquakes,
    )
    latitudes = random_seed.uniform(
        center_coordinates[1] - square_side / 2,
        center_coordinates[1] + square_side / 2,
        n_earthquakes,
    )
    return pd.DataFrame.from_dict({
        'magnitude': magnitudes,
        'longitude': longitudes,
        'latitude': latitudes,
    })


class MiscFunctionsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('full_overlap', (0, 0), 1, (0, 0), 1, np.pi),
      ('half_overlap_square_shift1', (0, 0), 1, (1, 0), 1, np.pi / 2),
      ('half_overlap_square_shift2', (0, 0), 1, (0, 1), 1, np.pi / 2),
      ('half_overlap_circle_shoft', (1, 0), 1, (0, 0), 1, np.pi / 2),
      ('quarter overlap', (0, 0), 1, (1, 1), 1, np.pi / 4),
      ('no_overlap', (0, 0), 1, (10, 10), 1, 0),
  )
  def test_circle_square_overlap(
      self,
      circle_center,
      circle_radius,
      square_center,
      square_side,
      expected,
  ):
    got = catalog_analysis.circle_square_overlap(
        circle_center=circle_center,
        circle_radius=circle_radius,
        square_center=square_center,
        square_side=square_side,
    )
    np.testing.assert_allclose(
        expected, got, rtol=0, atol=_REL_NUMERICAL_ERROR * 100
    )


class SpatialMcMeasureTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    rs = tfp.random.sanitize_seed(1905)
    rs_np = np.random.RandomState(1905)

    self.events_spread = 1
    # Set left side distribution
    n_sample_left = 1000
    self.mc_left = 1.5
    pdf_left = tfp.distributions.Gamma(6, 4)
    magnitudes_left = np.array(pdf_left.sample(n_sample_left, seed=rs))
    self.left_center = (-2, -2)
    lons_left = rs_np.uniform(
        self.left_center[0] - self.events_spread,
        self.left_center[1] + self.events_spread,
        n_sample_left,
    )
    lats_left = rs_np.uniform(
        self.left_center[0] - self.events_spread,
        self.left_center[1] + self.events_spread,
        n_sample_left,
    )

    # Set late time distribution
    n_sample_right = 1500
    self.mc_right = 0.8
    pdf_right = tfp.distributions.Gamma(3, 4)
    magnitudes_right = np.array(pdf_right.sample(n_sample_right, seed=rs))
    self.right_center = (2, 2)
    lons_right = rs_np.uniform(
        self.right_center[0] - self.events_spread,
        self.right_center[1] + self.events_spread,
        n_sample_right,
    )
    lats_right = rs_np.uniform(
        self.right_center[0] - self.events_spread,
        self.right_center[1] + self.events_spread,
        n_sample_right,
    )

    all_lons = np.concatenate((lons_left, lons_right))
    all_lats = np.concatenate((lats_left, lats_right))
    all_times = np.arange(n_sample_left + n_sample_right)
    all_magnitudes = np.concatenate((magnitudes_left, magnitudes_right))
    self.n_events = 500
    self.time_window = 10
    self.test_catalog = (
        pd.DataFrame({
            'magnitude': all_magnitudes,
            'time': all_times,
            'longitude': all_lons,
            'latitude': all_lats,
        })
        .sort_values('time')
        .reset_index(drop=True)
    )
    self.estimation_times = np.array([-11, 0, 11])

  @parameterized.named_parameters(
      ('rad_0.5', 0.5, 800),
      ('rad_1', 1, 800),
      ('small_rad_many_events', 0.001, 1000),
  )
  def test_calc_mc_on_grid(
      self,
      minimal_radius,
      minimal_events,
  ):
    number_of_events, radius, measured_m_c = (
        catalog_analysis._calc_mc_at_coordinates(
            longitudes=[self.left_center[0], self.right_center[0]],
            latitudes=[self.left_center[1], self.right_center[1]],
            catalog=self.test_catalog,
            minimal_radius=minimal_radius,
            minimal_events=minimal_events,
        )
    )
    expected = np.array([self.mc_left, self.mc_right])
    np.testing.assert_allclose(
        expected,
        measured_m_c,
        rtol=0,
        atol=0.8,
    )
    _ = [self.assertGreaterEqual(n, minimal_events) for n in number_of_events]
    _ = [self.assertGreaterEqual(r, 0.95 * minimal_radius) for r in radius]




if __name__ == '__main__':
  absltest.main()
