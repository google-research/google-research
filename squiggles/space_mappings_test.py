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

"""Tests for space_mappings."""
import math

from absl.testing import absltest
import numpy as np

from squiggles import space_mappings


class DerivsToPathPointsTest(absltest.TestCase):

  def test_shape(self):
    dummy_input = np.zeros(shape=(11, 2, 5), dtype=np.float32)
    expected_output_shape = (11, 17, 2)

    output = space_mappings.derivs_to_path_points(dummy_input, num_points=17)
    self.assertSequenceEqual(expected_output_shape, output.shape)

  def test_entries(self):
    exp_derivs = np.ones(shape=(16,), dtype=np.float32)
    sin_derivs = np.array(4 * [1, 0, -1, 0], dtype=np.float32)
    input_array = np.stack([exp_derivs, sin_derivs], axis=0)
    input_array = input_array[np.newaxis, :]

    output = space_mappings.derivs_to_path_points(input_array, num_points=7)
    expected_output = np.array(
        [[
            [math.exp(-3) - 1, math.sin(-3)],  # autoformat: \n
            [math.exp(-2) - 1, math.sin(-2)],
            [math.exp(-1) - 1, math.sin(-1)],
            [math.exp(0) - 1, math.sin(0)],
            [math.exp(1) - 1, math.sin(1)],
            [math.exp(2) - 1, math.sin(2)],
            [math.exp(3) - 1, math.sin(3)]
        ]],
        dtype=np.float32)

    np.testing.assert_allclose(expected_output, output, atol=1e-6, rtol=1e-6)


def _sine_net_to_path_points_test_function_to_approximate(t):
  """Function whose Fourier series over [-1, 1] converges reasonably quickly."""
  # The Fourier series converges reasonably quickly on the interval [-1, 1]
  # since
  #
  #        f(1) = f(-1)
  #       f'(1) = f'(-1)
  #      f''(1) = f''(-1)
  #     f'''(1) = f'''(-1).
  #
  # Multiplying by (2 - t) makes the function neither even nor odd, ensuring
  # that the Fourier series will contain both sine and cosine terms. Since we
  # use phases rather than separate sine and cosine terms, this ensures the
  # phases will be nontrivial.
  return (1 - t)**4 * (1 + t)**4 * (2 - t)


class SinNetToPathPointsTest(absltest.TestCase):

  def test_shape(self):
    dummy_input = np.zeros(shape=(11, 31, 4), dtype=np.float32)
    expected_output_shape = (11, 17, 2)
    output = space_mappings.sine_net_to_path_points(dummy_input, num_points=17)
    self.assertSequenceEqual(expected_output_shape, output.shape)

  def test_entries_x(self):
    num_points = 7
    t_scale = 2
    # `const_coeff`, `x_amplitudes` and `phases` construct the (first few terms
    # of the) Fourier series for
    # _sine_net_to_path_points_test_function_to_approximate. They were computed
    # using SageMath; for the computations and verifications see the notebook at
    # https://sagecell.sagemath.org/?z=eJydk9tugzAMhu_zFNZuGrf0AKp2x5OgDmXUbNEgyZKgtW8_F2gp0jpN44IEH77fJk4NOThNFX3pQLKQcp0mkGICMoU1nPBlD8vLfnXbZ70dDyiE6doykm8DQ7KdCNqUlaW6vnwX9aa2ndfky0C8hJLdNPh1pclEqZmaItTWgwZtwCvzRvIGxYOobPiNyO7_ME2IQw5D_8LcIWwhE0K3zvoIDHNnUAGMm7ds3EZ5r85ysiZwjGdHObvqxqr4vMd5U7ecyfpTjlCta3TsjjQmhU8fZb92ytOdIvIPmOwTFVE4r7mbu_6vpgnOSu5dhV6lF5dObzPkI-8LraIy2ay9O4ERNuQzSLrGzuS4stC1spjkCn3gkWKelHrFx7bkWeTXiSMHDAeggEfP41PGZMySpwTGkR4tlW2szxeejosEWuU_yOdP9ollWPRScc3Df70FY_Br09ECBX4DtRYGjQ==&lang=sage&interacts=eJyLjgUAARUAuQ==
    const_coeff = 256 / 315
    x_amplitudes = np.array([
        1.03251546e+00, 2.01003369e-01, 1.93362046e-02, 3.75423077e-03,
        1.08763061e-03, 4.04024912e-04, 1.77361643e-04, 8.77096053e-05,
        4.74116246e-05, 2.74572061e-05, 1.67990298e-05, 1.07490437e-05,
        7.13888030e-06, 4.89272148e-06, 3.44479004e-06, 2.48254596e-06,
        1.82592189e-06, 1.36732661e-06, 1.04039452e-06, 8.03023841e-07
    ],
                            dtype=np.float32)
    y_amplitudes = np.zeros_like(x_amplitudes)
    phases = np.array([
        1.72361702, 1.99433456, -1.53750914, 1.32483942, 4.26165659, 0.9658349,
        3.98821988, 0.75230197, 3.81770253, 0.61345077, 3.70270494, 0.51679847,
        3.62042536, 0.4459663, 3.55884523, 0.39196216, 3.51111752, 0.34948904,
        3.47308523, 0.31524202
    ],
                      dtype=np.float32)
    radian_frequencies = np.array([
        (i + 1) * math.pi for i in range(len(phases))
    ])
    input_array = np.stack(
        [radian_frequencies, phases, x_amplitudes, y_amplitudes], axis=1)
    input_array = input_array[np.newaxis, :]

    expected_output_x = np.array([
        _sine_net_to_path_points_test_function_to_approximate(t)
        for t in np.linspace(-1.0, 1.0, 7)
    ],
                                 dtype=np.float32)
    expected_output_y = np.zeros_like(expected_output_x)
    expected_output = np.stack([expected_output_x, expected_output_y], axis=1)
    expected_output = expected_output[np.newaxis, :]

    # The const_coeff could theoretically have been included as a coefficient of
    # sin(0*t + pi/2), but that was more effort than it was worth.
    output = (
        space_mappings.sine_net_to_path_points(
            input_array, num_points=num_points, t_scale=t_scale) +
        np.array([const_coeff, 0.0]))

    np.testing.assert_allclose(expected_output, output, atol=1e-5)


class RescalePointsTest(absltest.TestCase):

  def test_shape(self):
    dummy_input = np.zeros(shape=(7, 11, 2), dtype=np.float32)
    expected_output_shape = (7, 11, 2)

    output = space_mappings.rescale_points(dummy_input)

    self.assertSequenceEqual(expected_output_shape, output.shape)

  def test_x_dominates(self):
    input_array = np.array([[(-1, 0), (0, 0), (1, 1)]], dtype=np.float32)
    output = space_mappings.rescale_points(input_array, margin=0.0)  # pytype: disable=wrong-arg-types  # jax-ndarray
    # Scale everything by factor of 0.5. (Also translate.)
    expected_output = np.array([[(0, 0.25), (0.5, 0.25), (1, 0.75)]],
                               dtype=np.float32)

    np.testing.assert_allclose(expected_output, output)

  def test_y_dominates(self):
    input_array = np.array([[(0, 0), (0, 2), (1, 1)]], dtype=np.float32)
    output = space_mappings.rescale_points(input_array, margin=0.0)  # pytype: disable=wrong-arg-types  # jax-ndarray
    # Scale everything by factor of 0.5. (Also translate.)
    expected_output = np.array([[(0.25, 0), (0.25, 1), (0.75, 0.5)]],
                               dtype=np.float32)

    np.testing.assert_allclose(expected_output, output)

  def test_margin(self):
    input_array = np.array([[(0, 0), (1, 1), (2, 2)]], dtype=np.float32)
    output = space_mappings.rescale_points(input_array)  # default margin is 0.1  # pytype: disable=wrong-arg-types  # jax-ndarray
    # Scale everything to fit inside [0.1, 0.9] x [0.1, 0.9].
    expected_output = np.array([[(0.1, 0.1), (0.5, 0.5), (0.9, 0.9)]],
                               dtype=np.float32)

    np.testing.assert_allclose(expected_output, output)

  def test_two_curves(self):
    """Distinct items in a batch should scale independently."""
    input_array = np.array(
        [
            [(0, 0), (0.5, 0), (1, 1)],  #
            [(0, 0), (1, 1), (2, 2)]
        ],
        dtype=np.float32)
    output = space_mappings.rescale_points(input_array, margin=0.0)  # pytype: disable=wrong-arg-types  # jax-ndarray
    expected_output = np.array(
        [
            [(0, 0), (0.5, 0), (1, 1)],  #
            [(0, 0), (0.5, 0.5), (1, 1)]
        ],
        dtype=np.float32)

    np.testing.assert_allclose(expected_output, output)


class GaussianActivationTest(absltest.TestCase):

  def test_shape(self):
    dummy_input = np.zeros(shape=(3, 5, 7, 1), dtype=np.float32)
    expected_output_shape = (3, 5, 7, 1)
    output = space_mappings.gaussian_activation(dummy_input)
    self.assertSequenceEqual(expected_output_shape, output.shape)

  def test_entries(self):
    input_array = np.array([0.0, 1 / 11, -1 / 11, -1.0, 1.0], dtype=np.float32)
    output = space_mappings.gaussian_activation(input_array, spread=1 / 11)
    expected_output = np.array([1.0] + 2 * [np.exp(-1)] + 2 * [np.exp(-121.0)])

    np.testing.assert_allclose(expected_output, output, atol=1e-6)

  def test_computed_spread(self):
    input_array = (
        np.random.RandomState(seed=328773902)  # Make test deterministic
        .standard_normal(size=(3, 13, 7, 2))  #
        .astype(np.float32, casting='same_kind'))
    expected_output = np.exp(-np.square(13 * input_array))
    output = space_mappings.gaussian_activation(
        input_array, spread=None
    )  # When spread is *explicitly* set to None, it defaults to 1 / sidelength.
    np.testing.assert_allclose(expected_output, output, atol=1e-6)


class NearestPointDistanceTest(absltest.TestCase):

  def test_one_point(self):
    one_point_input = np.array([[(0.0, 1.0)]], dtype=np.float32)
    expected_output = np.array([[
        [1.0, 0.0],  #
        [math.sqrt(2.0), 1.0]
    ]])
    output = space_mappings.nearest_point_distance(
        one_point_input, x_pixels=2, y_pixels=2)

    np.testing.assert_allclose(expected_output, output)

  def test_two_points(self):
    two_points_input = np.array([[(0, 0), (0.5, 1.0)]], dtype=np.float32)
    expected_output = np.array([[
        [0.0, 0.5],  #
        [1.0, 0.5]
    ]])
    output = space_mappings.nearest_point_distance(
        two_points_input, x_pixels=2, y_pixels=2)

    np.testing.assert_allclose(expected_output, output)

  def test_shape(self):
    dummy_input = np.zeros(shape=(17, 19, 2), dtype=np.float32)
    expected_output_shape = (17, 11, 13)
    output = space_mappings.nearest_point_distance(
        dummy_input, x_pixels=11, y_pixels=13)

    self.assertSequenceEqual(expected_output_shape, output.shape)


class CoordsToPixelsTest(absltest.TestCase):

  def test_two_points(self):
    two_points_input = np.array([[(0, 0), (0.5, 1.0)]], dtype=np.float32)
    expected_output = np.array(
        [[
            [[1.0], [0.36787944]],  #
            [[0.01831563888], [0.36787944]]
        ]],
        dtype=np.float32)
    output = space_mappings.coords_to_pixels(
        two_points_input, 2, 2, spread=1 / 2)
    np.testing.assert_allclose(expected_output, output)


if __name__ == '__main__':
  absltest.main()
