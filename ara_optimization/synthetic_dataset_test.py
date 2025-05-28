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

from absl.testing import absltest
import numpy as np

from ara_optimization import synthetic_dataset


class SyntheticDatasetTest(absltest.TestCase):

  def test_generate_slice_distribution_for_impressions(self):
    b = -1.4
    num_slices = 150
    slice_dist = synthetic_dataset._generate_slice_distribution_for_impressions(
        param_b=b, number_of_slices=num_slices
    )

    # check slice frequencies summing to num_slices
    self.assertAlmostEqual(sum(slice_dist.values()), num_slices)

  def test_generate_slice_distribution_for_impressions_approximate_fit(self):
    b = -1.5
    num_slices = 28
    slice_dist = synthetic_dataset._generate_slice_distribution_for_impressions(
        param_b=b, number_of_slices=num_slices
    )

    # check slice frequencies summing to num_slices
    self.assertAlmostEqual(sum(slice_dist.values()), num_slices)

  def test_get_conversion_distribution(self):
    total_keys = 500
    num_slices = 10
    conv_dist = synthetic_dataset.get_conversion_distribution_uniformly(
        num_slices=num_slices, total_conversions=total_keys
    )
    # conv_dist supposed to meet 2 criteria:
    # 1. have num_slices many slices
    # 2. slice frequencies summing to total_keys
    # check criteria 1
    self.assertLen(conv_dist, num_slices)
    # check criteria 2
    self.assertAlmostEqual(sum(conv_dist), total_keys)

  def test_get_conversion_distribution_extrame_values(self):
    total_keys = 15000
    num_slices = 3000
    conv_dist = synthetic_dataset.get_conversion_distribution_uniformly(
        num_slices=num_slices, total_conversions=total_keys
    )
    # check have num_slices many slices
    self.assertLen(conv_dist, num_slices)

  def test_get_conversion_distribution_extrame_values_have_remaining(self):
    total_keys = 251
    num_slices = 13
    conv_dist = synthetic_dataset.get_conversion_distribution_uniformly(
        num_slices=num_slices, total_conversions=total_keys
    )
    # conv_dist supposed to meet 2 criteria:
    # 1. have num_slices many slices
    # 2. slice frequencies summing to total_keys
    # check criteria 1
    self.assertLen(conv_dist, num_slices)
    # check criteria 2
    self.assertAlmostEqual(
        sum(conv_dist), total_keys, delta=abs(total_keys * 0.05)
    )

  def test_get_b_param_from_slice_size_rates(self):
    rate_of_size_1 = 0.40
    rate_of_size_2 = 0.15
    param_b_expected = -1.41503749928  # log2(0.15/0.40)

    param_b = synthetic_dataset._get_b_param_from_slice_size_rates(
        rate_of_size_1, rate_of_size_2
    )
    self.assertAlmostEqual(param_b, param_b_expected)

  def test_generate_slice_distribution_with_conversions(self):
    b = -1.5
    impression_side_dimensions = [3, 8]
    conversion_side_dimensions = [2]
    value_mean = 30
    value_mode = 4
    impression_key_cardinality = np.prod(impression_side_dimensions)
    conversion_key_cardinality = np.prod(conversion_side_dimensions)
    average_conversion_per_impression = 40

    impression_dist = (
        synthetic_dataset._generate_slice_distribution_for_impressions(
            b, impression_key_cardinality
        )
    )

    attributed_conversions = (
        synthetic_dataset.generate_slice_distribution_with_conversions_raw(
            impression_dist=impression_dist,
            average_conversion_per_impression=average_conversion_per_impression,
            impression_side_dimensions=impression_side_dimensions,
            conversion_side_dimensions=conversion_side_dimensions,
            value_mean=value_mean,
            value_mode=value_mode,
        )
    )

    # check if all impression side keys appears in attributed_conversions
    records = set([(conv[1][0], conv[1][1]) for conv in attributed_conversions])
    self.assertLen(records, impression_key_cardinality)

    # check if all conversion side keys appears in attributed_conversions
    records = set([(conv[1][2]) for conv in attributed_conversions])
    self.assertLen(records, conversion_key_cardinality)

  def test_generate_dataset_with_values(self):
    np.random.seed(42)
    slice_1_rate = 0.39
    slice_2_rate = 0.14
    average_conversion_per_impression = 6
    impression_side_dimensions = [2, 4]
    conversion_side_dimensions = [2]

    dataset_expected = [
        ['imp-1', (0, 0, 0), 1, 57.28520163225153],
        ['imp-1', (0, 0, 0), 1, 63.37565751300641],
        ['imp-1', (0, 0, 0), 1, 13.845256540510109],
        ['imp-1', (0, 0, 0), 1, 429.4064822140339],
        ['imp-1', (0, 0, 1), 1, 74.48945212699607],
        ['imp-2', (0, 0, 0), 1, 115.70953483448099],
        ['imp-2', (0, 0, 0), 1, 54.541010374919225],
        ['imp-2', (0, 0, 0), 1, 74.0261353024022],
        ['imp-2', (0, 0, 0), 1, 85.56285331545836],
        ['imp-3', (0, 1, 1), 1, 165.5596916925544],
        ['imp-3', (0, 1, 1), 1, 15.517627320489568],
        ['imp-4', (0, 1, 0), 1, 50.89377568833895],
        ['imp-4', (0, 1, 0), 1, 6.136920930256484],
        ['imp-4', (0, 1, 1), 1, 22.854100743615316],
        ['imp-4', (0, 1, 1), 1, 93.61138792507235],
        ['imp-4', (0, 1, 1), 1, 111.58042177829326],
        ['imp-4', (0, 1, 1), 1, 181.89838351382755],
        ['imp-5', (0, 2, 0), 1, 9.119461356799928],
        ['imp-5', (0, 2, 0), 1, 97.92767310602233],
        ['imp-5', (0, 2, 0), 1, 153.08181632286716],
        ['imp-5', (0, 2, 0), 1, 27.116793993235884],
        ['imp-5', (0, 2, 0), 1, 67.30319201295315],
        ['imp-5', (0, 2, 0), 1, 420.52170559893904],
        ['imp-5', (0, 2, 0), 1, 20.8303020827322],
        ['imp-5', (0, 2, 0), 1, 230.45420323187116],
        ['imp-5', (0, 2, 0), 1, 566.810121504647],
        ['imp-5', (0, 2, 1), 1, 130.28017512128383],
        ['imp-6', (0, 2, 0), 1, 10.879941167466312],
        ['imp-6', (0, 2, 0), 1, 236.66852144983423],
        ['imp-6', (0, 2, 0), 1, 192.6917704854611],
        ['imp-7', (0, 3, 0), 1, 108.19944551718868],
        ['imp-7', (0, 3, 1), 1, 23.27594072127602],
        ['imp-7', (0, 3, 1), 1, 14.803374962982078],
        ['imp-7', (0, 3, 1), 1, 95.54530261032083],
        ['imp-7', (0, 3, 1), 1, 89.68817398819138],
        ['imp-8', (0, 3, 0), 1, 155.0540305404866],
        ['imp-8', (0, 3, 1), 1, 107.62522594004291],
        ['imp-9', (1, 0, 0), 1, 216.6299943913337],
        ['imp-9', (1, 0, 0), 1, 88.78408973593353],
        ['imp-9', (1, 0, 0), 1, 18.0095526466048],
        ['imp-9', (1, 0, 0), 1, 730.2446487261744],
        ['imp-10', (1, 0, 0), 1, 64.15389553353378],
        ['imp-10', (1, 0, 0), 1, 108.66094943475025],
        ['imp-10', (1, 0, 1), 1, 14.134061510227115],
        ['imp-10', (1, 0, 1), 1, 35.548244398465116],
        ['imp-10', (1, 0, 1), 1, 64.66627084178333],
        ['imp-10', (1, 0, 1), 1, 28.745226774938036],
        ['imp-12', (1, 1, 0), 1, 29.53984845276234],
        ['imp-12', (1, 1, 0), 1, 48.862223826193265],
        ['imp-12', (1, 1, 0), 1, 27.459077870625173],
        ['imp-12', (1, 1, 1), 1, 50.68020320792792],
        ['imp-12', (1, 1, 1), 1, 145.9907352860136],
        ['imp-12', (1, 1, 1), 1, 256.9367264701665],
        ['imp-12', (1, 1, 1), 1, 141.0552383411823],
        ['imp-13', (1, 1, 0), 1, 82.41886304013596],
        ['imp-13', (1, 1, 0), 1, 137.05036355673676],
        ['imp-13', (1, 1, 0), 1, 64.63086121527817],
        ['imp-13', (1, 1, 1), 1, 94.47563971817995],
        ['imp-13', (1, 1, 1), 1, 234.44271939485154],
        ['imp-13', (1, 1, 1), 1, 22.306258896567066],
        ['imp-13', (1, 1, 1), 1, 86.92079936482894],
        ['imp-14', (1, 1, 0), 1, 86.62430710670266],
        ['imp-14', (1, 1, 0), 1, 163.34760490881894],
        ['imp-14', (1, 1, 0), 1, 275.0656690878604],
        ['imp-14', (1, 1, 1), 1, 8.980801938392347],
        ['imp-14', (1, 1, 1), 1, 296.91275874514065],
        ['imp-14', (1, 1, 1), 1, 24.45778788573441],
        ['imp-14', (1, 1, 1), 1, 21.566255200636487],
        ['imp-14', (1, 1, 1), 1, 2.722976866518531],
        ['imp-15', (1, 1, 0), 1, 30.208468799102377],
        ['imp-15', (1, 1, 0), 1, 44.832565380675376],
        ['imp-15', (1, 1, 0), 1, 21.107373468685484],
        ['imp-15', (1, 1, 0), 1, 209.3725778621213],
        ['imp-15', (1, 1, 0), 1, 232.71103629586008],
        ['imp-15', (1, 1, 1), 1, 124.4119328147427],
        ['imp-15', (1, 1, 1), 1, 8.606562342333739],
        ['imp-15', (1, 1, 1), 1, 1068.2701208894125],
        ['imp-16', (1, 2, 0), 1, 90.86217451714266],
        ['imp-16', (1, 2, 0), 1, 12.691053015760481],
        ['imp-16', (1, 2, 0), 1, 42.408212121113806],
        ['imp-16', (1, 2, 0), 1, 32.77003346360243],
        ['imp-16', (1, 2, 0), 1, 96.10164922441334],
        ['imp-16', (1, 2, 0), 1, 40.23204317907693],
        ['imp-16', (1, 2, 1), 1, 66.08600280776427],
        ['imp-16', (1, 2, 1), 1, 12.55756912094819],
        ['imp-16', (1, 2, 1), 1, 778.5192809587189],
        ['imp-16', (1, 2, 1), 1, 19.068822057949046],
        ['imp-17', (1, 2, 0), 1, 308.39903224182575],
        ['imp-17', (1, 2, 0), 1, 47.49442985560207],
        ['imp-17', (1, 2, 1), 1, 152.83345217835407],
        ['imp-17', (1, 2, 1), 1, 7.620326708544707],
        ['imp-17', (1, 2, 1), 1, 125.55735540369804],
        ['imp-17', (1, 2, 1), 1, 40.92143479435512],
        ['imp-18', (1, 2, 0), 1, 188.54461955642327],
        ['imp-18', (1, 2, 0), 1, 527.0530972761997],
        ['imp-18', (1, 2, 0), 1, 32.53218343010536],
        ['imp-18', (1, 2, 1), 1, 67.67813284427548],
        ['imp-18', (1, 2, 1), 1, 43.61833998932426],
        ['imp-18', (1, 2, 1), 1, 485.5384042670982],
        ['imp-18', (1, 2, 1), 1, 46.342753973682754],
        ['imp-18', (1, 2, 1), 1, 32.779869319139976],
        ['imp-19', (1, 2, 0), 1, 39.20409023361329],
        ['imp-19', (1, 2, 0), 1, 52.019068418702474],
        ['imp-19', (1, 2, 0), 1, 53.017289564732586],
        ['imp-19', (1, 2, 0), 1, 110.75175503837713],
        ['imp-19', (1, 2, 0), 1, 11.827627087349667],
        ['imp-19', (1, 2, 1), 1, 229.22253682845474],
        ['imp-19', (1, 2, 1), 1, 332.53987197310363],
        ['imp-20', (1, 2, 0), 1, 30.758750588173516],
        ['imp-20', (1, 2, 0), 1, 712.2802360586452],
        ['imp-20', (1, 2, 0), 1, 19.438311586268814],
        ['imp-20', (1, 2, 0), 1, 99.92065968651967],
        ['imp-20', (1, 2, 1), 1, 51.710147760414216],
        ['imp-21', (1, 3, 0), 1, 29.78801917234388],
        ['imp-21', (1, 3, 0), 1, 5.869601085617152],
        ['imp-21', (1, 3, 0), 1, 383.5618767028249],
        ['imp-21', (1, 3, 0), 1, 669.6882533445241],
        ['imp-22', (1, 3, 0), 1, 542.0746502190372],
        ['imp-22', (1, 3, 1), 1, 135.55667202746454],
        ['imp-22', (1, 3, 1), 1, 13.048738711689497],
        ['imp-22', (1, 3, 1), 1, 94.4613549480219],
        ['imp-23', (1, 3, 0), 1, 265.1597270553973],
        ['imp-23', (1, 3, 0), 1, 47.76194668716046],
        ['imp-23', (1, 3, 0), 1, 103.77454249633779],
        ['imp-23', (1, 3, 0), 1, 77.71736291701087],
        ['imp-23', (1, 3, 1), 1, 225.41623117027817],
        ['imp-24', (1, 3, 0), 1, 55.972835833364265],
        ['imp-24', (1, 3, 0), 1, 37.653558870434246],
        ['imp-24', (1, 3, 0), 1, 703.0070603532364],
        ['imp-24', (1, 3, 0), 1, 263.37884767673404],
        ['imp-24', (1, 3, 0), 1, 9.602564586166466],
        ['imp-24', (1, 3, 0), 1, 45.658638765722976],
        ['imp-24', (1, 3, 0), 1, 309.66783293775813],
        ['imp-24', (1, 3, 0), 1, 19.1456939261324],
        ['imp-24', (1, 3, 1), 1, 187.20664156221596],
        ['imp-24', (1, 3, 1), 1, 166.021190293246],
        ['imp-24', (1, 3, 1), 1, 34.07942349935298],
        ['imp-24', (1, 3, 1), 1, 75.85184608879726],
        ['imp-25', (1, 3, 0), 1, 156.091099233627],
        ['imp-25', (1, 3, 0), 1, 369.7316863259624],
        ['imp-25', (1, 3, 0), 1, 46.377044593670874],
        ['imp-25', (1, 3, 0), 1, 97.0503981223615],
        ['imp-25', (1, 3, 0), 1, 146.4970875715329],
        ['imp-25', (1, 3, 0), 1, 210.66559554221405],
        ['imp-25', (1, 3, 0), 1, 292.0402096266422],
        ['imp-25', (1, 3, 1), 1, 44.06730240384354],
        ['imp-26', (1, 3, 0), 1, 35.48694482196721],
        ['imp-26', (1, 3, 0), 1, 370.688763713492],
        ['imp-26', (1, 3, 0), 1, 7.2285666914082665],
        ['imp-26', (1, 3, 0), 1, 472.7941167932397],
        ['imp-27', (1, 3, 0), 1, 408.4200980414033],
        ['imp-27', (1, 3, 0), 1, 8.328084600770142],
        ['imp-27', (1, 3, 0), 1, 205.6961467123334],
        ['imp-27', (1, 3, 0), 1, 186.6998594802865],
        ['imp-27', (1, 3, 0), 1, 145.74914432150592],
        ['imp-27', (1, 3, 1), 1, 179.2434588418143],
        ['imp-27', (1, 3, 1), 1, 21.310795380288504],
    ]

    dataset = synthetic_dataset.generate_counts_and_values_dataset_raw(
        rate_of_size_1=slice_1_rate,
        rate_of_size_2=slice_2_rate,
        average_conversion_per_impression=average_conversion_per_impression,
        impression_side_dimensions=impression_side_dimensions,
        conversion_side_dimensions=conversion_side_dimensions,
        value_mean=150.0,
        value_mode=20.0,
    )

    self.assertEqual(dataset, dataset_expected)


if __name__ == '__main__':
  absltest.main()
