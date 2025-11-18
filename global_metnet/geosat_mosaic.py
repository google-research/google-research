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

"""Utilities related to mosaic satellite data."""

from typing import Optional

from global_metnet import normalizers

mosaic_means = {
    'C00': 11.529593010925941,
    'C01': 10.738041817884756,
    'C02': 8.17022257047685,
    'C03': 9.43819732569827,
    'C04': 1.3411066645053442,
    'C05': 6.700432552769698,
    'C06': 5.6050104537553365,
    'C07': 274.16567601352824,
    'C08': 233.05963092544008,
    'C09': 239.55619553100306,
    'C10': 247.6793465247521,
    'C11': 265.30717181521516,
    'C12': 248.00248450100622,
    'C13': 266.191669456038,
    'C14': 266.9443420562457,
    'C15': 265.07119609663425,
    'C16': 252.4215450626244,
}
mosaic_stds = {
    'C00': 15.803094404381264,
    'C01': 14.997628284572462,
    'C02': 12.946044274726331,
    'C03': 15.009583126880035,
    'C04': 4.530947238428484,
    'C05': 11.157139997897897,
    'C06': 10.210652434062736,
    'C07': 21.775293856263502,
    'C08': 10.356291668554702,
    'C09': 12.005967401841446,
    'C10': 14.564458608706836,
    'C11': 22.68601589280422,
    'C12': 18.352325078610868,
    'C13': 22.77347858453953,
    'C14': 23.599256902981175,
    'C15': 23.440982105698225,
    'C16': 18.185488090719506,
}

mosaic_min = {
    'C00': -4.17578125,
    'C01': -1.140625,
    'C02': -4.046875,
    'C03': -4.08984375,
    'C04': -4.0703125,
    'C05': -4.1484375,
    'C06': -4.08984375,
    'C07': -3.453125,
    'C08': -2.2265625,
    'C09': 119.9375,
    'C10': -0.493408203125,
    'C11': -0.1790771484375,
    'C12': -0.05999755859375,
    'C13': 89.6875,
    'C14': -0.64111328125,
    'C15': -0.408447265625,
    'C16': -0.5791015625,
}

mosaic_max = {
    'C00': 8560.0,
    'C01': 119.25,
    'C02': 2062.0,
    'C03': 140.0,
    'C04': 3968.0,
    'C05': 8496.0,
    'C06': 126.9375,
    'C07': 412.25,
    'C08': 329.25,
    'C09': 330.0,
    'C10': 331.25,
    'C11': 341.25,
    'C12': 328.5,
    'C13': 341.25,
    'C14': 343.75,
    'C15': 343.75,
    'C16': 325.25,
}

mosaic_channels = [
    'C00',
    'C01',
    'C02',
    'C03',
    'C04',
    'C05',
    'C06',
    'C07',
    'C08',
    'C09',
    'C10',
    'C11',
    'C12',
    'C13',
    'C14',
    'C15',
    'C16',
]


def create_mosaic_normalizer(
    channels = None,
):
  if channels is None:
    channels = mosaic_channels
  return normalizers.Normalizer(
      center=[mosaic_means[c] for c in channels],
      scale=[mosaic_stds[c] for c in channels],
      lower_bound=[mosaic_min[c] for c in channels],
      upper_bound=[mosaic_max[c] for c in channels],
  )


def create_mosaic_normalizer_no_clipping(
    channels = None,
):
  if channels is None:
    channels = mosaic_channels
  return normalizers.Normalizer(
      center=[mosaic_means[c] for c in channels],
      scale=[mosaic_stds[c] for c in channels],
  )
