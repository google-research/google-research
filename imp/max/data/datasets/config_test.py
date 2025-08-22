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

"""Tests for all known datasets."""
import collections

from absl.testing import absltest
from absl.testing import parameterized

from imp.max.core import constants
from imp.max.data import config as data_config
from imp.max.data.datasets import config as datasets_config
from imp.max.data.datasets import dataloader
from imp.max.data.datasets import factories

Modality = constants.Modality
DataFeatureType = constants.DataFeatureType
DataFeatureRoute = constants.DataFeatureRoute
_VISION_TEXT = (Modality.VISION, Modality.TEXT)
_AUDIO_TEXT = (Modality.SPECTROGRAM, Modality.WAVEFORM, Modality.TEXT)


def get_lowest_dict_entries(element):
  if isinstance(element, dict):
    for element_value in element.values():
      yield from get_lowest_dict_entries(element_value)
  else:
    yield element


# Add your local dataset tests below


if __name__ == '__main__':
  absltest.main()
