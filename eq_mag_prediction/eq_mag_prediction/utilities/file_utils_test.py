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

"""Tests for utilities.file_utils."""

import tempfile
import numpy as np
import xarray as xr
from absl.testing import absltest
from eq_mag_prediction.utilities import file_utils


class XarraySaveLoadTest(absltest.TestCase):

  def test_save_load_dataset(self):
    random_state = np.random.RandomState(seed=1984)
    dataset = xr.Dataset(
        {
            'a': (['x', 'y'], random_state.randn(4, 20)),
            'b': (
                ['x', 'y', 'z'],
                random_state.randint(low=-100, high=100, size=(4, 20, 30)),
            ),
        },
        coords={'x': (['x'], ['a', 'b', 'c', 'd']), 'y': (['y'], range(20))},
        attrs={'foo': 'bar'},
    )
    tmpfile = tempfile.NamedTemporaryFile()
    file_utils.save_xr_dataset(tmpfile.name, dataset)
    self.assertTrue(dataset.identical(file_utils.load_xr_dataset(tmpfile.name)))


if __name__ == '__main__':
  absltest.main()
