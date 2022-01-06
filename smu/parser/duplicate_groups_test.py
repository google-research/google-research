# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Tests for query_sqlite."""

import os
import pandas as pd

from absl.testing import absltest

from smu.parser import duplicate_groups

TESTDATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'testdata')


class ParseDuplicatesFileTest(absltest.TestCase):

  def test_basic(self):
    df = duplicate_groups.parse_duplicates_file(
        os.path.join(TESTDATA_PATH, 'small.equivalent_isomers.dat'))
    pd.testing.assert_frame_equal(
        pd.DataFrame(
            columns=[
                'name1', 'stoich1', 'btid1', 'shortconfid1', 'confid1', 'name2',
                'stoich2', 'btid2', 'shortconfid2', 'confid2'
            ],
            data=[
                [
                    'x07_c2n2o2fh3.224227.004', 'c2n2o2fh3', 224227, 4,
                    224227004, 'x07_c2n2o2fh3.224176.005', 'c2n2o2fh3', 224176,
                    5, 224176005
                ],
                [
                    'x07_c2n2o2fh3.260543.005', 'c2n2o2fh3', 260543, 5,
                    260543005, 'x07_c2n2o2fh3.224050.001', 'c2n2o2fh3', 224050,
                    1, 224050001
                ],
            ]),
        df,
        check_like=True)


if __name__ == '__main__':
  absltest.main()
