# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

import re

from absl.testing import absltest

from eeg_modelling.eeg_viewer import lookup

CHANNEL_KEYS = ['eeg_channel/EEG feat_1-REF/samples',
                'eeg_channel/EEG feat_2/samples']

CHANNEL_MATCHERS = [
    re.compile(r'eeg_channel/EEG (\w+)(-\w+)*/samples'),
    re.compile(r'eeg_channel/POL (EKG\w+)/samples'),
    re.compile(r'eeg_channel/(\w+)/samples'),
    re.compile(r'eeg_channel/EEG (\w+)(-\w+)*/resampled_samples'),
    re.compile(r'(seizure_bin)ary_per_sec'),
]


class LookupTest(absltest.TestCase):

  def setUp(self):
    super(LookupTest, self).setUp()
    self.test_lookup = lookup.Lookup(CHANNEL_KEYS, CHANNEL_MATCHERS)

  def testGetKeyFromIndex(self):
    self.assertEqual('eeg_channel/EEG feat_1-REF/samples',
                     self.test_lookup.GetKeyFromIndex(0))
    self.assertEqual('eeg_channel/EEG feat_2/samples',
                     self.test_lookup.GetKeyFromIndex(1))

  def testGetKeyFromIndexReturnsNone(self):
    self.assertIsNone(self.test_lookup.GetKeyFromIndex(3))

  def testGetIndexFromShorthand(self):
    self.assertEqual('0', self.test_lookup.GetIndexFromShorthand('FEAT_1'))
    self.assertEqual('1', self.test_lookup.GetIndexFromShorthand('FEAT_2'))

  def testGetIndexFromShorthandReturnsNone(self):
    self.assertIsNone(self.test_lookup.GetIndexFromShorthand('FEAT_3'))

  def testGetShorthandFromKey(self):
    self.assertEqual('FEAT_1', self.test_lookup.GetShorthandFromKey(
        'eeg_channel/EEG feat_1-REF/samples'))
    self.assertEqual('FEAT_2', self.test_lookup.GetShorthandFromKey(
        'eeg_channel/EEG feat_2/samples'))

  def testGetShorthandFromKeyReturnsNone(self):
    self.assertIsNone(self.test_lookup.GetShorthandFromKey(
        'eeg_channel/EEG feat_3/samples'))


if __name__ == '__main__':
  absltest.main()
