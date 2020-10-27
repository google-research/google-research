# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Tests for beam_utils."""

from absl.testing import absltest
import apache_beam as beam
import apache_beam.testing.util as beam_testing

from etcmodel.models.openkp import beam_utils


class BeamUtilsTest(absltest.TestCase):

  def test_singletons_to_dict(self):

    def pipeline(root):
      number_singleton = root | 'CreateNumber' >> beam.Create([3.14])
      string_singleton = root | 'CreateString' >> beam.Create(['test'])
      list_singleton = root | 'CreateList' >> beam.Create([[1, 2, 3]])

      expected = [dict(number=3.14, string='test', list=[1, 2, 3])]
      result = beam_utils.singletons_to_dict(
          number=number_singleton, string=string_singleton, list=list_singleton)

      beam_testing.assert_that(result, beam_testing.equal_to(expected))

    p = beam.Pipeline()
    pipeline(p)
    p.run().wait_until_finish()

  def test_singletons_to_dict_raises_if_not_singleton(self):

    def pipeline(root):
      numbers_not_singleton = root | 'CreateNumber' >> beam.Create([3.14, -1])
      string_singleton = root | 'CreateString' >> beam.Create(['test'])
      list_singleton = root | 'CreateList' >> beam.Create([[1, 2, 3]])

      return beam_utils.singletons_to_dict(
          number=numbers_not_singleton,
          string=string_singleton,
          list=list_singleton)

    with self.assertRaises(ValueError):
      p = beam.Pipeline()
      pipeline(p)
      p.run().wait_until_finish()

  def test_singletons_to_dict_raises_for_empty_kwargs(self):

    def pipeline(unused_root):
      return beam_utils.singletons_to_dict()

    with self.assertRaises(ValueError):
      p = beam.Pipeline()
      pipeline(p)
      p.run().wait_until_finish()


if __name__ == '__main__':
  absltest.main()
