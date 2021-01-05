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

"""Tests for rewrite_sparql.py."""
from absl.testing import absltest

from cfq_pt_vs_sa import rewrite_sparql


class RewriteSparqlTest(absltest.TestCase):

  def assert_rewrite(self, query, expected,
                     f):
    expected = 'COUNT ' + expected
    query = 'SELECT count { ' + query + ' }'
    actual = rewrite_sparql.rewrite(query, f)
    self.assertEqual(expected, actual)

  def test_group_subjects(self):
    f = rewrite_sparql.SimplifyFunction.GROUP_SUBJECTS
    # No rewrite.
    query = 's1 r1 o1 . s2 r2 o2 . s3 r1 o2'
    expected = 's1 { r1 o1 } s2 { r2 o2 } s3 { r1 o2 }'
    self.assert_rewrite(query, expected, f)

    # Simple rewrite.
    query = 's1 r1 o1 . s1 r1 o2'
    expected = 's1 { r1 o1 . r1 o2 }'
    self.assert_rewrite(query, expected, f)

    # A bit more complex rewrite.
    query = 's1 r1 o1 . s1 r2 o2 . s2 r1 o1'
    expected = 's1 { r1 o1 . r2 o2 } s2 { r1 o1 }'
    self.assert_rewrite(query, expected, f)

  def test_group_subjects_and_objects(self):
    f = rewrite_sparql.SimplifyFunction.GROUP_SUBJECTS_AND_OBJECTS
    # No rewrite.
    query = 's1 r1 o1 . s2 r2 o2 . s3 r1 o2'
    expected = 's1 { r1 { o1 } } s2 { r2 { o2 } } s3 { r1 { o2 } }'
    self.assert_rewrite(query, expected, f)

    # Rewrite.
    query = 's1 r1 o1 . s1 r1 o2 . s2 r2 o3 . s3 r2 o3'
    expected = 's1 { r1 { o1 o2 } } s2 { r2 { o3 } } s3 { r2 { o3 } }'
    self.assert_rewrite(query, expected, f)


if __name__ == '__main__':
  absltest.main()
