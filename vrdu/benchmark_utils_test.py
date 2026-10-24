# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Tests for missing-groundtruth handling in vrdu.benchmark_utils."""

import types

from absl.testing import absltest

from vrdu import benchmark_utils


def _benchmark(annotations, test_filenames):
  """Builds the minimal benchmark surface the scorer reads."""
  return types.SimpleNamespace(
      annotations=annotations,
      test_filenames=list(test_filenames),
      entity_name_to_match_func={},
      appearance_pattern_to_entity_names={
          'repeated': [],
          'unrepeated': [],
      },
  )


class MissingGroundtruthTest(absltest.TestCase):
  """An entity name absent from the groundtruth scores -1, not a free recall."""

  def test_absent_entity_name_scores_minus_one(self):
    benchmark = _benchmark(annotations={}, test_filenames=[])
    result = benchmark_utils.evaluate_for_target_entity_names(
        {'results': {}}, benchmark, {'unrepeated_entity_name'}
    )
    self.assertEqual(result['precision'], -1)
    self.assertEqual(result['recall'], -1)
    self.assertEqual(result['f1'], -1)

  def test_absent_entity_name_scores_minus_one_with_extractions(self):
    """A spurious extraction does not turn an unscorable entity into 0/1/0."""
    benchmark = _benchmark(annotations={}, test_filenames=[])
    # The document is not in `annotations`, so there is no groundtruth for it.
    result = benchmark_utils.evaluate_for_target_entity_names(
        {'results': {'doc.txt': []}}, benchmark, {'unrepeated_entity_name'}
    )
    self.assertEqual(result['recall'], -1)


class MacroAverageTest(absltest.TestCase):
  """The macro average skips -1 values and survives an all-missing corpus."""

  def test_all_missing_does_not_divide_by_zero(self):
    benchmark = types.SimpleNamespace(
        annotations={},
        test_filenames=[],
        entity_name_to_match_func={},
        appearance_pattern_to_entity_names={
            'repeated': [],
            'unrepeated': ['entity_a', 'entity_b'],
        },
    )
    results = benchmark_utils.evaluate({'meta': {}, 'results': {}},
                                       benchmark)
    self.assertEqual(results['macro_precision'], -1)
    self.assertEqual(results['macro_recall'], -1)
    self.assertEqual(results['macro_f1'], -1)
    # Per-entity entries still report the sentinel.
    self.assertEqual(results['entity_a_recall'], -1)


if __name__ == '__main__':
  absltest.main()
