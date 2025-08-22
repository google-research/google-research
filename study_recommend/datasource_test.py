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

"""Tests for DataSource class and related utilities."""

import collections
import itertools
from typing import Sequence

import unittest

import numpy as np
import pandas as pd
from study_recommend import datasource
from study_recommend import types
from study_recommend.utils import input_pipeline_utils


def get_sampler_test_data():
  classroom_indices: dict[
      types.ClassroomID, Sequence[types.StudentIndexRange]
  ] = {
      (1, 1): [(0, 2), (4, 1), (2, 1), (10, 1)],
      (2, 2): [(3, 1), (5, 2), (7, 3)],
  }
  student_ids = [1, 1, 2, 3, 4, 5, 5, 6, 6, 6, 7]
  titles = [
      [0] * 2,
      [1] * 4,
      [2] * 3,
      [3] * 3,
      [4] * 2,
      [5] * 4,
      [6] * 3,
      [7] * 2,
      [8] * 4,
      [9] * 3,
      [10] * 2,
  ]
  titles = np.array([np.array(x) for x in titles], dtype=object)
  return classroom_indices, titles, student_ids


def sample_exhaustively(sampler, rng, seq_len, max_samples=20):
  samples = []
  for _ in range(max_samples):
    try:
      classroom = sampler.sample_classroom(rng)
      samples.append(
          sampler.sample_indices(
              classroom, rng, seq_len=seq_len, using_separators=True
          )
      )
    except StopIteration:
      break
  return samples


class SamplerWithoutReplacementTest(unittest.TestCase):

  def test_sampled_without_replacement(self):
    """Test that no sequence index is return twice."""
    classroom_indices, titles, _ = get_sampler_test_data()
    sampler = datasource.ClassroomSamplerWithoutReplacement(
        classroom_indices, titles, with_automatic_reset=False
    )
    rng = np.random.RandomState(0)

    samples = sample_exhaustively(sampler, rng, seq_len=10)
    all_used_sequence_indices = list(itertools.chain.from_iterable(samples))

    self.assertLen(
        set(all_used_sequence_indices), len(all_used_sequence_indices)
    )

  def test_all_points_lengths_under_seq_len(self):
    classroom_indices, titles, _ = get_sampler_test_data()
    sampler = datasource.ClassroomSamplerWithoutReplacement(
        classroom_indices, titles, with_automatic_reset=False
    )
    rng = np.random.RandomState(0)
    seq_len = 10
    samples = sample_exhaustively(sampler, rng, seq_len=seq_len)

    for sample in samples:
      # The length of a composed sample is the sum of the constituent lengths
      # plus separator tokens
      len_sample = sum(titles[i].size for i in sample) + len(sample) - 1
      self.assertLessEqual(len_sample, seq_len)

  def test_all_consumed_when_stop_iteration(self):
    classroom_indices, titles, _ = get_sampler_test_data()
    sampler = datasource.ClassroomSamplerWithoutReplacement(
        classroom_indices, titles, with_automatic_reset=False
    )
    rng = np.random.RandomState(0)
    seq_len = 10
    samples = sample_exhaustively(sampler, rng, seq_len=seq_len)

    samples = sum(samples, start=[])
    self.assertLen(samples, len(titles))

  def test_ordered_within_student(self):
    classroom_indices, titles, student_ids = get_sampler_test_data()
    sampler = datasource.ClassroomSamplerWithoutReplacement(
        classroom_indices,
        titles,
        with_automatic_reset=False,
        ordered_within_student=True,
    )
    rng = np.random.RandomState(0)
    seq_len = 10

    samples = sample_exhaustively(sampler, rng, seq_len=seq_len)

    unraveled_index = collections.defaultdict(list)
    for i, student_id in enumerate(student_ids):
      unraveled_index[student_id].append(i)

    for sample in samples:
      for sequence_index in sample:
        student_id = student_ids[sequence_index]
        next_sequence_for_student = unraveled_index[student_id].pop(0)
        self.assertEqual(next_sequence_for_student, sequence_index)

  def test_all_indices_in_sample_same_classroom(self):
    classroom_indices, titles, student_ids = get_sampler_test_data()
    sampler = datasource.ClassroomSamplerWithoutReplacement(
        classroom_indices, titles, with_automatic_reset=False
    )
    rng = np.random.RandomState(0)
    seq_len = 10

    samples = sample_exhaustively(sampler, rng, seq_len=seq_len)

    def get_classroom(sequence_index):
      student_id = student_ids[sequence_index]
      for classroom, student_index_ranges in classroom_indices.items():
        for start_index, count in student_index_ranges:
          if start_index <= sequence_index < start_index + count:
            return classroom
      raise ValueError(
          f'student_id: {student_id} is not in any student index range'
      )

    for sample in samples:
      sample_classroom = get_classroom(sample[0])
      for sequence_index in sample:
        self.assertEqual(sample_classroom, get_classroom(sequence_index))

  def test_single_index_per_student(self):
    classroom_indices, titles, student_ids = get_sampler_test_data()
    sampler = datasource.ClassroomSamplerWithoutReplacement(
        classroom_indices, titles, with_automatic_reset=False
    )
    rng = np.random.RandomState(0)
    seq_len = 10

    samples = sample_exhaustively(sampler, rng, seq_len=seq_len)

    for sample in samples:
      sample_student_ids = map(lambda i: student_ids[i], sample)
      sample_student_ids = set(sample_student_ids)
      self.assertLen(sample_student_ids, len(sample))

  def test_raises_stop_iteration_upon_finish(self):
    classroom_indices, titles, _ = get_sampler_test_data()
    sampler = datasource.ClassroomSamplerWithoutReplacement(
        classroom_indices, titles, with_automatic_reset=False
    )
    rng = np.random.RandomState(0)
    with self.assertRaises(StopIteration):
      for _ in range(20):
        classroom = sampler.sample_classroom(rng)
        sampler.sample_indices(
            classroom, rng, seq_len=10, using_separators=True
        )

  def test_automatic_reset(self):
    classroom_indices, titles, _ = get_sampler_test_data()
    sampler = datasource.ClassroomSamplerWithoutReplacement(
        classroom_indices, titles, with_automatic_reset=True
    )
    rng = np.random.RandomState(0)

    # Test we can sample more datapoints that we have sequences and no
    # StopIteration is raised.
    for _ in range(20):
      classroom = sampler.sample_classroom(rng)
      sampler.sample_indices(classroom, rng, seq_len=10, using_separators=True)


class SamplerWithReplacementTest(unittest.TestCase):
  # def setUp(self):
  #   port = 5670
  #   debugpy.listen(('localhost', port))
  def test_all_points_lengths_under_seq_len(self):
    classroom_indices, titles, _ = get_sampler_test_data()
    sampler = datasource.ClassroomSamplerWithReplacement(
        classroom_indices, titles
    )
    rng = np.random.RandomState(0)
    seq_len = 10
    samples = sample_exhaustively(sampler, rng, seq_len=seq_len)

    for sample in samples:
      # The length of a composed sample is the sum of the constituent lengths
      # plus separator tokens
      len_sample = sum(titles[i].size for i in sample) + len(sample) - 1
      self.assertLessEqual(len_sample, seq_len)

  def test_all_indices_in_sample_same_classroom(self):
    classroom_indices, titles, student_ids = get_sampler_test_data()
    sampler = datasource.ClassroomSamplerWithReplacement(
        classroom_indices, titles
    )
    rng = np.random.RandomState(0)
    seq_len = 10

    samples = sample_exhaustively(sampler, rng, seq_len=seq_len)

    def get_classroom(sequence_index):
      student_id = student_ids[sequence_index]
      for classroom, student_index_ranges in classroom_indices.items():
        for start_index, count in student_index_ranges:
          if start_index <= sequence_index < start_index + count:
            return classroom
      raise ValueError(
          f'student_id: {student_id} is not in any student index range'
      )

    for sample in samples:
      sample_classroom = get_classroom(sample[0])
      for sequence_index in sample:
        self.assertEqual(sample_classroom, get_classroom(sequence_index))

  def test_single_index_per_student(self):
    classroom_indices, titles, student_ids = get_sampler_test_data()
    sampler = datasource.ClassroomSamplerWithReplacement(
        classroom_indices, titles
    )
    rng = np.random.RandomState(0)
    seq_len = 10

    samples = sample_exhaustively(sampler, rng, seq_len=seq_len)

    for sample in samples:
      sample_student_ids = map(lambda i: student_ids[i], sample)
      sample_student_ids = set(sample_student_ids)
      self.assertLen(sample_student_ids, len(sample))

  def test_sampling_with_replacement(self):
    """Test that we can draw more samples than there exist sequences."""
    classroom_indices, titles, _ = get_sampler_test_data()
    sampler = datasource.ClassroomSamplerWithReplacement(
        classroom_indices, titles
    )
    rng = np.random.RandomState(0)

    samples = sample_exhaustively(sampler, rng, seq_len=10, max_samples=20)

    self.assertLen(samples, 20)


class DatasourceTest(unittest.TestCase):

  def test_get_sample(self):
    classroom_indices, titles, student_ids = get_sampler_test_data()
    sampler = unittest.mock.create_autospec(
        datasource.SequenceSampler, instance=True
    )
    sampler.sample_classroom.return_value = (1, 1)
    sampler.sample_indices.return_value = [2, 4, 10]

    my_datasource = datasource.ClassroomGroupedDataSource(
        titles,
        dates=titles,
        classroom_lookup=classroom_indices,
        seq_len=10,
        separator_token=types.TokenIndex(50),
        sampler=sampler,
        student_ids=student_ids,
    )
    sample = my_datasource[[0]][0]

    reference_sample = {
        types.ModelInputFields.TITLES: np.array(
            [2, 2, 2, 50, 4, 4, 50, 10, 10, 0]
        ),
        types.ModelInputFields.TIMESTAMPS: np.array(
            [2, 2, 2, 50, 4, 4, 50, 10, 10, 0]
        ),
        types.ModelInputFields.STUDENT_IDS: np.array(
            [2, 2, 2, 50, 4, 4, 50, 7, 7, 0]
        ),
        types.ModelInputFields.INPUT_POSITIONS: np.array(
            [0, 1, 2, 0, 0, 1, 0, 0, 1, 0]
        ),
    }

    # Assert keys are equal
    self.assertCountEqual(list(sample), list(reference_sample))

    # Check arrays are equal
    for field in reference_sample:
      self.assertTrue(np.all(sample[field] == reference_sample[field]))

  def test_factory_method_without_replacement(self):
    """Assert caling factory method does not raise exception."""
    fields = types.StudentActivityFields
    student_activity_dataframe = pd.DataFrame(
        data=[
            [1, '2020-12-31', 'A'],
            [1, '2020-12-31', 'B'],
            [1, '2021-01-01', 'B'],
        ],
        columns=[
            fields.STUDENT_ID.value,
            fields.DATE.value,
            fields.BOOK_ID.value,
        ],
    ).groupby(fields.STUDENT_ID)
    student_info = pd.DataFrame(
        data=[[1, 2, 3]],
        columns=[
            fields.STUDENT_ID.value,
            fields.SCHOOL_ID.value,
            fields.GRADE_LEVEL.value,
        ],
    )
    my_datasource, vocab = (
        datasource.ClassroomGroupedDataSource.datasource_from_grouped_activity_dataframe(
            student_activity_dataframe,
            student_info,
            seq_len=10,
            vocab_size=10,
            with_replacement=False,
        )
    )
    self.assertIsInstance(my_datasource, datasource.ClassroomGroupedDataSource)
    self.assertIsInstance(vocab, input_pipeline_utils.Vocabulary)

  def test_factory_method_with_replacement(self):
    """Assert caling factory method does not raise exception."""
    fields = types.StudentActivityFields
    student_activity_dataframe = pd.DataFrame(
        data=[
            [1, '2020-12-31', 'A'],
            [1, '2020-12-31', 'B'],
            [1, '2021-01-01', 'B'],
        ],
        columns=[
            fields.STUDENT_ID.value,
            fields.DATE.value,
            fields.BOOK_ID.value,
        ],
    ).groupby(fields.STUDENT_ID)
    student_info = pd.DataFrame(
        data=[[1, 2, 3]],
        columns=[
            fields.STUDENT_ID.value,
            fields.SCHOOL_ID.value,
            fields.GRADE_LEVEL.value,
        ],
    )
    my_datasource, vocab = (
        datasource.ClassroomGroupedDataSource.datasource_from_grouped_activity_dataframe(
            student_activity_dataframe,
            student_info,
            seq_len=10,
            vocab_size=10,
            with_replacement=True,
        )
    )
    self.assertIsInstance(my_datasource, datasource.ClassroomGroupedDataSource)
    self.assertIsInstance(vocab, input_pipeline_utils.Vocabulary)

  def test_grade_level_attribute_from_factory_method(self):
    """Check correctness of grade level array when grade level exists."""
    fields = types.StudentActivityFields
    student_activity_dataframe = pd.DataFrame(
        data=[
            [1, '2020-12-31', 'A'],
            [1, '2020-12-31', 'B'],
            [2, '2021-01-01', 'B'],
            [2, '2022-01-01', 'C'],
        ],
        columns=[
            fields.STUDENT_ID.value,
            fields.DATE.value,
            fields.BOOK_ID.value,
        ],
    ).groupby(fields.STUDENT_ID)
    student_info = pd.DataFrame(
        data=[[1, 2, 3], [2, 2, 3]],
        columns=[
            fields.STUDENT_ID.value,
            fields.SCHOOL_ID.value,
            fields.GRADE_LEVEL.value,
        ],
    )
    my_datasource, _ = (
        datasource.ClassroomGroupedDataSource.datasource_from_grouped_activity_dataframe(
            student_activity_dataframe,
            student_info,
            seq_len=10,
            vocab_size=10,
            with_replacement=True,
            max_grade_level=5,
        )
    )
    grade_level = my_datasource[[1]][0][types.ModelInputFields.GRADE_LEVELS]
    # The composed entry will consist of sequences from two students in
    # grade 3 joint by separator token.
    self.assertEqual(grade_level.item(), 3)

  def test_grade_level_coerced_to_zero_attribute_from_factory_method(self):
    """Check grade levels > max_grade_level or <0 are coerced to zero."""
    fields = types.StudentActivityFields
    student_activity_dataframe = pd.DataFrame(
        data=[
            [1, '2020-12-31', 'A'],
            [1, '2020-12-31', 'B'],
            [2, '2021-01-01', 'B'],
            [2, '2022-01-01', 'C'],
        ],
        columns=[
            fields.STUDENT_ID.value,
            fields.DATE.value,
            fields.BOOK_ID.value,
        ],
    ).groupby(fields.STUDENT_ID)
    student_info = pd.DataFrame(
        data=[[1, 2, 3], [2, 2, 3]],
        columns=[
            fields.STUDENT_ID.value,
            fields.SCHOOL_ID.value,
            fields.GRADE_LEVEL.value,
        ],
    )
    my_datasource, _ = (
        datasource.ClassroomGroupedDataSource.datasource_from_grouped_activity_dataframe(
            student_activity_dataframe,
            student_info,
            seq_len=10,
            vocab_size=10,
            with_replacement=True,
            max_grade_level=2,
        )
    )
    grade_level = my_datasource[[1]][0][types.ModelInputFields.GRADE_LEVELS]

    self.assertEqual(grade_level.item(), 0)

  def test_iterator_with_replacement_with_sharding(self):
    """Test the sharding behaviour of the iterator with replacement.

    Assert each shard gets a unique subset of datapoint indices and that
    the shards together cover consecutive datapoints.
    """

    reference_samples_per_shard = {
        0: [
            [0, 4, 8, 12, 16],
            [20, 24, 28, 32, 36],
            [40, 44, 48, 52, 56],
            [60, 64, 68, 72, 76],
        ],
        1: [
            [1, 5, 9, 13, 17],
            [21, 25, 29, 33, 37],
            [41, 45, 49, 53, 57],
            [61, 65, 69, 73, 77],
        ],
        2: [
            [2, 6, 10, 14, 18],
            [22, 26, 30, 34, 38],
            [42, 46, 50, 54, 58],
            [62, 66, 70, 74, 78],
        ],
        3: [
            [3, 7, 11, 15, 19],
            [23, 27, 31, 35, 39],
            [43, 47, 51, 55, 59],
            [63, 67, 71, 75, 79],
        ],
    }

    for shard_index in range(4):
      my_datasource = unittest.mock.MagicMock()

      iterator = datasource.iterator_with_replacement(
          my_datasource,
          batch_size=5,
          num_shards=4,
          shard_index=shard_index,
          num_batches=4,
      )
      for _ in iterator:
        pass
      reference_calls = reference_samples_per_shard[shard_index]
      self.assertLen(my_datasource.__getitem__.call_args_list, 4)
      for i, call in enumerate(my_datasource.__getitem__.call_args_list):
        args, _ = call
        indices = list(args[0])
        self.assertEqual(indices, reference_calls[i])


if __name__ == '__main__':
  unittest.main()
