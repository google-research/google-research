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

"""Implementation of DataSource objecy to generate training data.

The ClassroomGroupedDatasource datasource generates datapoints by sampling
sequences of multiple students within a classroom then composing them to make
a single datapoint. The ClassroomGroupedDatasource provides sampling with
and without replacement. The without replacement sampler can be used for doing
a single exhuastive pass over a dataset for evaluation.
"""

import abc
import collections
from collections.abc import Callable, Mapping, Sequence
import itertools
import sys
from typing import Any, Optional

import numpy as np
import numpy.typing as np_typing
import pandas as pd
import pandas.core.groupby as pd_groupby
from study_recommend import types
from study_recommend.utils import input_pipeline_utils as utils


class SequenceSampler(abc.ABC):
  """Base class for sampling sequences to compose data points."""

  def __init__(self):
    self._titles = []

  @abc.abstractmethod
  def sample_classroom(self, rng):
    """Sample a classroom to generate a data point from.

    Classrooms are sampled with probability proportionated to the number of
    students in the classroom.
    Args:
      rng: A random generator to supply randomness for sampling.

    Returns:
      The ClassroomID of the sampled classroom
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def sample_indices(
      self,
      classroom,
      rng,
      seq_len,
      using_separators,
      start_with_separator = False,
  ):
    """Sample a group of indices that together form a single sample.

    The sampled indices will correspond to title-interaction sequences coming
    from distinct students in the same classroom. The total length of sequence
    formed by concatenating the sequences (and separator tokens if used) will be
    less than seq_len. The indices are being sampled without replacement.

    Args:
      classroom: The classroom from which to sample
      rng: A random generator to provide randomness for sampling
      seq_len: The maximum length of the overall sequence.
      using_separators: Flag to set whether or not separator tokens are used.
      start_with_separator: Whether or not to start the sequence with a
        separator token.

    Returns:
      sampled_sequences: A list of integer indices of the sequences that form
      the generated data point.
    """

  def _get_proposed_seq_len(
      self,
      current_seq_len,
      sequence_index,
      using_separators,
      seq_len,
  ):
    """Get the overall length of the data point if we add sequence_index to it.

    Args:
      current_seq_len: The current length of the sequence.
      sequence_index: The index of the proposed sequence to add.
      using_separators: A flag that indicates whether constituent sequences are
        separated with separator tokens.
      seq_len: The maximum allowed overall sequence length.

    Returns:
      The new proposed sequence length. Returns None if this would be > seq_len.
    """
    if using_separators and current_seq_len:
      proposed_seq_len = current_seq_len + 1
    else:
      proposed_seq_len = current_seq_len

    proposed_seq_len += self._titles[sequence_index].size

    if proposed_seq_len > seq_len:
      # We cannot sample this sequence. Reject this sample and try again.
      return None
    return proposed_seq_len

  @classmethod
  def _classroom_size(
      cls, student_index_ranges
  ):
    """Return the total number of sequences in a classroom.

    Args:
      student_index_ranges: The student index ranges for all students in the
        classroom.

    Returns:
      size: Number of sequences in the classroom.
    """
    return sum(map(lambda x: x[1], student_index_ranges))

  @abc.abstractmethod
  def is_ordered_within_student(self):
    """Returns True if the datasource yields sublists within students in order."""
    raise NotImplementedError()


class ClassroomSamplerWithoutReplacement(SequenceSampler):
  """Sample without replacement the indices of title sequences.

  The indices of student-title interaction sequences are sampled in groups such
  that the sequences  in each group 1) all come from the same classroom, 2) no
  two sequences in a group come from the same student, 3) the sum of the lengths
  of all the sequences in the group is less than provided seq_len.
  """

  def __init__(
      self,
      classroom_indices,
      titles,
      with_automatic_reset = True,
      ordered_within_student = True,
  ):
    """Initialize the sampler.

    Args:
      classroom_indices: A dictionary that maps each classroomID to the of
        StudentIndexRanges of all student the given classroom.
      titles: A jagged np.ndarray of titles. The i'th entry is an np.ndarray of
        titles read by the student whose StudentIndexRange contains the index i.
      with_automatic_reset: When true, when the sampler is completely depleted
        through sampling without replacement it will automatically reset and
        continue generating samples. If false a StopIteration exception will be
        raised when sampled from a depleted sampler.
      ordered_within_student: When True, we will randomly sample students per
        the described method, and then return the data with the earliest
        timestamps that have not been yet sampled for that student. When False
        we sample a random sequence belonging the student.

    Raises:
      RuntimeError: If python version < 3.7. This is because deterministic
        dict ordering is needed for reproducibility.
    """
    super().__init__()
    self._classroom_indices = classroom_indices
    self._titles = titles
    self._with_automatic_reset = with_automatic_reset
    self._ordered_within_student = ordered_within_student
    min_python = (3, 7)
    if sys.version_info < min_python:
      # See ClassroomSamplerWithoutReplacement.sample_classroom
      raise RuntimeError(
          'This classs assumes dicts are ordered (for classroom_indices), use'
          ' python 3.7+.'
      )
    self.reset()

  def reset(self):
    """Reset the cache of which sequences have already been sampled."""
    # self._student_by_classroom_indices map ClassroomIDs to the StartIndices of
    # the students in the classroom.
    self._student_by_classroom_indices: dict[
        types.ClassroomID, np_typing.NDArray[types.StartIndex]
    ] = {}
    # self._student_by_classroom_masks is used to keep track of which students
    # have had all their data exhausted by sampling. We do this to avoid
    # sampling from them again.
    self._student_by_classroom_masks: dict[
        types.ClassroomID, np_typing.NDArray[bool]
    ] = {}

    # self._student_sequence_unraveled_indices maps each StartIndex to an
    # ndarray containing the indices between StartIndex:StartIndex + NumItems.
    self._student_sequence_unraveled_indices: dict[
        types.StartIndex, np_typing.NDArray[int]
    ] = {}
    # self._student_segment_indices_mask is used to keep track of which indices
    # have already been sampled so we can sample without replacement.
    self._student_segment_indices_mask: dict[
        types.StartIndex, np_typing.NDArray[bool]
    ] = {}

    for classroom, student_index_ranges in self._classroom_indices.items():
      for start_index, length in student_index_ranges:
        self._student_sequence_unraveled_indices[start_index] = np.arange(
            start_index, start_index + length, dtype=np.int32
        )
        self._student_segment_indices_mask[start_index] = np.zeros_like(
            self._student_sequence_unraveled_indices[start_index],
            dtype=np.bool_,
        )

      # self._student_by_classroom_indices maps ClassroomIDs to students, where
      # each student is represented by the StartIndex of the student's data.
      self._student_by_classroom_indices[classroom] = np.array(
          [start_index for start_index, _ in student_index_ranges],
          dtype=np.int32,
      )
      # self._student_by_classroom_masks is used to keep track of which students
      # have data which is still yet to be exhausted for sampling.
      self._student_by_classroom_masks[classroom] = np.zeros_like(
          self._student_by_classroom_indices[classroom], dtype=np.bool_
      )

    max_index = max(
        map(
            len,
            itertools.chain(
                self._student_sequence_unraveled_indices.values(),
                self._student_by_classroom_indices.values(),
            ),
        )
    )
    # A utility variable that will be used. We want to avoid repeated calls to
    # np.arange.
    self.arange = np.arange(max_index, dtype=np.int32)

  def sample_classroom(self, rng):
    """Return a classroom that has not been exhausted by sampling."""
    # We assume self._student_by_classroom_masks has deterministic iteration
    # order. This is the case in python 3.7+
    for classroom in self._student_by_classroom_masks:
      # Return a classroom that has not been exhausted by sampling and had its
      # mask removed.
      return classroom
    # If all data has been sampled and with_reset=True, we reset the sampling
    # history and return the first sample.
    if self._with_automatic_reset:
      self.reset()
      return self.sample_classroom(rng)
    else:
      # All data has been sampled and no automatic reset, raise StopIteration
      raise StopIteration

  def sample_indices(
      self,
      classroom,
      rng,
      seq_len,
      using_separators,
      start_with_separator = False,
  ):
    """Sample a group of indices that together form a single sample.

    The sampled indices will correspond to title-interaction sequences coming
    from distinct students in the same classroom. The total length of sequence
    formed by concatenating the sequences (and separator tokens if used) will be
    less than seq_len. The indices are being sampled without replacement.

    Args:
      classroom: The classroom from which to sample
      rng: A random generator to provide randomness for sampling
      seq_len: The maximum length of the overall sequence.
      using_separators: Flag to set whether or not separator tokens are used.
      start_with_separator: Whether or not to start the sequence with a
        separator token.

    Returns:
      sampled_sequences: A list of integer indices of the sequences that form
      the generated data point.
    """
    if start_with_separator:
      seq_len -= 1

    classroom_mask = self._student_by_classroom_masks[classroom]
    # We get the indices of the students within the classroom that have not been
    # sampled. These indices are in [0,n) where n is the number of students
    # in the classroom.
    student_classroom_indices = np.ma.masked_array(
        self.arange[: classroom_mask.size], classroom_mask
    ).compressed()

    if not student_classroom_indices.size:
      # No unsampled data available in the specified classroom.
      return []

    sampled_sequences = []
    # If sampling a certain student-interaction sequence causes the overall
    # sequence to go over length we reject it. If we reject 3 sequences we
    # stop sampling and return the sequences we have selected so far.
    rejects = 0
    current_seq_len = 0

    def need_to_continue_sampling():
      return rejects < 3 and current_seq_len < seq_len - 1

    while need_to_continue_sampling():
      student_within_classroom_index = rng.choice(student_classroom_indices)
      # We get the StartIndex of the student we have chosen to sample from.
      sampled_start_index = self._student_by_classroom_indices[classroom][
          student_within_classroom_index
      ]

      # We get the indices of sequences pertaining to the current student that
      # have yet to sampled. within_student_indices are values in [0, n) where
      # n is the number of sequences pertaining to the current student.
      within_student_mask = self._student_segment_indices_mask[
          sampled_start_index
      ]
      indices = self.arange[: within_student_mask.size]
      within_student_indices = np.ma.masked_array(
          indices, within_student_mask
      ).compressed()

      if self._ordered_within_student:
        sequence_within_student_index = within_student_indices.min()
      else:
        sequence_within_student_index = rng.choice(within_student_indices)

      # We convert the within student relative index to an absolute index.
      sequence_index = self._student_sequence_unraveled_indices[
          sampled_start_index
      ][sequence_within_student_index]

      # We check if sampling the sequence with index sequence_index would make
      # make the the length of the overall sequence > seq_len

      proposed_seq_len = self._get_proposed_seq_len(
          current_seq_len, sequence_index, using_separators, seq_len
      )

      if proposed_seq_len is None:
        rejects += 1
        continue

      current_seq_len = proposed_seq_len
      # Add sequence_index to sampled_sequences
      sampled_sequences.append(sequence_index)

      # Mark sequence_index as sampled.
      self._mark_as_sampled(
          classroom=classroom,
          student_start_index=sampled_start_index,
          student_classroom_index=student_within_classroom_index,
          sequence_within_student_index=sequence_within_student_index,
      )
      # Remove the student from student_classroom_indices
      student_classroom_indices = student_classroom_indices[
          student_classroom_indices != student_within_classroom_index
      ]
      if not student_classroom_indices.size:
        break

    return sampled_sequences

  def _mark_as_sampled(
      self,
      classroom,
      student_start_index,
      student_classroom_index,
      sequence_within_student_index,
  ):
    """Mark sequence_index as sampled."""

    self._student_segment_indices_mask[student_start_index][
        sequence_within_student_index
    ] = 1
    if self._student_segment_indices_mask[student_start_index].all():
      self._student_by_classroom_masks[classroom][student_classroom_index] = 1
      if self._student_by_classroom_masks[classroom].all():
        del self._student_by_classroom_masks[classroom]

  def is_ordered_within_student(self):
    """Returns True if the datasource yields sublists within students in order."""
    return self._ordered_within_student


class ClassroomSamplerWithReplacement(ClassroomSamplerWithoutReplacement):
  """Sample with replacement the indices of title sequences.

  The indices of student-title interaction sequences are sampled in groups such
  that the sequences  in each group 1) all come from the same classroom, 2) no
  two sequences in a group come from the same student, 3) the sum of the lengths
  of all the sequences in the group is less than provided seq_len.
  """

  def __init__(
      self,
      classroom_indices,
      titles,
  ):
    super().__init__(classroom_indices, titles)
    self._classroom_ids = list(sorted(classroom_indices.keys()))
    self._classroom_weights = np.array(
        [len(classroom_indices[x]) for x in self._classroom_ids],
        dtype=np.float32,
    )
    self._classroom_weights /= self._classroom_weights.sum()

  def sample_classroom(self, rng):
    """Sample a classroom to generate a data point from.

    Classrooms are sampled with probability proportionated to the number of
    students in the classroom.
    Args:
      rng: A random generator to supply randomness for sampling.

    Returns:
      The ClassroomID of the sampled classroom
    """
    i = rng.choice(len(self._classroom_ids), p=self._classroom_weights)
    return self._classroom_ids[i]

  def sample_indices(
      self,
      classroom,
      rng,
      seq_len,
      using_separators,
      start_with_separator = False,
  ):
    """Sample a group of indices that together form a single sample.

    The sampled indices will correspond to title-interaction sequences coming
    from distinct students in the same classroom. The total length of sequence
    formed by concatenating the sequences (and separator tokens if used) will be
    less than seq_len. The indices are being sampled without replacement.

    Args:
      classroom: The classroom from which to sample
      rng: A random generator to provide randomness for sampling
      seq_len: The maximum length of the overall sequence.
      using_separators: Flag to set whether or not separator tokens are used.
      start_with_separator: Whether or not to start the sequence with a
        separator token.

    Returns:
      sampled_sequences: A list of integer indices of the sequences that form
      the generated data point.
    """
    if start_with_separator:
      seq_len -= 1

    # Obtain the StartIndexes for each student in the classroom.
    student_start_indexes = self._student_by_classroom_indices[classroom]

    # The indexes of the sequences for the current data point.
    sequence_indices = []

    # If sampling a certain student-interaction sequence causes the overall
    # sequence to go over length we reject it. If we reject 3 sequences we
    # stop sampling and return the sequences we have selected so far.
    rejects = 0
    current_seq_len = 0

    def need_to_continue_sampling():
      return rejects < 3 and current_seq_len < seq_len - 1

    while need_to_continue_sampling():
      # Sample a student from the classroom
      student_start_index = rng.choice(student_start_indexes)

      # Sample a sequence of data relating to the student
      proposed_sequence_index = rng.choice(
          self._student_sequence_unraveled_indices[student_start_index]
      )

      proposed_seq_len = self._get_proposed_seq_len(
          current_seq_len, proposed_sequence_index, using_separators, seq_len
      )
      if proposed_seq_len is None:
        rejects += 1
        continue

      current_seq_len = proposed_seq_len
      # Remove the student from student_start_indices
      student_start_indexes = student_start_indexes[
          student_start_indexes != student_start_index
      ]
      sequence_indices.append(proposed_sequence_index)
      current_seq_len = proposed_seq_len

      if not student_start_indexes.size:
        break

    return sequence_indices

  def is_ordered_within_student(self):
    """Returns True if the datasource yields sublists within students in order."""
    # Sampler with replacement will repeat sublists and hence a total order
    # is not defined.
    return False


class ClassroomGroupedDataSource:
  """A class that generates datapoint in accordance to the STUDY preprocessing.

  This classes __getitem__ involves mapping sequences of integers to sequences
  of datapoints. When this class is operating in with replacement then this
  mapping is static. When operating without replacement subsequent attempts
  to retrieve datapoints with the same index will generate new datapoints from
  the remaining data which has yet to be used to generate datapoints.
  """

  def __init__(
      self,
      titles,
      dates,
      classroom_lookup,
      seq_len,
      separator_token,
      sampler,
      fields = (
          types.ModelInputFields.TITLES,
          types.ModelInputFields.STUDENT_IDS,
          types.ModelInputFields.TIMESTAMPS,
          types.ModelInputFields.INPUT_POSITIONS,
      ),
      start_with_separator_token = False,
      student_ids = None,
      grade_levels = None,
  ):
    """Initializes the class.

    Args:
      titles: A jagged np.ndarray of titles. The i'th entry is an np.ndarray of
        titles read by the student whose StudentIndexRange contains the index i.
      dates: A jagged np.ndarray of timestamps specifying the date of the
        interactions described in titles.
      classroom_lookup: A dictionary that maps each ClassRoomID to the
        StudentIndexRanges for all students in that classroom.
      seq_len: The length of datapoints to be generated. Datapoints shorter than
        seq_len will be zero padded.
      separator_token: The TokenIndex of the separator token.
      sampler: A sampler object to sample classrooms and title interaction
        sequences
      fields: The list of fields to be produced for each sample. Valid values
        are be elements of types.ModelInputFields.
      start_with_separator_token: If true, all datapoints will start with a
        separator token at the beginning.
      student_ids: An array of StudentIDs where the i'th entry is the student
        who the data in the i'th row of titles belongs to.
      grade_levels: An optional sequence where the i'th entry in the grade level
        of the student who the data in the i'th row of titles belongs to.
    """

    super().__init__()
    # Assert that we have been given the data for the field we have been
    # requested to generate.
    if types.ModelInputFields.STUDENT_IDS in fields:
      assert student_ids is not None
    if types.ModelInputFields.GRADE_LEVELS in fields:
      assert grade_levels is not None

    self.titles = titles
    self.dates = dates
    self.classroom_lookup = classroom_lookup

    self.seq_len = seq_len
    self.seperator_token = np.array([separator_token], dtype=np.int32)
    self.zero = np.array([0], dtype=np.int32)
    self.student_ids = student_ids
    self.grade_levels = grade_levels

    self.fields = fields
    self._sampler = sampler

    self.start_with_seperator_token = start_with_separator_token

  @property
  def with_replacement(self):
    """Return True if sampling with replacement. False otherwise."""
    return isinstance(self._sampler, ClassroomSamplerWithReplacement)

  @classmethod
  def datasource_from_grouped_activity_dataframe(
      cls,
      student_activity,
      student_info,
      seq_len,
      *,
      max_grade_level = None,
      fields = (
          types.ModelInputFields.TITLES,
          types.ModelInputFields.STUDENT_IDS,
          types.ModelInputFields.TIMESTAMPS,
          types.ModelInputFields.INPUT_POSITIONS,
      ),
      student_chunk_len = None,
      vocab_size = None,
      vocab = None,
      classroom='school_year',
      school_info = None,
      with_replacement = True,
      with_automatic_reset = False,
      ordered_within_student = True,
  ):
    """Creates a ClassroomGroupedDataSource from a grouped activity dataframe.

    Args:
      student_activity: A dataframe with student-title interaction activity. It
        should be grouped by StudentID
      student_info: A dataframe with student information.
      seq_len: The maximum length of datapoints to be composed. Shorter
        datapoints will be padded to this length.
      max_grade_level: The maxmimum value for the field grade_lvl. Values higher
        than this are reassigned to 0 as are values < 0. If None then we will
        return a datasource which does not produce grade lvl fields.
      fields: The list of fields to be produced for each sample. Valid values
        are be elements of types.ModelInputFields.
      student_chunk_len: The length of sequence to break long student sequences
        to. This is done in preprocessing.
      vocab_size: The number of titles to assign unique vocabulary tokens. The
        rest are assigned to a single OOV token. Should not be provided if a
        prebuilt vocabulary is supplied
      vocab: a prebuilt vocabulary
      classroom: One of 'school_year', 'district_year' or 'none'. Controls how
        students are placed into classrooms.
      school_info: A dataframe containing details about schools.
      with_replacement: When true datapoints will be generated by sampling with
        replacement from the underlying data. When False, sequences used to
        previously generate a datapoint will not be used again.
      with_automatic_reset: When True, when sampling without replacement and all
        data is depleted, the datasource will automatically reset and continue
        sampling the data from the beginning. When False, attempting to sample
        without replacement from a depleted DataSource will raise a
        StopIteration exception.
      ordered_within_student: When True, we will randomly sample students per
        the described method, and then return the data with the earliest
        timestamps that have not been yet sampled for that student. When False
        we sample a random sequence belonging the student. This parameter only
        has effect when with_replacement=False.

    Returns:
      datasource: ClassroomGroupedDataSource from given data.
      vocab: The Vocabulary used to represent tokens.
    """
    assert (vocab_size is None) ^ (
        vocab is None
    ), 'Exactly one of vocab_size or vocab must be provided'

    if student_chunk_len is None:
      student_chunk_len = seq_len

    if vocab is None:
      vocab = utils.build_vocab(
          student_activity.obj['SHLF_NUM'],
          vocab_size,
          special_tokens=[types.Token('<SEP>')],
      )

    # Join given input dataframes together to gather necessary information
    # to assign students to classrooms.
    student_info = student_info.merge(
        student_activity.obj,
        on=types.StudentActivityFields.STUDENT_ID,
        how='right',
    ).drop_duplicates(subset=[types.StudentActivityFields.STUDENT_ID])
    if school_info is not None:
      student_info = student_info.merge(
          school_info, on=types.StudentActivityFields.SCHOOL_ID, how='left'
      ).drop_duplicates(subset=[types.StudentActivityFields.SCHOOL_ID])

    if classroom == 'school_year':
      classroom_columns = (
          types.StudentActivityFields.GRADE_LEVEL,
          types.StudentActivityFields.GRADE_LEVEL,
      )
      classroom_lookup = utils.build_classroom_lookup(
          student_info,
          classroom_columns=classroom_columns,
      )
    elif classroom == 'district_year':
      assert school_info is not None
      classroom_columns = (
          types.StudentActivityFields.GRADE_LEVEL,
          types.StudentActivityFields.DISTRICT_ID,
      )
      classroom_lookup = utils.build_classroom_lookup(
          student_info, classroom_columns=classroom_columns
      )
    elif classroom == 'none':
      classroom_lookup = collections.defaultdict(int)
    else:
      raise ValueError('Unknown classroom: %s' % classroom)

    titles, dates, student_ids, classroom_lookup = utils.preprocess_dataframe(
        student_activity.obj, vocab, student_chunk_len, classroom_lookup
    )

    if with_replacement:
      sampler = ClassroomSamplerWithReplacement(classroom_lookup, titles)
    else:
      sampler = ClassroomSamplerWithoutReplacement(
          classroom_lookup,
          titles,
          with_automatic_reset=with_automatic_reset,
          ordered_within_student=ordered_within_student,
      )

    # Build lookup table for grade levels
    if max_grade_level is not None:
      if types.ModelInputFields.GRADE_LEVELS not in fields:
        fields = fields + (types.ModelInputFields.GRADE_LEVELS,)
      grade_lvl_lookup = utils.build_classroom_lookup(
          student_info,
          classroom_columns=(
              types.StudentActivityFields.GRADE_LEVEL,
              types.StudentActivityFields.GRADE_LEVEL,
          ),
          offset=utils.UndefinedClassroom(),
      )
      def get_final_grade_level_from_classroom(classroom):
        """Extract grade level from classroom adjusting for UndefinedClassroom."""
        if isinstance(classroom[0], utils.UndefinedClassroom):
          return 0
        return classroom[0]

      grade_lvl_lookup = {
          student_id: get_final_grade_level_from_classroom(classroom)
          for student_id, classroom in grade_lvl_lookup.items()
      }

      grade_levels = [
          grade_lvl_lookup[student_id] for student_id in student_ids
      ]
      grade_levels = [
          grade_lvl if 0 < grade_lvl <= max_grade_level else 0
          for grade_lvl in grade_levels
      ]
      grade_levels = np.array(grade_levels, dtype=np.int32)
    else:
      grade_levels = []

    datasource = cls(
        titles,
        dates,
        classroom_lookup,
        seq_len,
        fields=fields,
        separator_token=vocab[types.Token('<SEP>')],
        sampler=sampler,
        start_with_separator_token=False,
        student_ids=student_ids,
        grade_levels=grade_levels,
    )

    return datasource, vocab

  def re_initialize_sampler(
      self,
      with_replacement,
      with_automatic_reset = False,
      ordered_within_student = True,
  ):
    """Reinitialize the index sampler.

    This allows the caller to change the datasource between with and without
    replacement sampling modes.
    Args:
      with_replacement: When true datapoints will be generated by sampling with
        replacement from the underlying data. When False, sequences used to
        previously generate a datapoint will not be used again.
      with_automatic_reset: When True, when sampling without replacement and all
        data is depleted, the datasource will automatically reset and continue
        sampling the data from the beginning. When False, attempting to sample
        without replacement from a depleted DataSource will raise a
        StopIteration exception.
      ordered_within_student: When True, we will randomly sample students per
        the described method, and then return the data with the earliest
        timestamps that have not been yet sampled for that student. When False
        we sample a random sequence belonging the student. This parameter only
        has effect when with_replacement=False.
    """

    if with_replacement:
      self._sampler = ClassroomSamplerWithReplacement(
          self.classroom_lookup, self.titles
      )

    else:
      self._sampler = ClassroomSamplerWithoutReplacement(
          self.classroom_lookup,
          self.titles,
          with_automatic_reset=with_automatic_reset,
          ordered_within_student=ordered_within_student,
      )

  def __len__(self):
    return 1_000_000_000_000  # this datasource is infinite

  def __getitem__(
      self, indices
  ):
    """Return a batch of samples with given indices .

    Args:
      indices: A list of indices of datapoints to be returned. When operating in
        without replacement indices do not map to static datapoints.

    Returns:
      A list with an entry for each index in indices. Each entry is a dictionary
      mapping str attribute name to the attribute value.
    """
    # We will use each index to seed a random number generator. This
    # generator will then be used to sample a datapoint.
    # Philox rngs still have acceptable random behaviour when seeded with
    # consecutive integers.
    rngs = [np.random.Generator(np.random.Philox(key=key)) for key in indices]
    batch = []
    for rng in rngs:
      classroom = self._sampler.sample_classroom(
          rng,
      )

      # Obtain the indices of the sequences to compose to generate this data
      # point.
      sequence_indices = self._sampler.sample_indices(
          classroom,
          rng,
          seq_len=self.seq_len,
          using_separators=True,
          start_with_separator=self.start_with_seperator_token,
      )

      sample = self._compose_sample(sequence_indices)
      batch.append(sample)

    return batch

  def _compose_batch_variable(
      self,
      sequence_indices,
      sequences,
      separator_token,
      transformation = None,
  ):
    """Compose the array for sample from the sequences at the given indices.

    Args:
      sequence_indices: A list of indices of the sequences to compose
      sequences: A list of all sequences to compose from.
      separator_token: The np.ndarray containing the separator token.
      transformation: A function f(sequences, index) that extracts the index'th
        entry from sequences and applies any necessary transformation. If None
        <lambda sequences,index: sequences[index]> is used.

    Returns:
      The composed np.ndarray.
    """

    if transformation is None:
      transformation = lambda x, i: x[i]

    entry = []
    # Add initial separator token if required
    if self.start_with_seperator_token:
      entry.append(separator_token)

    # Add the first subsequence
    first_subsequence = transformation(sequences, sequence_indices[0])
    entry.append(first_subsequence)

    # Add separator token then subsequent subsequence for remainder of indices.
    for sequence_index in sequence_indices[1:]:
      entry.append(separator_token)

      subsequence = transformation(sequences, sequence_index)
      entry.append(subsequence)

    # Compute needed padding length and add padding if necessary
    sequence_length = sum(x.size for x in entry)
    padding_needed = self.seq_len - sequence_length
    if padding_needed:
      padding = np.zeros(padding_needed, dtype=np.int32)
      entry.append(padding)
    # pylint: disable=unexpected-keyword-arg
    return np.concatenate(entry, dtype=np.int32)

  def _compose_sample(
      self, sequence_indices
  ):
    """Compose a sample from the given sequence indicies.

    Args:
      sequence_indices: The chosen sequence indices to compose to make the
        sample.

    Returns:
      A sample that is a dict[str, np.ndarray] mapping field names to their
      values.
    """
    sample = {}

    if types.ModelInputFields.TITLES in self.fields:
      sample[types.ModelInputFields.TITLES] = self._compose_batch_variable(
          sequence_indices, self.titles, self.seperator_token, None
      )
    if types.ModelInputFields.TIMESTAMPS in self.fields:
      sample[types.ModelInputFields.TIMESTAMPS] = self._compose_batch_variable(
          sequence_indices, self.dates, self.seperator_token, None
      )

    def make_student_ids(titles_and_student_ids, index):
      return (
          np.zeros_like(titles_and_student_ids[0][index])
          + titles_and_student_ids[1][index]
      )

    # STUDENT_IDS are arrays that match the length of title arrays and have
    # a constant value equal to the StudentID so both titles and student_id
    # arrays are needed to generate it.
    titles_and_student_ids = (self.titles, self.student_ids)
    if types.ModelInputFields.STUDENT_IDS in self.fields:
      sample[types.ModelInputFields.STUDENT_IDS] = self._compose_batch_variable(
          sequence_indices,
          titles_and_student_ids,
          self.seperator_token,
          make_student_ids,
      )

    # As all students in a separator token joint sampler are in the same
    # grade level we will return a 1-D array with one entry which is the grade
    # level for the entire batch.
    if types.ModelInputFields.GRADE_LEVELS in self.fields:
      sample[types.ModelInputFields.GRADE_LEVELS] = self.grade_levels[
          sequence_indices[0]
      ]

    def make_positions(titles, index):
      return np.arange(titles[index].size, dtype=np.int32)

    if types.ModelInputFields.INPUT_POSITIONS in self.fields:
      sample[types.ModelInputFields.INPUT_POSITIONS] = (
          self._compose_batch_variable(
              sequence_indices, self.titles, self.zero, make_positions
          )
      )

    return sample

  def is_ordered_within_student(self):
    """Returns True if the datasource yields sublists within students in order."""
    return self._sampler.is_ordered_within_student()


def collate(
    batch
):
  """Collate a batch from a sequence of samples.

  Args:
    batch: A sequence of dicts, one dict per sample in the batch

  Returns:
    A single dict[str, np.ndarray] where each array consists of all entries in
      the batch concatenated along a new axis for batch.
  """
  final_result = {attribute: [] for attribute in batch[0]}
  for sample in batch:
    for attribute, value in sample.items():
      final_result[attribute].append(value)

  final_result = {
      attribute: np.stack(value) for attribute, value in final_result.items()
  }
  return final_result


def exhaustive_batch_sampler(
    datasource, batch_size
):
  """A generator to exhaustively sample datapoints until datasource is consumed.

  Args:
    datasource: A datasource that samples without replacement
    batch_size: The number of samples per batch. Final batch will have less
      samples.

  Yields:
    A batch of datapoints from the datasource.
  """
  assert not datasource.with_replacement

  while 1:
    batch = []

    for _ in range(batch_size):
      try:
        batch.extend(datasource[[0]])
      except StopIteration:
        if batch:
          yield collate(batch)
        return

    yield collate(batch)


def iterator_with_replacement(
    datasource,
    batch_size,
    num_shards = 1,
    shard_index = 0,
    start_index = 0,
    num_batches = None,
):
  """An iterator for samplers with replacements.

  When non-default sharding parameters are passed, the iterator will return
  a batch composed of every n'th sample of the datasource with an offset of
  i, where n=num_shards and i=shard_index.
  Args:
    datasource: A datasource to fetch samples from.
    batch_size: the number of samples in each batch yielded from the iterator.
      when sharding is used, this is the number of samples in each sharded
      batch.
    num_shards: The number of parallel shards to produce.
    shard_index: The index of the current shard.
    start_index: The index from which to start sampling from.
    num_batches: The number of batches to produce if starting from batch 0. Will
      sample indefinitely if set to None.

  Yields:
    A batch of samples.
  """

  total_batch_size = batch_size * num_shards
  batch_start = start_index

  while num_batches is None or (batch_start < total_batch_size * num_batches):
    offset_batch_start = batch_start + shard_index

    sample_indices = range(
        offset_batch_start, offset_batch_start + total_batch_size, num_shards
    )
    yield collate(datasource[sample_indices])

    batch_start += total_batch_size
