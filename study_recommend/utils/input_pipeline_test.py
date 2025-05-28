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

import io

import unittest

import pandas as pd
from study_recommend import types
from study_recommend.utils import input_pipeline_utils


VOCAB_CONTENTS = {'A': 1, 'B': 2, 'C': 3}
OOV_VALUE = 4
FIELDS = types.StudentActivityFields


def build_test_vocab():
  vocab_contents = {
      types.Token(token): types.TokenIndex(value)
      for token, value in VOCAB_CONTENTS.items()
  }

  return input_pipeline_utils.Vocabulary(vocab_contents, OOV_VALUE)


class VocabularyTest(unittest.TestCase):

  def test_getitem(self):
    vocab = build_test_vocab()
    for token, value in VOCAB_CONTENTS.items():
      self.assertEqual(vocab[token], value)

  def test_encode(self):
    vocab = build_test_vocab()
    for token, value in VOCAB_CONTENTS.items():
      self.assertEqual(vocab.encode(token), value)

  def test_decode(self):
    vocab = build_test_vocab()
    for token, value in VOCAB_CONTENTS.items():
      self.assertEqual(vocab.decode(value), token)

  def test_endecode_oov(self):
    vocab = build_test_vocab()
    self.assertEqual(vocab.encode('Z'), OOV_VALUE)

  def test_decode_unknown(self):
    vocab = build_test_vocab()
    self.assertEqual(vocab.decode(15), input_pipeline_utils.UNDEF)

  def test_serialize_deserialize_buffer(self):
    vocab = build_test_vocab()
    buffer = io.StringIO()
    vocab.serialize(buffer)
    buffer.seek(0)

    new_vocab = input_pipeline_utils.Vocabulary().deserialize(buffer)

    # Assert tokens are encoded the same before and after serialization.
    for token in VOCAB_CONTENTS:
      self.assertEqual(new_vocab[token], vocab[token])
    # Assert out of vocabulary tokens are encoded the same.
    self.assertEqual(new_vocab.encode('Z'), vocab.encode('Z'))
    # Test token indices are decoded the same. This loop will test in vocab
    # and out of vocab token indices.
    for i in range(5):
      self.assertEqual(new_vocab.decode(i), vocab.decode(i))


STUDENT_INFO_DATA = [
    [1, 1, 2, 3],
    [2, 1, 2, 5],
    [3, 1, 6, 2],
    [4, 1, 6, 5],
    [5, 2, 7, 5],
    [6, 3, 7, 5],
]
STUDENT_INFO_COLUMNS = [
    FIELDS.STUDENT_ID.value,
    'attribute1',
    'attribute2',
    'attribute3',
]
CLASSROOMS = {1: (1, 2), 2: (1, 2), 3: (1, 6), 4: (1, 6), 5: (2, 7), 6: (3, 7)}


class ClassroomLookupTest(unittest.TestCase):

  def test_build_classroom_lookup(self):
    student_activity = pd.DataFrame(
        STUDENT_INFO_DATA, columns=STUDENT_INFO_COLUMNS
    )
    classrooms = input_pipeline_utils.build_classroom_lookup(
        student_activity, classroom_columns=['attribute1', 'attribute2']
    )

    self.assertEqual(classrooms, CLASSROOMS)


def ragged_array_to_nest_list(array):
  if isinstance(array, list):
    return [ragged_array_to_nest_list(x) for x in array]
  else:
    return array.tolist()


class PreprocessDataframe(unittest.TestCase):

  def test_preprocess_dataframe(self):
    activity_data = [
        [1, '2023-01-18', 'A'],
        [1, '2023-01-19', 'B'],
        [1, '2023-01-22', 'A'],
        [1, '2023-01-23', 'A'],
        [3, '2023-01-10', 'C'],
        [3, '2023-01-23', 'C'],
        [3, '2023-01-30', 'A'],
        [2, '2023-04-10', 'C'],
        [2, '2023-04-10', 'B'],
    ]
    student_activity = pd.DataFrame(
        activity_data, columns=[FIELDS.STUDENT_ID, FIELDS.DATE, FIELDS.BOOK_ID]
    )
    vocabulary = input_pipeline_utils.Vocabulary(VOCAB_CONTENTS, OOV_VALUE)

    titles_array, dates_array, student_id_array, clasrooms_to_indices = (
        input_pipeline_utils.preprocess_dataframe(
            student_activity,
            vocabulary,
            seq_len=3,
            student_classrooms_lookup=CLASSROOMS,
        )
    )

    self.assertSequenceEqual(
        ragged_array_to_nest_list(titles_array),
        [[1, 2, 1], [1], [3, 2], [3, 3, 1]],
    )
    self.assertSequenceEqual(
        ragged_array_to_nest_list(dates_array),
        [
            [1674000000, 1674086400, 1674345600],
            [1674432000],
            [1681084800, 1681084800],
            [1673308800, 1674432000, 1675036800],
        ],
    )
    self.assertSequenceEqual(student_id_array, [1, 1, 2, 3])
    # sort students to prepare for exact equality test.
    clasrooms_to_indices = {
        classroom: sorted(students)
        for classroom, students in clasrooms_to_indices.items()
    }

    self.assertEqual(
        clasrooms_to_indices, {(1, 2): [(0, 2), (2, 1)], (1, 6): [(3, 1)]}
    )


if __name__ == '__main__':
  unittest.main()
