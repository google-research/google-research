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

# Lint as: python3
"""Tests for T5 CBQA preprocessors."""

from absl.testing import absltest
from t5.data import test_utils
import tensorflow.compat.v1 as tf

from t5_closed_book_qa.t5_cbqa import preprocessors

tf.disable_v2_behavior()
tf.enable_eager_execution()


class PreprocessorsTest(absltest.TestCase):

  def test_natural_questions_nocontext(self):
    input_examples = [
        {
            'question': {
                'text': 'is the answer to this question no',
            },
            'annotations': {
                'short_answers': {
                    'start_token': ([], [0, 0]),
                    'end_token': ([], [0, 0]),
                    'text': ([], [0, 0])
                },
                'yes_no_answer': [-1, -1]
            }
        },
        {
            'question': {
                'text': 'is the answer to this question yes',
            },
            'annotations': {
                'short_answers': {
                    'start_token': ([3, 3], [1, 0, 1, 0]),
                    'end_token': ([7, 5], [1, 0, 1, 0]),
                    'text': (['not sure sir', 'not sure'], [1, 0, 1, 0]),
                },
                'yes_no_answer': [-1, 0, -1, 1]
            }
        },
        {
            'question': {
                'text': 'what are the names of the olsen twins',
            },
            'annotations': {
                'short_answers': {
                    'start_token': ([0, 3], [2, 0]),
                    'end_token': ([3, 4], [2, 0]),
                    'text': (['Mary-Kate', 'Ashley'], [2, 0])
                },
                'yes_no_answer': [-1, -1]
            }
        }
    ]

    def _short_ans_to_ragged(ex):
      for field in ['start_token', 'end_token', 'text']:
        values, row_lengths = ex['annotations']['short_answers'][field]
        ex['annotations']['short_answers'][field] = (
            tf.RaggedTensor.from_row_lengths(values, row_lengths))
      return ex

    og_dataset = tf.data.Dataset.from_generator(
        lambda: (x for x in input_examples),
        output_types={
            'question': {'text': tf.string},
            'annotations': {
                'short_answers': {
                    'start_token': (tf.int64, tf.int64),
                    'end_token': (tf.int64, tf.int64),
                    'text': (tf.string, tf.int64)
                },
                'yes_no_answer': tf.int64
            }
        },
        output_shapes={
            'question': {'text': []},
            'annotations': {
                'short_answers': {
                    'start_token': ([None], [None]),
                    'end_token': ([None], [None]),
                    'text': ([None], [None]),
                },
                'yes_no_answer': [None]
            }
        }).map(_short_ans_to_ragged)

    dataset = preprocessors.natural_questions_nocontext(og_dataset)
    test_utils.assert_dataset(
        dataset,
        [
            {
                'inputs': 'nq question: is the answer to this question yes',
                'targets':
                    'answer: no answer: yes answer: not sure sir '
                    'answer: not sure',
                'short_answers/values': ['not sure sir', 'not sure'],
                'short_answers/row_starts': [0, 1, 1, 2],
                'yes_no_answers': [-1, 0, -1, 1],
            },
            {
                'inputs': 'nq question: what are the names of the olsen twins',
                'targets': 'answer: Mary-Kate answer: Ashley',
                'short_answers/values': ['Mary-Kate', 'Ashley'],
                'short_answers/row_starts': [0, 2],
                'yes_no_answers': [-1, -1],
            }
        ]
    )

    dataset = preprocessors.natural_questions_nocontext(
        og_dataset, drop_yes_no=True)
    test_utils.assert_dataset(
        dataset,
        [
            {
                'inputs': 'nq question: is the answer to this question yes',
                'targets': 'answer: not sure sir answer: not sure',
                'short_answers/values': ['not sure sir', 'not sure'],
                'short_answers/row_starts': [0, 1, 1, 2],
                'yes_no_answers': [-1, -1, -1, -1],
            },
            {
                'inputs': 'nq question: what are the names of the olsen twins',
                'targets': 'answer: Mary-Kate answer: Ashley',
                'short_answers/values': ['Mary-Kate', 'Ashley'],
                'short_answers/row_starts': [0, 2],
                'yes_no_answers': [-1, -1],
            }
        ]
    )

    dataset = preprocessors.natural_questions_nocontext(
        og_dataset, max_tokens=2)
    test_utils.assert_dataset(
        dataset,
        [
            {
                'inputs': 'nq question: is the answer to this question yes',
                'targets': 'answer: no answer: yes answer: not sure',
                'short_answers/values': ['not sure'],
                'short_answers/row_starts': [0, 0, 0, 1],
                'yes_no_answers': [-1, 0, -1, 1],
            },
            {
                'inputs': 'nq question: what are the names of the olsen twins',
                'targets': 'answer: Ashley',
                'short_answers/values': ['Ashley'],
                'short_answers/row_starts': [0, 1],
                'yes_no_answers': [-1, -1],
            }
        ]
    )

    dataset = preprocessors.natural_questions_nocontext(
        og_dataset, max_answers=1)
    test_utils.assert_dataset(
        dataset,
        [
            {
                'inputs': 'nq question: is the answer to this question yes',
                'targets': 'answer: no',
                'short_answers/values': ['not sure sir', 'not sure'],
                'short_answers/row_starts': [0, 1, 1, 2],
                'yes_no_answers': [-1, 0, -1, 1],
            },
            {
                'inputs': 'nq question: what are the names of the olsen twins',
                'targets': 'answer: Mary-Kate',
                'short_answers/values': ['Mary-Kate', 'Ashley'],
                'short_answers/row_starts': [0, 2],
                'yes_no_answers': [-1, -1],
            }
        ]
    )

    dataset = preprocessors.natural_questions_nocontext(
        og_dataset, drop_yes_no=True, max_tokens=2, max_answers=1)
    test_utils.assert_dataset(
        dataset,
        [
            {
                'inputs': 'nq question: is the answer to this question yes',
                'targets': 'answer: not sure',
                'short_answers/values': ['not sure'],
                'short_answers/row_starts': [0, 0, 0, 1],
                'yes_no_answers': [-1, -1, -1, -1],
            },
            {
                'inputs': 'nq question: what are the names of the olsen twins',
                'targets': 'answer: Ashley',
                'short_answers/values': ['Ashley'],
                'short_answers/row_starts': [0, 1],
                'yes_no_answers': [-1, -1],
            }
        ]
    )

    dataset = preprocessors.natural_questions_nocontext(
        og_dataset, drop_yes_no=True, max_tokens=1)
    test_utils.assert_dataset(
        dataset,
        [
            {
                'inputs': 'nq question: what are the names of the olsen twins',
                'targets': 'answer: Ashley',
                'short_answers/values': ['Ashley'],
                'short_answers/row_starts': [0, 1],
                'yes_no_answers': [-1, -1],
            }
        ]
    )

  def test_natural_questions_open(self):
    input_data = {
        'question': ['What are the names of the Olsen Twins?'],
        'answer': ['Mary-Kate', 'Ashley']
    }
    og_dataset = tf.data.Dataset.from_tensors(input_data)
    dataset = preprocessors.natural_questions_open(og_dataset)
    test_utils.assert_dataset(
        dataset,
        {
            'inputs': 'nq question: What are the names of the Olsen Twins?',
            'targets': 'Mary-Kate',
            'answers': ['Mary-Kate', 'Ashley'],
        }
    )

  def test_trivia_qa_open(self):
    input_data = {
        'question': ['What are the names of the Olsen Twins?'],
        'answer': {
            'value': 'Mary-Kate and Ashley',
            'aliases': ['Mary-Kate and Ashley', 'Ashley and Mary-Kate']
        }
    }

    og_dataset = tf.data.Dataset.from_tensors(input_data)

    dataset = preprocessors.trivia_qa_open(og_dataset)

    test_utils.assert_dataset(
        dataset,
        {
            'inputs':
                'trivia_qa question: What are the names of the Olsen Twins?',
            'targets': 'Mary-Kate and Ashley',
            'answers': ['Mary-Kate and Ashley', 'Ashley and Mary-Kate'],
        }
    )

  def test_web_questions_open(self):
    input_data = {
        'question': ['What are the names of the Olsen Twins?'],
        'answers': ['Mary-Kate and Ashley', 'Ashley and Mary-Kate']
    }

    og_dataset = tf.data.Dataset.from_tensors(input_data)

    dataset = preprocessors.web_questions_open(og_dataset)

    test_utils.assert_dataset(
        dataset,
        {
            'inputs': 'wq question: What are the names of the Olsen Twins?',
            'targets': 'Mary-Kate and Ashley',
            'answers': ['Mary-Kate and Ashley', 'Ashley and Mary-Kate'],
        }
    )

  def test_sample_answer(self):
    input_data = {
        'inputs': ['What are the names of the Olsen Twins?'],
        'targets': ['Mary-Kate'],
        'answers': ['Mary-Kate', 'Ashley']
    }
    og_dataset = tf.data.Dataset.from_tensors(input_data)

    tf.set_random_seed(42)
    test_utils.assert_dataset(
        preprocessors.sample_answer(og_dataset),
        {
            'inputs': 'What are the names of the Olsen Twins?',
            'targets': 'Ashley',
            'answers': ['Ashley', 'Mary-Kate'],
        }
    )
    tf.set_random_seed(420)
    test_utils.assert_dataset(
        preprocessors.sample_answer(og_dataset),
        {
            'inputs': ['What are the names of the Olsen Twins?'],
            'targets': ['Mary-Kate'],
            'answers': ['Mary-Kate', 'Ashley']
        }
    )

  def test_mask_salient_spans(self):
    input_examples = [
        {
            'text': 'He was confident that it would be well received.',
            'spans': {
                'start': [],
                'limit': [],
            }
        },
        {
            'text':
                'The episode was filmed over three days at the end of October '
                'and beginning of November 2002.',
            'spans': {
                'start': [53, 78],
                'limit': [60, 91],
            }
        }
    ]

    og_dataset = tf.data.Dataset.from_generator(
        lambda: (x for x in input_examples),
        output_types={
            'text': tf.string,
            'spans': {
                'start': tf.int64,
                'limit': tf.int64,
            },
        },
        output_shapes={
            'text': [],
            'spans': {
                'start': [None],
                'limit': [None],
            },
        })

    dataset = preprocessors.mask_salient_spans(og_dataset)

    test_utils.assert_dataset(
        dataset,
        [
            {
                'inputs':
                    'nem: The episode was filmed over three days at the end of '
                    '_X_ and beginning of November 2002.',
                'targets': 'October'
            },
            {
                'inputs':
                    'nem: The episode was filmed over three days at the end of '
                    'October and beginning of _X_.',
                'targets': 'November 2002'
            }
        ]
    )

if __name__ == '__main__':
  absltest.main()
