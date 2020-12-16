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

"""Tests for WT5 preprocessors."""
import functools

from absl.testing import absltest
import t5.data
import tensorflow.compat.v1 as tf

from wt5.wt5 import preprocessors


class PreprocessorsTest(absltest.TestCase):

  def test_cos_e(self):
    input_data = {
        'question': 'Question?',
        'choices': ['First', 'Second', 'Third'],
        'abstractive_explanation': 'Abstractive explanation.',
        'extractive_explanation': 'Not currently used.',
        'answer': 'First',
    }
    og_dataset = tf.data.Dataset.from_tensors(input_data)
    dataset = preprocessors.cos_e(og_dataset)
    t5.data.assert_dataset(
        dataset,
        {
            'inputs': 'explain cos_e question: Question? choice: First choice: '
                      'Second choice: Third',
            'targets': 'First explanation: Abstractive explanation.'
        }
    )

  def test_cos_e_zero_shot_like_esnli(self):
    input_data = {
        'question': 'Question?',
        'choices': ['First', 'Second', 'Third'],
        'abstractive_explanation': 'Abstractive explanation.',
        'extractive_explanation': 'Not currently used.',
        'answer': 'First',
    }
    og_dataset = tf.data.Dataset.from_tensors(input_data)
    dataset = preprocessors.cos_e(og_dataset, prefix='explain nli',
                                  question_prefix='premise:')
    t5.data.assert_dataset(
        dataset,
        {
            'inputs': 'explain nli premise: Question? choice: First choice: '
                      'Second choice: Third',
            'targets': 'First explanation: Abstractive explanation.'
        }
    )

  def test_cos_e_zero_shot_like_esnli_functools(self):
    input_data = {
        'question': 'Question?',
        'choices': ['First', 'Second', 'Third'],
        'abstractive_explanation': 'Abstractive explanation.',
        'extractive_explanation': 'Not currently used.',
        'answer': 'First',
    }
    og_dataset = tf.data.Dataset.from_tensors(input_data)
    dataset = functools.partial(preprocessors.cos_e, prefix='explain nli',
                                question_prefix='premise:')(og_dataset)
    t5.data.assert_dataset(
        dataset,
        {
            'inputs': 'explain nli premise: Question? choice: First choice: '
                      'Second choice: Third',
            'targets': 'First explanation: Abstractive explanation.'
        }
    )

  def test_cos_e_zero_shot_like_esnli_functools_hypothesis(self):
    input_data = {
        'question': 'Question?',
        'choices': ['First', 'Second', 'Third'],
        'abstractive_explanation': 'Abstractive explanation.',
        'extractive_explanation': 'Not currently used.',
        'answer': 'First',
    }
    og_dataset = tf.data.Dataset.from_tensors(input_data)
    dataset = functools.partial(
        preprocessors.cos_e,
        prefix='explain nli',
        question_prefix='premise:',
        choice_prefix='hypothesis:')(
            og_dataset)
    t5.data.assert_dataset(
        dataset, {
            'inputs':
                'explain nli premise: Question? hypothesis: First hypothesis: '
                'Second hypothesis: Third',
            'targets': 'First explanation: Abstractive explanation.'
        })

  def test_esnli(self):
    input_data = {
        'premise': 'It is hot.',
        'hypothesis': 'It is sunny.',
        'label': 0,
        'explanation_1': 'hot implies that it is sunny.'
    }

    og_dataset = tf.data.Dataset.from_tensors(input_data)
    dataset = preprocessors.esnli(og_dataset)

    t5.data.assert_dataset(
        dataset,
        {
            'inputs':
                'explain nli hypothesis: It is sunny. premise: It is hot.',
            'targets':
                'entailment explanation: hot implies that it is sunny.'
        }
    )

  def test_esnli_with_choices_like_cos_e(self):
    input_data = {
        'premise': 'It is hot.',
        'hypothesis': 'It is sunny.',
        'label': 0,
        'explanation_1': 'hot implies that it is sunny.'
    }

    og_dataset = tf.data.Dataset.from_tensors(input_data)
    dataset = dataset = functools.partial(
        preprocessors.esnli, add_choices=True)(
            og_dataset)

    t5.data.assert_dataset(
        dataset, {
            'inputs':
                ('explain nli hypothesis: It is sunny. premise: It is hot. '
                 'choice: entailment choice: neutral choice: contradiction'),
            'targets':
                'entailment explanation: hot implies that it is sunny.'
        })

  def test_esnli_multiple_explanations(self):
    input_data = {
        'premise': 'It is hot.',
        'hypothesis': 'It is sunny.',
        'label': 0,
        'explanation_1': 'hot implies that it is sunny.',
        'explanation_2': 'sunny equals hot.',
        'explanation_3': 'hot means sunny.',
    }

    og_dataset = tf.data.Dataset.from_tensors(input_data)
    dataset = preprocessors.esnli(og_dataset)

    t5.data.assert_dataset(
        dataset,
        {
            'inputs':
                'explain nli hypothesis: It is sunny. premise: It is hot.',
            'targets':
                'entailment explanation: hot implies that it is sunny. '
                'explanation: sunny equals hot. '
                'explanation: hot means sunny.'
        }
    )

  def test_esnli_drop_explanations(self):
    input_data = {
        'premise': 'It is hot.',
        'hypothesis': 'It is sunny.',
        'label': 0,
        'explanation_1': 'hot implies that it is sunny.',
    }

    og_dataset = tf.data.Dataset.from_tensors(input_data)
    dataset = preprocessors.esnli(
        og_dataset, prefix='nli', drop_explanations=True)

    t5.data.assert_dataset(
        dataset,
        {
            'inputs': 'nli hypothesis: It is sunny. premise: It is hot.',
            'targets': 'entailment'
        }
    )

  def test_rationales_preprocessor(self):
    input_data = {
        'review': 'This was a terrible movie. Complete waste of time.',
        'label': 0,
        'evidences': ['terrible movie', 'waste of time']
    }

    og_dataset = tf.data.Dataset.from_tensors(input_data)

    dataset = preprocessors.extractive_explanations(og_dataset)
    t5.data.assert_dataset(
        dataset,
        {
            'inputs': 'explain sentiment review: This was a terrible movie. '
                      'Complete waste of time.',
            'targets': 'negative '
                       'explanation: terrible movie explanation: waste of time'
        }
    )

  def test_rationales_preprocessor_no_explanations(self):
    input_data = {
        'review': 'This was a terrible movie. Complete waste of time.',
        'label': 0,
        'evidences': ['terrible movie', 'waste of time']
    }

    og_dataset = tf.data.Dataset.from_tensors(input_data)

    dataset = preprocessors.extractive_explanations(
        og_dataset, drop_explanations=True)
    t5.data.assert_dataset(
        dataset,
        {
            'inputs': 'explain sentiment review: This was a terrible movie. '
                      'Complete waste of time.',
            'targets': 'negative'
        }
    )

  def test_amazon_reviews(self):
    input_data = {
        'data': {
            'review_headline': 'Great headphones',
            'review_body': 'Loved the sound quality of these headphones',
            'star_rating': 5,
        }
    }

    og_dataset = tf.data.Dataset.from_tensors(input_data)

    dataset = preprocessors.amazon_reviews(og_dataset)

    t5.data.assert_dataset(
        dataset,
        {
            'inputs': 'sentiment review: Great headphones Loved the '
                      'sound quality of these headphones',
            'targets': 'positive'
        })

    dataset = preprocessors.amazon_reviews(og_dataset, binary_output=False)

    t5.data.assert_dataset(
        dataset,
        {
            'inputs': 'sentiment review: Great headphones Loved the '
                      'sound quality of these headphones',
            'targets': '5'
        })

  def test_amazon_reviews_neutral(self):
    input_data = {
        'data': {
            'review_headline': 'okay headphones',
            'review_body': 'the sound quality of these headphones is not bad',
            'star_rating': 3,
        }
    }

    og_dataset = tf.data.Dataset.from_tensors(input_data)

    dataset = preprocessors.amazon_reviews(og_dataset)

    t5.data.assert_dataset(dataset, [])

    dataset = preprocessors.amazon_reviews(og_dataset, binary_output=False)

    t5.data.assert_dataset(
        dataset,
        {
            'inputs': 'sentiment review: okay headphones the sound quality of '
                      'these headphones is not bad',
            'targets': '3'
        })

  def test_eraser_multi_rc(self):
    input_data = {
        'passage':
            'This is a multi line passage. \nIt is about multiple things. '
            '\nThere is more than one thing in it.',
        'query_and_answer':
            'Is it about one thing? || Nope.',
        'label':
            1,
        'evidences': [
            'It is about multiple things.',
            'There is more than one thing in it.'
        ]
    }

    og_dataset = tf.data.Dataset.from_tensors(input_data)

    dataset = preprocessors.eraser_multi_rc(og_dataset)
    t5.data.assert_dataset(
        dataset, {
            'inputs':
                'explain multirc passage: This is a multi line passage. \nIt is '
                'about multiple things. \nThere is more than one thing in it. '
                'query: Is it about one thing? answer: Nope.',
            'targets':
                'True explanation: It is about multiple things. explanation: '
                'There is more than one thing in it.'
        })

  def test_eraser_multi_rc_drop_examples(self):
    input_data = {
        'passage':
            'This is a multi line passage. \nIt is about multiple things. '
            '\nThere is more than one thing in it.',
        'query_and_answer':
            'Is it about one thing? || Nope.',
        'label':
            1,
        'evidences': [
            'It is about multiple things.',
            'There is more than one thing in it.'
        ]
    }

    og_dataset = tf.data.Dataset.from_tensors(input_data)

    dataset = preprocessors.eraser_multi_rc(og_dataset, drop_explanations=True)
    t5.data.assert_dataset(
        dataset, {
            'inputs':
                'explain multirc passage: This is a multi line passage. \nIt is '
                'about multiple things. \nThere is more than one thing in it. '
                'query: Is it about one thing? answer: Nope.',
            'targets':
                'True'
        })

  def test_imdb_movie_reviews(self):
    input_data = {
        'text': ['great movie', 'terrible movie'],
        'label': [1, -1],
    }

    og_dataset = tf.data.Dataset.from_tensor_slices(input_data)

    dataset = preprocessors.imdb_reviews(og_dataset)

    t5.data.assert_dataset(
        dataset,
        [
            {'inputs': 'sentiment: great movie', 'targets': 'positive'},
            {'inputs': 'sentiment: terrible movie', 'targets': '<unk>'}

        ]
    )


if __name__ == '__main__':
  tf.disable_v2_behavior()
  tf.enable_eager_execution()
  absltest.main()
