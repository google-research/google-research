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

"""Tests for wt5.metrics."""
from absl.testing import absltest
import numpy as np
import tensorflow.compat.v1 as tf

from wt5.wt5 import postprocessors

tf.disable_v2_behavior()
tf.enable_eager_execution()


class PostprocessorsTest(absltest.TestCase):

  def test_cos_e(self):
    self.assertEqual(
        postprocessors.abstractive_explanations(
            output="cheeseburger explanation: she ordered a cheeseburger.",
            example=None,
            is_target=True), {
                "label": "cheeseburger",
                "explanations": ["she ordered a cheeseburger."]
            })

  def test_cos_e_explanation_contains_because(self):
    self.assertEqual(
        postprocessors.abstractive_explanations(
            output="cheeseburger explanation: because she was hungry.",
            example=None,
            is_target=True), {
                "label": "cheeseburger",
                "explanations": ["because she was hungry."]
            })

  def test_cos_e_multi_word_answer(self):
    self.assertEqual(
        postprocessors.abstractive_explanations(
            output="goodnight moon explanation: it's a popular kids book.",
            example=None,
            is_target=True), {
                "label": "goodnight moon",
                "explanations": ["it's a popular kids book."]
            })

  def test_esnli(self):
    self.assertEqual(
        postprocessors.abstractive_explanations(
            "entailment explanation: this is correct.",
            example=None,
            is_target=False),
        {
            "label": "entailment",
            "explanations": ["this is correct."]
        })

    self.assertEqual(
        postprocessors.abstractive_explanations(
            "entailment explanation: this is correct.",
            example={
                "inputs_pretokenized": b"This is incorrect",
                "targets_pretokenized": b"Incorrect answer."
            },
            is_target=True),
        {
            "label": "entailment",
            "explanations": ["this is correct."]
        })

  def test_esnli_multiple_explanations(self):
    self.assertEqual(
        postprocessors.abstractive_explanations(
            "entailment explanation: this is correct. "
            "explanation: correct. explanation: also correct.",
            example=None,
            is_target=False),
        {
            "label": "entailment",
            "explanations": ["this is correct.", "correct.", "also correct."]
        })

  def test_movies(self):
    self.assertEqual(
        postprocessors.abstractive_explanations(
            "negative explanation: the movie was boring explanation: acting "
            "was bad explanation: poor script explanation: skip it.",
            separator=" explanation: "),
        {
            "label": "negative",
            "explanations": [
                "the movie was boring", "acting was bad", "poor script",
                "skip it."
            ]
        })

  def test_extractive_explanations_em(self):
    inputs = (
        b"explain review: the movie was boring, did not have a good time. "
        b"acting was bad and a poor script"
    )
    answer = (
        "negative explanation: the movie was boring explanation: acting was "
        "bad explanation: poor script"
    )
    expected_span_array = np.zeros(len(inputs), np.int)
    explanations = [b"the movie was boring", b"acting was bad", b"poor script"]
    for exp in explanations:
      expected_span_array[inputs.find(exp):inputs.find(exp) + len(exp)] = 1

    self.assertEqual(
        postprocessors.extractive_explanations(
            answer,
            separator=" explanation: ",
            example={"inputs_pretokenized": inputs},
            tokenizer_fn=list,
        ),
        {
            "label": "negative",
            "overlap_spans": [(16, 36), (64, 78), (85, 96)],
            "span_array": list(expected_span_array),
            "explanations": [tf.compat.as_text(e) for e in explanations],
        })

    inputs = (
        "explain review: the movie was boring, did not have a good time. "
        "acting was bad"
    )
    expected_span_array = np.zeros(len(inputs), np.int)
    explanations = ["the movie was boring", "acting was bad"]
    for exp in explanations:
      expected_span_array[inputs.find(exp):inputs.find(exp) + len(exp)] = 1
    self.assertEqual(
        postprocessors.extractive_explanations(
            answer,
            separator=" explanation: ",
            example={"inputs_pretokenized": inputs},
            tokenizer_fn=list,
        ),
        {
            "label": "negative",
            "overlap_spans": [(16, 36), (64, 78)],
            "span_array": list(expected_span_array),
            "explanations": explanations,
        })

  def test_extractive_explanations_duplicates(self):
    inputs = (
        b"explain review: the movie was boring, did not have a good time. "
        b"acting was bad and a poor script"
    )
    answer = ("negative explanation: the movie was boring explanation: "
              "poor script explanation: poor script")
    expected_span_array = np.zeros(len(inputs), np.int16)
    explanations = [b"the movie was boring", b"poor script"]
    for exp in explanations:
      expected_span_array[inputs.find(exp):inputs.find(exp) + len(exp)] = 1

    self.assertEqual(
        postprocessors.extractive_explanations(
            answer,
            separator=" explanation: ",
            example={"inputs_pretokenized": inputs},
            tokenizer_fn=list,
        ),
        {
            "label": "negative",
            "overlap_spans": [(16, 36), (85, 96)],
            "span_array": list(expected_span_array),
            "explanations": [tf.compat.as_text(e) for e in explanations],
        })


if __name__ == "__main__":
  absltest.main()
