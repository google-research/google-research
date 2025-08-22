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

"""Tests for input_pipeline."""

import functools
from typing import Sequence

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow_datasets as tfds

from sparse_mixers import input_pipeline
import sentencepiece as spm


class MockTokenizer(spm.SentencePieceProcessor):
  """Mock tokenizer returning pre-specified tokens."""

  def EncodeAsIds(self, text):
    del text  # Ignore input and return dummy output
    return np.array([6, 7, 8])

  def pad_id(self):
    return 1

  def eos_id(self):
    return 2

  def bos_id(self):
    return 3

  def PieceToId(self, text):
    del text  # Ignore input and return dummy output
    return np.random.randint(5, 20)

  def GetPieceSize(self):
    return 20


class InputPipelineTest(parameterized.TestCase):

  def test_clean_multirc_inputs(self):
    self.assertEqual(
        input_pipeline._clean_multirc_inputs(
            dataset_name="super_glue/multirc", text=b"<br>html</b>"), " html ")
    self.assertEqual(
        input_pipeline._clean_multirc_inputs(
            dataset_name="super_glue/multirc", text=b"clean"), "clean")
    self.assertEqual(
        input_pipeline._clean_multirc_inputs(
            dataset_name="not_multirc", text="<br>html</b>"), "<br>html</b>")

  @parameterized.parameters(
      "glue/cola", "glue/sst2", "glue/mrpc", "glue/qqp", "glue/stsb",
      "glue/mnli", "glue/qnli", "glue/rte", "glue/wnli", "super_glue/boolq",
      "super_glue/cb", "super_glue/copa", "super_glue/multirc",
      "super_glue/record", "super_glue/rte", "super_glue/wic", "super_glue/wsc",
      "super_glue/wsc.fixed", "super_glue/axb", "super_glue/axg")
  def test_classification_inputs(self, dataset_name):
    batch_size = 2
    max_seq_length = 4
    data_pipeline = functools.partial(
        input_pipeline.classification_inputs,
        split=tfds.Split.TRAIN,
        batch_size=batch_size,
        tokenizer=MockTokenizer(),
        max_seq_length=max_seq_length)

    with tfds.testing.mock_data(num_examples=10):
      for batch, _ in zip(data_pipeline(dataset_name=dataset_name), range(1)):
        self.assertSetEqual(
            set(batch.keys()), {"input_ids", "type_ids", "idx", "label"})
        self.assertTupleEqual(batch["input_ids"].shape,
                              (batch_size, max_seq_length))
        self.assertTupleEqual(batch["type_ids"].shape,
                              (batch_size, max_seq_length))
        self.assertTupleEqual(batch["idx"].shape, (batch_size,))
        self.assertEqual(batch["label"].shape[0], batch_size)

  def test_singularize_copa_examples(self):
    dataset = [{
        "idx":
            np.array([0]),
        "premise":
            np.array(["I packed up my belongings."], dtype="object"),
        "question":
            np.array(["cause"], dtype="object"),
        "choice1":
            np.array(["I was hunting for a new apartment."], dtype="object"),
        "choice2":
            np.array(["I was moving out of my apartment."], dtype="object"),
        "label":
            np.array([1]),
    }, {
        "idx":
            np.array([1]),
        "premise":
            np.array(["The putrid odor filled the room."], dtype="object"),
        "question":
            np.array(["effect"], dtype="object"),
        "choice1":
            np.array(["I clamped my hand over my nose."], dtype="object"),
        "choice2":
            np.array(["I put the rubber gloves on."], dtype="object"),
        "label":
            np.array([0]),
    }]
    expected_result = {
        "idx":
            np.array([[0], [0], [1], [1]]),
        "premise":
            np.array(
                [["I packed up my belongings."], ["I packed up my belongings."],
                 ["The putrid odor filled the room."],
                 ["The putrid odor filled the room."]],
                dtype="object"),
        "question":
            np.array([["cause"], ["cause"], ["effect"], ["effect"]],
                     dtype="object"),
        "choice":
            np.array([["I was hunting for a new apartment."],
                      ["I was moving out of my apartment."],
                      ["I clamped my hand over my nose."],
                      ["I put the rubber gloves on."]],
                     dtype="object"),
        "label":
            np.array([[False], [True], [True], [False]])
    }
    np.testing.assert_equal(
        input_pipeline._singularize_copa_examples(dataset), expected_result)

  def test_singularize_record_examples(self):
    dataset = [{
        "idx": {
            "query": np.array([0])
        },
        "passage": np.array(["Easy as 123."], dtype="object"),
        "query": np.array(["Let's count from @placeholder."], dtype="object"),
        "entities": np.array(["123", "the start"], dtype="object"),
        "answers": np.array(["123"], dtype="object"),
        "label": np.array([1]),
    }, {
        "idx": {
            "query": np.array([1])
        },
        "passage": np.array(["ABC."], dtype="object"),
        "query": np.array(["First letters of @placeholder."], dtype="object"),
        "entities": np.array(["google", "alphabet"], dtype="object"),
        "answers": np.array(["alphabet"], dtype="object"),
        "label": np.array([0]),
    }]
    expected_result = {
        "idx":
            np.array([[0], [0], [1], [1]]),
        "passage":
            np.array([
                ["Easy as 123."],
                ["Easy as 123."],
                ["ABC."],
                ["ABC."],
            ],
                     dtype="object"),
        "query":
            np.array([["Let's count from @placeholder."],
                      ["Let's count from @placeholder."],
                      ["First letters of @placeholder."],
                      ["First letters of @placeholder."]],
                     dtype="object"),
        "entity":
            np.array(["123", "the start", "google", "alphabet"],
                     dtype="object"),
        "label":
            np.array([True, False, False, True])
    }
    np.testing.assert_equal(
        input_pipeline._singularize_record_examples(dataset), expected_result)


if __name__ == "__main__":
  absltest.main()
