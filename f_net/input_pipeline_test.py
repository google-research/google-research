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

"""Tests for f_net.input_pipeline."""

import functools
from typing import Sequence

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow_datasets as tfds
import tensorflow_text as tf_text

from f_net import input_pipeline


class MockTokenizer(tf_text.TokenizerWithOffsets):
  """Mock tokenizer returning pre-specified tokens."""

  def EncodeAsIds(self, text):
    del text  # Ignore input and return dummy output
    return np.random.randint(5, 20, size=3)

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

  @parameterized.parameters("glue/cola", "glue/sst2", "glue/mrpc", "glue/qqp",
                            "glue/stsb", "glue/mnli", "glue/qnli", "glue/rte",
                            "glue/wnli")
  def test_glue_inputs(self, dataset_name):
    batch_size = 2
    max_seq_length = 4
    glue_pipeline = functools.partial(
        input_pipeline.glue_inputs,
        split=tfds.Split.TRAIN,
        batch_size=batch_size,
        tokenizer=MockTokenizer(),
        max_seq_length=max_seq_length)

    with tfds.testing.mock_data(num_examples=10):
      for batch, _ in zip(glue_pipeline(dataset_name=dataset_name), range(1)):
        self.assertSetEqual(
            set(batch.keys()), {"input_ids", "type_ids", "idx", "label"})
        self.assertTupleEqual(batch["input_ids"].shape,
                              (batch_size, max_seq_length))
        self.assertTupleEqual(batch["type_ids"].shape,
                              (batch_size, max_seq_length))
        self.assertTupleEqual(batch["idx"].shape, (batch_size,))
        self.assertTupleEqual(batch["label"].shape, (batch_size,))

  # TODO(b/181607810): Modify C4 pipeline to load smaller batches of text
  #  so that we can test it.


if __name__ == "__main__":
  absltest.main()
