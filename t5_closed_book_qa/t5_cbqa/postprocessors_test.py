# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Tests for T5 CBQA postprocessors."""

from absl.testing import absltest
import numpy as np
import tensorflow.compat.v1 as tf

from t5_closed_book_qa.t5_cbqa import postprocessors

tf.disable_v2_behavior()
tf.enable_eager_execution()


class PostprocessorsTest(absltest.TestCase):

  def test_natural_questions(self):
    output = "answer: yes answer: This is correct"
    self.assertListEqual(
        postprocessors.natural_questions(output, is_target=False),
        [("yes", "This is correct")]
    )

    short_answers = tf.RaggedTensor.from_row_lengths(
        [b"a", b"B", b"c"], row_lengths=[0, 2, 1, 0, 0])
    self.assertListEqual(
        postprocessors.natural_questions(
            output,  # will be ignored
            example={
                "short_answers/values": short_answers.values,
                "short_answers/row_starts": short_answers.row_starts(),
                "yes_no_answers": np.array([-1, -1, -1, 0, 1])
            },
            is_target=True),
        [(), ("a", "B"), ("c",), ("no",), ("yes",)]
    )


if __name__ == "__main__":
  absltest.main()
