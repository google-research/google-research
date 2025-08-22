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

from absl.testing import absltest
from absl.testing import parameterized
from cisc.src.datasets import mmlu_pro


class MmluProTest(parameterized.TestCase):

  @parameterized.parameters(
      ("The proposed answer is: (A).", "A"),
      ("The proposed answer is: B.", "B"),
      ("The proposed answer is:    C.", "C"),
      ("The proposed answer is:    (d.", "D"),
      ("The proposed answer is:    e something", "E"),
      ("The answer is:    h", "H"),
      ("answer is:    J", "J"),
      ("The proposed answer is: X.", None),
      ("The proposed answer is: CC", None),
      ("The proposed answer is: NONE.", None),
      ("The proposed answer is: ELSE.", None),
  )
  def test_answer_extraction(self, text, expected):
    answer, _ = mmlu_pro.get_final_answer(text)
    self.assertEqual(expected, answer)


if __name__ == "__main__":
  absltest.main()
