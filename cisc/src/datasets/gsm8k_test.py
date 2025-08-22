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
from cisc.src.datasets import gsm8k


class GSM8KTest(parameterized.TestCase):

  @parameterized.parameters(
      ("The proposed answer is: (11).", "11"),
      ("The proposed answer is: (-1).", "-1"),
      ("The proposed answer is: -1", "-1"),
      ("The proposed answer is: (- 1).", "-1"),
      ("The answer is: 1337.", "1337"),
      ("The answer is: 1,337.", "1337"),
      # Only take the int part.
      ("The answer is: -13.37.", "-13"),
      ("The proposed answer is: (A).", None),
  )
  def test_answer_extraction(self, text, expected):
    answer, _ = gsm8k.get_final_answer(text)
    self.assertEqual(expected, answer)


if __name__ == "__main__":
  absltest.main()
