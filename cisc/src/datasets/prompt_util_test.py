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
from cisc.src.datasets import prompt_util


class UtilTest(parameterized.TestCase):

  @parameterized.parameters(
      ("The is: (Z).", None),
      ("is (Z).", None),
      ("is (X).", None),
      ("answer X", None),
      ("answer: (X)", "X"),
      ("The proposed answer is: (Z).", "Z"),
      ("The proposed answer      is : (Z).", "Z"),
      ("the proposed answer is: (z).", "Z"),
      ("the proposed answer is (z).", "Z"),
      ("proposed answer is (z).", "Z"),
      ("proposed answer is (50).", "50"),
      ("The proposed answer: (z).", "Z"),
      ("he proposed answer is: (G).", "G"),
      ("The proposed answer is: *(A)*.", "A"),
      ("answer is: *(A)*.", "A"),
      ("The proposed answer is: 50.", "50"),
      ("The proposed answer is ( $35 )", "35"),
      ("The proposed answer is (35.5)", "35"),
      ("The proposed answer is (-35)", "-35"),
      ("The proposed answer is (3,500)", "3500"),
      ("The proposed answer is: B. Moche.", "B"),
      (
          "The proposed answer is: ( - 60. ). Something else: (1)",
          "-60",
      ),
      ("**proposed Answer:** 21. Confidence: 1.", "21"),
      ("proposed Answer: $1.75. Confidence: (1)", "1"),
      ("The answer is: C. Confidence: 1.", "C"),
      ("answer is:    J", "J"),
      (
          (
              "Step-by-step explanation: The question asks about the"
              " development of one of the earliest kingdoms in South America."
              " The options provided are all ancient civilizations in"
              " Mesoamerica, which is a region that includes parts of Mexico"
              " and Central America, but not South America. Therefore, I will"
              " focus on the options that are more likely to be in South"
              " America. The Olmec civilization is known to have flourished in"
              " the region of Veracruz, Mexico, and the Wari culture was found"
              " in Peru. The Inca civilization is more commonly associated with"
              " the Andean region of South America. Given this information, I"
              " will eliminate options A, B, D, E, and F as they are not"
              " directly associated with South America. I will also eliminate"
              " option H as the Aztecs were not a South American civilization."
              " This leaves me with options C and G. I will choose option G,"
              " Wari, as it is the only remaining option that is associated"
              " with South America. The Wari culture was a pre-Columbian"
              " civilization that flourished in the Andean region of Peru."
              " Proposed answer: G. Proposed confidence: 1.<|eot_id|>."
          ),
          "G",
      ),
      # TODO(amirt): currently we don't support floats. Enable this test once
      # we do.
      # ("**[proposed Answer]**: $1.75.", "1.75"),
  )
  def test_extract_final_answer(self, text, expected):
    predicted_answer, _ = prompt_util.get_final_answer(
        text, match_part_pattern=r"((?:-?\s*[0-9,]+)|(?:[a-zA-Z]+))"
    )
    self.assertEqual(
        expected.lower() if expected is not None else None,
        predicted_answer.lower() if predicted_answer is not None else None,
    )

  @parameterized.parameters(
      ("proposed Answer. - The proposed answer is: 2. XXX: 1.", "2"),
      (
          " * [proposed Answer] The proposed answer is: 56. Confidence: (1)",
          "56",
      ),
  )
  def test_extract_final_answer_only_numbers(self, text, expected):
    predicted_answer, _ = prompt_util.get_final_answer(
        text, match_part_pattern=r"(-?\s*[0-9,]+)"
    )
    self.assertEqual(expected, predicted_answer)

  def test_extract_final_answer_with_span(self):
    part1 = "The proposed answer is          "
    part2 = "2. Bla bla: 1."

    ans, span = prompt_util.get_final_answer(
        part1 + part2, match_part_pattern=r"(-?\s*[0-9,]+)"
    )

    self.assertEqual(ans, "2")
    self.assertEqual(span, (len(part1), len(part1) + 1))

  def test_answer_format_instruction(self):
    output = prompt_util.general_instructions()
    expected = """Before giving your answer, provide a step-by-step explanation of your thought process.
Then on a new line, give your proposed answer adhering to this precise format: 'Proposed answer: (X).', where X is your proposed answer."""
    self.assertEqual(expected, output)


if __name__ == "__main__":
  absltest.main()
