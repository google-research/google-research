# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Test for utility library of instructions."""

from absl.testing import absltest
from absl.testing import parameterized
from instruction_following_eval import instructions_util


class InstructionsUtilTest(parameterized.TestCase):

  TEST_WORD_COUNT_CASE_1 = ("word1, word2, word3, word4.", 4)

  TEST_WORD_COUNT_CASE_2 = (
      """
      Bard can you tell me which is the best optimization method for the
      transition from an hydro-thermal system to an hydro-renewables system""",
      24)

  TEST_WORD_COUNT_CASE_3 = (
      """
      Hyphenated-word has two word counts.
      """, 6)

  def test_word_count(self):
    """Tests word counter."""
    with self.subTest(f"{self.TEST_WORD_COUNT_CASE_1[0]}"):
      text, expected_num_words = self.TEST_WORD_COUNT_CASE_1
      actual_num_words = instructions_util.count_words(text)
      self.assertEqual(expected_num_words, actual_num_words)

    with self.subTest(f"{self.TEST_WORD_COUNT_CASE_2[0]}"):
      text, expected_num_words = self.TEST_WORD_COUNT_CASE_2
      actual_num_words = instructions_util.count_words(text)
      self.assertEqual(expected_num_words, actual_num_words)

    with self.subTest(f"{self.TEST_WORD_COUNT_CASE_3[0]}"):
      text, expected_num_words = self.TEST_WORD_COUNT_CASE_3
      actual_num_words = instructions_util.count_words(text)
      self.assertEqual(expected_num_words, actual_num_words)

  @parameterized.named_parameters(
      [
          {  # pylint: disable=g-complex-comprehension
              "testcase_name": (
                  f"_response={response}_num_sentences={num_sentences}"
              ),
              "response": response,
              "num_sentences": num_sentences,
          }
          for response, num_sentences in [
              ("xx,x. xx,x! xx/x. x{x}x? x.", 5),
              ("xx,x! xxxx. x(x)x?", 3),
              ("xxxx. xx,x! xx|x. x&x x?", 4),
              ("xx-x]xx,x! x{x}xx,x.", 2),
          ]
      ]
  )
  def test_count_sentences(self, response, num_sentences):
    """Tests sentence counter."""
    actual_num_sentences = instructions_util.count_sentences(response)
    self.assertEqual(num_sentences, actual_num_sentences)

  TEST_SENTENCE_SPLIT_1 = """
  Google is a technology company. It was founded in 1998 by Larry Page
and Sergey Brin. Google's mission is to organize the world's information
and make it universally accessible and useful.
  """

  TEST_SENTENCE_SPLIT_2 = """
  The U.S.A has many Ph.D. students. They will often haven a .com website
sharing the research that they have done.
  """

  EXPECTED_SENTENCE_SPLIT_1 = [
      "Google is a technology company.",
      "It was founded in 1998 by Larry Page and Sergey Brin.",
      (
          "Google's mission is to organize the world's information and make it"
          " universally accessible and useful."
      ),
  ]

  EXPECTED_SENTENCE_SPLIT_2 = [
      "The U.S.A has many Ph.D. students.",
      (
          "They will often haven a .com website sharing the research that they"
          " have done."
      ),
  ]

  def test_sentence_splitter(self):
    """Tests sentence splitter."""
    sentence_split_1 = instructions_util.split_into_sentences(
        self.TEST_SENTENCE_SPLIT_1
    )
    sentence_split_2 = instructions_util.split_into_sentences(
        self.TEST_SENTENCE_SPLIT_2
    )

    self.assertEqual(self.EXPECTED_SENTENCE_SPLIT_1, sentence_split_1)
    self.assertEqual(self.EXPECTED_SENTENCE_SPLIT_2, sentence_split_2)

  def test_generate_keywords(self):
    """Tests generate keywords."""
    self.assertLen(instructions_util.generate_keywords(10), 10)


if __name__ == "__main__":
  absltest.main()
