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
from cisc.src.datasets import math


class MathTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='simple',
          text='Proposed answer: 100.',
          expected='100',
      ),
      dict(
          testcase_name='parentheses',
          text='Proposed answer: (100).....',
          expected='100',
      ),
      dict(
          testcase_name='parentheses_tex',
          text='Proposed answer: $(100)$. Proposed Proposed: 1.',
          expected='100',
      ),
      dict(
          testcase_name='answer_in_the_middle',
          text='Proposed answer: $140$. \n<end_of_turn><eos>',
          expected='140',
      ),
      dict(
          testcase_name='no_confidence_frac',
          text=r'Proposed answer: (frac{a}{b}).',
          expected='frac{a}{b}',
      ),
      dict(
          testcase_name='no_confidence_sqrt',
          text=r'Proposed answer: (sqrt{8}).',
          expected='sqrt{8}',
      ),
      dict(
          testcase_name='surronded_by_dollars',
          text=r'Proposed answer: $100$.',
          expected='100',
      ),
      dict(
          testcase_name='surronded_boxed',
          text=r'Proposed answer: $\boxed{117}$.',
          expected='117',
      ),
      dict(
          testcase_name='no_answer',
          text='Proposed something: 1.',
          expected='',
      ),
      dict(
          testcase_name='answer_with_</s>',
          text='Proposed answer: \\(-2\\).</s>',
          expected='-2',
      ),
      dict(
          testcase_name='answer_with_</s>_and_boxed',
          text='Proposed answer: \\( \\boxed{9} \\).</s>',
          expected='9',
      ),
  )
  def test_get_final_normalized_answer(self, text, expected):
    self.assertEqual(expected, math.get_final_normalized_answer(text)[0])

  @parameterized.named_parameters(
      dict(
          testcase_name='simple',
          text=r'Some gound truth the includes $\boxed{117}$ expression.',
          expected='117',
      ),
      dict(
          testcase_name='parentheses',
          text=r'Some gound truth the includes $\boxed{(118)}$ expression.',
          expected='118',
      ),
      dict(
          testcase_name='boxed_fraction',
          text=r'Gound truth the includes $\boxed{\frac{a}{b}}$ expression.',
          expected=r'frac{a}{b}',
      ),
      dict(
          testcase_name='boxed_sqrt',
          text=r'Some gound truth the includes $\boxed{\sqrt{8}}$ expression.',
          expected=r'sqrt{8}',
      ),
      dict(
          testcase_name='no_boxed',
          text=r'Some gound truth the includes $117$ expression.',
          expected='',
      ),
  )
  def test_extract_answer_from_last_box(self, text, expected):
    self.assertEqual(expected, math.extract_answer_from_last_box(text))


if __name__ == '__main__':
  absltest.main()
