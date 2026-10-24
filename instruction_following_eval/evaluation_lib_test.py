# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Tests for instruction_following_eval.evaluation_lib."""

import json
import os

from absl.testing import absltest
from instruction_following_eval import evaluation_lib

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


class RepeatPromptArgumentTest(absltest.TestCase):
  """`combination:repeat_prompt` needs the prompt passed to build_description.

  `RepeatPromptThenAnswer.build_description` raises when `prompt_to_repeat` is
  missing, and the prompt is not part of the record's kwargs, so the evaluator
  has to supply it.
  """

  PROMPT = 'Write a haiku about the sea.'
  FOLLOWED = 'Write a haiku about the sea. Waves curl on the sand.'
  NOT_FOLLOWED = 'Waves curl on the sand, gulls cry above the grey sea.'

  def _example(self):
    return evaluation_lib.InputExample(
        key=1,
        instruction_id_list=['combination:repeat_prompt'],
        prompt=self.PROMPT,
        kwargs=[{}],
    )

  def test_strict_evaluation_of_repeated_prompt(self):
    example = self._example()
    output = evaluation_lib.test_instruction_following_strict(
        example, {self.PROMPT: self.FOLLOWED}
    )
    self.assertEqual(output.follow_instruction_list, [True])
    self.assertTrue(output.follow_all_instructions)

  def test_strict_evaluation_when_prompt_not_repeated(self):
    example = self._example()
    output = evaluation_lib.test_instruction_following_strict(
        example, {self.PROMPT: self.NOT_FOLLOWED}
    )
    self.assertEqual(output.follow_instruction_list, [False])
    self.assertFalse(output.follow_all_instructions)

  def test_loose_evaluation_of_repeated_prompt(self):
    example = self._example()
    output = evaluation_lib.test_instruction_following_loose(
        example, {self.PROMPT: self.FOLLOWED}
    )
    self.assertEqual(output.follow_instruction_list, [True])


class CorpusBuildsTest(absltest.TestCase):
  """Every instruction in the shipped data must still build a description."""

  def test_all_instructions_build(self):
    path = os.path.join(_DATA_DIR, 'input_data.jsonl')
    if not os.path.exists(path):
      self.skipTest('input_data.jsonl not available')
    with open(path) as f:
      rows = [json.loads(line) for line in f if line.strip()]

    for row in rows:
      for instruction_id, kwargs in zip(row['instruction_id_list'],
                                        row['kwargs']):
        checker = evaluation_lib.instructions_registry.INSTRUCTION_DICT[
            instruction_id](instruction_id)
        with self.subTest(f'key={row["key"]} instruction={instruction_id}'):
          checker.build_description(
              **evaluation_lib._with_prompt_kwargs(
                  checker, kwargs, row['prompt']
              )
          )


if __name__ == '__main__':
  absltest.main()
