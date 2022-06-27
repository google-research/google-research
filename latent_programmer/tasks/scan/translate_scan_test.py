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

"""Tests for latent_programmer.tasks.scan.translate_scan."""

from absl.testing import absltest
from absl.testing import parameterized

from latent_programmer.tasks.scan import scan_vocab
from latent_programmer.tasks.scan import translate_scan


class TranslateScanTest(parameterized.TestCase):

  @parameterized.named_parameters(
      # Simple tests from README at https://github.com/brendenlake/SCAN.
      ('jump', 'jump', 'I_JUMP'),
      ('jump_left', 'jump left', 'I_TURN_LEFT I_JUMP'),
      ('jump_around', 'jump around right',
       ('I_TURN_RIGHT I_JUMP I_TURN_RIGHT I_JUMP '
        'I_TURN_RIGHT I_JUMP I_TURN_RIGHT I_JUMP')),
      ('turn_twice', 'turn left twice', 'I_TURN_LEFT I_TURN_LEFT'),
      ('jump_thrice', 'jump thrice', 'I_JUMP I_JUMP I_JUMP'),
      ('jump_and_walk', 'jump opposite left and walk thrice',
       'I_TURN_LEFT I_TURN_LEFT I_JUMP I_WALK I_WALK I_WALK'),
      ('jump_after_walk', 'jump opposite left after walk around left',
       ('I_TURN_LEFT I_WALK I_TURN_LEFT I_WALK '
        'I_TURN_LEFT I_WALK I_TURN_LEFT I_WALK '
        'I_TURN_LEFT I_TURN_LEFT I_JUMP')),

      # Manual tasks with more parts.
      ('long_program',
       'jump and walk and run twice after turn left and look after jump thrice',
       ('I_JUMP I_JUMP I_JUMP '
        'I_TURN_LEFT I_LOOK '
        'I_JUMP I_WALK I_RUN I_RUN')),
  )
  def test_translate(self, input_str, expected_output_str):
    input_tokens = input_str.split()
    expected_output_tokens = expected_output_str.split()
    output_tokens = translate_scan.translate(input_tokens, add_separators=False)
    self.assertEqual(expected_output_tokens, output_tokens)

    output_tokens = translate_scan.translate(input_tokens, add_separators=True)
    if 'and' in input_tokens or 'after' in input_tokens:
      self.assertIn(scan_vocab.SEP, output_tokens)
    output_tokens = [t for t in output_tokens if t != scan_vocab.SEP]
    self.assertEqual(expected_output_tokens, output_tokens)


if __name__ == '__main__':
  absltest.main()
