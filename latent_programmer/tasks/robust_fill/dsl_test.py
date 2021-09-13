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

"""Tests for latent_programmer.tasks.robust_fill.dsl."""

from absl.testing import absltest

from latent_programmer.tasks.robust_fill import dsl
from latent_programmer.tasks.robust_fill import sample_random
from latent_programmer.tasks.robust_fill import tokens


class DSLTest(absltest.TestCase):

  def test_programs(self):
    program1 = dsl.Concat(
        dsl.GetToken(dsl.Type.ALPHANUM, 3),
        dsl.GetFrom(':'),
        dsl.GetFirst(dsl.Type.CHAR, 4))
    self.assertEqual(program1('Ud 9:25,JV3 Obb'), '2525,JV3 ObbUd 9')
    self.assertEqual(program1('zLny xmHg 8:43 A44q'), '843 A44qzLny')

    program2 = dsl.Concat(
        dsl.Compose(
            dsl.Replace(' ', ','),
            dsl.GetSpan(dsl.Type.PROP_CASE, 1, dsl.Boundary.START,
                        dsl.Type.PROP_CASE, 4, dsl.Boundary.END)),
        dsl.ConstStr('.'),
        dsl.GetToken(dsl.Type.PROP_CASE, -1))
    self.assertEqual(program2('Jacob Ethan James Alexander Michael'),
                     'Jacob,Ethan,James,Alexander.Michael')
    self.assertEqual(program2('Earth Fire Wind Water Pluto Sun'),
                     'Earth,Fire,Wind,Water.Sun')

  def test_decode(self):
    id_token_table, token_id_table = tokens.build_token_tables()
    self.assertEqual(len(token_id_table), len(id_token_table))
    program = dsl.Concat(
        dsl.Compose(
            dsl.Replace(' ', ','),
            dsl.GetSpan(dsl.Type.PROP_CASE, 1, dsl.Boundary.START,
                        dsl.Type.PROP_CASE, 4, dsl.Boundary.END)),
        dsl.ConstStr('.'),
        dsl.GetToken(dsl.Type.PROP_CASE, -1))
    encoding = program.encode(token_id_table)
    self.assertEqual(encoding[-1], token_id_table[dsl.EOS])

    decoded_program = dsl.decode_program(encoding, id_token_table)
    self.assertEqual(decoded_program('Jacob Ethan James Alexander Michael'),
                     'Jacob,Ethan,James,Alexander.Michael')
    self.assertEqual(decoded_program('Earth Fire Wind Water Pluto Sun'),
                     'Earth,Fire,Wind,Water.Sun')

  def test_sample_random(self):
    for _ in range(10):
      example = sample_random.random_task(
          max_expressions=10,
          max_k=5,
          max_input_tokens=10,
          max_input_length=100,
          max_output_length=100,
          num_examples=4)
      self.assertGreater(min(len(out) for out in example.outputs), 0)

      outputs = [example.program(inp) for inp in example.inputs]
      self.assertListEqual(outputs, example.outputs)


if __name__ == '__main__':
  absltest.main()
