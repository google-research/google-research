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

"""Tests for deepcoder_dsl."""

from absl.testing import absltest
from absl.testing import parameterized

from latent_programmer.tasks.deepcoder import deepcoder_dsl
# import deepcoder_dsl


class DeepcoderDslTest(parameterized.TestCase):

  def setUp(self):
    super(DeepcoderDslTest, self).setUp()
    self.head_op = deepcoder_dsl.TOKEN_TO_OPERATION['Head']
    self.map_op = deepcoder_dsl.TOKEN_TO_OPERATION['Map']
    self.plus_one_lambda = deepcoder_dsl.TOKEN_TO_LAMBDA['+1']
    self.square_lambda = deepcoder_dsl.TOKEN_TO_LAMBDA['**2']

  @parameterized.named_parameters(
      ('single_list', [['a', 'b']], ['a', 'b']),
      ('two_lists', [['a', 'b'], ['c']], ['a', 'b', ',', 'c']),
  )
  def test_join_token_lists(self, token_lists, expected_output):
    actual_output = deepcoder_dsl.join_token_lists(token_lists,
                                                   separator_token=',')
    self.assertEqual(actual_output, expected_output)

  @parameterized.named_parameters(
      ('int', 3, True),
      ('too_big_int', 3000, False),
      ('list', [1, 4, 2], True),
      ('list_with_big_int', [1, 4000, 2], False),
      ('None', None, False),
  )
  def test_validate_result(self, result, expected):
    self.assertEqual(deepcoder_dsl.validate_result(result), expected)

  def test_variable_token(self):
    self.assertEqual(deepcoder_dsl.variable_token(3), 'x3')

  def test_variable_index_from_token(self):
    self.assertEqual(deepcoder_dsl.variable_index_from_token('x3'), 3)
    with self.assertRaises(deepcoder_dsl.ParseError):
      deepcoder_dsl.variable_index_from_token('y1')
    with self.assertRaises(deepcoder_dsl.ParseError):
      deepcoder_dsl.variable_index_from_token('xx1')

  @parameterized.named_parameters(
      ('int', 7, ['7']),
      ('list_one', [4], ['[', '4', ']']),
      ('list_multi', [4, 2, 5], ['[', '4', ',', '2', ',', '5', ']']),
  )
  def test_tokenize_result_succeeds(self, result, expected):
    self.assertEqual(deepcoder_dsl.tokenize_result(result), expected)

  @parameterized.named_parameters(
      ('None', None),
      ('list_with_none', [3, 5, None, 2]),
  )
  def test_tokenize_result_raises(self, result):
    with self.assertRaises(deepcoder_dsl.DeepCoderError):
      deepcoder_dsl.tokenize_result(result)

  def test_program_state(self):
    state = deepcoder_dsl.ProgramState([2, [6, 7]])
    self.assertLen(state, 2)
    self.assertEqual(state[0], 2)
    self.assertLen(state, 2)  # Original is unaffected.
    with self.assertRaises(deepcoder_dsl.RunError):
      _ = state[-1]
    with self.assertRaises(deepcoder_dsl.RunError):
      _ = state[2]

    state_copy = state.copy()
    self.assertEqual(state, state_copy)
    self.assertIsNot(state, state_copy)
    state_copy.add_result(0)
    self.assertLen(state_copy, 3)

    self.assertEqual(state.get_output(), [6, 7])
    self.assertEqual(state.tokenize(),
                     ['x0', '=', '2', '|', 'x1', '=', '[', '6', ',', '7', ']'])
    self.assertEqual(str(state), 'x0 = 2 | x1 = [ 6 , 7 ]')

  def test_program_state_from_str(self):
    state_str = 'x0 = 2 | x1 = [6, 7]'
    state = deepcoder_dsl.ProgramState.from_str(state_str)
    # str(state) has different whitespace than state_str.
    self.assertEqual(str(state), 'x0 = 2 | x1 = [ 6 , 7 ]')

  @parameterized.named_parameters(
      ('bad_lhs_name', 'y0 = 3'),
      ('lhs_wrong_index', 'x0 = 3 | x2 = 4'),
      ('bad_equal_sign', 'x0 : 3'),
      ('invalid_result', 'x0 = None'),
  )
  def test_program_state_from_str_raises(self, bad_str):
    with self.assertRaises(deepcoder_dsl.ParseError):
      deepcoder_dsl.ProgramState.from_str(bad_str)

  def test_program_state_from_tokens(self):
    tokens = ['x0', '=', '[', '3', ']']
    state = deepcoder_dsl.ProgramState.from_tokens(tokens)
    self.assertEqual(state.get_output(), [3])

  def test_operation_run(self):
    self.assertEqual(self.head_op.run([[7, 2, 3]]), 7)
    # Operation corner case.
    self.assertIsNone(self.head_op.run([[]]))

    self.assertEqual(self.map_op.run(
        [self.plus_one_lambda.func, [7, 2, 3]]), [8, 3, 4])
    # Result is too big.
    self.assertIsNone(self.map_op.run([self.square_lambda.func, [20]]))

  def test_operation_run_raises(self):
    with self.assertRaises(deepcoder_dsl.RunError):
      self.head_op.run([[7, 2, 3], 0])  # Wrong arity.
    with self.assertRaises(deepcoder_dsl.RunError):
      self.map_op.run([0, [7, 2, 3]])  # First arg should be lambda.

  def test_statement(self):
    statement_1 = deepcoder_dsl.Statement.from_tokens(['x1', '=', 'Head', 'x0'])
    self.assertEqual(str(statement_1), 'x1 = Head x0')
    initial_state_1 = deepcoder_dsl.ProgramState([[3, 6]])
    result_state_1 = deepcoder_dsl.ProgramState([[3, 6], 3])
    self.assertEqual(statement_1.run(initial_state_1), result_state_1)

    statement_2 = deepcoder_dsl.Statement.from_str('x3 = Map +1 x1')
    self.assertEqual(statement_2.tokenize(), ['x3', '=', 'Map', '+1', 'x1'])
    initial_state_2 = deepcoder_dsl.ProgramState([[3], [5, 2, 8], 4])
    result_state_2 = deepcoder_dsl.ProgramState([[3], [5, 2, 8], 4, [6, 3, 9]])
    self.assertEqual(statement_2.run(initial_state_2), result_state_2)

  @parameterized.named_parameters(
      ('too_few_tokens', 'x1 = INPUT'),
      ('bad_equals', 'x1 == Head x0'),
      ('wrong_arity', 'x2 = Head x0 x1'),
      ('unexpected_lambda_head', 'x1 = Head +1'),
      ('unexpected_lambda_map', 'x1 = Map +1 +1'),
      ('needs_lambda_got_variable', 'x2 = Map x1 x0'),
      ('needs_lambda_got_operation', 'x1 = Map Map x0'),
      ('bad_variable', 'x1 = Map +1 y0'),
      ('unknown_operation', 'x1 = NotAnOp x0'),
  )
  def test_statement_from_string_raises(self, statement_str):
    with self.assertRaises(deepcoder_dsl.ParseError):
      deepcoder_dsl.Statement.from_str(statement_str)

  @parameterized.named_parameters(
      ('wrong_lhs_index', 'x2 = Head x0'),
      ('wrong_rhs_index', 'x1 = Head x1'),
  )
  def test_statement_run_raises(self, statement_str):
    statement = deepcoder_dsl.Statement.from_str(statement_str)
    with self.assertRaises(deepcoder_dsl.RunError):
      statement.run(deepcoder_dsl.ProgramState([[3, 6]]))

  def test_program(self):
    program_1 = deepcoder_dsl.Program.from_tokens(
        ['x0', '=', 'INPUT', '|', 'x1', '=', 'Head', 'x0'])
    self.assertEqual(str(program_1), 'x0 = INPUT | x1 = Head x0')
    self.assertEqual(program_1.run([[5, 3, 6, 4]]),
                     deepcoder_dsl.ProgramState([[5, 3, 6, 4], 5]))

    program_2 = deepcoder_dsl.Program.from_str(
        'x0 = INPUT | x1 = INPUT | x2 = Reverse x1 | x3 = ZipWith + x0 x2')
    self.assertEqual(
        program_2.run([[3, 2], [1, 4]]),
        deepcoder_dsl.ProgramState([[3, 2], [1, 4], [4, 1], [7, 3]]))

    program_3 = deepcoder_dsl.Program.from_str('x0 = INPUT | x1 = Head x0')
    self.assertEqual(program_3.run([[4]]), deepcoder_dsl.ProgramState([[4], 4]))
    self.assertIsNone(program_3.run([[]]))

  @parameterized.named_parameters(
      ('inputs_not_at_beginning', 'x0 = INPUT | x1 = Head x0 | x2 = INPUT'),
      ('bad_input_line', 'x0 = Head INPUT | x1 = Head x0'),
      ('bad_statement', 'x0 = INPUT | x1 = Head +1 x0'),
  )
  def test_program_from_string_raises(self, program_str):
    with self.assertRaises(deepcoder_dsl.ParseError):
      deepcoder_dsl.Program.from_str(program_str)

  def test_program_run_raises(self):
    program = deepcoder_dsl.Program.from_str('x0 = INPUT | x1 = Head x0')
    with self.assertRaises(deepcoder_dsl.RunError):
      program.run([[4], 7])


if __name__ == '__main__':
  absltest.main()
