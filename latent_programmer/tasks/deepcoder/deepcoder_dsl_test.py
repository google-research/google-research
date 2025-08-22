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

"""Tests for deepcoder_dsl."""

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized

from latent_programmer.tasks.deepcoder import deepcoder_dsl as dsl

FLAGS = flags.FLAGS


class DeepcoderDslTest(parameterized.TestCase):

  def setUp(self):
    super(DeepcoderDslTest, self).setUp()
    self.head_op = dsl.TOKEN_TO_OPERATION['Head']
    self.map_op = dsl.TOKEN_TO_OPERATION['Map']
    self.plus_one_lambda = dsl.TOKEN_TO_LAMBDA['(+1)']
    self.square_lambda = dsl.TOKEN_TO_LAMBDA['(**2)']
    self.times_3_lambda = dsl.TOKEN_TO_LAMBDA['(*3)']
    self._saved_flags = flagsaver.save_flag_values()

  def tearDown(self):
    flagsaver.restore_flag_values(self._saved_flags)
    super(DeepcoderDslTest, self).tearDown()

  @parameterized.named_parameters(
      ('single_list', [['a', 'b']], ['a', 'b']),
      ('two_lists', [['a', 'b'], ['c']], ['a', 'b', ',', 'c']),
  )
  def test_join_token_lists(self, token_lists, expected_output):
    actual_output = dsl.join_token_lists(token_lists, separator_token=',')
    self.assertEqual(actual_output, expected_output)

  @parameterized.named_parameters(
      ('int', 3, True),
      ('too_big_int', 3000, False),
      ('list', [1, 4, 2], True),
      ('list_with_big_int', [1, 4000, 2], False),
      ('None', None, False),
  )
  def test_validate_result(self, result, expected):
    self.assertEqual(dsl.validate_result(result), expected)

  def test_variable_token(self):
    self.assertEqual(dsl.variable_token(3), 'x3')

  def test_variable_index_from_token(self):
    self.assertEqual(dsl.variable_index_from_token('x3'), 3)
    with self.assertRaises(dsl.ParseError):
      dsl.variable_index_from_token('y1')
    with self.assertRaises(dsl.ParseError):
      dsl.variable_index_from_token('xx1')
    with self.assertRaises(dsl.ParseError):
      # Index too large.
      dsl.variable_index_from_token('x100')

  @parameterized.named_parameters(
      ('int', 7, ['7']),
      ('list_one', [4], ['[', '4', ']']),
      ('list_multi', [4, 2, 5], ['[', '4', '2', '5', ']']),
  )
  def test_tokenize_result_succeeds(self, result, expected):
    self.assertEqual(dsl.tokenize_result(result), expected)

  @parameterized.named_parameters(
      ('None', None),
      ('list_with_none', [3, 5, None, 2]),
  )
  def test_tokenize_result_raises(self, result):
    with self.assertRaises(dsl.DeepCoderError):
      dsl.tokenize_result(result)

  def test_program_state(self):
    state = dsl.ProgramState([2, [6, 7]], ['x4', 'x2'])
    self.assertLen(state, 2)
    self.assertEqual(state.get_index(0), 2)
    self.assertEqual(state.get_variable('x2'), [6, 7])
    with self.assertRaises(dsl.RunError):
      _ = state.get_index(-1)
    with self.assertRaises(dsl.RunError):
      _ = state.get_index(2)
    with self.assertRaises(dsl.RunError):
      _ = state.get_variable('x0')

    self.assertEqual(state.get_output(), [6, 7])
    self.assertEqual(state.get_output_variable(), 'x2')
    self.assertEqual(state.tokenize(),
                     ['x4', '=', '2', '|', 'x2', '=', '[', '6', '7', ']'])
    self.assertEqual(str(state), 'x4 = 2 | x2 = [ 6 7 ]')

    state_copy = state.copy()
    self.assertEqual(state, state_copy)
    self.assertIsNot(state, state_copy)
    state_copy.add_result(0, 'x1')
    self.assertLen(state_copy, 3)
    self.assertLen(state, 2)  # Original is unaffected.
    self.assertEqual(state_copy.get_output(), 0)
    self.assertEqual(state_copy.get_output_variable(), 'x1')

  def test_program_state_from_str(self):
    state_str = 'x3 = 2 | x5 = [6 7]'
    state = dsl.ProgramState.from_str(state_str)
    # str(state) has different whitespace than state_str.
    self.assertEqual(str(state), 'x3 = 2 | x5 = [ 6 7 ]')

  @parameterized.named_parameters(
      ('bad_lhs_name', 'y0 = 3'),
      ('lhs_wrong_format', 'x0 x1 = 3'),
      ('duplicate_variable', 'x0 = 3 | x0 = 4'),
      ('bad_equal_sign', 'x0 : 3'),
      ('invalid_result', 'x0 = None'),
  )
  def test_program_state_from_str_raises(self, bad_str):
    with self.assertRaises(dsl.ParseError):
      dsl.ProgramState.from_str(bad_str)

  def test_program_state_from_tokens(self):
    tokens = ['x0', '=', '[', '3', ']']
    state = dsl.ProgramState.from_tokens(tokens)
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
    with self.assertRaises(dsl.RunError):
      self.head_op.run([[7, 2, 3], 0])  # Wrong arity.
    with self.assertRaises(dsl.RunError):
      self.map_op.run([0, [7, 2, 3]])  # First arg should be lambda.

  def test_statement(self):
    statement_1 = dsl.Statement.from_tokens(['x1', '=', 'Head', 'x3'])
    self.assertEqual(str(statement_1), 'x1 = Head x3')
    initial_state_1 = dsl.ProgramState([[3, 6]], ['x3'])
    result_state_1 = dsl.ProgramState([[3, 6], 3], ['x3', 'x1'])
    self.assertEqual(statement_1.run(initial_state_1), result_state_1)

    statement_2 = dsl.Statement.from_str('x0 = Map (+1) x1')
    self.assertEqual(statement_2.tokenize(), ['x0', '=', 'Map', '(+1)', 'x1'])
    initial_state_2 = dsl.ProgramState([[3], [5, 2, 8], 4], ['x4', 'x1', 'x3'])
    result_state_2 = dsl.ProgramState([[3], [5, 2, 8], 4, [6, 3, 9]],
                                      ['x4', 'x1', 'x3', 'x0'])
    self.assertEqual(statement_2.run(initial_state_2), result_state_2)

  @parameterized.named_parameters(
      ('too_few_tokens', 'x1 = INPUT'),
      ('bad_lhs_variable', 'y1 = Map (+1) x0'),
      ('bad_equals', 'x1 == Head x0'),
      ('unknown_operation', 'x1 = NotAnOp x0'),
      ('unexpected_lambda_head', 'x1 = Head (+1)'),
      ('unexpected_lambda_map', 'x1 = Map (+1) (+1)'),
      ('needs_lambda_got_variable', 'x2 = Map x1 x0'),
      ('needs_lambda_got_operation', 'x1 = Map Map x0'),
      ('lhs_as_arg', 'x1 = Map (+1) x1'),
      ('bad_arg_variable', 'x1 = Map (+1) y0'),
      ('wrong_arity', 'x2 = Head x0 x1'),
  )
  def test_statement_from_string_raises(self, statement_str):
    with self.assertRaises(dsl.ParseError):
      dsl.Statement.from_str(statement_str)

  def test_statement_run(self):
    initial_state = dsl.ProgramState([1, [3, 7]], ['x0', 'x6'])
    with self.assertRaises(dsl.RunError):
      # Variable x0 already exists.
      dsl.Statement.from_str('x0 = Head x6').run(initial_state)
    statement = dsl.Statement.from_str('x1 = Access x0 x6')
    self.assertEqual(
        statement.run(initial_state).get_output(), 7)
    bad_initial_state = dsl.ProgramState([4, [3, 7]], ['x0', 'x6'])
    # Access index out of bounds.
    self.assertIsNone(statement.run(bad_initial_state))

  def test_program(self):
    program_1 = dsl.Program.from_tokens(
        ['x3', '=', 'INPUT', '|', 'x1', '=', 'Head', 'x3'])
    self.assertEqual(str(program_1), 'x3 = INPUT | x1 = Head x3')
    self.assertEqual(program_1.run([[5, 3, 6, 4]]),
                     dsl.ProgramState([[5, 3, 6, 4], 5], ['x3', 'x1']))
    self.assertEqual(program_1.num_inputs, 1)
    self.assertEqual(program_1.get_variables(), ['x3', 'x1'])
    self.assertLen(program_1, 1)

    program_2 = dsl.Program.from_str(
        'x0 = INPUT | x1 = INPUT | x2 = Reverse x1 | x3 = ZipWith (+) x0 x2')
    self.assertEqual(program_2.run([[3, 2], [1, 4]]),
                     dsl.ProgramState([[3, 2], [1, 4], [4, 1], [7, 3]],
                                      ['x0', 'x1', 'x2', 'x3']))
    self.assertEqual(program_2.num_inputs, 2)
    self.assertEqual(program_2.get_variables(), ['x0', 'x1', 'x2', 'x3'])
    self.assertLen(program_2, 2)

    program_3 = dsl.Program.from_str('x1 = INPUT | x0 = Head x1')
    self.assertEqual(program_3.run([[4]]),
                     dsl.ProgramState([[4], 4], ['x1', 'x0']))
    self.assertIsNone(program_3.run([[]]))

  @parameterized.named_parameters(
      ('inputs_not_at_beginning', 'x0 = INPUT | x1 = Head x0 | x2 = INPUT'),
      ('bad_variable', 'y0 = INPUT | x1 = Head x0'),
      ('bad_input_line', 'x0 = Head INPUT | x1 = Head x0'),
      ('bad_statement', 'x0 = INPUT | x1 = Head (+1) x0'),
  )
  def test_program_from_string_raises(self, program_str):
    with self.assertRaises(dsl.ParseError):
      dsl.Program.from_str(program_str)

  def test_program_run_raises(self):
    program = dsl.Program.from_str('x0 = INPUT | x1 = Head x0')
    with self.assertRaises(dsl.RunError):
      # Wrong number of inputs.
      program.run([[4], 7])

  @parameterized.parameters(
      ('Head', [[5, 6, 7]], 5),
      ('Head', [[]], None),
      ('Last', [[5, 6, 7]], 7),
      ('Last', [[]], None),
      ('Take', [2, [3, 5, 8, 4]], [3, 5]),
      ('Take', [0, [3, 5, 8, 4]], []),
      ('Take', [-3, [3, 5, 8, 4]], [3]),
      ('Take', [5, [3, 5, 8, 4]], [3, 5, 8, 4]),
      ('Drop', [2, [6, 1, 3]], [3]),
      ('Drop', [0, [6, 1, 3]], [6, 1, 3]),
      ('Drop', [-2, [6, 1, 3]], [1, 3]),
      ('Drop', [4, [6, 1, 3]], []),
      ('Access', [-1, [7, 8, 9]], None),
      ('Access', [0, [7, 8, 9]], 7),
      ('Access', [2, [7, 8, 9]], 9),
      ('Access', [3, [7, 8, 9]], None),
      ('Maximum', [[6, 8, 4]], 8),
      ('Maximum', [[]], None),
      ('Minimum', [[6, 2, 4]], 2),
      ('Minimum', [[]], None),
      ('Reverse', [[3, 7, 2]], [2, 7, 3]),
      ('Reverse', [[]], []),
      ('Sort', [[3, 6, 3, 1, 5]], [1, 3, 3, 5, 6]),
      ('Sort', [[]], []),
      ('Sum', [[3, 5, 1]], 9),
      ('Sum', [[]], 0),
  )
  def test_first_order_operations(self, token, inputs, expected):
    op = dsl.TOKEN_TO_OPERATION[token]
    self.assertIsInstance(op, dsl.FirstOrderOperation)
    self.assertLen(inputs, op.arity)
    self.assertLen(op.inputs_type, op.arity)
    for inp, inp_type in zip(inputs, op.inputs_type):
      self.assertEqual(type(inp), inp_type)
    result = op.run(inputs)
    self.assertEqual(result, expected)
    if result is not None:
      self.assertEqual(type(result), op.output_type)

  @parameterized.parameters(
      ('Map', '(+1)', [[5, 2, 7]], [6, 3, 8]),
      ('Map', '(+1)', [[-4]], [-3]),
      ('Map', '(+1)', [[]], []),
      ('Map', '(-1)', [[5, 2, 7]], [4, 1, 6]),
      ('Map', '(*2)', [[2, 0, 3, 1]], [4, 0, 6, 2]),
      ('Map', '(/2)', [[4, 3, 0, 7, 6, -3]], [2, 1, 0, 3, 3, -2]),
      ('Map', '(*(-1))', [[4, -6, 0]], [-4, 6, 0]),
      ('Map', '(**2)', [[0, -3, 2]], [0, 9, 4]),
      ('Map', '(*3)', [[1, 3, 0]], [3, 9, 0]),
      ('Map', '(/3)', [[-6, -5, 0, 3, 4, 7]], [-2, -2, 0, 1, 1, 2]),
      ('Map', '(*4)', [[2]], [8]),
      ('Map', '(/4)', [[8, 1, 0, -1]], [2, 0, 0, -1]),
      ('Filter', '(>0)', [[4, -1, 0, 2, -4]], [4, 2]),
      ('Filter', '(%2==0)', [[4, -1, 0, 2, -4]], [4, 0, 2, -4]),
      ('Count', '(<0)', [[4, -1, 0, 2, -4]], 2),
      ('Count', '(%2==1)', [[4, -1, 0, 2, -4]], 1),
      ('ZipWith', '(-)', [[3, 2, 5], [-2, 4, 1]], [5, -2, 4]),
      ('ZipWith', '(*)', [[3, 2, 5], [-2, 4, 1, 3]], [-6, 8, 5]),
      ('ZipWith', '(min)', [[3, 2, 5, 0], [-2, 4, 1]], [-2, 2, 1]),
      ('ZipWith', '(+)', [[], [1]], []),
      ('Scanl1', '(+)', [[]], []),
      ('Scanl1', '(+)', [[6]], [6]),
      ('Scanl1', '(+)', [[6, -2, -5, 3]], [6, 4, -1, 2]),
      ('Scanl1', '(max)', [[-3, 2, -1, 3, 2, 5]], [-3, 2, 2, 3, 3, 5]),
  )
  def test_higher_order_operations(self, op_token, lambda_token, inputs,
                                   expected):
    op = dsl.TOKEN_TO_OPERATION[op_token]
    self.assertIsInstance(op, dsl.HigherOrderOperation)
    lambda_object = dsl.TOKEN_TO_LAMBDA[lambda_token]
    self.assertEqual((lambda_object.inputs_type, lambda_object.output_type),
                     op.inputs_type[0])
    # Here `inputs` excludes the lambda which is normally the first input.
    self.assertLen(inputs, op.arity - 1)
    self.assertLen(op.inputs_type, op.arity)
    for inp, inp_type in zip(inputs, op.inputs_type[1:]):
      self.assertEqual(type(inp), inp_type)
    result = op.run([lambda_object.func] + inputs)
    self.assertEqual(result, expected)
    if result is not None:
      self.assertEqual(type(result), op.output_type)

  def test_vocab_tables(self):
    id_to_token, token_to_id = dsl.vocab_tables()

    tokens = list(id_to_token.values())
    for token in tokens:
      if tokens.count(token) > 1:
        indices = [i for i in range(len(tokens)) if tokens[i] == token]
        self.assertIsNone(f'Token {token} duplicated at indices {indices}.')
      self.assertNotIn(' ', token)  # No token should have spaces in it.

    self.assertLen(id_to_token, len(token_to_id))
    for i in range(len(id_to_token)):
      self.assertEqual(i, token_to_id[id_to_token[i]])

    program = dsl.Program.from_str(
        'x0 = INPUT | x9 = INPUT | x2 = Reverse x9 | x5 = ZipWith (+) x0 x2')
    for token in program.tokenize():
      self.assertIn(token, token_to_id)

    state = program.run([[5, 2, 6, 8], [9, 1, 3, 4]])
    self.assertEqual(state.get_output(), [9, 5, 7, 17])
    for token in state.tokenize():
      self.assertIn(token, token_to_id)


if __name__ == '__main__':
  absltest.main()
