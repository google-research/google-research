# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Tests for two_sat.dnf_lib."""

from absl.testing import absltest
from absl.testing import parameterized
import mock

from bangbang_qaoa.two_sat import dnf_lib


class ClauseTest(parameterized.TestCase):

  def test_init(self):
    dnf_lib.Clause(2, 4, False, False)

  @parameterized.parameters(
      ((6, 6, True, False), 'index1 and index2 must be different'),
      ((-4, 3, False, True), 'index1 must be a non-negative integer, not -4'),
      ((12, -10, True, True), 'index2 must be a non-negative integer, not -10'),
  )
  def test_init_neg_index(self, parameters, error_message):
    with self.assertRaisesRegex(
        ValueError,
        error_message):
      dnf_lib.Clause(parameters[0], parameters[1], parameters[2], parameters[3])

  @parameterized.parameters(
      (dnf_lib.Clause(3, 2, False, False), '(x_2 || x_3)'),
      (dnf_lib.Clause(0, 1, True, True), '(!x_0 || !x_1)'),
      (dnf_lib.Clause(5, 7, False, True), '(x_5 || !x_7)'),
      (dnf_lib.Clause(4, 5, True, False), '(!x_4 || x_5)'),
  )
  def test_str(self, clause, expected_string):
    self.assertEqual(str(clause), expected_string)

  @parameterized.parameters(
      (dnf_lib.Clause(4, 0, True, False), dnf_lib.Clause(4, 0, True, False)),
      # Order doesn't matter
      (dnf_lib.Clause(22, 7, False, True), dnf_lib.Clause(7, 22, True, False)),
  )
  def test_eq(self, clause1, clause2):
    self.assertEqual(clause1, clause2)

  @parameterized.parameters(
      # Indices off
      (dnf_lib.Clause(26, 27, False, False),
       dnf_lib.Clause(26, 28, False, False)),
      # Negations off
      (dnf_lib.Clause(2, 1, False, True), dnf_lib.Clause(2, 1, True, True)),
  )
  def test_not_eq(self, clause1, clause2):
    self.assertNotEqual(clause1, clause2)

  @parameterized.parameters(
      (dnf_lib.Clause(25, 2, False, True), dnf_lib.Clause(25, 2, False, True)),
      (dnf_lib.Clause(3, 5, True, False), dnf_lib.Clause(5, 3, False, True)),
  )
  def test_clause_hash(self, clause1, clause2):
    self.assertEqual(clause1.__hash__(), clause2.__hash__())

  @parameterized.parameters(
      ([True, True], True),
      ([True, False], False),
      ([False, False, True], True),
      ([True, False, False], False),
  )
  def test_eval(self, literals, expected_evaluation):
    # not(x_0) OR x_1
    clause1 = dnf_lib.Clause(0, 1, True, False)
    self.assertEqual(clause1.eval(literals), expected_evaluation)

  @parameterized.parameters(
      ([True, True, False, False], True),
      ([True, False, True, False], False)
  )
  def test_eval_index_error(self, literals, expected_evaluation):
    # x_1 OR x_3
    clause2 = dnf_lib.Clause(1, 3, False, False)
    self.assertEqual(clause2.eval(literals), expected_evaluation)
    with self.assertRaises(IndexError):
      clause2.eval([True])
    with self.assertRaises(IndexError):
      clause2.eval([True, False, True])

  def test_get_random_clause(self):
    clause = dnf_lib.get_random_clause(5)
    self.assertIsInstance(clause, dnf_lib.Clause)
    self.assertLessEqual(clause.index1, 4)
    self.assertGreaterEqual(clause.index1, 0)
    self.assertLessEqual(clause.index2, 4)
    self.assertGreaterEqual(clause.index2, 0)


class DNFTest(parameterized.TestCase):

  @parameterized.parameters(
      (
          dnf_lib.DNF(20, []),
          20,
          set(),
          0
      ),
      (
          dnf_lib.DNF(5, [dnf_lib.Clause(0, 1, True, True),
                          dnf_lib.Clause(3, 4, False, False)]),
          5,
          set([dnf_lib.Clause(0, 1, True, True),
               dnf_lib.Clause(3, 4, False, False)]),
          2
      ),
      (
          dnf_lib.DNF(4, [dnf_lib.Clause(0, 1, False, False),
                          dnf_lib.Clause(0, 1, True, True),
                          dnf_lib.Clause(0, 1, False, True),
                          dnf_lib.Clause(0, 1, True, False)]),
          4,
          set([dnf_lib.Clause(0, 1, False, False),
               dnf_lib.Clause(0, 1, True, True),
               dnf_lib.Clause(0, 1, False, True),
               dnf_lib.Clause(0, 1, True, False)]),
          3
      ),
      # Duplicated clauses.
      (
          dnf_lib.DNF(5, [dnf_lib.Clause(0, 1, True, True),
                          dnf_lib.Clause(0, 1, True, True)]),
          5,
          set([dnf_lib.Clause(0, 1, True, True)]),
          1
      ),
  )
  def test_init(
      self, dnf, expected_literals, expected_clauses, expected_optimal):
    self.assertEqual(dnf.num_literals, expected_literals)
    self.assertSetEqual(dnf.clauses, expected_clauses)
    self.assertEqual(dnf.optimal_num_satisfied, expected_optimal)

  @parameterized.parameters(
      (
          dnf_lib.DNF(5, [dnf_lib.Clause(0, 1, True, True),
                          dnf_lib.Clause(3, 4, False, False)]),
          2
      ),
      (
          dnf_lib.DNF(4, [dnf_lib.Clause(0, 1, False, False),
                          dnf_lib.Clause(0, 1, True, True),
                          dnf_lib.Clause(0, 1, False, True),
                          dnf_lib.Clause(0, 1, True, False)]),
          3
      ),
  )
  def test_get_optimal_num_satisfied(self, dnf, expected_value):
    self.assertEqual(dnf._get_optimal_num_satisfied(), expected_value)

  @parameterized.parameters(
      (-5, 'num_literals must be at least 2, not -5'),
      (1, 'num_literals must be at least 2, not 1'),
  )
  def test_init_neg_num(self, num_literals, error_message):
    with self.assertRaisesRegex(
        ValueError,
        error_message):
      dnf_lib.DNF(num_literals, set())

  def test_str(self):
    clauses = [
        dnf_lib.Clause(0, 2, False, False),
        dnf_lib.Clause(1, 2, False, True)
    ]
    dnf = dnf_lib.DNF(3, clauses)
    expected_regex = (r'^Number of Literals: 3, '
                      r'DNF: \(\!?x_\d \|\| \!?x_\d\) && '
                      r'\(\!?x_\d \|\| \!?x_\d\)$')
    self.assertRegex(str(dnf), expected_regex)
    self.assertIn('(x_0 || x_2)', str(dnf))
    self.assertIn('(x_1 || !x_2)', str(dnf))

  @parameterized.parameters(
      ([True, True, True], True),
      ([True, True, False], True),
      ([True, False, True], False),
      ([True, False, False], True),
      ([False, True, True], True),
      ([False, True, False], False),
      ([False, False, True], False),
      ([False, False, False], False),
  )
  def test_eval(self, literals, expected_evaluation):
    # (x_0 OR x_2) AND (x_1 OR not(x_2))
    clauses = [
        dnf_lib.Clause(0, 2, False, False),
        dnf_lib.Clause(1, 2, False, True)
    ]
    dnf = dnf_lib.DNF(3, clauses)
    self.assertEqual(dnf.eval(literals), expected_evaluation)

  @parameterized.parameters(
      ([True, True, True], 2),
      ([True, True, False], 2),
      ([True, False, True], 1),
      ([True, False, False], 2),
      ([False, True, True], 2),
      ([False, True, False], 1),
      ([False, False, True], 1),
      ([False, False, False], 1),
  )
  def test_get_num_clauses_satisfied(self,
                                     literals,
                                     expected_clauses_satisfied):
    # (x_0 OR x_2) AND (x_1 OR not(x_2))
    clauses = [
        dnf_lib.Clause(0, 2, False, False),
        dnf_lib.Clause(1, 2, False, True)
    ]
    dnf = dnf_lib.DNF(3, clauses)
    self.assertEqual(dnf.get_num_clauses_satisfied(literals),
                     expected_clauses_satisfied)

  @parameterized.parameters((2, 4), (5, 40), (10, 180))
  def test_get_number_of_possible_clauses(
      self, num_literals, expected_number_of_possible_clauses):
    self.assertEqual(
        dnf_lib.get_number_of_possible_clauses(num_literals),
        expected_number_of_possible_clauses)

  def test_get_random_dnf(self):
    with mock.patch.object(
        dnf_lib, 'get_random_clause') as mock_get_random_clause:
      mock_get_random_clause.side_effect = [
          dnf_lib.Clause(0, 1, True, True),
          dnf_lib.Clause(0, 1, True, True),  # Duplicaed clause.
          dnf_lib.Clause(0, 1, True, False),
          dnf_lib.Clause(1, 2, True, True),
          dnf_lib.Clause(3, 4, True, True),
          dnf_lib.Clause(5, 6, True, True),
          # Not used since we only need 5 unique clauses.
          dnf_lib.Clause(7, 8, True, True),
          dnf_lib.Clause(7, 8, False, True),
      ]
      dnf = dnf_lib.get_random_dnf(num_literals=10, num_clauses=5)
    self.assertEqual(dnf.num_literals, 10)
    self.assertSetEqual(
        dnf.clauses,
        set([dnf_lib.Clause(0, 1, True, True),
             dnf_lib.Clause(0, 1, True, False),
             dnf_lib.Clause(1, 2, True, True),
             dnf_lib.Clause(3, 4, True, True),
             dnf_lib.Clause(5, 6, True, True)]))

  def test_get_random_dnf_num_clauses_too_large(self):
    with self.assertRaisesRegex(
        ValueError,
        'num_clauses 10 can not be greater than number of possible clauses 4'):
      dnf_lib.get_random_dnf(num_literals=2, num_clauses=10)

  @parameterized.parameters(
      ('(x_32 || x_0)', dnf_lib.Clause(0, 32, False, False)),
      ('(x_2 || !x_39)', dnf_lib.Clause(2, 39, False, True)),
      ('(!x_11 || x_12)', dnf_lib.Clause(11, 12, True, False)),
      ('(!x_3 || !x_2)', dnf_lib.Clause(2, 3, True, True))
  )
  def test_clause_from_string(self, clause_string, expected_clause):
    self.assertEqual(dnf_lib.clause_from_string(clause_string), expected_clause)

  @parameterized.parameters(
      ('as;lkj;lkajsdf'),
      ('x_2'),
      ('x_-5'),
      ('(x_0 | x_1)'),
      ('x_0 || x_1)'),
      ('x_0 || x_1'),
      ('(!!x_3 || x_5)'),
      ('(!x_3 || x_4 || x_5)')
  )
  def test_clause_from_string_invalid(self, clause_string):
    with self.assertRaisesRegex(
        ValueError,
        'Not the output of a clause string'
    ):
      dnf_lib.clause_from_string(clause_string)

  @parameterized.parameters(
      (
          'Number of Literals: 12, DNF:',
          dnf_lib.DNF(12, [])
      ),
      (
          'Number of Literals: 4, DNF: ',
          dnf_lib.DNF(4, [])
      ),
      (
          'Number of Literals: 24, DNF: (!x_0 || x_1)',
          dnf_lib.DNF(24, [dnf_lib.Clause(0, 1, True, False)])
      ),
      (
          'Number of Literals: 5, DNF:(x_2 || x_4) && (x_3 || !x_1)',
          dnf_lib.DNF(5, [dnf_lib.Clause(2, 4, False, False),
                          dnf_lib.Clause(1, 3, True, False)])
      )
  )
  def test_dnf_from_string(self, dnf_string, expected_dnf):
    dnf = dnf_lib.dnf_from_string(dnf_string)
    self.assertEqual(dnf.num_literals, expected_dnf.num_literals)
    self.assertEqual(dnf.clauses, expected_dnf.clauses)

  @parameterized.parameters(
      ('Number of Literals: , DNF:'),
      ('Number of Literals: 5, DNF: (x_0 || x_1) && ')
  )
  def test_dnf_from_string_invalid(self, dnf_string):
    with self.assertRaisesRegex(
        ValueError,
        'Not the output of a dnf string'
    ):
      dnf_lib.dnf_from_string(dnf_string)


if __name__ == '__main__':
  absltest.main()
