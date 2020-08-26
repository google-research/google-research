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

"""Class containing a 2-DNF.

A k-DNF is a disjunction (logical AND) over conjunctions (logical OR) with each
conjunction clause containing at most k literals. We will restrict ourselves to
2-DNFs.

This library contains the building blocks for producing problem instances,
as well as evaluating problems related to the DNF.
"""

import itertools
import random
import re

_CLAUSE_REGEX = r'\(\!?x_\d+ \|\| \!?x_\d+\)'


def get_number_of_possible_clauses(num_literals):
  """Gets number of possible clauses for num_literals.

  We select two literals from num_literals. Each clause has 4 pairs of
  negativities (True, True), (False, False), (True, False), (False, True).

  Args:
    num_literals: Positive integer, the number of literals in the DNF.

  Returns:
    Integer.
  """
  return 4 * num_literals * (num_literals - 1) / 2


def get_random_clause(num_literals):
  """Returns a clause uniformly at random.

  Generates two distince indices from num_literals uniformly at random. Also
  generates is_negative1 and is_negative2 uniformly at random.

  Args:
    num_literals: Positive integer, the number of literals in the DNF.

  Returns:
    Clause instance generated uniformly at random.
  """
  indices = random.sample(range(num_literals), 2)
  is_negative1 = bool(random.randrange(2))
  is_negative2 = bool(random.randrange(2))
  return Clause(indices[0], indices[1], is_negative1, is_negative2)


def get_random_dnf(num_literals, num_clauses):
  """Outputs a random DNF.

  Create random DNF from num_clauses unique random clauses.

  Args:
    num_literals: Positive integer, the number of literals in our DNF. Must be
                  greater than or equal to 2.
    num_clauses: Positive integer, the number of random clauses that will be
                 added to the set of all clauses. Not guaranteed that the
                 size of this final set is equal to num_clauses. If
                 non-positive, will return an empty DNF.

  Returns:
    A DNF with uniformly random clauses defined by num_literals and num_clauses.

  Raises:
    ValueError: If num_clauses is greater than number of possible clauses.
  """
  number_of_possible_clauses = get_number_of_possible_clauses(num_literals)
  if num_clauses > number_of_possible_clauses:
    raise ValueError(
        'num_clauses %d can not be greater than number of possible clauses %d.'
        % (num_clauses, number_of_possible_clauses))
  clauses = set()
  while len(clauses) < num_clauses:
    new_clause = get_random_clause(num_literals)
    clauses.add(new_clause)
  return DNF(num_literals, clauses)


def clause_from_string(clause_string):
  """Converts a serialized clause string into a Clause.

  Args:
    clause_string: The human-readable string to be converted into a Clause.
                   An example would be: (!x_0 || x_5)

  Returns:
    A Clause object that represents the string

  Raises:
    ValueError: If clause_string is not in the proper format
  """
  if re.match(r'%s\Z' % _CLAUSE_REGEX, clause_string) is None:
    raise ValueError('Not the output of a clause string')
  # Remove parentheses.
  clause_string = clause_string[1:-1]
  literal_string1, literal_string2 = clause_string.split(' || ')
  is_negative1 = literal_string1[0] == '!'
  is_negative2 = literal_string2[0] == '!'
  index1 = int(re.search(r'\d+', literal_string1).group(0))  # type: ignore
  index2 = int(re.search(r'\d+', literal_string2).group(0))  # type: ignore
  return Clause(index1, index2, is_negative1, is_negative2)


def dnf_from_string(dnf_string):
  """Converts a serialized dnf string into a Dnf.

  Args:
    dnf_string: The human-readable string to be converted into a DNF.

  Returns:
    A DNF object that represents the string

  Raises:
    ValueError: If dnf_string is not in the proper format
  """
  expected_regex = (r'Number of Literals: \d+, '
                    r'DNF: ?((%s && )*%s)?' % ((_CLAUSE_REGEX,) * 2))
  if re.match(r'%s\Z' % expected_regex, dnf_string) is None:
    raise ValueError('Not the output of a dnf string')
  num_literals = int(re.match(r'Number of Literals: \d+',
                              dnf_string).group(0)[20:])  # type: ignore
  clauses = [clause_from_string(clause_match.group(0))
             for clause_match in re.finditer(_CLAUSE_REGEX, dnf_string)]
  dnf = DNF(num_literals, clauses)
  return dnf


class Clause(object):
  """Singular clause in a DNF.

  Each clause is represented by two indices, which represent the literals. Each
  literal is accompanied by a boolean value as to whether the literal is
  negative.

  Attributes:
    index1: Integer, index of the first variable in the clause.
    index2: Integer, index of the second variable in the clause.
    is_negative1: Boolean, true if literal at index1 is negative.
    is_negative2: Boolean, true if literal at index2 is negative.
  """

  def __init__(self, index1, index2, is_negative1, is_negative2):
    """Initializer.

    This class stores the smaller index first as an invariant.

    Args:
      index1: Non-negative Integer, contains index of first literal.
      index2: Non-negative Integer, contains index of second literal. Must be
          different from index1.
      is_negative1: Boolean, true if literal at index1 is negative.
      is_negative2: Boolean, true if literal at index2 is negative.

    Raises:
      ValueError: If indices are negative or are equal to one another.
    """
    if index1 < 0:
      raise ValueError('index1 must be a non-negative integer, not %d.'
                       % index1)
    if index2 < 0:
      raise ValueError('index2 must be a non-negative integer, not %d.'
                       % index2)
    if index1 == index2:
      raise ValueError('index1 and index2 must be different')

    if index1 < index2:
      self.index1 = index1
      self.index2 = index2
      self.is_negative1 = is_negative1
      self.is_negative2 = is_negative2
    else:
      # Swap the indices to make the smaller one come first
      self.index1 = index2
      self.index2 = index1
      self.is_negative1 = is_negative2
      self.is_negative2 = is_negative1

  def __str__(self):
    """Returns a string representation of the Clause.

    '!' is used to represent logical 'NOT'.
    '||' is used to represent logical 'OR'.
    '&&' is used to represent logical 'AND'.
    'x_5' refers to the 5th literal in the input.

    Returns:
      String that represents a single clause in human-readable form.
    """
    negation1 = '!' if self.is_negative1 else ''
    negation2 = '!' if self.is_negative2 else ''
    return '(%sx_%d || %sx_%d)' % (negation1,
                                   self.index1,
                                   negation2,
                                   self.index2)

  def __eq__(self, other):
    """Checks whether two clauses are equivalent or not.

    Makes sure indices match, and that the negations match. Don't have to check
    order because of our invariant in how we saved indices.

    Args:
      other: Clause, the clause to compare equality to.

    Returns:
      A boolean value that is True if and only if the indices and negations are
      the same.
    """
    if isinstance(other, Clause):
      return (self.index1 == other.index1 and self.index2 == other.index2 and
              self.is_negative1 == other.is_negative1 and
              self.is_negative2 == other.is_negative2)
    else:
      return False

  def __hash__(self):
    """Hashes the clause to an integer.

    Turns the clause into a bunch of tuples, that are then hashed using Python's
    built-in hash function.

    Returns:
      Integer that acts as the hashed value of the clause.
    """
    return ((self.index1, self.is_negative1),
            (self.index2, self.is_negative2)).__hash__()

  def eval(self, literals):
    """Evalutes the clause on the list of literals.

    Args:
      literals: Boolean list, index i corresponds to the truth assignment of
          literal i. Number of assigned literals must be greater than both
          self.index1 and self.index2.

    Returns:
      A Boolean mapping to the evaluation of the clause on the given assignment
      of literals.

    Raises:
      IndexError: An error occurring if the list of literals is not large
                  enough.
    """
    def xor(value1, value2):
      """Evaluates the boolean exclusive-or of two boolean values.

      Opposed to doing bitwise evaluation using '^'.

      Args:
        value1: Boolean.
        value2: Boolean.

      Returns:
        The boolean exclusive-or of value1 and value2.
      """
      return bool(value1) != bool(value2)

    return (xor(literals[self.index1], self.is_negative1) or
            xor(literals[self.index2], self.is_negative2))


class DNF(object):
  """Represents 2-DNF as a set of clauses.

  Attributes:
    num_literals: Integer, the number of literals in the 2-DNF.
    clauses: Clause set, the set of clauses in the 2-DNF.
    optimal_num_satisfied: Positive integer, the maximum number of satisfiable
        clauses by any assignment of literals.
  """

  def __init__(self, num_literals, clauses):
    """Initializer.

    Args:
      num_literals: Positive Integer, the number of literals in our DNF. Must be
          greater than or equal to 2.
      clauses: Iterable container of Clause, (abstract) set of clauses to copy
          over.

    Raises:
      ValueError: If number of literals is negative.
    """
    if num_literals < 2:
      raise ValueError('num_literals must be at least 2, not %d.'
                       % num_literals)

    self.num_literals = num_literals

    self.clauses = set(clauses)

    self.optimal_num_satisfied = self._get_optimal_num_satisfied()

  def _get_optimal_num_satisfied(self):
    """Gets the maximum number of satisfiable clauses.

    Brute force search through all possible literal assignments and takes
    the max. Complexity is O(2 ** num_literals).

    Returns:
      Integer, the maximum number of satisfied clauses in the DNF.
    """
    return max([
        self.get_num_clauses_satisfied(assignment)
        for assignment in itertools.product([0, 1], repeat=self.num_literals)])

  def __str__(self):
    """Returns a string representation of the DNF.

    '!' is used to represent logical 'NOT'.
    '||' is used to represent logical 'OR'.
    '&&' is used to represent logical 'AND'.
    'x_5' refers to the 5th literal in the input.

    Returns:
      String that displays the number of literals, then the clauses of the DNF
      in human-readable form.
    """
    clause_string = ' && '.join(str(clause) for clause in self.clauses)
    return 'Number of Literals: %d, DNF: %s' % (self.num_literals,
                                                clause_string)

  def _evaluate_clauses(self, literals):
    """Evaluates clauses by the assignment of literals.

    Args:
      literals: Boolean list, index i corresponds to the truth assignment of
          literal i. Number of assigned literals must be equal to
          self.num_literals.

    Returns:
      List of boolean. Note the results in this list is unordered since
      self.clauses is a Set.

    Raises:
      ValueError: If the number of assigned literals does not match
          self.num_literals.
    """
    if len(literals) != self.num_literals:
      raise ValueError('Number of literals should be %d, not %d'
                       % (self.num_literals, len(literals)))
    return [clause.eval(literals) for clause in self.clauses]

  def eval(self, literals):
    """Evaluates the dnf on the assignment of literals given.

    Args:
      literals: Boolean list, index i corresponds to the truth assignment of
          literal i. Number of assigned literals must be equal to
          self.num_literals.

    Returns:
      The logical AND over all of the clauses in the DNF.
    """
    return all(self._evaluate_clauses(literals))

  def get_num_clauses_satisfied(self, literals):
    """Counts the number of clauses satisfied by the assignment of literals.

    Args:
      literals: Boolean list, index i corresponds to the truth assignment of
          literal i. Number of assigned literals must be equal to
          self.num_literals.

    Returns:
      An integer giving the number of satisfied clauses in the DNF.
    """
    return sum(self._evaluate_clauses(literals))

