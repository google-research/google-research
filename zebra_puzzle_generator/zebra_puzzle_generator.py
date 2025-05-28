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

"""Main file for generating random zebra puzzles."""
import random
from typing import Dict, Optional, List, Tuple

from zebra_puzzle_generator import zebra_solver
from zebra_puzzle_generator import zebra_utils

Clue = zebra_utils.Clue
SymbolicZebraPuzzle = zebra_utils.SymbolicZebraPuzzle
ZebraPuzzleSolver = zebra_solver.ZebraPuzzleSolver
ZebraSolverStep = zebra_utils.ZebraSolverStep
SymbolicZebraGroundTruth = zebra_utils.SymbolicZebraGroundTruth
SymbolicToNaturalLanguageMapper = zebra_utils.SymbolicToNaturalLanguageMapper
get_attr_num = zebra_utils.get_attr_num
convert_to_readable_entity = zebra_utils.convert_to_readable_entity
check_redundant = zebra_utils.check_redundant


def find_entity_in_answer_table(
    answer_table: List[List[int | None]],
    attr: int,
    attr_value: int) -> int:
  n = len(answer_table[0])
  index = 0
  for i in range(n):
    if answer_table[attr][i] == attr_value:
      index = i
  return index


def fraction_answer_table_solved(answer_table: List[List[int | None]]) -> float:
  m = len(answer_table)
  n = len(answer_table[0])
  none_count = 0
  for attr in range(1, m):
    for i in range(n):
      if not answer_table[attr][i]:
        none_count += 1
  return 1 - none_count / ((m-1) * n)


def find_clues_used(solution: List[List[ZebraSolverStep]]) -> List[int]:
  used_clue_numbers = set()
  for step_block in solution:
    for step in step_block:
      for clue in step.clue_list:
        used_clue_numbers.add(clue.number)
  return list(used_clue_numbers)


class RandomZebraPuzzleGenerator:
  """Class for generating random zebra puzzles.

  Attributes:
    sampled_clue_dict: A dictionary containing the clues generated so far.
    sampled_attr_dict:
    clue_type_weights:
    n: Number of entities.
    m1: Number of categorical attributes.
    m2: Number of numerical attributes. Currently only support 1 numerical
      attribute per puzzle.
  """
  sampled_clue_dict: Dict[str, List[Clue]] = {}
  sampled_attr_dict: Dict[int, int] = {}
  clue_type_weights: Dict[str, int] = {
      '=': 4, '!=': 1, 'nbr': 2, 'ends': 2,
      'immediate-left': 2, 'left-of': 2, 'inbetween': 1}
  n: int = 0
  m1: int = 0
  m2: int = 0

  def __init__(self, n: int, m1: int, m2: int = 1,
               clue_type_weights: Optional[Dict[str, int]] = None):
    self.n = n
    self.m1 = m1
    self.m2 = m2
    if clue_type_weights:
      self.clue_type_weights = clue_type_weights

  def check_duplicate(
      self,
      clue_type: str,
      lhs_list: List[Tuple[str, int, int]],
      rhs_list: List[Tuple[str, int, int]]) -> bool:
    """Check if a clue is a semantic duplicate of another clue in the clue_dict.

    Args:
      clue_type: String indicating the clue type.
      lhs_list: LHS list of entities.
      rhs_list: RHS list of entities.

    Returns:
      A boolean indicating whether the combination of clue_type, lhs_list and
      rhs_list already exists semantically in the clue_dict.
    """
    match clue_type:
      case 'inbetween':
        if clue_type in self.sampled_clue_dict:
          for clue in self.sampled_clue_dict[clue_type]:
            if (clue.lhs_list[0] == lhs_list[0]
                and clue.rhs_list[0] == rhs_list[0]
                and clue.rhs_list[1] == rhs_list[1]):
              return True
      case 'ends':
        if clue_type in self.sampled_clue_dict:
          for clue in self.sampled_clue_dict[clue_type]:
            if clue.lhs_list[0] == lhs_list[0]:
              return True
      case 'nbr' | '=' | '!=':
        if clue_type in self.sampled_clue_dict:
          for clue in self.sampled_clue_dict[clue_type]:
            if (clue.lhs_list[0] == lhs_list[0]
                and clue.rhs_list[0] == rhs_list[0]):
              return True
            if (clue.lhs_list[0] == rhs_list[0]
                and clue.rhs_list[0] == lhs_list[0]):
              return True
      case 'immediate-left' | 'left-of':
        if clue_type in self.sampled_clue_dict:
          for clue in self.sampled_clue_dict[clue_type]:
            if (clue.lhs_list[0] == lhs_list[0]
                and clue.rhs_list[0] == rhs_list[0]):
              return True
    return False

  def add_to_clue_dict(self, clue: Clue):
    clue_type = clue.clue_type
    if clue_type in self.sampled_clue_dict:
      self.sampled_clue_dict[clue_type].append(clue)
    else:
      self.sampled_clue_dict[clue_type] = [clue]

  def add_to_attr_dict(self, clue: Clue):
    clue_type = clue.clue_type
    match clue_type:
      case 'inbetween':
        self.sampled_attr_dict[get_attr_num(clue.lhs_list[0], self.m2)] += 1
        self.sampled_attr_dict[get_attr_num(clue.rhs_list[0], self.m2)] += 1
        self.sampled_attr_dict[get_attr_num(clue.rhs_list[1], self.m2)] += 1
      case 'ends':
        self.sampled_attr_dict[get_attr_num(clue.lhs_list[0], self.m2)] += 1
      case 'nbr' | '=' | '!=' | 'immediate-left' | 'left-of':
        self.sampled_attr_dict[get_attr_num(clue.lhs_list[0], self.m2)] += 1
        self.sampled_attr_dict[get_attr_num(clue.rhs_list[0], self.m2)] += 1

  def sample_attr(
      self,
      incomplete_answer: Optional[List[List[int | None]]] = None) -> int:
    """Sample an attribute biasing towards less filled attributes.

    Args:
      incomplete_answer: The current answer that the solver has been able to
        produce.

    Returns:
      A sampled attribute index.
    """
    if incomplete_answer is None:
      eps = 3
      m = len(self.sampled_attr_dict)
      attrs = list(range(m))
      weights = [1/(eps + self.sampled_attr_dict[i]) for i in attrs]
      for i in range(self.m2):
        weights[i] /= 5  # keep the numerical attributes rare
    else:
      m = len(self.sampled_attr_dict)
      n = len(incomplete_answer[0])
      attrs = list(range(m))
      filled = []
      for j in attrs:
        s = 0
        for i in range(n):
          if incomplete_answer[j][i] is not None:
            s += 1
        filled.append(s)
      weights = [(n-f+1) for f in filled]
    return random.choices(attrs, weights=weights, k=1)[0]

  def sample_entity(
      self,
      complete_answer: List[List[int]],
      incomplete_answer: List[List[int | None]]) -> Tuple[int, int, int]:
    """Sample an entity biasing towards unsolved parts of the answer table.

    Args:
      complete_answer: The fully solved answer table.
      incomplete_answer: A partially solver answer table.

    Returns:
      An entity tuple indicating the attritube, value and the position of the
      entity.
    """
    m = self.m1 + self.m2
    n = len(incomplete_answer[0])
    attrs = list(range(m))
    filled = []
    # First sample an attribute biasing for less filled rows
    for j in attrs:
      s = 0
      for i in range(n):
        if incomplete_answer[j][i] is not None:
          s += 1
      filled.append(s)
    weights = [(n - f + 1) for f in filled]
    attr = random.choices(attrs, weights=weights, k=1)[0]
    # Sample index now biasing towards unfilled positions
    indices = list(range(n))
    weights = [
        5 if incomplete_answer[attr][i] is not None else 1 for i in indices
    ]
    index = random.choices(indices, weights=weights, k=1)[0]
    value = complete_answer[attr][index]
    return attr, index, value

  def generate_symbolic_zebra_puzzle(self, verbose: bool = False) -> Tuple[
      SymbolicZebraPuzzle,
      SymbolicZebraGroundTruth,
      bool,
      List[List[ZebraSolverStep | None]],
      List[Tuple[Tuple[str, int, int], int]],
  ]:
    """Generate a symbolic zebra puzzle.

    Arguments:
      verbose: Whether to be be verbose during generation process.

    Returns:
      A generated symbolic zebra puzzle together with ground truth and list of
      clues and the solution.
    """
    # First sample a ground truth answer table.
    assert self.m2 == 1  # only support one numerical attribute currently
    answer_table = []
    for i in range(self.m1+1):
      if i == 0:
        answer_table.append(list(range(self.n)))
      else:
        vals = list(range(self.n))
        random.shuffle(vals)
        answer_table.append(vals)

    ground_truth = SymbolicZebraGroundTruth(
        n=self.n, m1=self.m1, m2=self.m2, answer_table=answer_table)

    # Until puzzle becomes solvable, generate and add clues to clue set.
    # Clue generation is biased so as to sample currently
    # unsolved entities more frequently. This helps cut down redundant clues
    # and drive the generator towards a solvable puzzle faster.
    clues = []
    self.sampled_clue_dict = {}
    self.sampled_attr_dict = {attr: 0 for attr in range(self.m1+1)}
    puzzle = SymbolicZebraPuzzle(n=self.n, m1=self.m1, m2=1, clues=clues)
    solved = False
    answer_attempt = None
    solution = None
    clue_types = list(self.clue_type_weights.keys())
    clue_weights = [self.clue_type_weights[clue_type]
                    for clue_type in clue_types]
    if self.n < 3:
      # Don't generate inbetween clues when less than 3 entities.
      self.clue_type_weights['inbetween'] = 0
    while not solved:
      # Sample a new clue
      clue_type = random.choices(clue_types, weights=clue_weights, k=1)[0]
      lhs_list = []
      rhs_list = []
      match clue_type:
        case '=':
          # Sample lhs attr and lhs attr value. Then sample rhs attribute.
          # rhs attr value automatically fixed by ground truth solution.
          if answer_attempt:
            lhs_attr, index, lhs_attr_value = self.sample_entity(answer_table,
                                                                 answer_attempt)
          else:
            lhs_attr = self.sample_attr()
            lhs_attr_value = random.randint(0, self.n-1)
            # Find position of lhs entity in answer_table
            index = find_entity_in_answer_table(answer_table, lhs_attr,
                                                lhs_attr_value)
          rhs_attr = self.sample_attr()
          while rhs_attr == lhs_attr:
            # we don't want a redundant clue, resample attribute.
            rhs_attr = self.sample_attr()
          rhs_attr_value = answer_table[rhs_attr][index]
          lhs_list = [convert_to_readable_entity(lhs_attr, lhs_attr_value)]
          rhs_list = [convert_to_readable_entity(rhs_attr, rhs_attr_value)]
        case '!=':
          # Sample lhs attr and lhs attr value.
          # Sample rhs attribute and rhs attr value.
          if answer_attempt:
            lhs_attr, index, lhs_attr_value = self.sample_entity(answer_table,
                                                                 answer_attempt)
          else:
            lhs_attr = self.sample_attr()
            lhs_attr_value = random.randint(0, self.n-1)
            # Find position of lhs entity in answer_table
            index = find_entity_in_answer_table(answer_table, lhs_attr,
                                                lhs_attr_value)
          rhs_attr = self.sample_attr()
          while rhs_attr == lhs_attr:
            # we don't want a redundant clue
            rhs_attr = self.sample_attr()
          rhs_attr_value = random.randint(0, self.n-1)
          while rhs_attr_value == answer_table[rhs_attr][index]:
            rhs_attr_value = random.randint(0, self.n-1)
          lhs_list = [convert_to_readable_entity(lhs_attr, lhs_attr_value)]
          rhs_list = [convert_to_readable_entity(rhs_attr, rhs_attr_value)]
        case 'nbr':
          # Sample lhs attr and lhs attr value.
          # Sample rhs attribute and rhs attr value.
          if answer_attempt:
            lhs_attr, index, lhs_attr_value = self.sample_entity(answer_table,
                                                                 answer_attempt)
          else:
            lhs_attr = self.sample_attr()
            lhs_attr_value = random.randint(0, self.n-1)
            # Find position of lhs entity in answer_table
            index = find_entity_in_answer_table(answer_table, lhs_attr,
                                                lhs_attr_value)
          rhs_attr = self.sample_attr()
          while lhs_attr == 0 and rhs_attr == 0:
            # redundant clue. resample rhs_attr
            rhs_attr = self.sample_attr()
          nbr_vals = []
          if index > 0:
            # has a left neighbor
            nbr_vals.append(answer_table[rhs_attr][index-1])
          if index < self.n-1:
            # has a right neighbor
            nbr_vals.append(answer_table[rhs_attr][index+1])
          rhs_attr_value = random.choice(nbr_vals)
          lhs_list = [convert_to_readable_entity(lhs_attr, lhs_attr_value)]
          rhs_list = [convert_to_readable_entity(rhs_attr, rhs_attr_value)]
        case 'ends':
          # Sample attr. Then sample from one of two entities at the ends.
          lhs_attr = self.sample_attr()
          while lhs_attr == 0:
            lhs_attr = self.sample_attr()
          end_vals = [answer_table[lhs_attr][0], answer_table[lhs_attr][-1]]
          lhs_val = random.choice(end_vals)
          lhs_list = [convert_to_readable_entity(lhs_attr, lhs_val)]
          rhs_list = []
        case 'immediate-left':
          # Sample lhs attr and lhs attr value. Sample rhs attribute.
          # rhs attr value automatically fixed by solution.
          if answer_attempt:
            lhs_attr, index, lhs_attr_value = self.sample_entity(answer_table,
                                                                 answer_attempt)
          else:
            lhs_attr = self.sample_attr()
            lhs_attr_value = random.randint(0, self.n-1)
            # Find position of lhs entity in answer_table
            index = find_entity_in_answer_table(answer_table, lhs_attr,
                                                lhs_attr_value)
          while index == self.n-1:
            # resample lhs
            if answer_attempt:
              lhs_attr, index, lhs_attr_value = self.sample_entity(
                  answer_table, answer_attempt)
            else:
              lhs_attr = self.sample_attr()
              lhs_attr_value = random.randint(0, self.n-1)
              index = find_entity_in_answer_table(answer_table, lhs_attr,
                                                  lhs_attr_value)
          rhs_attr = self.sample_attr()
          while lhs_attr == 0 and rhs_attr == 0:
            # redundant clue. resample rhs_attr
            rhs_attr = self.sample_attr()
          rhs_attr_value = answer_table[rhs_attr][index+1]
          lhs_list = [convert_to_readable_entity(lhs_attr, lhs_attr_value)]
          rhs_list = [convert_to_readable_entity(rhs_attr, rhs_attr_value)]
        case 'left-of':
          if answer_attempt:
            lhs_attr, index, lhs_attr_value = self.sample_entity(answer_table,
                                                                 answer_attempt)
          else:
            lhs_attr = self.sample_attr()
            lhs_attr_value = random.randint(0, self.n-1)
            # Find position of lhs entity in answer_table
            index = find_entity_in_answer_table(answer_table, lhs_attr,
                                                lhs_attr_value)
          while index == self.n-1:
            # resample lhs
            if answer_attempt:
              lhs_attr, index, lhs_attr_value = self.sample_entity(
                  answer_table, answer_attempt)
            else:
              lhs_attr = self.sample_attr()
              lhs_attr_value = random.randint(0, self.n-1)
              index = find_entity_in_answer_table(answer_table, lhs_attr,
                                                  lhs_attr_value)
          rhs_attr = random.randint(0, self.m1)
          while lhs_attr == 0 and rhs_attr == 0:
            # redundant clue. resample rhs_attr
            rhs_attr = self.sample_attr()
          rhs_vals = answer_table[rhs_attr][index+1:]
          rhs_attr_value = random.choice(rhs_vals)
          lhs_list = [convert_to_readable_entity(lhs_attr, lhs_attr_value)]
          rhs_list = [convert_to_readable_entity(rhs_attr, rhs_attr_value)]
        case 'inbetween':
          # first sample lhs
          lhs_attr = self.sample_attr()
          lhs_attr_value = random.randint(0, self.n-1)
          # Find position of lhs entity in answer_table
          lhs_index = find_entity_in_answer_table(answer_table, lhs_attr,
                                                  lhs_attr_value)
          while lhs_index > self.n-3:
            # resample lhs
            lhs_attr = self.sample_attr()
            lhs_attr_value = random.randint(0, self.n-1)
            lhs_index = find_entity_in_answer_table(answer_table, lhs_attr,
                                                    lhs_attr_value)
          # next sample rhs2
          rhs2_attr = self.sample_attr()
          while rhs2_attr == 0:
            rhs2_attr = self.sample_attr()
          rhs2_attr_value = random.randint(0, self.n-1)
          rhs2_index = find_entity_in_answer_table(answer_table, rhs2_attr,
                                                   rhs2_attr_value)
          while rhs2_index < lhs_index + 2:
            # resample rhs2
            rhs2_attr_value = random.randint(0, self.n-1)
            rhs2_index = find_entity_in_answer_table(answer_table, rhs2_attr,
                                                     rhs2_attr_value)
          # finally sample rhs1
          rhs1_attr = self.sample_attr()
          while rhs1_attr == 0:
            rhs1_attr = self.sample_attr()
          rhs1_vals = answer_table[rhs1_attr][lhs_index+1:rhs2_index]
          rhs1_attr_value = random.choice(rhs1_vals)
          lhs_list = [convert_to_readable_entity(lhs_attr, lhs_attr_value)]
          rhs_list = [convert_to_readable_entity(rhs1_attr, rhs1_attr_value),
                      convert_to_readable_entity(rhs2_attr, rhs2_attr_value)]

      if self.check_duplicate(clue_type, lhs_list, rhs_list):
        continue
      if check_redundant(clue_type, lhs_list, rhs_list, answer_attempt):
        continue
      clue = Clue(number=len(clues)+1, clue_type=clue_type,
                  lhs_list=lhs_list, rhs_list=rhs_list)
      clues.append(clue)
      self.add_to_clue_dict(clue)
      self.add_to_attr_dict(clue)
      # Check if puzzle is solvable.
      # Running the solver is expensive. Only run after we have accrued at least
      # 5 clues.
      if len(clues) > 4:
        hard_deduce = False
        solver = ZebraPuzzleSolver(puzzle)
        solved, _, answer_attempt, _, solution = solver.solve(
            verbose=False, hard_deduce=hard_deduce
        )
        if verbose:
          print(
              'Percent of puzzle solved:',
              fraction_answer_table_solved(answer_attempt) * 100,
          )
      if verbose:
        print('Current clue list has', len(clues), 'clues.')

    # By now the puzzle is solvable
    # First filter clues unused by the solver
    used_clue_numbers = find_clues_used(solution)
    used_clues = []
    for clue in clues:
      if clue.number in used_clue_numbers:
        used_clues.append(clue)
    # fix the numbering in the filtered clues
    for i, clue in enumerate(used_clues):
      clue.number = i+1
    puzzle.clues = used_clues
    if verbose:
      print('Number of clues after filtering unused ones', len(puzzle.clues))

    # Make the clues a bit more diverse by exploiting symmetries.
    # For e.g. a R b can be written as b R a some fraction of the time
    # for R \in {=, !=, nbr}
    for clue in used_clues:
      if clue.clue_type in ['=', '!=', 'nbr']:
        flip = random.uniform(0.0, 1.0)
        if flip > 0.75:
          lhs = clue.lhs_list[0]
          rhs = clue.rhs_list[0]
          clue.lhs_list[0] = rhs
          clue.rhs_list[0] = lhs

    # Re-solve the puzzle
    hard_deduce = False
    final_solver = ZebraPuzzleSolver(puzzle)
    _, used_hard_deduce, _, _, solution = final_solver.solve(
        verbose=False, hard_deduce=hard_deduce
    )
    fills = []
    for reasoning_block in solution:
      fills.extend(zebra_utils.new_fills_in_reasoning_block(reasoning_block))
    return puzzle, ground_truth, used_hard_deduce, solution, fills
