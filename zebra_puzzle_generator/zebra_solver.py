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

"""Deductive solver for Zebra puzzles."""

import copy
import itertools
from typing import List, Tuple

from zebra_puzzle_generator import zebra_utils


class ZebraPuzzleSolver:
  """Zebra puzzle solver."""
  symbolic_puzzle: zebra_utils.SymbolicZebraPuzzle
  solution: List[List[zebra_utils.ZebraSolverStep]] = None
  answer: List[List[int | None]] = None

  def __init__(self,
               symbolic_puzzle: zebra_utils.SymbolicZebraPuzzle):
    self.symbolic_puzzle = symbolic_puzzle
    self.n = symbolic_puzzle.n
    self.m1 = symbolic_puzzle.m1
    self.m2 = symbolic_puzzle.m2
    self.m = self.m1 + self.m2
    # Initialize the answer table
    answer = []
    for j in range(self.m1 + self.m2):
      answer.append([])
      for i in range(self.n):
        if j == 0:
          answer[j].append(i)
        else:
          answer[j].append(None)
    self.answer = answer

  def check_invalid_state(
      self,
      possible_answers: List[List[List[int]]]) -> bool:
    """Checks whether the current state of possible_answers and answer_table are invalid.

    Args:
      possible_answers: A data structure containing a list of possible answers
        for each position in the answer_table.

    Returns:
      A boolean indicating whether the current state of possible_answers and
      answer_table are invalid or not.
    """
    # 1. Answers in every row must be unique
    # 2. Possible answers must be a list.
    for j in range(self.m):
      unique_answers = set()
      for i in range(self.n):
        if len(possible_answers[j][i]) > self.n:
          return True
        if self.answer[j][i] not in unique_answers:
          unique_answers.add(self.answer[j][i])
        elif (self.answer[j][i] in unique_answers
              and self.answer[j][i] is not None):
          return True
    return False

  def is_solved(self) -> bool:
    """Checks whether the answer table is solved."""
    for j in range(self.m):
      for i in range(self.n):
        if self.answer[j][i] is None:
          return False
    return True

  def print_possible_answers(self, possible_answers: List[List[List[int]]]):
    print('Possible Answers')
    for j in range(self.m):
      print(j, ': ', end=' ')
      for i in range(self.n):
        print(possible_answers[j][i], end='  ')
    print('\n')

  def print_answer_table(self):
    print('Answer Table')
    for j in range(self.m):
      print(j, ': ', end=' ')
      for i in range(self.n):
        print(self.answer[j][i], end='  ')
    print('\n')

  def filter_possible_answers(
      self,
      possible_answers: List[List[List[int]]]
      ) -> Tuple[bool, List[List[List[int]]]]:
    """Remove those options as possible_answers which already appear as answer in the same row.

    Args:
      possible_answers: A table of possible answers for each cell.

    Returns:
      progress: A boolean indicating whether any changes were made by
        this function call.
      possible_answers: The modified possible_answers table.
    """
    progress = False
    for j in range(1, self.m):
      for i in range(self.n):
        if self.answer[j][i] is not None:
          for ii in range(self.n):
            if ii != i and (self.answer[j][i] in possible_answers[j][ii]):
              possible_answers[j][ii].remove(self.answer[j][i])
              progress = True
            if ii == i and len(possible_answers[j][i]) > 1:
              possible_answers[j][i] = [self.answer[j][i]]
              progress = True
    return progress, possible_answers

  def get_grounded_entity_position(self, entity: Tuple[str, int, int]) -> int:
    entity_attr = zebra_utils.get_attr_num(entity, self.m2)
    index = -1
    for i in range(self.n):
      if self.answer[entity_attr][i] == entity[2]:
        index = i
        break
    return index

  def get_possible_locations(
      self,
      possible_answers: List[List[List[int]]],
      entity: Tuple[str, int, int]) -> List[int]:
    entity_attr = zebra_utils.get_attr_num(entity, self.m2)
    locations = []
    for i in range(self.n):
      if entity[2] in possible_answers[entity_attr][i]:
        locations.append(i)
    return locations

  def fill_by_elimination(
      self,
      possible_answers: List[List[List[int]]],
      allow_multiple_fills: bool = False,
  ) -> Tuple[bool, zebra_utils.ZebraSolverStep | None]:
    """Fill the answer table by elimination using possible_answers.

    Args:
      possible_answers:
      allow_multiple_fills: If true, allows for multiple cells to be filled with
        a single function call. Otherwise we return after the first cell has
        been filled.

    Returns:
      progress: A boolean indicating whether progress has been made in filling
        the answer table.
      reasoning_step: An instance of ZebraSolverStep which contains the info
        for the reasoning undertaken by this function call.
    """
    progress = False
    new_fills1 = []
    new_fills2 = []
    for j in range(1, self.m):
      # 1. Check if any cell has only one option for possible_answers
      for i in range(self.n):
        if len(possible_answers[j][i]) == 1 and self.answer[j][i] is None:
          self.answer[j][i] = possible_answers[j][i][0]
          progress = True
          new_fills1.append((j, i, possible_answers[j][i][0]))
          _, possible_answers = self.filter_possible_answers(possible_answers)
          if not allow_multiple_fills:
            break
      if progress and not allow_multiple_fills:
        break
      # 2. Check if any value is present only in one location for
      # its possible_answers row
      for i in range(self.n):
        count, index = 0, 0
        for ii in range(self.n):
          if i in possible_answers[j][ii]:
            count += 1
            index = ii
        if count == 1 and self.answer[j][index] is None:
          self.answer[j][index] = i
          progress = True
          new_fills2.append((j, index, i))
          _, possible_answers = self.filter_possible_answers(possible_answers)
          if not allow_multiple_fills:
            break
      if progress and not allow_multiple_fills:
        break
    reasoning_step = None
    if progress:
      reasoning_step = zebra_utils.ZebraSolverStep(
          clue_list=[], reason='fill-by-elimination',
          auxiliary_info=[new_fills1, new_fills2, allow_multiple_fills],
          current_answer_table=copy.deepcopy(self.answer))
    return progress, reasoning_step

  def check_valid_assignment(
      self,
      entities: List[Tuple[str, int, int]],
      assignment: Tuple[int, ...],
      clue_list: List[zebra_utils.Clue]) -> bool:
    """Check that a given assignment of entities to positions is valid.

    We check that the assignment is valid with respect to each of the clues
    in the clue_list.

    Args:
      entities: List of entity tuples.
      assignment: Dictionary which assigns entities to positions.
      clue_list: List of clues to check validity for.

    Returns:
      A boolean indicating whether the assignment is valid or not.
    """
    # 1. Check that there aren't two distinct entities occupying same spot
    for i, entity in enumerate(entities):
      for j in range(i + 1, len(entities)):
        if (
            zebra_utils.get_attr_num(entity, self.m2)
            == zebra_utils.get_attr_num(entities[j], self.m2)
            and entity[2] != entities[j][2]
        ):
          # entities i and j are distinct
          if assignment[i] == assignment[j]:
            return False

    # 2. Check that no clue in clue_list is violated
    def find_entity_index(entity):
      for i, e in enumerate(entities):
        if e == entity:
          return i

    for clue in clue_list:
      match clue.clue_type:
        case '=':
          lhs_index = find_entity_index(clue.lhs_list[0])
          rhs_index = find_entity_index(clue.rhs_list[0])
          if assignment[lhs_index] != assignment[rhs_index]:
            return False
        case '!=':
          lhs_index = find_entity_index(clue.lhs_list[0])
          rhs_index = find_entity_index(clue.rhs_list[0])
          if assignment[lhs_index] == assignment[rhs_index]:
            return False
        case 'nbr':
          lhs_index = find_entity_index(clue.lhs_list[0])
          rhs_index = find_entity_index(clue.rhs_list[0])
          if assignment[lhs_index] not in [assignment[rhs_index]-1,
                                           assignment[rhs_index]+1]:
            return False
        case 'ends':
          lhs_index = find_entity_index(clue.lhs_list[0])
          if assignment[lhs_index] not in [0, self.n-1]:
            return False
        case 'immediate-left':
          lhs_index = find_entity_index(clue.lhs_list[0])
          rhs_index = find_entity_index(clue.rhs_list[0])
          if assignment[lhs_index] != assignment[rhs_index] - 1:
            return False
        case 'left-of':
          lhs_index = find_entity_index(clue.lhs_list[0])
          rhs_index = find_entity_index(clue.rhs_list[0])
          if assignment[lhs_index] >= assignment[rhs_index]:
            return False
        case 'inbetween':
          lhs_index = find_entity_index(clue.lhs_list[0])
          rhs1_index = find_entity_index(clue.rhs_list[0])
          rhs2_index = find_entity_index(clue.rhs_list[1])
          if not (assignment[lhs_index] < assignment[rhs1_index]):
            return False
          if not (assignment[rhs1_index] < assignment[rhs2_index]):
            return False
    return True

  def hard_deduce_from_clue_list(
      self,
      p_ans: List[List[List[int]]],
      clue_list: List[zebra_utils.Clue]
  ) -> Tuple[List[List[List[int]]], bool, List[zebra_utils.ZebraSolverStep]]:
    """Hard deduction using a clue list.

    Hard deduction might involve exhaustive search over all possible assignments
    of values to certain entities until we find one which has only a single
    valid assignment. This can be different from how humans traditionally
    perform deductive reasoning.
    Args:
      p_ans: The possible answers table
      clue_list: The list of clues using which to perform hard deduction.

    Returns:
      p_ans: The modified possible answers table.
      successful_deduction: Boolean indicating whether a successful deduction
        was possible or not.
      reasoning_steps: A list of the reasoning steps formatted as instances of
        ZebraSolverSteps.
    """
    reasoning_steps = []
    # Construct a list of all entities being referred to in the clue list.
    # For each entity, construct a map to possible positions for the entity.
    entities_positions_map = {}
    for clue in clue_list:
      for t in clue.lhs_list + clue.rhs_list:
        entities_positions_map[t] = self.get_possible_locations(p_ans, t)

    # Find entities with only 2 or 3 possible values and try if we can
    # eliminate them to have only unique value.
    for i, key in enumerate(entities_positions_map):
      values = entities_positions_map[key]
      if len(values) > 3:
        continue
      # len(v) is 2 or 3
      emap_copy = copy.deepcopy(entities_positions_map)
      del emap_copy[key]
      num_possible_assignments = 1
      for _, v in emap_copy.items():
        num_possible_assignments *= len(v)
      if num_possible_assignments > 2000:
        continue
      for v in values:
        # see if there is a valid assignment where entity[i] gets value v
        valid_assignment = False
        for assignment in itertools.product(*emap_copy.values()):
          assignment_list = list(assignment)
          assignment_list.insert(i, v)
          assignment = tuple(assignment_list)
          # check if assignment is valid for clue_list
          if self.check_valid_assignment(
              list(entities_positions_map.keys()), assignment, clue_list):
            valid_assignment = True
            break
        if not valid_assignment:
          # no valid assignment possible with. remove value from possible values
          entities_positions_map[key].remove(v)

    # Check if any entity has only one possible value left
    successful_deduction = False
    for k, v in entities_positions_map.items():
      if len(v) == 1:
        if self.answer[zebra_utils.get_attr_num(k, self.m2)][v[0]] is None:
          successful_deduction = True
          self.answer[zebra_utils.get_attr_num(k, self.m2)][v[0]] = k[2]
          # aux info includes the entity k and its position v[0]
          reasoning_step = zebra_utils.ZebraSolverStep(
              clue_list=clue_list, reason='hard-deduce',
              auxiliary_info=[k, v[0]],
              current_answer_table=copy.deepcopy(self.answer))
          reasoning_steps.append(reasoning_step)
    _, p_ans = self.filter_possible_answers(p_ans)
    return p_ans, successful_deduction, reasoning_steps

  def deduce_from_clue_list(
      self,
      clue_list: List[zebra_utils.Clue],
      possible_answers: List[List[List[int]]],
      hard_deduce: bool = False,
      verbose: bool = False,
  ) -> Tuple[bool, List[List[List[int]]], List[zebra_utils.ZebraSolverStep]]:
    """Try to deduce any cell in the answer table using a given clue list.

    Args:
      clue_list: List of Clues.
      possible_answers: Possible answers table.
      hard_deduce: Whether to use hard_deduce.
      verbose: Whether to be verbose and print intermediate steps.

    Returns:
      successful_deduction: Whether successful deduction took place
      possible_answers: The modified possible answers table
      reasoning_steps: A list of reasoning steps represented as ZebraSolverStep
        instances.
    """
    successful_deduction = False
    reasoning_steps = []
    # Iteratively go about restricting possible answers using the
    # clues in clue_list
    p_ans = copy.deepcopy(possible_answers)
    for clue in clue_list:
      match clue.clue_type:
        case '=':
          lhs, rhs = clue.lhs_list[0], clue.rhs_list[0]
          lhs_attr = zebra_utils.get_attr_num(lhs, self.m2)
          rhs_attr = zebra_utils.get_attr_num(rhs, self.m2)
          # 1. If lhs entity or rhs entity has been grounded in the
          # answer table, then can enter the other into the answer table.
          if lhs[2] in self.answer[lhs_attr]:
            # lhs entity is grounded
            index = self.get_grounded_entity_position(lhs)
            if self.answer[rhs_attr][index] is None:
              self.answer[rhs_attr][index] = rhs[2]
              _, p_ans = self.filter_possible_answers(p_ans)
              successful_deduction = True
              # aux info specifies which of lhs or rhs is being deduced in
              # reasoning_step below, the index of grounding and attribute of
              # the grounding entity
              reasoning_step = zebra_utils.ZebraSolverStep(
                  clue_list=[clue], reason='=,grounded',
                  auxiliary_info=[1, index],
                  current_answer_table=copy.deepcopy(self.answer))
              reasoning_steps.append(reasoning_step)
              break
          if rhs[2] in self.answer[rhs_attr]:
            # rhs entity is grounded
            index = self.get_grounded_entity_position(rhs)
            if self.answer[lhs_attr][index] is None:
              self.answer[lhs_attr][index] = lhs[2]
              _, p_ans = self.filter_possible_answers(p_ans)
              successful_deduction = True
              # aux info specifies which of lhs or rhs is being deduced in
              # reasoning_step below and the index of grounding
              reasoning_step = zebra_utils.ZebraSolverStep(
                  clue_list=[clue], reason='=,grounded',
                  auxiliary_info=[0, index],
                  current_answer_table=copy.deepcopy(self.answer))
              reasoning_steps.append(reasoning_step)
              break
          # 2. If neither lhs nor rhs are grounded, can still try to restrict
          # possible answers in following manner. Scan both lhs_attr and
          # rhs_attr and for any entity having value different from lhs_val,
          # remove rhs_val from possible_answers at rhs_attr and vice versa.
          if (lhs[2] not in self.answer[lhs_attr]
              and rhs[2] not in self.answer[rhs_attr]):
            rhs_indices = []
            lhs_indices = []
            for _ in range(2):
              for i in range(self.n):
                if (self.answer[lhs_attr][i] is not None
                    and rhs[2] in p_ans[rhs_attr][i]):
                  # if position i for lhs_attr is filled,
                  # rhs_attr can't occur at i
                  p_ans[rhs_attr][i].remove(rhs[2])
                  rhs_indices.append(i)
                if (self.answer[rhs_attr][i] is not None
                    and lhs[2] in p_ans[lhs_attr][i]):
                  # if position i for rhs_attr is filled,
                  # lhs_attr can't occur at i
                  p_ans[lhs_attr][i].remove(lhs[2])
                  lhs_indices.append(i)
                if (lhs[2] not in p_ans[lhs_attr][i]
                    and rhs[2] in p_ans[rhs_attr][i]):
                  p_ans[rhs_attr][i].remove(rhs[2])
                  rhs_indices.append(i)
                if (rhs[2] not in p_ans[rhs_attr][i]
                    and lhs[2] in p_ans[lhs_attr][i]):
                  p_ans[lhs_attr][i].remove(lhs[2])
                  lhs_indices.append(i)
            if rhs_indices or lhs_indices:
              # aux info specifies the lhs_indices and rhs_indices from which
              # lhs vals and rhs vals have been filtered respectively
              reasoning_step = zebra_utils.ZebraSolverStep(
                  clue_list=[clue], reason='=,negative-grounded',
                  auxiliary_info=[lhs_indices, rhs_indices],
                  current_answer_table=copy.deepcopy(self.answer))
              reasoning_steps.append(reasoning_step)

        case '!=':
          lhs, rhs = clue.lhs_list[0], clue.rhs_list[0]
          lhs_attr = zebra_utils.get_attr_num(lhs, self.m2)
          rhs_attr = zebra_utils.get_attr_num(rhs, self.m2)
          # 1. If lhs entity or rhs entity has been grounded in the answer
          # table, remove it from corresponding possible_answers entry.
          if lhs[2] in self.answer[lhs_attr]:
            # lhs entity is grounded
            index = self.get_grounded_entity_position(lhs)
            if rhs[2] in p_ans[rhs_attr][index]:
              p_ans[rhs_attr][index].remove(rhs[2])
              # auxiliary_info specifies which of lhs or rhs is being deduced in
              # reasoning_step below and the index of grounding
              reasoning_step = zebra_utils.ZebraSolverStep(
                  clue_list=[clue], reason='!=,grounded',
                  auxiliary_info=[1, index],
                  current_answer_table=copy.deepcopy(self.answer))
              reasoning_steps.append(reasoning_step)
          elif rhs[2] in self.answer[rhs_attr]:
            # rhs entity is grounded
            index = self.get_grounded_entity_position(rhs)
            if lhs[2] in p_ans[lhs_attr][index]:
              p_ans[lhs_attr][index].remove(lhs[2])
              # auxiliary_info specifies which of lhs or rhs is being deduced
              # in reasoning_step below and the index of grounding
              reasoning_step = zebra_utils.ZebraSolverStep(
                  clue_list=[clue], reason='!=,grounded',
                  auxiliary_info=[0, index],
                  current_answer_table=copy.deepcopy(self.answer))
              reasoning_steps.append(reasoning_step)

        case 'nbr':
          lhs, rhs = clue.lhs_list[0], clue.rhs_list[0]
          lhs_attr = zebra_utils.get_attr_num(lhs, self.m2)
          rhs_attr = zebra_utils.get_attr_num(rhs, self.m2)
          # 1. If lhs entity or rhs entity is grounded, then update
          # possible_answers table
          if lhs[2] in self.answer[lhs_attr]:
            # lhs entity is grounded
            index = self.get_grounded_entity_position(lhs)
            indices = []
            for i in range(self.n):
              if (i not in [index - 1, index + 1]
                  and rhs[2] in p_ans[rhs_attr][i]):
                p_ans[rhs_attr][i].remove(rhs[2])
                indices.append(i)
            rhs_possible_locations = self.get_possible_locations(p_ans, rhs)
            if indices and len(rhs_possible_locations) > 1:
              reasoning_step = zebra_utils.ZebraSolverStep(
                  clue_list=[clue], reason='nbr,grounded',
                  auxiliary_info=[1, indices],
                  current_answer_table=copy.deepcopy(self.answer))
              reasoning_steps.append(reasoning_step)
            elif indices and len(rhs_possible_locations) == 1:
              self.answer[rhs_attr][rhs_possible_locations[0]] = rhs[2]
              _, p_ans = self.filter_possible_answers(p_ans)
              reasoning_step = zebra_utils.ZebraSolverStep(
                  clue_list=[clue], reason='nbr,grounded',
                  auxiliary_info=[1, indices, rhs_possible_locations[0]],
                  current_answer_table=copy.deepcopy(self.answer))
              reasoning_steps.append(reasoning_step)
              successful_deduction = True
              break
          if rhs[2] in self.answer[rhs_attr]:
            # rhs entity is grounded
            index = self.get_grounded_entity_position(rhs)
            indices = []
            for i in range(self.n):
              if (i not in [index - 1, index + 1]
                  and lhs[2] in p_ans[lhs_attr][i]):
                p_ans[lhs_attr][i].remove(lhs[2])
                indices.append(i)
            lhs_possible_locations = self.get_possible_locations(p_ans, lhs)
            if indices and len(lhs_possible_locations) > 1:
              reasoning_step = zebra_utils.ZebraSolverStep(
                  clue_list=[clue], reason='nbr,grounded',
                  auxiliary_info=[0, indices],
                  current_answer_table=copy.deepcopy(self.answer))
              reasoning_steps.append(reasoning_step)
            elif indices and len(lhs_possible_locations) == 1:
              self.answer[lhs_attr][lhs_possible_locations[0]] = lhs[2]
              _, p_ans = self.filter_possible_answers(p_ans)
              reasoning_step = zebra_utils.ZebraSolverStep(
                  clue_list=[clue], reason='nbr,grounded',
                  auxiliary_info=[0, indices, lhs_possible_locations[0]],
                  current_answer_table=copy.deepcopy(self.answer))
              reasoning_steps.append(reasoning_step)
              successful_deduction = True
              break
          # 2. If neither have been grounded, using possible locations of lhs,
          # narrow down possible locations of rhs and vice versa.
          if (lhs[2] not in self.answer[lhs_attr]
              and rhs[2] not in self.answer[rhs_attr]):
            lhs_indices = []
            rhs_indices = []
            for _ in range(2):
              lhs_possible_locations = self.get_possible_locations(p_ans, lhs)
              rhs_possible_locations = self.get_possible_locations(p_ans, rhs)
              for i in range(self.n):
                if ((i-1) not in lhs_possible_locations
                    and (i+1) not in lhs_possible_locations):
                  # rhs cannot occur at position i
                  if rhs[2] in p_ans[rhs_attr][i]:
                    p_ans[rhs_attr][i].remove(rhs[2])
                    rhs_indices.append(i)
                if ((i-1) not in rhs_possible_locations
                    and (i+1) not in rhs_possible_locations):
                  # lhs cannot occur at position i
                  if lhs[2] in p_ans[lhs_attr][i]:
                    p_ans[lhs_attr][i].remove(lhs[2])
                    lhs_indices.append(i)
            if lhs_indices or rhs_indices:
              reasoning_step = zebra_utils.ZebraSolverStep(
                  clue_list=[clue], reason='nbr,possible-locations',
                  auxiliary_info=[lhs_indices, rhs_indices],
                  current_answer_table=copy.deepcopy(self.answer))
              reasoning_steps.append(reasoning_step)

        case 'ends':
          entity = clue.lhs_list[0]
          entity_attr = zebra_utils.get_attr_num(entity, self.m2)
          # 1. If one end is filled and not equal to current entity,
          # fill in the other end.
          if (self.answer[entity_attr][0] is not None
              and self.answer[entity_attr][0] != entity[2]):
            self.answer[entity_attr][self.n-1] = entity[2]
            _, p_ans = self.filter_possible_answers(p_ans)
            successful_deduction = True
            # auxiliary_nfo indicates which end the entity is
            # being filled in - 0 for left end, 1 for right end.
            reasoning_step = zebra_utils.ZebraSolverStep(
                clue_list=[clue], reason='ends,one-end-filled',
                auxiliary_info=[1, len(self.answer[0])],
                current_answer_table=copy.deepcopy(self.answer))
            reasoning_steps.append(reasoning_step)
            break
          if (self.answer[entity_attr][self.n-1] is not None
              and self.answer[entity_attr][self.n-1] != entity[2]):
            self.answer[entity_attr][0] = entity[2]
            _, p_ans = self.filter_possible_answers(p_ans)
            successful_deduction = True
            # auxiliary_info indicates which end the entity is
            # being filled in - 0 for left end, 1 for right end
            reasoning_step = zebra_utils.ZebraSolverStep(
                clue_list=[clue], reason='ends,one-end-filled',
                auxiliary_info=[0, len(self.answer[0])],
                current_answer_table=copy.deepcopy(self.answer))
            reasoning_steps.append(reasoning_step)
            break
          # 2. Remove entity from possible_answers list of middle positions.
          indices = []
          for i in range(1, self.n-1):
            if entity[2] in p_ans[entity_attr][i]:
              p_ans[entity_attr][i].remove(entity[2])
              indices.append(i)
          if indices:
            reasoning_step = zebra_utils.ZebraSolverStep(
                clue_list=[clue], reason='ends,middle-positions',
                auxiliary_info=[indices],
                current_answer_table=copy.deepcopy(self.answer))
            reasoning_steps.append(reasoning_step)

        case 'immediate-left':
          lhs, rhs = clue.lhs_list[0], clue.rhs_list[0]
          lhs_attr = zebra_utils.get_attr_num(lhs, self.m2)
          rhs_attr = zebra_utils.get_attr_num(rhs, self.m2)
          # 1. If lhs entity or rhs entity is grounded, then
          # update possible_answers table
          if lhs[2] in self.answer[lhs_attr]:
            # lhs entity is grounded
            index = self.get_grounded_entity_position(lhs)
            if self.answer[rhs_attr][index+1] is None:
              self.answer[rhs_attr][index+1] = rhs[2]
              _, p_ans = self.filter_possible_answers(p_ans)
              successful_deduction = True
              reasoning_step = zebra_utils.ZebraSolverStep(
                  clue_list=[clue], reason='immediate-left,grounded',
                  auxiliary_info=[1, index+1],
                  current_answer_table=copy.deepcopy(self.answer))
              reasoning_steps.append(reasoning_step)
              break
          if rhs[2] in self.answer[rhs_attr]:
            # rhs entity is grounded
            index = self.get_grounded_entity_position(rhs)
            if self.answer[lhs_attr][index-1] is None:
              self.answer[lhs_attr][index-1] = lhs[2]
              _, p_ans = self.filter_possible_answers(p_ans)
              successful_deduction = True
              reasoning_step = zebra_utils.ZebraSolverStep(
                  clue_list=[clue], reason='immediate-left,grounded',
                  auxiliary_info=[0, index-1],
                  current_answer_table=copy.deepcopy(self.answer))
              reasoning_steps.append(reasoning_step)
              break
          # 2. If neither have been grounded, using possible locations of lhs,
          # narrow down possible locations of rhs and vice versa.
          if (lhs[2] not in self.answer[lhs_attr]
              and rhs[2] not in self.answer[rhs_attr]):
            lhs_indices = []
            rhs_indices = []
            for _ in range(2):
              lhs_possible_locations = self.get_possible_locations(p_ans, lhs)
              rhs_possible_locations = self.get_possible_locations(p_ans, rhs)
              for i in range(self.n):
                if (i-1) not in lhs_possible_locations:
                  # rhs cannot occur at position i
                  if rhs[2] in p_ans[rhs_attr][i]:
                    p_ans[rhs_attr][i].remove(rhs[2])
                    rhs_indices.append(i)
                if (i+1) not in rhs_possible_locations:
                  # lhs cannot occur at position i
                  if lhs[2] in p_ans[lhs_attr][i]:
                    p_ans[lhs_attr][i].remove(lhs[2])
                    lhs_indices.append(i)
            if lhs_indices or rhs_indices:
              reasoning_step = zebra_utils.ZebraSolverStep(
                  clue_list=[clue], reason='immediate-left,possible-locations',
                  auxiliary_info=[lhs_indices, rhs_indices],
                  current_answer_table=copy.deepcopy(self.answer))
              reasoning_steps.append(reasoning_step)

        case 'left-of':
          lhs, rhs = clue.lhs_list[0], clue.rhs_list[0]
          lhs_attr = zebra_utils.get_attr_num(lhs, self.m2)
          rhs_attr = zebra_utils.get_attr_num(rhs, self.m2)
          # 1. Based on possible positions for rhs entity,
          # update possible positions for lhs entity and vice versa.
          lhs_indices = []
          rhs_indices = []
          leftmost_lhs_possible_location = self.get_possible_locations(
              p_ans, lhs)[0]
          rightmost_rhs_possible_location = self.get_possible_locations(
              p_ans, rhs)[-1]
          for i in range(self.n):
            if (i >= rightmost_rhs_possible_location
                and lhs[2] in p_ans[lhs_attr][i]):
              p_ans[lhs_attr][i].remove(lhs[2])
              lhs_indices.append(i)
            if (i <= leftmost_lhs_possible_location
                and rhs[2] in p_ans[rhs_attr][i]):
              p_ans[rhs_attr][i].remove(rhs[2])
              rhs_indices.append(i)
          if lhs_indices or rhs_indices:
            lhs_index, rhs_index = -1, -1
            lhs_possible_locations = self.get_possible_locations(p_ans, lhs)
            rhs_possible_locations = self.get_possible_locations(p_ans, rhs)
            if ((lhs[2] in self.answer[lhs_attr])
                and (len(rhs_possible_locations) == 1)):
              # lhs is grounded
              lhs_index = self.get_grounded_entity_position(lhs)
              if self.answer[rhs_attr][rhs_possible_locations[0]] is None:
                self.answer[rhs_attr][rhs_possible_locations[0]] = rhs[2]
                _, p_ans = self.filter_possible_answers(p_ans)
                successful_deduction = True
                reasoning_step = zebra_utils.ZebraSolverStep(
                    clue_list=[clue], reason='left-of,unique-grounded',
                    auxiliary_info=[1, rhs_possible_locations[0]],
                    current_answer_table=copy.deepcopy(self.answer))
                reasoning_steps.append(reasoning_step)
                break
            if ((rhs[2] in self.answer[rhs_attr])
                and (len(lhs_possible_locations) == 1)):
              # rhs is grounded
              rhs_index = self.get_grounded_entity_position(rhs)
              if self.answer[lhs_attr][lhs_possible_locations[0]] is None:
                self.answer[lhs_attr][lhs_possible_locations[0]] = lhs[2]
                _, p_ans = self.filter_possible_answers(p_ans)
                successful_deduction = True
                reasoning_step = zebra_utils.ZebraSolverStep(
                    clue_list=[clue], reason='left-of,unique-grounded',
                    auxiliary_info=[0, lhs_possible_locations[0]],
                    current_answer_table=copy.deepcopy(self.answer))
                reasoning_steps.append(reasoning_step)
                break
            # auxiliary_info contains lhs_index and rhs_index to deduce
            # whether lhs or rhs was grounded before this clue was applied.
            reasoning_step = zebra_utils.ZebraSolverStep(
                clue_list=[clue], reason='left-of,possible-locations',
                auxiliary_info=[lhs_indices, rhs_indices,
                                lhs_index, rhs_index],
                current_answer_table=copy.deepcopy(self.answer))
            reasoning_steps.append(reasoning_step)

        case 'inbetween':
          lhs, rhs1, rhs2 = clue.lhs_list[0], clue.rhs_list[0], clue.rhs_list[1]
          lhs_attr = zebra_utils.get_attr_num(lhs, self.m2)
          rhs1_attr = zebra_utils.get_attr_num(rhs1, self.m2)
          rhs2_attr = zebra_utils.get_attr_num(rhs2, self.m2)
          for _ in range(2):
            lhs_indices = []
            rhs1_indices = []
            rhs2_indices = []
            leftmost_lhs_possible_location = self.get_possible_locations(
                p_ans, lhs)[0]
            rightmost_rhs1_possible_location = self.get_possible_locations(
                p_ans, rhs1)[-1]
            leftmost_rhs1_possible_location = self.get_possible_locations(
                p_ans, rhs1)[0]
            rightmost_rhs2_possible_location = self.get_possible_locations(
                p_ans, rhs2)[-1]
            for i in range(self.n):
              if (i >= rightmost_rhs1_possible_location
                  and lhs[2] in p_ans[lhs_attr][i]):
                p_ans[lhs_attr][i].remove(lhs[2])
                lhs_indices.append(i)
              if (i <= leftmost_lhs_possible_location
                  and rhs1[2] in p_ans[rhs1_attr][i]):
                p_ans[rhs1_attr][i].remove(rhs1[2])
                rhs1_indices.append(i)
              if (i >= rightmost_rhs2_possible_location
                  and rhs1[2] in p_ans[rhs1_attr][i]):
                p_ans[rhs1_attr][i].remove(rhs1[2])
                rhs2_indices.append(i)
              if (i <= leftmost_rhs1_possible_location
                  and rhs2[2] in p_ans[rhs2_attr][i]):
                p_ans[rhs2_attr][i].remove(rhs2[2])
                rhs2_indices.append(i)
            if lhs_indices or rhs1_indices or rhs2_indices:
              lhs_index, rhs1_index, rhs2_index = -1, -1, -1
              if lhs[2] in self.answer[lhs_attr]:
                lhs_index = self.get_grounded_entity_position(lhs)
              if rhs1[2] in self.answer[rhs1_attr]:
                rhs1_index = self.get_grounded_entity_position(rhs1)
              if rhs2[2] in self.answer[rhs2_attr]:
                rhs2_index = self.get_grounded_entity_position(rhs2)
              # auxiliary_info contains lhs_index, rhs1_index ans rhs2_index
              # to deduce if any of the entities were grounded before this
              # clue was applied
              reasoning_step = zebra_utils.ZebraSolverStep(
                  clue_list=[clue], reason='inbetween,possible-locations',
                  auxiliary_info=[lhs_indices, rhs1_indices, rhs2_indices,
                                  lhs_index, rhs1_index, rhs2_index],
                  current_answer_table=copy.deepcopy(self.answer))
              reasoning_steps.append(reasoning_step)
      progress, reasoning_step = self.fill_by_elimination(p_ans)
      if reasoning_step is not None:
        reasoning_steps.append(reasoning_step)
      successful_deduction = progress or successful_deduction
      if hard_deduce and verbose:
        print(clue)
        self.print_possible_answers(p_ans)
      if successful_deduction:
        break
    # If we got a successful deduction using clue_list
    # (i.e. at least one new entry filled in answer table),
    # accept the p_ans as the possible_answers and call fill by elimination
    if successful_deduction:
      if verbose:
        print('Successful deduction!')
        print(clue_list)
      possible_answers = copy.deepcopy(p_ans)
      _, possible_answers = self.filter_possible_answers(possible_answers)
      _, possible_answers = self.filter_possible_answers(possible_answers)
    elif hard_deduce:
      # if no successful deduction, call hard_deduce with the restricted p_ans
      p_ans, successful_deduction, hard_reasoning_steps = (
          self.hard_deduce_from_clue_list(p_ans, clue_list)
      )
      if successful_deduction:
        if verbose:
          print('Successful hard deduction!')
          print(clue_list)
        possible_answers = copy.deepcopy(p_ans)
        _, possible_answers = self.filter_possible_answers(possible_answers)
        reasoning_steps.extend(hard_reasoning_steps)
    return successful_deduction, possible_answers, reasoning_steps

  def solve(self, verbose: bool = True, hard_deduce: bool = False) -> Tuple[
      bool,
      bool,
      List[List[int | None]],
      List[List[List[int]]],
      List[List[zebra_utils.ZebraSolverStep | None]],
  ]:
    """Solve the symbolic zebra puzzle.

    Args:
      verbose: Whether to print some intermediate messages along the way.
      hard_deduce: Whether to use hard_deduce function.

    Returns:
      solved: Whether the solver was able to solve the puzzle.
      used_hard_deduce: Whether the solver had to use hard deduce (if allowed)
      answer: The answer table
      possible_answers: The possible answers table
      solution: The solution represented as a list of reasoning blocks
        each of which is a list of ZebraSolverSteps.
    """
    solved = False
    solution = []
    possible_answers = []
    for j in range(self.m):
      possible_answers.append([])
      for i in range(self.n):
        if j == 0:
          possible_answers[j].append([i])
        else:
          possible_answers[j].append(list(range(self.n)))
    clues = self.symbolic_puzzle.clues
    clue_singlets = [(clue,) for clue in clues]
    # pairs and triplets might not be required
    clue_pair_perms = list(itertools.permutations(clues, 2))
    clue_triplet_perms = list(itertools.permutations(clues, 3))
    clue_permutations = clue_singlets + clue_pair_perms + clue_triplet_perms
    hard_clue_permutations = None
    if hard_deduce:
      clue_quadruplet_combs = list(itertools.combinations(clues, 4))
      clue_quintuplet_combs = list(itertools.combinations(clues, 5))
      hard_clue_permutations = (
          clue_singlets + clue_pair_perms + clue_triplet_perms
      )
      if len(clue_quadruplet_combs) <= 20000:
        hard_clue_permutations = clue_quintuplet_combs + hard_clue_permutations
      if len(clue_quintuplet_combs) <= 10000:
        hard_clue_permutations = clue_quadruplet_combs + hard_clue_permutations
    if verbose:
      print('Number of clue permutations', len(clue_permutations))
      if hard_deduce:
        print('Number of hard clue combinations', len(hard_clue_permutations))
      print(clues)

    # Iterate over clues - try to solve without hard deduce first
    non_redundant_clues = copy.deepcopy(clues)
    progress = True
    used_hard_deduce = False
    while progress:
      if self.check_invalid_state(possible_answers):
        if verbose:
          print('Invalid state!')
          self.print_possible_answers(possible_answers)
          self.print_answer_table()
        break
      progress = False
      for clue_perm in clue_permutations:
        deduce, _, reasoning_steps = self.deduce_from_clue_list(
            list(clue_perm), possible_answers, hard_deduce=False
        )
        _, possible_answers = self.filter_possible_answers(possible_answers)
        progress = progress or deduce
        if progress:
          solution.append(reasoning_steps)
          _, reasoning_step = self.fill_by_elimination(
              possible_answers, allow_multiple_fills=True
          )
          if reasoning_step is not None:
            solution.append([reasoning_step])
          if verbose:
            print('Progress!')
            self.print_answer_table()
          break
      # Mark redundant clues and re-compute clue_permutations
      # with remaining clues
      redundant_clue_found = False
      for clue in non_redundant_clues:
        if zebra_utils.check_redundant(
            clue.clue_type, clue.lhs_list, clue.rhs_list, self.answer, self.m2):
          non_redundant_clues.remove(clue)
          redundant_clue_found = True
          if verbose:
            print('Redundant clue found!')
            print(clue)
      if redundant_clue_found:
        clue_singlets = [(clue,) for clue in non_redundant_clues]
        clue_pair_perms = list(itertools.permutations(non_redundant_clues, 2))
        clue_triplet_perms = list(itertools.permutations(non_redundant_clues,
                                                         3))
        clue_permutations = clue_singlets + clue_pair_perms + clue_triplet_perms
        if hard_deduce:
          clue_quadruplet_combs = list(itertools.combinations(clues, 4))
          clue_quintuplet_combs = list(itertools.combinations(clues, 5))
          hard_clue_permutations = (
              clue_singlets + clue_pair_perms + clue_triplet_perms)
          if len(clue_quadruplet_combs) <= 40000:
            hard_clue_permutations = (clue_quintuplet_combs
                                      + hard_clue_permutations)
          if len(clue_quintuplet_combs) <= 10000:
            hard_clue_permutations = (clue_quadruplet_combs
                                      + hard_clue_permutations)

      if not self.is_solved() and not progress and hard_deduce:
        # Call hard deduce while trying to adhere as much as possible to
        # easy deduce still.
        used_hard_deduce = True
        for clue_perm in hard_clue_permutations:
          if verbose:
            print('Trying hard deduce with ',
                  len(clue_perm), 'clues.', list(clue_perm))
          deduce, _, reasoning_steps = self.deduce_from_clue_list(
              list(clue_perm), possible_answers, hard_deduce=True)
          _, possible_answers = self.filter_possible_answers(possible_answers)
          progress = progress or deduce
          if progress:
            if verbose:
              print('Successful hard deduce with ', list(clue_perm))
              print('Progress!')
              self.print_answer_table()
            solution.append(reasoning_steps)
            _, reasoning_step = self.fill_by_elimination(
                possible_answers, allow_multiple_fills=True)
            if reasoning_step is not None:
              solution.append([reasoning_step])
            break
      elif self.is_solved():
        solved = True
        if verbose:
          print('Solved!')
          self.print_possible_answers(possible_answers)
          self.print_answer_table()
        break
    return solved, used_hard_deduce, self.answer, possible_answers, solution
