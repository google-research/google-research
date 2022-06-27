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

"""Global constants and classes for LogicInference dataset generation.
"""

import dataclasses
from typing import Any, List


# Global variables:
NEXT_RENAME_INDEX = 1

PROPOSITION_NAMES = ["p", "q", "r", "s", "t", "u", "w"]
FUNCTOR_NAMES = ["P", "Q", "R", "S", "T", "U", "W"]
CONSTANT_NAMES = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
VARIABLE_NAMES = ["var x", "var y", "var z"]

# Each rule is a tuple with 6 elements:
# (premises, conclusions, contradictions, irrelevant, propositions, name)
PROPOSITIONAL_INFERENCE_RULES = []
QUANTIFIED_INFERENCE_RULES = []
ALL_INFERENCE_RULES = []
ALL_RULE_NAMES = []

# Counters to keep track of how many examples of each type are actually
# generated. And for each problem type, we also count the number of times we
# generate examples with contradictions, and with certain other special cases.
EXAMPLE_PROBLEM_TYPES = [
    "1",
    "2a", "2a-cont", "2a-empty",
    "2b", "2b-cont", "2b-empty",
    "3a", "3a-cont", "3a-premise", "3a-no", "3a-no-1",
    "3a-unrelated",
    "3b", "3b-cont", "3b-premise", "3b-no", "3b-no-1",
    "3b-unrelated"]
EXAMPLE_TYPE_STATS = {}


# Common logic inference rules. For each rule we provide:
# - the premises (list of clauses)
# - the conclusions that can be drawn (list of clauses)
# - a collection of clauses that would contradict the premises
# - a collection of clauses that cannot be inferred from the premises, and
#   are hence, unrelated.
# - the set of atomic clauses (propositions) that appear in the lists above.
# - the rule name.
# These lists are non-exhaustive, and are just used to generate candidate
# inferences for the training examples.
@dataclasses.dataclass
class InferenceRule:
  premises: List[Any]
  inferences: List[Any]
  contradictions: List[Any]
  unrelated: List[Any]
  propositions: List[Any]
  rule_name: str


# Each inference problem is a 6-tuple consisting of:
# - premises
# - inferences: [clause, reasoning chain]. Where reasoning chains is
#               a ("premises, inferences, name") list where premises is a subset
#               of original premises + inferences already obtained.
# - contradictions: same as above, with the last step being the contradiction.
# - unrelated: just a plain list of clauses.
# - propositions: list of all atomic clauses appearning in all the lists before.
# - contains_contradiction: whether the premises lead to a contradiction or not.
@dataclasses.dataclass
class InferenceProblem:
  premises: List[Any]
  inferences: List[Any]
  contradictions: List[Any]
  unrelated: List[Any]
  propositions: List[Any]
  contains_contradiction: bool


@dataclasses.dataclass
class Example:
  inputs: str
  targets: str
  example_type: str
  problem: InferenceProblem
