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

"""Inference problem definition for the LogicInference dataset.
"""

import random

import inference_methods as im
import logic_inference_lib as lib


def sample_from_distribution(d):
  r = random.random()*sum(d)
  for i in range(len(d)):
    r -= d[i]
    if r <= 0:
      return i
  return len(d)-1


def apply_bindings_to_problem(bindings, problem):
  return lib.InferenceProblem(
      im.apply_bindings(bindings, problem.premises),
      im.apply_bindings(bindings, problem.inferences),
      im.apply_bindings(bindings, problem.contradictions),
      im.apply_bindings(bindings, problem.unrelated),
      im.apply_bindings(bindings, problem.propositions),
      problem.contains_contradiction)


def generate_multistep_problem(length_distribution=None):
  """Returns a new infrence problem.

  Args:
    length_distribution: a list of weights for candidate problem lengths, where
      the "length" represents the number of inference steps in the problem. If
      not passing anything, the default is [0.5, 0.25, 0.25], representing a
      50% chance of problems of length 1, and 25% chance of problems of lengths
      2 and 3 each.

  Returns:
    problem: a lib.InferenceProblem.
  """

  if length_distribution is None:
    length_distribution = [0.5, 0.25, 0.25]

  # This only defines the number of attempts that will be made to add new
  # inference steps fo the base chain of one step. So, it's an upper bound on
  # the inference length:
  target_length = sample_from_distribution(length_distribution)

  # - Pick a random rule, this defines a set of premises, and a set of
  #   inferences/contradictions (ignore the unrelated)
  lib.NEXT_RENAME_INDEX = 1
  # Since there are more quantified than propositional rules, increase the
  # probability of the propositional rules, to balance it out:
  if random.random() < 0.5:
    rule = random.choice(lib.PROPOSITIONAL_INFERENCE_RULES)
  else:
    rule = random.choice(lib.ALL_INFERENCE_RULES)
  premises = list(rule.premises)
  inferences = []
  contradictions = []
  for rule_inference in rule.inferences:
    inferences.append([rule_inference, [[rule.premises, [rule_inference],
                                         rule.rule_name]]])
  for contradiction in rule.contradictions:
    rule_inference = rule.inferences[0]
    contradictions.append([contradiction, [[rule.premises, [rule_inference],
                                            rule.rule_name]]])
  unrelated = rule.unrelated
  propositions = list(rule.propositions)

  # Some rules only have "unrelated":
  if not inferences and not contradictions:
    return lib.InferenceProblem(premises, inferences, contradictions, unrelated,
                                propositions, False)

  while target_length > 0:
    target_length -= 1

    # - Pick a random premise, and look for a rule for which one of the
    # inferences unify it.
    premise = random.choice(premises)
    next_rule_candidates = []
    for r in lib.ALL_INFERENCE_RULES:
      r = im.rename_clauses(r, lib.NEXT_RENAME_INDEX)
      bindings = im.unify_clause_with_rule(premise, r)
      if bindings is not None:
        lib.NEXT_RENAME_INDEX += 1
        next_rule_candidates.append([r, bindings])
    if next_rule_candidates:
      [next_rule, bindings] = random.choice(next_rule_candidates)

      # - Apply the bindings to the new rule and the current state:
      premises.remove(premise)
      next_rule = im.apply_bindings(bindings, next_rule)
      premise = im.apply_bindings(bindings, premise)
      premises = im.apply_bindings(bindings, premises)
      inferences = im.apply_bindings(bindings, inferences)
      contradictions = im.apply_bindings(bindings, contradictions)
      unrelated = im.apply_bindings(bindings, unrelated)
      propositions = im.apply_bindings(bindings, propositions)

      # - Add the new step to the chains:
      for p in next_rule.premises:
        premises.append(p)
      for p in next_rule.propositions:
        if p not in propositions:
          propositions.append(p)
      for problem_inference in inferences:
        problem_inference[1] = [[
            next_rule.premises, [premise], next_rule.rule_name
        ]] + problem_inference[1]
      for problem_inference in contradictions:
        problem_inference[1] = [[
            next_rule.premises, [premise], next_rule.rule_name
        ]] + problem_inference[1]

      im.split_compound_propositions(propositions)

  problem = lib.InferenceProblem(premises, inferences, contradictions,
                                 unrelated, propositions, False)
  shorten_chains_if_possible(problem)
  return problem


def generate_multistep_problems(target_n, length_distribution=None,
                                n_failures_before_giving_up=100,
                                contradictory_ratio=0.1):
  """Attempts to generate 'target_n' different problems.

  This method will try to generate 'target_n' different problems. However,
  since 'target_n' might not be possible, it will actually generate as many as
  it can up to 'target_n'. If after 'n_failures_before_giving_up' tries, it
  cannot generate any new problem, it will give up, and return the current
  set.

  Args:
    target_n: the number of lib.InferenceProblem instances to try to generate.
    length_distribution: the probability distribution over inference chain
      lengths.
    n_failures_before_giving_up: how many times to try to generate a new
      lib.InferenceProblem before giving up if we cannot generate any new
      problem not already in the generated list.
    contradictory_ratio: the ratio of how many problems with contradictions to
    have at most.

  Returns:
    A list of lib.InferenceProblem instances.
  """

  problems = []
  problems_with_contradictions = []
  problem_hash_set = {}
  n_failed_attempts_in_a_row = 0
  while len(problems) < target_n:
    problem = generate_multistep_problem(
        length_distribution=length_distribution)
    problem_hash = str(problem)
    if problem_hash not in problem_hash_set:
      if im.detect_contradiction_in_problem(problem):
        problem.contains_contradiction = True
        problems_with_contradictions.append(problem)
      else:
        problems.append(problem)
      problem_hash_set[problem_hash] = True
      n_failed_attempts_in_a_row = 0
      if len(problems) % 100 == 0:
        print(f"generate_multistep_problems: {len(problems)}/{target_n}")
    else:
      n_failed_attempts_in_a_row += 1
      if n_failed_attempts_in_a_row > n_failures_before_giving_up:
        break
  n_contradictory = min(int(len(problems) * contradictory_ratio),
                        len(problems_with_contradictions))
  problems += problems_with_contradictions[:n_contradictory]
  random.shuffle(problems)
  print(f"generate_multistep_problems: {len(problems)} "
        f"({n_contradictory} with contradictions)")
  return problems


def propositions_in_order_of_appearence(problem):
  """List of propositions, sorted by their order in the premises."""
  def depth_first_search(subtree, propositions, problem):
    if subtree in problem.propositions:
      if subtree not in propositions:
        propositions.append(subtree)
    elif isinstance(subtree, list):
      for child in subtree:
        depth_first_search(child, propositions, problem)

  # Get all the propositions that appear in the premises in the correct order.
  propositions = []
  depth_first_search(problem.premises, propositions, problem)
  # Add any additional propositions in the order in which they appear in the
  # problem.propositions list.
  for proposition in problem.propositions:
    if proposition not in propositions:
      propositions.append(proposition)
  return propositions


def generate_problem_canonical_renaming(problem):
  """Generates a problem canonical renaming.

  Given an lib.InferenceProblem, renames all its variables, propositions and
  constants so that the first proposition is "p", the next is "q", etc.

  Args:
    problem: the input lib.InferenceProblem.

  Returns:
    A copy of "problem" with the renaming already applied.
  """

  new_propositions = []
  bindings = {}
  propositions = list(lib.PROPOSITION_NAMES)
  functors = list(lib.FUNCTOR_NAMES)
  constants = list(lib.CONSTANT_NAMES)

  propositions_in_order = propositions_in_order_of_appearence(problem)
  for proposition in propositions_in_order:
    if im.is_base_proposition(proposition):
      if not propositions:
        # We ran out of canonical names:
        return None
      new_propositions.append(propositions[0])
      bindings[proposition] = propositions[0]
      propositions = propositions[1:]
    elif (isinstance(proposition, list) and
          len(proposition) == 2 and
          im.is_functor(proposition[0])):
      if proposition[0] in bindings:
        new_propositions.append(bindings[proposition[0]])
      else:
        if not functors:
          # We ran out of canonical names:
          return None
        new_propositions.append(functors[0])
        bindings[proposition[0]] = functors[0]
        functors = functors[1:]
      if proposition[1] not in bindings:
        if im.is_constant(proposition[1]):
          if not constants:
            # We ran out of canonical names:
            return None
          bindings[proposition[1]] = constants[0]
          constants = constants[1:]
        else:
          # a variable (for now they are all called "x"):
          bindings[proposition[1]] = "var x"
    else:
      raise ValueError(f"Unsupported propositions {proposition}")

  return apply_bindings_to_problem(bindings, problem)


def generate_problem_renaming(problem, attempts=100):
  """Generates a copy of the problem, but renaming variables.

  Args:
    problem: the input lib.InferenceProblem.
    attempts: how many times to try to rename a single variable trying to find
              a name without a clash, before giving up.

  Returns:
    A new instance of lib.InferenceProblem with variables renamed, or None if
    there was a name clash.
  """

  used_names = []
  for proposition in problem.propositions:
    if im.is_base_proposition(proposition):
      if proposition not in used_names:
        used_names.append(proposition)
    elif (isinstance(proposition, list) and
          len(proposition) == 2 and
          im.is_functor(proposition[0])):
      if proposition[0] not in used_names:
        used_names.append(proposition[0])
      if proposition[1] not in used_names:
        used_names.append(proposition[1])

  bindings = {}
  for proposition in problem.propositions:
    # 75% chance of renaming a proposition:
    if random.random() < 0.75:
      found = False
      for _ in range(attempts):
        if im.is_base_proposition(proposition):
          new_proposition = random.choice(lib.PROPOSITION_NAMES)
          if random.random() < 0.33:
            new_proposition += random.choice(["_1", "_2", "_3", "_4"])
          if new_proposition not in used_names:
            found = True
            break

        elif (isinstance(proposition, list) and
              len(proposition) == 2 and
              im.is_functor(proposition[0])):

          if proposition[0] not in bindings:
            new_functor = random.choice(lib.FUNCTOR_NAMES)
            if random.random() < 0.33:
              new_functor += random.choice(["_1", "_2", "_3", "_4"])
          else:
            new_functor = bindings[proposition[0]]

          if proposition[1] not in bindings:
            if im.is_constant(proposition[1]):
              new_argument = random.choice(lib.CONSTANT_NAMES)
              if random.random() < 0.33:
                new_argument += random.choice(["_1", "_2", "_3", "_4"])
            else:
              # a variable:
              new_argument = random.choice(lib.VARIABLE_NAMES)
              if random.random() < 0.33:
                new_argument += random.choice(["_1", "_2", "_3", "_4"])
          else:
            new_argument = bindings[proposition[1]]

          found = True
          if new_functor in used_names:
            found = False
          if new_argument in used_names:
            found = False
          if found:
            new_proposition = [new_functor, new_argument]
            break

        else:
          raise ValueError(f"Unsupported propositions {proposition}")

      if not found:
        new_proposition = proposition

    else:
      # Keep the name:
      new_proposition = proposition

    if isinstance(proposition, list):
      bindings[proposition[0]] = new_proposition[0]
      bindings[proposition[1]] = new_proposition[1]
    else:
      bindings[proposition] = new_proposition

  # Make sure the variation did not introduce a name clash:
  values = []
  for key in bindings:
    value = bindings[key]
    if value in values:
      return None
    values.append(value)

  return apply_bindings_to_problem(bindings, problem)


def generate_problem_renaming_variations(problem, target_n):
  """Generates a target number of renaming variations of a problem.

  Args:
    problem: the inference problem to start with.
    target_n: how many variations to generate.

  Returns:
    A list with all the variations generated, with duplicates removed. So, this
    list can be smaller than "target_n".
  """

  variations = []
  variations_hash_set = {}
  for _ in range(target_n):
    variation = generate_problem_renaming(problem)
    if variation:
      variation_hash = str(variation)
      if variation_hash not in variations_hash_set:
        variations.append(variation)
        variations_hash_set[variation_hash] = True
  return variations


def shorten_chains_if_possible(problem):
  """Reduces inference/contradiction chains if possible.

  Because of the method of construction, some times, inference chains are longer
  than they should be. For example, they often contain an extra inference step
  at the end when they originated from the rule of addition or of conjunction.
  This function detects that and removes unnecessary inference steps.

  Args:
    problem: an lib.InferenceProblem.

  This method does not return anything, but modifies the input problem.
  """

  def shortest_inference_chain(premises, proposition, chain):
    propositions = list(premises)
    if proposition in propositions:
      return 0
    length = 1
    for step in chain:
      propositions.extend(step[1])
      if proposition in propositions:
        return length
      length += 1
    return len(chain)

  def shortest_contradictory_chain(premises, proposition, chain):
    propositions = list(premises)
    propositions.append(proposition)
    if im.detect_contradiction(propositions, check_rules=False):
      return 0
    length = 1
    for step in chain:
      propositions.extend(step[1])
      if im.detect_contradiction(propositions, check_rules=False):
        return length
      length += 1
    return len(chain)

  # Inferences:
  for problem_inference in problem.inferences:
    length = shortest_inference_chain(problem.premises,
                                      problem_inference[0],
                                      problem_inference[1])
    if length < len(problem_inference[1]):
      problem_inference[1] = problem_inference[1][:length]

  # Contradictions:
  for contradiction in problem.contradictions:
    length = shortest_contradictory_chain(problem.premises,
                                          contradiction[0], contradiction[1])
    if length < len(contradiction[1]):
      contradiction[1] = contradiction[1][:length]


def rules_used_in_problem(problem):
  """Returns the number of times each rule is used in this problem.

  Args:
    problem: an lib.InferenceProblem.

  Returns:
    A dictionary where keys are rule names, and values are the number of times
    the corresponding rule appears in this problem.
  """

  counts = {}
  for problem_inference in problem.inferences + problem.contradictions:
    for step in problem_inference[1]:
      rule_name = step[2]
      if rule_name not in counts:
        counts[rule_name] = 0
      counts[rule_name] += 1
  return counts


def rules_used_in_problems(problems):
  """Returns the number of times each rule is used in a list of problems.

  Args:
    problems: a list of lib.InferenceProblem.

  Returns:
    A dictionary where keys are rule names, and values are the number of times
    the corresponding rule appears in this problem.
  """

  counts = {}
  for problem in problems:
    problem_counts = rules_used_in_problem(problem)
    for rule_name in problem_counts:
      if rule_name not in counts:
        counts[rule_name] = 0
      counts[rule_name] += problem_counts[rule_name]
  return counts


def problem_length_stats(problems):
  counts = []
  for problem in problems:
    for problem_inference in problem.inferences + problem.contradictions:
      l = len(problem_inference[1])
      while len(counts) <= l:
        counts.append(0)
      counts[l] += 1
  return counts
