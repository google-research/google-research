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

"""Example generation functions for the LogicInference dataset.
"""


import random

import inference_methods as im
import inference_problems as ip
import logic_inference_lib as lib
import rules


def reset_example_type_stats():
  for t in lib.EXAMPLE_PROBLEM_TYPES:
    lib.EXAMPLE_TYPE_STATS[t] = 0


def generate_example_with_problem(problem, example_types=None,
                                  answer_at_the_end=True,
                                  example_type_weights_map=None):
  """Generates a logic inference example.

  Examples can be of different types:
  - language to logic:
    (1) - Given a natural language sentence, generate the logical form in
      propositional or first order logic.
  - inference steps:
    (2a) - Given some propositional or first order logic formulas, predict what
      can be inferred from them (it should just be one or a small set of things,
      inferrable in one inference step).
    (2b) - The same, but in natural language.
    - Examples can potentially ask to name the inference rule being used.
  - inference chains:
    (3a) - Given some propositional or first order logic formulas, determine
      if we can infer or not a certain other formula, and provide the
      derivation for the proof or the proof of contradiction.
    (3b) - The same as above but in natural language.
    - Examples can potentially ask to name the inference rule being used at
      each step.

  Args:
    problem: the inference problem to use to generate an example.
    example_types: the list of problem types to consider for generating
      examples.
    answer_at_the_end: if "True", the final answer (yes/no/contradiction/etc.)
      will be added at the end of the final answer, rather than at the
      beginning.
    example_type_weights_map: probability distribution for generating each type
      of problem type.

  Returns:
    An instance of the Example class.
  """

  # Sample problem types with different weights, to balance the dataset to have
  # them in the desired proportion:
  if example_type_weights_map is None:
    example_type_weights_map = {"1": 0.5,
                                "2a": 1.0, "2b": 0.75,
                                "3a": 1.0, "3b": 0.75}
  if not example_types:
    example_types = ["1", "2a", "2b", "3a", "3b"]

  # Shuffle the premises to prevent models picking up on any pattern that might
  # resulf from the generation process.
  shuffled_premises = list(problem.premises)
  random.shuffle(shuffled_premises)
  problem = lib.InferenceProblem(
      shuffled_premises,
      problem.inferences,
      problem.contradictions,
      problem.unrelated,
      problem.propositions,
      problem.contains_contradiction)

  nl_propositions = []
  bindings = {}
  for proposition in problem.propositions:
    if isinstance(proposition, str):
      nl_proposition = rules.generate_nl_proposition(proposition, [], bindings)
    else:
      nl_proposition = rules.generate_nl_proposition(proposition[0],
                                                     proposition[1:],
                                                     bindings)
    nl_propositions.append(nl_proposition)

  chain_length = 0
  for problem_inference in problem.inferences:
    if len(problem_inference[1]) > chain_length:
      chain_length = len(problem_inference[1])
  for contradiction in problem.contradictions:
    if len(contradiction[1]) > chain_length:
      chain_length = len(contradiction[1])
  if not problem.inferences:
    if "1" in example_types:
      example_types.remove("1")
  if not example_types:
    # No example can be generated this time
    return None
  example_type_weights = [example_type_weights_map[x] for x in example_types]

  example_type = example_types[ip.sample_from_distribution(
      example_type_weights)]
  if example_type == "1":
    example = generate_example_type_1(problem, nl_propositions)
  elif example_type == "2a" or example_type == "2b":
    one_step_inferences = im.one_step_inferences_from_premises(
        problem.premises, rules_to_ignore=["addition", "instantiation"])
    propositions = list(problem.propositions)
    for [_, rule] in one_step_inferences:
      for proposition in rule.propositions:
        if proposition not in propositions:
          propositions.append(proposition)
          if isinstance(proposition, str):
            nl_proposition = rules.generate_nl_proposition(proposition, [],
                                                           bindings)
          else:
            nl_proposition = rules.generate_nl_proposition(proposition[0],
                                                           proposition[1:],
                                                           bindings)
          nl_propositions.append(nl_proposition)
    if example_type == "2a":
      example = generate_example_type_2a(problem, one_step_inferences)
    else:
      example = generate_example_type_2b(problem, one_step_inferences,
                                         propositions, nl_propositions)
  elif example_type == "3a":
    example = generate_example_type_3a(problem,
                                       answer_at_the_end=answer_at_the_end)
  elif example_type == "3b":
    example = generate_example_type_3b(problem, nl_propositions,
                                       answer_at_the_end=answer_at_the_end)
  else:
    raise ValueError(f"Example type {example_type} not supported.")

  return example


# Type 1: Given a natural language sentence, generate the logical form in
#         propositional or first order logic.
def generate_example_type_1(problem, nl_propositions):
  """Generates a type 1 training example.

  Args:
    problem: a lib.InferenceProblem instance.
    nl_propositions: the natural language versions of all the propositions
      appearing in "problem".

  Returns:
    An instance of "Example", or None if any issue was found.
  """
  problem = ip.generate_problem_canonical_renaming(problem)
  if not problem:
    return None
  premises = problem.premises
  inferences = problem.inferences
  propositions = problem.propositions
  [problem_inference, _] = random.choice(inferences)
  nl_inference = rules.render_language_clause(problem_inference, propositions,
                                              nl_propositions)
  nl_premises = []
  for premise in premises:
    nl_premises.append(
        rules.capitalize(
            rules.render_language_clause(premise, propositions,
                                         nl_propositions)))
  inputs = "Translate the following inference to logic notation: "
  inputs += (". ".join(nl_premises)) + f". Therefore {nl_inference}."
  targets = ". ".join([rules.render_logic_clause(x) for x in premises])
  targets = (f"{targets}. Therefore " +
             f"{rules.render_logic_clause(problem_inference)}.")
  return lib.Example(inputs, targets, "1", problem)


# Type 2a: Given some propositional or first order logic formulas, predict what
#          can be inferred from them (it should just be one or a small set of
#          things, inferrable in one inference step).
# - Examples can potentially ask to name the inference rule being used.
def generate_example_type_2a(problem, one_step_inferences):
  """Generates a type 2a training example.

  Args:
    problem: a lib.InferenceProblem instance.
    one_step_inferences: the list of one step inferences that can be reahced
      form the premises.

  Returns:
    An instance of "Example", or None if any issue was found.
  """

  premises = problem.premises
  example_type = "2a"
  name_rule = random.choice([True, False])
  inputs = ("What can be inferred from the following premises in a single "
            "inference step (ignoring inferences that add new predicates or "
            "constants)? ")
  if name_rule:
    inputs += "Name the inference rule being used: "
  inputs += (". ".join([rules.render_logic_clause(x) for x in premises])) + "."
  inferences_str = []
  for [rule_inference, rule] in one_step_inferences:
    rule_name = rule.rule_name
    inference_str = rules.render_logic_clause(rule_inference)
    if name_rule:
      inference_str += f" can be inferred via the {rule_name} rule"
    inferences_str.append(inference_str)
  targets = (". ".join(inferences_str)) + "."

  if not inferences_str:
    example_type = "2a-empty"
    targets = "Nothing can be inferred from these premises."
  elif problem.contains_contradiction:
    example_type = "2a-cont"
    targets = ("Since the premises are contradictory, we can infer anything "
               "from them.")

  return lib.Example(inputs, targets, example_type, problem)


# Type 2b: The same as 2a, but in natural language.
# - Examples can potentially ask to name the inference rule being used.
def generate_example_type_2b(problem, one_step_inferences,
                             propositions, nl_propositions):
  """Generates a type 2b training example.

  Args:
    problem: an InferenceProblem instance.
    one_step_inferences: the list of one step inferences that can be reahced
      form the premises.
    propositions: the list of propositions in the problem.
    nl_propositions: the natural language versions of "propositions".

  Returns:
    An instance of "lib.Example", or None if any issue was found.
  """

  premises = problem.premises
  example_type = "2b"
  nl_premises = []
  for premise in premises:
    nl_premises.append(
        rules.capitalize(
            rules.render_language_clause(premise, propositions,
                                         nl_propositions)))
  name_rule = random.choice([True, False])
  rule_name = None
  nl_inferences = []
  for [rule_inference, rule] in one_step_inferences:
    rule_name = rule.rule_name
    inference_str = rules.capitalize(rules.render_language_clause(
        rule_inference, propositions, nl_propositions))
    if name_rule:
      inference_str += f" can be inferred via the {rule_name} rule"
    nl_inferences.append(inference_str)
  inputs = ("What can be inferred from the following premises in a single "
            "inference step (ignoring inferences that add new predicates or "
            "constants)? ")
  if name_rule:
    inputs += "Name the inference rule being used: "
  inputs += (". ".join(nl_premises)) + "."
  targets = (". ".join(nl_inferences)) + "."

  if not nl_inferences:
    example_type = "2b-empty"
    targets = "Nothing can be inferred from these premises."
  elif problem.contains_contradiction:
    example_type = "2b-cont"
    targets = ("Since the premises are contradictory, we can infer anything "
               "from them.")

  return lib.Example(inputs, targets, example_type, problem)


# Type 3a: Given some propositional or first order logic formulas, determine
#          if we can infer or not a certain other formula, and provide the
#          derivation for the proof or the proof of contradiction.
# - Examples can potentially ask to name the inference rule being used at
#   each step.
def generate_example_type_3a(problem,
                             probability_of_adding_direct_inference=0.1,
                             answer_at_the_end=True):
  """Generates a type 3a training example.

  Args:
    problem: an InferenceProblem instance.
    probability_of_adding_direct_inference: the probability of having one of the
      premises as the target im.
    answer_at_the_end: whether to put the answer at the end of the targets, or
      at the beginning.

  Returns:
    An instance of "Example", or None if any issue was found.
  """

  # Since "unrelated" is usually larger, we randomly reduce it to size 1:
  example_type = "3a"
  premises = problem.premises
  unrelated = list(problem.unrelated)
  while len(unrelated) > 1:
    unrelated.remove(random.choice(unrelated))
  choices = ([(x, "inference") for x in problem.inferences] +
             [(x, "contradiction") for x in problem.contradictions] +
             [([x, []], "unrelated") for x in unrelated])
  # With some probability we also allow direct inferences (which are directly
  # given in the premises):
  if random.random() < probability_of_adding_direct_inference:
    for premise in problem.premises:
      choices.append(([premise, []], "inference"))
  if not choices:
    return None
  ([target_inference, chain], inference_type) = random.choice(choices)
  name_rule = random.choice([True, False])
  inputs = "Consider the following premises. "
  inputs += (". ".join([rules.render_logic_clause(x) for x in premises])) + ". "
  inputs += (f"Can we infer {rules.render_logic_clause(target_inference)} " +
             "from them? ")
  if name_rule:
    inputs += "If possible, name the inference rules being used at each step."
  if inference_type == "inference":
    if not chain:
      example_type = "3a-premise"
      if answer_at_the_end:
        targets = "That is one of the premises. Therefore, the answer is yes."
      else:
        targets = "Yes, that is one of the premises."
    elif len(chain) == 1:
      if name_rule:
        rule_name = chain[0][2]
        if answer_at_the_end:
          targets = ("We can infer " +
                     f"{rules.render_logic_clause(target_inference)} via the "
                     f"{rule_name} rule. Therefore, the answer is yes.")
        else:
          targets = ("Yes, we can infer " +
                     f"{rules.render_logic_clause(target_inference)} via the "
                     f"{rule_name} rule.")
      else:
        targets = "Yes."
    else:
      if problem.contains_contradiction:
        example_type = "3a-cont"
      if answer_at_the_end:
        targets = ""
      else:
        if problem.contains_contradiction:
          targets = ("Yes, the premises are contradictory, so we can infer "
                     "anything from them. For example via the following "
                     "inference chain.")
        else:
          targets = "Yes, via the following inference chain."
      for i in range(len(chain)):
        # Each element in the chain is: ("premises, inferences, name")
        if i == len(chain) - 1:
          targets += " Finally, from"
        else:
          targets += " From"
        # Premises:
        for j in range(len(chain[i][0])):
          premise = chain[i][0][j]
          premise_str = rules.render_logic_clause(premise)
          if j != 0:
            targets += ","
          targets += f" {premise_str}"
        targets += " we can infer"
        # Inferences:
        for j in range(len(chain[i][1])):
          chain_inference = chain[i][1][j]
          inference_str = rules.render_logic_clause(chain_inference)
          if j != 0:
            targets += ","
          targets += f" {inference_str}"
        if name_rule:
          targets += f" via {chain[i][2]}"
        targets += "."
      if answer_at_the_end:
        if problem.contains_contradiction:
          targets += (" Therefore, the answer is yes. Notice, however, that "
                      "the premises were contradictory, so we can infer "
                      "anything from them.")
        else:
          targets += " Therefore, the answer is yes."

  elif inference_type == "contradiction":
    if problem.contains_contradiction:
      example_type = "3a-cont"
      if answer_at_the_end:
        targets = ("The premises are contradictory and we can infer "
                   "anything from them. Therefore, the answer is yes.")
      else:
        targets = ("Yes, the premises are contradictory, so we can infer "
                   "anything from them.")
    elif len(chain) <= 1:
      example_type = "3a-no-1"
      if answer_at_the_end:
        targets = "That contradicts the premises. Therefore the answer is no."
      else:
        targets = "No, that contradicts the premises."
    else:
      example_type = "3a-no"
      if answer_at_the_end:
        targets = ""
      else:
        targets = "No, we can see why via the following inference chain."
      for i in range(len(chain)):
        # Each element in the chain is: ("premises, inferences, name")
        if i == len(chain) - 1:
          targets += " Finally, from"
        else:
          targets += " From"
        # Premises:
        for j in range(len(chain[i][0])):
          premise = chain[i][0][j]
          premise_str = rules.render_logic_clause(premise)
          if j != 0:
            targets += ","
          targets += f" {premise_str}"
        targets += " we can infer"
        # Inferences:
        for j in range(len(chain[i][1])):
          chain_inference = chain[i][1][j]
          inference_str = rules.render_logic_clause(chain_inference)
          if j != 0:
            targets += ","
          targets += f" {inference_str}"
        if name_rule:
          targets += f" via {chain[i][2]}"
        targets += "."
      targets = targets[:-1]  # Remove the last dot.
      targets += (", which contradicts " +
                  f"{rules.render_logic_clause(target_inference)}.")
      if answer_at_the_end:
        targets += " Therefore, the answer is no."

  else:
    example_type = "3a-unrelated"
    if answer_at_the_end:
      targets = "We cannot infer that from the premises."
      targets += " Therefore the answer is no."
    else:
      targets = "No, we cannot infer that from the premises."

  return lib.Example(inputs, targets, example_type, problem)


# Type 3b: The same as 3a but in natural language.
# - Examples can potentially ask to name the inference rule being used at
#   each step.
def generate_example_type_3b(problem, nl_propositions,
                             probability_of_adding_direct_inference=0.1,
                             answer_at_the_end=True):
  """Generates a type 3b training example.

  Args:
    problem: an InferenceProblem instance.
    nl_propositions: the natural language versions of "propositions".
    probability_of_adding_direct_inference: the probability of having one of the
      premises as the target im.
    answer_at_the_end: whether to put the answer at the end of the targets, or
      at the beginning.

  Returns:
    An instance of "Example", or None if any issue was found.
  """

  # Since "unrelated" is usually larger, we randomly reduce it to size 1:
  example_type = "3b"
  premises = problem.premises
  unrelated = list(problem.unrelated)
  while len(unrelated) > 1:
    unrelated.remove(random.choice(unrelated))
  choices = ([(x, "inference") for x in problem.inferences] +
             [(x, "contradiction") for x in problem.contradictions] +
             [([x, []], "unrelated") for x in unrelated])
  # With some probability we also allow direct inferences (which are directly
  # given in the premises):
  if random.random() < probability_of_adding_direct_inference:
    for premise in premises:
      choices.append(([premise, []], "inference"))
  if not choices:
    return None
  ([target_inference, chain], inference_type) = random.choice(choices)
  name_rule = random.choice([True, False])

  nl_premises = []
  for premise in premises:
    nl_premises.append(rules.capitalize(rules.render_language_clause(
        premise, problem.propositions, nl_propositions)))
  nl_target_inference = rules.capitalize(rules.render_language_clause(
      target_inference, problem.propositions, nl_propositions))

  inputs = "Consider the following premises. "
  inputs += (". ".join(nl_premises)) + ". "
  inputs += "Can we infer the following from them? "
  if name_rule:
    inputs += "If we can, name the inference rule being used: "
  inputs += nl_target_inference + "."
  if inference_type == "inference":
    if not chain:
      example_type = "3b-premise"
      if answer_at_the_end:
        targets = "That is one of the premises. Therefore, the answer is yes."
      else:
        targets = "Yes, that is one of the premises."
    elif len(chain) == 1:
      if name_rule:
        rule_name = chain[0][2]
        if answer_at_the_end:
          targets = ("We can infer this via the "
                     f"{rule_name} rule. Therefore, the answer is yes.")
        else:
          targets = f"Yes, we can infer this via the {rule_name} rule."
      else:
        targets = "Yes."
    else:
      if problem.contains_contradiction:
        example_type = "3b-cont"
      if answer_at_the_end:
        targets = ""
      else:
        if problem.contains_contradiction:
          targets = ("Yes, the premises are contradictory, so we can infer "
                     "anything from them. For example via the following "
                     "inference chain.")
        else:
          targets = "Yes, via the following inference chain."
      for i in range(len(chain)):
        # Each element in the chain is: ("premises, inferences, name")
        if i == len(chain) - 1:
          targets += " Finally, from the fact that"
        else:
          targets += " From the fact that"
        # Premises:
        for j in range(len(chain[i][0])):
          premise = chain[i][0][j]
          nl_premise = rules.render_language_clause(premise,
                                                    problem.propositions,
                                                    nl_propositions)
          if j != 0:
            if j == len(chain[i][0]) - 1:
              targets += ", and that"
            else:
              targets += ", that"
          targets += f" {nl_premise}"
        targets += " we can infer that"
        # Inferences:
        for j in range(len(chain[i][1])):
          chain_inference = chain[i][1][j]
          nl_inference = rules.render_language_clause(chain_inference,
                                                      problem.propositions,
                                                      nl_propositions)
          if j != 0:
            targets += ","
          targets += f" {nl_inference}"
        if name_rule:
          targets += f" via {chain[i][2]}"
        targets += "."
      if answer_at_the_end:
        if problem.contains_contradiction:
          targets += (" Therefore, the answer is yes. Notice, however, that "
                      "the premises were contradictory, so we can infer "
                      "anything from them.")
        else:
          targets += " Therefore, the answer is yes."

  elif inference_type == "contradiction":
    if problem.contains_contradiction:
      example_type = "3b-cont"
      if answer_at_the_end:
        targets = ("The premises are contradictory and we can infer "
                   "anything from them. Therefore, the answer is yes.")
      else:
        targets = ("Yes, the premises are contradictory, so we can infer "
                   "anything from them.")
    elif len(chain) <= 1:
      example_type = "3b-no-1"
      if answer_at_the_end:
        targets = "That contradicts the premises. Therefore the answer is no."
      else:
        targets = "No, that contradicts the premises."
    else:
      example_type = "3b-no"
      if answer_at_the_end:
        targets = ""
      else:
        targets = "No, we can see why via the following inference chain."
      for i in range(len(chain)):
        # Each element in the chain is: ("premises, inferences, name")
        if i == len(chain) - 1:
          targets += " Finally, from the fact that"
        else:
          targets += " From the fact that"
        # Premises:
        for j in range(len(chain[i][0])):
          premise = chain[i][0][j]
          nl_premise = rules.render_language_clause(premise,
                                                    problem.propositions,
                                                    nl_propositions)
          if j != 0:
            if j == len(chain[i][0]) - 1:
              targets += ", and that"
            else:
              targets += ", that"
          targets += f" {nl_premise}"
        targets += " we can infer that"
        # Inferences:
        for j in range(len(chain[i][1])):
          chain_inference = chain[i][1][j]
          nl_inference = rules.render_language_clause(chain_inference,
                                                      problem.propositions,
                                                      nl_propositions)
          if j != 0:
            targets += ","
          targets += f" {nl_inference}"
        if name_rule:
          targets += f" via {chain[i][2]}"
        targets += "."
      targets = targets[:-1]  # Remove the last dot.
      targets += f", which contradicts that {nl_target_inference}."
      if answer_at_the_end:
        targets += " Therefore, the answer is no."

  else:
    example_type = "3b-unrelated"
    if answer_at_the_end:
      targets = "We cannot infer that from the premises."
      targets += " Therefore the answer is no."
    else:
      targets = "No, we cannot infer that from the premises."

  return lib.Example(inputs, targets, example_type, problem)


def generate_examples_from_problems(problems, n_variations, n_examples,
                                    example_types=None,
                                    answer_at_the_end=True):
  """Generates a set of training examples from a set of inference problems.

  Args:
    problems: a list of InferenceProblem.
    n_variations: how many renaming variations to generate per problem.
    n_examples: how many examples to generate (before deduplication).
    example_types: the set of example types to consider.
    answer_at_the_end: whether to put the final answer at the end of an example
      or at the beginning.

  Returns:
    A list of deduplicated training examples (instances of "lib.Example").
  """
  reset_example_type_stats()
  variations = []
  print((f"generate_examples_from_problems generating renaming variations for "
         f"{len(problems)} problems..."))
  for i in range(len(problems)):
    if i % 100 == 0:
      print(f"    {i}/{len(problems)} with {len(variations)} variations")
    problem_variations = ip.generate_problem_renaming_variations(problems[i],
                                                                 n_variations)
    variations.extend(problem_variations)
  examples = []
  examples_hash = {}
  print((f"generate_examples_from_problems generating examples with "
         f"{len(variations)} variations..."))
  for i in range(n_examples):
    if i%10000 == 0:
      print(f"    {i}/{n_examples} with {len(examples)} so far")
    problem = random.choice(problems)
    example = generate_example_with_problem(problem,
                                            example_types=example_types,
                                            answer_at_the_end=answer_at_the_end)
    if example is not None:
      example_hash = str(example)
      if example_hash not in examples_hash:
        examples.append(example)
        examples_hash[example_hash] = True
        if example.example_type:
          lib.EXAMPLE_TYPE_STATS[example.example_type] += 1
  print(f"EXAMPLE_TYPE_STATS: {lib.EXAMPLE_TYPE_STATS}")
  return examples
