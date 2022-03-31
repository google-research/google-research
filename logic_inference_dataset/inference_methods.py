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

"""Basic logic inference functions for generating the LogicInference dataset.
"""


import logic_inference_lib as lib


def is_base_proposition(p):
  if isinstance(p, str):
    for prefix in lib.PROPOSITION_NAMES:
      if p.startswith(prefix):
        return True
  return False


def is_functor(p):
  if isinstance(p, str):
    for prefix in lib.FUNCTOR_NAMES:
      if p.startswith(prefix):
        return True
  return False


def is_constant(p):
  if isinstance(p, str):
    for prefix in lib.CONSTANT_NAMES:
      if p.startswith(prefix):
        return True
  return False


def is_variable(p):
  if isinstance(p, str):
    for prefix in lib.VARIABLE_NAMES:
      if p.startswith(prefix):
        return True
  return False


def is_operator(p):
  return p in ["~", "->", "<->", "and", "or", "forall", "exists"]


def rename_clauses(exp, suffix):
  """Rename all the propositions, constants, variables in a clase.

  Args:
    exp: the expression to rename. It can be a clause, list of clauses,
      an InferenceRule or an InferenceProblem.
    suffix: the suffix to add to all the names.

  Returns:
    A copy of "exp" with the renaming applied.
  """

  if isinstance(exp, list):
    return [rename_clauses(x, suffix) for x in exp]

  elif isinstance(exp, str):
    if exp in lib.ALL_RULE_NAMES:
      return exp
    if is_operator(exp):
      return exp
    if "_" in exp:
      exp = exp[:exp.index("_")]
    return exp + f"_{suffix}"

  elif isinstance(exp, lib.InferenceRule):
    return lib.InferenceRule(rename_clauses(exp.premises, suffix),
                             rename_clauses(exp.inferences, suffix),
                             rename_clauses(exp.contradictions, suffix),
                             rename_clauses(exp.unrelated, suffix),
                             rename_clauses(exp.propositions, suffix),
                             exp.rule_name)

  elif isinstance(exp, lib.InferenceProblem):
    return lib.InferenceProblem(rename_clauses(exp.premises, suffix),
                                rename_clauses(exp.inferences, suffix),
                                rename_clauses(exp.contradictions, suffix),
                                rename_clauses(exp.unrelated, suffix),
                                rename_clauses(exp.propositions, suffix),
                                exp.contains_contradiction)

  else:
    raise ValueError(f"rename_clauses with unsupported exp: {exp}")


def unify_clause_with_rule(premise, rule):
  for inference in rule.inferences:
    bindings = unify_clauses(premise, inference)
    if bindings is not None:
      return bindings

  return None


def unify_clauses(p1, p2, bindings=None, allow_p2_bindings=True):
  """Tries to unify p1 with p2, and returns the resulting bindings.

  Instead of full unification, we will use a set of predefined rules, that are
  enough for dataset generation. This is for simplicity, and also to control
  which chains are generated, and prevent some results that, while valid,
  would result in overly complex inferences (e.g., binding "p" to "r->q", for
  example).

  Args:
    p1: the first clause.
    p2: the second clause.
    bindings: any already established bindings.
    allow_p2_bindings: if this is False, only matches that do not require
    binding any proposition/functor/constant from p2 are allowed.

  Returns:
    The bindings necessary to unify p1 and p2 (this is an extension of the input
    "bindings").
  """

  if bindings is None:
    bindings = {}

  # match primitive propositions:
  if len(p1) == 1 and len(p2) == 1:
    if is_base_proposition(p1[0]) and is_base_proposition(p2[0]):
      if ((p1[0] not in bindings) or
          p2 == bindings[p1[0]] or p2 == [bindings[p1[0]]]):
        bindings1 = bindings.copy()
        bindings1[p1[0]] = p2
        return bindings1

  # ->, <->, and, or:
  if len(p1) == 3 and len(p2) == 3 and p1[0] == p2[0]:
    bindings1 = unify_clauses(p1[1], p2[1], bindings)
    if bindings1 is not None:
      bindings2 = unify_clauses(p1[2], p2[2], bindings1)
      if bindings2 is not None:
        return bindings2

  # ~:
  if len(p1) == 2 and len(p2) == 2 and p1[0] == "~" and p2[0] == "~":
    bindings1 = unify_clauses(p1[1], p2[1], bindings)
    if bindings1 is not None:
      return bindings1

  # match a "p" with a "~p":
  if len(p1) == 1 and len(p2) == 2 and p2[0] == "~":
    if is_base_proposition(p1[0]):
      if (p1[0] not in bindings) or p2 == bindings[p1[0]]:
        bindings1 = bindings.copy()
        bindings1[p1[0]] = p2
        return bindings1

  # match propositions to P(c):
  if (len(p1) == 1 and is_base_proposition(p1[0]) and
      len(p2) == 2 and is_functor(p2[0]) and is_constant(p2[1])):
    if (p1[0] not in bindings) or p2[0] == bindings[p1[0]]:
      bindings1 = bindings.copy()
      bindings1[p1[0]] = p2
      return bindings1

  # match functors with constants/variables:
  if (len(p1) == 2 and is_functor(p1[0]) and
      len(p2) == 2 and is_functor(p2[0])):
    if ((is_constant(p1[1]) and is_constant(p2[1])) or
        (is_variable(p1[1]) and is_variable(p2[1]))):
      if (p1[0] not in bindings) or p2[0] == bindings[p1[0]]:
        bindings1 = bindings.copy()
        bindings1[p1[0]] = p2[0]
        if (p1[1] not in bindings) or p2[1] == bindings[p1[1]]:
          bindings1[p1[1]] = p2[1]
          return bindings1

  # match "->"", "<->", "or", "and", "~" to P(c):
  if ((p1[0] in ["->", "<->", "or", "and", "~"]) and
      is_functor(p2[0]) and
      is_constant(p2[1]) and
      allow_p2_bindings):
    ground_propositions = []
    ground = True
    for p in p1[1:]:
      if len(p) == 1 and is_base_proposition(p[0]):
        if p[0] not in ground_propositions:
          ground_propositions.append(p[0])
      else:
        ground = False
        break

    if ground:
      # map each ground proposition in p1 to a new functor with the same
      # the same constant:
      bindings1 = bindings.copy()
      new_functor_names = ["R", "S"]  # These are never used in the rules above
      new_expression = [p1[0]]
      failed = False
      for i in range(len(ground_propositions)):
        p = ground_propositions[i]
        new_name = rename_clauses(new_functor_names[i], lib.NEXT_RENAME_INDEX)
        new_expression.append([new_name])
        if (p not in bindings1) or [new_name, p2[1]] == bindings1[p]:
          bindings1[p] = [new_name, p2[1]]
        else:
          failed = True
          break

      if not failed:
        # map the p2 functor to the operation in p1:
        if (p2[0] not in bindings1) or new_expression == bindings1[p2[0]]:
          bindings1[p2[0]] = new_expression
          return bindings1

  # match "exists"/"forall":
  if (len(p1) == 3 and len(p2) == 3 and p1[0] == p2[0] and
      (p1[0] in ["exists", "forall"])):
    # match variable:
    if (p1[1] not in bindings) or p2[1] == bindings[p1[1]]:
      bindings1 = bindings.copy()
      bindings1[p1[1]] = p2[1]
      # match clause:
      bindings2 = unify_clauses(p1[2], p2[2], bindings1)
      if bindings2 is not None:
        return bindings2

  return None


def apply_bindings(bindings, exp):
  """Applies bindings to an expression.

  Args:
    bindings: a dictionary with the bindings to apply.
    exp: the expression to apply bindings to.

  Returns:
    A copy of "exp" with the bindings applied.
  """

  if not exp:
    return exp
  if isinstance(exp, list):
    if isinstance(exp[0], str):
      if len(exp) == 1 and is_base_proposition(exp[0]) and exp[0] in bindings:
        # base propositions special case:
        new_exp = bindings[exp[0]]
        if isinstance(new_exp, str):
          return [new_exp]
        return new_exp
      elif len(exp) == 2 and is_functor(exp[0]) and exp[0] in bindings:
        # functor special case:
        if isinstance(bindings[exp[0]], str):
          new_exp = [bindings[exp[0]], apply_bindings(bindings, exp[1])]
          return new_exp
        else:
          new_exp = list(bindings[exp[0]])
          for i in range(len(new_exp)):
            if (isinstance(new_exp[i], list) and len(new_exp[i]) == 1
                and is_functor(new_exp[i][0])):
              # Add the appropriate argument
              new_exp[i] = [new_exp[i][0], apply_bindings(bindings, exp[1])]
          return new_exp
      else:
        # general case:
        return [apply_bindings(bindings, x) for x in exp]
    else:
      return [apply_bindings(bindings, x) for x in exp]
  elif isinstance(exp, str):
    if exp in bindings:
      # base propositions special case:
      new_exp = bindings[exp]
      if (is_base_proposition(exp) and
          isinstance(new_exp, list) and len(new_exp) == 1 and
          is_base_proposition(new_exp[0])):
        return new_exp[0]
      return new_exp

  elif isinstance(exp, lib.InferenceRule):
    return lib.InferenceRule(
        apply_bindings(bindings, exp.premises),
        apply_bindings(bindings, exp.inferences),
        apply_bindings(bindings, exp.contradictions),
        apply_bindings(bindings, exp.unrelated),
        apply_bindings(bindings, exp.propositions),
        exp.rule_name)

  elif isinstance(exp, lib.InferenceProblem):
    return lib.InferenceProblem(
        apply_bindings(bindings, exp.premises),
        apply_bindings(bindings, exp.inferences),
        apply_bindings(bindings, exp.contradictions),
        apply_bindings(bindings, exp.unrelated),
        apply_bindings(bindings, exp.propositions),
        exp.contains_contradiction)

  return exp


def premises_contradict_rule(premises, rule):
  """Checks if a set of premises contradict a rule.

  For example, the premises are p, p->q, and ~q would contradict the modus
  ponens rule.

  Args:
    premises: the premises to check.
    rule: the rule to check.

  Returns:
    Whether the premises contradict the rule (a boolean).
  """

  in_contradiction = []
  bindings = {}
  for rule_premise in rule.premises:
    match = False
    for premise in premises:
      bindings2 = unify_clauses(rule_premise, premise, bindings)
      if bindings2 is not None:
        bindings = bindings2
        match = True
        in_contradiction.append(premise)
        break
    if not match:
      return False

  for rule_contradiction in rule.contradictions:
    rule_contradiction2 = apply_bindings(bindings, rule_contradiction)
    if rule_contradiction2 in premises:
      in_contradiction.append(rule_contradiction2)
      return in_contradiction
  return False


def detect_contradiction(premises, check_rules=True):
  """Tries to identify contradictions in the premises of a problem.

  This is done in a simple way (and might not detect all the contradictions),
  but hopefully it eliminates most of them. Specifically, this method just looks
  at the following things:
  (1) if there is any rule for which both the premises and one of its
      contradictions appear in the problem premises.
  (2) the presence of "p" and "~p"
  (3) the presence of "p" and "~p" considering quantification operators, e.g.
      "forall x: P(x)" and "~P(c1)"
  Also, since we are doing no "occurs check" in the unification routine, it
  might eliminate some problems that are not contradictions. But that is ok, as
  it happens rarely.

  Args:
    premises: a list of propositions.
    check_rules: whether to check rules (step (1) above) or not.

  Returns:
    contradicting_premises: a list of contradicting premises, or False if there
                            is no contradiction.
  """

  if check_rules:
    for rule in lib.ALL_INFERENCE_RULES:
      # "*" is a suffix that is never used during generation, so, we guarantee
      # no conflicts.
      rule2 = rename_clauses(rule, "*")
      rule_contradiction = premises_contradict_rule(premises, rule2)
      if rule_contradiction:
        # Contradiction! (1)
        return rule_contradiction
  for p1 in premises:
    for p2 in premises:
      if p1 == ["~", p2]:
        # Contradiction! (2)
        return [p1, p2]
      if (isinstance(p1, list) and
          isinstance(p2, list) and
          p1[0] == "forall"):
        p1_sign = True
        p1_internal = p1[2]
        if isinstance(p1_internal, list) and p1_internal[0] == "~":
          p1_sign = False
          p1_internal = p1_internal[1]
        p2_sign = True
        p2_internal = p2
        if p2[0] == "forall":
          p2_internal = p2[2]
        if isinstance(p2_internal, list) and p2_internal[0] == "~":
          p2_sign = False
          p2_internal = p2_internal[1]
        if (p1_internal[0] == p2_internal[0] and
            p1_sign != p2_sign):
          # Contradiction! (3)
          return [p1, p2]

  return False


def detect_contradiction_in_problem(problem):
  """Returns a list of contradictory clauses from the problem.

  We try to detect if the problem in volves a contradiction by pooling
  together all the premises and all the inferences that the problem contains.

  Args:
    problem: an InferenceProblem.

  Returns:
    contradicting_premises: a list of contradicting premises, or False if there
                            is no contradiction.
  """

  premises = list(problem.premises)
  for inference in problem.inferences:
    for step in inference[1]:
      for proposition in step[0]:
        if proposition not in premises:
          premises.append(proposition)
  return detect_contradiction(premises)


def premises_match_rule(premises, rule, next_idx=0, bindings=None,
                        prevent_equal_bindings=True):
  """Determines if a rule can be fired given some premises.

  If all the premises of "rule" can be matched to a premise in "premises",
  this function calculates the bindings necessary for making those premises
  match. It returns a list of all the possible ways in which the rule matches.
  So, if the returned list is empty, the rule does not match the premises.

  Args:
    premises: the premises.
    rule: the rule to check.
    next_idx: the index of the next premise in "premises" to check.
    bindings: the current set of variable bindings.
    prevent_equal_bindings: if this is true, this function will prevent
      returning matches where two separate propositions in a rule match to the
      same in the premises, e.g. if bindings['p'] == bindings['q']. This is to
      prevent trivial rule firings that result in inferences like: 'p' implies
      'p and p'. However, notice that this might prevent some matches.

  Returns:
    A list where every element is a "bindings" dictionary, with the bindings
    necessary to match the rule to a subset of premises. The length of this list
    corresponds to the number of different ways in which the rule matches the
    premises.
  """

  def bindings_contain_equal_binding(bindings):
    for p1 in bindings:
      for p2 in bindings:
        if p1 == p2:
          continue
        if bindings[p1] == bindings[p2]:
          return True
    return False

  if bindings is None:
    bindings = {}
  if next_idx < len(rule.premises):
    results = []
    rule_premise = rule.premises[next_idx]
    for premise in premises:
      bindings2 = unify_clauses(rule_premise, premise, bindings,
                                allow_p2_bindings=False)
      if ((bindings2 is not None) and
          ((not prevent_equal_bindings) or
           (not bindings_contain_equal_binding(bindings2)))):
        results2 = premises_match_rule(premises, rule, next_idx + 1,
                                       bindings2, prevent_equal_bindings)
        results += results2
    return results
  else:
    return [bindings]


def split_compound_propositions(propositions):
  """Makes sure the list of propositions only contains atoms.

  Some times, after applying bindings, a proposition turns into something like
  "~p". We only want primitive propositions in the list of propositions. This
  function will locate all of those non-atomic propositions, remove them from
  the list of propositions, and add all the primitive propositions in them that
  were not there before.

  Args:
    propositions: the list of propositions to check.

  Returns:
    The same propositions list, but where all compound propositions have been
    split into atomic ones.
  """
  compound_propositions = []

  new_propositions = []
  for p in propositions:
    if isinstance(p, list) and (p[0] in ["->", "<->", "or", "and", "~"]):
      compound_propositions.append(p)
      for p2 in p[1:]:
        if p2 not in propositions and p2 not in new_propositions:
          new_propositions.append(p2)
    elif isinstance(p, list) and len(p) == 1:
      compound_propositions.append(p)
      if p[0] not in propositions and p[0] not in new_propositions:
        new_propositions.append(p[0])
  for p in compound_propositions:
    propositions.remove(p)
  for p in new_propositions:
    propositions.append(p)
  if compound_propositions or new_propositions:
    split_compound_propositions(propositions)


def rules_matching_premises(premises):
  """Returns the set of rules that can be fired given some premises.

  The set of candidate rules is the global variable "ALL_INFERENCE_RULES".

  Args:
    premises: the premises to check.

  Returns:
    The list of rules that fire. If any rule matches in more than one way, it
    will appear more than once in this list, with each copy having the
    corresponding instantiation to match the given premises.
  """

  l = []
  for rule in lib.ALL_INFERENCE_RULES:
    results = premises_match_rule(premises, rule)
    for bindings in results:
      rule2 = apply_bindings(bindings, rule)
      split_compound_propositions(rule2.propositions)
      l.append(rule2)
  return l


def one_step_inferences_from_premises(premises,
                                      rules_to_ignore=None):
  """Returns all the one-step inferences that can be reached from the premises.

  Specifically, this function uses the "rules_matching_premises" function above,
  to get the set of rules that would trigger from "premises", and returns all
  the inferences those rules would make.

  Args:
    premises: the set of premises to check.
    rules_to_ignore: a list of names of rules we do not want to consider.

  Returns:
    The list of clauses that can be inferred from the premises with a single
    rule application.
  """

  inferences = []
  l = rules_matching_premises(premises)
  for rule in l:
    add_inferences = True
    if rules_to_ignore:
      for name in rules_to_ignore:
        if name in rule.rule_name:
          add_inferences = False
          break
    if add_inferences:
      for inference in rule.inferences:
        inferences.append([inference, rule])
  return inferences
