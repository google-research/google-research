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

"""Training set split generation.

Specifically, we support three types of training/test splits:
- IID: random split of examples in train/test.
- OOD: reandomly splitting InferenceProblems, and then generating train/test
  examples separately from each subset of InferenceProblems.
- length: splits examples by length (defined as the number of premises in the
  InferenceProblem used to generate the example).
"""


import example_generation as eg
import inference_problems as ip
import logic_inference_lib as lib


def generate_training_and_test_sets_iid(n_problems, n_variations, n_examples,
                                        train_ratio,
                                        example_types=None,
                                        length_distribution=None,
                                        answer_at_the_end=True):
  """Generates an IID split."""

  problems = ip.generate_multistep_problems(
      n_problems, length_distribution=length_distribution)
  actual_length_distribution = ip.problem_length_stats(problems)
  print(f"problem counts by chain length: {actual_length_distribution}")
  examples = eg.generate_examples_from_problems(
      problems, n_variations, n_examples,
      example_types=example_types,
      answer_at_the_end=answer_at_the_end)
  split_point = int(train_ratio * len(examples))
  examples_train = examples[:split_point]
  examples_test = examples[split_point:]
  return examples_train, examples_test


def generate_training_and_test_sets_ood(n_problems, n_variations, n_examples,
                                        train_ratio,
                                        example_types=None,
                                        length_distribution=None,
                                        answer_at_the_end=True):
  """Generates an OOD split."""

  def ensure_all_rules_appear_in_training(problems_train, problems_test):
    counts_train = ip.rules_used_in_problems(problems_train)
    counts_test = ip.rules_used_in_problems(problems_test)
    missing_rules = []
    for rule_name in lib.ALL_RULE_NAMES:
      if rule_name not in counts_train:
        counts_train[rule_name] = 0
      if rule_name not in counts_test:
        counts_test[rule_name] = 0
      if counts_test[rule_name] > 0:
        if counts_train[rule_name] <= 0:
          if rule_name not in missing_rules:
            missing_rules.append(rule_name)
    if not missing_rules:
      # We are fine!
      return
    print(f"missing: {missing_rules}")
    for rule_name in lib.ALL_RULE_NAMES:
      print(f"{rule_name}\t{counts_train[rule_name]}\t{counts_test[rule_name]}")
    raise ValueError("Some rule types appear in testing, but not in training!")

  problems = ip.generate_multistep_problems(
      n_problems, length_distribution=length_distribution)
  actual_length_distribution = ip.problem_length_stats(problems)
  print(f"problem counts by chain length: {actual_length_distribution}")
  # For the OOD setting, we split by the problem type early on, so that testing
  # has constructs, not seen during training:
  problems_split_point = int(train_ratio * len(problems))
  problems_train = problems[:problems_split_point]
  problems_test = problems[problems_split_point:]

  ensure_all_rules_appear_in_training(problems_train, problems_test)

  examples_train = eg.generate_examples_from_problems(
      problems_train, n_variations, int(n_examples*train_ratio),
      example_types=example_types,
      answer_at_the_end=answer_at_the_end)
  examples_test = eg.generate_examples_from_problems(
      problems_test, n_variations, int(n_examples*(1-train_ratio)),
      example_types=example_types,
      answer_at_the_end=answer_at_the_end)

  return examples_train, examples_test


def generate_training_and_test_sets_length(n_problems, n_variations, n_examples,
                                           length_threshold,
                                           example_types=None,
                                           length_distribution=None,
                                           answer_at_the_end=True):
  """Generates a length split."""

  problems = ip.generate_multistep_problems(
      n_problems, length_distribution=length_distribution)
  actual_length_distribution = ip.problem_length_stats(problems)
  print(f"problem counts by chain length: {actual_length_distribution}")
  examples = eg.generate_examples_from_problems(
      problems, n_variations, n_examples,
      example_types=example_types,
      answer_at_the_end=answer_at_the_end)
  examples_train = []
  examples_test = []
  for example in examples:
    if len(example.problem.premises) <= length_threshold:
      examples_train.append(example)
    else:
      examples_test.append(example)
  return examples_train, examples_test
