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

"""Augment dataset with backward reasoning."""

import argparse
import json

from prompt import consistency_check_prompt_math
from prompt import consistency_check_prompt_mcq
from prompt import gen_reasoning_prompt
from prompt import icl_samples
from prompt import prompt_for_backward_question
import tqdm
from utils import get_alphabet_choice
from utils import get_gemini_output
from utils import get_true_false
from utils import get_yes_no
from utils import parse_math_boxed
from utils import parse_number
from utils import remove_backward_answer


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--task", default="SQA", type=str)
  args = parser.parse_args()

  with open(f"./training_data/{args.task}.json", "r") as f:
    train_samples = json.load(f)

  is_math = False
  if args.task == "SQA":
    answer_extraction = get_yes_no
    consistency_check_prompt = consistency_check_prompt_mcq
  elif args.task in ["ANLI", "ARC", "Date", "CSQA", "ESNLI"]:
    answer_extraction = get_alphabet_choice
    consistency_check_prompt = consistency_check_prompt_mcq
  elif args.task in ["GSM8K", "GSM8K-Rev"]:
    answer_extraction = parse_number
    consistency_check_prompt = consistency_check_prompt_math
    is_math = True
  elif args.task in ["TabMWP", "MATH"]:
    answer_extraction = parse_math_boxed
    consistency_check_prompt = consistency_check_prompt_math
    is_math = True
  else:
    raise ValueError(f"Unsupported task: {args.task}")

  # backward question generation
  print("Generating backward question...")
  results = []
  for sample in tqdm(train_samples[len(results):]):
    try:
      tmp = {}
      tmp["question"] = sample["question"]
      tmp["gold_answer"] = sample["gold_answer"]

      q = f"{sample['question']} The correct answer is {sample['gold_answer']}."

      prompt = prompt_for_backward_question.format(
          icl_samples=icl_samples[args.task],
          input_question=q)
      backward_question = get_gemini_output(prompt, model="pro")
      tmp["backward_question"] = remove_backward_answer(backward_question)
      results.append(tmp)
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"error in backward question generation: {e}")
      continue

  # forward reasoning generation
  print("Generating forward reasoning...")
  for i in tqdm.tqdm(results):
    try:
      prompt = i["question"] + gen_reasoning_prompt[args.task]
      forward_reasoning = get_gemini_output(prompt, model="pro")
      i["forward_reasoning"] = forward_reasoning
      i["forward_pred"] = answer_extraction(forward_reasoning)
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"error in forward reasoning generation: {e}")
      continue

    # backward reasoning generation
    print("Generating backward reasoning...")
    for i in tqdm.tqdm(results):
      try:
        prompt = i["backward_question"] + gen_reasoning_prompt[args.task]
        backward_reasoning = get_gemini_output(prompt, model="pro")
        i["backward_reasoning"] = backward_reasoning
        i["backward_pred"] = answer_extraction(backward_reasoning)
      except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"error in backward reasoning generation: {e}")
        continue

    # Consistency check
    print("Validating the augmented data...")
    for i in tqdm.tqdm(results):
      try:
        prompt = consistency_check_prompt.format(
            question=i["question"],
            gold_answer=i["gold_answer"],
            backward_question=i["backward_question"],
            backward_pred=i["backward_pred"]
        )
        consistency = get_gemini_output(prompt, model="pro")
        i["consistency_reasoning"] = consistency
        i["is_consistent"] = get_true_false(consistency)
      except Exception as e:  # pylint: disable=broad-exception-caught
        print(e)
        i["consistency_reasoning"] = "N/A"
        i["is_consistent"] = "false"

    with open(f"./training_data/{args.task}.json", "w") as f:
      json.dump(results, f)
