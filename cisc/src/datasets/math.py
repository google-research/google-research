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

"""Load the Hendrycks et al. (2021) Math dataset.

https://arxiv.org/pdf/2103.03874
"""

import re
import pandas as pd
from cisc.src.datasets import dataset
from cisc.src.datasets import math_util
from cisc.src.datasets import prompt_util

_ds = None


def _cached_ds():
  """Huggingface version of the dataset."""
  import datasets as hf_datasets  # pylint: disable=g-import-not-at-top

  global _ds
  if _ds is None:
    configs = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]
    _ds = pd.concat([
        hf_datasets.load_dataset(
            "EleutherAI/hendrycks_math", config, split="test"
        ).to_pandas()
        for config in configs
    ])
  return _ds




def _get_instructions():
  """Returns the task's instructions."""
  return f"""You will be given a question and your goal is to answer it correctly.
Your proposed answer should be a TeX expression, such as '$5$', '$3.14$', or '$\\sqrt{8}$'.

{prompt_util.general_instructions()}"""


def remove_suffix(text, suffix):
  if text.endswith(suffix):
    return text[: -len(suffix)]
  return text


def normalize_math(math_text):
  math_text = math_util.normalize_math(math_text)
  # Strip some additional characters that can be added due to prompt formatting.
  return remove_suffix(math_text, "</s>").strip(".*()\\t")


def extract_answer_from_last_box(text):
  r"""Extracts answer from \boxed Latex command.

  Args:
    text: An text string.

  Returns:
    Extracted answer string.
  """
  boxed_str = math_util.last_boxed_only_string(text)
  if not boxed_str:
    return ""
  # Strips \boxed command and cleans up the Latex.
  return normalize_math(boxed_str)


def get_final_normalized_answer(
    text,
):
  """Extracts the final answer from the text and normalize it.

  It is expected that the output is using the pattern
  "Proposed answer: (<answer>)."

  Args:
    text: The text to extract the answer from.

  Returns:
    The normalized answer.
  """

  patterns = [  # Ordered by match priority.
      r"Proposed answer:? \$(.*)\$",
      r"Proposed answer:? (.*)\)",
      r"Proposed answer:? (.*)\.",
      r"answer:? \$(.*)\$",
      r"answer:? (.*)\)",
      r"answer:? (.*)\.",
  ]
  for pattern in patterns:
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
      return normalize_math(match.group(1).strip()), match.span(1)
  return "", None


def get_dataset():
  """Loads the Math dataset; potentially filters to only int answers."""
  ds = _cached_ds()
  assert sorted(ds.columns.tolist()) == sorted(
      ["problem", "level", "solution", "type"]
  )
  ds = ds.rename(columns={"problem": "question"})
  ds["golden_label"] = ds.solution.apply(extract_answer_from_last_box)

  print(f"Total questions: {len(ds)}")

  instructions = _get_instructions()
  ds["question_id"] = ds.index
  return dataset.Dataset(
      ds,
      instructions,
      get_final_normalized_answer,
  )
