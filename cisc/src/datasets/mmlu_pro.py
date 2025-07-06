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

"""MMLU-pro dataset."""

import pandas as pd
from cisc.src.datasets import dataset
from cisc.src.datasets import prompt_util

_ds = None


def _cached_ds():
  """Huggingface version of the dataset."""
  import datasets as hf_datasets  # pylint: disable=g-import-not-at-top

  global _ds
  if _ds is None:
    _ds = hf_datasets.load_dataset(
        "TIGER-Lab/MMLU-Pro", split="test"
    ).to_pandas()
  return _ds




def _format_question_with_options(row):
  """Formats the question with the options."""
  # Decode options and add letters (A, B, C, ...)
  formatted_options = [
      f"({chr(65 + i)}): {str(opt, 'utf-8')}"
      for i, opt in enumerate(row["options"])
  ]
  question_with_options = (
      f"{row['question_no_choices']}\nOptions are:\n"
      + "\n".join(formatted_options)
  )
  return question_with_options


def _parse_df(ds):
  ds.columns = ds.columns.str.replace("^default/", "", regex=True)
  ds.rename(
      columns={"question": "question_no_choices", "answer": "golden_label"},
      inplace=True,
  )
  ds["question"] = ds.apply(_format_question_with_options, axis=1)
  return ds


def _get_instructions():
  """Returns the task's instructions."""
  return f"""You will be given a single-choice question. Answer the question by selecting the letter of the best fitting option.

{prompt_util.general_instructions()}

The answer MUST ALWAYS be the letter of one of the available options; it CANNOT be "None of the Above"."""


def get_final_answer(text):
  ans, span = prompt_util.get_final_answer(
      text,
      # Any letter from A to H as MMLU contains 10 options.
      match_part_pattern=r"((?:[A-J]\b))",
  )
  if ans is not None:
    ans = ans.upper()
  return ans, span


def get_dataset():
  """Returns the MMLU-pro dataset."""
  ds = _parse_df(_cached_ds())
  instructions = _get_instructions()
  ds["question_id"] = ds.index
  return dataset.Dataset(
      ds,
      instructions,
      get_final_answer,
  )
