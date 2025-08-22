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

"""GSM8K dataset."""

import pandas as pd
from cisc.src.datasets import dataset
from cisc.src.datasets import prompt_util

_ds = None


def _cached_ds():
  """Huggingface version of the dataset."""
  import datasets as hf_datasets  # pylint: disable=g-import-not-at-top

  global _ds
  if _ds is None:
    _ds = hf_datasets.load_dataset("gsm8k", "main", split="test").to_pandas()
    assert all(_ds.answer.str.contains("####"))
    _ds["short_answer"] = _ds["answer"].str.split("####").str[-1].str.strip()
  return _ds




def _get_instructions():
  """Returns the task's instructions."""
  return f"""You will be given a question and your goal is to answer it correctly.

{prompt_util.general_instructions()}"""


def get_final_answer(text):
  ans, span = prompt_util.get_final_answer(
      text,
      # Only ints.
      match_part_pattern=r"((?:-?\s*[0-9,]+))",
  )
  if ans is not None:
    ans = ans.replace(",", "")
  return ans, span


def get_dataset():
  """Returns the GSM8K dataset."""
  ds = _cached_ds()
  ds["golden_label"] = ds.short_answer.apply(lambda x: x.replace(",", ""))

  instructions = _get_instructions()
  ds["question_id"] = ds.index
  return dataset.Dataset(
      ds,
      instructions,
      get_final_answer,
  )
