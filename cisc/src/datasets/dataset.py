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

"""Simple interface for loading datasets."""

import abc
from typing import Callable
import pandas as pd
from cisc.src.datasets import prompt_util


def decode_if_bytes(ds):
  """Decodes the bytes columns to strings."""
  for col, _ in ds.dtypes.items():
    ds[col] = ds[col].apply(
        lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
    )
  return ds


class Dataset(abc.ABC):
  """Simple interface for loading datasets."""

  data: pd.DataFrame

  def __init__(
      self,
      data,
      instructions,
      extract_answer_func,
      upper_case_answer_and_golden_label = True,
  ):
    """Constructs the dataset. Fails if the data is invalid.

    Args:
      data: the dataset dataframe.
      instructions: the instructions for the specific dataset.
      extract_answer_func: a function to extract the answer from the model
        output.
      upper_case_answer_and_golden_label: if true, converts the answer and
        golden_label to upper case.
    """
    assert data is not None
    if not data.index.is_unique:
      raise ValueError("data must have unique indices")
    if "question" not in data.columns:
      raise ValueError("data must contain a column named 'question'")
    if "question_id" not in data.columns:
      raise ValueError("data must contain a column named 'question_id'")
    if "golden_label" not in data.columns:
      raise ValueError("data must contain a column named 'golden_label'")
    self.data = data
    self.upper_case_answer_and_golden_label = upper_case_answer_and_golden_label
    if self.upper_case_answer_and_golden_label:
      self.data["golden_label"] = self.data["golden_label"].apply(
          lambda x: x.upper()
      )
    self.instructions = instructions
    self.extract_answer_func = extract_answer_func

  def get_instructions(self):
    return self.instructions

  def format_question(self, question):
    return prompt_util.build_prompt(
        instructions=self.instructions,
        question=question,
    )

  def extract_answer(
      self, text
  ):
    """Extracts the answer from the text and return it and its span."""
    answer, span = self.extract_answer_func(text)
    if self.upper_case_answer_and_golden_label and answer is not None:
      answer = answer.upper()
    return answer, span
