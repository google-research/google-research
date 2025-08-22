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

"""bbh dataset."""

import enum
import re

import pandas as pd

from cisc.src.datasets import dataset
from cisc.src.datasets import prompt_util

_ds = None


class QuestionType(enum.Enum):
  """The type of answer extraction to use."""

  MULTI_CHOICE = 'multi_choice'
  OTHER = 'other'


# we are not using the following tasks: "dyck_languages", "web_of_lies" and
# "word_sorting" because they require complex post-processing for answer
# extraction.
BBH_TASKS = {
    'boolean_expressions': QuestionType.OTHER,
    'causal_judgement': QuestionType.OTHER,
    'date_understanding': QuestionType.MULTI_CHOICE,
    'disambiguation_qa': QuestionType.MULTI_CHOICE,
    # 'dyck_languages',
    'formal_fallacies': QuestionType.OTHER,
    'geometric_shapes': QuestionType.MULTI_CHOICE,
    'hyperbaton': QuestionType.MULTI_CHOICE,
    'logical_deduction_five_objects': QuestionType.MULTI_CHOICE,
    'logical_deduction_seven_objects': QuestionType.MULTI_CHOICE,
    'logical_deduction_three_objects': QuestionType.MULTI_CHOICE,
    'movie_recommendation': QuestionType.MULTI_CHOICE,
    'multistep_arithmetic_two': QuestionType.OTHER,
    'navigate': QuestionType.OTHER,
    'object_counting': QuestionType.OTHER,
    'penguins_in_a_table': QuestionType.MULTI_CHOICE,
    'reasoning_about_colored_objects': QuestionType.MULTI_CHOICE,
    'ruin_names': QuestionType.MULTI_CHOICE,
    'salient_translation_error_detection': QuestionType.MULTI_CHOICE,
    'snarks': QuestionType.MULTI_CHOICE,
    'sports_understanding': QuestionType.OTHER,
    'temporal_sequences': QuestionType.MULTI_CHOICE,
    'tracking_shuffled_objects_five_objects': QuestionType.MULTI_CHOICE,
    'tracking_shuffled_objects_seven_objects': QuestionType.MULTI_CHOICE,
    'tracking_shuffled_objects_three_objects': QuestionType.MULTI_CHOICE,
    # 'web_of_lies',
    # 'word_sorting',
}


def _load_task_core(task_name):
  """Huggingface version of the dataset."""
  import datasets as hf_datasets  # pylint: disable=g-import-not-at-top

  return hf_datasets.load_dataset(
      'lukaemon/bbh', task_name, split='test'
  ).to_pandas()




def _load_task(task_name, task_type):
  """Load a single BBH task."""
  df = _load_task_core(task_name)
  df['task_name'] = task_name

  if task_type == QuestionType.MULTI_CHOICE:
    df['input'] = df.input.apply(
        lambda input: input
        + '\nSelect the letter of the best fitting option. The answer CANNOT be'
        ' "None of the Above"'
    )
  return df


def load_dataset(tasks_list):
  """Load BBH raw data from cns.

  Args:
    tasks_list: Which bbh tasks to load

  Returns:
    A dataframe containing the bbh data with the following columns:
    'question': question
    'golden_label': correct answer
    'task_name': name of the subtask
  """
  all_dfs = [
      _load_task(task_name, task_type)
      for task_name, task_type in tasks_list.items()
  ]
  combined_tasks_df = pd.concat(all_dfs, ignore_index=True)
  combined_tasks_df.rename(
      columns={'input': 'question', 'target': 'golden_label'}, inplace=True
  )
  return combined_tasks_df


def _cached_ds():
  global _ds
  if _ds is None:
    _ds = load_dataset(BBH_TASKS)
  return _ds


def _format_df(df):
  """Formats the dataframe."""
  df['golden_label'] = df.apply(
      lambda row: prompt_util.remove_non_alphanumeric(row['golden_label']),
      axis=1,
  )
  df['question_id'] = df.index
  return df


def _get_instructions():
  """Returns the task's instructions."""
  return f"""You will be given a question and your goal is to answer it correctly.

{prompt_util.general_instructions()}"""


def get_final_answer(text):
  """Extracts the final answer for all type of answers in BBH."""
  ans, span = prompt_util.get_final_answer(
      text,
      # BBH contains a variety of answers, including numbers, multi-choice
      # letter options, true/false, yes/no, valid/invalid. We have a single
      # pattern for all of them.
      match_part_pattern=(
          r'((?:[A-R]\b)|(?:-?\s*[0-9,]+)|true\b|false\b|yes\b|no\b|yes\b|no\b|valid\b|invalid\b|not.plausible\b|not\b|plausible\b)'
      ),
  )
  if ans is not None:
    # The ground truth answer does not contain the 'NOT' or 'PLAUSIBLE'
    # keywords, but we observed that the model sometimes adds them, so we
    # covert them as best effort.
    ans = ans.upper()
    ans = re.sub(r'NOT.?PLAUSIBLE', 'NO', ans)
    ans = re.sub(r'NOT', 'NO', ans)
    ans = re.sub(r'PLAUSIBLE', 'YES', ans)
  return ans, span


def get_dataset():
  """Returns the bbh dataset."""
  ds = _cached_ds()
  ds = _format_df(ds)
  instructions = _get_instructions()

  return dataset.Dataset(
      ds,
      instructions,
      get_final_answer,
  )
