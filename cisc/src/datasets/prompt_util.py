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

"""Utils for extracting answers and managing the dataset prompts.

These utils are placed in the datasets folder because they are specific to the
datasets we are using.
"""

import re


def format_answer_line(answer):
  return f"Proposed answer: ({answer})."


def general_instructions():
  """Basic instructions that can be used in all datasets."""

  return """Before giving your answer, provide a step-by-step explanation of your thought process.
Then on a new line, give your proposed answer adhering to this precise format: 'Proposed answer: (X).', where X is your proposed answer."""


def get_final_answer(
    text,
    match_part_pattern,
):
  """Extracts answers from patterns like: The proposed answer is: (C).

  Args:
    text: the text to extract the answer from.
    match_part_pattern: a regex pattern to match the answer part. For example,
      `(-?[0-9,.]+)` for matching numbers or [A-D] to much for matching multiple
      choice letter answers.

  Returns:
    A tuple of the extracted answer and the span of the answer in the text.
  """
  ignore = r"[\s\:\*\.\$\,]"

  def get_match(pattern):
    m = re.search(pattern, text, re.IGNORECASE)
    if not m or not m.group(1):
      return None
    return m

  # Patterns are matched in order.
  patterns = [
      # With parenthesis.
      f"proposed answer]?(?: is)?.?{ignore}*\\({ignore}*",
      # Without preciding proposed we require the "is" literal.
      f"answer{ignore}*is.?{ignore}*\\(",
      # Without parenthesis.
      f"proposed answer(?: is)?.?{ignore}*",
      # finally, the most permissive patterns.
      f"answer:{ignore}*\\(",
      "answer.{1,5}\\(",
      f"answer is{ignore}*",
  ]
  patterns = [p + match_part_pattern for p in patterns]

  for pattern in patterns:
    m = get_match(pattern)
    if m is not None:
      extracted_answer = re.sub(f"{ignore}+", "", m.group(1))
      return extracted_answer, m.span(1)
  return None, None


def build_prompt(
    instructions,
    question,
):
  return f"""{instructions}

### Question:

{question}

### Your Answer:
"""


def remove_non_alphanumeric(s):
  # Remove all non-alphanumeric characters.
  if s is None:
    return ""
  return "".join(c for c in s if c.isalnum())
