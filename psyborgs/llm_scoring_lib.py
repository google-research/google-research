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

"""Helper for running scoring using various LLMs services."""

from typing import Sequence, Union

import typing_extensions


class LanguageModelScoringFn(typing_extensions.Protocol):
  """Specifcation of LLMs scoring function."""

  def __call__(self, prompt,
               continuations):
    """Computes the score of continuations for a given prompt.

    Arguments:
      prompt: prompt to use as context
      continuations: continuations to score

    Returns:
      score: the probability normalized by length of each continuation
              (in regular scale / not in log scale)
    """


def score_with_dummy_llm(
    prompt, continuations
):
  """Retrieves the scores of a LLM - this function is a placeholder for actual calls."""
  del prompt
  return [0.0] * len(continuations)
