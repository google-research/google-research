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

"""A fake runner for testing."""

from typing import Sequence
from cisc.src.runners import runner as runner_lib


class FakeRunner(runner_lib.Runner):
  """A fake runner that returns the prompt as the response."""

  def __init__(self, fail=False):
    self._fail = fail

  def generate(
      self,
      prompts,
      max_new_tokens,
      temperature,
      enable_formatting = False,
  ):
    if self._fail:
      raise SystemError("Failed to generate")
    return [
        runner_lib.GenerationOutput(
            prompt=prompt,
            response=prompt + " response",
            exception="",
        )
        for prompt in prompts
    ]

  def get_completion_likelihoods(
      self,
      prefixes,
      completions,
      enable_formatting,
  ):
    assert (
        len(prefixes) == 1
    ), "For now fake runner only supports single prefix."
    return [[float(i) for i, _ in enumerate(completions)]]

  def get_normalized_probability_for_sequence(
      self,
      prefix,
      completion,
  ):
    """Return the normalized probability for all tokens in the sequence."""
    return 0.0
