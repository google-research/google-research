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

"""Interface for running different models."""

import abc
from collections.abc import Sequence
import dataclasses
import numpy as np


@dataclasses.dataclass(frozen=True)
class GenerationOutput:
  """Holds the output of a single generation."""

  # The prompt as inputed to the model, after formatting. This might include
  # post-processing of the raw prompt (e.g. adding instructions tags).
  prompt: str

  response: str | None  # Might be None if an exception was raised.
  exception: str = ""  # Optional exception message.

  # Optional.The scores of the model.
  # Shape of [# tokens-in-response, # vocab-size].
  # Note that the scores are only for the *generated* tokens not the `prompt`.
  scores: np.ndarray | None = None

  # Optional. The embeddings from the last layer of the model.
  # Shape of [# tokens-in-prompt, # embeddings-size].
  # Note that the embeddings are only for tokens in the `prompt`.
  embeddings: np.ndarray | None = None


class Runner(abc.ABC):
  """Interface for running a model."""

  def generate(
      self,
      prompts,
      max_new_tokens,
      temperature,
      enable_formatting,
  ):
    """Generates a single response for each prompt."""
    raise NotImplementedError()

  def get_completion_likelihoods(
      self,
      prefixes,
      completions,
      enable_formatting,
  ):
    """For each prefix, returns the likelihood of each `completions`."""
    raise NotImplementedError()

  def get_normalized_probability_for_sequence(
      self,
      prefix,
      completion,
  ):
    """Return the normalized probability for all tokens in the sequence."""
    raise NotImplementedError()
