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

"""Utilities for running the self-consistency algorithm."""

from concurrent import futures
import dataclasses
import re
from typing import Any
import pandas as pd
from cisc.src import confidence_extraction
from cisc.src.datasets import dataset as dataset_lib
from cisc.src.runners import runner as runner_lib


@dataclasses.dataclass
class SelfConsistencyResult:
  """The result of running the self-consistency algorithm on a single question."""

  question_id: str
  prompt: str  # The orginal prompt.
  temperature: float  # The temperature used for the model.

  @dataclasses.dataclass()
  class Trace:
    """A single attempt to answer the question."""

    prompt: str  # The orginal prompt. Duplicated here for convenience.
    response: str | None  # Might be None if an exception was raised.
    exception: str | None  # Optional exception if the generation failed.

    # The answer to the question. Might be None if couldn't extract an answer.
    answer: str | None
    # Answer span in `response``. E,g, if the reponse is "answer A. bla bla",
    # and the answer is "A", then the span would be [7, 8]. Note that the span
    # is in the response and not including the original prompt.
    answer_span: tuple[int, int] | None
    # The confidence extracted for this trace.
    confidence: confidence_extraction.Confidence | None = None

  traces: list[Trace]

  # Optional
  golden_label: str | None = None
  original_row: Any = None  # The original row from the dataset.


Trace = SelfConsistencyResult.Trace


def results_to_dataframe(
    self_consistency_results,
):
  """Converts a list of `SelfConsistencyResult` to a `pd.DataFrame`.

  Args:
    self_consistency_results: A list of `SelfConsistencyResult`.

  Returns:
    Dataframe with a row per trace.
  """
  df = pd.DataFrame(self_consistency_results)
  # Create row per trace and check each trace's answer.
  df = df.explode("traces").reset_index()

  # Flatten the trace into columns so it would be easier to debug.
  def flatten_trace(trace):
    """trace is a dict representation of a `SelfConsistencyResult.Trace`."""
    has_conf = pd.notna(trace["confidence"])
    return pd.Series([
        trace["response"],
        trace["exception"],
        trace["answer"],
        trace["confidence"]["verbal_conf"] if has_conf else None,
        trace["confidence"]["confidence_likelihoods"] if has_conf else None,
        trace["confidence"]["response_probability"] if has_conf else None,
    ])

  df[[
      "response",
      "exception",
      "answer",
      "verbal_confidence",
      "confidence_likelihoods",
      "response_probability",
  ]] = df.traces.apply(flatten_trace)

  normalize_str = lambda s: "" if s is None else re.sub(r"\W", "", s)
  df["is_correct"] = df.apply(
      lambda row: normalize_str(row.answer) == normalize_str(row.golden_label),
      axis=1,
  )
  return df


def run_self_consistency(
    runner,
    question_id,
    prompt,
    temp,
    num_tokens,
    num_traces,
    dataset,
):
  """Runs the self-consistency algorithm on a single question.

  Args:
    runner: the runner to use for querying the model.
    question_id: the question id.
    prompt: the prompt to use for querying the model.
    temp: the temperature to use for querying the model.
    num_tokens: the number of tokens to use for querying the model.
    num_traces: the number of traces to run. Each would be a different call to
      the model.
    dataset: the dataset which includes the instructions on how to format the
      prompts and how to extract the answers.

  Returns:
    The result of running the self-consistency algorithm on a single question.
  """

  # We send the same prompt multiple times to the model. This is inefficent, as
  # it would encode the prompt multiple times, but it should be negligable in
  # our case, as the input is much shroter than the generation part. Consider
  # optimizing this further.
  prompts = [prompt] * num_traces
  responses = runner.generate(prompts, num_tokens, temp, enable_formatting=True)

  with futures.ThreadPoolExecutor(len(prompts)) as executor:

    def extract_trace_from_response(
        response,
    ):
      trace = Trace(
          prompt=response.prompt,
          response=response.response,
          exception=response.exception,
          answer=None,
          answer_span=None,
      )
      response_text = response.response
      if response_text is None:
        return trace
      ans, span = dataset.extract_answer(response_text)
      if (not ans) or (not span):
        return trace
      return Trace(
          prompt=response.prompt,
          response=response.response,
          exception=response.exception,
          answer=ans,
          answer_span=span,
      )

    traces = list(executor.map(extract_trace_from_response, responses))

  return SelfConsistencyResult(
      question_id=question_id,
      prompt=prompt,
      temperature=temp,
      traces=traces,
  )
