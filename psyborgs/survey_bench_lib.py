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

"""Library that administers social science survey instruments to LLMs."""

import dataclasses
import enum
import json

from typing import Dict, Iterator, List, Optional, Tuple

import dacite
import pandas as pd


from psyborgs import llm_scoring_lib


# build data classes w/ `dacite`
# data classes for Administration Sessions
@dataclasses.dataclass(frozen=True)
class Scale:
  user_readable_name: str
  item_ids: List[str]
  reverse_keyed_item_ids: List[str]
  item_preamble_ids: List[str]
  item_postamble_ids: List[str]
  response_scale_ids: List[str]
  response_choice_postamble_ids: List[str]


@dataclasses.dataclass(frozen=True)
class MeasureSpecification:
  user_readable_name: str
  items: Dict[str, str]
  scales: Dict[str, Scale]


@dataclasses.dataclass(frozen=True)
class ResponseScale:
  user_readable_name: str
  response_choices: Dict[str, str]


class ModelFamily(str, enum.Enum):
  OTHER = 'Other'


@dataclasses.dataclass(frozen=True)
class ModelSpec:
  """Represents LLM scoring functions by name, endpoint, and LLM family."""
  user_readable_name: str
  model_family: ModelFamily
  model_endpoint: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class AdministrationSession:
  """Representation of all measures and options fed to specified models."""
  measures: Dict[str, MeasureSpecification]
  item_preambles: Dict[str, str]
  item_postambles: Dict[str, str]
  response_scales: Dict[str, ResponseScale]
  response_choice_postambles: Dict[str, str]
  models: Dict[str, ModelSpec]

  @property
  def n_measures(self):
    """Returns number of measures in an `AdministrationSession`."""
    return len(self.measures)


# data classes for iterative scoring
@dataclasses.dataclass(frozen=True)
class NamedEntry:
  entry_id: str
  text: str


@dataclasses.dataclass(frozen=True)
class Measure:
  measure_id: str
  measure: MeasureSpecification
  scale_id: str
  scale: Scale


@dataclasses.dataclass(frozen=True)
class Prompt:
  preamble: NamedEntry
  item: NamedEntry
  postamble: NamedEntry

  @property
  def payload(self):
    return ''.join([self.preamble.text, self.item.text, self.postamble.text])


@dataclasses.dataclass(frozen=True)
class Continuation:
  response_value: int
  response_scale_id: str
  response_choice: NamedEntry
  response_choice_postamble: NamedEntry

  @property
  def payload(self):
    return ''.join([self.response_choice.text,
                    self.response_choice_postamble.text])


@dataclasses.dataclass(frozen=True)
class PayloadSpec():
  """Stores the exact payload specification used to produce a score."""
  prompt_text: str
  continuation_text: str
  score: float
  measure_id: str
  measure_name: str
  scale_id: str
  item_preamble_id: str
  item_id: str
  item_postamble_id: str
  response_scale_id: str
  response_value: int
  response_choice: str
  response_choice_postamble_id: str
  model_id: str


def load_admin_session(json_path):
  """Builds session data classes from user-friendly JSON.

  Args:
    json_path: Path of the AdministrationSession in .json format.

  Returns:
    The Admin Session object.
  """
  json_file = open(json_path)
  admin_session_dict = json.load(json_file)

  # dacite documentation on casting input values to objects can be found here:
  # https://github.com/konradhalas/dacite#casting
  admin_session = dacite.from_dict(data_class=AdministrationSession,
                                   data=admin_session_dict,
                                   config=dacite.Config(cast=[enum.Enum]))
  return admin_session


def create_llm_scoring_fn(
    model_spec):
  """Creates a LanguageModelScoringFn instance given a model specification."""
  # this is currently a dummy function to be enhanced with real world models.
  del model_spec
  return llm_scoring_lib.score_with_dummy_llm


def create_llm_scoring_fns_dict(
    admin_session
):
  """Creates a dict of LLM scoring functions easily accessible by `model_id`.

  Args:
    admin_session: An `AdministrationSession` containing a specification of
      desired survey measures, item framing options, and LLM scoring functions.

  Returns:
    A dict of LLM scoring functions.
  """

  return {
      model_id: create_llm_scoring_fn(model_spec)
      for model_id, model_spec in admin_session.models.items()
  }


def assemble_payload(prompt,
                     continuation
                     ):
  """Forms a scorable prompt, continuation from an item and framing options."""
  return prompt.payload, continuation.payload


def generate_payload_spec(measure,
                          prompt,
                          continuation,
                          score,
                          model_id):
  """Returns all info needed to reproduce a single prompt-continuation score.

  Generates exact specification used (measure name; IDs for scale, item,
    preamble, postambles, response scale, model; response value) by unpacking
    prompt and continuation objects into their constituent parts.

  Args:
    measure: A `Measure` iterator that specifies one combination of a measure
      and a scale to be scored.
    prompt: A `Prompt` iterator that specifes one combination of an item
      preamble, item text, and item postamble to be scored.
    continuation: A `Continuation` that specifies one combination of a response
      scale, response choice and integer value, and response choice postamble.
    score: A float indicating the LLM score for the current prompt-continuation
      combination.
    model_id: A string indicating the LLM scoring function used to calculate the
      score.
  """
  # rehydrate full prompt and continuation strings
  prompt_text, continuation_text = assemble_payload(prompt,
                                                    continuation)

  # rehydrate measure specification from measure
  measure_id = measure.measure_id
  measure_name = measure.measure.user_readable_name
  scale_id = measure.scale_id

  # rehydrate prompt specification from prompt
  item_preamble_id = prompt.preamble.entry_id
  item_id = prompt.item.entry_id
  item_postamble_id = prompt.postamble.entry_id

  # rehydrate continuation specification from continuation
  response_scale_id = continuation.response_scale_id
  response_value = continuation.response_value
  response_choice = continuation.response_choice.text
  response_choice_postamble_id = continuation.response_choice_postamble.entry_id

  # save full payload specification info as a PayloadSpec
  return PayloadSpec(
      prompt_text, continuation_text, score, measure_id, measure_name, scale_id,
      item_preamble_id, item_id, item_postamble_id, response_scale_id,
      response_value, response_choice, response_choice_postamble_id, model_id
  )


def print_payload_specification(payload_spec):
  """Prints payload (item + options, etc.) specification for debugging."""
  print(f'prompt: \'{payload_spec.prompt_text}\'')
  print(f'continuation: \'{payload_spec.continuation_text}\'')

  print(f'score: {payload_spec.score}; '
        f'measure_id: {payload_spec.measure_id}; '
        f'measure_name: {payload_spec.measure_name}; '
        f'scale_id: {payload_spec.scale_id}; '
        f'item_preamble_id: {payload_spec.item_preamble_id}; '
        f'item_id: {payload_spec.item_id}; '
        f'item_postamble_id: {payload_spec.item_postamble_id}; '
        f'response_scale_id: {payload_spec.response_scale_id}; '
        f'response_value: {payload_spec.response_value}; '
        f'response_choice: {payload_spec.response_choice}; '
        f'response_choice_postamble_id: {payload_spec.response_choice_postamble_id}; '  # pylint: disable=line-too-long
        f'model_id: {payload_spec.model_id}')


def assemble_and_score_payload(
    measure,
    prompt,
    continuation,
    model_scoring_fn,
    model_id,
    verbose = False):
  """Assembles and scores a payload given an LLM scoring function."""
  # assemble item-choice and options payload
  prompt_text, continuation_text = assemble_payload(prompt, continuation)

  # score payload
  (score,) = model_scoring_fn(prompt_text, continuation_text)

  # generate payload_specification
  payload_spec = generate_payload_spec(measure, prompt, continuation, score,
                                       model_id)

  if verbose:
    # view payload and specification if verbose
    print_payload_specification(payload_spec)

  # append result to raw scores list
  return payload_spec


def continuation_generator(
    measure,
    admin_session):
  """Iterates through all specified continuations."""
  # retrieve response scales based on ID
  for response_scale_id in measure.scale.response_scale_ids:
    response_scale = admin_session.response_scales[response_scale_id]

    # retrieve response choices
    for response_choice_id, response_choice in response_scale.response_choices.items():  # pylint: disable=line-too-long

      # retrieve response choice postambles based on ID
      for response_choice_postamble_id in measure.scale.response_choice_postamble_ids:  # pylint: disable=line-too-long
        yield Continuation(
            response_value=int(response_choice_id),
            response_scale_id=response_scale_id,
            response_choice=NamedEntry(
                entry_id=response_choice_id,
                text=response_choice,
            ),
            response_choice_postamble=NamedEntry(
                entry_id=response_choice_postamble_id,
                text=admin_session
                .response_choice_postambles[response_choice_postamble_id]),
        )


def prompt_generator(
    measure,
    admin_session):
  for item_preamble_id in measure.scale.item_preamble_ids:
    for item_id in measure.scale.item_ids:
      for item_postamble_id in measure.scale.item_postamble_ids:
        yield Prompt(
            preamble=NamedEntry(
                entry_id=item_preamble_id,
                text=admin_session.item_preambles[item_preamble_id]),
            item=NamedEntry(
                entry_id=item_id,
                text=measure.measure.items[item_id]),
            postamble=NamedEntry(
                entry_id=item_postamble_id,
                text=admin_session.item_postambles[item_postamble_id]),
        )


def measure_generator(
    admin_session):
  """Iterates through all specified measures and scales."""
  measures = admin_session.measures

  for measure_id, measure in measures.items():
    for scale_id, scale in measure.scales.items():
      yield Measure(
          measure_id=measure_id,
          measure=measure,
          scale_id=scale_id,
          scale=scale
      )


def administer_session_serially(
    admin_session,
    verbose = False):
  """Administers specified survey items to LLMs and returns raw LLM scores.

  This key function (serial version) 'administers' a battery of survey measures
    to various LLMs specified within an `AdministrationSession` object. Since
    items (e.g., 'I like ice cream') within a measure can be presented to LLMs
    in a variety of ways, each item is administered multiple times across an
    assortment of compatible framing options and standardized response choices
    derived from response scales.

    Framing options within an `AdministrationSession` consist of item preambles
    (e.g., 'With regards to the following statement, "'), item postambles
    (e.g., '", I tend to '), and response choice postambles (e.g., '.').

    Prompts and continuations are assembled in the following format:

    Prompt:
    {item preamble} {item} {item postamble}

    Continuation:
    {response choice} {response choice postamble}

  Args:
    admin_session: An `AdministrationSession` containing a specification of
      desired survey measures, item framing options, and LLM scoring functions.
    verbose: If True, output is printed for debugging.

  Returns:
    A Pandas DataFrame containing raw LLM scores for each item-response choice
      pair and specification information needed to reproduce the score.
  """
  # create dict of LLM scoring functions for reuse
  llm_scoring_fns = create_llm_scoring_fns_dict(admin_session)

  # for efficiency, accumulate raw score data for each item + response choice +
  # options combination in a list, then this list to a pd.DataFrame at the end
  # of the loop
  raw_response_scores_list = []

  # iterate through all measures and scale combinations
  for measure_object in measure_generator(admin_session):

    # iterate through all prompt combinations
    for prompt_object in prompt_generator(measure_object, admin_session):

      # iterate through all continuation combinations
      for continuation_object in continuation_generator(measure_object,
                                                        admin_session):

        # iterate through LLM scoring functions to use (this is done here to
        # preempt potential RPC rate limits)
        for model_id, model_scoring_fn in llm_scoring_fns.items():

          # assemble and score payload
          raw_score = assemble_and_score_payload(measure_object, prompt_object,
                                                 continuation_object,
                                                 model_scoring_fn, model_id,
                                                 verbose)

          # append single score + specification info
          raw_response_scores_list.append(raw_score)

  # convert raw scores list into pd.DataFrame
  raw_response_scores_df = pd.DataFrame(raw_response_scores_list)

  return raw_response_scores_df
