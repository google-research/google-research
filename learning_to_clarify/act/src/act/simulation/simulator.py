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

"""Utils for user simulation ($U$) for ACT."""

from typing import Any, Optional, Union

from act.data.constants import _MODEL_ROLE, _USER_ROLE
from act.models.action_classifier_model import ActionClassifierModel
from act.models.intent_model import UserIntentModel
from act.models.simulator_model import SimulatorModel
import torch
from torch import nn
from transformers import PreTrainedTokenizerBase


class Simulator:

  def __init__(
      self,
      model,
      tokenizer,
      action_model,
      user_intent_model,
      user_simulator_model,
  ):
    self.model = model
    self.tokenizer = tokenizer
    self.user_intent_model = user_intent_model
    self.action_model = action_model
    self.user_simulator_model = user_simulator_model

  def summarize_user_intent(self, history):
    return self.user_intent_model.generate(history)

  def simulate_user_response(self, history, intent):
    output = self.user_simulator_model.generate(
        self.user_simulator_model.simulation_prompt(history, intent)
    )
    return (
        output.split("Assistant")[0].rstrip().split(":")[-1].lstrip().rstrip()
    )

  def generate_response(
      self,
      inputs,
      max_input_length,
      max_new_tokens = 128,
      **generation_kwargs,
  ):
    """Generate response from the model."""
    generation_config = {"max_new_tokens": max_new_tokens, **generation_kwargs}
    if isinstance(inputs, str):
      tokens = self.tokenizer(
          inputs,
          truncation=max_input_length is not None,
          max_length=4096 if max_input_length is None else max_input_length,
          add_special_tokens=True,
          return_tensors="pt",
      )
      inputs = tokens["input_ids"]
      attention_mask = tokens["attention_mask"]
      attention_mask = attention_mask.to(self.model.device)
      generation_config["attention_mask"] = attention_mask.reshape(1, -1)
    inputs = inputs.to(self.model.device)
    tok = self.model.generate(
        input_ids=inputs.reshape(1, -1), **generation_config
    )
    response = self.tokenizer.decode(
        tok.squeeze()[len(inputs.squeeze()) :], skip_special_tokens=True
    )
    return response.lstrip().rstrip().split("\n")[0]

  def should_simulate(self, inferred_action, chosen_policy):
    if inferred_action == "AskClarification" and (
        chosen_policy is None or chosen_policy == inferred_action
    ):
      return True
    return False

  def generate_trajectory(
      self,
      inputs,
      chosen_policy,
      prompt,
      gold_trajectory,
      max_input_length,
      max_new_tokens = 128,
      **generation_kwargs,
  ):
    response = self.generate_response(
        inputs,
        max_input_length,
        max_new_tokens,
        **generation_kwargs,
    )
    inferred_action = self.action_model.classify(
        self.action_model.generate(response)
    )

    if chosen_policy is not None:
      chosen_policy = self.action_model.mapper(chosen_policy)

    if self.should_simulate(inferred_action, chosen_policy):
      intent = self.summarize_user_intent(gold_trajectory).split("\n")[0]
      rollout_prompt = prompt + response + "\nUser:"
      clarification_answer = (
          self.simulate_user_response(rollout_prompt, intent)
          .split("\n")[0]
          .strip()
      )
      final_answer = self.generate_response(
          rollout_prompt + " " + clarification_answer,
          max_input_length,
          max_new_tokens,
          **generation_kwargs,
      )
      response = (
          response + "\nUser: " + clarification_answer + "\n" + final_answer
      )
    else:
      # We still need to call `generate_response` for to match the
      # number of forward passes in each card.
      self.generate_response(
          prompt,
          max_input_length,
          max_new_tokens,
          **generation_kwargs,
      )

      clarification_answer = None
      final_answer = response

    # response: the simulated trajectory.
    # final_answer: the final answer to the question.
    #
    # Example 1:
    #   User: <question>
    #   Assistant: <answer>
    # In this case:
    #   response = "Assistant: <answer>""
    #   final_answer = "Assistant: <answer>"
    #   inferred_action: "DirectlyAnswerQuestion"
    #   clarification_answer: None
    #
    # Example 2:
    #   User: <question>
    #   Assistant: <clarify>
    #   User: <explain>
    #   Assistant: <answer>
    # In this case:
    #   response = "Assistant: <clarify>\nUser: <explain>\nAssistant: <answer>"
    #   final_answer = "Assistant: <answer>"
    #   inferred_action: "AskClarification"
    #   clarification_answer: "<explain>"

    return {
        "response": response,
        "final_answer": final_answer,
        "inferred_action": inferred_action,
        "clarification_answer": clarification_answer,
    }
