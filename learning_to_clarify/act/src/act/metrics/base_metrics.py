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

from typing import Any, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from act.config.base_config import BaseConfig
from act.simulation.simulator import Simulator


class BaseMetrics:

  def __init__(
      self,
      act_config,
      max_input_length = 4096,
      debug = False,
  ):
    self.act_config = act_config
    self.max_input_length = max_input_length
    self.debug = debug

  def get_metrics(
      self,
      predicted,
      gold,
  ):
    # This should be overriden by child class
    return None, None

  def conditon_checker(self, **metadata):
    # This should be overriden by child class.
    # metadata contains following fields:
    #   - prompt
    #   - gold_target
    #   - final_answer
    #   - gold_trajectory
    #   - response
    return None


class BaseEvaluator:

  def __init__(
      self,
      simulator,
      metrics,
      max_input_length = 4096,
      debug = False,
  ):
    self.metrics = metrics
    self.simulator = simulator
    self.max_input_length = max_input_length
    self.debug = debug

    self.count = 0
    self.running_accuracy = 0
    self.y = []
    self.y_pred = []
    self.dropf1 = []
    self.trajectory_dropf1 = []
    self.post_clarification_dropf1 = []

    self.samples = []

  def evaluate_one(self, instance):
    self.count += 1
    response = self.simulator.generate_response(
        instance["input_text"],
        max_input_length=self.max_input_length,
        max_new_tokens=128,
    )

    trajectory = self.simulator.generate_trajectory(
        inputs=instance["input_text"],
        chosen_policy=None,
        prompt=instance["input_text"],
        gold_trajectory=instance["gold_trajectory"],
        max_input_length=self.max_input_length,
    )

    inferred_action = trajectory["inferred_action"]
    target_action = self.simulator.action_model.mapper(
        instance["dialogue_policy"]
    )

    clarification_answer = trajectory["clarification_answer"]
    final_answer = trajectory["final_answer"]

    if self.debug:
      print("> Input text:", instance["input_text"].strip())
      print("> Predicted response:", response.strip())
      print("> Output text:", instance["output_text"].strip())
      print("> Inferred_action:", inferred_action)
      print("> Target action:", target_action)
      if inferred_action == "AskClarification":
        print("> Simulated clarification question:", clarification_answer)
        print("> Final answer:", final_answer)

    self.y_pred.append(inferred_action)
    self.y.append(target_action)
    self.running_accuracy += int(inferred_action == target_action)
    print("Current Accuracy: {}".format(self.running_accuracy / self.count))
    print(
        "Current Weighted F1:",
        f1_score(self.y, self.y_pred, average="weighted"),
    )
    print("Current Macro F1:", f1_score(self.y, self.y_pred, average="macro"))
    print(
        "Current Binary F1:",
        f1_score(self.y, self.y_pred, pos_label="AskClarification"),
    )
    print(
        "Current Binary Recall:",
        recall_score(self.y, self.y_pred, pos_label="AskClarification"),
    )
    print(
        "Current Binary Precision:",
        precision_score(self.y, self.y_pred, pos_label="AskClarification"),
    )

    _, turn_f1 = self.metrics.get_metrics(response, instance["output_text"])
    self.dropf1.append(turn_f1)
    print("Running Turn-level DROP F1: {}".format(np.mean(self.dropf1)))

    _, trajectory_f1 = self.metrics.get_metrics(
        final_answer, instance["gold_target"]
    )
    self.trajectory_dropf1.append(trajectory_f1)
    print(
        "Running Trajectory-level DROP F1: {}".format(
            np.mean(self.trajectory_dropf1)
        )
    )
    if inferred_action == "AskClarification":
      self.post_clarification_dropf1.append(trajectory_f1)
      print(
          "Running Post-Clarification DROP F1: {}".format(
              np.mean(self.post_clarification_dropf1)
          )
      )

    self.samples.append({
        "input_text": instance["input_text"],
        "response": response,
        "output_text": instance["output_text"],
        "inferred_action": inferred_action,
        "target_action": target_action,
        "clarification_answer": clarification_answer,
        "final_answer": final_answer,
        "gold_target": instance["gold_target"],
    })

  def final_metrics(self):
    return {
        "Accuracy": self.running_accuracy / self.count,
        "Weighted F1": f1_score(self.y, self.y_pred, average="weighted"),
        "Macro F1": f1_score(self.y, self.y_pred, average="macro"),
        "Binary F1": f1_score(
            self.y, self.y_pred, pos_label="AskClarification"
        ),
        "Binary Recall": recall_score(
            self.y, self.y_pred, pos_label="AskClarification"
        ),
        "Binary Precision": precision_score(
            self.y, self.y_pred, pos_label="AskClarification"
        ),
        "Turn-Level DROP F1": np.mean(self.dropf1),
        "Trajectory-level DROP F1": np.mean(self.trajectory_dropf1),
        "Post-Clarification DROP F1": np.mean(self.post_clarification_dropf1),
    }
