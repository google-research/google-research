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

"""Evaluator Module for Dynamic Link Prediction"""

import dataclasses
from typing import Any
import numpy as np
import torch


@dataclasses.dataclass(frozen=False)
class MetricsLogger:
  global_total_loss: float = dataclasses.field(default_factory=float)
  global_train_steps: int = dataclasses.field(default_factory=int)
  global_num_events: int = dataclasses.field(default_factory=int)
  num_loss_logs: int = dataclasses.field(default_factory=int)


class Evaluator(object):
  r"""Evaluator for Link Property Prediction"""

  def __init__(self, name, k_value = 10):
    r"""Parameters:

    name: name of the dataset
    k_value: the desired 'k' value for calculating metric@k
    """
    self.name = name
    self.k_value = k_value  # for computing `hits@k`
    self.valid_metric_list = ["hits@", "mrr"]

  def _parse_and_check_input(self, input_dict):
    r"""Helper function to check whether the input has the appropriate format.

    Parameters:
        input_dict: a dictionary containing "y_pred_pos", "y_pred_neg", and
        "eval_metric"
        note: "eval_metric" should be a list including one or more of the
        followin metrics: ["hits@", "mrr"]
    Returns:
        y_pred_pos: positive predicted scores
        y_pred_neg: negative predicted scores
    """

    if "eval_metric" not in input_dict:
      raise RuntimeError("Missing key of eval_metric!")

    y_pred_pos = None
    y_pred_neg = None
    for eval_metric in input_dict["eval_metric"]:
      if eval_metric in self.valid_metric_list:
        if "y_pred_pos" not in input_dict:
          raise RuntimeError("Missing key of y_true")
        if "y_pred_neg" not in input_dict:
          raise RuntimeError("Missing key of y_pred")

        y_pred_pos, y_pred_neg = (
            input_dict["y_pred_pos"],
            input_dict["y_pred_neg"],
        )

        # converting to numpy on cpu
        if torch is not None and isinstance(y_pred_pos, torch.Tensor):
          y_pred_pos = y_pred_pos.detach().cpu().numpy()
        if torch is not None and isinstance(y_pred_neg, torch.Tensor):
          y_pred_neg = y_pred_neg.detach().cpu().numpy()

        # check type and shape
        if not isinstance(y_pred_pos, np.ndarray) or not isinstance(
            y_pred_neg, np.ndarray
        ):
          raise RuntimeError(
              "Arguments to Evaluator need to be either numpy ndarray or torch"
              " tensor!"
          )
      else:
        print(
            "ERROR: The evaluation metric should be in:", self.valid_metric_list
        )
        raise ValueError("Unsupported eval metric %s " % eval_metric)
    self.eval_metric = input_dict["eval_metric"]

    return y_pred_pos, y_pred_neg

  def _eval_hits_and_mrr(
      self,
      y_pred_pos,
      y_pred_neg,
      type_info,
      k_value,
  ):
    r"""compute hist@k and mrr

    reference:
        -
        https://github.com/snap-stanford/ogb/blob/d5c11d91c9e1c22ed090a2e0bbda3fe357de66e7/ogb/linkproppred/evaluate.py#L214

    Parameters:
        y_pred_pos: positive predicted scores
        y_pred_neg: negative predicted scores
        type_info: type of the predicted scores; could be 'torch' or 'numpy'
        k_value: the desired 'k' value for calculating metric@k

    Returns:
        a dictionary containing the computed performance metrics
    """
    if type_info == "torch":
      # calculate ranks
      y_pred_pos = y_pred_pos.view(-1, 1)
      # optimistic rank: "how many negatives have a larger score than the positive?"
      # ~> the positive is ranked first among those with equal score
      optimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
      # pessimistic rank: "how many negatives have at least the positive score?"
      # ~> the positive is ranked last among those with equal score
      pessimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
      ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
      hitsK_list = (ranking_list <= k_value).to(torch.float)
      mrr_list = 1.0 / ranking_list.to(torch.float)

      return {f"hits@{k_value}": hitsK_list.mean(), "mrr": mrr_list.mean()}

    else:
      y_pred_pos = y_pred_pos.reshape(-1, 1)
      optimistic_rank = (y_pred_neg >= y_pred_pos).sum(axis=1)
      pessimistic_rank = (y_pred_neg > y_pred_pos).sum(axis=1)
      ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
      hitsK_list = (ranking_list <= k_value).astype(np.float32)
      mrr_list = 1.0 / ranking_list.astype(np.float32)

      return {f"hits@{k_value}": hitsK_list.mean(), "mrr": mrr_list.mean()}

  def eval(self, input_dict, verbose = False):
    r"""evaluate the link prediction task

    this method is callable through an instance of this object to compute the
    metric

    Parameters:
        input_dict: a dictionary containing "y_pred_pos", "y_pred_neg", and
        "eval_metric"
                    the performance metric is calculated for the provided scores
        verbose: whether to print out the computed metric

    Returns:
        perf_dict: a dictionary containing the computed performance metric
    """
    y_pred_pos, y_pred_neg = self._parse_and_check_input(
        input_dict
    )  # convert the predictions to numpy
    perf_dict = self._eval_hits_and_mrr(
        y_pred_pos, y_pred_neg, type_info="numpy", k_value=self.k_value
    )

    return perf_dict
