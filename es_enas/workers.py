# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""This contains the GeneralTopologyBlackboxWorker, which allows RPC requests for updating topologies along with normal ES perturbation evaluations."""


class GeneralTopologyBlackboxWorker(object):
  """Worker class for server."""

  def __init__(self, worker_id, blackbox_object):
    self.worker_id = worker_id
    self.blackbox_object = blackbox_object

  def EvaluateBlackboxInput(self, request):
    """Evaluates a blackbox request.

    Args:
      request: Dict containing inputs.

    Returns:
      result: Dict containing results.
    """
    current_input = request["current_input"]
    core_hyperparameters = list(request["hyperparameters"])
    hyperparameters = [self.worker_id] + core_hyperparameters
    tag = request["tag"]

    function_value, evaluation_stat = self.blackbox_object.execute_with_topology(
        current_input + request["perturbation"], request["topology_str"],
        hyperparameters)

    result = {
        "function_value": function_value,
        "evaluation_stat": evaluation_stat,
        "tag": tag
    }

    if tag != 0:
      result["perturbation"] = request["perturbation"]
    else:
      result["perturbation"] = []

    return result
