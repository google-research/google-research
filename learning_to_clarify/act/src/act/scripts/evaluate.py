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

"""Used to evaluate an ACT model.

Eval data should be in a format that HuggingFace `DPOTrainer` can understand
(not in Vertex AI format).
"""

import json
from typing import cast

from absl import app
from act.config.flags import FLAGS, initialize_flags
from act.config.utils import (
    get_config_from_dict,
    get_config_from_flags,
    get_config_from_json_path,
    get_default_config,
)
from act.metrics.base_metrics import BaseEvaluator
from act.metrics.pacific_metrics import PacificMetrics
from act.models.utils import initialize_env, load_models
from act.simulation.simulator import Simulator
from act.utils.storage_utils import (
    read_json,
    read_jsonl,
    write_json,
)
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import torch

initialize_flags(get_default_config())

def main(argv):
  """Main function for evaluating an ACT model."""
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if FLAGS.config != "":
    config = get_config_from_json_path(FLAGS.config)
  else:
    config = get_config_from_flags(FLAGS)

  initialize_env()

  eval_data = read_jsonl(config.data_config.eval_data)

  model, _, action_model, user_simulator, intent_model = (
      load_models(config, load_policy_from_checkpoint=True)
  )

  if torch.cuda.is_available():
    model.device = 'cuda'

  simulator = Simulator(
      model.model,
      model.tokenizer,
      action_model,
      intent_model,
      user_simulator,
  )
  metrics = PacificMetrics()
  evaluator = BaseEvaluator(
      simulator,
      metrics,
      max_input_length=config.data_config.eval_max_input_length,
      debug=FLAGS.debug,
  )

  length = len(eval_data)
  for i in range(length):
    evaluator.evaluate_one({
        "input_text": eval_data[i]["input_text"],
        "output_text": eval_data[i]["output_text"],
        "dialogue_policy": eval_data[i]["dialogue_policy"],
        "gold_trajectory": eval_data[i]["gold_trajectory"],
        "gold_target": eval_data[i]["gold_target"],
    })
    print("Evaluated: {} of {}".format(i + 1, length))
    print("------------------------------------")

  if config.data_config.eval_sample_output_path:
    write_json(config.data_config.eval_sample_output_path, evaluator.samples)

  if config.data_config.eval_result_output_path:
    write_json(
        config.data_config.eval_result_output_path, evaluator.final_metrics()
    )


if __name__ == "__main__":
  app.run(main)
