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

"""Binary of evaluating instruction following. See README.md."""

import os
from typing import Sequence

from absl import app
from absl import flags
from absl import logging

from instruction_following_eval import evaluation_lib


_INPUT_DATA = flags.DEFINE_string(
    "input_data", None, "path to input data", required=True
)

_INPUT_RESPONSE_DATA = flags.DEFINE_string(
    "input_response_data", None, "path to input response data", required=False
)

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    None,
    "Output directory for inference and eval results.",
    required=True,
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  inputs = evaluation_lib.read_prompt_list(_INPUT_DATA.value)
  prompt_to_response = evaluation_lib.read_prompt_to_response_dict(
      _INPUT_RESPONSE_DATA.value)

  # get instruction following results
  for func, output_file_name in [
      (evaluation_lib.test_instruction_following_strict, "eval_results_strict"),
      (evaluation_lib.test_instruction_following_loose, "eval_results_loose"),
  ]:
    logging.info("Generating %s...", output_file_name)
    outputs = []
    for inp in inputs:
      outputs.append(func(inp, prompt_to_response))
    follow_all_instructions = [o.follow_all_instructions for o in outputs]
    accuracy = sum(follow_all_instructions) / len(outputs)
    logging.info("Accuracy: %f", accuracy)

    output_file_name = os.path.join(
        _OUTPUT_DIR.value, output_file_name + ".jsonl"
    )
    evaluation_lib.write_outputs(output_file_name, outputs)
    logging.info("Generated: %s", output_file_name)

    # Prints instruction following accuracy report.
    print("=" * 64)
    print(f"{output_file_name} Accuracy Scores:")
    evaluation_lib.print_report(outputs)


if __name__ == "__main__":
  app.run(main)
