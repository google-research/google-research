# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Tests for input/response coverage before evaluation."""

import json
import pathlib
import subprocess
import sys

from absl.testing import absltest
from instruction_following_eval import evaluation_lib


class EvaluationMainTest(absltest.TestCase):

  def _write_jsonl(self, rows):
    return self.create_tempfile(
        content="".join(json.dumps(row) + "\n" for row in rows)
    ).full_path

  def _input(self, key, prompt):
    return {
        "key": key,
        "instruction_id_list": ["keywords:existence"],
        "prompt": prompt,
        "kwargs": [{"keywords": ["hello"]}],
    }

  def _run_evaluation(self, input_path, response_path, output_dir):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "instruction_following_eval.evaluation_main",
            f"--input_data={input_path}",
            f"--input_response_data={response_path}",
            f"--output_dir={output_dir}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

  def test_missing_responses_rejected_before_scoring(self):
    input_path = self._write_jsonl([
        self._input(1, "First prompt"),
        self._input(2785, "Revised prompt"),
        self._input(9, "Another missing prompt"),
    ])
    response_path = self._write_jsonl([
        {"prompt": "First prompt", "response": "hello"},
        {"prompt": "Old prompt", "response": "hello"},
    ])
    output_dir = self.create_tempdir().full_path

    result = self._run_evaluation(input_path, response_path, output_dir)

    self.assertNotEqual(result.returncode, 0)
    self.assertIn(
        "ValueError: Missing responses for input task keys: [2785, 9]",
        result.stderr,
    )
    self.assertNotIn("Generating eval_results", result.stderr)
    self.assertEmpty(list(pathlib.Path(output_dir).iterdir()))

  def test_matching_prompts_preserve_scores_and_allow_extra_responses(self):
    input_path = self._write_jsonl([
        self._input(1, "First prompt"),
        self._input(2, "Second prompt"),
    ])
    response_path = self._write_jsonl([
        {"prompt": "Second prompt", "response": ""},
        {"prompt": "Unused prompt", "response": "hello"},
        {"prompt": "First prompt", "response": "hello"},
    ])
    output_dir = self.create_tempdir().full_path

    result = self._run_evaluation(input_path, response_path, output_dir)
    self.assertEqual(result.returncode, 0, msg=result.stderr)

    for mode in ("strict", "loose"):
      output_path = pathlib.Path(output_dir) / f"eval_results_{mode}.jsonl"
      outputs = [json.loads(line) for line in output_path.read_text().splitlines()]
      self.assertEqual([row["response"] for row in outputs], ["hello", ""])
      self.assertEqual(
          [row["follow_all_instructions"] for row in outputs], [True, False]
      )

  def test_archived_demo_has_complete_prompt_coverage(self):
    data_dir = pathlib.Path(__file__).resolve().parent / "data"
    inputs = evaluation_lib.read_prompt_list(
        data_dir / "input_data_gpt4_20231107_145030.jsonl"
    )
    responses = evaluation_lib.read_prompt_to_response_dict(
        data_dir / "input_response_data_gpt4_20231107_145030.jsonl"
    )

    self.assertLen(inputs, 541)
    self.assertLen(responses, 541)
    evaluation_lib.validate_prompt_coverage(inputs, responses)
    self.assertEqual({inp.prompt for inp in inputs}, set(responses))


if __name__ == "__main__":
  absltest.main()
