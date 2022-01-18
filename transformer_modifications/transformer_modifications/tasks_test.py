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

"""Run this test to confirm that a new task or mixture works.
"""

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized

import seqio
import tensorflow.compat.v1 as tf

from transformer_modifications.transformer_modifications import tasks  # pylint: disable=unused-import

tf.disable_v2_behavior()
tf.enable_eager_execution()

TaskRegistry = seqio.TaskRegistry
_SEQUENCE_LENGTH = {"inputs": 512, "targets": 256}
_TASKS = [
    "c4_v220_unsupervised_en32k",
    "xsum_v110",
    "super_glue_boolq_v102_envocab",
    "super_glue_cb_v102_envocab",
    "super_glue_copa_v102_envocab",
    "super_glue_multirc_v102_envocab",
    "super_glue_record_v102_envocab",
    "super_glue_rte_v102_envocab",
    "super_glue_wic_v102_envocab",
    "super_glue_wsc_v102_simple_eval_envocab",
    "super_glue_wsc_v102_simple_train_envocab",
    "web_questions_open_envocab",
    "web_questions_open_test_envocab",
]


class TasksTest(parameterized.TestCase):

  @parameterized.parameters(((name,) for name in _TASKS))
  def test_task(self, name):
    task = TaskRegistry.get(name)
    split = "train" if "train" in task.splits else "validation"
    logging.info("task=%s, split=%s", name, split)
    ds = task.get_dataset(_SEQUENCE_LENGTH, split)
    for d in ds:
      logging.info(d)
      break

if __name__ == "__main__":
  absltest.main()
