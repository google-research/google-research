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

"""Tests for AP Parsing TF-NLP task."""

from absl import logging
from absl.testing import parameterized
import tensorflow as tf

from assessment_plan_modeling.ap_parsing import ap_parsing_dataloader
from assessment_plan_modeling.ap_parsing import ap_parsing_task

TRAIN_DATA_CONFIG = ap_parsing_dataloader.APParsingDataConfig(
    input_path="test", seq_length=16, global_batch_size=1)


class APParsingTaskTestTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="base",
          task_config=ap_parsing_task.APParsingConfig(
              use_crf=False, train_data=TRAIN_DATA_CONFIG)),
      dict(
          testcase_name="crf",
          task_config=ap_parsing_task.APParsingConfig(
              use_crf=True, train_data=TRAIN_DATA_CONFIG)),
  )
  def test_task_with_test_data(self, task_config):

    if task_config.use_crf:
      task = ap_parsing_task.APParsingTaskBase(task_config)
    else:
      task = ap_parsing_task.APParsingTaskCRF(task_config)

    model = task.build_model()
    logging.info("Model summary:\n %s", model.summary())

    metrics = task.build_metrics()
    dataset = task.build_inputs(task_config.train_data)

    iterator = iter(dataset)
    features, labels = next(iterator)
    logging.info(features, labels)
    optimizer = tf.keras.optimizers.SGD(lr=0.1)
    task.initialize(model)
    train_outputs = task.train_step((features, labels),
                                    model,
                                    optimizer,
                                    metrics=metrics)
    logging.info("train_step:\n %s",
                 {k: v.shape for k, v in train_outputs.items()})

    val_outputs = task.validation_step((features, labels), model, metrics)
    logging.info("validation_step:\n %s",
                 {k: v.shape for k, v in val_outputs.items()})

    infer_outputs = task.inference_step((features), model)
    logging.info("inference_step:\n %s",
                 {k: len(v) for k, v in infer_outputs.items()})


if __name__ == "__main__":
  tf.test.main()
