# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Creates Tasks given the task names."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from bam.bert import tokenization
from bam.task_specific.classification import classification_tasks


def get_tasks(config):
  tokenizer = tokenization.FullTokenizer(vocab_file=config.vocab_file,
                                         do_lower_case=config.do_lower_case)
  return [get_task(config, task_name, tokenizer)
          for task_name in config.task_names]


def get_task(config, task_name, tokenizer):
  """Returns a single task based on the provided name."""

  if task_name == "cola":
    return classification_tasks.CoLA(config, tokenizer)
  elif task_name == "mrpc":
    return classification_tasks.MRPC(config, tokenizer)
  elif task_name == "mnli":
    return classification_tasks.MNLI(config, tokenizer)
  elif task_name == "sst":
    return classification_tasks.SST(config, tokenizer)
  elif task_name == "rte":
    return classification_tasks.RTE(config, tokenizer)
  elif task_name == "qnli":
    return classification_tasks.QNLI(config, tokenizer)
  elif task_name == "qqp":
    return classification_tasks.QQP(config, tokenizer)
  elif task_name == "sts":
    return classification_tasks.STS(config, tokenizer)
  else:
    raise ValueError("Unknown task " + task_name)
