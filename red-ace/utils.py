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

"""Utility functions for RED-ACE."""
import json


def read_input(input_file,):
  """Read JSON input_file."""
  with open(input_file, 'r') as f:
    d = json.load(f)
    for entry in d:
      utterance_id = entry['id']
      source = ' '.join([word[0] for word in entry['asr']])
      target = entry['truth']
      confidence_scores = [word[1] for word in entry['asr']]
      yield (source, target, confidence_scores, utterance_id)


def batch_generator(
    predict_input_file,
    batch_size,
):
  """Produces batches for RED-ACE to predict.

  Args:
    predict_input_file: Root of the Flume pipeline.
    batch_size: The batch size.

  Yields:
    A tuple of (list of source texts, variant ids, target texts).
  """
  source_batch = []
  target_batch = []
  confidence_scores_batch = []
  utterance_id_batch = []
  for source, target, confidence_scores, utterance_id in read_input(
      predict_input_file):
    source_batch.append(source)
    target_batch.append('\t'.join(target))
    confidence_scores_batch.append(confidence_scores)
    utterance_id_batch.append(utterance_id)
    if len(source_batch) == batch_size:
      yield source_batch, confidence_scores_batch, target_batch, utterance_id_batch
      source_batch = []
      target_batch = []
      confidence_scores_batch = []
      utterance_id_batch = []

  if source_batch:
    yield source_batch, confidence_scores_batch, target_batch, utterance_id_batch
