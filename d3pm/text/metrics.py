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

"""Metric functions for the D3PM library."""

from d3pm.text import types


def sequence_accuracy(predictions,
                      batch,
                      *,
                      dataset_info,
                      prediction_key='text/final_text',
                      target_key='targets'):
  """Computes the sequence-level exact match accuracy."""
  del dataset_info

  if prediction_key not in predictions:
    raise ValueError(
        f'Expected key "{prediction_key}" in sequence postprocessor.')

  text = predictions[prediction_key]

  text = text.reshape((-1,) + text.shape[-1:])  # [N, seq]

  if target_key not in batch:
    raise ValueError(f'Expected key "{target_key}" not found in input batch.')

  reference_text = batch[target_key]

  acc = (reference_text == text).all(-1).mean()

  return [types.Metric(
      name='sequence_accuracy',
      type='scalar',
      value=acc,
  )]


def take_n(predictions, batch, *, dataset_info, n=1):
  """Extracts any named features from the prediction if specified."""
  del batch

  metrics = []

  for key, value in predictions.items():
    if key.startswith('img/'):
      value = value.reshape((-1,) + value.shape[-3:])
      metrics.append(types.Metric(
          name=key,
          type='image',
          value=value[:n],
      ))
    elif key.startswith('text/'):
      value = value.reshape((-1,) + value.shape[-1:])
      text = dataset_info.vocab.decode_tf(value)
      metrics.append(types.Metric(
          name=key,
          type='text',
          value=text[:n],
      ))

  return metrics
