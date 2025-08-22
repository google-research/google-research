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

# Lint as: python3
"""Utilities for model training."""

import json
import numpy as np
import tensorflow as tf

VAL_IDXS = {
    'ALL_TRUE': 0,
    'ALL_FALSE': 1,
    'MIXED': 2,
    'TYPE_MISMATCH': 3,
    'EMPTY': 4,
}
TRAIN_FRACTION = 0.9


def sigmoid(x, temp):
  return 1 / (1 + np.exp(-x / temp))


def load_data_from_json(filename):
  """Loads data from the JSON file."""
  with tf.io.gfile.GFile(filename, 'r') as f:
    lines = f.readlines()
  dicts = [json.loads(line) for line in lines]
  data_points = []
  for d in dicts:
    data_points.append({
        'is_positive': d['isPositiveExample'],
        'example_sig': d['exampleSignature'],
        'value_sig': d['valueSignature'],
        'sub_expr_operation_list': d['subExpressionOperationPresenceList']
    })

  n_data_points = len(data_points)
  example_signature_size = len(data_points[0]['example_sig'])
  value_signature_size = len(data_points[0]['value_sig'])
  signature_size = example_signature_size + value_signature_size
  n_operations = len(data_points[0]['sub_expr_operation_list'])
  print('n_data_points: {}'.format(n_data_points))
  print('example_signature_size: {}'.format(example_signature_size))
  print('value_signature_size: {}'.format(value_signature_size))
  # each data point has sig for I/O example and intermediate value
  # so data matrix will have `signature_size` int indices (for embeddings)
  # the target will just be a 1 or a 0.
  data_matrix = np.zeros((n_data_points, signature_size), dtype=np.uint8)
  label_matrix = np.zeros((n_data_points, 1), dtype=np.uint8)
  op_matrix = np.zeros((n_data_points, n_operations), dtype=np.uint8)

  for dp_idx, dp in enumerate(data_points):
    concated_signature = dp['example_sig'] + dp['value_sig']

    for val_idx, val in enumerate(concated_signature):
      data_matrix[dp_idx, val_idx] = VAL_IDXS[val]

    if dp['is_positive']:
      label_matrix[dp_idx, 0] = 1
    else:
      label_matrix[dp_idx, 0] = 0

    # casting bools to ints does what you'd expect
    op_matrix[dp_idx, :] = np.array(
        dp['sub_expr_operation_list'], dtype=np.uint8)

  np.random.seed(0)
  shuffled_idxs = np.random.permutation(range(n_data_points))
  print('shuffled_idxs first 20: {}'.format(shuffled_idxs[:20]))
  shuffled_data_matrix = data_matrix[shuffled_idxs]
  shuffled_label_matrix = label_matrix[shuffled_idxs]
  shuffled_op_matrix = op_matrix[shuffled_idxs]
  train_size = int(TRAIN_FRACTION * n_data_points)
  x_train = shuffled_data_matrix[:train_size]
  y_train = shuffled_label_matrix[:train_size]
  op_train = shuffled_op_matrix[:train_size]
  x_valid = shuffled_data_matrix[train_size:]
  y_valid = shuffled_label_matrix[train_size:]
  op_valid = shuffled_op_matrix[train_size:]

  return (x_train, y_train, op_train, x_valid, y_valid, op_valid,
          example_signature_size)
