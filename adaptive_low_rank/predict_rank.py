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

# Copyright 2024 Google LLC
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

"""Utility functions for module rank prediction."""

from typing import List
import tensorflow as tf


def create_ranks_from_disagreement(
    disagreement, average_rank = 4, n_layers = 12
):
  """Creates a list of ranks from a list of model disagreement values."""
  # Disagreement values are assumed to be non-negative.
  disagreement = tf.convert_to_tensor(disagreement)
  disagreement_sum = tf.math.reduce_sum(disagreement)
  total_rank = average_rank * n_layers
  scaled_disagreement = (
      tf.math.divide(total_rank, disagreement_sum) * disagreement
  )
  print(scaled_disagreement)
  disagreement = tf.cast(tf.math.floor(scaled_disagreement), dtype=tf.int32)
  return ','.join(str(int(d)) for d in disagreement)
