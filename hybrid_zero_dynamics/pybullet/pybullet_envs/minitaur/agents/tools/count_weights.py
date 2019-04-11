# Copyright 2017 The TensorFlow Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Count learnable parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import numpy as np
import tensorflow as tf


def count_weights(scope=None, exclude=None, graph=None):
  """Count learnable parameters.

  Args:
    scope: Resrict the count to a variable scope.
    exclude: Regex to match variable names to exclude.
    graph: Operate on a graph other than the current default graph.

  Returns:
    Number of learnable parameters as integer.
  """
  if scope:
    scope = scope if scope.endswith('/') else scope + '/'
  graph = graph or tf.get_default_graph()
  vars_ = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
  if scope:
    vars_ = [var for var in vars_ if var.name.startswith(scope)]
  if exclude:
    exclude = re.compile(exclude)
    vars_ = [var for var in vars_ if not exclude.match(var.name)]
  shapes = [var.get_shape().as_list() for var in vars_]
  return int(sum(np.prod(shape) for shape in shapes))
