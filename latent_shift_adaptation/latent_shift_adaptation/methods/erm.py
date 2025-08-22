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

"""Baseline model. Train from X, C, W or a combination of them to Y directly.

Assumes that the input is known at test time in both source and target. The
data is a tuple (X, Y, C, W, U).
"""

from latent_shift_adaptation.methods import baseline
import tensorflow as tf


class Method(baseline.Method):
  """Baseline method that predicts Y directly from the chosen input."""

  def __init__(self, model, evaluate=None, inputs="x", outputs="y",
               dtype=tf.float32, pos=None):  # pylint: disable=g-doc-args
    """Constructor.

    Args:
      model: Compiled keras model.
      evaluate: a tf.keras.metrics method.
      inputs: the input of a model, e.g. 'x' if x -> y, 'cw' if from C,W to Y.
      outputs: the ouptut of a model, e.g. 'y'
      dtype: desired dtype (e.g. tf.float32).
      pos: config_dict that specifies the index of x, y, c, w, u in data tuple.
        Default: data is of the form (x, y, c, w, u).
    """
    super(Method, self).__init__(evaluate, dtype, pos)
    self.model = model
    self.inputs = inputs
    if outputs not in ["y", "c", "w", "u", "cy"]:
      raise ValueError('Unrecognized value of "outputs".'
                       'Supported values are "y", "c", "w", "u", and "cy".')
    self.outputs = outputs
