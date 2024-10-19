# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Scaling law estimator M1.
"""
from revisiting_neural_scaling_laws.methods import m2


class Estimator(m2.Estimator):
  """Scaling law estimator M1."""

  def __init__(self,
               loss_values,
               c = -0.5,
               update_c = True
               ):
    """Constructor.

    Args:
      loss_values: a dictionary {x: y}, where x is the data size and y is the
        error/loss of the model (lower is better).
      c: initial value of the scaling exponent.
      update_c: set to True if the exponent c is learnable. If False, only the
        beta parameter is optimized.
    """
    super(Estimator, self).__init__(loss_values, c, err_inf=0,
                                    update_c=update_c, update_err_inf=False)
