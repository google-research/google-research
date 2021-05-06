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

from jax import lax
from jax.experimental.callback import rewrite


def high_precision_dot_general(*args, **kwargs):
  kwargs.pop('precision')
  return lax.dot_general(*args, precision=lax.Precision.HIGH, **kwargs)


def high_precision_conv(*args, **kwargs):
  kwargs.pop('precision')
  kwargs.pop('lhs_shape')
  kwargs.pop('rhs_shape')
  return lax.conv_general_dilated(*args, precision=lax.Precision.HIGH, **kwargs)


HIGH_PRECISION_RULES = {
    lax.dot_general_p: high_precision_dot_general,
    lax.conv_general_dilated_p: high_precision_conv
}


def rewrite_high_precision(fn):
  return rewrite(fn, HIGH_PRECISION_RULES)
