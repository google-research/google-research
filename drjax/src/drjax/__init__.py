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

"""The FAX package."""

import functools as _functools
import sys as _sys

from . import api as _api


# Import the public API.
broadcast = _api.broadcast
map_fn = _api.map_fn
reduce_mean = _api.reduce_mean
reduce_sum = _api.reduce_sum
reduce_weighted_mean = _api.reduce_weighted_mean


@_functools.wraps(_api.drjax_program)
def program(*, placements, use_spmd_axis_name = True):
  # We wrap here and send in this module as the one to be modified, as it
  # will be the one that users interact with and requires the API changes.
  return _api.drjax_program(
      placements=placements,
      self_module=_sys.modules[__name__],
      use_spmd_axis_name=use_spmd_axis_name,
  )
