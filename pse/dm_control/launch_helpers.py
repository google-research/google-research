# coding=utf-8
# Copyright 2022 The Google Research Authors.
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
"""Launch Helpers."""

import os
from distracting_control import suite_utils




def get_camera_kwargs(domain_name, scale, dynamic=True):
  """Default camera arguments."""
  return suite_utils.get_camera_kwargs(domain_name, scale, dynamic)


def prefix_result_dict(f, prefix):
  def call_prefix_result_dict(*args, **kwargs):
    d = f(*args, **kwargs)
    return {f'{prefix}{k}': v for k, v in d.items()}
  return call_prefix_result_dict


def get_default_kwargs(domain_name, scale=0.0):
  get_camera_kwargs_prefixed = prefix_result_dict(get_camera_kwargs, 'camera_')
  default_kwargs = get_camera_kwargs_prefixed(domain_name, scale=scale)
  default_kwargs.update({
      'background_ground_plane_alpha': 0.3,
  })
  return default_kwargs
