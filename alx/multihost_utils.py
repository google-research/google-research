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

r"""Utilities for synchronizing and communication across multiple hosts.

Derived from p5x.
"""

import functools
import os

import jax
import numpy as np


# NB: This needs to be top-level for the jax compilation cache.
@functools.partial(jax.pmap, axis_name='hosts')
def _host_allgather_psum(x):
  """Host psum for host_allgather."""
  return jax.lax.psum(x, 'hosts')


def sync_devices():
  """Creates a barrier across all hosts/devices."""
  x = np.ones([jax.local_device_count()])
  x = jax.device_get(_host_allgather_psum(x))
  if x[0] != jax.device_count():
    raise ValueError(f'x[0] != jax.device_count(). '
                     f'x: {x}, '
                     f'jax.local_device_count(): {jax.local_device_count()}, '
                     f'jax.device_count(): {jax.device_count()}')


def get_host_dir(workdir, host_id):
  """Returns a host level dir for storing partitioned variables."""
  return os.path.join(workdir, f'host_{host_id}')
