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

"""LogSNR schedules (t==0 => logsnr_max, t==1 => logsnr_min)."""

import functools
import jax.numpy as jnp
import numpy as onp


def _logsnr_schedule_cosine(t, *, logsnr_min, logsnr_max):
  b = onp.arctan(onp.exp(-0.5 * logsnr_max))
  a = onp.arctan(onp.exp(-0.5 * logsnr_min)) - b
  return -2. * jnp.log(jnp.tan(a * t + b))


def get_logsnr_schedule(name, **kwargs):
  """Get log SNR schedule (t==0 => logsnr_max, t==1 => logsnr_min)."""
  schedules = {
      'cosine': _logsnr_schedule_cosine,
  }
  return functools.partial(schedules[name], **kwargs)
