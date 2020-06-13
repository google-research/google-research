# coding=utf-8
# Copyright 2020 The Google Research Authors.
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
"""Base class for particles samplers."""
import jax.numpy as np
import numpy as onp


class Sampler:
  """Base class for (particle) samplers."""

  def __init__(self):
    self.particle_weights = None
    self.particles = None
    self.convergence_metric = onp.NaN

  def reset(self):
    self.particle_weights = None
    self.particles = None
    self.convergence_metric = onp.NaN

  @property
  def is_cheap(self):
    """Whether the sampling is cheap or expensive."""
    return True

  @property
  def marginal(self):
    """Produces the marginal distribution of the posterior currently stored."""
    return np.sum(self.particle_weights[:, np.newaxis] * self.particles, axis=0)

  def reset_convergence_metric(self):
    self.convergence_metric = onp.NaN
