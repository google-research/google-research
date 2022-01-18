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

"""Utiliy functions for domain advarserial training of neural networks."""

import jax


@jax.custom_jvp
def flip_grad_identity(x):
  """Identity function with custom jvp that flips the direction of gradients."""
  return x


@flip_grad_identity.defjvp
def flip_grad_identity_jvp(primals, tangents):
  """Flips the direction of gradients."""

  def identity(x):
    return x

  primal_out, tangent_out = jax.jvp(identity, primals, tangents)

  return primal_out, -1 * tangent_out
