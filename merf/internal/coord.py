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

"""Tools for manipulating coordinate spaces and distances along rays."""

import jax
import jax.numpy as jnp


def pos_enc(x, min_deg, max_deg, append_identity=True):
  """The positional encoding used by the original NeRF paper."""
  scales = 2 ** jnp.arange(min_deg, max_deg)
  shape = x.shape[:-1] + (-1,)
  scaled_x = jnp.reshape((x[Ellipsis, None, :] * scales[:, None]), shape)
  # Note that we're not using safe_sin, unlike IPE.
  four_feat = jnp.sin(
      jnp.concatenate([scaled_x, scaled_x + 0.5 * jnp.pi], axis=-1)
  )
  if append_identity:
    return jnp.concatenate([x] + [four_feat], axis=-1)
  else:
    return four_feat


def piecewise_warp_fwd(x, eps=jnp.finfo(jnp.float32).eps):
  """A piecewise combo of linear and reciprocal to allow t_near=0."""
  return jnp.where(x < 1, 0.5 * x, 1 - 0.5 / jnp.maximum(eps, x))


def piecewise_warp_inv(x, eps=jnp.finfo(jnp.float32).eps):
  """The inverse of `piecewise_warp_fwd`."""
  return jnp.where(x < 0.5, 2 * x, 0.5 / jnp.maximum(eps, 1 - x))


def s_to_t(s, t_near, t_far):
  """Convert normalized distances ([0,1]) to world distances ([t_near, t_far])."""
  s_near, s_far = [piecewise_warp_fwd(x) for x in (t_near, t_far)]
  return piecewise_warp_inv(s * s_far + (1 - s) * s_near)


def contract(x):
  """The contraction function we proposed in MERF."""
  # For more info check out MERF: Memory-Efficient Radiance Fields for Real-time
  # View Synthesis in Unbounded Scenes: https://arxiv.org/abs/2302.12249,
  # Section 4.2
  # After contraction points lie within [-2,2]^3.
  x_abs = jnp.abs(x)
  # Clamping to 1 produces correct scale inside |x| < 1.
  x_max = jnp.maximum(1, jnp.amax(x_abs, axis=-1, keepdims=True))
  scale = 1 / x_max  # no divide by 0 because of previous maximum(1, ...)
  z = scale * x
  # The above produces coordinates like (x/z, y/z, 1)
  # but we still need to replace the "1" with \pm (2-1/z).
  idx = jnp.argmax(x_abs, axis=-1, keepdims=True)
  negative = jnp.take_along_axis(z, idx, axis=-1) < 0
  o = jnp.where(negative, -2 + scale, 2 - scale)
  # Select the final values by coordinate.
  ival_shape = [1] * (x.ndim - 1) + [x.shape[-1]]
  ival = jnp.arange(x.shape[-1]).reshape(ival_shape)
  result = jnp.where(x_max <= 1, x, jnp.where(ival == idx, o, z))
  return result


def stepsize_in_squash(x, d, v):
  """Computes step size in contracted space."""
  # Approximately computes s such that ||c(x+d*s) - c(x)||_2 = v, where c is
  # the contraction function, i.e. we often need to know by how much (s) the ray
  # needs to be advanced to get an advancement of v in contracted space.
  #
  # The further we are from the scene's center the larger steps in world space
  # we have to take to get the same advancement in contracted space.
  contract_0_grad = jax.grad(lambda x: contract(x)[0])
  contract_1_grad = jax.grad(lambda x: contract(x)[1])
  contract_2_grad = jax.grad(lambda x: contract(x)[2])

  def helper(x, d):
    return jnp.sqrt(
        d.dot(contract_0_grad(x)) ** 2
        + d.dot(contract_1_grad(x)) ** 2
        + d.dot(contract_2_grad(x)) ** 2
    )

  return v / jax.vmap(helper)(x, d)
