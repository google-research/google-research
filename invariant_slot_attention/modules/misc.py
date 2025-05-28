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

"""Miscellaneous modules."""

from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

from flax import linen as nn
import jax
import jax.numpy as jnp

from invariant_slot_attention.lib import utils

Shape = Tuple[int]

DType = Any
Array = Any  # jnp.ndarray
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[str, "ArrayTree"]]  # pytype: disable=not-supported-yet
ProcessorState = ArrayTree
PRNGKey = Array
NestedDict = Dict[str, Any]


class Identity(nn.Module):
  """Module that applies the identity function, ignoring any additional args."""

  @nn.compact
  def __call__(self, inputs, **args):
    return inputs


class Readout(nn.Module):
  """Module for reading out multiple targets from an embedding."""

  keys: Sequence[str]
  readout_modules: Sequence[Callable[[], nn.Module]]
  stop_gradient: Optional[Sequence[bool]] = None

  @nn.compact
  def __call__(self, inputs, train = False):
    num_targets = len(self.keys)
    assert num_targets >= 1, "Need to have at least one target."
    assert len(self.readout_modules) == num_targets, (
        "len(modules) and len(keys) must match.")
    if self.stop_gradient is not None:
      assert len(self.stop_gradient) == num_targets, (
          "len(stop_gradient) and len(keys) must match.")
    outputs = {}
    for i in range(num_targets):
      if self.stop_gradient is not None and self.stop_gradient[i]:
        x = jax.lax.stop_gradient(inputs)
      else:
        x = inputs
      outputs[self.keys[i]] = self.readout_modules[i]()(x, train)  # pytype: disable=not-callable
    return outputs


class MLP(nn.Module):
  """Simple MLP with one hidden layer and optional pre-/post-layernorm."""

  hidden_size: int
  output_size: Optional[int] = None
  num_hidden_layers: int = 1
  activation_fn: Callable[[Array], Array] = nn.relu
  layernorm: Optional[str] = None
  activate_output: bool = False
  residual: bool = False

  @nn.compact
  def __call__(self, inputs, train = False):
    del train  # Unused.

    output_size = self.output_size or inputs.shape[-1]

    x = inputs

    if self.layernorm == "pre":
      x = nn.LayerNorm()(x)

    for i in range(self.num_hidden_layers):
      x = nn.Dense(self.hidden_size, name=f"dense_mlp_{i}")(x)
      x = self.activation_fn(x)
    x = nn.Dense(output_size, name=f"dense_mlp_{self.num_hidden_layers}")(x)

    if self.activate_output:
      x = self.activation_fn(x)

    if self.residual:
      x = x + inputs

    if self.layernorm == "post":
      x = nn.LayerNorm()(x)

    return x


class GRU(nn.Module):
  """GRU cell as nn.Module."""

  @nn.compact
  def __call__(self, carry, inputs,
               train = False):
    del train  # Unused.
    features = carry.shape[-1]
    carry, _ = nn.GRUCell(features)(carry, inputs)
    return carry


class Dense(nn.Module):
  """Dense layer as nn.Module accepting "train" flag."""

  features: int
  use_bias: bool = True

  @nn.compact
  def __call__(self, inputs, train = False):
    del train  # Unused.
    return nn.Dense(features=self.features, use_bias=self.use_bias)(inputs)


class PositionEmbedding(nn.Module):
  """A module for applying N-dimensional position embedding.

  Attr:
    embedding_type: A string defining the type of position embedding to use. One
      of ["linear", "discrete_1d", "fourier", "gaussian_fourier"].
    update_type: A string defining how the input is updated with the position
      embedding. One of ["proj_add", "concat"].
    num_fourier_bases: The number of Fourier bases to use. For embedding_type ==
      "fourier", the embedding dimensionality is 2 x number of position
      dimensions x num_fourier_bases. For embedding_type == "gaussian_fourier",
      the embedding dimensionality is 2 x num_fourier_bases. For embedding_type
      == "linear", this parameter is ignored.
    gaussian_sigma: Standard deviation of sampled Gaussians.
    pos_transform: Optional transform for the embedding.
    output_transform: Optional transform for the combined input and embedding.
    trainable_pos_embedding: Boolean flag for allowing gradients to flow into
      the position embedding, so that the optimizer can update it.
  """

  embedding_type: str
  update_type: str
  num_fourier_bases: int = 0
  gaussian_sigma: float = 1.0
  pos_transform: Callable[[], nn.Module] = Identity
  output_transform: Callable[[], nn.Module] = Identity
  trainable_pos_embedding: bool = False

  def _make_pos_embedding_tensor(self, rng, input_shape):
    if self.embedding_type == "discrete_1d":
      # An integer tensor in [0, input_shape[-2]-1] reflecting
      # 1D discrete position encoding (encode the second-to-last axis).
      pos_embedding = jnp.broadcast_to(
          jnp.arange(input_shape[-2]), input_shape[1:-1])
    else:
      # A tensor grid in [-1, +1] for each input dimension.
      pos_embedding = utils.create_gradient_grid(input_shape[1:-1], [-1.0, 1.0])

    if self.embedding_type == "linear":
      pass
    elif self.embedding_type == "discrete_1d":
      pos_embedding = jax.nn.one_hot(pos_embedding, input_shape[-2])
    elif self.embedding_type == "fourier":
      # NeRF-style Fourier/sinusoidal position encoding.
      pos_embedding = utils.convert_to_fourier_features(
          pos_embedding * jnp.pi, basis_degree=self.num_fourier_bases)
    elif self.embedding_type == "gaussian_fourier":
      # Gaussian Fourier features. Reference: https://arxiv.org/abs/2006.10739
      num_dims = pos_embedding.shape[-1]
      projection = jax.random.normal(
          rng, [num_dims, self.num_fourier_bases]) * self.gaussian_sigma
      pos_embedding = jnp.pi * pos_embedding.dot(projection)
      # A slightly faster implementation of sin and cos.
      pos_embedding = jnp.sin(
          jnp.concatenate([pos_embedding, pos_embedding + 0.5 * jnp.pi],
                          axis=-1))
    else:
      raise ValueError("Invalid embedding type provided.")

    # Add batch dimension.
    pos_embedding = jnp.expand_dims(pos_embedding, axis=0)

    return pos_embedding

  @nn.compact
  def __call__(self, inputs):

    # Compute the position embedding only in the initial call use the same rng
    # as is used for initializing learnable parameters.
    pos_embedding = self.param("pos_embedding", self._make_pos_embedding_tensor,
                               inputs.shape)

    if not self.trainable_pos_embedding:
      pos_embedding = jax.lax.stop_gradient(pos_embedding)

    # Apply optional transformation on the position embedding.
    pos_embedding = self.pos_transform()(pos_embedding)  # pytype: disable=not-callable

    # Apply position encoding to inputs.
    if self.update_type == "project_add":
      # Here, we project the position encodings to the same dimensionality as
      # the inputs and add them to the inputs (broadcast along batch dimension).
      # This is roughly equivalent to concatenation of position encodings to the
      # inputs (if followed by a Dense layer), but is slightly more efficient.
      n_features = inputs.shape[-1]
      x = inputs + nn.Dense(n_features, name="dense_pe_0")(pos_embedding)
    elif self.update_type == "concat":
      # Repeat the position embedding along the first (batch) dimension.
      pos_embedding = jnp.broadcast_to(
          pos_embedding, shape=inputs.shape[:-1] + pos_embedding.shape[-1:])
      # concatenate along the channel dimension.
      x = jnp.concatenate((inputs, pos_embedding), axis=-1)
    else:
      raise ValueError("Invalid update type provided.")

    # Apply optional output transformation.
    x = self.output_transform()(x)  # pytype: disable=not-callable
    return x


class RelativePositionEmbedding(nn.Module):
  """A module for applying embedding of input position relative to slots.

  Attr
    update_type: A string defining how the input is updated with the position
      embedding. One of ["proj_add", "concat"].
    embedding_type: A string defining the type of position embedding to use.
      Currently only "linear" is supported.
    num_fourier_bases: The number of Fourier bases to use. For embedding_type ==
      "fourier", the embedding dimensionality is 2 x number of position
      dimensions x num_fourier_bases. For embedding_type == "gaussian_fourier",
      the embedding dimensionality is 2 x num_fourier_bases. For embedding_type
      == "linear", this parameter is ignored.
    gaussian_sigma: Standard deviation of sampled Gaussians.
    pos_transform: Optional transform for the embedding.
    output_transform: Optional transform for the combined input and embedding.
    trainable_pos_embedding: Boolean flag for allowing gradients to flow into
      the position embedding, so that the optimizer can update it.
  """

  update_type: str
  embedding_type: str = "linear"
  num_fourier_bases: int = 0
  gaussian_sigma: float = 1.0
  pos_transform: Callable[[], nn.Module] = Identity
  output_transform: Callable[[], nn.Module] = Identity
  trainable_pos_embedding: bool = False
  scales_factor: float = 1.0

  def _make_pos_embedding_tensor(self, rng, input_shape):

    # A tensor grid in [-1, +1] for each input dimension.
    pos_embedding = utils.create_gradient_grid(input_shape[1:-1], [-1.0, 1.0])

    # Add batch dimension.
    pos_embedding = jnp.expand_dims(pos_embedding, axis=0)

    return pos_embedding

  @nn.compact
  def __call__(self, inputs, slot_positions,
               slot_scales = None,
               slot_rotm = None):

    # Compute the position embedding only in the initial call use the same rng
    # as is used for initializing learnable parameters.
    pos_embedding = self.param("pos_embedding", self._make_pos_embedding_tensor,
                               inputs.shape)

    if not self.trainable_pos_embedding:
      pos_embedding = jax.lax.stop_gradient(pos_embedding)

    # Relativize pos_embedding with respect to slot positions
    # and optionally slot scales.
    slot_positions = jnp.expand_dims(
        jnp.expand_dims(slot_positions, axis=-2), axis=-2)
    if slot_scales is not None:
      slot_scales = jnp.expand_dims(
          jnp.expand_dims(slot_scales, axis=-2), axis=-2)

    if self.embedding_type == "linear":
      pos_embedding = pos_embedding - slot_positions
      if slot_rotm is not None:
        pos_embedding = self.transform(slot_rotm, pos_embedding)
      if slot_scales is not None:
        # Scales are usually small so the grid might get too large.
        pos_embedding = pos_embedding / self.scales_factor
        pos_embedding = pos_embedding / slot_scales
    else:
      raise ValueError("Invalid embedding type provided.")

    # Apply optional transformation on the position embedding.
    pos_embedding = self.pos_transform()(pos_embedding)  # pytype: disable=not-callable

    # Define intermediate for logging.
    pos_embedding = Identity(name="pos_emb")(pos_embedding)

    # Apply position encoding to inputs.
    if self.update_type == "project_add":
      # Here, we project the position encodings to the same dimensionality as
      # the inputs and add them to the inputs (broadcast along batch dimension).
      # This is roughly equivalent to concatenation of position encodings to the
      # inputs (if followed by a Dense layer), but is slightly more efficient.
      n_features = inputs.shape[-1]
      x = inputs + nn.Dense(n_features, name="dense_pe_0")(pos_embedding)
    elif self.update_type == "concat":
      # Repeat the position embedding along the first (batch) dimension.
      pos_embedding = jnp.broadcast_to(
          pos_embedding, shape=inputs.shape[:-1] + pos_embedding.shape[-1:])
      # concatenate along the channel dimension.
      x = jnp.concatenate((inputs, pos_embedding), axis=-1)
    else:
      raise ValueError("Invalid update type provided.")

    # Apply optional output transformation.
    x = self.output_transform()(x)  # pytype: disable=not-callable
    return x

  @classmethod
  def transform(cls, rot, coords):
    # The coordinate grid coords is in the (y, x) format, so we need to swap
    # the coordinates on the input and output.
    coords = jnp.stack([coords[Ellipsis, 1], coords[Ellipsis, 0]], axis=-1)
    # Equivalent to inv(R) * coords^T = R^T * coords^T = (coords * R)^T.
    # We are multiplying by the inverse of the rotation matrix because
    # we are rotating the coordinate grid *against* the rotation of the object.
    new_coords = jnp.einsum("...hij,...jk->...hik", coords, rot)
    # Swap coordinates again.
    return jnp.stack([new_coords[Ellipsis, 1], new_coords[Ellipsis, 0]], axis=-1)
