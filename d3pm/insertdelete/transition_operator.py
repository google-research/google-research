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

"""Discrete token transition operators."""
import abc
from typing import Any

import flax.struct
import jax
import jax.numpy as jnp

from d3pm.insertdelete import math_util

NDArray = Any


class TransitionOperator(abc.ABC):
  """Base class for transition operators."""

  @abc.abstractmethod
  def prob_matrix(self, log = False):
    """Materializes the transition matrix of probabilities."""
    Ellipsis

  @abc.abstractmethod
  def apply(
      self, before, is_distn, log = False
  ):
    """Apply the transition matrix to a vector (left multiplication)."""
    Ellipsis

  @abc.abstractmethod
  def observe(
      self, after, is_distn, log = False
  ):
    """Observe the result of a transition (right multiplication)."""
    Ellipsis

  def then(self, other):
    """Builds an operator that runs two operators in order."""
    raise NotImplementedError(f"`then` not implemented for {self}")

  def left_fold_identity(self):
    """Builds an identity operator that can be composed with this operator.

    Should be an operator whose type is sufficient for a "left fold", e.g.
    an operator for which

      foo0.left_fold_identity(),
      foo0.left_fold_identity().then(foo0),
      foo0.left_fold_identity().then(foo0).then(foo1),
      foo0.left_fold_identity().then(foo0).then(foo1).then(foo2),
      ...

    are all well defined and have the same type. This is used to allow jumping
    from an initial timestep to the current timestep.
    """
    raise NotImplementedError(f"No left identity for {self}")


@flax.struct.dataclass
class IdentityOperator(TransitionOperator):
  """Simple identity function operator."""

  dim: int = flax.struct.field(pytree_node=False)

  def prob_matrix(self, log = False):
    return jnp.eye(self.dim)

  def apply(
      self, before, is_distn, log = False
  ):
    if not is_distn:
      before = jax.nn.one_hot(before, self.dim)
      if log:
        before = jnp.log(before)
    return before

  def observe(
      self, after, is_distn, log = False
  ):
    if not is_distn:
      after = jax.nn.one_hot(after, self.dim)
      if log:
        after = jnp.log(after)
    return after

  def then(self, other):
    """Builds an operator that runs two operators in order."""
    return other

  def left_fold_identity(self):
    return self


@flax.struct.dataclass
class MatrixOperator(TransitionOperator):
  """Explicit transition matrix operator."""

  matrix: NDArray

  def prob_matrix(self, log = False):
    if log:
      return jnp.log(self.matrix)
    else:
      return self.matrix

  def apply(
      self, before, is_distn, log = False
  ):
    if is_distn:
      if log:
        return math_util.safe_logsumexp(
            before[:, None] + jnp.log(self.matrix), axis=0
        )
      else:
        return before @ self.matrix
    else:
      probs = self.matrix[before, :]
      if log:
        return jnp.log(probs)
      else:
        return probs

  def observe(
      self, after, is_distn, log = False
  ):
    if is_distn:
      if log:
        return math_util.safe_logsumexp(
            after[None, :] + jnp.log(self.matrix), axis=1
        )
      else:
        return self.matrix @ after
    else:
      likelihoods = self.matrix[:, after]
      if log:
        return jnp.log(likelihoods)
      else:
        return likelihoods

  def then(self, other):
    return MatrixOperator(
        jnp.dot(
            self.matrix,
            other.prob_matrix(),
            precision=jax.lax.Precision.HIGHEST,
        )
    )

  def left_fold_identity(self):
    return MatrixOperator(jnp.eye(self.matrix.shape[0]))


@flax.struct.dataclass
class LogMatrixOperator(TransitionOperator):
  """Explicit transition matrix operator."""

  log_matrix: NDArray

  @staticmethod
  def from_probs(matrix):
    return LogMatrixOperator(jnp.log(matrix))

  def prob_matrix(self, log = False):
    if log:
      return self.log_matrix
    else:
      return jnp.exp(self.log_matrix)

  def apply(
      self, before, is_distn, log = False
  ):
    if is_distn:
      if log:
        return math_util.safe_logsumexp(
            before[:, None] + self.log_matrix, axis=0
        )
      else:
        return before @ self.prob_matrix()
    else:
      log_probs = self.log_matrix[before, :]
      if log:
        return log_probs
      else:
        return jnp.exp(log_probs)

  def observe(
      self, after, is_distn, log = False
  ):
    if is_distn:
      if log:
        return math_util.safe_logsumexp(
            after[None, :] + self.log_matrix, axis=1
        )
      else:
        return self.prob_matrix() @ after
    else:
      log_likelihoods = self.log_matrix[:, after]
      if log:
        return log_likelihoods
      else:
        return jnp.exp(log_likelihoods)

  def then(self, other):
    assert isinstance(other, LogMatrixOperator)
    return LogMatrixOperator(
        math_util.safe_logsumexp(
            self.log_matrix[:, :, None] + other.log_matrix[None, :, :], axis=1
        )
    )

  def left_fold_identity(self):
    return LogMatrixOperator(jnp.log(jnp.eye(self.log_matrix.shape[0])))


@flax.struct.dataclass
class RerollOperator(TransitionOperator):
  """Operator that either applies a child operator or redraws."""

  transition: TransitionOperator
  reroll_dist: NDArray
  reroll_prob: NDArray

  def prob_matrix(self, log = False):
    matrix = (
        self.reroll_prob * self.reroll_dist[None, :]
        + (1 - self.reroll_prob) * self.transition.prob_matrix()
    )
    if log:
      return jnp.log(matrix)
    else:
      return matrix

  def apply(
      self, before, is_distn, log = False
  ):
    if log:
      return math_util.safe_logaddexp(
          jnp.log(self.reroll_prob) + jnp.log(self.reroll_dist),
          jnp.log1p(-self.reroll_prob)
          + self.transition.apply(before, is_distn=is_distn, log=True),
      )
    else:
      return self.reroll_prob * self.reroll_dist + (
          1 - self.reroll_prob
      ) * self.transition.apply(before, is_distn=is_distn, log=False)

  def observe(
      self, after, is_distn, log = False
  ):
    if log:
      if is_distn:
        reroll_likelihood = math_util.safe_logsumexp(
            jnp.log(self.reroll_dist) + after
        )
      else:
        reroll_likelihood = jnp.log(self.reroll_dist[after])
      return math_util.safe_logaddexp(
          jnp.log(self.reroll_prob) + reroll_likelihood,
          jnp.log1p(-self.reroll_prob)
          + self.transition.observe(after, is_distn=is_distn, log=True),
      )
    else:
      if is_distn:
        reroll_likelihood = jnp.sum(self.reroll_dist * after)
      else:
        reroll_likelihood = self.reroll_dist[after]
      return self.reroll_prob * reroll_likelihood + (
          1 - self.reroll_prob
      ) * self.transition.observe(after, is_distn=is_distn, log=False)


# pytype: disable=invalid-function-definition
@flax.struct.dataclass
class UniformDiffusionOperator(TransitionOperator):
# pytype: enable=invalid-function-definition
  """Operator with simple diagonal/off-diagonal pairing.

  Attributes:
    dim: Dimension.
    lp_no_randomize: Float log probability of not randomizing.
    lp_randomize: (Inferred) Log prob of randomizing.
    lp_off_diag: (Inferred) Log prob of each off-diagonal transition.
    lp_diag: (Inferred) Log prob of each diagonal (stay-the-same) transition.
  """

  dim: int = flax.struct.field(pytree_node=False)
  lp_no_randomize: NDArray

  @property
  def lp_randomize(self):
    return jnp.log(-jnp.expm1(self.lp_no_randomize))

  @property
  def lp_off_diag(self):
    return self.lp_randomize - jnp.log(self.dim - 1)

  @property
  def lp_diag(self):
    return math_util.safe_logaddexp(self.lp_off_diag, self.lp_no_randomize)

  def prob_matrix(self, log = False):
    lp_matrix = jnp.full((self.dim, self.dim), self.lp_off_diag)
    i = jnp.arange(self.dim)
    lp_matrix = lp_matrix.at[i, i].set(self.lp_diag)
    if log:
      return lp_matrix
    else:
      return jnp.exp(lp_matrix)

  def apply(
      self, before, is_distn, log = False
  ):
    if is_distn:
      if log:
        return math_util.safe_logaddexp(
            self.lp_no_randomize + before, self.lp_off_diag
        )
      else:
        return jnp.exp(self.lp_no_randomize) * before + jnp.exp(
            self.lp_off_diag
        )
    else:
      v = (
          jnp.full((self.dim,), self.lp_off_diag)
          .at[before]
          .set(self.lp_no_randomize)
      )
      if log:
        return v
      else:
        return jnp.exp(v)

  def observe(
      self, after, is_distn, log = False
  ):
    # Symmetric matrix.
    return self.apply(before=after, is_distn=is_distn, log=log)

  def then(self, other):
    compatible = (
        isinstance(other, UniformDiffusionOperator) and other.dim == self.dim
    )
    if not compatible:
      raise ValueError(
          "Can only compose same-dim UniformDiffusionOperator with another."
      )

    return UniformDiffusionOperator(
        self.dim, self.lp_no_randomize + other.lp_no_randomize
    )

  def left_fold_identity(self):
    return UniformDiffusionOperator(self.dim, 0.0)


# pytype: disable=invalid-function-definition
@flax.struct.dataclass
class MaskDiffusionOperator(TransitionOperator):
# pytype: enable=invalid-function-definition
  """Operator that transitions to a mask.

  Attributes:
    dim: Dimension.
    mask_token: Which token is the mask.
    lp_no_mask: Float log probability of not masking.
    lp_mask: (Inferred) Log probability of masking.
  """

  dim: int = flax.struct.field(pytree_node=False)
  mask_token: int = flax.struct.field(pytree_node=False)
  lp_no_mask: NDArray

  @property
  def lp_mask(self):
    return jnp.log(-jnp.expm1(self.lp_no_mask))

  def prob_matrix(self, log = False):
    lp_matrix = jnp.log(jnp.eye(self.dim)) + self.lp_no_mask
    lp_matrix = lp_matrix.at[:, self.mask_token].set(self.lp_mask)
    lp_matrix = lp_matrix.at[self.mask_token, self.mask_token].set(0.0)
    if log:
      return lp_matrix
    else:
      return jnp.exp(lp_matrix)

  def _log_one_hot_on_mask(self):
    return jnp.full([self.dim], -jnp.inf).at[self.mask_token].set(0.0)

  def _log_distn_apply(self, before):
    return math_util.safe_logaddexp(
        self.lp_no_mask + before, self.lp_mask + self._log_one_hot_on_mask()
    )

  def apply(
      self, before, is_distn, log = False
  ):
    if not is_distn:
      before = jnp.log(jax.nn.one_hot(before, self.dim))
    elif not log:
      before = jnp.log(before)

    v = self._log_distn_apply(before)

    if log:
      return v
    else:
      return jnp.exp(v)

  def _log_distn_observe(self, before):
    return math_util.safe_logaddexp(
        self.lp_no_mask + before, self.lp_mask + before[self.mask_token]
    )

  def observe(
      self, after, is_distn, log = False
  ):
    if not is_distn:
      after = jnp.log(jax.nn.one_hot(after, self.dim))
    elif not log:
      after = jnp.log(after)

    v = self._log_distn_observe(after)

    if log:
      return v
    else:
      return jnp.exp(v)

  def then(self, other):
    compatible = (
        isinstance(other, MaskDiffusionOperator) and other.dim == self.dim
    )
    if not compatible:
      raise ValueError(
          "Can only compose same-dim MaskDiff0usionOperator with another."
      )

    return MaskDiffusionOperator(
        self.dim, self.mask_token, self.lp_no_mask + other.lp_no_mask
    )

  def left_fold_identity(self):
    return MaskDiffusionOperator(self.dim, self.mask_token, 0.0)
