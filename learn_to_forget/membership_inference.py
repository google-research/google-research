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

"""Membership inference attacks."""

from typing import Any, Callable, Optional, Tuple

import jax
from jax import flatten_util
from jax import numpy as jnp
from jax.experimental.sparse import linalg as experimental_splinalg
from jaxopt import tree_util
import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize
from scipy.sparse import linalg as splinalg
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit


def evaluate_attack_model(sample_loss,
                          members,
                          n_splits = 5,
                          random_state = None):
  """Computes the cross-validation score of a membership inference attack.

  Args:
    sample_loss : array_like of shape (n,).
      objective function evaluated on n samples.
    members : array_like of shape (n,),
      whether a sample was used for training.
    n_splits: int
      number of splits to use in the cross-validation.
    random_state: int, RandomState instance or None, default=None
      random state to use in cross-validation splitting.

  Returns:
    score : array_like of size (n_splits,)
  """

  unique_members = np.unique(members)
  if not np.all(unique_members == np.array([0, 1])):
    raise ValueError("members should only have 0 and 1s")

  attack_model = LogisticRegression()
  cv = StratifiedShuffleSplit(
      n_splits=n_splits, random_state=random_state)
  return cross_val_score(attack_model, sample_loss, members, cv=cv)


def second_order_defense(loss,
                         params,
                         data_obliterate,
                         target_loss,
                         data_keep = None,
                         l2reg = 1e-1,
                         lobpcg_maxiter = 20,
                         ls_maxiter = 20):
  """Defense based on a second-order expansion of objective function.

  Args:
    loss: callable
      Returns the per-sample loss
    params: array_like or pytree
      Parameters of the trained model
    data_obliterate: tuple (X, y)
      Tuple or list (X, y) with the data to obliterate. X and y are array_like
      of size (n_samples, n_features) and (n_samples,) respectively
    target_loss: float
      Target value for the loss on obliterated samples.
    data_keep: tuple (X, y)
      Tuple or list (X, y) with the data to keep (i.e., not obliterate)
    l2reg: float, optional
      Regularization term for the generalized eigenvalue problem.
    lobpcg_maxiter: int, optional
      Maximum number of iterations in the lobpcg eigenvalue solver.
    ls_maxiter: int, optional
      Maximum number of iterations in the line-search routine to compute
      the magnitude of the update.

  Returns:
    obliterated_params : array_like or pytree, same size as params
      Updated parameters with the solution of this defense.
  """

  flat_params, unflatten = flatten_util.ravel_pytree(params)
  n_features = len(flat_params)

  def _loss_obliterate(x):
    pytree_x = unflatten(x)
    return jnp.mean(loss(pytree_x, data_obliterate))

  def _hessian_obliterate_matvec(x):
    # forward-over-reverse differentiation
    return jax.jvp(jax.grad(_loss_obliterate), (flat_params,), (x,))[1]

  eigenvalues, eigenvectors, _ = experimental_splinalg.lobpcg_standard(
      jax.vmap(_hessian_obliterate_matvec, in_axes=1, out_axes=1),
      flat_params.reshape((-1, 1)),
      m=lobpcg_maxiter)
  normalized_l2_reg = l2reg * eigenvalues[0]

  if data_keep is None:
    update_direction = eigenvectors.ravel()
  else:
    def _loss_keep(x):
      pytree_params = unflatten(x)
      regularization = normalized_l2_reg * jnp.linalg.norm(x)
      return jnp.mean(loss(pytree_params, data_keep)) + regularization

    def _hessian_keep_matvec(x):
      # forward-over-reverse differentiation
      return jax.jvp(jax.grad(_loss_keep), (flat_params,), (x,))[1]

    hessian_obliterate = splinalg.LinearOperator(
        (n_features, n_features), matvec=_hessian_obliterate_matvec)
    hessian_keep = splinalg.LinearOperator(
        (n_features, n_features), matvec=_hessian_keep_matvec)
    _, generalized_eigenvectors = splinalg.eigsh(
        hessian_obliterate, k=1, M=hessian_keep, v0=eigenvectors)
    update_direction = generalized_eigenvectors.ravel()

  # find the step-size by minimizing a 1D objective function (that is, by
  # exact line-search)
  def scalr_fun(step):
    candidate_params = flat_params + step * update_direction
    return (_loss_obliterate(candidate_params) - target_loss) ** 2
  step = optimize.minimize_scalar(scalr_fun, options={"maxiter": ls_maxiter}).x
  # there are two solutions of this problem, one positive and one
  # negative. Do we need to run it twice and get the smallest one?
  return tree_util.tree_add_scalar_mul(params, step,
                                       unflatten(update_direction))
