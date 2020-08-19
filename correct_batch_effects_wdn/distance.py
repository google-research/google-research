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
"""Compute various distance metrics for probability densities."""

import numpy as np
import pandas as pd
import sklearn.metrics.pairwise


def _combine(v1, v2):
  """Combine a vector and a vector or array into a single vector."""
  return np.concatenate((v1, v2.reshape(-1)))


def _split(v, col1, col2):
  """Split a vector into a vector + a vector or array.

  The first vector is 1D with col1 columns.  The second has col2 columns and
  is a 1-D vector if len(v) == col1 + col2 or an array otherwise.

  Args:
    v: vector to split
    col1: number of columns for the first portion
    col2: number of columns for the second portion

  Returns:
    A tuple consisting of the first split vector and the second.
  """
  v1 = v[:col1]
  v2 = v[col1:]
  if len(v2) == col2:
    return v1, v2
  return v1, v2.reshape([-1, col2])


def _wrapped_dist_fn(v1, v2, dist_fn=None, dfcol=None, auxcol=None):
  """Wrapper for a distance function that splits the inputs.

  This allows us to use distances that require auxiliary quantities with
  sklearn's parwise_distances function.

  Args:
    v1: first input vector - will be split
    v2: second input vector - will be split
    dist_fn: distance function to call on split vectors
    dfcol: number of columns for the first split portion
    auxcol: number of columns for the second split portion

  Returns:
    The value of dist_fn called on the split versions of v1 and v2.
  """
  v11, v12 = _split(v1, dfcol, auxcol)
  v21, v22 = _split(v2, dfcol, auxcol)
  return dist_fn(v11, v21, v12, v22)


def matrix(dist_fn, df, aux_df=None, n_jobs=1, **kwds):
  """Compute a distance matrix between rows of a DataFrame.

  Args:
    dist_fn: A distance function.  If aux_df = None, should take 2 Series
      as arguments; if aux_df is a data frame, should take 4 Series as
      arguments (row1, row2, aux1, aux2).
    df: DataFrame for which we want to compute row distances
    aux_df: optional auxiliary DataFrame whose rows provide additional
      distance function arguments
    n_jobs: number of parallel jobs to use in computing the distance matrix.
      Note that setting n_jobs > 1 does not work well in Colab.
    **kwds: additional keyword arguments are passed to sklearn's
      pairwise_distances function

  Returns:
    A matrix of distances.
  """
  dfrow, dfcol = df.shape
  if aux_df is not None:
    auxrow, auxcol = aux_df.shape

  # aux_df specifies either a set of vectors of variances or arrays of
  # covariances for use with the distance functions below.  sklearn's
  # pairwise distance function doesn't allow for this kind of side info,
  # so we need to flatten the side information and append it to the vectors
  # in df, then we need to wrap the distance functions so the side info is
  # split out before computing distances.
  if aux_df is not None:
    combined = np.zeros([dfrow, dfcol + int(auxrow / dfrow) * auxcol])
    for i, (idx, row) in enumerate(df.iterrows()):
      combined[i, :] = _combine(row.values, aux_df.loc[idx].values)
    kwds.update(dist_fn=dist_fn, dfcol=dfcol, auxcol=auxcol)
    dist = sklearn.metrics.pairwise.pairwise_distances(
        X=combined, metric=_wrapped_dist_fn, n_jobs=n_jobs, **kwds)
  else:
    dist = sklearn.metrics.pairwise.pairwise_distances(
        X=df.values, metric=dist_fn, n_jobs=n_jobs, **kwds)

  return pd.DataFrame(dist, columns=df.index, index=df.index)
