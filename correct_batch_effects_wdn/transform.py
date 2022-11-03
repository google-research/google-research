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

"""Functions to transform raw embeddings into more meaningful representations.
"""

import numpy as np
import pandas as pd
from sklearn import decomposition
import tensorflow.compat.v1 as tf

from correct_batch_effects_wdn import metadata


def keep_rows_by_multi_index(df, multi_index_name, multi_index_value):
  """Keeps rows with given multi-index values from a DataFrame.

  Args:
    df: DataFrame with samples as rows and multi-indices.
    multi_index_name: String, the name of the multi-index.
    multi_index_value: String or a list of strings, the values of the
      multi-index.

  Returns:
    A subset of the DataFrame with only the specified rows.

  Raises:
    ValueError: Multi-index to drop is not contained in the DataFrame.
  """
  if multi_index_name not in df.index.names:
    raise ValueError(
        'Multi-index %s is not contained in the DataFrame.' % multi_index_name)
  if not isinstance(multi_index_value, list):
    multi_index_value = [multi_index_value]
  return df[df.index.get_level_values(
      level=multi_index_name).isin(multi_index_value)]


def get_negative_controls(df):
  """Get negative control samples from a data frame of samples.

  Args:
    df: DataFrame with samples as rows and metadata.TREATMENT_GROUP as an index.

  Returns:
    The input DataFrame filtered to negative control samples.
  """
  treatment_name = metadata.TREATMENT_GROUP
  return keep_rows_by_multi_index(df, treatment_name, metadata.NEGATIVE_CONTROL)


def eig_symmetric(m):
  """Get the eigenvalues and eigenvectors for a real, symmetric matrix.

  Uses linalg.eigh, which is optimized for symmetric matrices.  Eigenvalues
  are then sorted in descending order.

  Args:
    m: a real, symmetric matrix

  Returns:
    A tuple consisting of a vector of eigenvalues in descending order and
    a matrix of eigenvectors (eigenvectors are columns).
  """
  lambda_m, q_m = np.linalg.eigh(m)
  idx = lambda_m.argsort()[::-1]  # sort by eigenvalues, descending
  return lambda_m[idx], q_m[:, idx]


def factor_analysis(embedding_dataframe, fraction, n_components):
  """Projects the embeddings by factor analysis using negative controls.

  It would be interesting to explore factor analysis because it is a natural way
  to extract important latent features from the data, and PCA is actually a
  special case of factor analysis. When the variances of the error term in
  factor analysis are identical and go towards zero, the posterior estimate of
  the latent variables becomes exactly PCA.

  TVN is essentially PCA without dimension reduction. Compared with TVN, the
  drawback of factor analysis is that it requires to specify the number of
  latent variables. As an ad-hoc approach, I would suggest specifying it as the
  number of unique treatments.

  Args:
    embedding_dataframe: Pandas dataframe of the embeddings with each row as a
      sample.
    fraction: Fraction of negative control samples used to estimate parameters
      in factor analysis.
    n_components: Number of latent variables. If -1, specify n_components as
      the number of unique treatments.

  Returns:
     A Pandas dataframe with a reduced number of dimensions.
  """
  # specify the number of latent variables as the number of unique treatments,
  # excluding the negative control
  if n_components == -1:
    n_components = embedding_dataframe.reset_index()[[
        metadata.COMPOUND, metadata.CONCENTRATION
    ]].drop_duplicates().shape[0] - 1
  factor_analysis_object = decomposition.FactorAnalysis(
      n_components=n_components)
  factor_analysis_object.fit(
      get_negative_controls(embedding_dataframe).sample(frac=fraction,
                                                        axis=0).values)
  return pd.DataFrame(
      data=factor_analysis_object.transform(embedding_dataframe.values),
      index=embedding_dataframe.index)


def get_bootstrap_sample(df):
  """Get a bootstrap sample of a DataFrame.

  To make different bootstrap samples comparable, we assume that the
  experimental condition, i.e., the layout of batches, plates and
  wells, stays the same. Otherwise the bootstrap samples would vary
  considerably. Thus, the bootstrapping is conducted for the embedding vectors
  within each well. Without making any distributional assumption, nonparametric
  bootstrapping is performed.

  Args:
    df: DataFrame with samples as rows.  Should have a MultiIndex with levels
      batch, plate and well

  Returns:
    A bootstrap sample of the DataFrame.
  """
  df_list = []
  levels = [metadata.BATCH, metadata.PLATE, metadata.WELL]
  if metadata.TIMEPOINT in df.index.names:
    levels.append(metadata.TIMEPOINT)
  if metadata.SEQUENCE in df.index.names:
    levels.append(metadata.SEQUENCE)
  for _, df_well in df.groupby(level=levels):
    index = np.random.choice(df_well.shape[0], size=df_well.shape[0])
    df_list.append(df_well.iloc[index, :])
  return pd.concat(df_list)


def normalize_df_by_coral(df,
                          center,
                          center_cov,
                          level_to_normalize=metadata.BATCH,
                          lambda_reg=1.0):
  """Remove batch effects by variants of CORAL method.

  This function is adapted from @geoffd's colab:
  https://drive.google.com/open?id=0B-umIy5UPrYIVWJYYnZaMzNRNGc.

  The main idea of CORAL method is (1) finding an affine transformation on each
  batch (or plate) such that after transformation the covariances of the
  negative control on each batch (or plate) are the same as the overall
  covariance of the negative control before transformation; (2) applying each
  learned transformation to all compounds on each batch (or plate).

  Args:
    df: Pandas dataframe with complete multi-index of metadata and embeddings
    center: A boolean giving whether the embedding vectors are centered by the
      mean of those of the negative control on each batch (or plate)
    center_cov: A boolean giving whether the covariance is centered
    level_to_normalize: (optional) A string giving the level on which CORAL
      method is applied
    lambda_reg: the weight of the regularization term (i.e., an identity matrix)
      added to the estimated covariance on each batch (or plate)

  Returns:
    A Pandas dataframe transformed by CORAL method, which has the same
      multi-index as df.
  """
  normalized = []
  for _, df_level in df.groupby(level=level_to_normalize):
    df_control = df_level.xs(
        metadata.NEGATIVE_CONTROL,
        level=metadata.TREATMENT_GROUP,
        drop_level=False)
    control_mean = np.mean(df_control, axis=0)
    if center_cov:
      control_cov = np.cov(df_control, rowvar=False)
    else:
      control_cov = (df_control.T.dot(df_control)) / df_control.shape[0]
    # regularize
    control_cov += lambda_reg * np.identity(control_cov.shape[1])
    lambda_cov, q_cov = eig_symmetric(control_cov)
    if center:
      df_level -= control_mean
    normalized_df = (df_level.dot(q_cov) / np.sqrt(lambda_cov)).dot(q_cov.T)
    normalized.append(normalized_df)
  return pd.concat(normalized)


def coral_without_mean_shift_batch(df):
  """Apply a CORAL normalization without mean shift on batch level."""
  return normalize_df_by_coral(df, center=False, center_cov=True)


def transform_df(df, rotate_mat, shift_vec):
  """Transform a DataFrame by rotating and shifting.

  Denote each row of df by x. Mathematically, it transforms x to
  x * rotate_mat^T + shift_vec^T. The transpose is due to that x is a row
  vector.

  Args:
    df: DataFrame with samples as rows
    rotate_mat: 2-D NumPy array of size p-by-p
    shift_vec: 2-D NumPy array of size p-by-1

  Returns:
    A transformed DataFrame.
  """
  return df.dot(rotate_mat.T) + np.squeeze(shift_vec)


def sum_of_square(a):
  """Sum of squared elements of Tensor a."""
  return tf.reduce_sum(tf.square(a))


def drop_unevaluated_comp(df):
  """Drop unevaluated compounds from a dataframe."""
  df = df[df.index.get_level_values(
      level=metadata.TREATMENT_GROUP) != metadata.NEGATIVE_CONTROL]
  df = df[df.index.get_level_values(level=metadata.MOA) != metadata.UNKNOWN]
  return df
