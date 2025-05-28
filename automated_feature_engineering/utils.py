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

"""Compute entropy, to be used as diversity regularizer."""

import json
import os
# TODO(yihed): replace pickle with protobuf.
import pickle
import random
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from absl import logging
import bq_data
import feature_engineering
from google.cloud import bigquery
import numpy as np
import pandas as pd
from sklearn import cluster
from sklearn import covariance
from sklearn import decomposition
from sklearn import metrics
import tensorflow as tf
from tensorflow.io import gfile
import tensorflow_probability as tfp


# Number of buckets used for histogramming feature distribution.
N_BUCKETS = 100
CAT_DATA_TYPE = True  # 'categorical'
CONT_DATA_TYPE = False  # 'continuous'
_TEMPORAL_FEATURE_SPLIT_PATTERN = r'\s*,\s*'
_DATA_IMPUTE_VALUE = '0'
_TEMPORAL_CONTEXT_LEN = 10
_TEMPORAL_PLACEHOLDER = tf.zeros((_TEMPORAL_CONTEXT_LEN,))
_DATA_DIR = 'data'
_LABEL_COL = 'label'


def read_json_config(config_path):
  """Reads config file from gcs or local files.

  Args:
    config_path: path for configuration file that indicates feature types. It
      can be gsc file or local file.

  Returns:
    The dict of config from json file.
  """
  if config_path.startswith('gs://'):
    with gfile.GFile(config_path) as f:
      data_types = json.load(f)
  else:
    with open(config_path, 'r') as f:
      data_types = json.load(f)
  return data_types


def create_temporal_dataset(
    data,
    config_path,
    target_col_name,
    embedders,
    context_len = _TEMPORAL_CONTEXT_LEN,
    cat_embed_dim = 1,
):
  """Process data with temporal dimension.

  Args:
    data: the raw data based on which to create the temporal TF dataset.
    config_path: path for configuration file that indicates feature types.
    target_col_name: name of target column.
    embedders: embedders for categorical data.
    context_len: historical context length for temporal features.
    cat_embed_dim: dimension used for categorical embeddings.

  Returns:
    The created TF dataset based on parsing the provided raw data.
  """
  # TODO(yihed): make context_len configurable as input argument.
  # TODO(yihed): consider how cat_embed_dim > 1 impacts temporal
  # features shape.
  data_types = read_json_config(config_path)

  def data_gen(data, attribute_feature_names, temporal_feature_names):
    column_names = data_types.keys()
    for _, row in data.iterrows():
      attribute_features = []
      temporal_features = []
      for col_name in column_names:
        # Each col can be singleton attribute or a temporal feature.
        data_str = row.at[col_name]
        data_str = re.sub(r'\[|\]|"', '', data_str)
        data_str = re.sub(r'__MISSING__', _DATA_IMPUTE_VALUE, data_str)
        col = np.array(re.split(_TEMPORAL_FEATURE_SPLIT_PATTERN, data_str))
        if col_name in embedders:
          # categorical feature
          col = tf.squeeze(embedders[col_name](col)[: context_len + 1], -1)
        else:
          # numerical feature
          if len(col) > context_len:
            # Case for numerical feature separated by commas.
            entire_col = []
            for j in range(context_len + 1):
              entire_col.append(float(col[j]))
            col = np.array(entire_col, dtype=np.float32)
          else:
            col = np.array([col[0]], dtype=np.float32)

        if col_name == target_col_name:
          target_col = create_temporal_dataset_target_label(
              col, col_name, context_len, embedders
          )

        col = col[:context_len]
        if col_name in attribute_feature_names:
          attribute_features.append(col)
        elif col_name in temporal_feature_names:
          temporal_features.append(col)

      # TODO(yihed): handle case when either attribute or temporal features is
      # empty.
      yield (
          (tf.concat(attribute_features, 0), tf.concat(temporal_features, 0)),
          target_col,
      )

  attribute_feature_names = set()
  temporal_feature_names = set()
  attribute_dim = 0
  temporal_dim = 0
  sample_row = data.iloc[0]
  for col_name, col_type in data_types.items():
    col = sample_row.loc[data.columns == col_name].to_numpy()
    if len(re.split(_TEMPORAL_FEATURE_SPLIT_PATTERN, col[0])) > context_len:
      # Column contains temporal feature.
      if col_type == 'str':
        temporal_dim += context_len * cat_embed_dim
      else:
        temporal_dim += context_len
      temporal_feature_names.add(col_name)
    else:
      # Column contains attribute feature.
      if col_type == 'str':
        attribute_dim += cat_embed_dim
      else:
        attribute_dim += 1
      if col_name != target_col_name:
        attribute_feature_names.add(col_name)

  data_signature = (
      (
          tf.TensorSpec(shape=(attribute_dim,), dtype=tf.float32),
          tf.TensorSpec(shape=(temporal_dim,), dtype=tf.float32),
      ),
      # This assumes the target corresponds to single timestamp,
      # i.e. as in classification / regression.
      tf.TensorSpec(shape=(), dtype=tf.float32),
  )

  return tf.data.Dataset.from_generator(
      lambda: data_gen(data, attribute_feature_names, temporal_feature_names),
      output_signature=data_signature,
  )


def create_temporal_dataset_target_label(
    col,
    col_name,
    context_len,
    embedders,
):
  """Creates the target label for the temporal dataset.

  Args:
    col: the target column to extract label from.
    col_name: feature name for column.
    context_len: length of the prediction context window.
    embedders: embedding layers for categorical features.

  Returns:
    The prediction target label.
  """
  if len(col) > context_len:
    # Predict the next timestamp value if the target column is temporal.
    target_col = col[context_len]
  else:
    target_col = col[0]

  if col_name in embedders:
    target_col = tf.squeeze(embedders[col_name](target_col), -1)
  else:
    target_col = tf.constant(target_col, dtype=tf.float32)
  return target_col


def infer_discovered_features_and_save(
    dataset,
    model,
    is_classification,
    dataset_name,
    cat_feature_embeds,
    temporal_embedders = None,
):
  """Infers discovered features from trained model.

  Args:
    dataset: dataset to infer additional discovered features for.
    model: the trained model used for feature inference.
    is_classification: whether the prediction task is classification.
    dataset_name: name of dataset to infer discovered features for.
    cat_feature_embeds: embedding modules for categorical variables.
    temporal_embedders: embeddings for temporal features.

  Returns:
    Discovered features inferred from the trained model.
  """
  inferred_features = []
  feature_indices = None
  for submodule in model.submodules:
    if isinstance(submodule, feature_engineering.FeatureDiscoveryModel):
      discovery_module = submodule
      discovery_module.infer_features = True
      break
  else:
    raise ValueError('No FeatureDiscoveryModel found in model.')

  for batch in dataset:
    y = batch[-1]
    if cat_feature_embeds:
      cat_features = perform_feature_embedding(batch, cat_feature_embeds)
      batch_features = (cat_features,)
      idx_features = batch[0][:, : len(cat_feature_embeds)]
      model_features = (cat_features, _TEMPORAL_PLACEHOLDER, idx_features)
    else:
      # batch consists of two components: features and labels.
      batch_features = batch[0]
      if temporal_embedders:
        model_features = batch[0]
      else:
        model_features = (batch[0], _TEMPORAL_PLACEHOLDER)

    model(model_features, training=False)
    learned_features = discovery_module.learned_features

    if feature_indices is None:
      feature_indices = tf.squeeze(tf.where(learned_features[0] != 0))

    learned_features = tf.gather(learned_features, feature_indices, axis=-1)
    if is_classification:
      n_classes = y.shape[-1]
      y = tf.cast(tf.argmax(y, axis=-1), tf.float32)
    y = tf.expand_dims(y, -1)

    if not isinstance(batch_features, (tuple, list)):
      batch_features = (batch_features,)

    learned_features = tf.concat(
        [*batch_features] + [learned_features, y], axis=-1
    )
    inferred_features.append(learned_features)

  inferred_features = tf.concat(inferred_features, axis=0)
  features_filepath = get_inferred_features_filepath(dataset_name)
  features_config_filepath = get_inferred_features_config_filepath(dataset_name)
  features_shape = tuple(inferred_features.shape)
  print('Inferred features shape: {}'.format(features_shape))
  n_raw_features = features_shape[-1] - len(feature_indices) - 1
  features_config = {
      'features_shape': features_shape,
      'n_raw_features': n_raw_features,
  }
  if is_classification:
    features_config.update({'n_classes': n_classes})
  with open(features_config_filepath, 'wb') as f:
    pickle.dump(features_config, f, protocol=pickle.HIGHEST_PROTOCOL)

  fp = np.memmap(
      features_filepath, dtype='float32', mode='w+', shape=features_shape
  )

  fp[:] = inferred_features.numpy()[:]
  return inferred_features


def _concat_feature_names(
    feature_names,
    cat_features,
    numerical_features,
):
  """Combines all feature names."""
  return feature_names + cat_features + numerical_features + [_LABEL_COL]


def infer_and_upload_discovered_features(
    dataset,
    model,
    bq_client,
    project_id,
    dataset_name,
    table_name,
    feature_names,
    feature_ranking,
    cat_features,
    numerical_features,
    upload_to_bq = False,
):
  """Infers discovered features from trained model.

  Args:
    dataset: dataset to infer additional discovered features for.
    model: the trained model used for feature inference.
    bq_client: BigQuery client.
    project_id: BigQuery project id.
    dataset_name: name of dataset to infer discovered features for.
    table_name: name of table to infer features for.
    feature_names: list of feature names.
    feature_ranking: ranking of discovered features.
    cat_features: names of categorical features.
    numerical_features: names of numerical features.
    upload_to_bq: whether to upload inferred features to BQ.

  Returns:
    Name of the newly created table containing inferred features.
  """
  # feature_indices = None
  new_table_name = ''
  new_table = None
  for submodule in model.submodules:
    if isinstance(submodule, feature_engineering.FeatureDiscoveryModel):
      discovery_module = submodule
      discovery_module.infer_features = True
      break
  else:
    raise ValueError('No FeatureDiscoveryModel found in model.')

  if upload_to_bq:
    all_feature_names = _concat_feature_names(
        feature_names, cat_features, numerical_features
    )
    new_table_name, table_feature_names = create_bq_table(
        bq_client,
        project_id,
        dataset_name,
        table_name,
        all_feature_names,
    )
    if not new_table_name:
      # Table creation erred out, do not proceed.
      logging.warn('Failed to create feature table.')
      return new_table_name
    new_table = bq_client.get_table(new_table_name)
  else:
    table_feature_names = []

  for batch in dataset:
    x = batch[bq_data.X_KEY]
    target = batch[bq_data.TARGET_KEY]
    embed_idx = batch[bq_data.EMBED_IDX_KEY]

    model([x, embed_idx], training=False)
    learned_features = discovery_module.learned_features

    # if feature_indices is None:
    #  # feature_indices = tf.squeeze(tf.where(learned_features[0] != 0))
    #  # logging.info('Inferred features dim: %d', len(feature_indices))

    # learned_features = tf.gather(learned_features, feature_indices, axis=-1)

    if upload_to_bq:
      learned_features = tf.gather(learned_features, feature_ranking, axis=-1)
      all_features = tf.concat([learned_features, x, target], axis=-1)
      upload_features_to_bq(
          all_features, new_table, bq_client, table_feature_names
      )

  return new_table_name


def upload_features_to_bq(
    learned_features,
    table,
    bq_client,
    feature_names,
):
  """Uploads learned features to BigQuery."""

  rows_to_insert = pd.DataFrame(learned_features, columns=feature_names)

  bq_client.insert_rows_from_dataframe(table, rows_to_insert)


def create_bq_table(
    bq_client,
    project_id,
    dataset_name,
    current_table_name,
    feature_names,
):
  """Creates a new table in BigQuery based on given table name."""
  creation_time = time.strftime('%H%M%b%d')
  table_name = (
      f'{project_id}.{dataset_name}.{current_table_name}_{creation_time}'
  )
  # TODO(yihed): use BQ table creation API to create table instead.
  if isinstance(feature_names[0], bytes):
    feature_names = [b.decode('utf-8') for b in feature_names]

  def sub_str(x):
    x = re.sub(r'\<|\>|\)', '', x)
    return re.sub(r'\(|\s|;|,', '_', x)

  new_feature_names = [sub_str(s) for s in feature_names]
  schema_names = [f'{s} FLOAT64' for s in new_feature_names]
  schema_str = ', '.join(schema_names)

  option_str = (
      'OPTIONS(expiration_timestamp=TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL'
      ' 48 HOUR))'
  )

  query = f'CREATE TABLE {table_name} ({schema_str}) {option_str}'
  query_job = bq_client.query(query)
  try:
    query_job.result()  # Wait for request to finish
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.warn('Failed to create feature table due to error: %s', e)
    table_name = ''
  return table_name, new_feature_names


def get_inferred_features_filepath(dataset_name):
  """Retrieves filepath for cached discovered features inferred from model."""

  # Create data directory if one does not exist.
  if not os.path.exists(_DATA_DIR):
    os.makedirs(_DATA_DIR)
  return os.path.join(
      _DATA_DIR, '{}_inferred_features.npy'.format(dataset_name)
  )


def get_inferred_features_config_filepath(dataset_name):
  """Retrieves config filepath for cached discovered features inferred from model."""
  if not os.path.exists(_DATA_DIR):
    os.makedirs(_DATA_DIR)
  return os.path.join(
      _DATA_DIR, '{}_inferred_features_config.npy'.format(dataset_name)
  )


def load_inferred_features(
    dataset_name, features_shape
):
  """Loads discovered features inferred from trained model.

  Args:
    dataset_name: dataset name to load inferred features for.
    features_shape: expected shape of inferred features.

  Returns:
    Discovered features inferred from trained model.
  """
  inferred_features_filepath = get_inferred_features_filepath(dataset_name)
  inferred_features = np.memmap(
      inferred_features_filepath,
      dtype='float32',
      mode='r+',
      shape=features_shape,
  )
  return inferred_features


def compute_hsic_loss():
  """Constructs function that computes the HSIC objective between features & labels.

  References: https://arxiv.org/pdf/1908.01580.pdf.
  https://pure.mpg.de/rest/items/item_1791468/component/file_3009860/content.

  Returns:
    Function that computes the HSIC (Hilbert-Schmidt independence criterion)
    objective between training features & labels.
  """

  def hsic_loss_fn(x_train, labels, sparse_mask):
    batch_sz = x_train.shape[0]

    # Only need to standard normalize the features, and not the labels,
    # since different features live on different scales, but the label
    # space has only one scale.
    x_std = tf.math.reduce_std(x_train, axis=0, keepdims=True)
    x_mean = tf.math.reduce_mean(x_train, axis=0, keepdims=True)
    x_train = tf.math.divide_no_nan((x_train - x_mean), x_std)

    # Condition loss on learned mask
    x_train *= sparse_mask
    # Square l2 distance between points
    x_dist = tf.reduce_sum(
        (tf.expand_dims(x_train, 1) - tf.expand_dims(x_train, 0)) ** 2, axis=-1
    )
    # Labels are 2d vectors, i.e. categorical labels have been converted to
    # one-hot, hence distance makes sense.
    y_dist = tf.reduce_sum(
        (tf.expand_dims(labels, 1) - tf.expand_dims(labels, 0)) ** 2, axis=-1
    )
    # Use gaussian kernel. 2 is normalization constant.
    x_dist = tf.math.exp(
        -tf.math.divide_no_nan(x_dist, 2 * tf.math.reduce_std(x_dist) ** 2)
    )
    y_dist = tf.math.exp(
        -tf.math.divide_no_nan(y_dist, 2 * tf.math.reduce_std(y_dist) ** 2)
    )

    # Centering
    x_centered = x_dist - tf.reduce_mean(x_dist, -1, keepdims=True)
    y_centered = y_dist - tf.reduce_mean(y_dist, -1, keepdims=True)
    return (
        -tf.linalg.trace(tf.matmul(x_centered, y_centered))
        / (batch_sz - 1) ** 2
    )

  return hsic_loss_fn


def compute_mi_loss(
    data_types,
    is_classification,
    do_probabilistic_loss = False,
    alpha = 20.0,
    cont_features_diff_weight = 0.7,
    verbose = False,
):
  """Constructs function that computes mutual information between data & labels.

  Args:
    data_types: Type for each feature, whether categorical or continuous.
    is_classification: Whether task if classification.
    do_probabilistic_loss: Whether to compute MI loss component based on the
      learned probabilistic mask. This requires more computation than the
      quadratic MI loss component.
    alpha: Weight for probability loss; controls the contribution from the
      consistency constraint.
    cont_features_diff_weight: Weight for the consistency .
    verbose: Whether to do verbose logging.

  Returns:
    Function that computes mutual information between training data & labels.
  """

  def mi_loss_fn(x_train, logits, labels, sparse_mask, step):

    batch_sz = x_train.shape[0]
    if is_classification:
      logits = tf.nn.softmax(logits, axis=-1)

    logits = tf.squeeze(logits)
    if do_probabilistic_loss:
      sparse_mask = tf.squeeze(sparse_mask)
      features1 = tf.expand_dims(x_train, axis=1)
      features2 = tf.expand_dims(x_train, axis=0)
      all_prob_loss = []
      # Breaking into chunks to ensure scalability.
      chunk_sz = 128
      i = 0

      while i < batch_sz:
        upper = min(batch_sz, i + chunk_sz)
        features_diff = features1[i:upper] - features2

        features_diff = tf.transpose(features_diff, [2, 0, 1])
        cat_features_diff = tf.boolean_mask(features_diff, data_types)
        cat_features_diff = tf.transpose(cat_features_diff, [1, 2, 0])
        cat_mask = tf.boolean_mask(sparse_mask, data_types)

        cat_prob = tf.math.reduce_prod(
            tf.where(cat_features_diff > 0, 1 - cat_mask, 1.0), axis=-1
        )

        cont_data_types = True ^ data_types
        cont_features_diff = tf.boolean_mask(features_diff, cont_data_types)
        cont_features_diff = tf.transpose(cont_features_diff, [1, 2, 0])
        cont_mask = tf.boolean_mask(sparse_mask, cont_data_types)
        cont_prob = tf.math.reduce_prod(
            1
            - (
                1
                - tf.math.exp(
                    -tf.math.abs(cont_features_diff * cont_features_diff_weight)
                )
            )
            * cont_mask,
            axis=-1,
        )

        prob = cat_prob * cont_prob
        logits_diff = (
            tf.expand_dims(logits[i:upper], 1) - tf.expand_dims(logits, 0)
        ) ** 2

        if not is_classification:
          prob_loss = tf.reduce_sum(tf.squeeze(logits_diff) * prob)
        else:
          prob_loss = tf.reduce_sum(tf.reduce_sum(logits_diff, axis=-1) * prob)

        all_prob_loss.append(prob_loss)
        i += chunk_sz

      prob_loss = tf.reduce_sum(all_prob_loss)
    else:
      prob_loss = 0

    # MI loss component not based on probability mask:
    if not is_classification:
      square_loss = tf.reduce_sum((logits - labels) ** 2)
    else:
      pos_loss = tf.reduce_sum(
          (1 - tf.ragged.boolean_mask(logits, labels > 0)) ** 2, -1
      )  ##
      neg_loss = tf.reduce_sum(
          tf.ragged.boolean_mask(logits, labels == 0) ** 2, -1
      )  ##
      square_loss = tf.reduce_sum(pos_loss + neg_loss)

    # add alpha hyperparam
    if verbose and step % 100 == 0:
      print(
          'prob_loss {} square_loss {} %%logits{} sparse_mask non-zero '
          'indices {}'.format(
              prob_loss,
              square_loss,
              logits[:100],
              tf.squeeze(tf.where(sparse_mask > 0)),
          )
      )

    return (prob_loss * alpha + square_loss) / batch_sz

  return mi_loss_fn


def lassonet_select(
    x_train,
    y_train,
    num_features,
    batch_size,
    is_classification,
    chunk_sz = 3000,
):
  """Uses LassonNet for feature extraction.

  Args:
    x_train: Training features. As a dataframe.
    y_train: Training labels. As a dataframe.
    num_features: Number of selected features.
    batch_size: Training batch size for lassonet.
    is_classification: Whether is classification task.
    chunk_sz: Chunk size for splitting the processing.

  Returns:
    Indices of selected features.
  """
  try:
    # lassonet is imported here, as it is not required to run the main SLM
    # functions.
    import lassonet  # pylint: disable=g-import-not-at-top, import-error
  except ImportError as e:
    logging.warning('Lassonet installation is required.')
    raise e
  hidden_dims = (random.randrange(64, 128),)
  dropout = random.uniform(0.0, 0.5)
  # l2 penalization on skip connection, defaults to 0.
  gamma = random.uniform(0.0, 0.4)
  # hiearchy param, defaults to 10
  m_param = random.uniform(8.0, 15.0)
  if is_classification:
    model = lassonet.LassoNetClassifier(
        batch_size=batch_size,
        M=m_param,
        dropout=dropout,
        hidden_dims=hidden_dims,
        gamma=gamma,
    )
  else:
    model = lassonet.LassoNetRegressor()

  start_idx = 0
  all_scores = []
  while True:
    if start_idx > len(y_train):
      break
    end_idx = start_idx + chunk_sz
    model_params = model.path(
        x_train.iloc[start_idx:end_idx, :].values,
        y_train[start_idx:end_idx].values,
    )
    for mod in model_params:
      model.load(mod.state_dict)
    scores = model.feature_importances_.numpy()
    all_scores.append(scores)
    start_idx += chunk_sz

  all_scores = np.mean(all_scores, axis=0)
  return np.argsort(all_scores)[:num_features]


def pfa_select(x_train, num_features):
  """Uses PFA (principal feature analysis) for feature extraction.

  Args:
    x_train: training features.
    num_features: number of features to be selected.

  Returns:
    Indices of selected features.
  """
  cov = covariance.EmpiricalCovariance().fit(x_train)
  x_cov = cov.covariance_
  pca = decomposition.PCA(svd_solver='randomized').fit(x_cov)
  cov_svecs = pca.components_
  kmeans = cluster.KMeans(num_features).fit(cov_svecs)
  centers = kmeans.cluster_centers_
  dist = metrics.pairwise_distances(centers, cov_svecs)
  rank_idx = np.argmin(dist, axis=-1)
  return rank_idx


def compute_entropy(feature_mx, eps = 1e-5):
  """Computes the entropy of the selected features.

  Args:
    feature_mx: Tensor of selected features.
    eps: Epsilon factor to achieve positive definiteness.

  Returns:
    Von Neumann entropy of the covariance of the features.
  """
  mx_max = tf.reduce_max(feature_mx, axis=0)
  mx_min = tf.reduce_min(feature_mx, axis=0)
  histograms = []

  for i, (cur_max, cur_min) in enumerate(zip(mx_max, mx_min)):
    interval_len = (cur_max - cur_min) / (N_BUCKETS - 1)
    edges = tf.concat([cur_min + j * interval_len for j in range(N_BUCKETS)], 0)
    histogram = tfp.stats.histogram(feature_mx[:, i], edges)
    histogram /= tf.reduce_sum(histogram)
    histograms.append(histogram)

  prob_mx = tf.stack(histograms, 0)
  # prob_mx has shape (n_features, N_BUCKETS-1)
  cov_mx = tfp.stats.covariance(prob_mx, sample_axis=-1, event_axis=0)
  # logm only takes complex entries

  cov_mx = tf.cast(
      cov_mx + tf.eye(cov_mx.shape[0], dtype=cov_mx.dtype) * eps, tf.complex64
  )

  # TODO(yihed): this entropy covers cross-feature entropy, also compute
  # intra-feature entropy (easier to compute.)
  entropy = -tf.linalg.trace(tf.matmul(cov_mx, tf.linalg.logm(cov_mx)))

  # Since the cov_mx is positive semidefinite, the complex part is 0.
  return tf.math.real(entropy)


def compare_entropy(
    x_train, mask, n_random_sampling = 5
):
  """Compares the entropy of the selected features w/ randomly sampled features.

  Args:
    x_train: Training data.
    mask: Feature selection mask.
    n_random_sampling: Number of random samplings to take to compare entropy.
  """
  selected_idx = tf.squeeze(tf.where(mask > 0), -1)
  num_selected_features = len(selected_idx)
  num_features = x_train.shape[-1]

  mask_features = tf.gather(x_train, selected_idx, axis=-1)
  mask_feature_entropy = compute_entropy(mask_features)
  avg_entropies = []

  # randomly sample subsets of features
  for _ in range(n_random_sampling):
    selected_idx = tf.random.uniform(
        (num_selected_features,), minval=0, maxval=num_features, dtype=tf.int64
    )
    cur_features = tf.gather(x_train, selected_idx, axis=-1)

    mask_feature_entropy = compute_entropy(cur_features)
    avg_entropies.append(mask_feature_entropy)

  avg_entropy = np.mean(avg_entropies)
  print(f'mask entropy {mask_feature_entropy}. Mean entropy {avg_entropy}')


def def_weighted_ce(y_train, max_pos_weight = 10.0):
  """Defines weighted cross entropy loss."""

  pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
  pos_weight = min(pos_weight, max_pos_weight)

  def weighted_ce(labels, pred, pos_weight=pos_weight):
    return tf.reduce_mean(
        tf.nn.weighted_cross_entropy_with_logits(
            labels, pred, pos_weight=pos_weight
        )
    )

  return weighted_ce


def get_feature_types(x_train):
  """Decides feature column types."""
  feature_types = []
  for i, col in enumerate(x_train.iloc[0]):
    if (
        isinstance(col, str)
        or col is None
        or x_train.dtypes[i] == np.dtype('O')
    ):
      feature_types.append(CAT_DATA_TYPE)
    else:
      feature_types.append(CONT_DATA_TYPE)

  return tf.constant(feature_types, dtype=tf.bool)


def create_temporal_embedders(
    x_train,
    config_path,
    context_len = _TEMPORAL_CONTEXT_LEN,
    cat_embed_dim = 1,
    n_rows_for_vocab = 1000,
):
  """Create embedders for categorical features, including temporal features.

  Args:
    x_train: training data that created embedders are based on.
    config_path: path for configuration file that indicates feature types.
    context_len: historical context length for temporal features.
    cat_embed_dim: dimension used for categorical embeddings.
    n_rows_for_vocab: number of samples from which to extract vocabulary.

  Returns:
    Categorical embedding layers that have been adapted to feature vocabularies.
  """
  data_types = read_json_config(config_path)

  embedders = {}
  n_temporal_features = 0
  for col_name, data_type in data_types.items():
    feature_col = x_train.loc[:, x_train.columns == col_name].to_numpy()
    if (
        len(re.split(_TEMPORAL_FEATURE_SPLIT_PATTERN, feature_col[0][0]))
        > context_len
    ):
      split_col = True
      n_temporal_features += 1
    else:
      split_col = False

    if data_type != 'str':
      continue

    vocab_set = set()
    for i, row in enumerate(feature_col):
      if split_col:
        row = re.split(_TEMPORAL_FEATURE_SPLIT_PATTERN, row[0])
      vocab_set.update(row)
      if i > n_rows_for_vocab:
        break

    lookup_layer = tf.keras.layers.StringLookup()

    lookup_layer.adapt(list(vocab_set))
    embed_layer = tf.keras.layers.Embedding(
        len(lookup_layer.get_vocabulary()), cat_embed_dim
    )

    def embed_func(x, embed_layer=embed_layer, lookup_layer=lookup_layer):
      return embed_layer(lookup_layer(x))

    embedders[col_name] = embed_func

  n_attribute_features = len(data_types) - n_temporal_features
  return embedders, n_attribute_features, n_temporal_features


def category_to_idx(
    x_train, x_test, normalize_range = False
):
  """Reconstruct dataframe to convert categories into indices.

  Args:
    x_train: Training set dataframe.
    x_test: Test set dataframe.
    normalize_range: Whether to range-normalize (max_min) the features.

  Returns:
    Updated dataframe where categorical features are indices;
    as well as supporting categorical data information.
  """
  cat_feat_train = []
  cont_feat_train = []
  cat_feat_test = []
  cont_feat_test = []
  vocab_lengths = []
  train_index = x_train.index
  test_index = x_test.index

  # TODO(yihed): check if affect MI datatypes computation.
  for i, col in enumerate(x_train.iloc[0]):
    if (
        isinstance(col, str)
        or col is None
        or x_train.dtypes[i] == np.dtype('O')
    ):

      col_train = x_train.iloc[:, i]
      lookup_layer = tf.keras.layers.StringLookup()
      # Replace missing values with a negative number that does not appear.
      missing_value_rep = '-1'
      col_train = col_train.fillna(missing_value_rep)
      col_test = x_test.iloc[:, i].fillna(missing_value_rep)
      lookup_layer.adapt(col_train)

      vocab_lengths.append(len(lookup_layer.get_vocabulary()))
      cat_feat_train.append(lookup_layer(col_train).numpy().astype(np.float32))
      cat_feat_test.append(lookup_layer(col_test).numpy().astype(np.float32))
    else:
      col_train = x_train.iloc[:, i]
      col_test = x_test.iloc[:, i]

      if normalize_range:
        col_min = col_train.dropna().min()
        col_range = max(col_train.dropna().max() - col_min, 0.001)
        col_train = (col_train - col_min) / col_range
        col_test = (col_test - col_min) / col_range

      missing_value_rep = col_train.dropna().median()
      cont_feat_train.append(
          col_train.fillna(missing_value_rep).values.astype(np.float32)
      )
      cont_feat_test.append(
          col_test.fillna(missing_value_rep).values.astype(np.float32)
      )

  if not cat_feat_train:
    raise ValueError('Input data do not contain any categorical features.')
  # Construct new data with categorical features at the beginning
  cat_feat_train.extend(cont_feat_train)
  cat_feat_test.extend(cont_feat_test)

  feat_train = np.stack(cat_feat_train, axis=-1)
  feat_test = np.stack(cat_feat_test, axis=-1)

  feat_train = pd.DataFrame(feat_train)
  feat_train.index = train_index
  feat_test = pd.DataFrame(feat_test)
  feat_test.index = test_index

  return (feat_train, feat_test, vocab_lengths)


def perform_feature_embedding(
    batch, cat_feature_embeds
):
  """Embeds categorical features.

  Args:
    batch: Dataset batch.
    cat_feature_embeds: Categorical feature embeddings

  Returns:
    Embedded features.
  """
  cat_features = []

  for i, feature_embed in enumerate(cat_feature_embeds):
    # Category features at beginning of x
    cat_features.append(feature_embed(batch[0][:, i]))

  cat_features = tf.concat(cat_features, axis=-1)
  return tf.concat(
      [cat_features, batch[0][:, len(cat_feature_embeds) :]], axis=-1
  )
