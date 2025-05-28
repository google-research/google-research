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

"""Retrieve dataset from BigQuery."""

import collections
import dataclasses
from typing import Callable, Dict, Optional, Tuple, Union

import bq_dataset
import feature_metadata
from google.cloud import bigquery
import tensorflow as tf


X_KEY = "x"
EMBED_IDX_KEY = "cat_embed_idx"
TARGET_KEY = "y"
_CAT_EMBED_DIM = 1
_PAGE_SIZE_MULTIPLIER = 300


@dataclasses.dataclass
class BQInfo:
  """Object containing BigQuery information.

  Attributes:
    bq_project: Name of BigQuery project.
    dataset_name: BiqQuery Dataset name.
    table_name: BigQuery table name.
  """

  bq_project: str
  dataset_name: str
  table_name: str


def create_embedding_fn(
    vocab,
):
  """Create embedding function for categorical features.

  Args:
    vocab: vocabulary for the categorical feature.

  Returns:
    Embedding function for the categorical feature.
  """
  if isinstance(list(vocab.keys())[0], int):
    lookup_layer = tf.keras.layers.IntegerLookup()
  else:
    lookup_layer = tf.keras.layers.StringLookup()

  lookup_layer.adapt(list(vocab.keys()))

  embed_layer = tf.keras.layers.Embedding(
      lookup_layer.vocabulary_size(), _CAT_EMBED_DIM
  )

  def look_up_and_embed(x):
    embed_idx = lookup_layer(x)
    return embed_layer(embed_idx), tf.expand_dims(embed_idx, -1)

  return look_up_and_embed


def create_target_process_fn(
    vocab,
    is_classification,
):
  """Create transform function for the target feature.

  Args:
    vocab: vocabulary for the target feature.
    is_classification: whether task is classification.

  Returns:
    Transform function for the target feature.
  """
  if is_classification:
    if isinstance(list(vocab.keys())[0], int):
      lookup_layer = tf.keras.layers.IntegerLookup()
    else:
      lookup_layer = tf.keras.layers.StringLookup()

    lookup_layer.adapt(list(vocab.keys()))
    return lookup_layer
  else:
    return lambda x: tf.expand_dims(x, -1)


def make_categorical_transform_fn(
    feature_data,
    target_key,
    task_type,
):
  """Make transform function for categorical features.

  Args:
    feature_data: metadata for the training BQ table.
    target_key: key of training target.
    task_type: training task type.

  Returns:
    Transform function for categorical features.

  Raises:
    ValueError: If the specified target key is not amongst the features.
  """
  cat_features = []
  numerical_features = []
  feature_embedding_fn = {}
  target_metadata = None
  for feature in feature_data:
    if feature.name == target_key:
      target_metadata = feature
      continue
    if feature.vocabulary:
      feature_embedding_fn[feature.name] = create_embedding_fn(
          feature.vocabulary
      )
      cat_features.append(feature.name)
    else:
      numerical_features.append(feature.name)
      assert feature.input_data_type.lower().startswith(
          "float"
      ), "The column dtype must be float"

  if target_metadata:
    target_process_fn = create_target_process_fn(
        target_metadata.vocabulary,
        is_classification=task_type == "classification",
    )
  else:
    raise ValueError("Target feature not found.")

  @tf.function
  def categorical_transform_fn(
      features,
  ):
    """Input is dictionary of all features in batch."""
    all_features = collections.defaultdict(list)
    # Putting categorical features first eases aggregation learning.
    for feature_name in cat_features:
      embedding, embed_idx = feature_embedding_fn[feature_name](
          features[feature_name]
      )
      all_features[X_KEY].append(embedding)
      all_features[EMBED_IDX_KEY].append(embed_idx)

    for feature_name in numerical_features:
      all_features[X_KEY].append(tf.expand_dims(features[feature_name], -1))

    if all_features[EMBED_IDX_KEY]:
      all_features[EMBED_IDX_KEY] = tf.concat(
          all_features[EMBED_IDX_KEY], axis=-1
      )

    all_features[X_KEY] = tf.concat(all_features[X_KEY], axis=-1)
    all_features[TARGET_KEY] = target_process_fn(features[target_key])
    return all_features

  return (
      categorical_transform_fn,
      list(cat_features),
      list(numerical_features),
  )


def get_data_from_bq_with_bq_info(
    bq_client,
    bq_info,
    batch_size = 256,
    limit = None,
    drop_remainder = False,
):
  """Obtains data or data generator from a BQ table, given the BQInfo.

  Args:
    bq_client: BigQuery client.
    bq_info: BQInfo object.
    batch_size: batch size.
    limit: Number of records to query from BigQuery table.
    drop_remainder: whether to drop last smaller-sized batch.

  Returns:
    TF Dataset containing specified data, and table metadata.
  """

  bq_metadata_builder = feature_metadata.BigQueryMetadataBuilder(
      bq_info.bq_project,
      bq_info.dataset_name,
      bq_info.table_name,
      bq_client=bq_client,
  )

  retrieval_options = feature_metadata.MetadataRetrievalOptions(
      get_mean=False,
      get_variance=False,
      get_min=True,
      get_max=False,
      get_median=False,
      get_log_mean=False,
      get_log_variance=False,
      min_log_value=0.0,
      number_of_quantiles=None,
      get_mode=False,
  )
  table_metadata = bq_metadata_builder.get_metadata_for_all_features(
      retrieval_options
  )

  dataset = bq_dataset.get_bigquery_dataset(
      table_metadata,
      bq_client=bq_client,
      batch_size=batch_size,
      page_size=batch_size * _PAGE_SIZE_MULTIPLIER,
      limit=limit,
      drop_remainder=drop_remainder,
  )
  options = tf.data.Options()
  # Avoid a large warning output by TF Dataset.
  options.deterministic = False
  dataset = dataset.with_options(options)
  return dataset, table_metadata
