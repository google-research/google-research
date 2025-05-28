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

"""Creates feature metadata for use in normalization and feature creation."""

import collections
import dataclasses
import enum
import logging
import textwrap
# TODO(b/249435057): Convert imports to use collections.abc once the container
#  is updated to have python 3.9.
from typing import Any, DefaultDict, Iterable, Mapping, MutableMapping, MutableSequence, Optional, Sequence, Set, Union

import bq_utils
from google.cloud import bigquery
import numpy as np
import tensorflow as tf


# https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types
_BIGQUERY_SUPPORTED_NUMERIC_DATA_TYPES = frozenset((
    'INT64',
    'NUMERIC',
    'BIGNUMERIC',
    'FLOAT64',
    'FLOAT',
    'INTEGER',
))
# Decide if we should handle dates and times as numerical as well.
# Research if there are other data types we should try to handle.
_BIGQUERY_SUPPORTED_DISCRETE_DATA_TYPES = frozenset((
    'STRING',
    'BYTES',
    'BOOL',
    'BOOLEAN',
    'INT64',
    'INTEGER',
    'DATE',
    'DATETIME',
    'TIMESTAMP',
    'TIME',
))
_BIGQUERY_SUPPORTED_DATA_TYPES = frozenset((
    _BIGQUERY_SUPPORTED_NUMERIC_DATA_TYPES
    | _BIGQUERY_SUPPORTED_DISCRETE_DATA_TYPES
))

_USE_LEGACY_SQL = False


@enum.unique
class Normalization(str, enum.Enum):
  """GAIN normalization methods."""

  MINMAX = 'minmax'
  STANDARD = 'standard'


@dataclasses.dataclass
class MetadataRetrievalOptions:
  """Describes what metadata should be collected."""

  # For numeric features
  get_mean: bool = True
  get_variance: bool = True
  get_min: bool = True
  get_max: bool = True
  get_median: bool = False
  get_log_mean: bool = True
  get_log_variance: bool = True
  number_of_quantiles: Optional[int] = None

  # Transform related parameters.
  min_log_value: float = 1.0e-4

  # For discrete features
  get_mode: bool = True
  max_vocab_size: int = 5000

  # BigQuery specific options.
  where_clauses: Sequence[str] = dataclasses.field(default_factory=tuple)

  def no_options(self, ignored_fields = None):
    """True if metadata statistics do not need to be retrieved.

    Args:
      ignored_fields: Any fields to not check for a boolean true value. If None
        is passed in the default of {'min_log_value'} will be used.

    Returns:
      True if none of the important attributes are true.
    """
    if ignored_fields is None:
      ignored_fields = {'min_log_value', 'where_clauses'}

    for field in dataclasses.fields(self):
      if field.name in ignored_fields:
        continue

      if getattr(self, field.name):
        return False

    return True

  @classmethod
  def get_all(
      cls,
      number_of_quantiles = 101,
      max_vocab_size = 10000,
      where_clauses = tuple(),
  ):
    """Creates a MetadataRetrievalOptions object with all values True.

    Args:
      number_of_quantiles: The number of quantiles to use.
      max_vocab_size: The maximum vocab size to use.
      where_clauses: A list of clauses that can be combined with and statements
        to get the correct values in the BigQuery table.

    Returns:
      A MetadataRetrievalOptions that gets all metadata classes.
    """
    return cls(
        get_mean=True,
        get_variance=True,
        get_min=True,
        get_max=True,
        get_median=True,
        get_log_mean=True,
        get_log_variance=True,
        number_of_quantiles=number_of_quantiles,
        get_mode=True,
        max_vocab_size=max_vocab_size,
        where_clauses=where_clauses,
    )

  @classmethod
  def get_none(
      cls,
      where_clauses = tuple(),
  ):
    """Creates a MetadataRetrievalOptions object that doesn't get any metadata.

    Args:
      where_clauses: A list of clauses that can be combined with and statements
        to get the correct values in the BigQuery table.

    Returns:
      A MetadataRetrievalOptions that gets no metadata.
    """
    return cls(
        get_mean=False,
        get_variance=False,
        get_min=False,
        get_max=False,
        get_median=False,
        get_log_mean=False,
        get_log_variance=False,
        min_log_value=0.0,
        number_of_quantiles=None,
        get_mode=False,
        max_vocab_size=0,
        where_clauses=where_clauses,
    )

  @classmethod
  def for_normalization(
      cls,
      normalization,
      where_clauses = tuple(),
  ):
    """Returns the metadata retrieval options needed for a normalization.

    Args:
      normalization: The normalization method to be used. Currently, supports
        minmax and standard.
      where_clauses: A list of clauses that can be combined with and statements
        to get the correct values in the BigQuery table.

    Returns:
      A MetadataRetrievalOptions with the necessary options being True.

    Raises:
      ValueError: If an invalid normalization is passed in.
    """
    if normalization == Normalization.MINMAX:
      return cls(
          get_mean=False,
          get_variance=False,
          get_min=True,
          get_max=True,
          get_median=False,
          get_log_mean=False,
          get_log_variance=False,
          get_mode=False,
          where_clauses=where_clauses,
      )
    elif normalization == Normalization.STANDARD:
      return cls(
          get_mean=True,
          get_variance=True,
          get_min=False,
          get_max=False,
          get_median=False,
          get_log_mean=False,
          get_log_variance=False,
          get_mode=False,
          where_clauses=where_clauses,
      )
    else:
      raise ValueError(f'The normalization {normalization} is not valid.')


# noinspection PyUnresolvedReferences
@dataclasses.dataclass
class FeatureMetadata:
  """The metadata for a single data column.

  Attributes:
    name: The name of the column.
    index: The index of the column (e.g. 0-N-1).
    input_data_type: The data_type of the column in BigQuery
    tf_data_type_str: The data type that should be used for the feature in
      TensorFlow as a string.
    mean: The mean of the feature. For numeric data only.
    variance: The variance of the feature. For numeric data only.
    min: The minimum value of the feature. For numeric data only.
    max: The maximum value of the feature. For numeric data only.
    median: The mean of the feature. For numeric data only.
    log_shift: The value to shift the data by before applying a log transform.
      This is effectively MIN(0, eps - min(X)). Therefore the log transform
      should be applied as log(X + log_shift). Will be present if log_mean or
      log_variance are present. For numeric data only.
    log_mean: The mean of the feature after the log transform. For numeric data
      only.
    log_variance: The variance of the feature after the log transform. For
      numeric data only.
    quantiles: A list of quantiles evenly spaced between 0% (the min) and 100%
      (the max). As an example if there are three values they are the min, the
      median, and the max.
    cardinality: The number of different values the feature takes on in the
      table. For discrete data only.
    mode: The most common value of the feature. For discrete data only.
    vocabulary: A mapping between the feature values and the number of times
      that they occur in the data. Note that this may not be present for some
      discrete data whose cardinality is too large. For discrete data only.
    is_numeric: True if the BigQuery datatype is numeric.
    is_discrete: True it fhe BigQuery datatype is discrete.
  """

  name: str
  index: int
  input_data_type: Optional[str] = None
  tf_data_type_str: Optional[str] = None
  # These are related to numeric features.
  mean: Optional[float] = None
  variance: Optional[float] = None
  min: Optional[float] = None
  max: Optional[float] = None
  median: Optional[float] = None
  log_shift: Optional[float] = None
  log_mean: Optional[float] = None
  log_variance: Optional[float] = None
  quantiles: Optional[Sequence[float]] = None
  # These are related to discrete features.
  # Consider if we should calculate cardinality for all features
  #  as sometimes float fields can only take on a set of discrete values.
  # This case was commonly seen in some of the Uber data.
  cardinality: Optional[int] = None
  mode: Optional[Any] = None
  vocabulary: Optional[Mapping[Any, int]] = None

  @property
  def tf_data_type(self):
    """The TensorFlor data type of the feature."""
    return (
        tf.dtypes.as_dtype(self.tf_data_type_str)
        if self.tf_data_type_str
        else None
    )

  @tf_data_type.setter
  def tf_data_type(self, tf_data_type):
    tf_dtype = tf.dtypes.as_dtype(tf_data_type)
    self.tf_data_type_str = tf_dtype.name

  @property
  def range(self):
    """The range (max-min) of the data."""
    if self.max is None or self.min is None:
      return None

    return self.max - self.min

  @property
  def is_numeric(self):
    """If the feature is numeric."""
    if self.tf_data_type_str:
      return self.tf_data_type.is_floating or self.tf_data_type.is_integer  # pytype: disable=attribute-error  # always-use-return-annotations
    # Handle other input sources more generically.
    else:
      return self.input_data_type in _BIGQUERY_SUPPORTED_NUMERIC_DATA_TYPES

  @property
  def is_discrete(self):
    """If the feature is discrete."""
    tf_data_type = self.tf_data_type
    if tf_data_type:
      return (
          tf_data_type.is_integer
          or tf_data_type.is_bool
          or tf_data_type == tf.dtypes.string
      )
    # Handle other input sources more generically.
    else:
      return self.input_data_type in _BIGQUERY_SUPPORTED_DISCRETE_DATA_TYPES

  def get_config(self):
    """Creates a config that can be converted to json."""
    config = dataclasses.asdict(self)
    # Because JSON only takes strings as keys we need to track and convert
    # types.
    if self.vocabulary:
      # Consider whether we should limit the vocabulary to one
      #  type.
      config['vocabulary_types'] = {
          v: type(v).__name__ for v in self.vocabulary
      }
    return config

  @classmethod
  def from_config(cls, config):
    """Creates a FeatureMetadata from a config."""

    def _update_vocabulary_key(new_key, old_key):
      config['vocabulary'][new_key] = config['vocabulary'][old_key]
      del config['vocabulary'][old_key]

    vocabulary_types = config.pop('vocabulary_types', {})
    for k, k_type in vocabulary_types.items():
      if k_type == 'bool':
        bool_key = k.lower() == 'true'
        _update_vocabulary_key(bool_key, k)
      elif k_type == 'NoneType':
        _update_vocabulary_key(None, k)
      elif k_type == 'int':
        _update_vocabulary_key(int(k), k)
      elif k_type == 'float':
        _update_vocabulary_key(float(k), k)

    return cls(**config)


class FeatureMetadataContainer:
  """Allows feature metadata to be easily retrieved.

  Example::

    # Pass retrieved metadata into the container.
    builder = BigQueryMetadataBuilder('project', 'dataset', 'table')
    all_metadata = builder.get_feature_names_and_types()
    metadata_container = FeatureMetadataContainer(all_metadata)

    # Get a feature by name.
    feature1_metadata = metadata_container.get_metadata_by_name('feature1')
    assert feature1_metadata.name == 'feature1'

    # Get features by dtype.
    float_features = metadata_container.get_metadata_by_name('FLOAT64')
    assert all([ff.input_data_type == 'FLOAT64' for ff in float_features])

    # Get a feature by index and check it matches.
    assert metadata_container[0] == all_metadata[0]

    # Iterate through the metadata and check it matches.
    for idx, c_metadata in enumerate(metadata_container):
      assert c_metadata == all_metadata[idx]
  """

  def __init__(
      self,
      all_feature_metadata,
  ):
    """Container that holds feature metadata for easy retrieval.

    Args:
      all_feature_metadata: A sequence of the metadata for the features in the
        table.
    """
    # Internal storage of the input metadata as a sequence.
    self._metadata_sequence = (
        all_feature_metadata._metadata_sequence
        if isinstance(all_feature_metadata, FeatureMetadataContainer)
        else all_feature_metadata
    )

    # Lazily created cache for mapping output features to data types.
    self._metadata_for_dtype: Optional[
        DefaultDict[str, MutableSequence[FeatureMetadata]]
    ] = None
    # Lazily created cache for quickly getting feature metadata by name.
    self._metadata_for_name: Optional[Mapping[str, FeatureMetadata]] = None

  @property
  def feature_metadata_by_names(self):
    """A mapping between the name and metadata of each feature.

    Raises:
      ValueError: If two features have the same name.
    """
    if not self._metadata_for_name:
      self._metadata_for_name = {m.name: m for m in self._metadata_sequence}
      # Make sure all names are unique
      if len(self._metadata_for_name) != len(self._metadata_sequence):
        raise ValueError('Two features may not have the same name.')

    return self._metadata_for_name

  def get_metadata_by_name(self, feature_name):
    """Returns the metadata for the feature with the given name.

    Args:
      feature_name: The name of the feature (i.e. metadata.name).

    Returns:
      The metadata object whose name matches the input name.
    """
    return self.feature_metadata_by_names[feature_name]

  def __getitem__(self, idx):
    """Allow user to use access the metadata sequence directly.

    Args:
      idx: The index (based on input order) of the feature to retrieve.

    Returns:
      The metadata for the specified feature.
    """
    return self._metadata_sequence[idx]

  @property
  def feature_metadata_by_dtypes(
      self,
  ):
    """A mapping between  data type and metadata for all features."""
    if not self._metadata_for_dtype:
      self._metadata_for_dtype = collections.defaultdict(list)
      for m in self._metadata_sequence:
        self._metadata_for_dtype[m.input_data_type].append(m)

    return self._metadata_for_dtype

  def get_metadata_by_dtype(
      self, input_data_type
  ):
    """Returns a set of metadata for features of the given type."""
    return self.feature_metadata_by_dtypes[input_data_type]

  @property
  def names(self):
    return tuple(self.feature_metadata_by_names.keys())

  def __iter__(self):
    """Iterates through the metadata in the order they were provided."""
    for metadata in self._metadata_sequence:
      yield metadata

  def __len__(self):
    """Returns the number of elements in the metadata."""
    return len(self._metadata_sequence)

  def __repr__(self):
    metadata_str = ','.join([f'{f!r}' for f in self])
    return f'{type(self)!r}(all_feature_metadata=[{metadata_str}])'

  def to_bigquery_schema(self):
    """Creates a BigQuery schema from the feature metadata.

    Returns:
      A list of the SchemaFields for each of the features.
    """
    # Add in the nullable attribute.
    return [bigquery.SchemaField(m.name, m.input_data_type) for m in self]

  def equal_names_and_types(
      self,
      other,
      difference_method = None,
  ):
    """Returns where the name and datatypes of the two containers is equal.

    Args:
      other: Another metadata container to compare to.
      difference_method: How to handle differences. Valid values are 'raise',
        logger levels or None. For logging levels see:
        https://docs.python.org/3/library/logging.html#logging-levels.

    Returns:
      True if all the names and types of the two schema match.

    Raises:
      ValueError: If the difference method is raise and the two values do not
        match or an invalid difference_method is specified.
    """
    fields_to_check = ('name', 'input_data_type', 'tf_data_type')

    if difference_method == 'raise':

      def difference_fn(message):
        raise ValueError(message)

    elif difference_method:
      numeric_level = getattr(logging, difference_method.upper(), None)
      if numeric_level is None:
        raise ValueError(
            f'The difference_method {difference_method} is not valid.'
        )

      def difference_fn(message):
        logging.log(numeric_level, message)

    else:

      def difference_fn(_):
        pass

    if len(self) != len(other):
      difference_fn(
          f'The length of the two containers is different: {self} vs {other}'
      )
      return False

    different_features = []
    for feature, other_feature in zip(self, other):
      for field_name in fields_to_check:
        if getattr(feature, field_name) != getattr(other_feature, field_name):
          different_features.append((feature, other_feature))
          continue

    if different_features:
      features_str = [
          f'{mine} vs {other}' for mine, other in different_features
      ]
      difference_fn(f'The following features differ: {";".join(features_str)}')
      return False

    return True

  def get_config(self):
    """Creates a config that can be converted to json."""
    return {
        'all_feature_metadata': [
            fm.get_config() for fm in self._metadata_sequence
        ],
    }

  @classmethod
  def from_config(
      cls,
      config,
  ):
    """Create a FeatureMetadataContainer from a config."""
    if 'all_feature_metadata' not in config:
      raise ValueError('The key all_feature_metadata must be in the config.')

    config['all_feature_metadata'] = [
        FeatureMetadata.from_config(fm) for fm in config['all_feature_metadata']
    ]
    return cls(**config)


class BigQueryTableMetadata(FeatureMetadataContainer):
  """Contains the metadata for features in a BigQuery table.

  This extends the normal FeatureMetadataContainer by including information
  about the datasource (i.e. the BigQuery Table).

  Attributes:
    project_id: The name of the GCP project that the BigQuery table is in.
    bq_dataset_name: The name of the dataset that the BigQuery table is in.
    bq_table_name: The name of the table from which the metadata was retrieved.
  """

  def __init__(
      self,
      all_feature_metadata,
      project_id,
      bq_dataset_name,
      bq_table_name,
  ):
    """Initializer for a BigQueryTableMetadata object.

    Args:
      all_feature_metadata: Sequence of FeatureMetadata for the columns.
      project_id: The name of the GCP project that the BigQuery table is in.
      bq_dataset_name: The name of the dataset that the BigQuery table is in.
      bq_table_name: The name of the table from which the metadata was
        retrieved.
    """
    super().__init__(all_feature_metadata)

    self.project_id = project_id
    self.bq_dataset_name = bq_dataset_name
    self.bq_table_name = bq_table_name

  @property
  def full_table_id(self):
    """The full path 'project.dataset.table' of the BigQuery table."""
    return f'{self.project_id}.{self.bq_dataset_name}.{self.bq_table_name}'

  @property
  def escaped_table_id(self):
    """The full path of the BigQuery table escaped with `."""
    return f'`{self.full_table_id}`'

  @property
  def bigquery_table(self):
    """The bigquery.Table object for this table."""
    return bigquery.Table.from_string(self.full_table_id)

  def update_bq_path_parts(self, bq_file_parts):
    self.project_id = bq_file_parts.project_id
    self.bq_dataset_name = bq_file_parts.bq_dataset_name
    self.bq_table_name = bq_file_parts.bq_table_name

  def __repr__(self):
    metadata_str = ','.join([f'{f!r}' for f in self])
    return (
        f'{type(self)!r}(project_id={self.project_id}, '
        f'bq_dataset_name={self.bq_dataset_name}, '
        f'bq_table_name={self.bq_table_name}, '
        f'all_feature_metadata=[{metadata_str}])'
    )

  def get_config(self):
    """Creates a config that can be converted to json."""
    config = super().get_config()
    config.update({
        'project_id': self.project_id,
        'bq_dataset_name': self.bq_dataset_name,
        'bq_table_name': self.bq_table_name,
    })
    return config


class BigQueryMetadataBuilder:
  """Gets metadata about the column in a BQ table."""

  def __init__(
      self,
      project_id,
      bq_dataset_name,
      bq_table_name,
      ignore_columns = (),
      bq_client = None,
  ):
    """Creates an object that can retrieve metadata about the input BQ table.

    Args:
      project_id: The BigQuery project that the table is in.
      bq_dataset_name: The BigQuery dataset that has the table.
      bq_table_name: The name of the table itself.
      ignore_columns: Any columns that should not be included in the output.
      bq_client: A BigQuery client to be used to interact with BigQuery. If this
        is not provided (default) a new client will be created for the input
        project.
    """
    self._project_id = project_id
    self._bq_dataset_name = bq_dataset_name
    self._bq_table_name = bq_table_name
    self._ignore_columns = ignore_columns
    self._bq_client = bq_client or bigquery.Client(project=project_id)
    self._query_config = bigquery.QueryJobConfig(use_legacy_sql=_USE_LEGACY_SQL)

    # Set up initial values for private variables that will be lazily updated
    # and cached.
    self._rows = None

  @property
  def full_table_id(self):
    """The full path 'project.dataset.table' of the BigQuery table."""
    return f'{self._project_id}.{self._bq_dataset_name}.{self._bq_table_name}'

  @property
  def escaped_table_id(self):
    """The full path of the BigQuery table escaped with `."""
    return f'`{self.full_table_id}`'

  def _query_bq(
      self,
      query,
      job_config = None,
      **kwargs,
  ):
    job_config = job_config or self._query_config
    return self._bq_client.query(query, job_config=job_config, **kwargs)

  @property
  def rows(self):
    if not self._rows:
      row_query = textwrap.dedent(f"""\
          SELECT COUNT(*)
          FROM {self.escaped_table_id}""")
      self._rows = next(self._query_bq(row_query).result())[0]

    return self._rows

  def get_feature_names_and_types(self):
    """Returns a metadata collection of all the columns in the table.

    Returns:
      A BigQueryTableMetadata with FeatureMetadata for each column in the
      object's BQ table.
    """
    table = self._bq_client.get_table(self.full_table_id)

    if self._ignore_columns:
      ignored_set = set(self._ignore_columns)
    else:
      ignored_set = set()

    columns = []
    for idx, column in enumerate(table.schema):
      if column.name not in ignored_set:
        if column.field_type not in _BIGQUERY_SUPPORTED_DATA_TYPES:
          raise NotImplementedError(
              f'The datatype {column.field_type} for {column.name} '
              'is not currently supported.'
          )
        columns.append(FeatureMetadata(column.name, idx, column.field_type))

    return BigQueryTableMetadata(
        columns,
        project_id=self._project_id,
        bq_dataset_name=self._bq_dataset_name,
        bq_table_name=self._bq_table_name,
    )

  def _construct_numeric_metadata_query(
      self, feature, options
  ):
    """Constructs a query to get the numeric metadata for the input feature.

    Note that the suffix of each of the query results must match the attribute
    that it will be assigned to.

    In my tests doing this as a single query for each feature appeared to give
    the best performance as it avoided multiple queries but also avoided the
    anti-pattern of throwing a bunch of queries together:
    https://cloud.google.com/bigquery/docs/best-practices-performance-compute#split_complex_queries_into_multiple_smaller_ones


    Args:
      feature: The current metadata for the feature. Must include the name.
      options: Specifications for what types of metadata should be retrieved.

    Returns:
      The query to run in bigquery to get the metadata.
    """
    if not feature.is_numeric:
      raise ValueError(f'This function only works for numeric data: {feature}')

    # Cache for brevity, performance and ease of use.
    name = feature.name
    # The CTE queries list can contain multiple CTE expression which will be
    # proceeded by the WITH statement and joined with commas.
    cte_queries = []

    # The where clauses list will be proceeded by the WHERE statement and
    # joined with ands.
    where_clauses = list(options.where_clauses)

    # The select components will be combined with commas to create the main
    # query. The output of this query is what will be used to create the
    # results.
    main_query_components = []
    if options.get_mean:
      main_query_components.append(f'AVG({name}) as {name}_mean')

    if options.get_variance:
      main_query_components.append(f'VARIANCE({name}) as {name}_variance')

    if options.get_min:
      main_query_components.append(f'MIN({name}) as {name}_min')

    if options.get_max:
      main_query_components.append(f'MAX({name}) as {name}_max')

    if options.get_median:
      # Look at other ways of doing this.
      main_query_components.append(
          f'APPROX_QUANTILES({name}, 3)[OFFSET(1)] as {name}_median'
      )

    if options.number_of_quantiles:
      main_query_components.append(
          f'APPROX_QUANTILES({name}, {options.number_of_quantiles}) '
          f'as {name}_quantiles'
      )

    is_float = feature.input_data_type.startswith('FLOAT')

    if options.get_log_mean or options.get_log_variance:
      cte_where = (
          where_clauses + [f'NOT IS_NAN({name})'] if is_float else where_clauses
      )
      cte_shift = 'log_shift'
      where_statement = bq_utils.where_statement_from_clauses(cte_where)
      cte_queries.append(
          f'{cte_shift} AS ('
          f'SELECT {options.min_log_value} '
          f'- LEAST(0, MIN({name})) as value '
          f'FROM {self.escaped_table_id}{where_statement})'
      )
      log_shift_subquery = f'(SELECT value from {cte_shift})'

      cte_transformed = 'log_transformed'
      cte_queries.append(
          f'{cte_transformed} AS ('
          f'SELECT LOG({name} + {log_shift_subquery}) as value '
          f'FROM {self.escaped_table_id}{where_statement})'
      )

      main_query_components.append(f'{log_shift_subquery} as {name}_log_shift')
      if options.get_log_mean:
        main_query_components.append(
            f'(SELECT AVG(value) FROM {cte_transformed}) as {name}_log_mean'
        )
      if options.get_log_variance:
        main_query_components.append(
            '(SELECT VARIANCE(value) '
            f'FROM {cte_transformed}) as {name}_log_variance'
        )

    query_parts = []
    if cte_queries:
      combined_with = ',\n'.join(cte_queries)
      query_parts.append(f'WITH {combined_with}')

    combined_select = ',\n  '.join(main_query_components)
    if is_float:
      where_clauses.append(f'NOT IS_NAN({name})')

    where_statement = bq_utils.where_statement_from_clauses(where_clauses)
    query_parts.append(
        f'SELECT\n{combined_select}\n'
        f'FROM {self.escaped_table_id}{where_statement}'
    )

    return '\n'.join(query_parts)

  def update_numeric_feature_metadata(
      self, feature, options
  ):
    """Updates a numeric feature's metadata based on BigQuery results.

    Args:
      feature: The feature to update.
      options: The options for getting the metadata.
    """
    query_string = self._construct_numeric_metadata_query(feature, options)
    # We only expect one row of results.
    query_result = next(self._query_bq(query_string).result())
    if not query_result:
      raise ValueError(f'Expected a result from query: {query_string}')

    # Do this more efficiently and safely by only iterating over
    #   numeric attributes.
    query_columns = set(query_result.keys())
    for metadata_field in dataclasses.fields(feature):
      field_name = metadata_field.name
      # This means we overwrite any already existing data.
      # Figure out a cleaner way to do this.
      query_result_name = f'{feature.name}_{field_name}'
      if query_result_name in query_columns:
        if query_result[query_result_name] is None:
          logging.warning('Got a None value for %s on %s', field_name, feature)
        setattr(feature, field_name, query_result[query_result_name])

  def update_discrete_feature_metadata(
      self, feature, options
  ):
    """Updates a discrete feature's metadata based on BigQuery results.

    Args:
      feature: The feature to update.
      options: The options for getting the metadata.
    """
    if not feature.is_discrete:
      raise ValueError(f'This function only works for discrete data: {feature}')

    # Cache for brevity, performance and ease of use.
    name = feature.name

    where_clauses = [f'{name} IS NOT NULL']
    where_clauses.extend(options.where_clauses)

    # We will get the mode from the first APPROX_TOP_COUNT output
    num_top_count = max(options.max_vocab_size, int(options.get_mode))

    select_parts = [
        # Should be close enough to use APPROX_COUNT_DISTINCT instead.
        # https://cloud.google.com/bigquery/docs/reference/standard-sql/approximate_aggregate_functions#approx_count_distinct
        f'APPROX_COUNT_DISTINCT({name}) as {name}_cardinality',
    ]
    if num_top_count <= 10000:
      # While it is not documented APPROX_TOP_COUNT does not appear to work
      # with numbers >10000. When this occurs we have to use 2 queries
      # instead.
      select_parts.append(
          f'APPROX_TOP_COUNT({name}, {num_top_count}) as {name}_top_count'
      )

    combined_select = ',\n  '.join(select_parts)
    # The where clauses list will be proceeded by the WHERE statement and
    # joined with ands.
    where_statement = bq_utils.where_statement_from_clauses(where_clauses)
    query_string = (
        f'SELECT\n  {combined_select}\n'
        f'FROM {self.escaped_table_id}{where_statement}'
    )

    # We only expect one row of results.
    query_result = next(self._query_bq(query_string).result())
    if not query_result:
      raise ValueError(f'Expected a result from query: {query_string}')

    # We need to handle mode independently since it comes from the vocab.
    feature.cardinality = int(query_result[f'{feature.name}_cardinality'])

    if num_top_count <= 10000:
      feature.vocabulary = {
          item['value']: int(item['count'])
          for item in query_result[f'{feature.name}_top_count']
      }
    else:
      # Since we are asking for a vocabulary of over 10k elements we need to use
      # a second query to get those results and the convert it to a dictionary.
      dataframe = (
          self._query_bq(
              f'SELECT {name} as value, count(*) as count '
              f'FROM {self.escaped_table_id}{where_statement} GROUP BY {name} '
              f'ORDER BY 2 DESC LIMIT {num_top_count}'
          )
          .result()
          .to_dataframe()
      )
      feature.vocabulary = {
          r['value']: int(r['count']) for r in dataframe.to_dict('records')
      }

    # The vocabulary is already ordered by the top count.
    if options.get_mode:
      feature.mode = next(iter(feature.vocabulary))

  def get_metadata_for_all_features(
      self, options
  ):
    """Gets the metadata_collection for all the features in the specified table.

    Args:
      options: The options to use when retrieving the metadata.

    Returns:
      A collection of metadata for each feature in the table with the specified
      values calculated.
    """
    feature_metadata = self.get_feature_names_and_types()

    if options.no_options():
      return feature_metadata

    for feature in feature_metadata:
      if feature.is_numeric:
        self.update_numeric_feature_metadata(feature, options)

      if feature.is_discrete:
        self.update_discrete_feature_metadata(feature, options)

    return feature_metadata

  # Add the ability to export the metadata_collection.

  @classmethod
  def from_table_parts(
      cls, table_parts, *args, **kwargs
  ):
    """Constructs a BigQueryMetadataBuilder directly from table_parts.

    Args:
      table_parts: The BigQuery path parts for the table for the metadata.
      *args: Positional arguments to be passed to __init__.
      **kwargs: Keyword arguments to be passed to __init__.

    Returns:
      The BigQueryMetadataBuilder for the specified table.
    """
    return cls(
        project_id=table_parts.project_id,
        bq_dataset_name=table_parts.bq_dataset_name,
        bq_table_name=table_parts.bq_table_name,
        *args,
        **kwargs,
    )

  @classmethod
  def from_table_path(
      cls, table_path, *args, **kwargs
  ):
    """Constructs a BigQueryMetadataBuilder directly from a full table path.

    Args:
      table_path: The full path of the BigQuery table in the format
        project.dataset.table.
      *args: Positional arguments to be passed to __init__.
      **kwargs: Keyword arguments to be passed to __init__.

    Returns:
      The BigQueryMetadataBuilder for the specified table.
    """
    table_parts = bq_utils.BQTablePathParts.from_full_path(table_path)
    return cls.from_table_parts(table_parts, *args, **kwargs)
