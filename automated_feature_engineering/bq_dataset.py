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

"""Module for creating a tf.data.Dataset from a BigQuery table."""

import collections
import dataclasses
import functools
import tempfile
from typing import Any, Callable, Generator, Iterable, Mapping, Optional, Sequence, Tuple, Union

import bq_utils
import feature_metadata
from google.cloud import bigquery
from google.cloud import bigquery_storage
import numpy as np
import pandas as pd
import tensorflow as tf


# Investigate using BigQueryReadClient with one generator per
#  stream.

NO_CACHE_LOCATION_NAME = 'none'

VALUES_KEY = 'values'
WAS_NULL_KEY = 'was_null'

# The value used to replace NULL strings.
NULL_STRING_PLACEHOLDER = '__NULL_PLACEHOLDER__'
# The value used to replace NULL floats.
NULL_FLOAT_PLACEHOLDER = 0.0
# The value used to replace NULL integers.
NULL_INT_PLACEHOLDER = 0
# The value used to replace NULL booleans.
NULL_BOOL_PLACEHOLDER = False

TensorAndMaskType = Union[
    Mapping[str, tf.Tensor], Mapping[str, Mapping[str, tf.Tensor]]
]
TensorAndMaskSpecType = Union[
    Mapping[str, tf.TensorSpec], Mapping[str, Mapping[str, tf.TensorSpec]]
]
KerasInputType = Callable[..., tf.Tensor]

_TF_FLOAT_DTYPE = tf.dtypes.float32
_TF_INT_DTYPE = tf.dtypes.int32
_TF_BOOL_DTYPE = tf.dtypes.bool
_NP_FLOAT_DTYPE = np.float32
_NP_INT_DTYPE = np.int32
_NP_BOOL_DTYPE = np.bool_

# The default value for variables related to input formatting.
NESTED_FORMAT_DEFAULT = False

_USE_LEGACY_SQL = False
_WITH_MASK_DEFAULT = False


def convert_series_to_tensor_dictionary(
    input_series: pd.Series,
    value_to_replace_null: Any,
    np_dtype: np.dtype,
    tf_dtype: tf.dtypes.DType,
    with_mask: bool = _WITH_MASK_DEFAULT,
) -> Union[tf.Tensor, Mapping[str, tf.Tensor]]:
  """Converts the input series into a tensor after filling null values.

  Args:
    input_series: The series to be converted.
    value_to_replace_null: Any nulls in this feature will be replaced with this
      value.
    np_dtype: The data in the series will be converted to this data type after
      filling and nulls and prior to converting to Tensorflow.
    tf_dtype: The data type for the output tensor(s).
    with_mask: If True (the default) the data will be output as a dictionary
      with the keys 'values' and 'was_null'. Otherwise, the tensor will be
      output directly.

  Returns:
    A tensor with the null values filled. If with_mask is true this tensor
    will be included in a dictionary with the key 'values' and the dictionary
    will also contain the key 'was_null' which will be a boolean tensor that
    is true if that value was pandas NA and false otherwise.
  """
  if with_mask:
    # Check performance to see if using the null mask is faster.
    missing_mask = input_series.isnull()
    # pandas disabled silent downcasting in version 2.2.1, suppressing
    # the warning here since we auto-inferred the dtype already.
    # Uncomment this when pandas version is updated to 2.2.1:
    # with pd.option_context('future.no_silent_downcasting', True):
    input_series = input_series.fillna(value_to_replace_null)
    return {
        VALUES_KEY: tf.convert_to_tensor(
            input_series.to_numpy(dtype=np_dtype),
            dtype=tf_dtype,
        ),
        WAS_NULL_KEY: tf.convert_to_tensor(
            missing_mask.to_numpy(dtype=bool),
            dtype=tf.dtypes.bool,
        ),
    }
  else:
    with pd.option_context('future.no_silent_downcasting', True):
      filled_inputs = input_series.fillna(value_to_replace_null)
    return tf.convert_to_tensor(
        filled_inputs.to_numpy(dtype=np_dtype),
        dtype=tf_dtype,
    )


# noinspection PyUnresolvedReferences
@dataclasses.dataclass
class BQFeatureConverter:
  """Converts an input series into a structure that can be used by tf.Dataset.

  Uses the input arguments to help convert the incoming data (i.e. pd.Series)
  into a dictionary of tensors (or dictionary of dictionaries) that can be
  passed into a TensorFlow dataset.

  Attributes:
    value_to_replace_null: Any nulls in this feature will be replaced with this
      value.
    np_dtype: The data in the series will be converted to this data type after
      filling and nulls and prior to converting to Tensorflow.
    tf_dtype: The data type for the output.
  """

  value_to_replace_null: Any
  np_dtype: np.dtype
  tf_dtype: tf.dtypes.DType

  def series_to_tensor(
      self,
      input_series: pd.Series,
      with_mask: bool = _WITH_MASK_DEFAULT,
  ) -> Union[tf.Tensor, Mapping[str, tf.Tensor]]:
    """Converts the input series into a tensor after filling null values.

    Args:
      input_series: The series to be converted.
      with_mask: If True (the default) the data will be output as a dictionary
        with the keys 'values' and 'was_null'. Otherwise the tensor will be
        output directly.

    Returns:
      A tensor with the null values filled. If with_mask is true this tensor
      will be included in a dictionary with the key 'values' and the dictionary
      will also contain the key 'was_null' which will be a boolean tensor that
      is true if that value was pandas NA and false otherwise.
    """
    return convert_series_to_tensor_dictionary(
        input_series,
        self.value_to_replace_null,
        self.np_dtype,
        self.tf_dtype,
        with_mask,
    )


# https://numpy.org/doc/stable/reference/arrays.dtypes.html#string-dtype-note
_STRING_CONVERSION_INFO = BQFeatureConverter(  # pytype: disable=wrong-arg-types  # numpy-scalars
    NULL_STRING_PLACEHOLDER,
    np.str_,
    tf.dtypes.string,
)
_FLOAT_CONVERSION_INFO = BQFeatureConverter(  # pytype: disable=wrong-arg-types  # numpy-scalars
    NULL_FLOAT_PLACEHOLDER,
    _NP_FLOAT_DTYPE,
    _TF_FLOAT_DTYPE,
)
_INT_CONVERSION_INFO = BQFeatureConverter(  # pytype: disable=wrong-arg-types  # numpy-scalars
    NULL_INT_PLACEHOLDER, _NP_INT_DTYPE, _TF_INT_DTYPE
)
_BOOL_CONVERSION_INFO = BQFeatureConverter(  # pytype: disable=wrong-arg-types  # numpy-scalars
    NULL_BOOL_PLACEHOLDER, _NP_BOOL_DTYPE, _TF_BOOL_DTYPE
)

# Create a mapping from the BigQuery data types to how we want to handle their
# conversion to tensors.  This uses a default type of converting the objects to
# strings if the conversion is not explicitly included in the dictionary.
_TF_INFO_FROM_BQ_DTYPE = collections.defaultdict(
    lambda: _STRING_CONVERSION_INFO,
    {
        'BIGNUMERIC': _FLOAT_CONVERSION_INFO,
        'FLOAT': _FLOAT_CONVERSION_INFO,
        'FLOAT64': _FLOAT_CONVERSION_INFO,
        'INT64': _INT_CONVERSION_INFO,
        'INTEGER': _INT_CONVERSION_INFO,
        'NUMERIC': _FLOAT_CONVERSION_INFO,
        'STRING': _STRING_CONVERSION_INFO,
        'BOOL': _BOOL_CONVERSION_INFO,
    },
)


# Make this cleaner and more dynamic.
def _dataframe_to_dict_of_tensors(
    df: pd.DataFrame,
    metadata_container: feature_metadata.FeatureMetadataContainer,
    with_mask: bool = _WITH_MASK_DEFAULT,
    nested: bool = NESTED_FORMAT_DEFAULT,
) -> TensorAndMaskType:
  """Converts a dataframe to a dictionary of tensors.

  Args:
    df: The input dataframe to be converted.
    metadata_container: The metadata collection for the data in the dataframe.
    with_mask: If true a mask specifying if the data was null will also be
      returned.
    nested: If true the output dictionary will have sub-dictionaries each with
      the fields 'values' and 'was_null'. If false the output dictionary will
      have keys '{feature_name}_values' and '{feature_name}_was_null' for each
      feature. Note that this only applies if `with_mask` is true.

  Returns:
    A dictionary of tensors and potentially masks of each of the features in the
    input dataframe.
  """
  output = {}
  for col_name, col_data in df.items():
    metadata = metadata_container.get_metadata_by_name(col_name)  # pytype: disable=wrong-arg-types  # pandas-drop-duplicates-overloads
    bq_conversion = _TF_INFO_FROM_BQ_DTYPE[metadata.input_data_type]
    feature_output = bq_conversion.series_to_tensor(
        col_data, with_mask=with_mask
    )
    if with_mask and not nested:
      for k, v in feature_output.items():
        output[f'{col_name}_{k}'] = v
    else:
      output[col_name] = feature_output

  return output  # pytype: disable=bad-return-type  # pandas-drop-duplicates-overloads


def _get_output_from_df_iterator(
    output_dataframes: Iterable[pd.DataFrame],
    metadata_container: feature_metadata.FeatureMetadataContainer,
    batch_size: int = 64,
    with_mask: bool = _WITH_MASK_DEFAULT,
    nested: bool = NESTED_FORMAT_DEFAULT,
    drop_remainder: bool = False,
    verbose: bool = False,
) -> Generator[TensorAndMaskType, None, None]:
  """Converts an iterator of DataFrame into the correct format and batch sizes.

  Args:
    output_dataframes: An iterator of pandas DataFrames as generated from
      bigquery.RowIterator.to_dataframe_iterable.
    metadata_container: The metadata generated for the features to be returned
      by the query.
    batch_size: The desired output batch size.
    with_mask: If false the output If true each feature will be returned as a
      dictionary with the keys 'values' and 'was_null'. The 'values' field is a
      dense tensor with null values filled and the 'was_null' tensor is a dense
      boolean tensor that is true where the 'values' tensor was null prior to
      being filled.
    nested: If true the output dictionary will have sub-dictionaries each with
      the fields 'values' and 'was_null'. If false the output dictionary will
      have keys '{feature_name}_values' and '{feature_name}_was_null' for each
      feature. Note that this only applies if `with_mask` is true.
    drop_remainder: If true no partial batches will be yielded.
    verbose: If true will print debugging messages using tf.print. These can be
      used to help debug when this function is running and when cached data is
      being used.

  Yields:
    A dictionary of features from the query.  If with_mask is true each feature
    will have a dictionary with keys 'values' and 'was_null' where 'values'
    contains a Tensor of the values from that column after null values have been
    filled and 'was_null' is a boolean tensor
  """
  # Investigate more performant ways to do this.
  data_frame_cache = []
  cur_rows = 0

  conversion_function = functools.partial(
      _dataframe_to_dict_of_tensors,
      metadata_container=metadata_container,
      with_mask=with_mask,
      nested=nested,
  )

  for df in output_dataframes:
    new_rows = df.shape[0]
    cur_rows += new_rows

    # Shortcut the happy path because eventually it might work.
    if new_rows == batch_size and not data_frame_cache:
      yield conversion_function(df)
      cur_rows = 0
      continue

    while cur_rows >= batch_size:
      # We now have enough data to output something
      if data_frame_cache:
        df = pd.concat(data_frame_cache + [df], axis=0, ignore_index=True)
        data_frame_cache = []

      output_data = df.iloc[:batch_size].reset_index(drop=True)

      if cur_rows > batch_size:
        df = df.iloc[batch_size:]
        cur_rows = df.shape[0]
      else:
        cur_rows = 0

      if verbose:
        tf.print('Yielding BQ Query')
      yield conversion_function(output_data)

    if cur_rows > 0:
      data_frame_cache.append(df)

  if data_frame_cache and not drop_remainder:
    if verbose:
      tf.print('Yielding BQ Query')
    yield conversion_function(
        pd.concat(data_frame_cache, axis=0, ignore_index=True)
    )


def bigquery_table_batch_generator(
    metadata_container: feature_metadata.BigQueryTableMetadata,
    bq_client: bigquery.Client,
    bqstorage_client: Optional[bigquery_storage.BigQueryReadClient] = None,
    batch_size: int = 64,
    with_mask: bool = _WITH_MASK_DEFAULT,
    nested: bool = NESTED_FORMAT_DEFAULT,
    drop_remainder: bool = False,
    page_size: Optional[int] = None,
    verbose: bool = False,
) -> Generator[TensorAndMaskType, None, None]:
  """Generates DataFrames iteratively from a Bigquery query.

  Args:
    metadata_container: The metadata generated for the features to be returned
      by the query.
    bq_client: The BigQuery client that will be used for the query.
    bqstorage_client: The BigQueryStorageClient that will be used for the query.
    batch_size: The desired output batch size.
    with_mask: If false the output If true each feature will be returned as a
      dictionary with the keys 'values' and 'was_null'. The 'values' field is a
      dense tensor with null values filled and the 'was_null' tensor is a dense
      boolean tensor that is true where the 'values' tensor was null prior to
      being filled.
    nested: If true the output dictionary will have sub-dictionaries each with
      the fields 'values' and 'was_null'. If false the output dictionary will
      have keys '{feature_name}_values' and '{feature_name}_was_null' for each
      feature. Note that this only applies if `with_mask` is true.
    drop_remainder: If true no partial batches will be yielded.
    page_size: the pagination size to use when retrieving data from BigQuery. A
      large value can result in fewer BQ calls, hence time savings.
    verbose: If true will print debugging messages using tf.print. These can be
      used to help debug when this function is running and when cached data is
      being used.

  Yields:
    A dictionary of features from the query.  If with_mask is true each feature
    will have a dictionary with keys 'values' and 'was_null' where 'values'
    contains a Tensor of the values from that column after null values have been
    filled and 'was_null' is a boolean tensor
  """
  if page_size is None:
    page_size = batch_size
  row_iterator = bq_client.list_rows(
      metadata_container.bigquery_table,
      selected_fields=metadata_container.to_bigquery_schema(),
      page_size=page_size,
  )

  output_dataframes = row_iterator.to_dataframe_iterable(
      bqstorage_client=bqstorage_client
  )

  yield from _get_output_from_df_iterator(
      output_dataframes=output_dataframes,
      metadata_container=metadata_container,
      batch_size=batch_size,
      with_mask=with_mask,
      nested=nested,
      drop_remainder=drop_remainder,
      verbose=verbose,
  )


def bigquery_query_batch_generator(
    query: str,
    metadata_container: feature_metadata.FeatureMetadataContainer,
    bq_client: bigquery.Client,
    bqstorage_client: Optional[bigquery_storage.BigQueryReadClient] = None,
    batch_size: int = 64,
    with_mask: bool = _WITH_MASK_DEFAULT,
    nested: bool = NESTED_FORMAT_DEFAULT,
    drop_remainder: bool = False,
    page_size: Optional[int] = None,
    verbose: bool = False,
) -> Generator[TensorAndMaskType, None, None]:
  """Generates DataFrames iteratively from a Bigquery query.

  Args:
    query: The query to be run.
    metadata_container: The metadata generated for the features to be returned
      by the query.
    bq_client: The BigQuery client that will be used for the query.
    bqstorage_client: The BigQueryStorageClient that will be used for the query.
    batch_size: The desired output batch size.
    with_mask: If false the output If true each feature will be returned as a
      dictionary with the keys 'values' and 'was_null'. The 'values' field is a
      dense tensor with null values filled and the 'was_null' tensor is a dense
      boolean tensor that is true where the 'values' tensor was null prior to
      being filled.
    nested: If true the output dictionary will have sub-dictionaries each with
      the fields 'values' and 'was_null'. If false the output dictionary will
      have keys '{feature_name}_values' and '{feature_name}_was_null' for each
      feature. Note that this only applies if `with_mask` is true.
    drop_remainder: If true no partial batches will be yielded.
    page_size: the pagination size to use when retrieving data from BigQuery. A
      large value can result in fewer BQ calls, hence time savings.
    verbose: If true will print debugging messages using tf.print. These can be
      used to help debug when this function is running and when cached data is
      being used.

  Yields:
    A dictionary of features from the query.  If with_mask is true each feature
    will have a dictionary with keys 'values' and 'was_null' where 'values'
    contains a Tensor of the values from that column after null values have been
    filled and 'was_null' is a boolean tensor
  """
  # Page size doesn't seem to have any impact here. Bug in client?
  # Instead, we will handle batching manually which is painful and slow.

  # It is recommended that legacy SQL not be used and this allows large results.
  # For more information see the following documentation:
  #  https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationQuery.FIELDS.allow_large_results
  #  https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationQuery.FIELDS.use_legacy_sql
  query_config = bigquery.QueryJobConfig(use_legacy_sql=_USE_LEGACY_SQL)
  if page_size is None:
    page_size = batch_size
  output_dataframes = (
      bq_client.query(query, job_config=query_config)
      .result(page_size=page_size)
      .to_dataframe_iterable(bqstorage_client=bqstorage_client)
  )

  yield from _get_output_from_df_iterator(
      output_dataframes=output_dataframes,
      metadata_container=metadata_container,
      batch_size=batch_size,
      with_mask=with_mask,
      nested=nested,
      drop_remainder=drop_remainder,
      verbose=verbose,
  )


def tensor_output_signature_from_metadata(
    metadata_container: feature_metadata.FeatureMetadataContainer,
    with_mask: bool = _WITH_MASK_DEFAULT,
    nested: bool = NESTED_FORMAT_DEFAULT,
) -> TensorAndMaskSpecType:
  """Converts the input metadata into a TensorSpec for Dataset creation.

  Args:
    metadata_container: The input feature metadata.
    with_mask: Whether was_null will be included in the output.
    nested: If true the output dictionary will have sub-dictionaries each with
      the fields 'values' and 'was_null'. If false the output dictionary will
      have keys '{feature_name}_values' and '{feature_name}_was_null' for each
      feature. Note that this only applies if `with_mask` is true.

  Returns:
    The TensorSpec corresponding to the expected output.
  """
  output_spec = {}
  for feature_md in metadata_container:
    if not feature_md.tf_data_type:
      raise ValueError(
          'The TF data type must be set before creating the output signature.'
      )
    name = feature_md.name
    values_spec = tf.TensorSpec(
        (None,), dtype=feature_md.tf_data_type, name=f'{name}_{VALUES_KEY}'
    )
    if with_mask:
      was_null_spec = tf.TensorSpec(
          (None,), dtype=tf.bool, name=f'{feature_md.name}_{WAS_NULL_KEY}'
      )
      if nested:
        output_spec[name] = {
            VALUES_KEY: values_spec,
            WAS_NULL_KEY: was_null_spec,
        }
      else:
        output_spec[f'{name}_{VALUES_KEY}'] = values_spec
        output_spec[f'{name}_{WAS_NULL_KEY}'] = was_null_spec
    else:
      output_spec[name] = values_spec

  return output_spec


def _generate_query_from_metadata(
    table_metadata: feature_metadata.BigQueryTableMetadata,
    limit: Optional[int] = None,
    where_clauses: Sequence[str] = tuple(),
) -> str:
  """Generates a query from column names, where clauses and a limit."""
  select_fields_string = '`,`'.join(col.name for col in table_metadata)
  limit_str = f' LIMIT {limit}' if limit else ''
  where_str = bq_utils.where_statement_from_clauses(where_clauses)
  return (
      f'SELECT `{select_fields_string}` FROM\n'
      f'{table_metadata.escaped_table_id}{where_str}{limit_str}'
  )


def update_tf_data_types_from_bq_data_types(
    metadata_container: feature_metadata.FeatureMetadataContainer,
) -> None:
  """Updates each feature's tf_data_type based on the input_data_type.

  Args:
    metadata_container: The metadata container to be updated.
  """
  for metadata in metadata_container:
    if not metadata.tf_data_type:
      metadata.tf_data_type = _TF_INFO_FROM_BQ_DTYPE[
          metadata.input_data_type
      ].tf_dtype


def get_bigquery_dataset(
    table_metadata: feature_metadata.BigQueryTableMetadata,
    bq_client: bigquery.Client,
    bqstorage_client: Optional[bigquery_storage.BigQueryReadClient] = None,
    batch_size: int = 64,
    with_mask: bool = _WITH_MASK_DEFAULT,
    nested: bool = NESTED_FORMAT_DEFAULT,
    limit: Optional[int] = None,
    cache_location: Optional[str] = NO_CACHE_LOCATION_NAME,
    where_clauses: Sequence[str] = tuple(),
    drop_remainder: bool = False,
    page_size: Optional[int] = None,
) -> tf.data.Dataset:
  """Creates a Big Query dataset for the specified Table.

  Args:
    table_metadata: The metadata for the BigQuery table.
    bq_client: The BigQuery client that will be used for the query.
    bqstorage_client: The BigQueryStorageClient that will be used for the query.
    batch_size: The desired output batch size.
    with_mask: If true each feature will be returned along with a tensor
      indicating if the values were null before being filled.
    nested: If true the output dictionary will have sub-dictionaries each with
      the fields 'values' and 'was_null'. If false the output dictionary will
      have keys '{feature_name}_values' and '{feature_name}_was_null' for each
      feature. Note that this only applies if `with_mask` is true.
    limit: Put a limit on the number of examples returned from the query. If
      falsy all the results will be returned.
    cache_location: If 'disk' the data will be cached to disk if None the data
      will not be cached and otherwise the data will be cached to memory.
    where_clauses: A list of clauses that can be combined with and statements to
      get the correct values in the BigQuery table.
    drop_remainder: If true no partial batches will be yielded.
    page_size: the pagination size to use when retrieving data from BigQuery. A
      large value can result in fewer BQ calls, hence time savings.

  Returns:
    A tf.data.Dataset for the table.
  """
  update_tf_data_types_from_bq_data_types(table_metadata)
  tensor_spec = tensor_output_signature_from_metadata(
      table_metadata, with_mask=with_mask, nested=nested
  )

  if limit or where_clauses:
    query = _generate_query_from_metadata(table_metadata, limit, where_clauses)
    tensor_generator_fn = functools.partial(
        bigquery_query_batch_generator,
        query=query,
        metadata_container=table_metadata,
        bq_client=bq_client,
        bqstorage_client=bqstorage_client,
        batch_size=batch_size,
        with_mask=with_mask,
        nested=nested,
        drop_remainder=drop_remainder,
        page_size=page_size,
    )
  else:
    tensor_generator_fn = functools.partial(
        bigquery_table_batch_generator,
        metadata_container=table_metadata,
        bq_client=bq_client,
        bqstorage_client=bqstorage_client,
        batch_size=batch_size,
        with_mask=with_mask,
        nested=nested,
        drop_remainder=drop_remainder,
        page_size=page_size,
    )

  dataset = tf.data.Dataset.from_generator(
      tensor_generator_fn, output_signature=tensor_spec
  )
  if cache_location is not None and cache_location != NO_CACHE_LOCATION_NAME:
    filename = tempfile.mkdtemp() if cache_location == 'disk' else ''
    # This appears to give a false warning: b/194670791
    dataset = dataset.cache(filename)

  return dataset


def keras_input_from_metadata(
    metadata_container: feature_metadata.FeatureMetadataContainer,
    with_mask: bool = _WITH_MASK_DEFAULT,
    nested: bool = NESTED_FORMAT_DEFAULT,
) -> Mapping[str, Union[Mapping[str, KerasInputType], KerasInputType]]:
  """Creates Keras Input objects for input metadata.

  Args:
    metadata_container: The metadata of the features to be used for input.
    with_mask: Whether a mask is included in the dataset or not. See
      get_bigquery_dataset for more information.
    nested: If True assumes the input dictionaries are nested. Otherwise,
      assumes that a flat dictionary is input. This is only relevant if
      with_mask is true.

  Returns:
    A dictionary of tf.keras.Input objets that can be input into a
    tf.keras.Model object.
  """
  model_input = {}
  for metadata in metadata_container:
    cur_name = metadata.name
    # Keras layers must take in JSON serializable inputs.
    values_input = tf.keras.Input(
        (), name=f'{cur_name}_{VALUES_KEY}', dtype=metadata.tf_data_type_str
    )
    if with_mask:
      was_null_input = tf.keras.Input(
          (), name=f'{cur_name}_{WAS_NULL_KEY}', dtype='bool'
      )
      if nested:
        model_input[cur_name] = {
            VALUES_KEY: values_input,
            WAS_NULL_KEY: was_null_input,
        }
      else:
        model_input[f'{cur_name}_{VALUES_KEY}'] = values_input
        model_input[f'{cur_name}_{WAS_NULL_KEY}'] = was_null_input
    else:
      model_input[cur_name] = values_input
  return model_input


def get_dataset_and_metadata_for_table(
    table_path: Optional[str] = None,
    table_parts: Optional[bq_utils.BQTablePathParts] = None,
    bigquery_client: Optional[bigquery.Client] = None,
    bigquery_storage_client: Optional[
        bigquery_storage.BigQueryReadClient
    ] = None,
    metadata_options: Optional[
        feature_metadata.MetadataRetrievalOptions
    ] = None,
    metadata_builder: Optional[feature_metadata.BigQueryMetadataBuilder] = None,
    batch_size: int = 64,
    with_mask: bool = _WITH_MASK_DEFAULT,
    drop_remainder: bool = False,
) -> Tuple[tf.data.Dataset, feature_metadata.BigQueryTableMetadata]:
  """Gets the metadata and dataset for a BigQuery table.

  Args:
    table_path: The full path ('project.dataset.table') of the BigQuery table.
    table_parts: The parsed potions of the BigQuery table path.
    bigquery_client: The BigQuery Client object to use for getting the metadata.
    bigquery_storage_client: The BigQuery storage client to use for the dataset.
    metadata_options: The metadata retrieval options to use.
    metadata_builder: The metadata builder to use to get the metadata.
    batch_size: The batch size to use for the dataset. Default is 64.
    with_mask: Whether the dataset should be returned with a mask format. For
      more information see get_bigquery_dataset.
    drop_remainder: If true no partial batches will be yielded.

  Returns:
    A tuple of the output dataset and metadata for the specified table.

  Raises:
    ValueError: If neither table_parts nor table_path are specified or both of
      them are.
  """
  if not (table_parts or table_path):
    raise ValueError('Either table_parts or table_path must be specified.')
  elif table_parts and table_path:
    raise ValueError('Only one of table_parts or table_path can be specified.')

  if not table_parts:
    table_parts = bq_utils.BQTablePathParts.from_full_path(table_path)

  if not bigquery_client:
    bigquery_client = bigquery.Client(project=table_parts.project_id)

  if not bigquery_storage_client:
    bigquery_storage_client = bigquery_storage.BigQueryReadClient()

  if not metadata_options:
    metadata_options = feature_metadata.MetadataRetrievalOptions.get_none()

  if not metadata_builder:
    metadata_builder = (
        feature_metadata.BigQueryMetadataBuilder.from_table_parts(
            table_parts, bq_client=bigquery_client
        )
    )

  all_metadata = metadata_builder.get_metadata_for_all_features(
      metadata_options
  )
  # Refactor the code so that this extra call is not needed.
  update_tf_data_types_from_bq_data_types(all_metadata)

  dataset = get_bigquery_dataset(
      all_metadata,
      bigquery_client,
      bqstorage_client=bigquery_storage_client,
      batch_size=batch_size,
      with_mask=with_mask,
      # Cache during prepare_dataset instead.
      cache_location=None,
      where_clauses=metadata_options.where_clauses,
      drop_remainder=drop_remainder,
  )

  return dataset, all_metadata
