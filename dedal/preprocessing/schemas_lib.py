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

"""Utility functions to process tabular data in Beam."""

from __future__ import annotations

import dataclasses
import re
from typing import Any, Collection, Iterable, Iterator, List, Mapping, Optional, Tuple, Union

import apache_beam as beam

from dedal.preprocessing import types


# Type aliases
SingleValue = types.SingleValue
Value = types.Value
Record = types.Record
Key = types.Key
KeyRecordPair = types.KeyRecordPair


def str2bool(s):
  """Parses `s` to `bool`, raising `ValueError` if `s` not 'True' or 'False'."""
  if s not in ('True', 'False'):
    raise ValueError("s must be 'True' or 'False'.")
  return s == 'True'  # bool('False') evaluates to `True`.


# For now, only `bool`, `int`, `float` and `str` types are supported. Fields
# with other type annotations will default to `str`, thus keeping the raw text
# representation of the field as read from the table.
SUPPORTED_TYPES: Mapping[str, Any] = {
    'bool': str2bool,
    'int': int,
    'float': float,
    'str': str
}
# Variable-length fields must be type-annotated as `List`. Other `Sequence`s are
# not yet supported.
VAR_LEN_ANNOTATION_PATTERN = r'List\[(bool|int|float|str)\]'


def _parse_single_value(single_value, type_str):
  """Casts `single_value` to `type_str`, treating unknown types as string."""
  try:
    return SUPPORTED_TYPES.get(type_str, str)(single_value)
  except ValueError as err:
    raise ValueError(f'Data is inconsistent with provided schema: '
                     f'{single_value} is not of type {type_str}.') from err


def parse_value(value, type_str, var_len_sep = ';'):
  """Parses raw text field `value` following its type annotation `type_str`.

  Note that `value` *must* be a text representation of a variable (or list of
  variables) with the base type indicated by `type_str`. In particular, type
  conversions are currently *not* supported and consistency of the input
  arguments is *not* verified. Thus, the behavior of this method when provided
  with inconsistent (`value`, `type_str`) inputs, such as attempting to parse
  `value = '0.0'` to `type_str = 'int'`, is undefined.

  Args:
    value: The string representation of a field value, stored in a flat text
      tabular database. It may represent a single (scalar) value or a variable
      length list of values of the same type.
    type_str: A type annotation string describing the type of the field. Only
      `bool`, `int`, `float` and `str` are supported as base types. Others will
      be treated as `str` (no casting). Variable-length fields must be type
      annotated as `list[base_type]`.
    var_len_sep: The character used to separate individual entries in
      variable-length fields.

  Returns:
    Either a single (scalar) value or a variable-length list of values, typed in
    accordance with the type annotation string `type_str`.
  """
  # Parses type annotation of variable-length fields.
  re_match = re.fullmatch(VAR_LEN_ANNOTATION_PATTERN, type_str)
  if re_match:  # Regex match <--> field annotated as variable length.
    if value:
      inner_type_str = re_match.group(1)
      return [_parse_single_value(x, inner_type_str)
              for x in value.split(var_len_sep)]
    else:  # Variable-length fields are allowed to be empty.
      return []
  else:  # No regex match <-> field *not* annotated as having variable length.
    return _parse_single_value(value, type_str)


class TableRowMixin:
  r"""Mixin for dataclasses representing a record of a tabular database.

  Dataclasses representing rows from a tabular database may subclass this Mixin
  to gain common functionality such as parsing raw text rows, casting fields to
  their right type following the dataclass' type annotations, dumping a set of
  rows to a flat text file or conversion to/from Python dicts with optional
  field filters, among others.

  Example usage:
  ```
    # A dataclass describing the columns of a tabular database, such as a CSV
    # file.
    @dataclasses.dataclass(frozen=True)
    class ExampleSchema(TableRowMixin):
      bool_field: bool
      int_field: int
      float_field: float
      str_field: str
      var_len_int_field: list[int]  # Var-len fields annotated as `list`.
      var_len_str_field: list[str]

    # Creating records from an open CSV file `f_in`.
    records = [ExampleSchema.from_table_row(line, sep=',') for line in f_in]

    # Writing records to an open CSV file `f_out`, with pipes delimiting entries
    # within the variable-length fields.
    for record in records:
      f_out.write(record.to_table_row(sep=',', var_len_sep='|') + '\n')

    # Converting records to Python dict while keeping only two fields.
    dict_records = [record.to_dict(fields_to_keep=('int_field', 'str_field'))
                    for record in records]
  ```
  """

  @classmethod
  def var_len_field_names(cls):
    """Returns the set of names of all fields annotated as variable length."""
    var_len_field_names = set()
    for field in dataclasses.fields(cls):
      if re.fullmatch(VAR_LEN_ANNOTATION_PATTERN, field.type) is not None:
        var_len_field_names.add(field.name)
    return var_len_field_names

  @classmethod
  def from_table_row(
      cls,
      row,
      sep = '\t',
      var_len_sep = ';',
  ):
    """Parses a row of a tabular database described by a dataclass.

    Note that it is expected that the number of fields in each `row` is fixed
    and equal to the number of fields in the dataclass subclassing this mixin.
    However, empty fields are valid.

    Args:
      row: A record ("row") from a flat text tabular database, consisting of a
        fixed number of fields ("columns"), each of which may be contain a
        single entry or be a variable length (possibly empty) list of entries.
      sep: The character used to separate the different fields ("columns").
      var_len_sep: The character used to separate individual entries in
        variable length fields.

    Returns:
      An instance of the dataclass subclassing this mixin with its fields
      populated according to the contents of `row`. Each field is cast following
      the type annotations of the dataclass.
    """
    field_values = row.strip('\n\r\x0b\x0c').split(sep)  # Excludes '\t'.
    field_names = tuple(field.name for field in dataclasses.fields(cls))
    field_types = tuple(field.type for field in dataclasses.fields(cls))
    return cls(
        **{
            k: parse_value(value=v, type_str=t, var_len_sep=var_len_sep)
            for k, v, t in zip(field_names, field_values, field_types)
        },
    )

  @classmethod
  def from_dict(cls, field_kvs):
    """Creates a dataclass instance for a database record from key-value pairs.

    Note that, unlike `from_table_row`, this method assumes the typing of values
    in the input matches those of the dataclass. No attempt to check or enforce
    this via casting has been implemented yet.

    Args:
      field_kvs: A mapping of key to values whose keys coincide with the fields
        of the dataclass subclassing this mixin. Extra keys are allowed and will
        be silently ignored. Missing keys require the dataclass to define
        default values for the corresponding fields.

    Returns:
      An instance of the dataclass subclassing this mixin with its fields
      populated according to the contents of `field_kvs`.
    """
    def maybe_to_list(v):
      """Enforces variable-length field `v` to be a Python `list`."""
      v = [v] if isinstance(v, (bool, int, float, str)) else v
      return v if isinstance(v, list) else list(v)
    var_len_field_names = cls.var_len_field_names()
    # Ignores any fields in `field_kvs` that are not defined in the dataclass
    # and makes sure variable-length fields are Python `list`s. However, base
    # type consistency with the dataclass's type annotations are not enforced.
    field_names = set(field.name for field in dataclasses.fields(cls))
    field_kvs = {k: maybe_to_list(v) if k in var_len_field_names else v
                 for k, v in field_kvs.items() if k in field_names}
    try:
      return cls(**field_kvs)
    except TypeError as err:
      # Note: missing fields are already reported by original exception `err`.
      raise TypeError('All schema fields without default values must be present'
                      ' in `field_kvs`.') from err

  def to_dict(
      self,
      fields_to_keep = None,
  ):
    """Converts dataclass-represented database records to key-value pairs.

    Args:
      fields_to_keep: A collection with the names of the dataclass fields (i.e.
        columns in the tabular database represented by the dataclass) to be
        kept. If None (default), all fields will be kept.

    Returns:
      A Python dictionary of `(field_name, value)` pairs for all fields in
      `fields_to_keep`.
    """
    field_kvs = dataclasses.asdict(self)
    if fields_to_keep is None:
      fields_to_keep = set(field_kvs.keys())
    return {k: v for k, v in field_kvs.items() if k in fields_to_keep}

  def to_table_row(
      self,
      sep = '\t',
      var_len_sep = ';',
  ):
    """Formats a record of a tabular database described by a dataclass as text.

    Args:
      sep: The character used to separate the different fields ("columns").
      var_len_sep: The character used to separate individual entries in
        variable length fields.

    Returns:
      A string representation of the (tabular) database record. Individual
      `bool`, `int`, and `float` fields are individually encoded as str using
      default methods. Variable-length fields have their entries delimited by
      `var_len_sep` and the textual representations of different fields are
      separated by `sep`. The newline character is *not* added by default.
    """
    cols = []
    for field in dataclasses.fields(self):
      value = getattr(self, field.name)
      value = value if isinstance(value, list) else [value]
      cols.append(var_len_sep.join([str(x) for x in value]))
    return sep.join(cols)

  @classmethod
  def generate_header(cls, sep = '\t'):
    """Returns a header string with field names separated by `sep`."""
    return sep.join([field.name for field in dataclasses.fields(cls)])


class ReadFromTable(beam.PTransform):
  r"""A `PTransform` for reading tabular flat text databases.

  Parses a text file as a collection of tabular records, with one record per
  line (delimited by `\n`). Each record is expected to consist of a fixed number
  of columns, delimited by some predefined character. Common examples include
  TSV and CSV formatted files. Additionally, variable-length "columns" are also
  supported, assuming the different entries within a column are delimited by
  another reserved character.

  Returns a `PCollection` of records, represented as `Record`s, that is,
  key-value pairs mapping database column names to their values. Optionally,
  each record in the `PCollection` may be paired with a key to be used by
  downstream `PTransform`s.

  Attributes:
    file_pattern: The file path to read from. The path can contain glob
      characters (`*`, `?` and `[...]` sets).
    schema_cls: A Python dataclass subclassing `TableRowMixin` whose fields
      describe the name and types of each column in the tabular database.
      Variable-length columns must be annotated as `list[base_type]`. Supported
      `base_type`s include `bool`, `int`, `float` and `str`. Other types will be
      treated as `str`, keeping the raw representation in the text file.
    skip_header_lines: The number of header lines to skip in each source file.
    key_field: The name of the field to serve as the key for each record. If
      provided, the output `PCollection` will consist of (key, record) tuples.
      If `None`, no keys will be included and the output `PCollection` will only
      contain the records.
    fields_to_keep: The names of the columns from the database to be kept. If
      `None`, all fields will be kept.
    delimiter: The character used to separate the different columns.
    var_len_delimiter: The character used to separate the different entries in a
      variable-length column.
  """

  def __init__(
      self,
      file_pattern,
      schema_cls,
      skip_header_lines = 0,
      key_field = None,
      fields_to_keep = None,
      delimiter = '\t',
      var_len_delimiter = ';',
  ):
    self.file_pattern = file_pattern
    self.schema_cls = schema_cls
    self.skip_header_lines = skip_header_lines
    self.key_field = key_field
    self.fields_to_keep = fields_to_keep
    self.delimiter = delimiter
    self.var_len_delimiter = var_len_delimiter

  def from_table_row(self, row):
    """Parses flat text row from the database into a dataclass instance."""
    return self.schema_cls.from_table_row(
        row, sep=self.delimiter, var_len_sep=self.var_len_delimiter)

  def to_dict(
      self,
      schema_record,
  ):
    """Converts record to dict or (key, dict) keeping the requested fields."""
    record = schema_record.to_dict(fields_to_keep=self.fields_to_keep)
    if self.key_field is not None:
      # Does not require `self.key_field` to be in `fields_to_keep`.
      key = getattr(schema_record, self.key_field)
      return key, record
    return record

  def expand(
      self,
      pcoll,
  ):
    return (
        pcoll
        | 'ReadRowsFromTable' >> beam.io.ReadFromText(
            file_pattern=self.file_pattern,
            skip_header_lines=self.skip_header_lines)
        | 'ParseRows' >> beam.Map(self.from_table_row)
        | 'ConvertRecordsToDict' >> beam.Map(self.to_dict))


class WriteToTable(beam.PTransform):
  """A `PTransform` for writing tabular flat text databases.

  Given a `PCollection` of `Records`s representing records from a tabular
  database (i.e. Python dictionaries mapping column names to their values),
  this `PTransform` will write them to flat text files with one line (row) per
  record. The different columns will be delimited by a customizable reserved
  character. Common use cases for this `PTransform` include writing TSV and CSV
  files, among others. Addionally, variable-length "columns" are also supported,
  in which case the entries within such columns will be delimited by another
  user-defined reserved character.

  Attributes:
    file_path_prefix: The file path to write to. The files written will begin
      with this prefix, followed by a shard identifier, and ending in a common
      extension given by `file_name_suffix`.
    schema_cls: A Python dataclass subclassing `TableRowMixin` whose fields
      describe the name and types of each column in the tabular database.
      Variable-length columns must be annotated as `list[base_type]`. Supported
      `base_type`s include `bool`, `int`, `float` and `str`.
    file_name_suffix: Suffix for the files written.
    num_shards: The number of files (shards) used for output. If not set (`0`),
      the service will decide on the optimal number of shards.
    delimiter: The character used to separate the different columns.
    var_len_delimiter: The character used to separate the different entries in a
      variable-length column.
    include_header: Whether to automatically generate a header for each shard
      based on `schema_cls`.
  """

  def __init__(
      self,
      file_path_prefix,
      schema_cls,
      file_name_suffix = '.tsv',
      num_shards = 0,
      delimiter = '\t',
      var_len_delimiter = ';',
      include_header = True,
  ):
    self.file_path_prefix = file_path_prefix
    self.schema_cls = schema_cls
    self.file_name_suffix = file_name_suffix
    self.num_shards = num_shards
    self.delimiter = delimiter
    self.var_len_delimiter = var_len_delimiter
    self.include_header = include_header

  def to_table_row(self, schema_record):
    """Formats dataclass record as flat text row for writing to the database."""
    return schema_record.to_table_row(
        sep=self.delimiter, var_len_sep=self.var_len_delimiter)

  def expand(self, pcoll):
    header = (self.schema_cls.generate_header(sep=self.delimiter)
              if self.include_header else None)
    return (
        pcoll
        | 'ReshuffleBeforeWrite' >> beam.Reshuffle()
        | 'ToDataclass' >> beam.Map(self.schema_cls.from_dict)
        | 'ToFlatTextRow' >> beam.Map(self.to_table_row)
        | 'WriteRowsToTextFile' >> beam.io.WriteToText(
            file_path_prefix=self.file_path_prefix,
            file_name_suffix=self.file_name_suffix,
            num_shards=self.num_shards,
            header=header))


class JoinTables(beam.PTransform):
  """A `PTransform` for joining multiple tabular databases.

  Given an input mapping of string table names to `PCollection`s representing
  tabular databases, it creates a single output `PCollection` that is the result
  of a "join" operation over all input records.

  Each individual tabular database is assumed to be a `PCollection` of `Record`s
  with their corresponding `Key`s. Every `Record` represents a map from column
  names to column values. Any set of `Record`s having the same `Key` will be
  merged, regardless of whether they originate from the same or different
  tables. When the same column name is present in two or more `Record`s being
  merged, their column values will be combined into a `Collection`.

  For example, given the following two tabular databases with three and four
  `(Key, Record)` pairs,
  ```
    table_1 = [
        ('key_a', {'field1': 1, 'var_len_field1': ['a']}),
        ('key_b', {'field1': 4, 'var_len_field1': ['d', 'e']}),
        ('key_c', {'field1': 6, 'var_len_field1': ['f']}),
    ]
    table_2 = [
        ('key_a', {'field1': 2, 'var_len_field1': ['b']}),
        ('key_b', {'field1': 5, 'var_len_field1': []}),
        ('key_a', {'field1': 3, 'var_len_field1': ['c']}),
        ('key_c', {'field2': 3.14}),
    ]
  ```
  the output of
  ```
    root = beam.Pipeline()
    table_1 = root | 'Create1' >> beam.Create(table_1)
    table_2 = root | 'Create2' >> beam.Create(table_2)
    joined = {'table_1': table_1, 'table_2': table_2} | schemas_lib.JoinTables()
  ```
  would be a `PCollection` with the following records
  ```
  [
      ('key_a', {'field1': [1, 2, 3], 'var_len_field1': ['a', 'b', 'c']}),
      ('key_b', {'field1': [4, 5], 'var_len_field1': ['d', 'e']}),
      ('key_c', {'field1': 6, 'var_len_field1': ['f'], 'field2': 3.14}),
  ]
  ```

  Remarks:
    + When merging records, some scalar columns maybe effectively "promoted" to
      variable-length if merged with another scalar or variable-length column
      with the same key. The responsibility of ensuring this may occur only in
      fields intended to be of variable length is delegated to the user.
    + Variable-length fields are internally stored as Python `list`s. However,
      their ordering is not guaranteed and thus should be thought of as
      `Collection`s instead.

  Attributes:
    left_join_tables: A collection of table names. If not None, any `Record`s
      that do *not* occur in *at least* one of these tables will be discarded.
  """

  def __init__(self, left_join_tables = None):
    self.left_join_tables = left_join_tables

  def merge_columns(
      self,
      key_grouped_records,
  ):
    """Merges a set of `Record`s sharing the same `Key`.

    Args:
      key_grouped_records: The result of a `beam.CoGroupByKey` `PTransform` over
        a map of `(table_name, PCollection[KeyRecordPair])` key-value pairs.

    Yields:
      Either 0 or 1 `Record`s. If the `left_join_tables` attribute is not
      specified (i.e., `None`) or is specified and the shared `Key` of the
      `Record`s being merged occurs in at least one of the `left_join_tables`,
      the iterator yields a single `Record` representing the union of all input
      `Record`s' columns. Otherwise, the iterator yields nothing, effectively
      discarding the input `Record`s.
    """
    key, grouped_records = key_grouped_records

    # If `self.left_join_tables` is specified, the record must occur in at least
    # one of these tables.
    if (self.left_join_tables is None or
        any(grouped_records[k] for k in self.left_join_tables)):
      merged_record: Record = {}
      for table_records in grouped_records.values():
        # The same key may occur more than once per table.
        for record in table_records:
          for col_name, col_val in record.items():
            # If the column is being encountered for the first time, the new
            # field may be of variable-length or not, depending on `col_val`.
            if col_name not in merged_record:
              # Copies variable-length column values to prevent input mutation.
              col_val = list(col_val) if isinstance(col_val, list) else col_val
              merged_record[col_name] = col_val
            # Otherwise, the new field has to be "promoted" to variable-length
            # if it wasn't already, regardless of whether `col_val` and
            # `merged_record[col_name]` were `list`s or not.
            else:
              col_val = col_val if isinstance(col_val, list) else [col_val]
              if isinstance(merged_record[col_name], list):
                merged_record[col_name].extend(col_val)
              else:
                values: SingleValue = merged_record[col_name]
                merged_record[col_name] = [values] + col_val
      yield key, merged_record

  def expand(
      self,
      tables,
  ):
    return tables | beam.CoGroupByKey() | beam.FlatMap(self.merge_columns)
