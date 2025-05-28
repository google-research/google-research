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

"""Table names that store statistics."""

# Before using replace:
# 1) project_name.table_name with your project and dataset name
# for the metadata database.
# 2) project_name.table_name_sampled_tables with your project and dataset name
# for the sample table database.
# 3) the json, query and training querygraph paths
# 4) the projectname and datasetnames of the data databases.

from CardBench_zero_shot_cardinality_training import database_connector
from CardBench_zero_shot_cardinality_training.definitions import Datatype

# Big Query schema pesudo-tables
BQ_INFO_SCHEMA_TABLES = ".INFORMATION_SCHEMA.TABLES"
BQ_INFO_SCHEMA_COLUMNS = ".INFORMATION_SCHEMA.COLUMNS"

DATA_DBTYPE = None  # replace with database_connector.DBType.X
METADATA_DBTYPE = None  # replace with database_connector.DBType.X

# Tables names where calculated statistics are stored
TABLES_INFO_TABLE = "X.tables_info"
COLUMNS_INFO_TABLE = "X.columns_info"
COLUMNS_STATS_TABLE = "X.columns_stats"
COLUMNS_INT64_EXTRA_STATS_TABLE = "X.columns_int64_extra_stats"
COLUMNS_UINT64_EXTRA_STATS_TABLE = "X.columns_uint64_extra_stats"
COLUMNS_FLOAT64_EXTRA_STATS_TABLE = "X.columns_float64_extra_stats"
COLUMNS_NUMERIC_EXTRA_STATS_TABLE = "X.columns_numeric_extra_stats"
# alias for numeric
COLUMNS_BIGNUMERIC_EXTRA_STATS_TABLE = "X.columns_bignumeric_extra_stats"
# alias for bignumeric
COLUMNS_STRING_EXTRA_STATS_TABLE = "X.columns_string_extra_stats"
COLUMNS_DATE_EXTRA_STATS_TABLE = "X.columns_date_extra_stats"
COLUMNS_DATETIME_EXTRA_STATS_TABLE = "X.columns_datetime_extra_stats"
COLUMNS_TIME_EXTRA_STATS_TABLE = "X.columns_time_extra_stats"
COLUMNS_TIMESTAMP_EXTRA_STATS_TABLE = "X.columns_timestamp_extra_stats"
CORRELATION_TABLE = "X.columns_correlation"
PK_FK_TABLE = "X.pk_fk"
COLUMNS_HISTOGRAM_TABLE = "X.histograms_table"
# 4k row tables
SAMPLE_PROJECTNAME_DATASET_NAME_4K = "X_sampled_tables"
WORKLOAD_DEFINITION_TABLE = "X.workload_definition"
QUERY_RUN_INFORMATION_TABLE = "X.query_run_information"
TEMP_QUERY_RUN_INFORMATION_TABLE_PREFIX = "X."
COLUMNS_BIGDECIMAL_EXTRA_STATS_TABLE = COLUMNS_BIGNUMERIC_EXTRA_STATS_TABLE
COLUMNS_UINT32_EXTRA_STATS_TABLE = COLUMNS_UINT64_EXTRA_STATS_TABLE
COLUMNS_INT32_EXTRA_STATS_TABLE = COLUMNS_INT64_EXTRA_STATS_TABLE
COLUMNS_DECIMAL_EXTRA_STATS_TABLE = COLUMNS_NUMERIC_EXTRA_STATS_TABLE

DIRECTORY_PATH_JSON_FILES = None
DIRECTORY_PATH_QUERY_FILES = None
DIRECTORY_TRAINING_QUERYGRAPH_OUTPUT = None

PROJECT_NAME = []
DATASET_NAMES = []



TYPES_TO_TABLES = {
    "INT64": COLUMNS_INT64_EXTRA_STATS_TABLE,
    "INT32": COLUMNS_INT32_EXTRA_STATS_TABLE,
    "UINT32": COLUMNS_UINT32_EXTRA_STATS_TABLE,
    "UINT64": COLUMNS_UINT64_EXTRA_STATS_TABLE,
    "FLOAT64": COLUMNS_FLOAT64_EXTRA_STATS_TABLE,
    "DOUBLE": COLUMNS_FLOAT64_EXTRA_STATS_TABLE,
    "NUMERIC": COLUMNS_NUMERIC_EXTRA_STATS_TABLE,
    "DECIMAL": COLUMNS_DECIMAL_EXTRA_STATS_TABLE,
    "BIGNUMERIC": COLUMNS_BIGNUMERIC_EXTRA_STATS_TABLE,
    "BIGDECIMAL": COLUMNS_BIGDECIMAL_EXTRA_STATS_TABLE,
    "STRING": COLUMNS_STRING_EXTRA_STATS_TABLE,
    "TIME": COLUMNS_TIME_EXTRA_STATS_TABLE,
    "TIMESTAMP": COLUMNS_TIMESTAMP_EXTRA_STATS_TABLE,
    "DATE": COLUMNS_DATE_EXTRA_STATS_TABLE,
    "DATETIME": COLUMNS_DATETIME_EXTRA_STATS_TABLE,
}


TYPES_TO_COLLECT_STATS = {
    "INT32": Datatype.INT,
    "INT64": Datatype.INT,
    "FLOAT64": Datatype.FLOAT,
    "UINT64": Datatype.INT,
    "NUMERIC": Datatype.NUMERIC,
    "TIME": Datatype.TIME,
    "TIMESTAMP": Datatype.TIMESTAMP,
    "DATE": Datatype.DATE,
    "DATETIME": Datatype.DATETIME,
    "STRING": Datatype.STRING,
    "DOUBLE": Datatype.FLOAT,
    "BIGNUMERIC": Datatype.NUMERIC,
    "BIGDECIMAL": Datatype.NUMERIC,
}


def is_type_we_dont_collect_stats(columntype):
  """We dont collect stats for these column types."""
  if "[]" in columntype:
    return True, Datatype.ARRAY
  elif "Enum" in columntype:
    return True, Datatype.ENUM
  elif "UNKNOWN_TYPE" in columntype:
    return True, Datatype.UNKNOWN_TYPE
  elif "ARRAY" in columntype:
    return True, Datatype.ARRAY
  elif "STRUCT" in columntype:
    return True, Datatype.STRUCT
  elif "ENUM" in columntype:
    return True, Datatype.ENUM
  elif "GEOGRAPHY" in columntype:
    return True, Datatype.GEOGRAPHY
  elif "JSON" in columntype:
    return True, Datatype.JSON
  elif "PROTO" in columntype:
    return True, Datatype.PROTO
  elif "BYTES" in columntype:
    return True, Datatype.BYTES
  elif "BOOLEAN" in columntype:
    return True, Datatype.BOOLEAN
  elif "BOOL" in columntype:
    return True, Datatype.BOOLEAN
  elif "INTERVAL" in columntype:
    return True, Datatype.INTERVAL
  elif "RANGE" in columntype:
    return True, Datatype.RANGE
  elif "INT" in columntype:
    return False, Datatype.INT
  elif "FLOAT" in columntype:
    return False, Datatype.FLOAT
  elif "STRING" in columntype:
    return False, Datatype.STRING
  elif "DATETIME" in columntype:
    return False, Datatype.DATETIME
  elif "TIMESTAMP" in columntype:
    return False, Datatype.TIMESTAMP
  elif "DATE" in columntype:
    return False, Datatype.DATE
  elif "TIME" in columntype:
    return False, Datatype.TIME
  elif "NUMERIC" in columntype:
    return False, Datatype.NUMERIC
  return False, "OOOPS"
