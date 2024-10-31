# Before using replace project_name.dataset_name. with your project
# and dataset names. These shoud match the names in the configuration.py file.

-- TABLES_INFO_TABLE
CREATE OR REPLACE TABLE `project_name.dataset_name.tables_info`(
  project_name STRING,
  dataset_name STRING,
  table_name STRING,
  row_count INT64,
  data_size_gib FLOAT64,
  file_count INT64,
  is_partitioned BOOL,
  is_clustered BOOL,
  partition_column STRING,
  clustered_columns ARRAY<STRING>,
  partition_column_type STRING);

-- COLUMNS_INFO_TABLE
CREATE OR REPLACE TABLE `project_name.dataset_name.columns_info`(
  project_name STRING,
  dataset_name STRING,
  table_name STRING,
  column_name STRING,
  column_type STRING,
  -- row_count INT64
);

-- COLUMNS_STATS_TABLE
CREATE OR REPLACE TABLE `project_name.dataset_name.columns_stats`(
  project_name STRING,
  dataset_name STRING,
  table_name STRING,
  column_name STRING,
  column_type STRING,
  null_frac FLOAT64,
  num_unique INT64,
  row_count INT64);

-- COLUMNS_INT64_EXTRA_STATS_TABLE
CREATE OR REPLACE TABLE `project_name.dataset_name.columns_int64_extra_stats`(
  project_name STRING,
  dataset_name STRING,
  table_name STRING,
  column_name STRING,
  min_val INT64,
  max_val INT64,
  percentiles ARRAY<INT64>,
  mean_val FLOAT64,
  allnull BOOL,
  row_count INT64);

-- COLUMNS_UINT64_EXTRA_STATS_TABLE
CREATE OR REPLACE TABLE `project_name.dataset_name.columns_uint64_extra_stats`(
  project_name STRING,
  dataset_name STRING,
  table_name STRING,
  column_name STRING,
  min_val bignumeric,
  max_val bignumeric,
  percentiles ARRAY<bignumeric>,
  mean_val FLOAT64,
  allnull BOOL,
  row_count INT64);

-- COLUMNS_FLOAT64_EXTRA_STATS_TABLE
CREATE OR REPLACE TABLE `project_name.dataset_name.columns_float64_extra_stats`(
  project_name STRING,
  dataset_name STRING,
  table_name STRING,
  column_name STRING,
  min_val FLOAT64,
  max_val FLOAT64,
  mean_val FLOAT64,
  percentiles ARRAY<FLOAT64>,
  allnull BOOL,
  row_count INT64);

-- COLUMNS_NUMERIC_EXTRA_STATS_TABLE
-- COLUMNS_DECIMAL_EXTRA_STATS_TABLE
CREATE OR REPLACE TABLE `project_name.dataset_name.columns_numeric_extra_stats`(
  project_name STRING,
  dataset_name STRING,
  table_name STRING,
  column_name STRING,
  min_val NUMERIC,
  max_val NUMERIC,
  percentiles ARRAY<NUMERIC>,
  mean_val NUMERIC,
  allnull BOOL,
  -- row_count INT64
);

-- COLUMNS_BIGNUMERIC_EXTRA_STATS_TABLE
-- COLUMNS_BIGDECIMAL_EXTRA_STATS_TABLE
CREATE OR REPLACE TABLE `project_name.dataset_name.columns_bignumeric_extra_stats`(
  project_name STRING,
  dataset_name STRING,
  table_name STRING,
  column_name STRING,
  min_val BIGNUMERIC,
  max_val BIGNUMERIC,
  mean_val BIGNUMERIC,
  percentiles BIGNUMERIC,
  allnull BOOL,
  row_count INT64);

-- COLUMNS_STRING_EXTRA_STATS_TABLE
CREATE OR REPLACE TABLE `project_name.dataset_name.columns_string_extra_stats`(
  project_name STRING,
  dataset_name STRING,
  table_name STRING,
  column_name STRING,
  min_val STRING,
  max_val STRING,
  freq_str_words ARRAY<STRING>,
  uniq_vals ARRAY<STRING>,
  allnull BOOL,
  max_length INT64,
  freq_str_words_do_not_need_to_collect BOOL,
  row_count INT64,
  did_not_find_enough_for_freq_word_picked_200_most_frequent BOOL);

-- COLUMNS_DATE_EXTRA_STATS_TABLE
CREATE OR REPLACE TABLE `project_name.dataset_name.columns_date_extra_stats`(
  project_name STRING,
  dataset_name STRING,
  table_name STRING,
  column_name STRING,
  min_val DATE,
  max_val DATE,
  uniq_vals ARRAY<DATE>,
  allnull BOOL,
  row_count INT64);

-- COLUMNS_DATETIME_EXTRA_STATS_TABLE
CREATE OR REPLACE TABLE `project_name.dataset_name.columns_datetime_extra_stats`(
  project_name STRING,
  dataset_name STRING,
  table_name STRING,
  column_name STRING,
  min_val DATETIME,
  max_val DATETIME,
  uniq_vals ARRAY<DATETIME>,
  allnull BOOL,
  row_count INT64);

-- COLUMNS_TIME_EXTRA_STATS_TABLE
CREATE OR REPLACE TABLE `project_name.dataset_name.columns_time_extra_stats`(
  project_name STRING,
  dataset_name STRING,
  table_name STRING,
  column_name STRING,
  min_val TIME,
  max_val TIME,
  uniq_vals ARRAY<TIME>,
  allnull BOOL,
  row_count INT64);

-- COLUMNS_TIMESTAMP_EXTRA_STATS_TABLE
CREATE OR REPLACE TABLE `project_name.dataset_name.columns_timestamp_extra_stats`(
  project_name STRING,
  dataset_name STRING,
  table_name STRING,
  column_name STRING,
  min_val TIMESTAMP,
  max_val TIMESTAMP,
  uniq_vals ARRAY<TIMESTAMP>,
  allnull BOOL,
  row_count INT64);

-- CORRELATION_TABLE
CREATE OR REPLACE TABLE `project_name.dataset_name.columns_correlation`(
  project_name STRING,
  dataset_name STRING,
  table_name_a STRING,
  table_name_b STRING,
  column_name_a STRING,
  column_name_b STRING,
  pearson_correlation FLOAT64);

-- PK_FK_TABLE
CREATE OR REPLACE TABLE `project_name.dataset_name.pk_fk`(
  project_name STRING,
  dataset_name STRING,
  primary_key_table_name STRING,
  primary_key_column_name STRING,
  foreign_key_table_name STRING,
  foreign_key_column_name STRING,
  column_type STRING);

-- COLUMNS_HISTOGRAM_TABLE
CREATE OR REPLACE TABLE `project_name.dataset_name.histograms_table`(
  project_name STRING,
  dataset_name STRING,
  table_name STRING,
  column_name STRING,
  column_type STRING,
  approx_quantiles ARRAY<STRING>,
  approx_quantiles_100 ARRAY<STRING>,
  approx_quantiles_1000 ARRAY<STRING>,
  approx_quantiles_500 ARRAY<STRING>);

-- WORKLOAD_DEFINITION_TABLE
CREATE OR REPLACE TABLE `project_name.dataset_name.workload_definition`(
  project_name STRING,
  dataset_name STRING,
  workload_id INT64,
  parameter_keys ARRAY<STRING>,
  parameter_values ARRAY<STRING>,
  queries_file_path STRING,
  creation_date TIMESTAMP);

-- QUERY_RUN_INFORMATION
CREATE OR REPLACE TABLE `project_name.dataset_name.query_run_information`(
  workload_id INT64,
  query_run_id INT64,
  query_string STRING,
  database_query_id STRING,
  cardinality INT64,);
