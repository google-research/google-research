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

"""Definitions of types used for query generation and training graph generation."""

import enum


class Operator(enum.Enum):
  """Filter predicate operators."""

  NEQ = "!="
  EQ = "="
  LEQ = "<="
  GEQ = ">="
  LIKE = "LIKE"
  NOT_LIKE = "NOT LIKE"
  IS_NOT_NULL = "IS NOT NULL"
  IS_NULL = "IS NULL"
  IN = "IN"
  BETWEEN = "BETWEEN"

  def __str__(self):
    return self.value


class Aggregator(enum.Enum):
  """Aggregation functions."""

  AVG = "AVG"
  SUM = "SUM"
  COUNT = "COUNT"

  def __str__(self):
    return self.value


class ExtendedAggregator(enum.Enum):
  MIN = "MIN"
  MAX = "MAX"

  def __str__(self):
    return self.value


class Datatype(enum.Enum):
  """Column Data Types."""

  INT = "int"
  FLOAT = "float"
  CATEGORICAL = "categorical"
  STRING = "string"
  MISC = ("misc",)
  ARRAY = ("array",)
  JSON = ("json",)
  ENUM = ("enum",)
  GEOGRAPHY = ("geography",)
  PROTO = ("proto",)
  STRUCT = ("struct",)
  UNKNOWN_TYPE = ("UNKNOWN_TYPE",)
  BYTES = ("bytes",)
  BOOLEAN = "boolean"
  INTERVAL = ("interval",)
  RANGE = "range"
  TIME = "time"
  TIMESTAMP = ("timestamp",)
  DATE = ("date",)
  DATETIME = ("datetime",)
  NUMERIC = "numeric"

  def __str__(self):
    return "%s" % self.value
