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

"""Common type annotations for DEDAL data preprocessing pipelines."""

from typing import Dict, Iterable, List, MutableMapping, Tuple, Union

# The data preprocessing pipelines for DEDAL operate for the most part on
# tabular data, with the caveat that values may be scalar or variable-length.
# Only `bool`, `int`, `float` and `str` types are supported as base types so
# far. Variable-length values are internally represented as Python `list`s, but
# should be thought of as `Collection`s, since their order is not guaranteed to
# be deterministic.
SingleValue = Union[bool, int, float, str]
Value = Union[SingleValue, List[SingleValue]]
# Table rows are represented as key-value pairs with the column names serving as
# the keys.
Record = MutableMapping[str, Value]

# Often, a specific `str`-valued field of a `Record` may be designed to serve as
# the `Record`'s key for a Beam pipeline.
Key = str
KeyRecordPair = Tuple[Key, Record]

# After a `CoGroupByKey`, `Record`s from different (named) `PCollection`s having
# the same `Key` get grouped as a `dict` of `PCollection` names to a list of
# `Record`s.
Records = Iterable[Record]
GroupedRecords = Dict[str, Records]
KeyGroupedRecordsPair = Tuple[Key, GroupedRecords]
