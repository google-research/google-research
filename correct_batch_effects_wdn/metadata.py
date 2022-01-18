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

# Lint as: python3
"""Metadata constants."""

TREATMENT_GROUP = "treatment_group"
MOA = "moa"
COMPOUND = "compound"
CONCENTRATION = "concentration"
ACTIVITY = "activity"

BATCH = "batch"
PLATE = "plate"
WELL = "well"
ROW = "row"
COLUMN = "column"
SITE = "site"
TIMEPOINT = "timepoint"
SEQUENCE = "sequence"
CELL_DENSITY = "cell_density"
PASSAGE = "passage"
CELL_LINE_ID = "cell_line_id"
EMBEDDING_TYPE = "embedding_type"
UNIQUE_ID = "unique_id"

METADATA_ORDER = (TREATMENT_GROUP, MOA, COMPOUND, CONCENTRATION, ACTIVITY,
                  BATCH, PLATE, WELL, ROW, COLUMN, SITE, TIMEPOINT, SEQUENCE,
                  CELL_DENSITY, PASSAGE, CELL_LINE_ID)

NEGATIVE_CONTROL = "NEGATIVE_CONTROL"

# To represent unknown MOA.
UNKNOWN = "UNKNOWN"
