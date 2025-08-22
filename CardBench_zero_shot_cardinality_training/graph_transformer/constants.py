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

"""Constants for the graph transformer."""

NODE_FEATURE_DIM = 150
MAX_NUM_NODES = 32

SCALING_NUMERICAL_FEATURES = {
    "g": ["cardinality", "exec_time"],
    "tables": ["rows"],
    "attributes": ["num_unique"],
}

# The features that needs to be always removed
REMOVE_FEATURE_DICT = {
    "attributes": [
        "name",
        "min_numeric",
        "max_numeric",
        "min_string",
        "max_string",
        "percentiles_100_string",
    ],
    "tables": ["name"],
    "correlations": ["type"],
    "predicates": ["constant", "encoded_constant"],
}


CATEGORICAL_FEATURE_UNIQUE_DICT = {
    "attributes": {
        "data_type": (
            6,
            [b"INT64", b"DATE", b"TIMESTAMP", b"STRING", b"FLOAT64", b"TIME"],
        )
    },
    "ops": {"operator": (2, [b"join", b"scan"])},
    "predicates": {"predicate_operator": (10, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])},
    "correlations": {
        "validity": (4, [b"valid", b"invalidtypes", b"nan", b"none"])
    },
}

NODE_TYPES = [
    "pseudo_node",
    "attributes",
    "ops",
    "predicates",
    "correlations",
    "tables",
]

EDGE_TYPES = {
    "pseudo_edge": (None, "pseudo_node"),
    "table_to_attr": ("tables", "attributes"),
    "attr_to_pred": ("attributes", "predicates"),
    "pred_to_pred": ("predicates", "predicates"),
    "attr_to_op": ("attributes", "ops"),
    "op_to_op": ("ops", "ops"),
    "pred_to_op": ("predicates", "ops"),
    "attr_to_corr": ("attributes", "correlations"),
    "corr_to_pred": ("correlations", "predicates"),
}

LABELS = ["exec_time", "cardinality"]
