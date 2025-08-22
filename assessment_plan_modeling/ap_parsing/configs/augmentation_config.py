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

"""Default augmentation config."""

from assessment_plan_modeling.ap_parsing import ap_parsing_lib
from assessment_plan_modeling.ap_parsing import augmentation_lib as aug_lib

FLATTEN_CLUSTER = aug_lib.AugmentationSequence(
    name="flatten_cluster",
    augmentation_sequence=[
        aug_lib.ShuffleClusters(),
        aug_lib.ShuffleFragments(),
        aug_lib.ChangeDelimAugmentation(
            fragment_types=[ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE],
            delims=["\n"],
            is_prefix=True),
        aug_lib.ChangeDelimAugmentation(
            fragment_types=[ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE],
            delims=[": ", " - ", ". ", ", "],
            is_prefix=False),
        aug_lib.ChangeDelimAugmentation(
            fragment_types=[
                ap_parsing_lib.LabeledSpanType.PROBLEM_DESCRIPTION,
                ap_parsing_lib.LabeledSpanType.ACTION_ITEM
            ],
            delims=[""],
            is_prefix=True),
        aug_lib.ChangeDelimAugmentation(
            fragment_types=[
                ap_parsing_lib.LabeledSpanType.PROBLEM_DESCRIPTION,
                ap_parsing_lib.LabeledSpanType.ACTION_ITEM
            ],
            delims=[" and ", "\n", ". ", ", "],
            is_prefix=False)
    ])

PARTIALLY_FLATTEN_CLUSTER = aug_lib.AugmentationSequence(
    name="partially_flatten_cluster",
    augmentation_sequence=[
        aug_lib.ShuffleClusters(),
        aug_lib.ShuffleFragments(),
        aug_lib.ChangeDelimAugmentation(
            fragment_types=[ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE],
            delims=[
                "\n",
                "\n# ",
                "\n*. ",
            ],
            probs=[0.5, 0.25, 0.25],
            is_prefix=True),
        aug_lib.ChangeDelimAugmentation(
            fragment_types=[ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE],
            delims=[": ", ":\n", " - ", ". "],
            is_prefix=False),
        aug_lib.ChangeDelimAugmentation(
            fragment_types=[
                ap_parsing_lib.LabeledSpanType.PROBLEM_DESCRIPTION,
                ap_parsing_lib.LabeledSpanType.ACTION_ITEM
            ],
            delims=["", "\n    - ", "\n- ", "\n    * "],
            is_prefix=True),
        aug_lib.ChangeDelimAugmentation(
            fragment_types=[
                ap_parsing_lib.LabeledSpanType.PROBLEM_DESCRIPTION,
                ap_parsing_lib.LabeledSpanType.ACTION_ITEM
            ],
            delims=[". "],
            is_prefix=False)
    ])

TITLES_MIXED_DELIMITERS = aug_lib.AugmentationSequence(
    name="titles_mixed_delimiters",
    augmentation_sequence=[
        aug_lib.ShuffleClusters(),
        aug_lib.ShuffleFragments(),
        aug_lib.ChangeDelimAugmentation(
            fragment_types=[ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE],
            delims=[
                "\n",
                "\n# ",
                "\n*. ",
            ],
            probs=[0.5, 0.25, 0.25],
            is_prefix=True),
    ])

TITLES_NUMBER_DELIMITERS = aug_lib.AugmentationSequence(
    name="titles_number_delimiters",
    augmentation_sequence=[
        aug_lib.ShuffleClusters(),
        aug_lib.ShuffleFragments(),
        aug_lib.NumberTitlesAugmentation(
            delims=["\n", "\n{}) ", "\n{}. "], probs=[0.5, 0.25, 0.25]),
    ])

DEFAULT_AUGMENTATION_CONFIG = aug_lib.AugmentationConfig(
    augmentation_sequences=[
        FLATTEN_CLUSTER, PARTIALLY_FLATTEN_CLUSTER, TITLES_MIXED_DELIMITERS,
        TITLES_NUMBER_DELIMITERS
    ],
    augmentation_number_poisson_lambda=2)
