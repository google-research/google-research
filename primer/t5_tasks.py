# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""T5 LM Tasks."""

import functools

import t5.data
from t5.data import preprocessors as t5_preprocessors
from t5.evaluation import metrics as t5_metrics


TaskRegistry = t5.data.TaskRegistry
TfdsTask = t5.data.TfdsTask

DEFAULT_OUTPUT_FEATURES = {
    "inputs": t5.data.Feature(
        vocabulary=t5.data.get_default_vocabulary(), add_eos=True,
        required=False),
    "targets": t5.data.Feature(t5.data.get_default_vocabulary(), add_eos=True)
}


TaskRegistry.add(
    "lm1b_autoregressive_language_modeling",
    TfdsTask,
    tfds_name="lm1b:1.1.0",
    text_preprocessor=functools.partial(
        t5_preprocessors.rekey, key_map={"targets": "text"}),
    token_preprocessor=t5_preprocessors.unsupervised,
    output_features=DEFAULT_OUTPUT_FEATURES,
    splits={
        "train": "train[:98%]",
        "validation": "train[98%:]",
        "test": "test"
    },
    metric_fns=[t5_metrics.accuracy])


TaskRegistry.add(
    "c4_v220_autoregressive_language_modeling",
    TfdsTask,
    tfds_name="c4/en:3.0.1",
    text_preprocessor=functools.partial(
        t5_preprocessors.rekey, key_map={"targets": "text"}),
    token_preprocessor=t5_preprocessors.unsupervised,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[t5_metrics.accuracy])


TaskRegistry.add(
    "pg19_autoregressive_language_modeling",
    TfdsTask,
    tfds_name="pg19:0.1.1",
    text_preprocessor=functools.partial(
        t5_preprocessors.rekey, key_map={"targets": "book_text"}),
    token_preprocessor=t5_preprocessors.unsupervised,
    output_features=DEFAULT_OUTPUT_FEATURES,
    splits={
        "train": "train",
        "validation": "validation",
        "test": "test"
    },
    metric_fns=[t5_metrics.accuracy])
