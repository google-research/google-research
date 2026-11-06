# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Flowracle: Streamflow forecasting across California's snow-fed basins."""

from flowracle.data_pipeline import build_terrax_dataloader
from flowracle.data_pipeline import consolidate_targets
from flowracle.data_pipeline import get_batch_predictors_ds
from flowracle.data_pipeline import get_batch_target_ds
from flowracle.data_pipeline import get_expanding_window_cv_slices
from flowracle.data_pipeline import get_issuance_times
from flowracle.data_pipeline import load_and_align_datasets
from flowracle.data_pipeline import resolve_dataset_path
from flowracle.features import AUGMENTATION_EPOCHS
from flowracle.features import batch_to_table
from flowracle.features import NUM_ENSEMBLE
from flowracle.features import prepare_shared_features
from flowracle.features import QUANTILES
from flowracle.features import TARGET_SCALE
from flowracle.metrics import compute_composite_cv_objective
from flowracle.metrics import compute_crps
from flowracle.metrics import compute_hydrograph_timing_metrics
from flowracle.metrics import evaluate_against_b120
from flowracle.metrics import quantile_loss
from flowracle.model import fit_and_predict_fn
from flowracle.model import fit_model
from flowracle.model import load_model
from flowracle.model import predict

__all__ = [
    "AUGMENTATION_EPOCHS",
    "NUM_ENSEMBLE",
    "QUANTILES",
    "TARGET_SCALE",
    "batch_to_table",
    "build_terrax_dataloader",
    "compute_composite_cv_objective",
    "compute_crps",
    "compute_hydrograph_timing_metrics",
    "consolidate_targets",
    "evaluate_against_b120",
    "fit_and_predict_fn",
    "fit_model",
    "get_batch_predictors_ds",
    "get_batch_target_ds",
    "get_expanding_window_cv_slices",
    "get_issuance_times",
    "load_and_align_datasets",
    "load_model",
    "predict",
    "prepare_shared_features",
    "quantile_loss",
    "resolve_dataset_path",
]
