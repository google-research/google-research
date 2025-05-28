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

"""Utilities for the Makita concentrated pipeline."""
from modAL import acquisition

from al_for_fep.data import utils
from al_for_fep.models import half_sample
from al_for_fep.models import sklearn_elasticnet_model
from al_for_fep.models import sklearn_gaussian_process_model
from al_for_fep.models import sklearn_gbm_model
from al_for_fep.models import sklearn_linear_model
from al_for_fep.models import sklearn_mlp_model
from al_for_fep.models import sklearn_rf_model
from al_for_fep.selection import acquisition_functions

QUERY_STRATEGIES = {
    'greedy': acquisition_functions.greedy,
    'half_sample': acquisition_functions.half_sample,
    'EI': acquisition.max_EI,
    'PI': acquisition.max_PI,
    'UCB': acquisition.max_UCB,
}

MODELS = {
    'rf': sklearn_rf_model.SklearnRfModel,
    'gbm': sklearn_gbm_model.SklearnGbmModel,
    'linear': sklearn_linear_model.SklearnLinearModel,
    'elasticnet': sklearn_elasticnet_model.SklearnElasticNetModel,
    'gp': sklearn_gaussian_process_model.SklearnGaussianProcessModel,
    'mlp': sklearn_mlp_model.SklearnMLPModel,
}

HALF_SAMPLE_WRAPPER = half_sample.HalfSampleRegressor

DATA_PARSERS = {
    'fingerprint':
        utils.parse_feature_smiles_morgan_fingerprint,
    'descriptors':
        utils.parse_feature_smiles_rdkit_properties,
    'fingerprints_and_descriptors':
        utils.parse_feature_smiles_morgan_fingerprint_with_descriptors,
    'vector':
        utils.parse_feature_vectors,
    'number':
        utils.parse_feature_numbers,
}
