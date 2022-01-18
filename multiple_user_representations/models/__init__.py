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

"""Models."""

import tensorflow as tf

from multiple_user_representations.models import density_smoothed_retrieval
from multiple_user_representations.models import retrieval
from multiple_user_representations.models.mlp_item_model import ItemModelMLP
from multiple_user_representations.models.parametric_attention import SimpleParametricAttention
from multiple_user_representations.models.transformer_encoder_parametric_attention import ParametricAttentionEncoder

USER_MODELS = {
    'parametric_attention_encoder': ParametricAttentionEncoder,
    'simple_parametric_attention': SimpleParametricAttention,
}

RETRIEVAL_MODELS = {
    'standard_retrieval':
        retrieval.RetrievalModel,
    'density_smoothed_retrieval':
        density_smoothed_retrieval.DensityWeightedRetrievalModel,
}
