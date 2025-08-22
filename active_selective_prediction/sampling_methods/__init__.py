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

"""Import sampling methods."""

from active_selective_prediction.sampling_methods.average_kl_divergence_sampling import AverageKLDivergenceSampling
from active_selective_prediction.sampling_methods.average_margin_sampling import AverageMarginSampling
from active_selective_prediction.sampling_methods.badge_sampling import BADGESampling
from active_selective_prediction.sampling_methods.clue_sampling import CLUESampling
from active_selective_prediction.sampling_methods.confidence_sampling import ConfidenceSampling
from active_selective_prediction.sampling_methods.entropy_sampling import EntropySampling
from active_selective_prediction.sampling_methods.kcenter_greedy_sampling import KCenterGreedySampling
from active_selective_prediction.sampling_methods.margin_sampling import MarginSampling
from active_selective_prediction.sampling_methods.uniform_sampling import UniformSampling
