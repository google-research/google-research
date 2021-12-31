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

"""This package provides core layers for ReadTwice.

Users of ReadTwice can import this package directly and refer to invidual layer
classes exposed from this file, for example:

  from readtwice import layers as readtwice_layers

  my_layer = readtwice_layers.TransformerWithSideInputLayers(...)
"""

from readtwice.layers import attention
from readtwice.layers import embedding
from readtwice.layers import helpers
from readtwice.layers import recomputing_dropout
from readtwice.layers import transformer
from readtwice.layers import wrappers

TransformerWithSideInputLayers = transformer.TransformerWithSideInputLayers

EmbeddingLookup = embedding.EmbeddingLookup

DenseLayers = helpers.DenseLayers
TrackedLambda = helpers.TrackedLambda

ResidualBlock = wrappers.ResidualBlock

RecomputingDropout = recomputing_dropout.RecomputingDropout

FusedSideAttention = attention.FusedSideAttention
SideAttention = attention.SideAttention
ProjectAttentionHeads = attention.ProjectAttentionHeads
