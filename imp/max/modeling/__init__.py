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

"""Aliases for all available layers."""

# Attention layers
from imp.max.modeling.attention import MultiHeadAttention

# Embedding layers
from imp.max.modeling.embeddings import Embed
from imp.max.modeling.embeddings import MaskFiller
from imp.max.modeling.embeddings import PosBiasEmbed
from imp.max.modeling.embeddings import PositionBias1D
from imp.max.modeling.embeddings import PositionBias3D
from imp.max.modeling.embeddings import SpatioTemporalPosEncode
from imp.max.modeling.embeddings import SpecialToken
from imp.max.modeling.embeddings import TemporalPosEncode

# Heads
from imp.max.modeling.heads import Classifier
from imp.max.modeling.heads import DisjointCommonSpace
from imp.max.modeling.heads import JointCommonSpace
from imp.max.modeling.heads import MAPHead
from imp.max.modeling.heads import MLP
from imp.max.modeling.heads import NonParametricAggregatorHead
from imp.max.modeling.heads import VitPostEncoderHead

# Linear projection layers
from imp.max.modeling.linear import Conv
from imp.max.modeling.linear import ConvLocal
from imp.max.modeling.linear import ConvTranspose
from imp.max.modeling.linear import Dense
from imp.max.modeling.linear import DenseGeneral

# Base Modeling Module
from imp.max.modeling.module import Model

# Mixture of Experts
from imp.max.modeling.moe import BaseMoE
from imp.max.modeling.moe import BaseSoftMoE
from imp.max.modeling.moe import BaseSparseMoE
from imp.max.modeling.moe import SoftMoEwithExpert
from imp.max.modeling.moe import SparseMoEwithExpert

# Multimodal layers
from imp.max.modeling.multimodal import extract_volume_patches
from imp.max.modeling.multimodal import FineAndCoarseCommonSpace
from imp.max.modeling.multimodal import PerModalityCLS
from imp.max.modeling.multimodal import PerModalityDense
from imp.max.modeling.multimodal import PerModalityDisjointCommonSpace
from imp.max.modeling.multimodal import PerModalityJointCommonSpace
from imp.max.modeling.multimodal import PerModalityMaskFiller
from imp.max.modeling.multimodal import PerModalitySpecialToken
from imp.max.modeling.multimodal import PerModalityTemperature
from imp.max.modeling.multimodal import RawToEmbed
from imp.max.modeling.multimodal import TokenIdToEmbed
from imp.max.modeling.multimodal import TokenRawToEmbed

# Normalization layers
from imp.max.modeling.normalization import LayerNorm

# Stacked layers
from imp.max.modeling.stacking import RematScannedStack
from imp.max.modeling.stacking import ReplicatedStack
from imp.max.modeling.stacking import SequentialStackCall

# Stochastic layers
from imp.max.modeling.stochastic import DropToken
from imp.max.modeling.stochastic import MaskToken

# Transformer layers/models
from imp.max.modeling.transformers import FeedForward
from imp.max.modeling.transformers import SoftMixtureOfFeedforward
from imp.max.modeling.transformers import SoftMoeTransformerDecoder
from imp.max.modeling.transformers import SoftMoeTransformerDecoderLayer
from imp.max.modeling.transformers import SoftMoeTransformerEncoder
from imp.max.modeling.transformers import SoftMoeTransformerEncoderLayer
from imp.max.modeling.transformers import SparseMixtureOfFeedforward
from imp.max.modeling.transformers import SparseMoeTransformerDecoder
from imp.max.modeling.transformers import SparseMoeTransformerDecoderLayer
from imp.max.modeling.transformers import SparseMoeTransformerEncoder
from imp.max.modeling.transformers import SparseMoeTransformerEncoderLayer
from imp.max.modeling.transformers import TransformerDecoder
from imp.max.modeling.transformers import TransformerDecoderLayer
from imp.max.modeling.transformers import TransformerEncoder
from imp.max.modeling.transformers import TransformerEncoderLayer
