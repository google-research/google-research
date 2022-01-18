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

"""Custom TensorFlow ops for Aptamer learning models.
"""

import tensorflow.compat.v1 as tf

from ..learning import gen_custom_ops


count_all_dna_kmers = gen_custom_ops.count_all_dna_kmers
dna_sequence_to_indices = gen_custom_ops.dna_sequence_to_indices
