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

"""Constants used by Felix models."""

# Edit operations.
KEEP = 'KEEP'
DELETE = 'DELETE'
PAD_TAG = 'PAD'


# Special tokens.
PAD = '[PAD]'
CLS = '[CLS]'
SEP = '[SEP]'
MASK = '[MASK]'

# Special tokens that indicate the start and end of a span of deleted tokens.
DELETE_SPAN_START = '[unused1]'
DELETE_SPAN_END = '[unused2]'

# For filtering out input tokens which are not used.
DELETED_TAGS = frozenset([DELETE, PAD_TAG, PAD])
