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

"""Constants for AP Parsing modeling."""
import frozendict

RESERVED_TOKENS = ("[PAD]", "[MASK]", "[UNK]")
METADATA_FEATURES = ("note_id", "char_offset", "seq_length")

CLASS_NAMES = frozendict.frozendict({
    "fragment_type": ("O", "B-PT", "I-PT", "B-PD", "I-PD", "B-AI", "I-AI"),
    "action_item_type":
        ("O", "MED", "IMG", "OBS", "CONS", "NUT", "THERP", "DIAG", "OTH")
})
FRAGMENT_TYPE_TO_ENUM = frozendict.frozendict({"PT": 1, "PD": 2, "AI": 3})
FEATURE_NAMES = ("token_ids", "token_type", "is_upper", "is_title")

# derived constants
TOKEN_IDS = FEATURE_NAMES[0]
LABEL_NAMES = tuple(CLASS_NAMES.keys())
MODEL_FEATURES = FEATURE_NAMES + LABEL_NAMES
