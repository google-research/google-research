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

# pylint: skip-file
import numpy as np
from aloe.common.configs import cmd_args

from aloe.rfill.utils.rfill_grammar_short import RFILL_VOCAB, trans_map, RFillNode, prod_rules, value_rules, expand_rules

RFILL_INV_VOCAB = {}
for key in RFILL_VOCAB:
    RFILL_INV_VOCAB[RFILL_VOCAB[key]] = key

state_map = {}
idx2state = {}
for key in trans_map:
    state = key[0]
    if not state in state_map:
        val = len(state_map)
        state_map[state] = val
    if not trans_map[key] in state_map:
        val = len(state_map)
        state_map[trans_map[key]] = val

for key in state_map:
    idx2state[state_map[key]] = key

IDX2STATE = idx2state
STATE_MAP = state_map

NUM_RFILL_STATES = len(state_map)

STATE_TRANS = np.zeros((NUM_RFILL_STATES, len(RFILL_VOCAB)), dtype=np.int64)

DECISION_MASK = np.zeros((NUM_RFILL_STATES, len(RFILL_VOCAB)), dtype=np.float32)

for key in trans_map:
    s1, v = key
    s2 = trans_map[key]
    s1 = state_map[s1]
    v = RFILL_VOCAB[v]
    s2 = state_map[s2]
    STATE_TRANS[s1, v] = s2
    DECISION_MASK[s1, v] = 1
