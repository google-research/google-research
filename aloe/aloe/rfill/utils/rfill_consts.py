# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Robustfill program graph representation.

   Constants are predefined here.
"""

# pylint: skip-file
import string

CONSTANTS = ['dummy', ']', ',', '-', '.', '@', "'", '\"', '(', ')', ':', '%']

REGEXES = [None] * 10

REGEXES[1] = '[A-Z]([a-z])+'  # ProperCase
REGEXES[2] = '[A-Z]+'  # CAPS
REGEXES[3] = '[a-z]+'  # lowercase
REGEXES[4] = r'\d+'  # Digits
REGEXES[5] = '[a-zA-Z]+'  # Alphabets
REGEXES[6] = '[a-zA-Z0-9]+'  # Alphanumeric
REGEXES[7] = r'\s+'  # Whitespace
REGEXES[8] = '^'
REGEXES[9] = '$'


STR_VOCAB = (''.join(CONSTANTS[1:]) + string.ascii_lowercase +
             string.ascii_uppercase + string.digits) + ' '


RFILL_NODE_TYPES = {
    'ConstPos--1': 0,
    'ConstPos--2': 1,
    'ConstPos--3': 2,
    'ConstPos--4': 3,
    'ConstPos-0': 4,
    'ConstPos-1': 5,
    'ConstPos-2': 6,
    'ConstPos-3': 7,
    'ConstPos-4': 8,
    'ConstStr-1': 9,
    'ConstStr-10': 10,
    'ConstStr-11': 11,
    'ConstStr-2': 12,
    'ConstStr-3': 13,
    'ConstStr-4': 14,
    'ConstStr-5': 15,
    'ConstStr-6': 16,
    'ConstStr-7': 17,
    'ConstStr-8': 18,
    'ConstStr-9': 19,
    'ConstTok': 20,
    'RegPos': 21,
    'RegexTok': 22,
    'SubStr': 23,
    'c1-1': 24,
    'c1-10': 25,
    'c1-11': 26,
    'c1-2': 27,
    'c1-3': 28,
    'c1-4': 29,
    'c1-5': 30,
    'c1-6': 31,
    'c1-7': 32,
    'c1-8': 33,
    'c1-9': 34,
    'direct-End': 35,
    'direct-Start': 36,
    'expr_root': 37,
    'idx--1': 38,
    'idx--2': 39,
    'idx--3': 40,
    'idx--4': 41,
    'idx-0': 42,
    'idx-1': 43,
    'idx-2': 44,
    'idx-3': 45,
    'idx-4': 46,
    'r1-1': 47,
    'r1-2': 48,
    'r1-3': 49,
    'r1-4': 50,
    'r1-5': 51,
    'r1-6': 52,
    'r1-7': 53,
    'r1-8': 54,
    'r1-9': 55
}


RFILL_EDGE_TYPES = {
    'c1': 0,
    'direct': 1,
    'idx': 2,
    'p1': 3,
    'p2': 4,
    'pos_param': 5,
    'r1': 6,
    'subexpr': 7,
    'succ': 8,
    'rev-c1': 9,
    'rev-direct': 10,
    'rev-idx': 11,
    'rev-p1': 12,
    'rev-p2': 13,
    'rev-pos_param': 14,
    'rev-r1': 15,
    'rev-subexpr': 16,
    'rev-succ': 17
}
