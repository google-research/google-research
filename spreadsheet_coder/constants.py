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

"""Some common constants."""

SPECIAL_TOKEN_LIST = [
    "PAD", "EOF", "UNK", "GO", "empty", "$RANGESPLIT$", "$ENDRANGE$",
    "$ENDFORMULASKETCH$", "R[0]", "R[1]", "R[2]", "R[3]", "R[4]", "R[5]",
    "R[6]", "R[7]", "R[8]", "R[9]", "R[10]", "R[-1]", "R[-2]", "R[-3]", "R[-4]",
    "R[-5]", "R[-6]", "R[-7]", "R[-8]", "R[-9]", "R[-10]", "C[0]", "C[1]",
    "C[2]", "C[3]", "C[4]", "C[5]", "C[6]", "C[7]", "C[8]", "C[9]", "C[10]",
    "C[-1]", "C[-2]", "C[-3]", "C[-4]", "C[-5]", "C[-6]", "C[-7]", "C[-8]",
    "C[-9]", "C[-10]", "str:", "doub:", "bool:", "quotedstr:", "err:", "image:",
    "sparkchart:"
]
SPECIAL_TOKEN_SIZE = len(SPECIAL_TOKEN_LIST)

COL_ID = 29
END_FORMULA_SKETCH_ID = 7
END_RANGE_ID = 6
EOF_ID = 1
GO_ID = 3
RANGE_SPLIT_ID = 5
RANGE_TOKEN_ID = SPECIAL_TOKEN_SIZE
ROW_ID = 8
