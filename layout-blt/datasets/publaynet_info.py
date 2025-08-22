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

"""Information about the PubLayNet dataset."""

import frozendict

NUMBER_LABELS = 5

ID_TO_LABEL = frozendict.frozendict({
    0: "Text",
    1: "Title",
    2: "List",
    3: "Table",
    4: "Figure",
})

LABEL_TO_ID = frozendict.frozendict(
    {l: i for i, l in enumerate(ID_TO_LABEL)})

COLORS = frozendict.frozendict({
    "Title": (193, 0, 0),
    "List": (64, 44, 105),
    "Figure": (36, 234, 5),
    "Table": (89, 130, 213),
    "Text": (253, 141, 28),
})

FRAME_WIDTH, FRAME_HEIGHTT = 1050, 1485
