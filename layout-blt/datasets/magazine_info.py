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

"""Information about the MAGAZINE dataset.

See https://xtqiao.com/projects/content_aware_layout for more details.
"""

import frozendict


LABEL_NAMES = (
    "text", "image", "text-over-image", "headline", "headline-over-image"
    )

COLORS = {
    "text": (254, 231, 44),
    "image": (27, 187, 146),
    "headline": (255, 0, 0),
    "text-over-image": (0, 102, 255),
    "headline-over-image": (204, 0, 255),
    "background": (200, 200, 200),
}


FRAME_WIDTH = 225
FRAME_HEIGHT = 300


ID_TO_LABEL = frozendict.frozendict(
    {i: v for (i, v) in enumerate(LABEL_NAMES)})

NUMBER_LABELS = len(ID_TO_LABEL)

LABEL_TO_ID_ = frozendict.frozendict(
    {l: i for i, l in ID_TO_LABEL.items()})


