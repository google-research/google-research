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

"""Information about the RICO dataset.

See http://interactionmining.org/rico for more details.
"""

import frozendict

COLORS = {
    "Web View": (66, 166, 246),
    "List Item": (0, 225, 179),
    "Multi-Tab": (55, 22, 18),
    "Input": (145, 0, 250),
    "Text Button": (100, 221, 57),
    "Slider": (26, 220, 221),
    "Background Image": (211, 20, 83),
    "Advertisement": (13, 71, 162),
    "Card": (216, 120, 227),
    "Bottom Navigation": (187, 104, 201),
    "Modal": (0, 256, 205),
    "On/Off Switch": (79, 196, 248),
    "Button Bar": (256, 206, 211),
    "Number Stepper": (175, 214, 130),
    "Text": (74, 20, 141),
    "Map View": (226, 81, 232),
    "Checkbox": (256, 139, 101),
    "Date Picker": (205, 102, 154),
    "Image": (241, 98, 147),
    "Drawer": (103, 58, 184),
    "Radio Button": (256, 184, 77),
    "Video": (0, 205, 0),
    "Toolbar": (77, 209, 226),
    "Pager Indicator": (19, 58, 69),
    "Icon": (255, 0, 0),
}

LABEL_NAMES = ("Text", "List Item", "Image", "Text Button", "Icon", "Toolbar",
               "Input", "Advertisement", "Card", "Web View", "Drawer",
               "Background Image", "Radio Button", "Modal", "Multi-Tab",
               "Pager Indicator", "Slider", "On/Off Switch", "Map View",
               "Bottom Navigation", "Video", "Checkbox", "Button Bar",
               "Number Stepper", "Date Picker")

ID_TO_LABEL = frozendict.frozendict(
    {i: v for (i, v) in enumerate(LABEL_NAMES)})

NUMBER_LABELS = len(ID_TO_LABEL)

LABEL_TO_ID_ = frozendict.frozendict(
    {l: i for i, l in ID_TO_LABEL.items()})
FRAME_WIDTH, FRAME_HEIGHT = 1440 // 3, 2560 // 3

