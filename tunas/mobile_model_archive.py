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

"""File containing specifications for Mobile baseline models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# pylint:disable=bad-whitespace
MOBILENET_V2_FILTERS        = (32, 16, 24, 32, 64,  96,  160, 320, 1280)
MNASNET_FILTERS             = (32, 16, 24, 40, 80,  96,  192, 320, 1280)
PROXYLESSNAS_MOBILE_FILTERS = (32, 16, 32, 40, 80,  96,  192, 320, 1280)
# pylint:enable=bad-whitespace

MOBILENET_V2_OPERATIONS = (
    0, 3, 3, 6, 6, 3, 3, 3, 6, 3, 3, 3, 3, 3, 3, 3, 6, 3, 3, 3, 6, 3
)
MNASNET_OPERATIONS = (
    0, 0, 0, 0, 6, 1, 1, 1, 6, 4, 4, 4, 6, 3, 3, 6, 6, 4, 4, 4, 4, 3
)
PROXYLESSNAS_MOBILE_OPERATIONS = (
    0, 1, 0, 6, 6, 2, 0, 1, 1, 5, 1, 1, 1, 4, 1, 1, 1, 5, 5, 2, 2, 5
)
