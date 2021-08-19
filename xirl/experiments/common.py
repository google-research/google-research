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

from ml_collections import FrozenConfigDict

# A mapping from x-MAGICAL embodiments to RL training iterations.
XMAGICALTrainingIterations = FrozenConfigDict({
    "longstick": 75_000,
    "mediumstick": 250_000,
    "shortstick": 500_000,
    "gripper": 500_000,
})

# A mapping from RLV environment to RL training iterations.
RLVTrainingIterations = FrozenConfigDict({
    "state_pusher": 500_000,
})
