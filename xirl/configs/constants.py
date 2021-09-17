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

"""Settings we used for the CoRL 2021 experiments."""

from typing import Dict
from ml_collections import FrozenConfigDict

# The embodiments we used in the x-MAGICAL experiments.
EMBODIMENTS = frozenset([
    "shortstick",
    "mediumstick",
    "longstick",
    "gripper",
])

# All baseline pretraining strategies we ran for the CoRL experiments.
ALGORITHMS = frozenset([
    "xirl",
    "tcn",
    "lifs",
    "goal_classifier",
    "raw_imagenet",
])

# A mapping from x-MAGICAL embodiment to RL training iterations.
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

# A mapping from x-MAGICAL embodiment to Gym environment name.
XMAGICAL_EMBODIMENT_TO_ENV_NAME: Dict[str, str] = {
    k: f"SweepToTop-{k.capitalize()}-State-Allo-TestLayout-v0"
    for k in EMBODIMENTS
}

# A mapping from embodiment to scalar multipliers used by the XIRL algo.
#
# To obtain these values, we run `interact.py`, visually inspect what the
# absolute maximum value of the raw distance in embedding space is, then
# compute the inverse of that value.
_XMAGICAL_EMBODIMENT_TO_XIRL_DISTANCE_SCALE: Dict[str, float] = {
    "longstick": 1.0,
    "mediumstick": 1.0,
    "shortstick": 1.0,
    "gripper": 1.0,
}

# TODO(kevin): Fill these out once we rerun all baselines.
_XMAGICAL_EMBODIMENT_TO_TCN_DISTANCE_SCALE: Dict[str, float] = {}
_XMAGICAL_EMBODIMENT_TO_LIFS_DISTANCE_SCALE: Dict[str, float] = {}
_XMAGICAL_EMBODIMENT_TO_RAW_IMAGENET_DISTANCE_SCALE: Dict[str, float] = {}

# A mapping from algo to distance scale dict.
ALGO_TO_DISTANCE_SCALE_DICT: Dict[str, Dict[str, float]] = {
    "xirl": _XMAGICAL_EMBODIMENT_TO_XIRL_DISTANCE_SCALE,
    "tcn": _XMAGICAL_EMBODIMENT_TO_TCN_DISTANCE_SCALE,
    "lifs": _XMAGICAL_EMBODIMENT_TO_LIFS_DISTANCE_SCALE,
    "raw_imagenet": _XMAGICAL_EMBODIMENT_TO_RAW_IMAGENET_DISTANCE_SCALE,
}
