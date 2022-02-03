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

"""Register visual Adroit environments."""

from d4rl import hand_manipulation_suite
from gym.envs.registration import register

from rrlfd.adroit_ext.door_v0 import VisualDoorEnvV0
from rrlfd.adroit_ext.hammer_v0 import VisualHammerEnvV0
from rrlfd.adroit_ext.pen_v0 import VisualPenEnvV0
from rrlfd.adroit_ext.relocate_v0 import VisualRelocateEnvV0


camera_kwargs = {
    'camera_id': 'vil_camera',
    'im_size': 128,
}


register(
    id='visual-door-v0',
    entry_point='rrlfd.adroit_ext:VisualDoorEnvV0',
    max_episode_steps=200,
    kwargs={
        'ref_min_score': hand_manipulation_suite.DOOR_RANDOM_SCORE,
        'ref_max_score': hand_manipulation_suite.DOOR_EXPERT_SCORE,
        **camera_kwargs,
    }
)


register(
    id='visual-hammer-v0',
    entry_point='rrlfd.adroit_ext:VisualHammerEnvV0',
    max_episode_steps=200,
    kwargs={
        'ref_min_score': hand_manipulation_suite.HAMMER_RANDOM_SCORE,
        'ref_max_score': hand_manipulation_suite.HAMMER_EXPERT_SCORE,
        **camera_kwargs,
    },
)

register(
    id='visual-pen-v0',
    entry_point='rrlfd.adroit_ext:VisualPenEnvV0',
    max_episode_steps=100,
    kwargs={
        'ref_min_score': hand_manipulation_suite.PEN_RANDOM_SCORE,
        'ref_max_score': hand_manipulation_suite.PEN_EXPERT_SCORE,
        **camera_kwargs,
    },
)

register(
    id='visual-relocate-v0',
    entry_point='rrlfd.adroit_ext:VisualRelocateEnvV0',
    max_episode_steps=200,
    kwargs={
        'ref_min_score': hand_manipulation_suite.RELOCATE_RANDOM_SCORE,
        'ref_max_score': hand_manipulation_suite.RELOCATE_EXPERT_SCORE,
        **camera_kwargs,
    },
)
