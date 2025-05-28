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

"""PuddleWorld arenas."""

import frozendict
from shapely import geometry

from aux_tasks.puddle_world import puddle_world

# Repeated shapes
_TOP_WALL = geometry.Polygon((
    (0.4, 1.0), (0.4, 0.6), (0.6, 0.6), (0.6, 1.0)))
_BOTTOM_WALL = geometry.Polygon(
    ((0.4, 0.0), (0.4, 0.4), (0.6, 0.4), (0.6, 0.0)))

EMPTY = ()

HYDROGEN = (
    puddle_world.SlowPuddle(
        puddle_world.circle(geometry.Point((0.5, 0.5)), 0.3)),
    puddle_world.SlowPuddle(
        puddle_world.circle(geometry.Point((0.5, 0.5)), 0.15)),
)

SUTTON = (
    puddle_world.SlowPuddle(
        geometry.LineString([(0.1, 0.75), (0.45, 0.75)]).buffer(0.1)),
    puddle_world.SlowPuddle(
        geometry.LineString([(0.45, 0.4), (0.45, 0.8)]).buffer(0.1)),
)

TWO_ROOM = (
    puddle_world.WallPuddle(_TOP_WALL),
    puddle_world.WallPuddle(_BOTTOM_WALL),
)

TWO_ROOM_SLOW = (
    puddle_world.SlowPuddle(_TOP_WALL),
    puddle_world.SlowPuddle(_BOTTOM_WALL),
)

# LINT.IfChange(arena_names)
_ARENAS = frozendict.frozendict({
    'empty': EMPTY,
    'hydrogen': HYDROGEN,
    'sutton': SUTTON,
    'two_room': TWO_ROOM,
    'two_room_slow': TWO_ROOM_SLOW,
})
# LINT.ThenChange(../google/xm_compute_sr.py:arena_names)

ARENA_NAMES = list(_ARENAS.keys())


def get_arena(arena_name):
  if arena_name not in _ARENAS:
    raise ValueError(f'Unknown arena name {arena_name}')
  return _ARENAS[arena_name]
