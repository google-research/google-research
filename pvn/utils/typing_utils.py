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

"""Typing Utils."""
from typing import Any, Dict

from typing_extensions import Literal

AtariGames = Literal[
    'AirRaid',
    'Alien',
    'Amidar',
    'Assault',
    'Asterix',
    'Asteroids',
    'Atlantis',
    'BankHeist',
    'BattleZone',
    'BeamRider',
    'Berzerk',
    'Bowling',
    'Boxing',
    'Breakout',
    'Carnival',
    'Centipede',
    'ChopperCommand',
    'CrazyClimber',
    'DemonAttack',
    'DoubleDunk',
    'ElevatorAction',
    'Enduro',
    'FishingDerby',
    'Freeway',
    'Frostbite',
    'Gopher',
    'Gravitar',
    'Hero',
    'IceHockey',
    'Jamesbond',
    'JourneyEscape',
    'Kangaroo',
    'Krull',
    'KungFuMaster',
    'MontezumaRevenge',
    'MsPacman',
    'NameThisGame',
    'Phoenix',
    'Pitfall',
    'Pong',
    'Pooyan',
    'PrivateEye',
    'Qbert',
    'Riverraid',
    'RoadRunner',
    'Robotank',
    'Seaquest',
    'Skiing',
    'Solaris',
    'SpaceInvaders',
    'StarGunner',
    'Tennis',
    'TimePilot',
    'Tutankham',
    'UpNDown',
    'Venture',
    'VideoPinball',
    'WizardOfWor',
    'YarsRevenge',
    'Zaxxon',
]

PyTree = Dict[Any, Any]
