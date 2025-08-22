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

"""Configuration for customized CTF environments.

This is an extension of the capture the flag environments in:
meltingpot/python/configs/substrates/paintball_capture_the_flag.py

The structure of all maps, prefabs, configs, etc. in this file (and folder as a
whole) follow the design patterns introduced there.
"""

from typing import Any, Dict, List, Optional, Tuple

from meltingpot.utils.substrates import shapes
from meltingpot.utils.substrates import specs
from ml_collections import config_dict
import numpy as np

from concept_marl.experiments.meltingpot.substrates import concept_specs

_COMPASS = ["N", "E", "S", "W"]

capture_the_flag_mod = """
IIIIIIIIIIIIIIIIIIIIIII
IWWWWWWWWWWWWWWWWWWWWWI
IWPPP,PPPP,F,PPPP,PPPWI
IWPPP,,PP,,,,,PP,,PPPWI
IWPPP,,,,,,,,,,,,,PPPWI
IWP,,WW,,,,,,,,,WW,,PWI
IWHHWWW,WWWWWWW,WWWHHWI
IWHHW,D,,,,,,,,,D,WHHWI
IWHH,,W,,,WWW,,,W,,HHWI
IW,,,,W,,,,,,,,,W,,,,WI
IW,,,,WWW,,,,,WWW,,,,WI
IW,,,,,,,,,I,,,,,,,,,WI
IW,,,,WWW,,,,,WWW,,,,WI
IW,,,,W,,,,,,,,,W,,,,WI
IWHH,,W,,,WWW,,,W,,HHWI
IWHHW,D,,,,,,,,,D,WHHWI
IWHHWWW,WWWWWWW,WWWHHWI
IWQ,,WW,,,,,,,,,WW,,QWI
IWQQQ,,,,,,,,,,,,,QQQWI
IWQQQ,,QQ,,,,,QQ,,QQQWI
IWQQQ,QQQQ,G,QQQQ,QQQWI
IWWWWWWWWWWWWWWWWWWWWWI
IIIIIIIIIIIIIIIIIIIIIII
"""

capture_the_flag_mod_mini = """
IIIIIIIIIIIIIIIII
IWWWWWWWWWWWWWWWI
IWP,PPP,F,PPP,PWI
IWP,,P,,,,,P,,PWI
IWW,W,WWWWW,W,WWI
IW,,W,,,,,,,W,,WI
IW,,WW,,,,,WW,,WI
IW,,,,,,I,,,,,,WI
IW,,WW,,,,,WW,,WI
IW,,W,,,,,,,W,,WI
IWW,W,WWWWW,W,WWI
IWQ,,Q,,,,,Q,,QWI
IWQ,QQQ,G,QQQ,QWI
IWWWWWWWWWWWWWWWI
IIIIIIIIIIIIIIIII
"""

ASCII_MAPS = {
    "capture_the_flag_mod": capture_the_flag_mod,
    "capture_the_flag_mod_mini": capture_the_flag_mod_mini,
}

# `prefab` determines which prefab game object to use for each `char` in the
# ascii map.
CHAR_PREFAB_MAP = {
    "P": {"type": "all", "list": ["spawn_point_red", "ground"]},
    "Q": {"type": "all", "list": ["spawn_point_blue", "ground"]},
    "W": "wall",
    "D": {"type": "choice",
          "list": ["destroyable_wall"] * 9 + ["destroyed_wall"]},
    "H": {"type": "choice",
          "list": ["destroyable_wall"] * 3 + ["destroyed_wall"]},
    ",": "ground",
    "I": {"type": "all", "list": ["indicator", "indicator_frame"]},
    "F": {"type": "all", "list": ["ground", "home_tile_red", "flag_red"]},
    "G": {"type": "all", "list": ["ground", "home_tile_blue", "flag_blue"]},
}

RED_COLOR = (225, 55, 85, 255)
DARKER_RED_COLOR = (200, 35, 55, 255)
DARKEST_RED_COLOR = (160, 5, 25, 255)

BLUE_COLOR = (85, 55, 225, 255)
DARKER_BLUE_COLOR = (55, 35, 200, 255)
DARKEST_BLUE_COLOR = (25, 5, 160, 255)

PURPLE_COLOR = (107, 63, 160, 255)


def multiply_tuple(color_tuple,
                   factor):
  alpha = color_tuple[3]
  return tuple([int(np.min([x * factor, alpha])) for x in color_tuple[0: 3]])

TEAMS_DATA = {
    "red": {"color": RED_COLOR,
            "spawn_group": "{}SpawnPoints".format("red")},
    "blue": {"color": BLUE_COLOR,
             "spawn_group": "{}SpawnPoints".format("blue")},
}

WALL = {
    "name": "wall",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "wall",
                "stateConfigs": [{
                    "state": "wall",
                    "layer": "upperPhysical",
                    "sprite": "Wall",
                }],
            }
        },
        {"component": "Transform",},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["Wall",],
                "spriteShapes": [shapes.WALL],
                "palettes": [{"*": (95, 95, 95, 255),
                              "&": (100, 100, 100, 255),
                              "@": (109, 109, 109, 255),
                              "#": (152, 152, 152, 255)}],
                "noRotates": [True]
            }
        },
        {
            "component": "AllBeamBlocker",
            "kwargs": {}
        },
    ]
}

INDICATOR_FRAME = {
    "name": "indicator_frame",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "inert",
                "stateConfigs": [
                    {"state": "inert",
                     "layer": "superOverlay",
                     "sprite": "InertFrame"}
                ]
            }
        },
        {"component": "Transform",},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["InertFrame"],
                "spriteShapes": [shapes.BUTTON],
                "palettes": [{"*": (0, 0, 0, 0),
                              "x": (55, 55, 55, 255),
                              "#": (0, 0, 0, 0)}],
                "noRotates": [True]
            }
        },
    ]
}


INDICATOR = {
    "name": "control_indicator",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "both",
                "stateConfigs": [
                    {
                        "state": "neither",
                        "layer": "background",
                        "sprite": "NeitherIndicator",
                    },
                    {
                        "state": "red",
                        "layer": "background",
                        "sprite": "RedIndicator",
                    },
                    {
                        "state": "blue",
                        "layer": "background",
                        "sprite": "BlueIndicator",
                    },
                    {
                        "state": "both",
                        "layer": "background",
                        "sprite": "BothIndicator",
                    },
                ]
            }
        },
        {"component": "Transform",},
        {
            "component": "Appearance",
            "kwargs": {
                "spriteNames": ["NeitherIndicator",
                                "RedIndicator",
                                "BlueIndicator",
                                "BothIndicator"],
                "spriteRGBColors": [(0, 0, 0, 0),
                                    DARKER_RED_COLOR,
                                    DARKER_BLUE_COLOR,
                                    PURPLE_COLOR]
            }
        },
        {"component": "ControlIndicator",},
    ]
}


def create_home_tile_prefab(team):
  """Return a home tile prefab, where the flag starts and must be brought."""
  sprite_name = "HomeTileFrame{}".format(team)
  prefab = {
      "name": "home_tile",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": "inert",
                  "stateConfigs": [
                      {"state": "inert",
                       "layer": "background",
                       "sprite": sprite_name}
                  ]
              }
          },
          {"component": "Transform",},
          {
              "component": "Appearance",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "spriteNames": [sprite_name],
                  "spriteShapes": [shapes.BUTTON],
                  "palettes": [{"*": (0, 0, 0, 0),
                                "x": (0, 0, 0, 0),
                                "#": (218, 165, 32, 255)}],
                  "noRotates": [True]
              }
          },
          {
              "component": "HomeTile",
              "kwargs": {
                  "team": team,
              }
          },
      ]
  }
  return prefab


def create_ground_prefab():
  """Return a prefab for a colorable ground prefab."""
  sprite_names = ["RedGround", "BlueGround"]
  sprite_colors = [DARKEST_RED_COLOR, DARKEST_BLUE_COLOR]
  prefab = {
      "name": "ground",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": "clean",
                  "stateConfigs": [
                      {
                          "state": "clean",
                          "layer": "alternateLogic",
                      },
                      {
                          "state": "red",
                          "layer": "alternateLogic",
                          "sprite": sprite_names[0],
                      },
                      {
                          "state": "blue",
                          "layer": "alternateLogic",
                          "sprite": sprite_names[1],
                      },
                  ]
              }
          },
          {"component": "Transform",},
          {
              "component": "Appearance",
              "kwargs": {
                  "spriteNames": sprite_names,
                  "spriteRGBColors": sprite_colors
              }
          },
          {
              "component": "Ground",
              "kwargs": {
                  "teamNames": ["red", "blue"],
              }
          },
      ]
  }
  return prefab


def create_destroyable_wall_prefab(initial_state):
  """Return destroyable wall prefab, potentially starting in destroyed state."""
  if initial_state == "destroyed":
    initial_health = 0
  else:
    initial_health = 5
  prefab = {
      "name": "destroyableWall",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": initial_state,
                  "stateConfigs": [
                      {
                          "state": "destroyable",
                          "layer": "upperPhysical",
                          "sprite": "DestroyableWall",
                      },
                      {
                          "state": "damaged",
                          "layer": "upperPhysical",
                          "sprite": "DamagedWall",
                      },
                      {
                          "state": "destroyed",
                          "layer": "alternateLogic",
                          "sprite": "Rubble",
                      },
                  ],
              }
          },
          {"component": "Transform",},
          {
              "component": "Appearance",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "spriteNames": ["DestroyableWall",
                                  "DamagedWall",
                                  "Rubble"],
                  "spriteShapes": [shapes.WALL,
                                   shapes.WALL,
                                   shapes.WALL],
                  "palettes": [{"*": (55, 55, 55, 255),
                                "&": (100, 100, 100, 255),
                                "@": (109, 109, 109, 255),
                                "#": (152, 152, 152, 255)},
                               {"*": (55, 55, 55, 255),
                                "&": (100, 100, 100, 255),
                                "@": (79, 79, 79, 255),
                                "#": (152, 152, 152, 255)},
                               {"*": (0, 0, 0, 255),
                                "&": (0, 0, 0, 255),
                                "@": (29, 29, 29, 255),
                                "#": (0, 0, 0, 255)}],
                  "noRotates": [True] * 3
              }
          },
          {
              "component": "Destroyable",
              "kwargs": {"hitNames": ["red", "blue"],
                         "initialHealth": initial_health,
                         "damagedHealthLevel": 2}
          }
      ]
  }
  return prefab


def create_spawn_point_prefab(team):
  """Return a team-specific spawn-point prefab."""
  prefab = {
      "name": "spawn_point",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": "playerSpawnPoint",
                  "stateConfigs": [{
                      "state": "playerSpawnPoint",
                      "layer": "logic",
                      "groups": [TEAMS_DATA[team]["spawn_group"]],
                  }],
              }
          },
          {"component": "Transform",},
          {
              "component": "Appearance",
              "kwargs": {
                  "renderMode": "invisible",
                  "spriteNames": [],
                  "spriteRGBColors": []
              }
          },
      ]
  }
  return prefab


def create_flag_prefab(team):
  """Return a team-specific flag prefab."""
  dropped_sprite_name = "DroppedFlag_{}".format(team)
  carried_sprite_name = "CarriedFlag_{}".format(team)
  if team == "red":
    flag_color = RED_COLOR
  elif team == "blue":
    flag_color = BLUE_COLOR

  prefab = {
      "name": "{}_flag".format(team),
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": "dropped",
                  "stateConfigs": [
                      {
                          "state": "dropped",
                          "layer": "lowerPhysical",
                          "sprite": dropped_sprite_name,
                      },
                      {
                          "state": "carried",
                          "layer": "overlay",
                          "sprite": carried_sprite_name,
                      },
                      {
                          "state": "wait",
                      }
                  ]
              }
          },
          {"component": "Transform",},
          {
              "component": "Appearance",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "spriteNames": [dropped_sprite_name, carried_sprite_name],
                  "spriteShapes": [shapes.FLAG,
                                   shapes.FLAG_HELD],
                  "palettes": [shapes.get_palette(flag_color)] * 2,
                  "noRotates": [True, True]
              }
          },
          {
              "component": "Flag",
              "kwargs": {
                  "team": team,
              }
          }
      ]
  }
  return prefab

# PREFABS is a dictionary mapping names to template game objects that can
# be cloned and placed in multiple locations accoring to an ascii map.
PREFABS = {
    "wall": WALL,
    "spawn_point_red": create_spawn_point_prefab("red"),
    "spawn_point_blue": create_spawn_point_prefab("blue"),
    "destroyable_wall": create_destroyable_wall_prefab("destroyable"),
    "destroyed_wall": create_destroyable_wall_prefab("destroyed"),
    "ground": create_ground_prefab(),
    "indicator": INDICATOR,
    "indicator_frame": INDICATOR_FRAME,
    "flag_red": create_flag_prefab("red"),
    "flag_blue": create_flag_prefab("blue"),
    "home_tile_red": create_home_tile_prefab("red"),
    "home_tile_blue": create_home_tile_prefab("blue"),
}

# Primitive action components.
# pylint: disable=bad-whitespace
# pyformat: disable
NOOP       = {"move": 0, "turn":  0, "fireZap": 0}
FORWARD    = {"move": 1, "turn":  0, "fireZap": 0}
STEP_RIGHT = {"move": 2, "turn":  0, "fireZap": 0}
BACKWARD   = {"move": 3, "turn":  0, "fireZap": 0}
STEP_LEFT  = {"move": 4, "turn":  0, "fireZap": 0}
TURN_LEFT  = {"move": 0, "turn": -1, "fireZap": 0}
TURN_RIGHT = {"move": 0, "turn":  1, "fireZap": 0}
FIRE_ZAP_A = {"move": 0, "turn":  0, "fireZap": 1}
FIRE_ZAP_B = {"move": 0, "turn":  0, "fireZap": 2}
# pyformat: enable
# pylint: enable=bad-whitespace

ACTION_SET = (
    NOOP,
    FORWARD,
    BACKWARD,
    STEP_LEFT,
    STEP_RIGHT,
    TURN_LEFT,
    TURN_RIGHT,
    FIRE_ZAP_A,  # a short-range beam with a wide area of effect
    FIRE_ZAP_B,  # a longer range beam with a thin area of effect
)


# The Scene is a non-physical object, its components implement global logic.
def create_scene(num_players, num_flags):
  """Creates the global scene."""
  scene = {
      "name": "scene",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": "scene",
                  "stateConfigs": [
                      {
                          "state": "scene",
                      }
                  ],
              },
          },
          {
              "component": "Transform",
          },
          {"component": "FlagManager", "kwargs": {}},
          {
              "component": "GlobalStateTracker",
              "kwargs": {
                  "numPlayers": num_players,
                  "numFlags": num_flags,
              },
          },
          {
              "component": "GlobalMetricReporter",
              "kwargs": {
                  "metrics": [
                      {
                          "name": "CONCEPT_AGENT_POSITIONS",
                          "type": "tensor.Int32Tensor",
                          "shape": (num_players, num_players, 2),
                          "component": "GlobalStateTracker",
                          "variable": "playerPositions",
                      },
                      {
                          "name": "CONCEPT_AGENT_ORIENTATIONS",
                          "type": "tensor.Int32Tensor",
                          "shape": (
                              num_players,
                              num_players,
                          ),
                          "component": "GlobalStateTracker",
                          "variable": "playerOrientations",
                      },
                      {
                          "name": "CONCEPT_AGENT_HEALTH_STATES",
                          "type": "tensor.Int32Tensor",
                          "shape": (
                              num_players,
                              num_players,
                          ),
                          "component": "GlobalStateTracker",
                          "variable": "playerHealthStates",
                      },
                      {
                          "name": "CONCEPT_FLAG_POSITIONS",
                          "type": "tensor.Int32Tensor",
                          "shape": (num_players, num_flags, 2),
                          "component": "GlobalStateTracker",
                          "variable": "flagPositions",
                      },
                      {
                          "name": "CONCEPT_AGENT_HAS_FLAG",
                          "type": "tensor.Int32Tensor",
                          "shape": (num_players, num_players,),
                          "component": "GlobalStateTracker",
                          "variable": "playerHasFlags",
                      },
                      {
                          "name": "CONCEPT_FLAG_STATE_INDICATOR",
                          "type": "tensor.Int32Tensor",
                          "shape": (num_players, 1,),
                          "component": "GlobalStateTracker",
                          "variable": "flagIndicatorStates",
                      },
                      {
                          "name": "CONCEPT_AGENT_TOP_CELL_STATE",
                          "type": "tensor.Int32Tensor",
                          "shape": (num_players, 1,),
                          "component": "GlobalStateTracker",
                          "variable": "playerSurroundingTop",
                      },
                      {
                          "name": "CONCEPT_AGENT_BOT_CELL_STATE",
                          "type": "tensor.Int32Tensor",
                          "shape": (num_players, 1,),
                          "component": "GlobalStateTracker",
                          "variable": "playerSurroundingBot",
                      },
                      {
                          "name": "CONCEPT_AGENT_LEFT_CELL_STATE",
                          "type": "tensor.Int32Tensor",
                          "shape": (num_players, 1,),
                          "component": "GlobalStateTracker",
                          "variable": "playerSurroundingLeft",
                      },
                      {
                          "name": "CONCEPT_AGENT_RIGHT_CELL_STATE",
                          "type": "tensor.Int32Tensor",
                          "shape": (num_players, 1,),
                          "component": "GlobalStateTracker",
                          "variable": "playerSurroundingRight",
                      },
                      {
                          "name": "CONCEPT_AGENT_TOP_LEFT_CELL_STATE",
                          "type": "tensor.Int32Tensor",
                          "shape": (num_players, 1,),
                          "component": "GlobalStateTracker",
                          "variable": "playerSurroundingTopLeft",
                      },
                      {
                          "name": "CONCEPT_AGENT_TOP_RIGHT_CELL_STATE",
                          "type": "tensor.Int32Tensor",
                          "shape": (num_players, 1,),
                          "component": "GlobalStateTracker",
                          "variable": "playerSurroundingTopRight",
                      },
                      {
                          "name": "CONCEPT_AGENT_BOT_LEFT_CELL_STATE",
                          "type": "tensor.Int32Tensor",
                          "shape": (num_players, 1,),
                          "component": "GlobalStateTracker",
                          "variable": "playerSurroundingBotLeft",
                      },
                      {
                          "name": "CONCEPT_AGENT_BOT_RIGHT_CELL_STATE",
                          "type": "tensor.Int32Tensor",
                          "shape": (num_players, 1,),
                          "component": "GlobalStateTracker",
                          "variable": "playerSurroundingBotRight",
                      },
                  ]
              },
          },
      ],
  }
  return scene


def create_avatar_object(
    player_idx,
    team,
    override_taste_kwargs = None):
  """Create an avatar object."""
  # Lua is 1-indexed.
  lua_index = player_idx + 1

  team_color = TEAMS_DATA[team]["color"]

  health1_avatar_sprite_name = "avatarSprite{}Health1".format(lua_index)
  health2_avatar_sprite_name = "avatarSprite{}Health2".format(lua_index)
  health3_avatar_sprite_name = "avatarSprite{}Health3".format(lua_index)

  health1_color_palette = shapes.get_palette(multiply_tuple(team_color, 0.35))
  health2_color_palette = shapes.get_palette(team_color)
  health3_color_palette = shapes.get_palette(multiply_tuple(team_color, 1.75))

  taste_kwargs = {
      "defaultTeamReward": 1.0,
      "rewardForZapping": 0.01,
      "extraRewardForZappingFlagCarrier": 0.01,
      "rewardForReturningFlag": 0.01,
      "rewardForPickingUpOpposingFlag": 0.01,
  }
  if override_taste_kwargs:
    taste_kwargs.update(override_taste_kwargs)

  avatar_object = {
      "name": "avatar",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": "health2",
                  "stateConfigs": [
                      {"state": "health1",
                       "layer": "upperPhysical",
                       "sprite": health1_avatar_sprite_name,
                       "contact": "avatar",
                       "groups": ["players"]},
                      {"state": "health2",
                       "layer": "upperPhysical",
                       "sprite": health2_avatar_sprite_name,
                       "contact": "avatar",
                       "groups": ["players"]},
                      {"state": "health3",
                       "layer": "upperPhysical",
                       "sprite": health3_avatar_sprite_name,
                       "contact": "avatar",
                       "groups": ["players"]},

                      # Player wait state used when they have been zapped out.
                      {"state": "playerWait",
                       "groups": ["playerWaits"]},
                  ]
              }
          },
          {"component": "Transform",},
          {
              "component": "Appearance",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "spriteNames": [health1_avatar_sprite_name,
                                  health2_avatar_sprite_name,
                                  health3_avatar_sprite_name],
                  "spriteShapes": [shapes.CUTE_AVATAR,
                                   shapes.CUTE_AVATAR,
                                   shapes.CUTE_AVATAR],
                  "palettes": [health1_color_palette,
                               health2_color_palette,
                               health3_color_palette],
                  "noRotates": [True] * 3
              }
          },
          {
              "component": "Avatar",
              "kwargs": {
                  "index": lua_index,
                  "aliveState": "health2",
                  "additionalLiveStates": ["health1", "health3"],
                  "waitState": "playerWait",
                  "spawnGroup": TEAMS_DATA[team]["spawn_group"],
                  "actionOrder": ["move",
                                  "turn",
                                  "fireZap"],
                  "actionSpec": {
                      "move": {"default": 0, "min": 0, "max": len(_COMPASS)},
                      "turn": {"default": 0, "min": -1, "max": 1},
                      "fireZap": {"default": 0, "min": 0, "max": 1},
                  },
                  "view": {
                      "left": 2,
                      "right": 2,
                      "forward": 3,
                      "backward": 1,
                      "centered": False
                  },
                  # The following kwarg makes it possible to get rewarded for
                  # team rewards even when an avatar is "dead".
                  "skipWaitStateRewards": False,
              }
          },
          {
              "component": "ColorZapper",
              "kwargs": {
                  "team": team,
                  # The color zapper beam is somewhat transparent.
                  "color": (team_color[0], team_color[1], team_color[2], 150),
                  "cooldownTime": 2,
                  "beamLength": 3,
                  "beamRadius": 1,
                  "secondaryBeamCooldownTime": 4,
                  "secondaryBeamLength": 6,
                  "secondaryBeamRadius": 0,
                  "aliveStates": ["health1", "health2", "health3"],
              }
          },
          {
              "component": "ReadyToShootObservation",
              "kwargs": {
                  "zapperComponent": "ColorZapper",
              }
          },
          {
              "component": "ZappedByColor",
              "kwargs": {
                  "team": team,
                  "allTeamNames": ["red", "blue"],
                  "framesTillRespawn": 80,
                  "penaltyForBeingZapped": 0,
                  "rewardForZapping": 0,
                  "healthRegenerationRate": 0.05,
                  "maxHealthOnGround": 2,
                  "maxHealthOnOwnColor": 3,
                  "maxHealthOnEnemyColor": 1,
                  "groundLayer": "alternateLogic",
              }
          },
          {
              "component": "TeamMember",
              "kwargs": {"team": team}
          },
          {
              "component": "Taste",
              "kwargs": taste_kwargs
          },
          {
              "component": "LocationObserver",
              "kwargs": {
                  "objectIsAvatar": True,
                  "alsoReportOrientation": True
              }
          },
      ]
  }

  return avatar_object


def _even_vs_odd_team_assignment(
    num_players,
    taste_kwargs = None):
  """Assign players with even ids to red team and odd ids to blue team."""
  avatar_objects = []
  for player_idx in range(0, num_players):
    if player_idx % 2 == 0:
      team = "red"
    elif player_idx % 2 == 1:
      team = "blue"
    game_object = create_avatar_object(player_idx, team,
                                       override_taste_kwargs=taste_kwargs)
    avatar_objects.append(game_object)

  return avatar_objects


def _low_vs_high_team_assignment(
    num_players,
    taste_kwargs = None):
  """Assign players with id below the median id to blue and above it to red."""
  median = np.median(range(num_players))
  avatar_objects = []
  for player_idx in range(0, num_players):
    if player_idx < median:
      team = "blue"
    elif player_idx > median:
      team = "red"
    game_object = create_avatar_object(player_idx, team,
                                       override_taste_kwargs=taste_kwargs)
    avatar_objects.append(game_object)

  return avatar_objects


def create_avatar_objects(
    num_players,
    taste_kwargs = None,
    fixed_teams = False):
  """Returns list of avatar objects of length 'num_players'."""
  assert num_players % 2 == 0, "num players must be divisible by 2"
  if fixed_teams:
    avatar_objects = _low_vs_high_team_assignment(num_players,
                                                  taste_kwargs=taste_kwargs)
  else:
    avatar_objects = _even_vs_odd_team_assignment(num_players,
                                                  taste_kwargs=taste_kwargs)
  return avatar_objects


def create_lab2d_settings(
    ascii_map,
    num_players,
    num_flags,
    avatar_taste_kwargs = None,
    fixed_teams = False):
  """Returns the lab2d settings."""
  ascii_map = ASCII_MAPS[ascii_map]

  # Lua script configuration.
  lab2d_settings = {
      "levelName":
          "paintball_capture_the_flag",
      "levelDirectory":
          "../experiments/meltingpot/lua/levels",
      "numPlayers":
          num_players,
      "maxEpisodeLengthFrames":
          1000,
      "spriteSize":
          8,
      "topology":
          "BOUNDED",  # Choose from ["BOUNDED", "TORUS"],
      "simulation": {
          "map": ascii_map,
          "gameObjects": create_avatar_objects(num_players,
                                               taste_kwargs=avatar_taste_kwargs,
                                               fixed_teams=fixed_teams),
          "scene": create_scene(num_players, num_flags),
          "prefabs": PREFABS,
          "charPrefabMap": CHAR_PREFAB_MAP,
      },
  }
  return lab2d_settings


def create_concept_spec(timestep_spec,
                        concept_prefix):
  """Create concept spec for this environment."""
  concepts = {}
  for key, value in timestep_spec.items():
    if key.startswith(concept_prefix):
      concepts[key] = {
          "name":
              key,
          "concept_type":
              value.concept_type,
          "object_type":
              value.object_type,
          "num_objs":
              value.shape[0],
          "num_values":
              value.num_categories if isinstance(
                  value, concept_specs.CategoricalConceptArray) else 1,
      }

  concept_spec = {"prefix": concept_prefix, "concepts": concepts}
  return concept_spec


def get_config(ascii_map):
  """Default configuration for training on the capture_the_flag level."""
  config = config_dict.ConfigDict()

  # Basic configuration.
  if ascii_map == "clean_up_mod":
    config.num_players = 8
  else:
    config.num_players = 4
  config.num_flags = 2
  # config.num_indicator_tiles = 1
  config.lab2d_settings = create_lab2d_settings(ascii_map,
                                                config.num_players,
                                                config.num_flags,
                                                fixed_teams=True)

  # Action set configuration.
  config.action_set = ACTION_SET
  # Observation format configuration.
  config.individual_observation_names = [
      "RGB",
      "READY_TO_SHOOT",
      "POSITION",
      "ORIENTATION",
  ]
  concept_prefix = "WORLD.CONCEPT"
  config.global_observation_names = [
      "WORLD.RGB",
      "WORLD.CONCEPT_AGENT_POSITIONS",
      "WORLD.CONCEPT_AGENT_ORIENTATIONS",
      "WORLD.CONCEPT_AGENT_HEALTH_STATES",
      "WORLD.CONCEPT_FLAG_POSITIONS",
      "WORLD.CONCEPT_AGENT_HAS_FLAG",
      "WORLD.CONCEPT_FLAG_STATE_INDICATOR",
      "CONCEPT_AGENT_TOP_CELL_STATE",
      "CONCEPT_AGENT_BOT_CELL_STATE",
      "CONCEPT_AGENT_LEFT_CELL_STATE",
      "CONCEPT_AGENT_RIGHT_CELL_STATE",
      "CONCEPT_AGENT_TOP_LEFT_CELL_STATE",
      "CONCEPT_AGENT_TOP_RIGHT_CELL_STATE",
      "CONCEPT_AGENT_BOT_LEFT_CELL_STATE",
      "CONCEPT_AGENT_BOT_RIGHT_CELL_STATE",
  ]

  # The specs of the environment (from a single-agent perspective).
  config.action_spec = specs.action(len(ACTION_SET))
  timestep_spec = {
      "RGB": specs.rgb(56, 56, name="RGB"),
      "READY_TO_SHOOT": specs.OBSERVATION["READY_TO_SHOOT"],
      "POSITION": specs.OBSERVATION["POSITION"],
      "ORIENTATION": specs.OBSERVATION["ORIENTATION"],
      "WORLD.RGB": specs.rgb(184, 184),
      "WORLD.CONCEPT_AGENT_POSITIONS":
          concept_specs.position_concept(
              config.num_players, config.num_players, is_agent=True),
      "WORLD.CONCEPT_AGENT_ORIENTATIONS":
          concept_specs.categorical_concept(
              config.num_players,
              config.num_players,
              num_values=4,
              is_agent=True),
      "WORLD.CONCEPT_AGENT_HEALTH_STATES":
          concept_specs.categorical_concept(
              config.num_players,
              config.num_players,
              num_values=4,
              is_agent=True),
      "WORLD.CONCEPT_FLAG_POSITIONS":
          concept_specs.position_concept(
              config.num_players, config.num_flags, is_agent=True),
      "WORLD.CONCEPT_AGENT_HAS_FLAG":
          concept_specs.binary_concept(
              config.num_players, config.num_players, is_agent=True),
      "WORLD.CONCEPT_FLAG_STATE_INDICATOR":
          concept_specs.categorical_concept(
              config.num_players,
              1,
              num_values=4,
              is_agent=False),
      "WORLD.CONCEPT_AGENT_TOP_CELL_STATE":
          concept_specs.categorical_concept(
              config.num_players,
              1,
              num_values=7,
              is_agent=True),
      "WORLD.CONCEPT_AGENT_BOT_CELL_STATE":
          concept_specs.categorical_concept(
              config.num_players,
              1,
              num_values=7,
              is_agent=True),
      "WORLD.CONCEPT_AGENT_LEFT_CELL_STATE":
          concept_specs.categorical_concept(
              config.num_players,
              1,
              num_values=7,
              is_agent=True),
      "WORLD.CONCEPT_AGENT_RIGHT_CELL_STATE":
          concept_specs.categorical_concept(
              config.num_players,
              1,
              num_values=7,
              is_agent=True),
      "WORLD.CONCEPT_AGENT_TOP_LEFT_CELL_STATE":
          concept_specs.categorical_concept(
              config.num_players,
              1,
              num_values=7,
              is_agent=True),
      "WORLD.CONCEPT_AGENT_TOP_RIGHT_CELL_STATE":
          concept_specs.categorical_concept(
              config.num_players,
              1,
              num_values=7,
              is_agent=True),
      "WORLD.CONCEPT_AGENT_BOT_LEFT_CELL_STATE":
          concept_specs.categorical_concept(
              config.num_players,
              1,
              num_values=7,
              is_agent=True),
      "WORLD.CONCEPT_AGENT_BOT_RIGHT_CELL_STATE":
          concept_specs.categorical_concept(
              config.num_players,
              1,
              num_values=7,
              is_agent=True),
  }
  config.timestep_spec = specs.timestep(timestep_spec)
  concept_spec = create_concept_spec(timestep_spec, concept_prefix)

  return config, concept_spec
