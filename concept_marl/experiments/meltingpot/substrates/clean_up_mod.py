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

"""Configuration for customized clean up environments.


This is an extension of the cooking environments in:
meltingpot/python/configs/substrates/clean_up.py

The structure of all maps, prefabs, configs, etc. in this file (and folder as a
whole) follow the design patterns introduced there.
"""

from typing import Any, Dict, Tuple

from meltingpot.utils.substrates import colors
from meltingpot.utils.substrates import game_object_utils
from meltingpot.utils.substrates import shapes
from meltingpot.utils.substrates import specs
from ml_collections import config_dict

from concept_marl.experiments.meltingpot.substrates import concept_specs

PrefabConfig = game_object_utils.PrefabConfig


clean_up_mod = """
WWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
WHFFFHFFHFHFHFHFHFHFHHFHFFFHFW
WHFHFHFFHFHFHFHFHFHFHHFHFFFHFW
WHFFHFFHHFHFHFHFHFHFHHFHFFFHFW
WHFHFHFFHFHFHFHFHFHFHHFHFFFHFW
WHFFFFFFHFHFHFHFHFHFHHFHFFFHFW
W               HFHHHHHH     W
W   P    P          SSS      W
W     P     P   P   SS   P   W
W             P   PPSS       W
W   P    P          SS    P  W
W               P   SS P     W
W     P           P SS       W
W           P       SS  P    W
W  P             P PSS       W
W B B B B B B B B B SSB B B BW
WBBBBBBBBBBBBBBBBBBBBBBBBBBBBW
WBBBBBBBBBBBBBBBBBBBBBBBBBBBBW
WBBBBBBBBBBBBBBBBBBBBBBBBBBBBW
WBBBBBBBBBBBBBBBBBBBBBBBBBBBBW
WWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
"""

clean_up_mod_mini = """
WWWWWWWWWWWWWWWWWW
WFHFHFHFHFHHHHFFFW
WHHFHFHFHFHHHHFFFW
WFHFHFHFHFHHHHFFFW
W  P    HFHHHH  PW
W          SS    W
W       P  SS  P W
W    P    PSS    W
W P        SS P  W
W        P SS    W
W B B B B BSS B BW
WBBBBBBBBBBBBBBBBW
WBBBBBBBBBBBBBBBBW
WBBBBBBBBBBBBBBBBW
WWWWWWWWWWWWWWWWWW
"""

ASCII_MAPS = {
    "clean_up_mod": clean_up_mod,
    "clean_up_mod_mini": clean_up_mod_mini,
}

# Map a character to the prefab it represents in the ASCII map.
CHAR_PREFAB_MAP = {
    "W": "wall",
    "P": "spawn_point",
    "B": "potential_apple",
    "S": "river",
    "H": {"type": "all", "list": ["river", "potential_dirt"]},
    "F": {"type": "all", "list": ["river", "actual_dirt"]},
}

_COMPASS = ["N", "E", "S", "W"]

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
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["Wall"],
                "spriteShapes": [shapes.WALL],
                "palettes": [{"*": (95, 95, 95, 255),
                              "&": (100, 100, 100, 255),
                              "@": (109, 109, 109, 255),
                              "#": (152, 152, 152, 255)}],
                "noRotates": [False]
            }
        },
        {
            "component": "Transform",
        },
        {
            "component": "BeamBlocker",
            "kwargs": {
                "beamType": "zapHit"
            }
        },
        {
            "component": "BeamBlocker",
            "kwargs": {
                "beamType": "cleanHit"
            }
        },
    ]
}

SPAWN_POINT = {
    "name": "spawnPoint",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "spawnPoint",
                "stateConfigs": [{
                    "state": "spawnPoint",
                    "layer": "logic",
                    "groups": ["spawnPoints"]
                }],
            }
        },
        {
            "component": "Transform",
            "kwargs": {
                "position": (0, 0),
                "orientation": "N"
            }
        },
    ]
}

POTENTIAL_APPLE = {
    "name": "potentialApple",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "appleWait",
                "stateConfigs": [
                    {
                        "state": "apple",
                        "sprite": "Apple",
                        "layer": "lowerPhysical",
                    },
                    {
                        "state": "appleWait"
                    }],
            }
        },
        {
            "component": "Transform",
            "kwargs": {
                "position": (0, 0),
                "orientation": "N"
            }
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["Apple"],
                "spriteShapes": [shapes.LEGACY_APPLE],
                "palettes": [{"*": (102, 255, 0, 255),
                              "@": (230, 255, 0, 255),
                              "&": (117, 255, 26, 255),
                              "#": (255, 153, 0, 255),
                              "x": (0, 0, 0, 0)}],
                "noRotates": [False]
            }
        },
        {
            "component": "Edible",
            "kwargs": {
                "liveState": "apple",
                "waitState": "appleWait",
                "rewardForEating": 0.1,
            }
        },
        {
            "component": "AppleGrow",
            "kwargs": {
                "maxAppleGrowthRate": 0.1,
                "thresholdDepletion": 0.75,
                "thresholdRestoration": 0.0,
            }
        }
    ]
}


def create_dirt_prefab(initial_state):
  """Create a dirt prefab with the given initial state."""
  dirt_prefab = {
      "name": "DirtContainer",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": initial_state,
                  "stateConfigs": [
                      {
                          "state": "dirtWait",
                          "layer": "logic",
                      },
                      {
                          "state": "dirt",
                          "layer": "lowerPhysical",
                          "sprite": "Dirt",
                      },
                  ],
              }
          },
          {
              "component": "Transform",
              "kwargs": {
                  "position": (0, 0),
                  "orientation": "N"
              }
          },
          {
              "component": "Appearance",
              "kwargs": {
                  "spriteNames": ["Dirt"],
                  # This color is greenish, and quite transparent to expose the
                  # animated water below.
                  "spriteRGBColors": [(2, 230, 80, 50)],
              }
          },
          {
              "component": "DirtTracker",
              "kwargs": {
                  "activeState": "dirt",
                  "inactiveState": "dirtWait",
              }
          },
          {
              "component": "DirtCleaning",
              "kwargs": {}
          },
      ]
  }
  return dirt_prefab

# Primitive action components.
# pylint: disable=bad-whitespace
# pyformat: disable
NOOP        = {"move": 0, "turn":  0, "fireZap": 0, "fireClean": 0}
FORWARD     = {"move": 1, "turn":  0, "fireZap": 0, "fireClean": 0}
STEP_RIGHT  = {"move": 2, "turn":  0, "fireZap": 0, "fireClean": 0}
BACKWARD    = {"move": 3, "turn":  0, "fireZap": 0, "fireClean": 0}
STEP_LEFT   = {"move": 4, "turn":  0, "fireZap": 0, "fireClean": 0}
TURN_LEFT   = {"move": 0, "turn": -1, "fireZap": 0, "fireClean": 0}
TURN_RIGHT  = {"move": 0, "turn":  1, "fireZap": 0, "fireClean": 0}
FIRE_ZAP    = {"move": 0, "turn":  0, "fireZap": 1, "fireClean": 0}
FIRE_CLEAN  = {"move": 0, "turn":  0, "fireZap": 0, "fireClean": 1}
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
    FIRE_ZAP,
    FIRE_CLEAN
)

TARGET_SPRITE_SELF = {
    "name": "Self",
    "shape": shapes.CUTE_AVATAR,
    "palette": shapes.get_palette((50, 100, 200)),
    "noRotate": True,
}


def get_water():
  """Get an animated water game object."""
  layer = "background"
  water = {
      "name": "water_{}".format(layer),
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": "water_1",
                  "stateConfigs": [
                      {"state": "water_1",
                       "layer": layer,
                       "sprite": "water_1",
                       "groups": ["water"]},
                      {"state": "water_2",
                       "layer": layer,
                       "sprite": "water_2",
                       "groups": ["water"]},
                      {"state": "water_3",
                       "layer": layer,
                       "sprite": "water_3",
                       "groups": ["water"]},
                      {"state": "water_4",
                       "layer": layer,
                       "sprite": "water_4",
                       "groups": ["water"]},
                  ]
              }
          },
          {"component": "Transform",},
          {
              "component": "Appearance",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "spriteNames": ["water_1", "water_2", "water_3", "water_4"],
                  "spriteShapes": [shapes.WATER_1, shapes.WATER_2,
                                   shapes.WATER_3, shapes.WATER_4],
                  "palettes": [shapes.WATER_PALETTE] * 4,
              }
          },
          {
              "component": "Animation",
              "kwargs": {
                  "states": ["water_1", "water_2", "water_3", "water_4"],
                  "gameFramesPerAnimationFrame": 2,
                  "loop": True,
                  "randomStartFrame": True,
                  "group": "water",
              }
          },
      ]
  }
  return water


def create_prefabs():
  """Returns the prefabs.

  Prefabs are a dictionary mapping names to template game objects that can
  be cloned and placed in multiple locations accoring to an ascii map.
  """
  prefabs = {
      "wall": WALL,
      "spawn_point": SPAWN_POINT,
      "potential_apple": POTENTIAL_APPLE,
      "river": get_water(),
      "potential_dirt": create_dirt_prefab("dirtWait"),
      "actual_dirt": create_dirt_prefab("dirt"),
  }
  return prefabs


def create_scene(num_players, num_closest_objs = 5):
  """Create the scene object, a non-physical object to hold global logic."""
  scene = {
      "name":
          "scene",
      "components": [{
          "component": "StateManager",
          "kwargs": {
              "initialState": "scene",
              "stateConfigs": [{
                  "state": "scene",
              }],
          }
      }, {
          "component": "Transform",
          "kwargs": {
              "position": (0, 0),
              "orientation": "N"
          },
      }, {
          "component": "RiverMonitor",
          "kwargs": {},
      }, {
          "component": "DirtSpawner",
          "kwargs": {
              "dirtSpawnProbability": 0.25,
              "delayStartOfDirtSpawning": 100,
          },
      }, {
          "component": "GlobalStateTracker",
          "kwargs": {
              "numPlayers": num_players,
              "numClosestObjects": num_closest_objs,
          }
      }, {
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
                      "shape": (num_players, num_players,),
                      "component": "GlobalStateTracker",
                      "variable": "playerOrientations",
                  },
                  {
                      "name": "CONCEPT_CLOSEST_APPLE_POSITIONS",
                      "type": "tensor.Int32Tensor",
                      "shape": (num_players, num_closest_objs, 2),
                      "component": "GlobalStateTracker",
                      "variable": "closestApplePositions",
                  },
                  {
                      "name": "CONCEPT_CLOSEST_POLLUTION_POSITIONS",
                      "type": "tensor.Int32Tensor",
                      "shape": (num_players, num_closest_objs, 2),
                      "component": "GlobalStateTracker",
                      "variable": "closestPollutionPositions",
                  },
              ]
          },
      }]
  }
  return scene


def create_avatar_object(player_idx,
                         target_sprite_self):
  """Create an avatar object that always sees itself as blue."""
  # Lua is 1-indexed.
  lua_index = player_idx + 1

  # Setup the self vs other sprite mapping.
  source_sprite_self = "Avatar" + str(lua_index)
  custom_sprite_map = {source_sprite_self: target_sprite_self["name"]}

  live_state_name = "player{}".format(lua_index)
  avatar_object = {
      "name": "avatar",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": live_state_name,
                  "stateConfigs": [
                      # Initial player state.
                      {"state": live_state_name,
                       "layer": "upperPhysical",
                       "sprite": source_sprite_self,
                       "contact": "avatar",
                       "groups": ["players"]},

                      # Player wait type for times when they are zapped out.
                      {"state": "playerWait",
                       "groups": ["playerWaits"]},
                  ]
              }
          },
          {
              "component": "Transform",
              "kwargs": {
                  "position": (0, 0),
                  "orientation": "N"
              }
          },
          {
              "component": "Appearance",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "spriteNames": [source_sprite_self],
                  "spriteShapes": [shapes.CUTE_AVATAR],
                  "palettes": [shapes.get_palette(colors.palette[player_idx])],
                  "noRotates": [True]
              }
          },
          {
              "component": "AdditionalSprites",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "customSpriteNames": [target_sprite_self["name"]],
                  "customSpriteShapes": [target_sprite_self["shape"]],
                  "customPalettes": [target_sprite_self["palette"]],
                  "customNoRotates": [target_sprite_self["noRotate"]],
              }
          },
          {
              "component": "Avatar",
              "kwargs": {
                  "index": lua_index,
                  "aliveState": live_state_name,
                  "waitState": "playerWait",
                  "spawnGroup": "spawnPoints",
                  "actionOrder": ["move",
                                  "turn",
                                  "fireZap",
                                  "fireClean"],
                  "actionSpec": {
                      "move": {"default": 0, "min": 0, "max": len(_COMPASS)},
                      "turn": {"default": 0, "min": -1, "max": 1},
                      "fireZap": {"default": 0, "min": 0, "max": 1},
                      "fireClean": {"default": 0, "min": 0, "max": 1},
                  },
                  "view": {
                      "left": 3,
                      "right": 3,
                      "forward": 5,
                      "backward": 1,
                      "centered": False
                  },
                  "spriteMap": custom_sprite_map,
              }
          },
          {
              "component": "Zapper",
              "kwargs": {
                  "cooldownTime": 10,
                  "beamLength": 3,
                  "beamRadius": 1,
                  "framesTillRespawn": 50,
                  "penaltyForBeingZapped": 0,
                  "rewardForZapping": 0,
                  "removeHitPlayer": True,
              }
          },
          {
              "component": "ReadyToShootObservation",
          },
          {
              "component": "Cleaner",
              "kwargs": {
                  "cooldownTime": 2,
                  "beamLength": 3,
                  "beamRadius": 1,
              }
          },
          {
              "component": "Taste",
              "kwargs": {
                  "role": "cleaner_consumer",
                  "cleanRewardAmount": 0.005,
                  "eatRewardAmount": 0.1,
              }
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


def create_avatar_objects(num_players):
  """Returns list of avatar objects of length 'num_players'."""
  avatar_objects = []
  for player_idx in range(0, num_players):
    game_object = create_avatar_object(player_idx,
                                       TARGET_SPRITE_SELF)
    avatar_objects.append(game_object)

  return avatar_objects


def create_lab2d_settings(ascii_map,
                          num_players,
                          num_closest_objs = 5):
  """Returns the lab2d settings."""
  ascii_map = ASCII_MAPS[ascii_map]

  lab2d_settings = {
      "levelName": "clean_up",
      "levelDirectory":
          "../experiments/meltingpot/lua/levels",
      "numPlayers": num_players,
      "maxEpisodeLengthFrames": 1000,
      "spriteSize": 8,
      "topology": "BOUNDED",  # Choose from ["BOUNDED", "TORUS"],
      "simulation": {
          "map": ascii_map,
          "gameObjects": create_avatar_objects(num_players),
          "prefabs": create_prefabs(),
          "charPrefabMap": CHAR_PREFAB_MAP,
          "scene": create_scene(num_players, num_closest_objs),
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
  """Default configuration for training on the clean_up level."""
  config = config_dict.ConfigDict()

  # Basic configuration.
  if ascii_map == "clean_up_mod":
    config.num_players = 7
  else:
    config.num_players = 4
  config.num_closest_objs = 5

  # Lua script configuration.
  config.lab2d_settings = create_lab2d_settings(ascii_map,
                                                config.num_players,
                                                config.num_closest_objs)

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
      "WORLD.CONCEPT_CLOSEST_APPLE_POSITIONS",
      "WORLD.CONCEPT_CLOSEST_POLLUTION_POSITIONS"
  ]

  # The specs of the environment (from a single-agent perspective).
  config.action_spec = specs.action(len(ACTION_SET))

  timestep_spec = {
      "RGB":
          specs.rgb(56, 56, name="RGB"),
      "READY_TO_SHOOT":
          specs.OBSERVATION["READY_TO_SHOOT"],
      "POSITION":
          specs.OBSERVATION["POSITION"],
      "ORIENTATION":
          specs.OBSERVATION["ORIENTATION"],
      "WORLD.RGB":
          specs.rgb(168, 240),
      "WORLD.CONCEPT_AGENT_POSITIONS":
          concept_specs.position_concept(
              config.num_players, config.num_players, is_agent=True),
      "WORLD.CONCEPT_AGENT_ORIENTATIONS":
          concept_specs.categorical_concept(
              config.num_players,
              config.num_players,
              num_values=4,
              is_agent=True),
      "WORLD.CONCEPT_CLOSEST_APPLE_POSITIONS":
          concept_specs.position_concept(
              config.num_players, config.num_closest_objs, is_agent=False),
      "WORLD.CONCEPT_CLOSEST_POLLUTION_POSITIONS":
          concept_specs.position_concept(
              config.num_players, config.num_closest_objs, is_agent=False),
  }
  config.timestep_spec = specs.timestep(timestep_spec)
  concept_spec = create_concept_spec(timestep_spec, concept_prefix)

  return config, concept_spec
