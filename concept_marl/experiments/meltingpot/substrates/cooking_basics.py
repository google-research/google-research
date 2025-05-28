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

"""Configuration for customized cooking environments.

This is an extension of the cooking environments in:
meltingpot/python/configs/substrates/collaborative_cooking.py

The structure of all maps, prefabs, configs, etc. in this file (and folder as a
whole) follow the design patterns introduced there.
"""

import copy
from typing import Any, Dict, List, Tuple

from meltingpot.utils.substrates import colors
from meltingpot.utils.substrates import game_object_utils
from meltingpot.utils.substrates import shapes
from meltingpot.utils.substrates import specs
from ml_collections import config_dict

from concept_marl.experiments.meltingpot.substrates import concept_specs

COOKING_TIME = 20
items = ["empty", "tomato", "dish", "soup"]

# Map a character to the prefab it represents in the ASCII map.
CHAR_PREFAB_MAP = {
    "P": "spawn_point",
    "#": "counter",
    "O": "tomato_dispenser",
    "D": "dish_dispenser",
    "T": "delivery_location",
    "C": "cooking_pot",
    "N": "nav_location",
}

##############
# ASCII maps #
##############
nav = """
####N##
O P   #
#     #
#     #
#   P O
##N####
"""

asym_mini = """
##T#T##
#     O
C #P# #
# #P# #
C     #
#D###DO
"""

asym = """
###T#T###
#       #
# ## ## O
C ##P## #
#       #
# ##P## #
# ## ## #
C       #
##D###D#O
"""

basic_mini = """
##T#T##
#D # D#
#     #
C P P C
#     #
##O#O##
"""

basic = """
##T###T##
#D  #  D#
#       #
#       #
#       #
C P   P C
#       #
#       #
#       #
##O###O##
"""

impassable_mini = """
##T#T##
D   P D
#     #
#######
# P   #
C     C
##O#O##
"""

impassable = """
##T###T##
## D#D ##
#       #
#    P  C
#       #
#########
#       #
C P     #
#       #
##O###O##
"""

passable_mini = """
#DD#OTT
C  #  O
#P # P#
C  #  #
#     #
#######
"""

passable = """
#D#D###O#O###
#    #     ##
#    #   P  T
C P  #     ##
#    #      T
C    #     ##
#    #     ##
#          ##
#############
"""

ASCII_MAPS = {
    "cooking_nav": nav,
    "cooking_asym_mini": asym_mini,
    "cooking_asym": asym,
    "cooking_basic_mini": basic_mini,
    "cooking_basic": basic,
    "cooking_impassable_mini": impassable_mini,
    "cooking_impassable": impassable,
    "cooking_passable_mini": passable_mini,
    "cooking_passable": passable,
}

###########
# SPRITES #
###########

TARGET_SPRITE_SELF = {
    "name": "Self",
    "shape": shapes.CUTE_AVATAR,
    "palette": shapes.get_palette((50, 100, 200)),
    "noRotate": True,
}

BACKGROUND_LIGHT = (255, 255, 255, 255)
BACKGROUND_DARK = (82, 82, 82, 255)
OUTLINE = (85, 58, 23, 255)
OUTLINE_DARK = (49, 49, 49, 255)
INVISIBLE = (0, 0, 0, 0)
COUNTER = (115, 81, 39, 255)

SPRITES = {}
PALETTES = {}

SPRITES["interact"] = """
PPPPPPPPPPPPPPPP
PPPPPPPPPPPPPPPP
PP~~~~~~~~~~~~PP
PP~~~~~~~~~~~~PP
PP~~~~~~~~~~~~PP
PP~~~~~~~~~~~~PP
PP~~~~~~~~~~~~PP
PP~~~~~~~~~~~~PP
PP~~~~~~~~~~~~PP
PP~~~~~~~~~~~~PP
PP~~~~~~~~~~~~PP
PP~~~~~~~~~~~~PP
PP~~~~~~~~~~~~PP
PP~~~~~~~~~~~~PP
PPPPPPPPPPPPPPPP
PPPPPPPPPPPPPPPP
"""

SPRITES["empty"] = """
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
"""
PALETTES["empty"] = {"~": INVISIBLE, "@": BACKGROUND_LIGHT}

SPRITES["counter"] = """
&&&&&&&&&&&&&&&&
&**************&
&**************&
&**************&
&**************&
&**************&
&**************&
&**************&
&**************&
&**************&
&**************&
&**************&
&**************&
&**************&
&**************&
&&&&&&&&&&&&&&&&
"""
PALETTES["counter"] = {"*": COUNTER, "&": OUTLINE}

SPRITES["delivery_location"] = SPRITES["counter"]
PALETTES["delivery_location"] = {"*": BACKGROUND_DARK, "&": OUTLINE_DARK}

SPRITES["nav_location"] = SPRITES["counter"]
PALETTES["nav_location"] = {"*": BACKGROUND_DARK, "&": OUTLINE_DARK}

SPRITES["dish"] = """
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~++++~~~~~~
~~~~~+^^^^+~~~~~
~~~~~+^^^^+~~~~~
~~~~~+^^^^+~~~~~
~~~~~&++++&~~~~~
~~~~~~&&&&~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
"""
PALETTES["dish"] = {
    "~": [0, 0, 0, 0],
    "+": [255, 255, 255, 255],
    "^": [233, 239, 248, 255],
    "&": [221, 222, 238, 255]
}

SPRITES["soup"] = SPRITES["dish"]
PALETTES["soup"] = {
    "~": [0, 0, 0, 0],
    "+": [255, 255, 255, 255],
    "^": [236, 58, 74, 255],
    "&": [221, 222, 238, 255]
}

SPRITES["dish_dispenser"] = """
&&&&&&&&&&&&&&&&
&~~~~~~~~~~~~~~&
&~~~~~~~~~~~~~~&
&~~~~~~~~~~~~~~&
&~~~~~~~~~~~~~~&
&~~~~~++++~~~~~&
&~~~~+^^^^+~~~~&
&~~~~+^^^^+~~~~&
&~~~~+^^^^+~~~~&
&~~~~X++++X~~~~&
&~~~~~XXXX~~~~~&
&~~~~~~~~~~~~~~&
&~~~~~~~~~~~~~~&
&~~~~~~~~~~~~~~&
&~~~~~~~~~~~~~~&
&&&&&&&&&&&&&&&&
"""

PALETTES["dish_dispenser"] = {
    "&": OUTLINE_DARK,
    "~": BACKGROUND_DARK,
    "+": [255, 255, 255, 255],
    "^": [233, 239, 248, 255],
    "X": [221, 222, 238, 255]
}

SPRITES["tomato"] = """
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~++^+~~~~~~
~~~~~&O^---~~~~~
~~~~~O-----~~~~~
~~~~~O&-@--~~~~~
~~~~~~OO&&~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
"""
PALETTES["tomato"] = {
    "~": [0, 0, 0, 0],
    "+": [239, 81, 90, 255],
    "^": [29, 139, 43, 255],
    "&": [190, 53, 62, 255],
    "O": [151, 47, 52, 255],
    "-": [236, 58, 74, 255],
    "@": [240, 57, 75, 255]
}

SPRITES["tomato_dispenser"] = """
&&&&&&&&&&&&&&&&
&,,,,,,,,,,,,,,&
&,,,,,,,,,,,,,,&
&,,,,,,,,,,,,,,&
&,,,,,,,,,,,,,,&
&,,,,,,,,,,,,,,&
&,,,,,++^+,,,,,&
&,,,,XO^---,,,,&
&,,,,O-----,,,,&
&,,,,O&-@--,,,,&
&,,,,,OOXX,,,,,&
&,,,,,,,,,,,,,,&
&,,,,,,,,,,,,,,&
&,,,,,,,,,,,,,,&
&,,,,,,,,,,,,,,&
&&&&&&&&&&&&&&&&
"""

PALETTES["tomato_dispenser"] = {
    ",": BACKGROUND_DARK,
    "&": OUTLINE_DARK,
    "~": [0, 0, 0, 0],
    "+": [239, 81, 90, 255],
    "^": [29, 139, 43, 255],
    "X": [190, 53, 62, 255],
    "O": [151, 47, 52, 255],
    "-": [236, 58, 74, 255],
    "@": [240, 57, 75, 255]
}

SPRITES["cooking_pot_empty"] = """
&&&&&&&&&&&&&&&&
&~~~++++++++~~~&
&~~+^^^^^^^XO~~&
&~~+^^^^^^XXO~~&
&^^+^^^^^XXXO--&
&^~+^^^^XXXXO~-&
&^~+@@@@AAAAO~-&
&^^+@@@@AAAAO--&
&~~+@@@@AAAAO~~&
&~~@OOOOOOOO-~~&
&~~@MMMMMMMM-~~&
&~~@MMMMMMMM-~~&
&~~@MMMMMMMM-~~&
&~~@MMMMMMMM-~~&
&~~~MMMMMMMM~~~&
&&&&&&&&&&&&&&&&
"""

SPRITES["cooking_pot_1"] = """
&&&&&&&&&&&&&&&&
&~~~++++++++~~~&
&~~+^^^^^^^XO~~&
&~~+^^^^^^XXO~~&
&^^+^^^^^XXXO--&
&^~+KKKKLLLLO~-&
&^~+KKKKLLLLO~-&
&^^+KKKKLLLLO--&
&~~+KKKKLLLLO~~&
&~~@OOOOOOOO-~~&
&~~@MMMMMMMM-~~&
&~~@MMMMMMMM-~~&
&~~@MMMMMMMM-~~&
&~~@MMMMMMMM-~~&
&~~~MMMMMMMM~~~&
&&&&&&&&&&&&&&&&
"""

SPRITES["cooking_pot_2"] = """
&&&&&&&&&&&&&&&&
&~~~++++++++~~~&
&~~+^^^^^^^XO~~&
&~~+^^^^^^XXO~~&
&^^+KKKKKKLLO--&
&^~+KKKKLKLLO~-&
&^~+KLKKKKLLO~-&
&^^+KKKKKKLLO--&
&~~+KKKKKKLLO~~&
&~~@OOOOOOOO-~~&
&~~@MMMMMMMM-~~&
&~~@MMMMMMMM-~~&
&~~@MMMMMMMM-~~&
&~~@MMMMMMMM-~~&
&~~~MMMMMMMM~~~&
&&&&&&&&&&&&&&&&
"""

SPRITES["cooking_pot_3"] = """
&&&&&&&&&&&&&&&&
&~~~++++++++~~~&
&~~+KKKKKKKKO~~&
&~~+KNKKKKKKO~~&
&^^+KKKKKKKKO--&
&^~+KKKKKKKKO~-&
&^~+KKKKKKKKO~-&
&^^+KKNKKKNKO--&
&~~+KKKKKKKKO~~&
&~~@OOOOOOOO-~~&
&~~@MMMMMMMM-~~&
&~~@MMMMMMMM-~~&
&~~@MMMMMMMM-~~&
&~~@MMMMMMMM-~~&
&~~~MMMMMMMM~~~&
&&&&&&&&&&&&&&&&
"""

PALETTES["cooking_pot"] = {
    "&": OUTLINE,
    "~": COUNTER,
    "+": [224, 231, 240, 255],
    "^": [140, 155, 181, 255],
    "X": [98, 95, 128, 255],
    "O": [238, 241, 241, 255],
    "-": [194, 206, 222, 255],
    "@": [92, 106, 135, 255],
    "A": [65, 66, 97, 255],
    "M": [139, 155, 181, 255],
    "K": [236, 58, 74, 255],
    "L": [161, 43, 43, 255],
    "N": [242, 226, 187, 255]
}

SPRITES["loading_bar"] = """
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~abcdefghij~~~
~~~abcdefghij~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
"""


def create_loading_bar_palette(count,
                               finished = False):
  """Creates an incrementally colored loading bar based on count."""
  characters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
  bars_palette = {"~": INVISIBLE}
  for idx in range(0, 10):
    if idx < count:
      if finished:
        bars_palette[characters[idx]] = (15, 188, 15, 255)
      else:
        bars_palette[characters[idx]] = (201, 178, 50, 255)
    else:
      bars_palette[characters[idx]] = (255, 255, 255, 255)

  return bars_palette


OFFSET_SIZE = 3
for s in ["empty", "tomato", "dish", "soup"]:
  SPRITES[s + "_offset"] = ("~~~~~~~~~~~~~~~~\n" * OFFSET_SIZE +
                            SPRITES[s][1:-OFFSET_SIZE * 17])

###########
# PREFABS #
###########

SPAWN_POINT = {
    "name":
        "spawnPoint",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "spawnPoint",
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

# pylint: disable=g-complex-comprehension
INVENTORY = {
    "name":
        "inventory",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "wait",
                "stateConfigs": [{
                    "state": "wait"
                }] + [{
                    "state": item,
                    "sprite": item,
                    "layer": "overlay"
                } for item in items] + [{
                    "state": item + "_offset",
                    "sprite": item + "_offset",
                    "layer": "overlay"
                } for item in items]
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
                "spriteNames": items + [item + "_offset" for item in items],
                "spriteShapes": ([SPRITES[item] for item in items] +
                                 [SPRITES[item + "_offset"] for item in items]),
                "palettes": ([PALETTES[item] for item in items] +
                             [PALETTES[item] for item in items]),
                "noRotates": [False]
            }
        },
        {
            "component": "Inventory",
            "kwargs": {
                "playerIndex": -1,  # Can be overwritten.
                "emptyState": "empty",
                "waitState": "wait"
            }
        },
    ]
}

loading_palettes = ([create_loading_bar_palette(idx) for idx in range(0, 10)] +
                    [create_loading_bar_palette(10, finished=True)])
LOADING_BAR = {
    "name":
        "loading_bar",
    "components": [{
        "component": "StateManager",
        "kwargs": {
            "initialState":
                "loading_bar_0",
            "stateConfigs": [{
                "state": "loading_bar_%d" % d,
                "layer": "overlay",
                "sprite": "loading_bar_%d" % d
            } for d in range(0, 11)],
        }
    }, {
        "component": "Transform",
        "kwargs": {
            "position": (0, 0),
            "orientation": "N"
        }
    }, {
        "component": "Appearance",
        "kwargs": {
            "renderMode": "ascii_shape",
            "spriteNames": ["loading_bar_%d" % d for d in range(0, 11)],
            "spriteShapes": [SPRITES["loading_bar"] for _ in range(0, 11)],
            "palettes": loading_palettes,
            "noRotates": [True]
        }
    }, {
        "component": "LoadingBarVisualiser",
        "kwargs": {
            "totalTime": COOKING_TIME,
            "customStateNames": ["loading_bar_%d" % d for d in range(0, 11)],
        }
    }]
}
# pylint: enable=g-complex-comprehension


def create_base_prefab(name,
                       layer = "upperPhysical"):
  """Returns a base prefab with a given name on the given layer."""
  return {
      "name":
          f"{name}",
      "components": [{
          "component": "StateManager",
          "kwargs": {
              "initialState":
                  f"{name}",
              "stateConfigs": [{
                  "state": f"{name}",
                  "layer": layer,
                  "sprite": f"{name}",
              }],
          }
      }, {
          "component": "Transform",
          "kwargs": {
              "position": (0, 0),
              "orientation": "N"
          }
      }, {
          "component": "Appearance",
          "kwargs": {
              "renderMode": "ascii_shape",
              "spriteNames": [name],
              "spriteShapes": [SPRITES[name]],
              "palettes": [PALETTES[name]],
              "noRotates": [True]
          }
      }]
  }


def create_counter():
  """Returns a prefab which can contain one of any item."""
  base_prefab = create_base_prefab("counter")
  base_prefab["components"] += [{
      "component": "Container",
      "kwargs": {
          "reward": 0.0
      }
  }]
  return base_prefab


def create_dispenser(prefab_name, item_name):
  """Returns a prefab which dispenses items to avatars upon interaction."""
  base_prefab = create_base_prefab(prefab_name)
  base_prefab["components"] += [{
      "component": "Container",
      "kwargs": {
          "startingItem": item_name,
          "infinite": True,
          "reward": 0.0
      }
  }]
  return base_prefab


def create_receiver(prefab_name,
                    item_name,
                    reward = 0,
                    global_reward = False):
  """Returns a prefab which can receive items from avatars.

  Args:
    prefab_name: the name of the prefab.
    item_name: the name of the accepted item.
    reward: value of reward given to avatar when object receives the item.
    global_reward: if true, reward all avatars.

  Returns:
    A prefab which can receive items.
  """
  base_prefab = create_base_prefab(prefab_name)
  base_prefab["components"] += [{
      "component": "Receiver",
      "kwargs": {
          "acceptedItems": item_name,
          "reward": reward,
          "globalReward": global_reward
      }
  }]
  return base_prefab


def create_cooking_pot(time_to_cook, reward = 1):
  """Creates a cooking pot for tomatoes."""

  state_configs = []
  sprite_names = []
  pot_sprites = []
  cooking_pot_palettes = []
  custom_state_names = []
  available_foods = ["tomato"]

  # Create state for each combination of available foods.
  foods_in_pot = ["empty"] + available_foods
  for food1 in foods_in_pot:
    for food2 in foods_in_pot:
      for food3 in foods_in_pot:
        sprite_name = "CookingPot_%s_%s_%s" % (food1, food2, food3)
        name = "cooking_pot_%s_%s_%s" % (food1, food2, food3)
        entry = {
            "state": name,
            "layer": "upperPhysical",
            "sprite": sprite_name,
            "groups": ["cooking_pot"]
        }
        state_configs.append(entry)
        sprite_names.append(sprite_name)
        pots_palette = PALETTES["cooking_pot"]
        cooking_pot_palettes.append(pots_palette)
        custom_state_names.append(name)

        if food1 == "empty":
          pot_sprites.append(SPRITES["cooking_pot_empty"])
        elif food2 == "empty":
          pot_sprites.append(SPRITES["cooking_pot_1"])
        elif food3 == "empty":
          pot_sprites.append(SPRITES["cooking_pot_2"])
        else:
          pot_sprites.append(SPRITES["cooking_pot_3"])

  # Create cooked state.
  sprite_name = "CookingPot_cooked"
  name = "cooking_pot_cooked"
  entry = {
      "state": "cooking_pot_cooked",
      "layer": "upperPhysical",
      "sprite": sprite_name,
      "groups": ["cooking_pot"]
  }
  state_configs.append(entry)
  sprite_names.append(sprite_name)
  pot_sprites.append(SPRITES["cooking_pot_3"])
  pots_palette = PALETTES["cooking_pot"]
  cooking_pot_palettes.append(pots_palette)
  custom_state_names.append(name)

  cooking_pot = {
      "name":
          "cooking_pot",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": ("cooking_pot_empty_empty_empty"),
                  "stateConfigs": state_configs
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
                  "spriteNames": sprite_names,
                  "spriteShapes": pot_sprites,
                  "palettes": cooking_pot_palettes,
                  "noRotates": [True for _ in sprite_names]
              }
          },
          {
              "component": "CookingPot",
              "kwargs": {
                  "acceptedItems": available_foods,
                  "cookingTime": time_to_cook,
                  "reward": reward,
                  "customStateNames": custom_state_names
              }
          },
      ]
  }

  return cooking_pot


# PREFABS is a dictionary mapping names to template game objects that can
# be cloned and placed in multiple locations according to an ascii map.
PREFABS = {
    "spawn_point":
        SPAWN_POINT,
    "inventory":
        INVENTORY,
    "loading_bar":
        LOADING_BAR,
    "counter":
        create_counter(),
    "dish_dispenser":
        create_dispenser(prefab_name="dish_dispenser", item_name="dish"),
    "tomato_dispenser":
        create_dispenser(prefab_name="tomato_dispenser", item_name="tomato"),
    "delivery_location":
        create_receiver(
            prefab_name="delivery_location",
            item_name="soup",
            reward=2,  # sparse reward for completing cooking task
            global_reward=True),
    "cooking_pot":
        create_cooking_pot(time_to_cook=COOKING_TIME, reward=0.0),
    "nav_location":
        create_receiver(
            prefab_name="nav_location",
            item_name="tomato",
            reward=0.1,  # sparse reward completing tomato-picking environment.
            global_reward=False),
}

###########
# ACTIONS #
###########

# Primitive action components.
# pylint: disable=bad-whitespace
# pyformat: disable
NOOP       = {"move": 0, "turn":  0, "interact": 0}
FORWARD    = {"move": 1, "turn":  0, "interact": 0}
STEP_RIGHT = {"move": 2, "turn":  0, "interact": 0}
BACKWARD   = {"move": 3, "turn":  0, "interact": 0}
STEP_LEFT  = {"move": 4, "turn":  0, "interact": 0}
TURN_LEFT  = {"move": 0, "turn": -1, "interact": 0}
TURN_RIGHT = {"move": 0, "turn":  1, "interact": 0}
INTERACT   = {"move": 0, "turn":  0, "interact": 1}
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
    INTERACT,
)

###########
# OBJECTS #
###########


def create_game_objects(ascii_map_string):
  """Returns list of game objects from 'ascii_map' and 'char_prefab' mapping."""

  # Create all game objects.
  game_objects = []
  for char, _ in CHAR_PREFAB_MAP.items():
    transforms = game_object_utils.get_game_object_positions_from_map(
        ascii_map_string, char)
    for transform in transforms:
      # Add inventory game object for holding and visualising items.
      if char == "#" or char == "O" or char == "D":
        inventory_object = copy.deepcopy(PREFABS["inventory"])
        go_transform = game_object_utils.get_first_named_component(
            inventory_object, "Transform")
        go_transform["kwargs"]["position"] = (transform.position.x,
                                              transform.position.y)
        game_objects.append(inventory_object)

      # Add loading bar object to cooking pots.
      if char == "C":
        loading_object = copy.deepcopy(PREFABS["loading_bar"])
        go_transform = game_object_utils.get_first_named_component(
            loading_object, "Transform")
        go_transform["kwargs"]["position"] = (transform.position.x,
                                              transform.position.y)
        game_objects.append(loading_object)

  return game_objects


def create_avatar_object(player_idx,
                         target_sprite_self):
  """Create an avatar object that always sees itself as blue."""
  # Lua is 1-indexed.
  lua_index = player_idx + 1

  # Setup the self vs other sprite mapping.
  source_sprite_self = "Avatar" + str(lua_index)
  custom_sprite_map = {source_sprite_self: target_sprite_self["name"]}

  interact_palette = {
      "P": colors.palette[player_idx] + (255,),
      "~": INVISIBLE,
  }

  live_state_name = "player{}".format(lua_index)
  avatar_object = {
      "name":
          "avatar",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState":
                      live_state_name,
                  "stateConfigs": [
                      {
                          "state": live_state_name,
                          "layer": "upperPhysical",
                          "sprite": source_sprite_self,
                          "contact": "avatar",
                          "groups": ["players"]
                      },
                      {
                          "state": "playerWait",
                          "groups": ["playerWaits"]
                      },
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
                  "spawnGroup": "spawnPoints",
                  "aliveState": live_state_name,
                  "waitState": "playerWait",
                  "speed": 1.0,
                  "actionOrder": ["move", "turn", "interact"],
                  "actionSpec": {
                      "move": {
                          "default": 0,
                          "min": 0,
                          "max": 4
                      },
                      "turn": {
                          "default": 0,
                          "min": -1,
                          "max": 1
                      },
                      "interact": {
                          "default": 0,
                          "min": 0,
                          "max": 1
                      },
                  },
                  "view": {
                      "left": 2,
                      "right": 2,
                      "forward": 3,
                      "backward": 1,
                      "centered": False
                  },
                  "spriteMap": custom_sprite_map,
              }
          },
          {
              "component": "InteractBeam",
              "kwargs": {
                  "cooldownTime": 1,
                  "shapes": [SPRITES["interact"]],
                  "palettes": [interact_palette]
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
  game_objects = []
  for player_idx in range(0, num_players):
    lua_index = player_idx + 1
    game_object = create_avatar_object(player_idx, TARGET_SPRITE_SELF)
    game_objects.append(game_object)

    # Add inventory game object which will be connected to player at init.
    inventory_object = copy.deepcopy(PREFABS["inventory"])
    game_object_utils.get_first_named_component(
        inventory_object, "Inventory")["kwargs"]["playerIndex"] = lua_index
    game_objects.append(inventory_object)

  return game_objects


def create_scene(num_players,
                 num_tomatoes,
                 num_dishes,
                 num_cooking_pots,
                 num_delivery_locations,
                 num_navigation_locations,
                 num_loose_objs,
                 num_counters):
  """Creates the global scene."""
  scene = {
      "name":
          "scene",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": "scene",
                  "stateConfigs": [{
                      "state": "scene",
                  }],
              }
          },
          {
              "component": "Transform",
          },
          {
              "component": "GlobalStateTracker",
              "kwargs": {
                  "numPlayers": num_players,
                  "numTomatoes": num_tomatoes,
                  "numDishes": num_dishes,
                  "numCookingPots": num_cooking_pots,
                  "numDeliveryLocations": num_delivery_locations,
                  "numNavigationLocations": num_navigation_locations,
                  "numLooseObjects": num_loose_objs,
                  "numCounters": num_counters
              }
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
                          "shape": (num_players, num_players,),
                          "component": "GlobalStateTracker",
                          "variable": "playerOrientations",
                      },
                      {
                          "name": "CONCEPT_TOMATO_POSITIONS",
                          "type": "tensor.Int32Tensor",
                          "shape": (num_players, num_tomatoes, 2),
                          "component": "GlobalStateTracker",
                          "variable": "tomatoPositions",
                      },
                      {
                          "name": "CONCEPT_DISH_POSITIONS",
                          "type": "tensor.Int32Tensor",
                          "shape": (num_players, num_dishes, 2),
                          "component": "GlobalStateTracker",
                          "variable": "dishPositions",
                      },
                      {
                          "name": "CONCEPT_COOKING_POT_POSITIONS",
                          "type": "tensor.Int32Tensor",
                          "shape": (num_players, num_cooking_pots, 2),
                          "component": "GlobalStateTracker",
                          "variable": "potPositions",
                      },
                      {
                          "name": "CONCEPT_DELIVERY_POSITIONS",
                          "type": "tensor.Int32Tensor",
                          "shape": (num_players, num_delivery_locations, 2),
                          "component": "GlobalStateTracker",
                          "variable": "deliveryPositions",
                      },
                      {
                          "name": "CONCEPT_NAVIGATION_POSITIONS",
                          "type": "tensor.Int32Tensor",
                          "shape": (num_navigation_locations, 2),
                          "component": "GlobalStateTracker",
                          "variable": "navigationPositions",
                      },
                      {
                          "name": "CONCEPT_AGENT_HAS_TOMATO",
                          "type": "tensor.Int32Tensor",
                          "shape": (num_players, num_players,),
                          "component": "GlobalStateTracker",
                          "variable": "playerHasTomatos",
                      },
                      {
                          "name": "CONCEPT_AGENT_HAS_DISH",
                          "type": "tensor.Int32Tensor",
                          "shape": (num_players, num_players,),
                          "component": "GlobalStateTracker",
                          "variable": "playerHasDishes",
                      },
                      {
                          "name": "CONCEPT_AGENT_HAS_SOUP",
                          "type": "tensor.Int32Tensor",
                          "shape": (num_players, num_players,),
                          "component": "GlobalStateTracker",
                          "variable": "playerHasSoups",
                      },
                      {
                          "name": "CONCEPT_COOKING_POT_TOMATO_COUNTS",
                          "type": "tensor.Int32Tensor",
                          "shape": (num_players, num_cooking_pots,),
                          "component": "GlobalStateTracker",
                          "variable": "potTomatoCounts",
                      },
                      {
                          "name": "CONCEPT_COOKING_POT_PROGRESS",
                          "type": "tensor.DoubleTensor",
                          "shape": (num_players, num_cooking_pots,),
                          "component": "GlobalStateTracker",
                          "variable": "potCookingProgress",
                      },
                      {
                          "name": "CONCEPT_LOOSE_TOMATO_POSITIONS",
                          "type": "tensor.Int32Tensor",
                          "shape": (num_players, num_loose_objs, 2),
                          "component": "GlobalStateTracker",
                          "variable": "looseTomatoPositions",
                      },
                      {
                          "name": "CONCEPT_LOOSE_DISH_POSITIONS",
                          "type": "tensor.Int32Tensor",
                          "shape": (num_players, num_loose_objs, 2),
                          "component": "GlobalStateTracker",
                          "variable": "looseDishPositions",
                      },
                      {
                          "name": "CONCEPT_LOOSE_SOUP_POSITIONS",
                          "type": "tensor.Int32Tensor",
                          "shape": (num_players, num_loose_objs, 2),
                          "component": "GlobalStateTracker",
                          "variable": "looseSoupPositions",
                      },
                      {
                          "name": "CONCEPT_COUNTER_TOMATOES",
                          "type": "tensor.Int32Tensor",
                          "shape": (num_players, num_counters,),
                          "component": "GlobalStateTracker",
                          "variable": "counterTomatoes",
                      },
                      {
                          "name": "CONCEPT_COUNTER_DISHES",
                          "type": "tensor.Int32Tensor",
                          "shape": (num_players, num_counters,),
                          "component": "GlobalStateTracker",
                          "variable": "counterDishes",
                      },
                      {
                          "name": "CONCEPT_COUNTER_SOUPS",
                          "type": "tensor.Int32Tensor",
                          "shape": (num_players, num_counters,),
                          "component": "GlobalStateTracker",
                          "variable": "counterSoups",
                      },
                  ]
              }
          }
      ]
  }

  return scene


def create_lab2d_settings(ascii_map,
                          config):
  """Returns the lab2d settings."""
  ascii_map = ASCII_MAPS[ascii_map]
  game_objects = create_game_objects(ascii_map)
  extra_game_objects = create_avatar_objects(config.num_players)
  game_objects += extra_game_objects

  # Lua script configuration.
  lab2d_settings = {
      "levelName":
          "cooking_basics",
      "levelDirectory":
          "../experiments/meltingpot/lua/levels",
      "numPlayers":
          config.num_players,
      "maxEpisodeLengthFrames":
          1000,
      "spriteSize":
          8,
      "simulation": {
          "map":
              ascii_map,
          "gameObjects":
              game_objects,
          "prefabs":
              PREFABS,
          "charPrefabMap":
              CHAR_PREFAB_MAP,
          "scene":
              create_scene(config.num_players, config.num_tomatoes,
                           config.num_dishes, config.num_cooking_pots,
                           config.num_delivery_locations,
                           config.num_navigation_locations,
                           config.num_loose_objs, config.num_counters),
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
  """Default configuration for training on the collaborative cooking level."""
  config = config_dict.ConfigDict()

  # The specs of the environment (from a single-agent perspective).
  config.action_spec = specs.action(len(ACTION_SET))
  if ascii_map == "cooking_passable":
    world_size = 72, 104
    config.num_players = 2
    config.num_tomatoes = 2
    config.num_dishes = 2
    config.num_cooking_pots = 2
    config.num_delivery_locations = 2
    config.num_navigation_locations = 0
    config.num_loose_objs = 3
    config.num_counters = 43
  elif ascii_map == "cooking_impassable":
    world_size = 80, 72
    config.num_players = 2
    config.num_tomatoes = 2
    config.num_dishes = 2
    config.num_cooking_pots = 2
    config.num_delivery_locations = 2
    config.num_navigation_locations = 0
    config.num_loose_objs = 3
    config.num_counters = 38
  elif ascii_map == "cooking_basic":
    world_size = 80, 72
    config.num_players = 2
    config.num_tomatoes = 2
    config.num_dishes = 2
    config.num_cooking_pots = 2
    config.num_delivery_locations = 2
    config.num_navigation_locations = 0
    config.num_loose_objs = 3
    config.num_counters = 29
  elif ascii_map == "cooking_basic_mini":
    world_size = 48, 56
    config.num_players = 2
    config.num_tomatoes = 2
    config.num_dishes = 2
    config.num_cooking_pots = 2
    config.num_delivery_locations = 2
    config.num_navigation_locations = 0
    config.num_loose_objs = 3
    config.num_counters = 17
  elif ascii_map == "cooking_asym":
    world_size = 72, 72
    config.num_players = 2
    config.num_tomatoes = 2
    config.num_dishes = 2
    config.num_cooking_pots = 2
    config.num_delivery_locations = 2
    config.num_navigation_locations = 0
    config.num_loose_objs = 3
    config.num_counters = 40
  elif ascii_map == "cooking_asym_mini":
    world_size = 48, 56
    config.num_players = 2
    config.num_tomatoes = 2
    config.num_dishes = 2
    config.num_cooking_pots = 2
    config.num_delivery_locations = 2
    config.num_navigation_locations = 0
    config.num_loose_objs = 3
    config.num_counters = 18
  elif ascii_map == "cooking_impassable_mini":
    world_size = 56, 56
    config.num_players = 2
    config.num_tomatoes = 2
    config.num_dishes = 2
    config.num_cooking_pots = 2
    config.num_delivery_locations = 2
    config.num_navigation_locations = 0
    config.num_loose_objs = 3
    config.num_counters = 21
  elif ascii_map == "cooking_passable_mini":
    world_size = 48, 56
    config.num_players = 2
    config.num_tomatoes = 2
    config.num_dishes = 2
    config.num_cooking_pots = 2
    config.num_delivery_locations = 2
    config.num_navigation_locations = 0
    config.num_loose_objs = 3
    config.num_counters = 17
  elif ascii_map == "cooking_nav":
    world_size = 40, 56
    config.num_players = 2
    config.num_tomatoes = 2
    config.num_dishes = 0
    config.num_cooking_pots = 0
    config.num_delivery_locations = 0
    config.num_navigation_locations = 2
    config.num_loose_objs = 3
    config.num_counters = 18

  # Lua script configuration.
  config.lab2d_settings = create_lab2d_settings(ascii_map, config)

  # Action set configuration.
  config.action_set = ACTION_SET
  # Observation format configuration.
  config.individual_observation_names = [
      "RGB",
      "POSITION",
      "ORIENTATION",
  ]
  concept_prefix = "WORLD.CONCEPT"
  config.global_observation_names = [
      "WORLD.RGB",
      "WORLD.CONCEPT_AGENT_POSITIONS",
      "WORLD.CONCEPT_AGENT_ORIENTATIONS",
      "WORLD.CONCEPT_TOMATO_POSITIONS",
      "WORLD.CONCEPT_DISH_POSITIONS",
      "WORLD.CONCEPT_COOKING_POT_POSITIONS",
      "WORLD.CONCEPT_DELIVERY_POSITIONS",
      "WORLD.CONCEPT_NAVIGATION_POSITIONS",
      "WORLD.CONCEPT_AGENT_HAS_TOMATO",
      "WORLD.CONCEPT_AGENT_HAS_DISH",
      "WORLD.CONCEPT_AGENT_HAS_SOUP",
      "WORLD.CONCEPT_COOKING_POT_TOMATO_COUNTS",
      "WORLD.CONCEPT_COOKING_POT_PROGRESS",
      "WORLD.CONCEPT_LOOSE_TOMATO_POSITIONS",
      "WORLD.CONCEPT_LOOSE_DISH_POSITIONS",
      "WORLD.CONCEPT_LOOSE_SOUP_POSITIONS",
      "WORLD.CONCEPT_COUNTER_TOMATOES",
      "WORLD.CONCEPT_COUNTER_DISHES",
      "WORLD.CONCEPT_COUNTER_SOUPS"
  ]

  timestep_spec = {
      "RGB":
          specs.rgb(40, 40),
      "POSITION":
          specs.OBSERVATION["POSITION"],
      "ORIENTATION":
          specs.OBSERVATION["ORIENTATION"],
      "WORLD.RGB":
          specs.rgb(*world_size),
      "WORLD.CONCEPT_AGENT_POSITIONS":
          concept_specs.position_concept(
              config.num_players, config.num_players, is_agent=True),
      "WORLD.CONCEPT_AGENT_ORIENTATIONS":
          concept_specs.categorical_concept(
              config.num_players,
              config.num_players,
              num_values=4,
              is_agent=True),
      "WORLD.CONCEPT_TOMATO_POSITIONS":
          concept_specs.position_concept(
              config.num_players, config.num_tomatoes, is_agent=False),
      "WORLD.CONCEPT_DISH_POSITIONS":
          concept_specs.position_concept(
              config.num_players, config.num_dishes, is_agent=False),
      "WORLD.CONCEPT_COOKING_POT_POSITIONS":
          concept_specs.position_concept(
              config.num_players, config.num_cooking_pots, is_agent=False),
      "WORLD.CONCEPT_DELIVERY_POSITIONS":
          concept_specs.position_concept(
              config.num_players, config.num_delivery_locations,
              is_agent=False),
      "WORLD.CONCEPT_NAVIGATION_POSITIONS":
          concept_specs.position_concept(
              config.num_players,
              config.num_navigation_locations,
              is_agent=False),
      "WORLD.CONCEPT_AGENT_HAS_TOMATO":
          concept_specs.binary_concept(
              config.num_players, config.num_players, is_agent=True),
      "WORLD.CONCEPT_AGENT_HAS_DISH":
          concept_specs.binary_concept(
              config.num_players, config.num_players, is_agent=True),
      "WORLD.CONCEPT_AGENT_HAS_SOUP":
          concept_specs.binary_concept(
              config.num_players, config.num_players, is_agent=True),
      "WORLD.CONCEPT_COOKING_POT_TOMATO_COUNTS":
          concept_specs.categorical_concept(
              config.num_players,
              config.num_cooking_pots,
              num_values=4,
              is_agent=False),
      "WORLD.CONCEPT_COOKING_POT_PROGRESS":
          concept_specs.scalar_concept(
              config.num_players, config.num_cooking_pots, is_agent=False),
      "WORLD.CONCEPT_LOOSE_TOMATO_POSITIONS":
          concept_specs.position_concept(
              config.num_players, config.num_loose_objs, is_agent=False),
      "WORLD.CONCEPT_LOOSE_DISH_POSITIONS":
          concept_specs.position_concept(
              config.num_players, config.num_loose_objs, is_agent=False),
      "WORLD.CONCEPT_LOOSE_SOUP_POSITIONS":
          concept_specs.position_concept(
              config.num_players, config.num_loose_objs, is_agent=False),
      "WORLD.CONCEPT_COUNTER_TOMATOES":
          concept_specs.binary_concept(
              config.num_players, config.num_counters, is_agent=False),
      "WORLD.CONCEPT_COUNTER_DISHES":
          concept_specs.binary_concept(
              config.num_players, config.num_counters, is_agent=False),
      "WORLD.CONCEPT_COUNTER_SOUPS":
          concept_specs.binary_concept(
              config.num_players, config.num_counters, is_agent=False),
  }
  config.timestep_spec = specs.timestep(timestep_spec)
  concept_spec = create_concept_spec(timestep_spec, concept_prefix)

  return config, concept_spec
