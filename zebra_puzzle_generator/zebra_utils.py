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

"""Function and Class utilities for the logic library."""

import copy
import dataclasses
import random
from typing import Any, Dict, List, Optional, Tuple


def th(i: int) -> str:
  """Turning a number i into i-th. Only works correctly for i < 100."""
  if i == 1:
    return '1st'
  if i == 2:
    return '2nd'
  if i == 3:
    return '3rd'
  if i in [21, 31, 41, 51, 61, 71, 81, 91]:
    return f'{i}st'
  if i in [22, 32, 42, 52, 62, 72, 82, 92]:
    return f'{i}nd'
  if i in [23, 33, 43, 53, 63, 73, 83, 93]:
    return f'{i}rd'
  return f'{i}th'


def get_attr_num(
    entity_tuple: Tuple[str, int, int],
    num_numerical_attrs: int = 1) -> int:
  if entity_tuple[0] == 'c':
    return entity_tuple[1] + num_numerical_attrs
  else:
    return entity_tuple[1]


def convert_to_readable_entity(
    attr: int, value: int, num_numerical_attrs: int = 1
    ) -> Tuple[str, int, int]:
  if attr < num_numerical_attrs:
    return ('n', attr, value)
  else:
    return ('c', attr-num_numerical_attrs, value)


def check_redundant(
    clue_type: str,
    lhs_list: List[Tuple[str, int, int]],
    rhs_list: List[Tuple[str, int, int]],
    answer_table: Optional[List[List[int | None]]] = None,
    num_numerical_attrs: int = 1) -> bool:
  """Checks if all entities in clue are already grounded, rendering the clue redundant.

  Args:
    clue_type: A string indicating the clue type.
    lhs_list: The lhs list of entities in the clue.
    rhs_list: The rhs list of entities in the clue.
    answer_table: A (potentially partially filled) answer table.
    num_numerical_attrs: The number of numerical attributes in the puzzle.

  Returns:
    A boolean indicating whether clue with clue_type, lhs_list and rhs_list
    is redundant given the answer table.

  """
  if answer_table is None:
    return False
  match clue_type:
    case 'inbetween':
      lhs = lhs_list[0]
      rhs1 = rhs_list[0]
      rhs2 = rhs_list[1]
      if (lhs[2] in answer_table[get_attr_num(lhs, num_numerical_attrs)]
          and rhs1[2] in answer_table[get_attr_num(rhs1, num_numerical_attrs)]
          and rhs2[2] in answer_table[get_attr_num(rhs2, num_numerical_attrs)]):
        return True
    case 'ends':
      entity = lhs_list[0]
      if entity[2] in answer_table[get_attr_num(entity, num_numerical_attrs)]:
        return True
    case 'nbr' | '=' | '!=' | 'immediate-left' | 'left-of':
      lhs = lhs_list[0]
      rhs = rhs_list[0]
      if (lhs[2] in answer_table[get_attr_num(lhs, num_numerical_attrs)]
          and rhs[2] in answer_table[get_attr_num(rhs, num_numerical_attrs)]):
        return True
  return False


class Attribute:
  """A class to hold different types of attributes used in synthesizing puzzles.

  Attributes:
    attr_type: Whether the attribute is categorical or numerical.
    name: Name of the attribute.
    attr_intro: Optional, is often set to the name itself.
    values: The possible values the attribute can take
    personal_identifier: Whether the attribute is a personal identifier, e.g.
      name, nationality, race etc.
    verb: Verb used in conjunction with this attribute.
    referring_phrase_generator: A function which generates the phrase which can
      be used to refer to this attribute.
    compatible_entities: A list of entity types wihch are compatible with this
      attribute, e.g. a person is compatible with a favorite drink, an animal
      is compatible with a favorite food, an organization is compatible with
      number of meetings.
    relative_pronouns: Relative pronouns to use for this attribute. E.g. "who",
      "which", "that"
    comparatory_phrases: Comparatory phrases for this attribute
      (only for numerical attributes), e.g. "is older/younger than"
    neighbor_phrase: Phrase to use to indicate two entities are neighbors
      (only for numerical attributes).
    immediate_left_phrase: Phrase to indicate one entity is immediately to the
      left of another (only for numerical attributes).
  """

  def __init__(self, attr_type='categorical', name='', attr_intro=None,
               values=None, personal_identifier=False, verb=None,
               referring_phrase_generator=None, compatible_entities=None,
               relative_pronouns=None, comparatory_phrases=None,
               neighbor_phrase=None, immediate_left_phrase=None):
    self.attr_type = attr_type
    self.name = name
    if attr_intro is None:
      self.attr_intro = self.name
    else:
      self.attr_intro = attr_intro
    self.values = values
    self.personal_identifier = personal_identifier
    self.verb = verb
    self.referring_phrase_generator = referring_phrase_generator
    self.compatible_entities = compatible_entities
    self.relative_pronouns = relative_pronouns
    self.comparatory_phrases = comparatory_phrases
    self.neighbor_phrase = neighbor_phrase
    self.immediate_left_phrase = immediate_left_phrase


CATEGORICAL_ATTRIBUTES = {
    'name': Attribute(
        name='name',
        personal_identifier=True,
        verb='has',
        values=[
            'Alex',
            'Barbara',
            'Bob',
            'Charlie',
            'Ali',
            'Maven',
            'Trinity',
            'Jones',
            'Doug',
            'Randy',
            'Katherine',
            'Michelle',
            'Margaret',
            'Drew',
            'Siva',
            'Ahmed',
            'Michael',
            'Mandy',
            'Rajiv',
            'Tao',
            'Wang',
            'Karan',
            'Mallika',
            'Nour',
            'Mohsen',
            'Mohan',
            'Noah',
            'Yair',
            'Joseph',
            'Jackson',
            'Rafael',
            'Rose',
        ],
        referring_phrase_generator=str,
    ),
    'nationality': Attribute(
        name='nationality',
        personal_identifier=True,
        verb='is',
        values=[
            'Brit',
            'German',
            'Dane',
            'Swede',
            'Indian',
            'Italian',
            'American',
            'Australian',
            'Chinese',
            'Japanese',
            'Korean',
            'Mexican',
            'Chilean',
            'Canadian',
            'Malaysian',
            'Bangladeshi',
            'Sri Lankan',
            'Nepali',
            'Croatian',
            'Serbian',
            'Russian',
            'Turkish',
            'Bulgarian',
            'Spanish',
            'Portuguese',
            'Irish',
            'Belarusian',
            'Swiss',
            'Hungarian',
            'Pakistani',
            'Iranian',
            'Israeli',
        ],
        referring_phrase_generator=(lambda x: 'the ' + str(x)),
    ),
    'house_color': Attribute(
        name='house color',
        attr_intro='colored house',
        verb='lives in',
        values=[
            'blue',
            'red',
            'green',
            'yellow',
            'purple',
            'orange',
            'white',
            'black',
            'brown',
            'grey',
            ],
        referring_phrase_generator=(
            lambda x: 'the person who lives in the ' + str(x) + ' house'
        ),
    ),
    'favorite_soft_drink': Attribute(
        name='favorite soft drink',
        verb='has',
        values=[
            'Gatorade',
            'Pepsi',
            '7up',
            'Fanta',
            'Coke',
            'Sprite',
            'Limca',
            'Powerade',
            'Mirinda',
            'Mountain Dew',
            'Dr Pepper',
            'Canada Dry',
        ],
        referring_phrase_generator=(lambda x: 'the person who likes ' + str(x)),
    ),
    'drink': Attribute(
        name='drink',
        verb='drinks',
        values=[
            'water',
            'tea',
            'coffee',
            'milk',
            'beer',
            'hot chocolate',
            'lemonade',
            'orange juice',
            'iced tea',
            'iced coffee',
        ],
        referring_phrase_generator=(
            lambda x: 'the person who drinks ' + str(x)
        ),
    ),
    'car': Attribute(
        name='car',
        verb='drives',
        values=[
            'Lexus',
            'Honda',
            'Hyundai',
            'Tesla',
            'Toyota',
            'Audi',
            'Mercedes',
            'BMW',
            'Ford',
            'Kia',
            'Ferrari',
            'McLaren',
        ],
        referring_phrase_generator=(
            lambda x: 'the person who drives the ' + str(x)
        ),
    ),
    'sport': Attribute(
        name='sport',
        verb='plays',
        values=[
            'tennis',
            'cricket',
            'basketball',
            'soccer',
            'baseball',
            'ice hockey',
            'table tennis',
            'badminton',
            'squash',
            'volleyball',
        ],
        referring_phrase_generator=(lambda x: 'the ' + str(x) + ' player'),
    ),
    'cigarette': Attribute(
        name='cigarette',
        verb='smokes',
        values=[
            'Blends',
            'Blue Master',
            'Marlboro',
            'Pall Mall',
            'Dunhill',
            'Winston',
            'Camel',
            'Newport',
            'Salem',
            'Davidoff',
        ],
        referring_phrase_generator=(
            lambda x: 'the person who smokes ' + str(x)
        ),
    ),
    'musical_instrument': Attribute(
        name='musical instrument',
        verb='plays',
        values=[
            'violin',
            'piano',
            'drums',
            'guitar',
            'cello',
            'flute',
            'saxophone',
            'harp',
            'oboe',
            'harmonica',
        ],
        referring_phrase_generator=(lambda x: 'the ' + str(x) + ' player'),
    ),
    'flowers': Attribute(
        name='flowers',
        attr_intro='type of flower',
        verb='likes',
        values=[
            'carnations',
            'lilies',
            'roses',
            'marigolds',
            'daffodils',
            'tulips',
            'orchids',
            'peonies',
            'dahlias',
            'begonias',
            'daisies',
        ],
        referring_phrase_generator=(lambda x: 'the person who likes ' + str(x)),
    ),
    'fruit': Attribute(
        name='fruit',
        verb='likes',
        values=[
            'apples',
            'bananas',
            'pears',
            'blueberries',
            'strawberries',
            'raspberries',
            'blackberries',
            'oranges',
            'grapes',
            'kiwis',
            'mangoes',
        ],
        referring_phrase_generator=(lambda x: 'the person who eats ' + str(x)),
    ),
    'food': Attribute(
        name='food',
        verb='likes',
        values=[
            'pasta',
            'burgers',
            'tacos',
            'pizza',
            'fried rice',
            'soups',
            'sandwiches',
            'kebabs',
            'curry',
            'noodles',
        ],
        referring_phrase_generator=(lambda x: 'the person who likes ' + str(x)),
    ),
    'profession': Attribute(
        name='profession',
        verb='has',
        personal_identifier=True,
        values=[
            'singer',
            'dancer',
            'doctor',
            'artist',
            'engineer',
            'programmer',
            'spy',
            'teacher',
            'designer',
            'architect',
            'actor',
            'athlete',
            'writer',
            'chef',
        ],
        referring_phrase_generator=(lambda x: 'the ' + str(x)),
    ),
    'motorbike': Attribute(
        name='motorbike',
        verb='drives',
        values=[
            'Yamaha',
            'Honda',
            'Ducati',
            'Harley-Davidson',
            'Bajaj',
            'Piaggio',
            'Suzuki',
            'Kawasaki',
            'Benelli',
            'Buell',
        ],
        referring_phrase_generator=(
            lambda x: 'the person who drives the ' + str(x)
        ),
    ),
    'nba_team': Attribute(
        name='NBA team',
        verb='supports',
        values=[
            'Lakers',
            'Cavaliers',
            'Knicks',
            'Celtics',
            'Warriors',
            'Suns',
            'Bucks',
            'Mavericks',
            'Hawks',
            'Bulls',
        ],
        referring_phrase_generator=(
            lambda x: 'the person who supports ' + str(x)
        ),
    ),
    'physical_activity': Attribute(
        name='physical activity',
        verb='likes',
        values=[
            'hiking',
            'skiing',
            'running',
            'biking',
            'swimming',
            'weight lifting',
            'base jumping',
            'bungee jumping',
            'paragliding',
            'scuba diving',
            'snorkeling',
        ],
        referring_phrase_generator=(lambda x: 'the person who likes ' + str(x)),
    ),
    'book_genre': Attribute(
        name='genre of books',
        verb='reads',
        values=[
            'fantasy',
            'science fiction',
            'history',
            'mystery',
            'romance',
            'short story',
            'humor',
            'magical realism',
            'spiritual',
        ],
        referring_phrase_generator=(
            lambda x: 'the person who reads ' + str(x) + ' books'
        ),
    ),
    'handbag_brand': Attribute(
        name='handbag brand',
        verb='owns',
        values=[
            'Gucci',
            'Coach',
            'Dolce and Gabbana',
            'Balenciaga',
            'Kate Spade',
            'Michael Kors',
            'Polo Ralph Lauren',
            'Tory Burch',
            'Tumi',
            'Versace',
        ],
        referring_phrase_generator=(
            lambda x: 'the person who owns a ' + str(x)
        ),
    ),
    'gemstones': Attribute(
        name='type of gemstone',
        verb='likes',
        values=[
            'Ambers',
            'Amethysts',
            'Aquamarines',
            'Corals',
            'Diamonds',
            'Dolomites',
            'Emeralds',
            'Gypsum',
            'Jade',
            'Moonstones',
            'Obsidians',
            'Opals',
            'Pearls',
            'Rubies',
            'Sapphires',
            'Topaz',
        ],
        referring_phrase_generator=(lambda x: 'the person who likes ' + str(x)),
    ),
    'hobby': Attribute(
        name='favorite hobby',
        verb='has',
        values=[
            'gardening',
            'stamp collecting',
            'bird watching',
            'biking',
            'knitting',
            'dancing',
            'painting',
            'juggling',
            'cooking',
            'astronomy',
            'archery',
            'skiing',
            'fishing',
            'camping',
            'sailing',
            'wine making',
            'horse riding',
            'baking',
            'cheese making',
            'mixology',
            'drone flying',
            'photography',
            'canoeing',
            'kayaking',
            'metal detecting',
            'beekeeping',
            'coin collecting',
            'kite flying',
            'origami',
            'parkour',
            'quilting',
            'crocheting',
            'pottery',
            'welding',
            'wood carving',
            'video games',
            'language learning',
        ],
        referring_phrase_generator=(lambda x: 'the person who likes ' + str(x)),
    ),
    'nfl_team': Attribute(
        name='NFL team',
        verb='supports the',
        values=[
            'New England Patriots',
            'Dallas Cowboys',
            'San Francisco 49ers',
            'Baltimore Ravens',
            'Buffalo Bills',
            'Kansas City Chiefs',
            'Philadelphia Eagles',
            'Green Bay Packers',
            'Pittsburgh Steelers',
            'Denver Broncos',
            'Atlanta Falcons',
            'Arizona Cardinals',
            'Seattle Seahawks',
            'Tennessee Titans',
            'Miami Dolphins',
            'Detroit Lions',
            'Chicago Bears',
            'Cincinnati Bengals',
            'Carolina Panthers',
            'Los Angeles Rams',
        ],
        referring_phrase_generator=(
            lambda x: 'the person who supports the ' + str(x)
        ),
    ),
    'animal': Attribute(
        name='favorite animal',
        verb='likes',
        values=[
            'cats',
            'dogs',
            'pigs',
            'parrots',
            'eagles',
            'squirrels',
            'penguins',
            'lions',
            'tigers',
            'donkeys',
            'leopards',
            'cheetahs',
            'grizzly bears',
            'polar bears',
            'sun bears',
            'panda bears',
            'black bears',
            'turtles',
            'crocodiles',
            'elephants',
            'panthers',
            'cows',
            'rabbits',
            'hares',
            'buffaloes',
            'baboons',
            'sheep',
            'whales',
            'jellyfish',
            'carp',
            'goldfish',
            'viperfish',
            'starfish',
            'catfish',
            'oscars',
            'zanders',
            'sea bass',
            'swordfish',
            'salmon',
            'halibut',
            'blobfish',
            'doctorfish',
            'tilapia',
            'kangaroos',
            'octopuses',
            'phoenixes',
            'aardvarks',
            'amberjacks',
            'eels',
            'hummingbirds',
            'canaries',
            'hippopotamuses',
            'snails',
            'caterpillars',
            'mosquitoes',
            'bats',
            'ferrets',
            'geckoes',
            'kudus',
            'moose',
            'cockroaches',
            'crickets',
            'grasshoppers',
            'meerkats',
            'spiders',
            'lobsters',
            'squids',
            'puffins',
            'ravens',
            'kiwis',
            'koalas',
            'wolverines',
            'akitas',
            'camels',
            'coyotes',
            'snakes',
            'monkeys',
            'ostriches',
            'pigeons',
            'dolphins',
            'frogs',
            'goats',
            'geese',
            'wolves',
            'gorillas',
            'beavers',
            'lizards',
            'flamingoes',
            'swans',
            'elk',
            'ducks',
            'reindeer',
            'bison',
            'sharks',
            'mice',
            'owls',
            'llamas',
            'zebras',
            'otters',
            'crabs',
            'peafowl',
            'rhinos',
            'dinosaurs',
            'doves',
            'badgers',
            'chinchillas',
            'cougars',
            'crows',
            'seals',
            'worms',
            'ants',
            'bees',
            'butterflies',
            'dragonflies',
            'dragons',
            'gadwalls',
            'mules',
            'ligers',
            'seahorses',
            'fangtooths',
            'dugongs',
            'walruses',
            'storks',
            'swallows',
            'songbirds',
            'woodpeckers',
            'starlings',
            'mannikins',
            'pelikans',
            'beetles',
            'finches',
        ],
        referring_phrase_generator=(lambda x: 'the person who likes ' + str(x)),
    ),
}

NUMERICAL_ATTRIBUTES = {
    'house_position': Attribute(
        name='house position',
        attr_intro='house among houses in a row on a street',
        verb='lives in',
        attr_type='numerical',
        values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        referring_phrase_generator=(
            lambda x: 'the person who lives in the '
            + th(x)
            + ' house'
        ),
        comparatory_phrases={
            '>': 'is to the right of',
            '<': 'is to the left of',
        },
        neighbor_phrase='lives next to',
        immediate_left_phrase='lives immediately to the left of',
    ),
    'age': Attribute(
        name='age',
        verb='is',
        attr_type='numerical',
        values=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        referring_phrase_generator=(
            lambda x: 'the person who is ' + str(x) + ' years old'
        ),
        comparatory_phrases={'>': 'is older than', '<': 'is younger than'},
        neighbor_phrase='is either immediately younger or immediately older to',
        immediate_left_phrase='is immediately younger to',
    ),
    'height_in_inches': Attribute(
        name='height in inches',
        verb='has',
        attr_type='numerical',
        values=[
            150,
            151,
            152,
            153,
            154,
            155,
            156,
            157,
            158,
            159,
            160,
            161,
            162,
            163,
            164,
            165,
            166,
            167,
            168,
            169,
            170,
        ],
        referring_phrase_generator=(
            lambda x: 'the person who is ' + str(x) + ' inches tall'
        ),
        comparatory_phrases={'>': 'is taller than', '<': 'is shorter than'},
        neighbor_phrase=(
            'is either immediately taller or immediately shorter than'
        ),
        immediate_left_phrase='is immediately shorter than',
    ),
}


ATTRIBUTE_UNIVERSE = {
    'categorical': CATEGORICAL_ATTRIBUTES,
    'numerical': NUMERICAL_ATTRIBUTES,
}


@dataclasses.dataclass
class SymbolicZebraGroundTruth:
  n: int = 1
  m1: int = 1
  m2: int = 1  # One 1 numerical attribute supported currently for Zebra puzzles
  answer_table: List[List[int | None]] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Clue:
  number: int = 0
  clue_type: str = ''
  lhs_list: List[Tuple[str, int, int]] = dataclasses.field(default_factory=list)
  rhs_list: List[Tuple[str, int, int]] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class SymbolicZebraPuzzle:
  n: int = 1
  m1: int = 1
  m2: int = 1  # One 1 numerical attribute supported currently for Zebra puzzles
  clues: List[Clue] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class ZebraSolverStep:
  clue_list: List[Clue] = dataclasses.field(default_factory=list)
  reason: Optional[str] = None
  auxiliary_info: List[Any] = dataclasses.field(default_factory=list)
  current_answer_table: List[List[int | None]] = dataclasses.field(
      default_factory=list
  )


def get_referring_phrase(attribute: Attribute, index: int) -> str:
  return attribute.referring_phrase_generator(attribute.values[index])


def convert_list_to_string_of_positions(vals: List[int]) -> str:
  """Converts list of integer values to string of positions.

  Args:
    vals: The list of integer values.

  Returns:
    answer: The string containing the values as positions in sequence.
  """
  answer = 'the '
  for i, val in enumerate(vals):
    answer += th(val+1)
    if i < len(vals)-2:
      answer += ', '
    elif i == len(vals) - 2:
      answer += ' and '
  if len(vals) == 1:
    answer += ' position'
  else:
    answer += ' positions'
  return answer


def is_relevant_to_deduction(
    entity: Tuple[str, int, int],
    position: int,
    fills: List[Tuple[Tuple[str, int, int], int]]) -> bool:
  """Compute whether a particular entity, position combination is relevant for a list of fills.

  Args:
    entity: An entity tuple
    position: An integer position
    fills: A list of fills, each fill a tuple of entity, position.

  Returns:
    relevant: Boolean indicating whether the entity, position combination was
      relevant.
  """
  relevant = False
  for fill in fills:
    if (position == fill[1]
        and entity[0] == fill[0][0]
        and entity[1] == fill[0][1]):
      # attribute and position match
      relevant = True
    if entity == fill[0]:
      # attribute and value match
      relevant = True
  return relevant


def new_fills_in_reasoning_block(
    block: List[ZebraSolverStep]) -> List[Tuple[Tuple[str, int, int], int]]:
  """Returns a list of newly filled cells by a reasoning block.

  Args:
    block: List of ZebraSolverStep instances.

  Returns:
    fills: Information newly filled in the answer table by the given reasoning
      block.
  """
  fills = []
  for step in block:
    match step.reason:
      case r if r in ['=,negative-grounded',
                      '!=,grounded',
                      'nbr,possible-locations',
                      'ends-middle-positions',
                      'immediate-left,possible-locations',
                      'left-of,possible-locations',
                      'inbetween,possible-locations']:
        pass
      case 'nbr_grounded':
        if len(step.auxiliary_info) == 3:
          clue = step.clue_list[0]
          lhs, rhs = clue.lhs_list[0], clue.rhs_list[0]
          entity = lhs if step.auxiliary_info[0] == 0 else rhs
          entity_location = step.auxiliary_info[2]
          fills.append((entity, entity_location))
      case 'hard_deduce':
        entity = step.auxiliary_info[0]
        entity_location = step.auxiliary_info[1]
        fills.append((entity, entity_location))
      case '=,0':
        clue = step.clue_list[0]
        lhs, rhs = clue.lhs_list[0], clue.rhs_list[0]
        entity = lhs if step.auxiliary_info[0] == 0 else rhs
        entity_location = rhs[2] if step.auxiliary_info[0] == 0 else lhs[2]
        fills.append((entity, entity_location))
      case r if r in [
          '=,grounded',
          'immediate-left,grounded',
          'left-of,unique-grounded',
      ]:
        clue = step.clue_list[0]
        lhs, rhs = clue.lhs_list[0], clue.rhs_list[0]
        entity = lhs if step.auxiliary_info[0] == 0 else rhs
        entity_location = step.auxiliary_info[1]
        fills.append((entity, entity_location))
      case 'ends,one-end-filled':
        clue = step.clue_list[0]
        entity = clue.lhs_list[0]
        entity_location = (
            step.auxiliary_info[1] - 1 if step.auxiliary_info[0] == 1 else 0
        )
        fills.append((entity, entity_location))
      case 'fill-by-elimination':
        fills1 = step.auxiliary_info[0]  # cell had only one possibility
        fills2 = step.auxiliary_info[1]  # entity had only one possible cell
        for fill in fills1:
          entity = (
              'n' if fill[0] == 0 else 'c',
              0 if fill[0] == 0 else fill[0] - 1,
              fill[2],
          )
          entity_location = fill[1]
          fills.append((entity, entity_location))
        for fill in fills2:
          entity = (
              'n' if fill[0] == 0 else 'c',
              0 if fill[0] == 0 else fill[0] - 1,
              fill[2],
          )
          entity_location = fill[1]
          fills.append((entity, entity_location))
  return fills


class SymbolicToNaturalLanguageMapper:
  """Symbolic to natural language mapper."""

  attribute_map: Optional[Dict[str, List[Attribute]]] = None
  symbolic_puzzle: Optional[SymbolicZebraPuzzle] = None
  symbolic_solution: Optional[List[List[ZebraSolverStep]]] = None
  ground_truth: Optional[SymbolicZebraGroundTruth] = None

  def __init__(
      self,
      symbolic_puzzle: SymbolicZebraPuzzle,
      symbolic_solution: List[List[ZebraSolverStep]],
      ground_truth: SymbolicZebraGroundTruth):
    self.symbolic_puzzle = symbolic_puzzle
    self.symbolic_solution = symbolic_solution
    self.ground_truth = ground_truth
    self.generate_attribute_map()

  def generate_attribute_map(self):
    """Generates attribute map for symbolic puzzle."""
    assert self.symbolic_puzzle is not None
    attribute_map = {
        'categorical': [],
        'numerical': [],
    }
    # sample the attributes
    categorical_attributes = list(
        ATTRIBUTE_UNIVERSE['categorical'].keys()
    )
    selected_categorical_attributes = random.sample(
        categorical_attributes, self.symbolic_puzzle.m1
    )
    # sample values for n entities for each of the attributes.
    for attribute in selected_categorical_attributes:
      universe_attribute = ATTRIBUTE_UNIVERSE['categorical'][
          attribute
      ]
      attribute_map['categorical'].append(copy.deepcopy(universe_attribute))
      attribute_map['categorical'][-1].values = random.sample(
          universe_attribute.values, self.symbolic_puzzle.n
      )
    self.attribute_map = attribute_map

  def create_puzzle_preamble(self) -> str:
    """Creates preamble for the puzzle question text.

    Returns:
      preamble: The preamble string.
    """
    preamble = (
        'There are '
        + str(self.symbolic_puzzle.n)
        + ' people next to each other in a row '
        + 'who have the following characteristics.\n'
    )
    for i in range(self.symbolic_puzzle.m1):
      preamble += (
          'Everyone '
          + self.attribute_map['categorical'][i].verb
          + ' a different '
          + self.attribute_map['categorical'][i].attr_intro
          + ': '
      )
      for j in range(self.symbolic_puzzle.n):
        preamble += self.attribute_map['categorical'][i].values[j]
        if j != self.symbolic_puzzle.n - 1:
          preamble += ', '
        else:
          preamble += '.\n'
    preamble += (
        'Match the people to the correct value for each of their '
        'characteristics using the clues provided below.\n'
    )
    return preamble

  def map_symbolic_entity_to_text(
      self, symbolic_entity: Tuple[str, int, int]
  ) -> str:
    if symbolic_entity[0] == 'c':
      attribute = self.attribute_map['categorical'][symbolic_entity[1]]
      return attribute.referring_phrase_generator(
          attribute.values[symbolic_entity[2]]
      )
    else:
      return (
          'the person at the '
          + th(symbolic_entity[2] + 1)
          + ' position'
      )

  def get_comparatory_phrase(
      self,
      numerical_attr: int,
      comparator: str = '>') -> str:
    attribute = self.attribute_map['numerical'][numerical_attr]
    return attribute.comparatory_phrases[comparator]

  def get_neighbor_phrase(self, numerical_attr: int) -> str:
    attribute = self.attribute_map['numerical'][numerical_attr]
    return attribute.neighbor_phrase

  def get_immediate_left_phrase(self, numerical_attr: int) -> str:
    attribute = self.attribute_map['numerical'][numerical_attr]
    return attribute.immediate_left_phrase

  def map_symbolic_clues(self) -> str:
    """Maps a list of symbolic clues to text.

    Returns:
      clues_text: A string containing textual descriptions of all the clues.
    """
    clues_text = ''
    for i, clue in enumerate(self.symbolic_puzzle.clues):
      clues_text += 'Clue '+str(i+1)+ ': '
      match clue.clue_type:
        case '=':
          lhs = clue.lhs_list[0]
          rhs = clue.rhs_list[0]
          clue_text = self.map_symbolic_entity_to_text(lhs)
          clue_text += ' is '
          clue_text += self.map_symbolic_entity_to_text(rhs)
          clue_text += '.\n'
          clue_text = clue_text[0].capitalize() + clue_text[1:]
          clues_text += clue_text
        case '!=':
          lhs = clue.lhs_list[0]
          rhs = clue.rhs_list[0]
          clue_text = self.map_symbolic_entity_to_text(lhs)
          clue_text += ' is not '
          clue_text += self.map_symbolic_entity_to_text(rhs)
          clue_text += '.\n'
          clue_text = clue_text[0].capitalize() + clue_text[1:]
          clues_text += clue_text
        case 'left-of':
          lhs = clue.lhs_list[0]
          rhs = clue.rhs_list[0]
          clue_text = self.map_symbolic_entity_to_text(lhs)
          clue_text += ' is somewhere to the left of '
          clue_text += self.map_symbolic_entity_to_text(rhs)
          clue_text += '.\n'
          clue_text = clue_text[0].capitalize() + clue_text[1:]
          clues_text += clue_text
        case 'nbr':
          lhs = clue.lhs_list[0]
          rhs = clue.rhs_list[0]
          clue_text = self.map_symbolic_entity_to_text(lhs)
          clue_text += ' is next to '
          clue_text += self.map_symbolic_entity_to_text(rhs)
          clue_text += '.\n'
          clue_text = clue_text[0].capitalize() + clue_text[1:]
          clues_text += clue_text
        case 'ends':
          lhs = clue.lhs_list[0]
          clue_text = self.map_symbolic_entity_to_text(lhs)
          clue_text += ' is at one of the ends.\n'
          clue_text = clue_text[0].capitalize() + clue_text[1:]
          clues_text += clue_text
        case 'immediate-left':
          lhs = clue.lhs_list[0]
          rhs = clue.rhs_list[0]
          clue_text = self.map_symbolic_entity_to_text(lhs)
          clue_text += ' is immediately to the left of '
          clue_text += self.map_symbolic_entity_to_text(rhs)
          clue_text += '.\n'
          clue_text = clue_text[0].capitalize() + clue_text[1:]
          clues_text += clue_text
        case 'inbetween':
          lhs = clue.lhs_list[0]
          rhs1 = clue.rhs_list[0]
          rhs2 = clue.rhs_list[1]
          clue_text = self.map_symbolic_entity_to_text(rhs1)
          clue_text += ' is somewhere in between '
          clue_text += self.map_symbolic_entity_to_text(lhs)
          clue_text += ' and '
          clue_text += self.map_symbolic_entity_to_text(rhs2)
          clue_text += ' in that order.\n'
          clue_text = clue_text[0].capitalize() + clue_text[1:]
          clues_text += clue_text
    return clues_text

  def create_puzzle_question_and_answer(self) -> Tuple[str, str, str]:
    """Generate a numerical question based on final tabular solution of the puzzle.

    Returns:
      question: The numerical question which tests whether the solver has solved
        two entities' positions correctly or not.
      answer_with_cot: The numerical answer along with reasoning steps in
        string form.
      answer: The numerical answer in string form.
    """
    assert self.symbolic_puzzle is not None
    assert self.attribute_map is not None
    assert self.symbolic_solution is not None
    num_reasoning_blocks = len(self.symbolic_solution)
    i = 1
    last_fills = new_fills_in_reasoning_block(self.symbolic_solution[-1])
    while not last_fills:
      last_fills = new_fills_in_reasoning_block(self.symbolic_solution[-1 - i])
      i += 1
    i = 0
    intermediate_fills = new_fills_in_reasoning_block(
        self.symbolic_solution[num_reasoning_blocks // 2]
    )
    while not intermediate_fills:
      intermediate_fills = new_fills_in_reasoning_block(
          self.symbolic_solution[num_reasoning_blocks // 2 + i]
      )
      i += 1
    fill1 = last_fills[0]
    fill2 = intermediate_fills[0]
    question = 'In the final answer, if '
    question += (
        self.map_symbolic_entity_to_text(fill1[0]) + ' is at position $x$'
    )
    question += (
        ' and '
        + self.map_symbolic_entity_to_text(fill2[0])
        + ' is at position $y$'
    )
    question += f', what is the value of ${self.symbolic_puzzle.n+1}y + x$?'
    answer_with_cot = 'Since ' + self.map_symbolic_entity_to_text(fill1[0])
    answer_with_cot += f' is at position ${fill1[1]+1}$ and '
    answer_with_cot += self.map_symbolic_entity_to_text(fill2[0])
    answer_with_cot += f' is at position ${fill2[1]+1}$, the final answer is '
    answer_with_cot += (
        f'${self.symbolic_puzzle.n+1}x{fill2[1]+1} + {fill1[1]+1}$ = '
    )
    answer = f'${(self.symbolic_puzzle.n+1)*(fill2[1]+1) + fill1[1] + 1}$'
    return question, answer_with_cot, answer

  def map_symbolic_puzzle(self) -> str:
    preamble = self.create_puzzle_preamble()
    clues_text = self.map_symbolic_clues()
    return preamble + clues_text

  def get_attribute(self, j: int) -> Attribute:
    assert self.attribute_map is not None
    if j > 0:
      return self.attribute_map['categorical'][j-1]
    else:
      return self.attribute_map['numerical'][0]

  def get_grounding_preamble(
      self,
      entity: Tuple[str, int, int], position: int) -> str:
    text = 'Since we know that '
    text += self.map_symbolic_entity_to_text(entity)
    text += ' is at the ' + th(position+1) + ' position'
    text += ', '
    text = text[0].capitalize() + text[1:]
    return text

  def get_answer_table_as_text(
      self,
      answer_table: List[List[int | None]]) -> str:
    """Convert answer table to readable text format.

    Args:
      answer_table: The answer table

    Returns:
      answer: The answer table in text format represented as a string.
    """
    m1 = len(self.attribute_map['categorical'])
    n = len(answer_table[0])
    answer = ''
    # to evenly space out columns, find maximum width of all
    # attribute names and values
    max_width = 0
    for j in range(m1):
      max_width = max(max_width, len(self.attribute_map['categorical'][j].name))
      for i in range(n):
        max_width = max(
            max_width, len(str(self.attribute_map['categorical'][j].values[i]))
        )
    max_width = max(max_width, len('Position'))
    table_width = (max_width + 2) * (n + 1)
    # print row of positions
    answer += ' Position ' + ' ' * (max_width - len('Position') + 1)
    for i in range(n):
      answer += ' ' + str(i + 1) + ' ' * (max_width - len(str(i + 1)) + 1)
    answer += '\n'
    answer += '-' * table_width + '\n'
    for j in range(m1):
      answer += (
          ' '
          + str(self.attribute_map['categorical'][j].name)
          + ' ' * (max_width
                   - len(self.attribute_map['categorical'][j].name) + 1)
      )
      for i in range(n):
        if answer_table[j + 1][i] is None:
          answer += ' ' * (max_width + 3)
        else:
          answer += (
              ' '
              + str(
                  self.attribute_map['categorical'][j].values[
                      answer_table[j + 1][i]
                  ]
              )
              + ' '
              * (
                  max_width
                  - len(
                      str(
                          self.attribute_map['categorical'][j].values[
                              answer_table[j + 1][i]
                          ]
                      )
                  )
                  + 1
              )
          )
      answer += '\n' + '-' * table_width + '\n'
    return answer

  def map_symbolic_solution(self) -> str:
    """Map symbolic solution to readable text format.

    Returns:
      solution: The natural language step-by-step solution as a string.
    """
    solution_prefix = 'Let us try to use the clues given to solve the puzzle.\n'
    solution_text = solution_prefix
    for block_num, step_block in enumerate(self.symbolic_solution):
      new_fills = new_fills_in_reasoning_block(step_block)
      answer_table = None
      for step in step_block:
        answer_table = step.current_answer_table
        match step.reason:
          case 'fill-by-elimination':
            fills1 = step.auxiliary_info[0]  # cell had only one possibility
            fills2 = step.auxiliary_info[1]  # entity had only one possible cell
            allow_multiple_fills = step.auxiliary_info[2]
            if fills1:
              text = (
                  'We can also deduce that'
                  if allow_multiple_fills
                  else 'Hence, we have that'
              )
              for i, fill in enumerate(fills1):
                symbolic_entity = (
                    'n' if fill[0] == 0 else 'c',
                    0 if fill[0] == 0 else fill[0] - 1,
                    fill[2],
                )
                if i == 0:
                  # first entry
                  text += ' only ' + self.map_symbolic_entity_to_text(
                      symbolic_entity
                  )
                elif i == len(fills1) - 1:
                  # last entry
                  text += ' and only '
                  text += self.map_symbolic_entity_to_text(symbolic_entity)
                else:
                  text += ', only '
                  text += self.map_symbolic_entity_to_text(symbolic_entity)
                text += (
                    ' can occur at the '
                    + th(fill[1] + 1)
                    + ' position'
                )
              text += '.\n'
              text = text[0].capitalize() + text[1:]
              solution_text += text
            if fills2:
              text = (
                  'We can also deduce that '
                  if allow_multiple_fills
                  else 'Hence, we have that '
              )
              for i, fill in enumerate(fills1):
                symbolic_entity = (
                    'n' if fill[0] == 0 else 'c',
                    0 if fill[0] == 0 else fill[0] - 1,
                    fill[2],
                )
                if i == 0:
                  # first entry
                  text += self.map_symbolic_entity_to_text(symbolic_entity)
                elif i == len(fills2) - 1:
                  # last entry
                  text += ' and '
                  text += self.map_symbolic_entity_to_text(symbolic_entity)
                else:
                  text += ', '
                  text += self.map_symbolic_entity_to_text(symbolic_entity)
                text += (
                    ' has to be at the '
                    + th(fill[1] + 1)
                    + ' position'
                )
              text += '.\n'
              text = text[0].capitalize() + text[1:]
              solution_text += text
          case 'hard-deduce':
            entity = step.auxiliary_info[0]
            entity_location = step.auxiliary_info[1]
            text = 'Clues '
            for i, clue in enumerate(step.clue_list):
              if i == 0:
                text += str(clue.number)
              elif i == len(step.clue_list) - 1:
                text += 'and '
                text += str(clue.number)
              else:
                text += ', '
                text += str(clue.number)
            text += ' together imply that '
            text += (self.map_symbolic_entity_to_text(entity)
                     + ' can only occur at the ')
            text += th(entity_location+1) + ' position'
            text += '.\n'
            text = text[0].capitalize() + text[1:]
            solution_text += text
          case '=,grounded':
            clue = step.clue_list[0]
            lhs, rhs = clue.lhs_list[0], clue.rhs_list[0]
            entity = lhs if step.auxiliary_info[0] == 0 else rhs
            grounding_entity = rhs if step.auxiliary_info[0] == 0 else lhs
            entity_location = step.auxiliary_info[1]
            text = ''
            if grounding_entity[0] != 'n':
              text += self.get_grounding_preamble(
                  grounding_entity, entity_location
              )
            text += 'we use Clue ' + str(clue.number)
            text += ' to deduce that '
            text += self.map_symbolic_entity_to_text(entity)
            text += (
                ' is also at the '
                + th(entity_location + 1)
                + ' position'
            )
            text += '.\n'
            text = text[0].capitalize() + text[1:]
            solution_text += text
          case '=,negative-grounded':
            clue = step.clue_list[0]
            lhs, rhs = clue.lhs_list[0], clue.rhs_list[0]
            lhs_indices, rhs_indices = (
                step.auxiliary_info[0],
                step.auxiliary_info[1],
            )
            lhs_indices = list(
                filter(
                    lambda i: is_relevant_to_deduction(lhs, i, new_fills),  # pylint: disable=cell-var-from-loop
                    lhs_indices,
                )
            )
            rhs_indices = list(
                filter(
                    lambda i: is_relevant_to_deduction(rhs, i, new_fills),  # pylint: disable=cell-var-from-loop
                    rhs_indices,
                )
            )
            text = (
                'Because '
                + self.map_symbolic_entity_to_text(lhs)
                + ' is '
                + self.map_symbolic_entity_to_text(rhs)
            )
            text += f' (Clue {clue.number}), '
            if lhs_indices:
              text += self.map_symbolic_entity_to_text(lhs)
              text += (
                  ' cannot occur at '
                  + convert_list_to_string_of_positions(lhs_indices)
                  + '. '
              )
            if rhs_indices:
              if lhs_indices:
                text += 'Similarly, '
              text += self.map_symbolic_entity_to_text(rhs)
              text += (
                  ' cannot occur at '
                  + convert_list_to_string_of_positions(rhs_indices)
                  + '. '
              )
            text += '\n'
            text = text[0].capitalize() + text[1:]
            solution_text += text
          case '!=,grounded':
            clue = step.clue_list[0]
            lhs, rhs = clue.lhs_list[0], clue.rhs_list[0]
            entity = lhs if step.auxiliary_info[0] == 0 else rhs
            grounding_entity = rhs if step.auxiliary_info[0] == 0 else lhs
            entity_location = step.auxiliary_info[1]
            text = ('Because '
                    + self.map_symbolic_entity_to_text(grounding_entity))
            text += ' is at the ' + th(entity_location+1) + ' position'
            text += ', from Clue '+str(clue.number) + ' we have that '
            text += self.map_symbolic_entity_to_text(entity)
            text += (' is not at the ' + th(entity_location+1)
                     + ' position')
            text += '.\n'
            text = text[0].capitalize() + text[1:]
            solution_text += text
          case 'nbr,grounded':
            clue = step.clue_list[0]
            lhs, rhs = clue.lhs_list[0], clue.rhs_list[0]
            entity = lhs if step.auxiliary_info[0] == 0 else rhs
            indices = step.auxiliary_info[1]
            text = ''
            if len(step.auxiliary_info) == 2:
              text = 'Since '+self.map_symbolic_entity_to_text(lhs)
              text += ' is adjacent to ' + self.map_symbolic_entity_to_text(rhs)
              text += ' (Clue '+str(clue.number)+'), '
              text += self.map_symbolic_entity_to_text(entity)
              text += (' cannot occur at '
                       + convert_list_to_string_of_positions(indices) + '. ')
              text += '\n'
            elif len(step.auxiliary_info) == 3:
              text = 'Since '+self.map_symbolic_entity_to_text(lhs)
              text += ' is adjacent to ' + self.map_symbolic_entity_to_text(rhs)
              text += ' (Clue '+str(clue.number)+'), '
              text += self.map_symbolic_entity_to_text(entity)
              text += (' has to be at the '
                       + th(step.auxiliary_info[2]+1) + ' position.\n')
            text = text[0].capitalize() + text[1:]
            solution_text += text
          case 'nbr,possible-locations':
            clue = step.clue_list[0]
            lhs, rhs = clue.lhs_list[0], clue.rhs_list[0]
            lhs_indices, rhs_indices = (step.auxiliary_info[0],
                                        step.auxiliary_info[1])
            lhs_indices = list(
                filter(lambda i: is_relevant_to_deduction(lhs, i, new_fills),  # pylint: disable=cell-var-from-loop
                       lhs_indices))
            rhs_indices = list(
                filter(lambda i: is_relevant_to_deduction(rhs, i, new_fills),  # pylint: disable=cell-var-from-loop
                       rhs_indices))
            text = 'Since '+ self.map_symbolic_entity_to_text(lhs)
            text += ' is adjacent to ' + self.map_symbolic_entity_to_text(rhs)
            text += ' (Clue '+str(clue.number)+'), we deduce that '
            if lhs_indices:
              text += self.map_symbolic_entity_to_text(lhs)
              text += (' cannot occur at '
                       + convert_list_to_string_of_positions(lhs_indices)
                       + '. ')
            if rhs_indices:
              if lhs_indices:
                text += 'Similarly, '
              text += self.map_symbolic_entity_to_text(rhs)
              text += (' cannot occur at '
                       + convert_list_to_string_of_positions(rhs_indices)
                       + '. ')
            text += '\n'
            text = text[0].capitalize() + text[1:]
            solution_text += text
          case 'ends,one-end-filled':
            clue = step.clue_list[0]
            entity = clue.lhs_list[0]
            entity_location = (step.auxiliary_info[1]-1
                               if step.auxiliary_info[0] == 1 else 0)
            text = self.map_symbolic_entity_to_text(entity)
            text += (' has to be at the '
                     + th(entity_location+1)
                     + ' position due to Clue '+str(clue.number))
            text += '.\n'
            text = text[0].capitalize() + text[1:]
            solution_text += text
          case 'ends,middle-positions':
            clue = step.clue_list[0]
            entity = clue.lhs_list[0]
            text = 'From Clue '+str(clue.number)
            text += ', we can remove '
            text += self.map_symbolic_entity_to_text(entity)
            text += ' as a possibility in the middle positions'
            text += '.\n'
            text = text[0].capitalize() + text[1:]
            solution_text += text
          case 'immediate-left,grounded':
            clue = step.clue_list[0]
            lhs, rhs = clue.lhs_list[0], clue.rhs_list[0]
            entity = lhs if step.auxiliary_info[0] == 0 else rhs
            entity_location = step.auxiliary_info[1]
            text = 'From Clue '+str(clue.number)
            text += ', we deduce that '
            text += self.map_symbolic_entity_to_text(entity)
            text += ' is at the ' + th(entity_location+1) + ' position'
            text += '.\n'
            text = text[0].capitalize() + text[1:]
            solution_text += text
          case 'immediate-left,possible-locations':
            clue = step.clue_list[0]
            lhs, rhs = clue.lhs_list[0], clue.rhs_list[0]
            lhs_indices, rhs_indices = (step.auxiliary_info[0],
                                        step.auxiliary_info[1])
            lhs_indices = list(
                filter(lambda i: is_relevant_to_deduction(lhs, i, new_fills),  # pylint: disable=cell-var-from-loop
                       lhs_indices))
            rhs_indices = list(
                filter(lambda i: is_relevant_to_deduction(rhs, i, new_fills),  # pylint: disable=cell-var-from-loop
                       rhs_indices))
            text = 'Since '+self.map_symbolic_entity_to_text(lhs)
            text += (' is immediately to the left of '
                     + self.map_symbolic_entity_to_text(rhs))
            text += ' (Clue '+str(clue.number)+'), '
            if lhs_indices:
              text += self.map_symbolic_entity_to_text(lhs)
              text += (' cannot occur at '
                       + convert_list_to_string_of_positions(lhs_indices)
                       + '. ')
            if rhs_indices:
              if lhs_indices:
                text += 'Similarly, '
              text += self.map_symbolic_entity_to_text(rhs)
              text += (' cannot occur at '
                       + convert_list_to_string_of_positions(rhs_indices)
                       + '. ')
            text += '\n'
            text = text[0].capitalize() + text[1:]
            solution_text += text
          case 'left-of,unique-grounded':
            clue = step.clue_list[0]
            lhs, rhs = clue.lhs_list[0], clue.rhs_list[0]
            entity = lhs if step.auxiliary_info[0] == 0 else rhs
            entity_location = step.auxiliary_info[1]
            text = 'Since '+self.map_symbolic_entity_to_text(lhs)
            text += ' is to the left of '+self.map_symbolic_entity_to_text(rhs)
            text += ' (Clue '+str(clue.number)+'), '
            text += self.map_symbolic_entity_to_text(entity)
            text += (' has to be at the '
                     + th(entity_location+1) + ' position.')
            text += '\n'
            text = text[0].capitalize() + text[1:]
            solution_text += text
          case 'left-of,possible-locations':
            clue = step.clue_list[0]
            lhs, rhs = clue.lhs_list[0], clue.rhs_list[0]
            lhs_indices, rhs_indices = (step.auxiliary_info[0],
                                        step.auxiliary_info[1])
            lhs_indices = list(
                filter(lambda i: is_relevant_to_deduction(lhs, i, new_fills),  # pylint: disable=cell-var-from-loop
                       lhs_indices))
            rhs_indices = list(
                filter(lambda i: is_relevant_to_deduction(rhs, i, new_fills),  # pylint: disable=cell-var-from-loop
                       rhs_indices))
            # lhs_index, rhs_index indicate whether lhs or rhs respectively
            # were grounded before this clue was applied.
            lhs_index, rhs_index = (step.auxiliary_info[2],
                                    step.auxiliary_info[3])
            text = ''
            if lhs_index >= 0 and rhs_index == -1 and rhs_indices:
              text = 'Since '+self.map_symbolic_entity_to_text(rhs)
              text += (' is to the right of '
                       + self.map_symbolic_entity_to_text(lhs))
              text += (' (Clue '+str(clue.number)
                       + '), the former cannot occur at '
                       + convert_list_to_string_of_positions(rhs_indices)
                       + '. ')
            if rhs_index >= 0 and lhs_index == -1 and lhs_indices:
              text = 'Since '+self.map_symbolic_entity_to_text(lhs)
              text += (' is to the left of '
                       + self.map_symbolic_entity_to_text(rhs))
              text += (' (Clue '+str(clue.number)
                       + '), the former cannot occur at '
                       + convert_list_to_string_of_positions(lhs_indices)
                       + '. ')
            if ((lhs_index == -1 and rhs_index == -1)
                and (lhs_indices or rhs_indices)):
              text = 'Since '+self.map_symbolic_entity_to_text(lhs)
              text += (' is to the left of '
                       + self.map_symbolic_entity_to_text(rhs))
              text += (' (Clue '+str(clue.number)
                       + '), we can eliminate certain possibilities.'
                       ' We deduce that ')
              if lhs_indices:
                text += self.map_symbolic_entity_to_text(lhs)
                text += (' cannot occur at '
                         + convert_list_to_string_of_positions(lhs_indices)
                         + '. ')
              if rhs_indices:
                if lhs_indices:
                  text += 'Similarly, '
                text += self.map_symbolic_entity_to_text(rhs)
                text += (' cannot occur at '
                         + convert_list_to_string_of_positions(rhs_indices)
                         + '. ')
            text += '\n'
            text = text[0].capitalize() + text[1:]
            solution_text += text
          case 'inbetween,possible-locations':
            clue = step.clue_list[0]
            lhs, rhs1, rhs2 = (clue.lhs_list[0],
                               clue.rhs_list[0],
                               clue.rhs_list[1])
            lhs_indices, rhs1_indices, rhs2_indices = (step.auxiliary_info[0],
                                                       step.auxiliary_info[1],
                                                       step.auxiliary_info[2])
            lhs_indices = list(
                filter(lambda i: is_relevant_to_deduction(lhs, i, new_fills),  # pylint: disable=cell-var-from-loop
                       lhs_indices))
            rhs1_indices = list(
                filter(lambda i: is_relevant_to_deduction(rhs1, i, new_fills),  # pylint: disable=cell-var-from-loop
                       rhs1_indices))
            rhs2_indices = list(
                filter(lambda i: is_relevant_to_deduction(rhs2, i, new_fills),  # pylint: disable=cell-var-from-loop
                       rhs2_indices))
            text = ''
            if lhs_indices or rhs1_indices or rhs2_indices:
              text = 'Since ' + self.map_symbolic_entity_to_text(rhs1)
              text += ' is in between ' + self.map_symbolic_entity_to_text(lhs)
              text += (' and '
                       + self.map_symbolic_entity_to_text(rhs2)
                       + ' in that order,')
              text += (' (Clue '
                       +str(clue.number)
                       +'), we can eliminate certain possibilities.\n')
              if lhs_indices:
                text += 'We deduce that '
                text += self.map_symbolic_entity_to_text(lhs)
                text += (' cannot occur at '
                         + convert_list_to_string_of_positions(lhs_indices)
                         + '. ')
              if rhs1_indices:
                text += 'We deduce that '
                text += self.map_symbolic_entity_to_text(rhs1)
                text += (' cannot occur at '
                         + convert_list_to_string_of_positions(rhs1_indices)
                         + '. ')
              if rhs2_indices:
                text += 'We deduce that '
                text += self.map_symbolic_entity_to_text(rhs2)
                text += (' cannot occur at '
                         + convert_list_to_string_of_positions(rhs2_indices)
                         + '. ')
            text += '\n'
            text = text[0].capitalize() + text[1:]
            solution_text += text
      if block_num == (len(self.symbolic_solution) - 1):
        solution_text += ("Hence the solved table indicating "
                          "everybody's position is:\n")
        solution_text += self.get_answer_table_as_text(answer_table)
      else:
        solution_text += self.get_answer_table_as_text(answer_table)
    return solution_text
