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

"""Creates a dictionary mapping integers to node names."""

import random

_RANDOM_SEED = 1234
random.seed(_RANDOM_SEED)

_INTEGER_NAMES = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
]

_POPULAR_NAMES = [
    "James",
    "Robert",
    "John",
    "Michael",
    "David",
    "Mary",
    "Patricia",
    "Jennifer",
    "Linda",
    "Elizabeth",
    "William",
    "Richard",
    "Joseph",
    "Thomas",
    "Christopher",
    "Barbara",
    "Susan",
    "Jessica",
    "Sarah",
    "Karen",
]


_SOUTH_PARK_NAMES = [
    "Eric",
    "Kenny",
    "Kyle",
    "Stan",
    "Tolkien",
    "Heidi",
    "Bebe",
    "Liane",
    "Sharon",
    "Linda",
    "Gerald",
    "Veronica",
    "Michael",
    "Jimbo",
    "Herbert",
    "Malcolm",
    "Gary",
    "Steve",
    "Chris",
    "Wendy",
]

_GOT_NAMES = [
    "Ned",
    "Cat",
    "Daenerys",
    "Jon",
    "Bran",
    "Sansa",
    "Arya",
    "Cersei",
    "Jaime",
    "Petyr",
    "Robert",
    "Jorah",
    "Viserys",
    "Joffrey",
    "Maester",
    "Theon",
    "Rodrik",
    "Lysa",
    "Stannis",
    "Osha",
]


_POLITICIAN_NAMES = [
    "Barack",
    "Jimmy",
    "Arnold",
    "Bernie",
    "Bill",
    "Kamala",
    "Hillary",
    "Elizabeth",
    "John",
    "Ben",
    "Joe",
    "Alexandria",
    "George",
    "Nancy",
    "Pete",
    "Madeleine",
    "Elijah",
    "Gabrielle",
    "Al",
]


_ALPHABET_NAMES = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
]


def create_name_dict(name, nnodes = 20):
  """The runner function to map integers to node names.

  Args:
    name: name of the approach for mapping.
    nnodes: optionally provide nnodes in the graph to be encoded.

  Returns:
    A dictionary from integers to strings.
  """
  if name == "alphabet":
    names_list = _ALPHABET_NAMES
  elif name == "integer":
    names_list = _INTEGER_NAMES
  elif name == "random_integer":
    names_list = []
    for _ in range(nnodes):
      names_list.append(str(random.randint(0, 1000000)))
  elif name == "popular":
    names_list = _POPULAR_NAMES
  elif name == "south_park":
    names_list = _SOUTH_PARK_NAMES
  elif name == "got":
    names_list = _GOT_NAMES
  elif name == "politician":
    names_list = _POLITICIAN_NAMES
  else:
    raise ValueError(f"Unknown approach: {name}")
  name_dict = {}
  for ind, value in enumerate(names_list):
    name_dict[ind] = value
  return name_dict
