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

#!/usr/bin/python3

import re


char_to_biblical = {}
char_to_modern = {}
char_to_bare_bones = {}


with open("hebrew_pron.tsv") as s:
  for line in s:
    char, name, biblical, modern = line.strip("\n").split("\t")
    char_to_biblical[char] = biblical
    char_to_modern[char] = modern

with open("hebrew_mapping.tsv") as s:
  for line in s:
    full, bare = line.strip("\n").split("\t")
    char_to_bare_bones[full] = bare


VOWELS = ["a", "e", "i", "o", "u", "ɑ"]

schwadel = re.compile("([aeiouɑ])_([jwv])_ə")


def clean_pron(pron):
  pron = "_".join(pron)
  pron = pron.replace("i_j", "i")
  pron = schwadel.sub(r"\1_\2", pron)
  # Deal with schwas:

  # if (len(pron) > 5 and
  #     pron[2] == "∅"):
  #   pron = pron[0] + "_e_" + pron[4:]
  # if pron.startswith("w_∅_"):
  #   pron = "w_e_" + pron[4:]
  # elif pron.startswith("v_∅_"):
  #   pron = "v_e_" + pron[4:]
  # elif pron.startswith("b_∅_"):
  #   pron = "b_e_" + pron[4:]
  # elif pron.startswith("l_∅_"):
  #   pron = "l_e_" + pron[4:]
  # elif pron.startswith("j_∅_"):
  #   pron = "j_e_" + pron[4:]
  if pron.endswith("_a_ʔ"):
    pron = pron[:-2]
  elif pron.endswith("_a_h"):
    pron = pron[:-2]
  pron = pron.replace("ə", "e")
  pron = pron.replace("∅_", "").replace("_∅", "")
  return pron


with open("hebrew.tsv") as s:
  for line in s:
    verse, contents = line.strip("\n").split("\t")
    print(verse)
    for token in contents.split():
      try:
        spelling, pron = token.split("/")
      except ValueError:
        print(token)
        continue
      nspelling = []
      pron_biblical = []
      pron_modern = []
      for c in spelling:
        nspelling.append(char_to_bare_bones[c])
      for c in pron:
        try:
          pron_biblical.append(char_to_biblical[c])
        except KeyError:
          pron_biblical.append(c)
        try:
          pron_modern.append(char_to_modern[c])
        except KeyError:
          pron_modern.append(c)
      print("{}\t{}\t{}\t{}".format(pron,
                                    "".join(nspelling),
                                    clean_pron(pron_biblical),
                                    clean_pron(pron_modern)))
