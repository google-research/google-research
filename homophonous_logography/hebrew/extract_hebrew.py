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


import unicodedata

pronchars = set()

with open("hebrew.tsv") as s:
  for line in s:
    for token in line.split():
      try:
        spell, pron = token.split("/")
      except ValueError:
        continue
      for c in pron:
        pronchars.add(c)


for c in pronchars:
  name = unicodedata.name(c)
  if "HEBREW" in name:
    print("{}\t{}\t{}".format(c, name, name.split()[2]))
