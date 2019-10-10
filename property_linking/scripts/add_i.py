# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Small data fix to add "i/" in front of all entity IDs."""

import sys

cats_file = open(sys.argv[1], "r").readlines()


def remap(split_line):
  return ("i/{}".format(split_line[0]),
          split_line[1],
          "|".join(["i/{}".format(entity)
                    for entity in split_line[2].split("|")]))

lines = [remap(line.split("\t")) for line in cats_file]
cats_file = open(sys.argv[1], "w+")
for line in lines:
  cats_file.write("\t".join(line))
