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

"""Restrict names file to only those in a kb."""

from __future__ import print_function

import csv
import sys

whitelist = set()
kb_file = open(sys.argv[1], "r")
kb_reader = csv.reader(kb_file, delimiter="\t")

for line in kb_reader:
  whitelist.add(line[0].strip())
  whitelist.add(line[1].strip())
  whitelist.add(line[2].strip())

print (len(whitelist))
print (list(whitelist)[:100])

name_file = open(sys.argv[2], "r")
name_new = open(sys.argv[3], "w+")
name_writer = csv.writer(name_new, delimiter="\t")

written = set()
for i, line in enumerate(name_file):
  line = line.strip().split("\t")
  if line[1].strip() in whitelist:
    if line[1].strip() in written:
      print ("Already wrote {}".format(line[1]))
    else:
      name_writer.writerow(line)
      written.add(line[1].strip())
print (len(written))

print (whitelist - written)
