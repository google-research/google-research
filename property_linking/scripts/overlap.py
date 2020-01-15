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

"""Compute overlap between tokens in name file and examples."""

from __future__ import print_function

import sys
import time

examples = open(sys.argv[1], "r")
names = open(sys.argv[2], "r")

examples_file = [line.strip().split("\t") for line in examples]
names_file = [line.strip().split("\t") for line in names]

example_bow = [line[1].strip() for line in examples_file]
name_bow = [(line[1].strip(), line[2].strip()) for line in names_file]

example_p = [(e,
              "".join([word[0] for word in e.lower().split()]),
              set(e.lower().split())) for e in example_bow]
name_p = [(idx, n.lower()[0], set(n.lower().split())) for (idx, n) in name_bow]

print (len(example_p), len(name_p))

output = []
old_time = time.time()
# one loop order
for i, (e, estarts, example) in enumerate(example_p):
  if i % 10 == 0:
    print ("{}: {}".format(i, time.time() - old_time))
    old_time = time.time()
  computed = (e, [n for n, n0, name in name_p
                  if n0 in estarts and name.issubset(example)])
  if len(examples_file[i]) == 2:
    examples_file[i].extend(["", "", "|".join(computed[1])])
  if len(examples_file[i]) == 3:
    examples_file[i].extend(["", "|".join(computed[1])])
  elif len(examples_file[i]) == 4:
    examples_file[i].extend(["|".join(computed[1])])
  elif len(examples_file[i]) == 5:
    examples_file[i][4] = "|".join(computed[1])
  else:
    print ("examples_file[i] has invalid length: {}".format(
        examples_file[i]))

  examples_file[i] = examples_file[i][:5]

  output.append("\t".join(examples_file[i]) + "\n")


new_examples = open(sys.argv[3], "w")
for row in output:
  new_examples.write(row)
