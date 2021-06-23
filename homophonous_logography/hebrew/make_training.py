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

b = open("hebrew_biblical.tsv", "w")
m = open("hebrew_modern.tsv", "w")

with open("hebrew_out.tsv") as s:
  biblical_line = []
  modern_line = []
  verse = ""
  for line in s:
    line = line.strip("\n").split("\t")
    if len(line) == 1:
      if verse and biblical_line and modern_line:
        b.write("{}\t{}\n".format(verse, " ".join(biblical_line)))
        m.write("{}\t{}\n".format(verse, " ".join(modern_line)))
      biblical_line = []
      modern_line = []
      verse = line[0]
    else:
      _, spelling, biblical, modern = line
      biblical_line.append("{}/{}".format(spelling, biblical))
      modern_line.append("{}/{}".format(spelling, modern))

if verse and biblical_line and modern_line:
  b.write("{}\t{}\n".format(verse, " ".join(biblical_line)))
  m.write("{}\t{}\n".format(verse, " ".join(modern_line)))

b.close()
m.close()
