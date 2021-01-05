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

"""Converts a json file to absl-py command line flags."""
import re
from typing import List

from absl import app
import colorama


def main(argv):
  base_pat = re.compile(r"_?FLAG_[\w_]+")
  good_pat = re.compile(r"^_?FLAG_[\w_]+\.value")
  define_pat = re.compile(r"^_?FLAG_[\w_]+\s*=\s*flags.DEFINE_")
  at_least_one_bad = False

  with open(argv[1]) as fin:
    for i, line in enumerate(fin):
      start = 0
      match = base_pat.search(line, start)
      while match:
        m_good_pat = good_pat.search(line[match.start():])
        m_define_pat = define_pat.search(line[match.start():])
        start = match.end()  # Matches should be non-overlapping

        # Verify if all m_base_pat matches overlap partially with at least one
        # match of an acceptable way to write things.
        # Matches can only start at the same position if they are for the
        # Same flag.
        if not (m_good_pat or m_define_pat):
          at_least_one_bad = True
          new_substr = (
              f"{colorama.Fore.RED}{colorama.Style.BRIGHT}"
              f"{match.group(0)}{colorama.Style.RESET_ALL}"
          )

          new_text = (
              line[:match.start(0)] + new_substr + line[match.end(0):].rstrip()
          )

          # Prints different errors on the same line in the original text
          # as separate errors in the output
          print(
              f"line {colorama.Style.BRIGHT}{i + 1}{colorama.Style.RESET_ALL}: "
              f"{new_text}"
          )

        match = base_pat.search(line, start)

  if at_least_one_bad:
    exit(-1)


if __name__ == "__main__":
  app.run(main)
