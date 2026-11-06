# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Strip <cite id="...">...</cite> tags from final_response in a JSONL file.

Keeps the cited text, removes only the XML tags.

Usage:
    python strip_citations.py input.jsonl output.jsonl
"""

import json
import re
import sys


def strip_citations(text):
  """Remove <cite id="...">...</cite> tags, keeping inner text."""
  return re.sub(
      r'<cite\s+id="[^"]*">(.*?)</cite>', r"\1", text, flags=re.DOTALL
  )


def main():
  if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <input.jsonl> <output.jsonl>")
    sys.exit(1)

  input_path, output_path = sys.argv[1], sys.argv[2]
  count = 0

  with open(input_path, "r") as fin, open(output_path, "w") as fout:
    for line in fin:
      if not line.strip():
        continue
      data = json.loads(line)
      if "final_response" in data:
        data["final_response"] = strip_citations(data["final_response"])
      fout.write(json.dumps(data, ensure_ascii=False) + "\n")
      count += 1

  print(f"Stripped citations from {count} examples: {output_path}")


if __name__ == "__main__":
  main()
