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

"""Code for using VMSST to score sentence pairs."""

from absl import app
from absl import flags
import numpy as np
import vmsst_encoder


_SENTENCE_PAIR_FILE = flags.DEFINE_string(
    "sentence_pair_file", None, "TSV file of sentence pairs to be scores."
)

_OUTPUT_FILE = flags.DEFINE_string(
    "output_file", None, "File of sentences, one per line."
)


def cosine(u, v):
  return np.sum(u * v, axis=1) / (np.linalg.norm(u) * np.linalg.norm(v))


def main(argv):
  if len(argv) > 1:
    raise RuntimeError(argv[1:])

  vmsst_enc = vmsst_encoder.VMSSTEncoder()

  lines = open(_SENTENCE_PAIR_FILE.value).readlines()
  left = []
  right = []
  for line in lines:
    sent1, sent2 = line.split("\t")
    left.append(sent1)
    right.append(sent2)

  left_encodings = vmsst_enc.encode(left)["embeddings"]
  right_encodings = vmsst_enc.encode(right)["embeddings"]

  scores = cosine(left_encodings, right_encodings)
  if _OUTPUT_FILE.value:
    fout = open(_OUTPUT_FILE.value, "w")
    for i, line in enumerate(lines):
      fout.write(line.strip() + f"\t{scores[i]})\n")
    fout.close()
  else:
    for i, line in enumerate(lines):
      print(line.strip() + f"\t{scores[i]})")


if __name__ == "__main__":
  app.run(main)
