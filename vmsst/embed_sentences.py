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

"""Code for using VMSST to encode sentences."""

from absl import app
from absl import flags
import numpy as np
import vmsst_encoder

_SENTENCE_FILE = flags.DEFINE_string(
    "sentence_file", None, "File of sentences, one per line."
)

_OUTPUT_FILE = flags.DEFINE_string(
    "output_file", None, "File of sentences, one per line."
)


def main(argv):
  if len(argv) > 1:
    raise RuntimeError(argv[1:])

  vmsst_enc = vmsst_encoder.VMSSTEncoder()

  lines = open(_SENTENCE_FILE.value).readlines()
  sentences = []
  for line in lines:
    sentences.append(line.strip())

  encodings = vmsst_enc.encode(sentences)["embeddings"]
  if _OUTPUT_FILE.value:
    np.save(_OUTPUT_FILE.value, encodings)
  else:
    print(encodings)


if __name__ == "__main__":
  app.run(main)
