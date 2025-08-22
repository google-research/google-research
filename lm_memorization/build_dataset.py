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

"""Builds the .numpy files of token sequences used for analyzing memorization.

Example usage:
PILE_DIR="/home/ncarlini/pile/the-eye.eu/public/AI/pile/train/"
python3 rebuild_dataset.py $PILE_DIR data
"""

import csv
import json
import os
import sys

import numpy as np
import tensorflow as tf
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("USAGE: python build_pile_dataset.py PILE_DIR OUTPUT_DIR")
    exit()

  pile_path = sys.argv[1]
  output_path = sys.argv[2]

  pile_files = [tf.io.gfile.GFile(pile_path+"%02d.jsonl"%x) for x in range(30)]

  fin = tf.io.gfile.GFile("gs://mem-data/pile_indices.csv")
  next(fin)  # skip header

  prompts = {}
  counts = {}

  # Load the examples indicated by the byte offsets in the scaling dataset csv.
  for i, row in enumerate(csv.reader(fin)):
    if i % 1000 == 0:
      print(i)
    (exid, fid, line_byte_offset, start, end, take_before, take_after,
     internal_offset, size, start_byte, end_byte, count) = map(int, row)

    pile_files[fid].seek(line_byte_offset)
    next_line = json.loads(next(pile_files[fid]))["text"]

    if start_byte < 0:
      # Faaaast!
      # Here be dragons...
      next_line = bytes(next_line, "utf8")
      sequence = tokenizer.encode(
          next_line[start - take_before:end + take_after].decode(
              "utf8", "ignore"))[internal_offset:internal_offset + size]
      if not sequence:
        sequence = tokenizer.encode(
            "z" + next_line[start:end + take_after].decode(
                "utf8", "ignore"))[1:size + 1]
    else:
      encoded = tokenizer.encode(next_line)
      sequence = encoded[start_byte:end_byte]

    if sequence:
      prompts[exid] = sequence
      counts[exid] = count

  if not os.path.exists(output_path):
    os.mkdir(output_path)

  lens = [100, 150, 200, 250, 300, 350, 400, 450, 500]
  sizes = [56000, 54000, 53000, 52000, 51000, 51000, 50000, 50000, 50000]
  cumsum = np.cumsum(sizes)

  out_prompt = [None] * sum(sizes)
  out_count = [None] * sum(sizes)

  for uid in range(len(out_prompt)):
    if uid in prompts:
      out_prompt[uid] = prompts[uid]
      out_count[uid] = counts[uid]
    else:
      bucket = np.where(uid < cumsum)[0][0]
      print("Write", uid, bucket)
      out_prompt[uid] = np.zeros(lens[bucket], dtype=np.uint16)
      out_count[uid] = 0

  cumsum = [0] + list(cumsum)

  if not tf.io.gfile.exists(output_path):
    tf.io.gfile.mkdir(output_path)
  for l, i, j in zip(lens, cumsum, cumsum[1:]):
    fname = os.path.join(output_path, "prompts_%d.npy" % l)
    with tf.io.gfile.GFile(fname, "wb") as f:
      np.save(f, out_prompt[i:j])
    fname = os.path.join(output_path, "counts_%d.npy" % l)
    with tf.io.gfile.GFile(fname, "wb") as f:
      np.save(f, out_count[i:j])
