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

"""Get embeddings for instructions."""

import argparse
import os
import pickle
import time

import numpy as np
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser()
parser.add_argument('--instruction_file', type=str, required=True)
parser.add_argument('--start_index', type=int, default=0)
parser.add_argument('--num_targets', type=int, default=180000)
parser.add_argument(
    '--model_name', type=str, default='sentence-transformers/sentence-t5-base'
)
parser.add_argument('--batch_size', type=int, default=16)

args = parser.parse_args()

instructions = pickle.load(open(args.instruction_file, 'rb'))
instructions = instructions[
    args.start_index : args.start_index + args.num_targets
]

model = SentenceTransformer(args.model_name)

num_targets = args.num_targets
batch_size = args.batch_size

num_batches = num_targets // batch_size
if num_targets % batch_size != 0:
  num_batches += 1

embeddings = []

start_time = time.time()

for i in range(num_batches):

  batch = instructions[i * batch_size : (i + 1) * batch_size]
  embeddings.append(model.encode(batch))

  if (i + 1) % 100 == 0:
    print(
        f'currently processing batch {i+1}/{num_batches}',
        'time elapsed:',
        time.time() - start_time,
        's',
    )

embeddings = np.concatenate(embeddings, axis=0)

model_type = 'base' if 'base' in args.model_name else 'xxl'

output_file = f'embeddings/embeddings_{model_type}_{args.start_index}_{args.start_index+args.num_targets}_' + os.path.basename(
    args.instruction_file
).replace(
    '.pkl', '.npy'
)
np.save(output_file, embeddings)
