# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

import json
import os, sys
import numpy as np
import csv

orig_path = 'evaluation_logs_reproducible'
tgt_path = 'cleaned_logs'

os.makedirs(tgt_path, exist_ok=True)

with open(os.path.join(tgt_path, 'stats.csv'), mode='w') as stats_file:
  files = os.listdir(orig_path)
  stats_writer = csv.writer(
      stats_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
  write_head = True

  for f in files:
    data = []
    if not f.endswith('.jsonl'):
      continue
    with open(os.path.join(orig_path, f), 'r') as jfile:
      print('processing ', f)
      for line in jfile:
        dic = json.loads(line)

        # Normalize reward
        reward = dic['evaluation_results']
        # max score is 4
        for k in reward.keys():
          reward[k] /= 4.0
        reward['reward'] = sum([r for k, r in reward.items()]) / len(reward)
        data.append(reward)

    print('  # of Samples:', len(data))
    reward = {}
    for d in data:
      for k, r in d.items():
        if k not in reward:
          reward[k] = []
        reward[k].append(r)
    r_keys = list(reward.keys())
    r_keys.sort()

    if write_head:
      write_head = False
      stats_writer.writerow( ['model_name'] + \
                            [k+k2 for k in r_keys for k2 in ['-mean','-std']])
    for k in r_keys:
      r = reward[k]
      r = np.array(r)
      reward[k] = r
      print('  ', k.replace('_guess', ''),
            '\t: {:.4f} (std: {:.4f})'.format(r.mean(), r.std()))

    stats_writer.writerow( [f.replace('.jsonl','')] + \
                          [r for k in r_keys for r in [reward[k].mean(), reward[k].std()] ])
