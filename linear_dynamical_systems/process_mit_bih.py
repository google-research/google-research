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

"""Generate timeseries in 2 clusters: NSR and SVT from mit-bih data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import random
from absl import app
from absl import flags
import numpy as np
import wfdb

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_dir', None,
    'Local input directory containing the mit-bih file that can be copied from '
    '/namespace/health-research/unencrypted/reference/user/milah/mit_bih/.')
flags.DEFINE_string('outfile_dir', None,
                    'Output filepath.')


def main(argv):
  del argv
  all_ = [100, 101, 102, 103, 104, 105, 106, 107, 108, 111, 112, 113, 114, 115,
          116, 117, 118, 119, 121, 122, 123, 124, 200, 201, 202, 203, 205, 207,
          208, 209, 210, 212, 213, 214, 215, 217, 219, 220, 221, 222, 223, 228,
          230, 231, 232, 233, 234]
  target_rhythms = ['AB', 'AFIB', 'AFL', 'B', 'BII', 'IVR', 'N', 'NOD', 'P',
                    'PREX', 'SBR', 'SVTA', 'T', 'VFL', 'VT']
  rhythms = dict()
  for a in all_:
    ann_ref = wfdb.rdann(FLAGS.input_dir + str(a), 'atr')
    for k, label in enumerate(ann_ref.aux_note):
      label = str(label).strip('\x00').strip('(')
      if label in target_rhythms:
        sampfrom = max(0, ann_ref.sample[k] - 140)
        sampto = ann_ref.sample[k] + 361
        sig, _ = wfdb.rdsamp(FLAGS.input_dir + str(a), channels=[0, 1],
                             sampfrom=sampfrom, sampto=sampto)
        for channel in [0, 1]:
          key = str(a) + ':' + str(k) + ':' + str(channel) + ':' + str(
              ann_ref.sample[k])
          x = np.array(sig)
          x = x[:, channel]
          record = ','.join([key, str(channel), str(label)] + [
              str(i) for i in x])
          if label not in rhythms:
            rhythms[label] = []
          rhythms[label].append(record)

  all_rhythms = sorted(rhythms.keys())
  print(all_rhythms)
  random.seed(1984)
  with file(FLAGS.outfile + 'all.csv', 'w') as f_all:
    for label in all_rhythms:
      records = rhythms[label]
      idxs = range(len(records)/2)
      random.shuffle(idxs)
      outfile = FLAGS.outfile + label + '.csv'
      with file(outfile, 'w') as f:
        for i in idxs:
          f.write(records[2*i] + '\n')
          f.write(records[2*i+1] + '\n')
          f_all.write(records[2*i] + '\n')
          f_all.write(records[2*i+1] + '\n')


if __name__ == '__main__':
  app.run(main)
