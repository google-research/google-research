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

# Lint as: python3

r"""This script analyzes a set of experiments collected in one directory.

It expects the following structure: parent_dir/experiment_name/*.csv

Each experiment is expected to have at least 4 csv files:
pristine.csv, corrupt.csv, train_sample.csv and test.csv

Note that when running without label noise the corrupt.csv file has no data
but it's still present.

Example usage:
    python -m coherent_gradients.imagenet_analyze_results
      /tmp/parent_dir -o /tmp/results
"""

from __future__ import print_function

import argparse
import collections
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams.update({'figure.autolayout': True})
plt.style.use('default')


def plot(args):
  """Plot experiment results."""
  experiments = [exp.name for exp in os.scandir(args.exptdir) if exp.is_dir()]
  results = {}
  for e in experiments:
    results[e] = collections.OrderedDict()
  for e in results:
    for key in ['pristine', 'corrupt', 'train_sample', 'test']:
      name = '{}/{}/{}.csv'.format(args.exptdir, e, key)
      print('-I- reading {}'.format(name))
      df = pd.read_csv(name)
      if not df:
        continue
      df = df.groupby('epoch').mean().reset_index()
      if key == 'train_sample':
        # Just better naming than train_sample
        results[e]['train'] = df
      else:
        results[e][key] = df

  desc = {'loss': 'loss', 'acc1': 'top-1 accuracy'}

  for dataset in (('pristine', 'corrupt'), ('train', 'test')):
    for stat in ['loss', 'acc1']:
      plt.figure(figsize=(8, 4))
      plt.xlabel('epoch')
      plt.ylabel(desc[stat])
      if stat == 'loss':
        plt.ylim(0, 9)
      for exp in experiments:
        p = plt.plot(
            results[exp][dataset[0]]['epoch'],
            results[exp][dataset[0]][stat],
            '-',
            label='{} [{}]'.format(exp, dataset[0]))
        color = p[-1].get_color()
        if dataset[1] in results[exp]:
          plt.plot(
              results[exp][dataset[1]]['epoch'],
              results[exp][dataset[1]][stat],
              '--',
              label='{} [{}]'.format(exp, dataset[1]),
              color=color)
      if stat in ['acc1', 'acc5']:
        plt.ylim(0, 100)
      plt.legend(loc=0)
      plt.tight_layout()
      name = '{}/{}_{}.pdf'.format(args.outdir, stat, '_'.join(dataset))
      print('-I- writing {}'.format(name))
      plt.savefig(name)
    # plt.show()


def main():
  parser = argparse.ArgumentParser(description='analysis')
  parser.add_argument('exptdir', help='parent directory with experiments')
  parser.add_argument(
      '-o',
      '--outdir',
      help='output directory for plots (if none specified defaults to exptdir)')
  args = parser.parse_args()
  if args.outdir is None:
    args.outdir = args.exptdir
  plot(args)


if __name__ == '__main__':
  main()
