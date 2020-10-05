# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

#!/usr/bin/env python
"""Script to plot training results."""

import os
import pickle
import matplotlib  # pylint: disable=unused-import
import matplotlib.pyplot as plt  # pylint: disable=unused-import
import numpy as np

from ravens import utils

# #-----------------------------------------------------------------------------
# # Specify Training Plots
# #-----------------------------------------------------------------------------

# title = 'Transporter Nets Performance on Various Tasks'
# # files = {'Sorting':     'sorting-transporter-1.pkl',
# #          'Insertion':   'insertion-transporter-1.pkl',
# #          'Hanoi':       'hanoi-transporter-1.pkl',
# #          'Aligning':    'aligning-transporter-1.pkl',
# #          'Stacking':    'stacking-transporter-1.pkl',
# #          'Sweeping':    'sweeping-transporter-1.pkl',
# #          'Pushing':     'pushing-transporter-1.pkl',
# #          'Palletizing': 'palletizing-transporter-1.pkl',
# #          'Kitting':     'kitting-transporter-1.pkl',
# #          'Packing':     'packing-transporter-1.pkl'}
# files = {'Sorting':     'sorting-transporter-10.pkl',
#          'Insertion':   'insertion-transporter-10.pkl',
#          'Hanoi':       'hanoi-transporter-10.pkl',
#          'Aligning':    'aligning-transporter-10.pkl',
#          'Stacking':    'stacking-transporter-10.pkl',
#          'Sweeping':    'sweeping-transporter-10.pkl',
#          # 'Pushing':     'pushing-transporter-10.pkl',
#          'Palletizing': 'palletizing-transporter-10.pkl',
#          'Kitting':     'kitting-transporter-10.pkl',
#          'Packing':     'packing-transporter-10.pkl'}
# ylabel = 'Task Success (%)'
# xlabel = 'Training Steps'

# #-----------------------------------------------------------------------------
# # Generate Training Plots
# #-----------------------------------------------------------------------------

# logs = {}
# for name, file in files.items():
#     if os.path.isfile(file):
#         data = pickle.load(open(file, 'rb'))
#         data = np.float32(data)
#         x = np.sort(np.unique(data[:, 0]))
#         y = np.float32([data[data[:, 0] == ix, 1] for ix in x])
#         logs[name] = (x, y)
# fname = os.path.join(os.getcwd(), 'plot.png')
# utils.plot(fname, title, ylabel, xlabel, data=logs, ylim=[0, 1])
# print(f'Done. Plot image saved to: {fname}')

#-----------------------------------------------------------------------------
# Specify Sample Efficiency Plots
#-----------------------------------------------------------------------------

title = 'Sample Efficiency on Various Tasks'
agent = 'form2fit'
files = {
    'Sorting': f'sorting-{agent}-',
    'Insertion': f'insertion-{agent}-',
    'Hanoi': f'hanoi-{agent}-',
    'Aligning': f'aligning-{agent}-',
    'Stacking': f'stacking-{agent}-',
    'Sweeping': f'sweeping-{agent}-',
    'Cable': f'cable-{agent}-',
    'Palletizing': f'palletizing-{agent}-',
    'Kitting': f'kitting-{agent}-',
    'Packing': f'packing-{agent}-'
}
ylabel = 'Task Success (%)'
xlabel = '# of Demonstrations'

#-----------------------------------------------------------------------------
# Generate Training Plots
#-----------------------------------------------------------------------------

logs = {}
for name, file in files.items():
  x, y, std = [], [], []
  for order in range(4):
    fname = file + f'{10**order}-0.pkl'
    if os.path.isfile(fname):
      x.append(len(x))
      data = pickle.load(open(fname, 'rb'))
      data = np.float32(data)
      ix = np.sort(np.unique(data[:, 0]))
      iy = np.float32([data[data[:, 0] == i, 1] for i in ix])
      imax = np.argsort(np.mean(iy, axis=1))[-5:]  # count best 5 runs
      iy = iy[imax, :].reshape(-1)
      std.append(np.std(iy))
      y.append(np.mean(iy))
      print(fname)
      print(np.mean(iy))
  logs[name] = (x, y, std)
xticks = ['1', '10', '100', '1000']
fname = os.path.join(os.getcwd(), 'plot.png')
utils.plot(fname, title, ylabel, xlabel, data=logs, xticks=xticks, ylim=[0, 1])
print(f'Done. Plot image saved to: {fname}')
