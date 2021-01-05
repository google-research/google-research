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

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_heatmap(pdf_func, out_name, size=3):
  w = 100
  x = np.linspace(-size, size, w)
  y = np.linspace(-size, size, w)
  xx, yy = np.meshgrid(x, y)
  coords = np.stack([xx.flatten(), yy.flatten()]).transpose()

  scores = pdf_func(coords)
  a = scores.reshape((w, w))

  plt.imshow(a)
  plt.axis('equal')
  plt.axis('off')
  plt.savefig(out_name, bbox_inches='tight')
  plt.close()


def plot_samples(samples, out_name):
  plt.scatter(samples[:, 0], samples[:, 1])
  plt.axis('equal')
  plt.savefig(out_name, bbox_inches='tight')
  plt.close()


def plot_joint(dataset, samples, out_name):
  x = np.max(dataset)
  y = np.max(-dataset)
  z = np.ceil(max((x, y)))
  plt.scatter(dataset[:, 0], dataset[:, 1], c='r', marker='x')
  plt.scatter(samples[:, 0], samples[:, 1], c='b', marker='.')
  plt.legend(['training data', 'ADE sampled'])
  plt.axis('equal')
  plt.xlim(-z, z)
  plt.ylim(-z, z)
  plt.savefig(out_name, bbox_inches='tight')
  plt.close()

  fname = out_name.split('/')[-1]
  out_name = '/'.join(out_name.split('/')[:-1]) + '/none-' + fname
  plt.figure(figsize=(8, 8))
  plt.scatter(dataset[:, 0], dataset[:, 1], c='r', marker='x')
  plt.scatter(samples[:, 0], samples[:, 1], c='b', marker='.')
  plt.axis('equal')
  plt.xlim(-z, z)
  plt.ylim(-z, z)
  plt.savefig(out_name, bbox_inches='tight')
  plt.close()
