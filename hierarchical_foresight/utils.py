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

"""Utilities."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np


def sample_batch_tdm_maze(bs, ims, vae=None):
  """Sample batch for the TDM."""
  b1 = np.random.choice(ims.shape[0], bs)
  batch = []
  labels = []
  for e in b1:
    tstep1 = np.random.choice(ims.shape[1], 1)
    tstep2 = (tstep1 +
              np.random.uniform(-30, 30,
                                (1))).clip(0, ims.shape[1]-1).astype(np.uint8)
    tstep3 = (tstep1 +
              np.random.uniform(-30, 30,
                                (1))).clip(0, ims.shape[1]-1).astype(np.uint8)
    b2 = np.concatenate([tstep1, tstep2, tstep3])
    im = ims[e]
    im0 = im[0]
    im = im[b2]

    if vae is not None:
      for j in range(3):
        if np.random.uniform() < 0.5:
          gen = vae.get_gen(im0, im[j])
          im[j] = gen[0]

    l = int(np.abs(b2[0] - b2[1]) < 3)
    lb = np.zeros(4)
    lb[l] = 1

    l = int(np.abs(b2[0] - b2[2]) < 3) + 2
    lb[l] = 1
    labels.append(lb)
    batch.append(im / 255.)
  labels = np.stack(labels)
  batch = np.stack(batch)
  return batch, labels


def sample_batch_vae(bs, ims, env='franka', epnum=10000, epsize=100):
  """Sample a batch for the VAE."""
  if env == 'franka':
    num = 1
  else:
    num = 2
  b1 = np.random.choice(epnum, bs)
  batch = []
  for e in b1:
    b2 = np.random.choice(epsize, num)
    im = ims[e]
    im = im[b2]
    batch.append(im / 255.)
  batch = np.stack(batch)
  return batch


def sample_batch_tap(bs, ims, epnum=10000, epsize=100):
  """Sample batch for training TAP."""
  b1 = np.random.choice(epnum, bs)
  batch = []
  trajs = []
  for e in b1:
    start = np.random.choice(epsize-20, 1)
    end = np.random.choice(range(start[0]+12, start[0]+20), 1)
    im = ims[e]
    traj = im[start[0]+5:end[0]-5]
    im = im[[start[0], end[0]]]
    batch.append(im / 255.)
    trajs.append(traj/ 255.)
  batch = np.array(batch)
  trajs = np.array(trajs)
  return batch, trajs


def save_im(im, name):
  """Save an image."""
  im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
  cv2.imwrite(name, im.astype(np.uint8))


def read_im(path):
  """Read an image."""
  with open(path, 'rb') as fid:
    raw_im = np.asarray(bytearray(fid.read()), dtype=np.uint8)
    im = cv2.imdecode(raw_im, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # Opencv using BGR order
    im = cv2.resize(im, (64, 48), interpolation=cv2.INTER_LANCZOS4)
  return im
