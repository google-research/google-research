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

"""Annealing schedules."""

import jax.numpy as np


def anneal_exponentially(step, i_split, val0, val1):
  t = np.minimum(step / i_split, 1.)
  return np.exp(np.log(val0) * (1 - t) + np.log(val1) * t)


def anneal_linearly(step, i_split, val0, val1):
  return val0 + (val1 - val0) * np.minimum(step / i_split, 1.)


def lr_fn(step, *, i_split, i_end, lr0, lr1, lr2, cosine_decay=False):
  """Anneal in the learning rate to avoid divergence.

  Decay learning rate afterward to a minimum if i_end is provided.

  Args:
    step (int): training iteration.
    i_split (int): step to switch from lr ramp to lr decay.
    i_end (int): maximum number of steps.
    lr0 (float): initial learning rate.
    lr1 (float): maximum learning rate, achieved at i_split.
    lr2 (float): final learning rate, achieved at i_end.
    cosine_decay (bool): decay with a cosine schedule rather than linearly.

  Returns:
    lr (float): learning rate.
  """
  if step < i_split:
    lr = anneal_exponentially(step, i_split, lr0, lr1)
  elif i_end:
    decay_progress = (step - i_split) / (i_end - i_split)
    if cosine_decay:
      return lr1 * np.cos(decay_progress * np.pi / 2)
    else:
      # linear decay
      lr = lr1 * (1 - decay_progress) + lr2 * decay_progress
  else:
    lr = lr1
  return lr


def sigma_noise_std_fn(step, *, i_split, sn0, sn1):
  return anneal_linearly(step, i_split, sn0, sn1)


def mask_rad_fn(step, *, i_split, mr0, mr1):
  return anneal_linearly(step, i_split, mr0, mr1)

