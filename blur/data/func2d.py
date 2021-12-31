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

"""Collection of simple 2d functions defined on [-1, 1]^2 domain."""
import functools
import numpy as np
import sklearn.datasets as skdatasets


def xor(bs):
  x = np.random.random(size=(bs, 2)) *2 - 1
  return x, np.sign(x[:, 0] * x[:, 1])[:, None]


def circ(bs, c=None, r=None):
  if c is None:
    c = (np.random.random(size=(1, 2)) * 2 -1)
  if r is None:
    r = np.random.random() * 0.5 + 0.2
  x = (np.random.random(size=(bs, 2))* 2 - 1)
  y = (np.linalg.norm(x - c, axis=1, ord=2) < r).astype(np.float32)
  return x, y[:, None] * 2 - 1


def two_moon(bs, noise=0.1):
  x, y = skdatasets.make_moons(n_samples=bs * 3, noise=noise)
  # The dataset is uniformly spaced, but shuffled, so we ask
  # for more samples and then truncate to get non-uniform samples.
  x = x[:bs]
  y = y[:bs]
  x[:, 0] = (x[:, 0] + 1) *2 / 3  - 1
  x[:, 1] = (x[:, 1] + 0.5) *2 / 1.5  - 1
  return x, (y * 2 - 1)[:, None]


def concentric_circles(bs, noise=0.1, factor=0.5):
  x1, y1 = skdatasets.make_circles(bs // 2, noise=noise, factor=factor)
  x2, y2 = skdatasets.make_circles(bs // 2, noise=noise, factor=factor)
  y2 += 2
  x2 *= 5
  x = np.concatenate([x1, x2])
  y = np.concatenate([y1, y2])

  return x, y[:, None]


def rotate(x, theta):
  c, s = np.cos(theta), np.sin(theta)
  return  x @ np.array(((c, -s), (s, c)))


def two_moon_rotated(bs, noise=0.1, angle=3.1415 / 2):
  x, y = skdatasets.make_moons(n_samples=bs * 3, noise=noise)
  # The dataset is uniformly spaced, but shuffled, so we ask
  # for more samples and then truncate to get non-uniform samples.
  x = x[:bs]
  y = y[:bs]
  x = rotate(x, angle)
  return x, (y * 2 - 1)[:, None]


def square(bs, angle=0, size=0.2, centers=((0, 0),)):
  x = np.random.random(size=(bs, 2)) *2 - 1
  y = np.zeros((bs,))
  for c in centers:
    y = np.logical_or(
        y, np.maximum(np.abs(x[:, 0] - c[0]), np.abs(x[:, 1] - c[1])) < size)
  x = rotate(x, angle)
  return x, (y * 2 - 1)[:, None]


def blobs(bs, n_centers, seed=None):
  x, y = skdatasets.make_blobs(n_samples=bs, centers=n_centers,
                               random_state=seed, center_box=[-1, 1],
                               cluster_std=0.1)
  return x, y[:, None]


def nxor(bs):
  x = np.random.random(size=(bs, 2)) *2 - 1
  return x, -np.sign(x[:, 0] * x[:, 1])[:, None]


def bool_or(bs):
  x = np.random.random(size=(bs, 2)) *2 - 1
  r = np.logical_or((x[:, 0] > 0), (x[:, 1] > 0))[:, None]
  r = (r.astype(np.float32) - 0.5) * 2
  return x, r


def bool_and(bs):
  x = np.random.random(size=(bs, 2)) *2 - 1
  r = np.logical_and((x[:, 0] > 0), (x[:, 1] > 0))[:, None]
  r = (r.astype(np.float32) - 0.5) * 2
  return x, r


def bool_qand(bs):
  """Positive if x[0] < 0 and x[1] > 0."""
  x = np.random.random(size=(bs, 2)) *2 - 1
  r = np.logical_and((x[:, 0] < 0), (x[:, 1] > 0))[:, None]
  r = (r.astype(np.float32) - 0.5) * 2
  return x, r


def random_xor(bs, proj=None, bias=None):
  """Random xor that first projects its data using proj and bias."""
  if proj is None:
    proj = (np.random.random(size=(2, 2)) - 0.5) * 2
  if bias is None:
    bias = (np.random.random(size=(1, 2)) - 0.5)
  x = np.random.random(size=(bs, 2)) *2 - 1
  xx = x @ proj + bias
  return x, np.sign(xx[:, 0] * xx[:, 1])[:, None]


def fixed_random_xor():
  """Random xor with fixed projection/bias."""
  return functools.partial(
      random_xor,
      proj=(np.random.random(size=(2, 2)) - 0.5) * 2,
      bias=np.random.random(size=(1, 2)) - 0.5
  )


def sine(bs, amplitude=None, noise_std=0):
  if not amplitude:
    amplitude = 10 * np.random.rand(1)
  x = 2 * np.pi * np.random.rand(bs)
  noise = noise_std * np.random.rand(bs)
  y = amplitude * np.sin(x) + noise
  return x[Ellipsis, None], y[Ellipsis, None]


def cross(bs, width=0.3):
  x = np.random.random(size=(bs, 2)) * 2 - 1
  y = (np.abs(x[:, 0]) < width) + ((np.abs(x[:, 1])) < width)
  return x, (y.astype(np.float32)[:, None])


def triangle(bs):
  x = np.random.random(size=(bs, 2)) * 2 - 1
  x1, x2 = x[:, 0], x[:, 1]
  y = np.logical_and(np.logical_and(x1 + x2 < 1, -x1 + x2 < 1), x2 > -0.5)
  return x, (y.astype(np.float32)[:, None])


def angular_slice(bs, angle_begin=-0.78, angle_end=.78):
  x = np.random.random(size=(bs, 2)) * 2 - 1
  angle = np.arctan2(x[:, 1], x[:, 0])
  return x, np.logical_and(
      np.less_equal(angle_begin, angle),
      np.less_equal(angle, angle_end)).astype(np.float32).reshape(-1, 1) * 2 - 1


def random_func(bs):
  x = np.random.random(size=(bs, 2)) * 2 - 1
  r = np.logical_or((x[:, 0] > 0), (x[:, 1] > 0))[:, None]
  r = (r.astype(np.float32) - 0.5) * 2
  return x, np.sign(np.random.random(size=(bs, 1))-0.5)
