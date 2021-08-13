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

"""JAX jittable/vmappable data augmentation utilities.
"""
import jax
import jax.numpy as jnp


# =======================================================================
# Data augmentation functions
# =======================================================================
@jax.jit
def rgb_to_hsv(rgb_image):
  # Adapted from the numpy implementation here: https://gist.github.com/PolarNick239/691387158ff1c41ad73c#file-rgb_to_hsv_np-py
  input_shape = rgb_image.shape
  rgb_image = rgb_image.reshape(-1, 3)
  r, g, b = rgb_image[:, 0], rgb_image[:, 1], rgb_image[:, 2]

  maxc = jnp.maximum(jnp.maximum(r, g), b)
  minc = jnp.minimum(jnp.minimum(r, g), b)
  v = maxc

  deltac = maxc - minc
  # s = deltac / maxc
  s = deltac / (maxc + 1e-9)

  deltac = jnp.where(deltac==0, 1, deltac)
  print(deltac)
  # rc = (maxc - r) / deltac
  # gc = (maxc - g) / deltac
  # bc = (maxc - b) / deltac
  rc = (maxc - r) / (deltac + 1e-9)  # NOT SURE WHY EXACTLY THIS IS NEEDED TO PREVENT NANS! OTHERWISE NANS CAN OCCUR!
  gc = (maxc - g) / (deltac + 1e-9)
  bc = (maxc - b) / (deltac + 1e-9)

  h = 4.0 + gc - rc
  h = jnp.where(g==maxc, 2.0 + jnp.where(g == maxc, rc, 0) - jnp.where(g==maxc, bc, 0), h)
  h = jnp.where(r==maxc, jnp.where(r==maxc, bc, 0) - jnp.where(r==maxc, gc, 0), h)
  h = jnp.where(minc==maxc, 0.0, h)

  h = (h / 6.0) % 1.0
  res = jnp.dstack([h, s, v])
  return res.reshape(input_shape)


@jax.jit
def hsv_to_rgb(hsv_image):
  # Adapted from the numpy implementation here: https://gist.github.com/PolarNick239/691387158ff1c41ad73c#file-rgb_to_hsv_np-py
  input_shape = hsv_image.shape
  hsv_image = hsv_image.reshape(-1, 3)
  h, s, v = hsv_image[:, 0], hsv_image[:, 1], hsv_image[:, 2]

  i = jnp.int32(h * 6.0)
  f = (h * 6.0) - i
  p = v * (1.0 - s)
  q = v * (1.0 - s * f)
  t = v * (1.0 - s * (1.0 - f))
  i = i % 6

  rgb_image = jnp.zeros_like(hsv_image)
  v, t, p, q = v.reshape(-1, 1), t.reshape(-1, 1), p.reshape(-1, 1), q.reshape(-1, 1)

  i = jnp.tile(i.reshape(-1,1), (1,3))
  rgb_image = jnp.where(i==0, jnp.hstack([v, t, p]), rgb_image)
  rgb_image = jnp.where(i==1, jnp.hstack([q, v, p]), rgb_image)
  rgb_image = jnp.where(i==2, jnp.hstack([p, v, t]), rgb_image)
  rgb_image = jnp.where(i==3, jnp.hstack([p, q, v]), rgb_image)
  rgb_image = jnp.where(i==4, jnp.hstack([t, p, v]), rgb_image)
  rgb_image = jnp.where(i==5, jnp.hstack([v, p, q]), rgb_image)

  s = jnp.tile(s.reshape(-1,1), (1,3))
  rgb_image = jnp.where(s==0, jnp.hstack([v, v, v]), rgb_image)

  return rgb_image.reshape(input_shape)


@jax.jit
def apply_random_hflip(key, image, prob):
  return jax.lax.cond(jax.random.uniform(key, ()) < prob, lambda x: x[:,:,::-1], lambda x: x, image)


@jax.jit
def apply_random_vflip(key, image, prob):
  return jax.lax.cond(jax.random.uniform(key, ()) < prob, lambda x: x[:,::-1,:], lambda x: x, image)


@jax.jit
def apply_dropout(key, image, prob):
  c,w,h = image.shape
  mask = jax.random.bernoulli(key, p=1-prob, shape=(w,h))
  return image * mask


@jax.jit
def apply_dropout_per_channel(key, image, prob):
  mask = jax.random.bernoulli(key, p=1-prob, shape=image.shape)
  return image * mask


@jax.jit
def apply_cutout(key, image, cutoutwidth, cutoutheight):
  channels, width, height = image.shape
  x0, y0 = jax.random.randint(key, (2,), minval=0, maxval=width+1-cutoutwidth)

  # Construct a mask
  xx, yy = jnp.meshgrid(jnp.arange(width), jnp.arange(height))
  xmask = jnp.where(jnp.logical_and(xx >= x0, xx < x0+cutoutwidth), 0, 1)
  ymask = jnp.where(jnp.logical_and(yy >= y0, yy < y0+cutoutheight), 0, 1)
  mask = jnp.logical_or(xmask, ymask)
  return image * mask


@jax.jit
def apply_crop(key, image, border):
  channels, width, height = image.shape
  padded_width = width + 2 * border
  padded_height = height + 2 * border

  # maxborder = 10  # Hardcoded max border
  maxborder = 20  # Hardcoded max border

  # These are nice but do not work with jit --- actually they do work as long as we have static_argnums=2
  # padded_image = jnp.pad(image, [(0,0), (border, border), (border, border)], mode='constant')
  # padded_image = jnp.pad(image, [(0,0), (border, border), (border, border)], mode='reflect')

  # New strategy: always pad with max possible padding of maxborder, then modify the min/max values for randint sampling to get the right size in the end
  # padded_image = jnp.pad(image, [(0,0), (maxborder, maxborder), (maxborder, maxborder)], mode='constant')
  padded_image = jnp.pad(image, [(0,0), (maxborder, maxborder), (maxborder, maxborder)], mode='reflect')

  x0, y0 = jax.random.randint(key, (2,), minval=maxborder-border, maxval=maxborder+border+1)  # Ensures that x0 and y0 are different
  cropped_image = jax.lax.dynamic_slice(padded_image, (0, x0, y0), (channels, width, height))
  return cropped_image
