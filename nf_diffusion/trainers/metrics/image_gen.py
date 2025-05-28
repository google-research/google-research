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

"""Utility functions for computing FID/Inception scores."""

from typing import Any, Dict, Optional, Union, Tuple

from absl import logging
import flax
from flax import jax_utils
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
from universal_diffusion.metrics import inception


# NOTE: I change the min_val from -1. to 0.
def normalize_to_0_255(
    images, min_val=0.0, max_val=1.0, clamp=True
):
  """Convert images to [0, 255]."""
  images = (images - min_val) * 255.0 / (max_val - min_val)
  return images.clip(0.0, 255.0) if clamp else images


def pad_batch(x, batch_size):
  """Pad an array of shape [partial_batch, ...] to the full batch size."""
  partial_batch_size = x.shape[0]
  if partial_batch_size < batch_size:
    zeros_fn = jnp.zeros if isinstance(x, jnp.ndarray) else np.zeros
    conca_fn = jnp.concatenate if isinstance(x, jnp.ndarray) else np.concatenate

    padding = zeros_fn(
        (batch_size - partial_batch_size,) + x.shape[1:], dtype=x.dtype
    )
    padded_x = conca_fn([x, padding], 0)
    return padded_x
  else:
    return x


def unshard(x):
  """Reshape an array of shape [A, B, ...] into one of shape [A * B, ...]."""
  return x.reshape((-1,) + x.shape[2:])


# NOTE: I change the min_val from -1. to 0.
def get_inception_stats(inception_params, images, min_val=0.0, max_val=1.0):
  """Run Inception on images."""
  # TODO(guandao) should I resize it to the shape eats by Inception score?
  images = normalize_to_0_255(images, min_val=min_val, max_val=max_val)

  n_devices = jax.local_device_count()
  batch_size = images.shape[0]
  if batch_size % n_devices:
    padded_size = int(np.ceil(batch_size / n_devices) * n_devices)
    images = pad_batch(images, padded_size)

  run_model = jax.pmap(inception.run_model, axis_name="batch")
  images = common_utils.shard(images)

  softmax, pool3 = run_model(inception_params, images)

  softmax = unshard(softmax)[:batch_size]
  pool3 = unshard(pool3)[:batch_size]

  return softmax, pool3


def compute_inception_scores_for_label(
    stats_real, stats_fake, splits=10, use_vmap=True, compute_kid=True
):
  """Computes the Inception scores."""
  _, pool3_real = stats_real
  softmax_fake, pool3_fake = stats_fake
  logging.info(
      "FID/IS score final stats: pool3_real=%s pool3_fake=%s softmax_fake=%s",
      pool3_real.shape,
      pool3_fake.shape,
      softmax_fake.shape,
  )

  # Compute IS
  is_mean, is_std = inception.get_inception_score_from_softmax(
      softmax_fake, splits
  )

  # Compute FID
  fake_pool3_mean, fake_pool3_cov = inception.get_stats_for_fid(pool3_fake)
  real_pool3_mean, real_pool3_cov = inception.get_stats_for_fid(pool3_real)
  fid = inception.get_fid_score(
      real_pool3_mean, real_pool3_cov, fake_pool3_mean, fake_pool3_cov
  )
  fid_num_examples = pool3_real.shape[0]

  # Compute KID (use_vmap = False since otherwise leads to OOM)
  if compute_kid:
    kid = inception.get_kid_score(pool3_fake, pool3_real, use_vmap=use_vmap)
  else:
    kid = -1

  return {
      "inception_mean": is_mean,
      "inception_std": is_std,
      "fid": fid,
      "kid": kid,
      "fid_num_examples": fid_num_examples,
  }


# @classmethod
# def get_inception_params(cls: "InceptionMetrics") -> "InceptionMetrics":
#   -> "InceptionMetrics":
def get_inception_params():
  # TODO(guandao) note the params here are replicated, so the metrics should
  #   not be replicated before put into use.
  inception_params = jax_utils.replicate(inception.load_params())
  return inception_params


@flax.struct.dataclass
class InceptionMetrics:
  """Computes the precision from model outputs `logits` and `labels`."""

  values: Dict[str, Tuple[np.ndarray, Ellipsis]]
  real_softmax: Optional[np.ndarray]
  real_pool3: Optional[np.ndarray]
  fake_softmax: Optional[np.ndarray]
  fake_pool3: Optional[np.ndarray]
  params: Any = None
  splits: int = 20
  # use_vmap: bool = True
  use_vmap: bool = False
  compute_kid: bool = False

  @classmethod
  def empty(cls, use_vmap=False, compute_kid=False):
    logging.info("Empty: Loading parameters.")
    params = get_inception_params()
    return InceptionMetrics(
        values={},
        params=params,
        real_softmax=None,
        fake_softmax=None,
        real_pool3=None,
        fake_pool3=None,
        use_vmap=use_vmap,
        compute_kid=compute_kid,
    )

  @classmethod
  def from_model_output(
      cls,
      *,
      real_images,
      fake_images,
      params = None,
      use_vmap=False,
      compute_kid=False,
      **_
  ):
    """Construct InceptionMetrics from model output."""
    if params is None:
      # print("(From model output) Loading parameters.")
      logging.info("(From model output) Loading parameters.")
      params = get_inception_params()
    else:
      # print("Params already loaded.")
      logging.info("Params already loaded.")

    # print("Get inception stats for the images.")
    logging.info("Get inception stats for the images.")
    real_softmax, real_pool3 = get_inception_stats(
        params, real_images, min_val=0.0, max_val=1.0
    )
    fake_softmax, fake_pool3 = get_inception_stats(
        params, fake_images, min_val=0.0, max_val=1.0
    )

    return InceptionMetrics(
        values={},
        params=params,
        real_softmax=np.array(real_softmax),
        fake_softmax=np.array(fake_softmax),
        real_pool3=np.array(real_pool3),
        fake_pool3=np.array(fake_pool3),
        use_vmap=use_vmap,
        compute_kid=compute_kid,
    )

  def merge(self, other):
    """Merge with another InceptionMetrics instance."""

    def _has_none_(ins):
      return (
          (ins.real_softmax is None)
          or (ins.fake_softmax is None)
          or (ins.real_pool3 is None)
          or (ins.fake_pool3 is None)
      )

    if _has_none_(other):
      return self
    elif _has_none_(self):
      return other
    else:
      real_softmax = np.concatenate(
          [self.real_softmax, other.real_softmax], axis=0
      )
      fake_softmax = np.concatenate(
          [self.fake_softmax, other.fake_softmax], axis=0
      )
      real_pool3 = np.concatenate([self.real_pool3, other.real_pool3], axis=0)
      fake_pool3 = np.concatenate([self.fake_pool3, other.fake_pool3], axis=0)
      logging.info("FID/IS score #images: %d", real_softmax.shape[0])
    return type(self)(
        values={},
        params=self.params,
        real_softmax=np.array(real_softmax),
        fake_softmax=np.array(fake_softmax),
        real_pool3=np.array(real_pool3),
        fake_pool3=np.array(fake_pool3),
        use_vmap=self.use_vmap,
        compute_kid=self.compute_kid,
    )

  def compute(self):
    # TODO(guandao)
    output = compute_inception_scores_for_label(
        (self.real_softmax, self.real_pool3),
        (self.fake_softmax, self.fake_pool3),
        splits=self.splits,
        use_vmap=self.use_vmap,
        compute_kid=self.compute_kid,
    )
    output = {k: np.array(v) for k, v in output.items()}
    return output
