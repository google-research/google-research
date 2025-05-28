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

"""Synthetic data generator, see README."""

import abc
import argparse
import dataclasses
import glob
import itertools
import os
import shutil

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # No TF Logs.

from typing import Iterable  # pylint: disable=g-import-not-at-top

# Imports from pip libraries
import mediapy
import numpy as np
import tensorflow_datasets as tfds
import tqdm


_MIN_IMAGE_RESOLUTION = 1512

NUM_VIDEOS = 100
NUM_FRAMES = 12


class CLICCache:
  """Cache helper of CLIC images."""

  def __init__(self, png_dir):
    self._png_dir = png_dir

  def _path_for_image(self, idx):
    return os.path.join(self._png_dir, f"image_{idx:02d}.png")

  @property
  def all_images_cached(self):
    pngs = glob.glob(os.path.join(self._png_dir, "image_*.png"))
    return len(pngs) == NUM_VIDEOS

  def store_clic_as_png(self):
    """Stores first N valid CLIC images as png."""
    if self.all_images_cached:
      return

    # First download CLIC via TFDS.
    tfds_cache_dir = os.path.join(self._png_dir, "tfds_cache")
    builder = tfds.builder("clic", data_dir=tfds_cache_dir)
    builder.download_and_prepare()
    tf_dataset = builder.as_dataset(split="test")

    # Store as png
    dataset = (x["image"] for x in tf_dataset)
    dataset = ((i, img, img.shape[:2]) for i, img in enumerate(dataset))
    dataset = (
        img for _, img, shape in dataset if min(shape) >= _MIN_IMAGE_RESOLUTION
    )
    with tqdm.tqdm(total=NUM_VIDEOS, desc="Store PNGs") as p:
      for i, image in enumerate(itertools.islice(dataset, NUM_VIDEOS)):
        mediapy.write_image(self._path_for_image(i), image)
        p.update()

    assert self.all_images_cached

    # Now we can remove the TFDS cache.
    shutil.rmtree(tfds_cache_dir)

  def get_image(self, idx):
    return mediapy.read_image(self._path_for_image(idx))


@dataclasses.dataclass
class _CropCoords:
  """Represents coordinates of a crop."""

  left: int
  top: int
  width: int
  height: int

  @classmethod
  def upper_left_coords(cls, crop_size):
    return cls(left=0, top=0, width=crop_size, height=crop_size)

  @classmethod
  def central_crop_coords(cls, image_shape, crop_size):
    h, w, _ = image_shape
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    return cls(left=left, top=top, width=crop_size, height=crop_size)

  def crop(self, image):
    assert self.can_crop(image.shape[0], image.shape[1])
    return image[self.top:self.top + self.height,
                 self.left:self.left + self.width, :,]

  def can_crop(self, h, w):
    if self.left < 0:
      return False
    if self.top < 0:
      return False
    if self.left + self.width > w:
      return False
    if self.top + self.height > h:
      return False
    return True

  def safe_shift(self, r, c, image_h, image_w):
    output = _CropCoords(self.left + c, self.top + r, self.width, self.height)
    if not output.can_crop(image_h, image_w):
      raise ValueError(f"Cannot safely shift by {r}, {c}: {self}")
    return output


def _gaussian_weight_mat(
    size,
    sigma,
    dtype,
):
  """Produces gaussian blur weight matrix."""

  sample_f = np.arange(size, dtype=dtype)

  # Compute gaussian kernel
  x = np.abs(
      sample_f[np.newaxis, :] - np.arange(size, dtype=dtype)[:, np.newaxis]
  )
  weights = np.exp(-0.5 * (x / sigma) ** 2)
  total_weight_sum = np.sum(weights, axis=0, keepdims=True)
  weights = np.where(
      np.abs(total_weight_sum) > 1000.0 * float(np.finfo(np.float32).eps),
      np.divide(weights, np.where(total_weight_sum != 0, total_weight_sum, 1)),
      0,
  )
  # Zero out weights where the sample location is completely outside the input
  # range.
  # Note sample_f has already had the 0.5 removed, hence the weird range below.
  input_size_minus_0_5 = size - 0.5
  return np.where(
      np.logical_and(
          sample_f >= -0.5,
          sample_f <= input_size_minus_0_5,
      )[np.newaxis, :],
      weights,
      0,
  )


def _gaussian_blur(x, sigma):
  """Gaussian blur a tensor."""
  if x.dtype != np.float32:
    raise NotImplementedError(f"image should be float32, got {x.dtype}.")
  if not sigma:
    return x
  spatial_dims = range(len(x.shape))[-3:-1]
  shape = x.shape
  contractions = []
  in_indices = list(range(len(shape)))
  out_indices = list(range(len(shape)))
  for i, d in enumerate(spatial_dims):
    m = shape[d]
    w = _gaussian_weight_mat(size=m, sigma=sigma, dtype=x.dtype)
    contractions.append(w)
    contractions.append([d, len(shape) + i])
    out_indices[d] = len(shape) + i
  contractions.append(out_indices)
  return np.einsum(x, in_indices, *contractions, optimize="optimal")


def _to_uint8(im):
  return im.clip(min=0.0, max=255.0).round().astype(np.uint8)


# ------------------------------------------------------------------------------


class BaseDataset(abc.ABC):
  """Base class for all datasets."""

  def __init__(self, x, clic_png_dir):
    self.x = x
    self._clic_cache = CLICCache(png_dir=clic_png_dir)
    assert self._clic_cache.all_images_cached

  def iter_videos(self):
    """Yields video of shape (NUM_FRAMES, 512, 512, 3)."""
    for video_idx in range(NUM_VIDEOS):
      yield self.get_video(video_idx)

  def get_video(self, video_idx):
    """Yields video of shape (NUM_FRAMES, 512, 512, 3)."""
    assert 0 <= video_idx < NUM_VIDEOS, video_idx
    frames = list(self._iter_frames(video_idx=video_idx))
    frames = np.stack(frames)
    assert frames.shape == (NUM_FRAMES, 512, 512, 3)
    assert frames.dtype == np.uint8
    return frames

  @abc.abstractmethod
  def _iter_frames(self, video_idx):
    """By default, this is called to get an eval video, should yield uint8."""

  def __str__(self):
    return f"{self.__class__.__name__}({self.x})"


# ------------------------------------------------------------------------------


class FadeDataset(BaseDataset):
  """Fade to next image."""

  @property
  def fade_rate(self):
    return self.x

  def _iter_frames(self, video_idx):
    """Yields uint8 frames."""
    # Here image is float32 until yielded.
    image0 = self._clic_cache.get_image(video_idx)
    image0 = _CropCoords.central_crop_coords(image0.shape, 512).crop(image0)
    yield image0

    next_idx = (video_idx + 1) % NUM_VIDEOS
    image1 = self._clic_cache.get_image(next_idx)
    image1 = _CropCoords.central_crop_coords(image1.shape, 512).crop(image1)

    # We interpolate in float space.
    image0 = image0.astype(np.float32)
    image1 = image1.astype(np.float32)
    for i in range(1, NUM_FRAMES):
      t = (1-self.fade_rate)**i
      image_i = image0 * t + image1*(1-t)
      yield _to_uint8(image_i)


class ShiftDataset(BaseDataset):
  """Shift images."""

  @property
  def abs_flow(self):
    return abs(round(self.x))

  def _iter_frames(self, video_idx):
    """Yields uint8 frames."""
    if self.abs_flow > 45:
      raise ValueError("x too high!")
    image = self._clic_cache.get_image(video_idx)
    h, w, _ = image.shape
    crop_coords = _CropCoords.central_crop_coords(image.shape, 512)
    yield crop_coords.crop(image)
    for _ in range(1, NUM_FRAMES):
      crop_coords = crop_coords.safe_shift(self.abs_flow,
                                           self.abs_flow,
                                           image_h=h, image_w=w)
      yield crop_coords.crop(image)


class SharpenOrBlurDataset(BaseDataset):
  """Blur/sharpen dataset."""

  @property
  def sigma(self):
    return self.x

  def _iter_forward(self, video_idx, sigma_abs):
    """Iterate in forward direction for a positive sigma."""
    # For negative sigma, this video gets reversed.
    assert sigma_abs >= 0, "Internal correctness."
    image = self._clic_cache.get_image(video_idx)
    image0 = _CropCoords.central_crop_coords(image.shape, 512).crop(image)
    yield _to_uint8(image0)

    # import pudb; pudb.set_trace()
    image0 = image0.astype(np.float32)
    for i in range(1, NUM_FRAMES):
      sigma_i = sigma_abs * i
      image_i = _gaussian_blur(image0, sigma_i)
      yield _to_uint8(image_i)

  def _iter_frames(self, video_idx):
    """Yields uint8 frames."""
    if self.sigma >= 0:
      yield from self._iter_forward(video_idx, self.sigma)
    else:
      yield from reversed(list(self._iter_forward(video_idx, -self.sigma)))


# These are the values for `x` that we used to make Fig. 5 in the paper.
VCT_DATASETS = [
    (
        ShiftDataset,
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32],
    ),
    (
        SharpenOrBlurDataset,
        [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
    ),
    (
        FadeDataset,
        [0.0, 0.05, 0.1, 0.15, 0.2],
    ),
]


def all_vct_datasets(clic_png_dir):
  """Yields all datasets we used for VCT."""
  for dataset_cls, xs in VCT_DATASETS:
    for x in xs:
      yield dataset_cls(x, clic_png_dir)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--clic_png_dir", type=str, required=True)
  p.add_argument("--create", action="store_true")
  p.add_argument("--show", type=str)
  p.add_argument("--x", "-x", type=float)
  p.add_argument("--video_idx", "-i", type=int, default=0)

  p.add_argument("--print_vct_dataset", action="store_true")

  flags = p.parse_args()
  clic_png_dir = flags.clic_png_dir

  if flags.create:
    cache = CLICCache(png_dir=clic_png_dir)
    cache.store_clic_as_png()
    # Test.
    image = cache.get_image(0)
    assert image.shape == (1512, 2016, 3)

    print(f"Cached CLIC at {clic_png_dir}!")

  elif dataset_name := flags.show:
    datasets = {
        "FadeDataset": FadeDataset,
        "ShiftDataset": ShiftDataset,
        "SharpenOrBlurDataset": SharpenOrBlurDataset,
    }
    if dataset_name not in datasets:
      raise KeyError(
          f"Unknown dataset: {dataset_name} must be in {datasets.keys()}"
      )
    dataset: BaseDataset = datasets[dataset_name](flags.x, clic_png_dir)
    video = dataset.get_video(flags.video_idx)
    for i, frame in enumerate(video):
      path = f"./{dataset_name}_{flags.x}_v{flags.video_idx:02d}_f{i:02d}.png"
      mediapy.write_image(path, frame)
      print(f"Created: `{path}`")

  elif flags.print_vct_dataset:
    for ds in all_vct_datasets(clic_png_dir):
      print(ds)

  else:
    raise ValueError("Either pass --create or --show <dataset>, see README.")


if __name__ == "__main__":
  main()
