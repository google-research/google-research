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

# pylint: skip-file
"""Functions for processing and loading raw image data."""

import json
import os
import types
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import rawpy

from google_research.yobo.internal import image as lib_image
from google_research.yobo.internal import math
from google_research.yobo.internal import utils
from multiprocessing import Pool
def ParallelMap(f, l):
  with Pool() as p:
    return p.map(f, l)

_Array = Union[np.ndarray, jnp.ndarray]
_Axis = Optional[Union[int, Tuple[int, Ellipsis]]]


def postprocess_raw(
    raw,
    camtorgb,
    exposure = None,
    xnp = np,
):
  """Converts demosaicked raw to sRGB with a minimal postprocessing pipeline.

  Numpy array inputs will be automatically converted to Jax arrays.

  Args:
    raw: [H, W, 3], demosaicked raw camera image.
    camtorgb: [3, 3], color correction transformation to apply to raw image.
    exposure: color value to be scaled to pure white after color correction. If
      None, "autoexposes" at the 97th percentile.
    xnp: either numpy or jax.numpy.

  Returns:
    srgb: [H, W, 3], color corrected + exposed + gamma mapped image.
  """
  if raw.shape[-1] != 3:
    raise ValueError(f'raw.shape[-1] is {raw.shape[-1]}, expected 3')
  if camtorgb.shape != (3, 3):
    raise ValueError(f'camtorgb.shape is {camtorgb.shape}, expected (3, 3)')
  # Convert from camera color space to standard linear RGB color space.
  matmul = math.matmul if xnp == jnp else np.matmul
  rgb_linear = matmul(raw, camtorgb.T)
  if exposure is None:
    exposure = xnp.percentile(rgb_linear, 97)
  # "Expose" image by mapping the input exposure level to white and clipping.
  rgb_linear_scaled = xnp.clip(rgb_linear / exposure, 0, 1)
  # Apply sRGB gamma curve to serve as a simple tonemap.
  srgb = lib_image.linear_to_srgb(rgb_linear_scaled, xnp=xnp)
  return srgb


def pixels_to_bayer_mask(pix_x, pix_y):
  """Computes binary RGB Bayer mask values from integer pixel coordinates."""
  # Red is top left (0, 0).
  r = (pix_x % 2 == 0) * (pix_y % 2 == 0)
  # Green is top right (0, 1) and bottom left (1, 0).
  g = (pix_x % 2 == 1) * (pix_y % 2 == 0) + (pix_x % 2 == 0) * (pix_y % 2 == 1)
  # Blue is bottom right (1, 1).
  b = (pix_x % 2 == 1) * (pix_y % 2 == 1)
  return np.stack([r, g, b], -1).astype(np.float32)


def bilinear_demosaic(bayer, xnp):
  """Converts Bayer data into a full RGB image using bilinear demosaicking.

  Input data should be ndarray of shape [height, width] with 2x2 mosaic pattern:
    -------------
    |red  |green|
    -------------
    |green|blue |
    -------------
  Red and blue channels are bilinearly upsampled 2x, missing green channel
  elements are the average of the neighboring 4 values in a cross pattern.

  Args:
    bayer: [H, W] array, Bayer mosaic pattern input image.
    xnp: either numpy or jax.numpy.

  Returns:
    rgb: [H, W, 3] array, full RGB image.
  """

  def reshape_quads(*planes):
    """Reshape pixels from four input images to make tiled 2x2 quads."""
    planes = xnp.stack(planes, -1)
    shape = planes.shape[:-1]
    # Create [2, 2] arrays out of 4 channels.
    zup = planes.reshape(shape + (2, 2))
    # Transpose so that x-axis dimensions come before y-axis dimensions.
    zup = xnp.transpose(zup, (0, 2, 1, 3))
    # Reshape to 2D.
    zup = zup.reshape((shape[0] * 2, shape[1] * 2))
    return zup

  def bilinear_upsample(z):
    """2x bilinear image upsample."""
    # Using np.roll makes the right and bottom edges wrap around. The raw image
    # data has a few garbage columns/rows at the edges that must be discarded
    # anyway, so this does not matter in practice.
    # Horizontally interpolated values.
    zx = 0.5 * (z + xnp.roll(z, -1, axis=-1))
    # Vertically interpolated values.
    zy = 0.5 * (z + xnp.roll(z, -1, axis=-2))
    # Diagonally interpolated values.
    zxy = 0.5 * (zx + xnp.roll(zx, -1, axis=-2))
    return reshape_quads(z, zx, zy, zxy)

  def upsample_green(g1, g2):
    """Special 2x upsample from the two green channels."""
    z = xnp.zeros_like(g1)
    z = reshape_quads(z, g1, g2, z)
    alt = 0
    # Grab the 4 directly adjacent neighbors in a "cross" pattern.
    for i in range(4):
      axis = -1 - (i // 2)
      roll = -1 + 2 * (i % 2)
      alt = alt + 0.25 * xnp.roll(z, roll, axis=axis)
    # For observed pixels, alt = 0, and for unobserved pixels, alt = avg(cross),
    # so alt + z will have every pixel filled in.
    return alt + z

  r, g1, g2, b = [bayer[(i // 2) :: 2, (i % 2) :: 2] for i in range(4)]
  r = bilinear_upsample(r)
  # Flip in x and y before and after calling upsample, as bilinear_upsample
  # assumes that the samples are at the top-left corner of the 2x2 sample.
  b = bilinear_upsample(b[::-1, ::-1])[::-1, ::-1]
  g = upsample_green(g1, g2)
  rgb = xnp.stack([r, g, b], -1)
  return rgb


bilinear_demosaic_jax = jax.jit(lambda bayer: bilinear_demosaic(bayer, xnp=jnp))


def load_raw_images(
    image_dir, image_names = None
):
  """Loads raw images and their metadata from disk, in parallel.

  Args:
    image_dir: directory containing raw image and EXIF data.
    image_names: files to load (ignores file extension), loads all DNGs if None.

  Returns:
    A tuple (images, exifs).
    images: [N, height, width, 3] array of raw sensor data.
    exifs: [N] list of dicts, one per image, containing the EXIF data.
  Raises:
    ValueError: The requested `image_dir` does not exist on disk.
  """

  if not utils.file_exists(image_dir):
    raise ValueError(f'Raw image folder {image_dir} does not exist.')

  # Load raw images (dng files) and exif metadata (json files).
  def load_raw_exif(image_name):
    base = os.path.join(image_dir, os.path.splitext(image_name)[0])
    with utils.open_file(base + '.dng', 'rb') as f:
      raw = rawpy.imread(f).raw_image
    with utils.open_file(base + '.json', 'rb') as f:
      exif = json.load(f)[0]
    return raw, exif

  if image_names is None:
    image_names = [
        os.path.basename(f)
        for f in sorted(gfile.Glob(os.path.join(image_dir, '*.dng')))
    ]

  data = ParallelMap(load_raw_exif, image_names)
  raws, exifs = zip(*data)
  raws = np.stack(raws, axis=0).astype(np.float32)

  return raws, exifs


# Brightness percentiles to use for re-exposing and tonemapping raw images.
_PERCENTILE_LIST = (80, 90, 97, 99, 100)

# Relevant fields to extract from raw image EXIF metadata.
# For details regarding EXIF parameters, see:
# https://www.adobe.com/content/dam/acom/en/products/photoshop/pdfs/dng_spec_1.4.0.0.pdf.
_EXIF_KEYS = (
    'BlackLevel',  # Black level offset added to sensor measurements.
    'WhiteLevel',  # Maximum possible sensor measurement.
    'AsShotNeutral',  # RGB white balance coefficients.
    'ColorMatrix2',  # XYZ to camera color space conversion matrix.
    'NoiseProfile',  # Shot and read noise levels.
)

# Color conversion from reference illuminant XYZ to RGB color space.
# See http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html.
_RGB2XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])


def process_exif(
    exifs
):
  """Processes list of raw image EXIF data into useful metadata dict.

  Input should be a list of dictionaries loaded from JSON files.
  These JSON files are produced by running
    $ exiftool -json IMAGE.dng > IMAGE.json
  for each input raw file.

  We extract only the parameters relevant to
  1. Rescaling the raw data to [0, 1],
  2. White balance and color correction, and
  3. Noise level estimation.

  Args:
    exifs: a list of dicts containing EXIF data as loaded from JSON files.

  Returns:
    meta: a dict of the relevant metadata for running RawNeRF.
  """
  meta = {}
  exif = exifs[0]
  # Convert from array of dicts (exifs) to dict of arrays (meta).
  for key in _EXIF_KEYS:
    exif_value = exif.get(key)
    if exif_value is None:
      continue
    # Values can be a single int or float...
    if isinstance(exif_value, int) or isinstance(exif_value, float):
      vals = [x[key] for x in exifs]
    # Or a string of numbers with ' ' between.
    elif isinstance(exif_value, str):
      vals = [[float(z) for z in x[key].split(' ')] for x in exifs]
    meta[key] = np.squeeze(np.array(vals))
  # Shutter speed is a special case, a string written like 1/N.
  meta['ShutterSpeed'] = np.fromiter(
      (1.0 / float(exif['ShutterSpeed'].split('/')[1]) for exif in exifs), float
  )

  # Create raw-to-sRGB color transform matrices. Pipeline is:
  # cam space -> white balanced cam space ("camwb") -> XYZ space -> RGB space.
  # 'AsShotNeutral' is an RGB triplet representing how pure white would measure
  # on the sensor, so dividing by these numbers corrects the white balance.
  whitebalance = meta['AsShotNeutral'].reshape(-1, 3)
  cam2camwb = np.array([np.diag(1.0 / x) for x in whitebalance])
  # ColorMatrix2 converts from XYZ color space to "reference illuminant" (white
  # balanced) camera space.
  xyz2camwb = meta['ColorMatrix2'].reshape(-1, 3, 3)
  rgb2camwb = xyz2camwb @ _RGB2XYZ
  # We normalize the rows of the full color correction matrix, as is done in
  # https://github.com/AbdoKamel/simple-camera-pipeline.
  rgb2camwb /= rgb2camwb.sum(axis=-1, keepdims=True)
  # Combining color correction with white balance gives the entire transform.
  cam2rgb = np.linalg.inv(rgb2camwb) @ cam2camwb
  meta['cam2rgb'] = cam2rgb

  return meta


def load_raw_dataset(
    split,
    data_dir,
    image_names,
    exposure_percentile,
    n_downsample,
):
  """Loads and processes a set of RawNeRF input images.

  Includes logic necessary for special "test" scenes that include a noiseless
  ground truth frame, produced by HDR+ merge.

  Args:
    split: DataSplit.TRAIN or DataSplit.TEST, only used for test scene logic.
    data_dir: base directory for scene data.
    image_names: which images were successfully posed by COLMAP.
    exposure_percentile: what brightness percentile to expose to white.
    n_downsample: returned images are downsampled by a factor of n_downsample.

  Returns:
    A tuple (images, meta, testscene).
    images: [N, height // n_downsample, width // n_downsample, 3] array of
      demosaicked raw image data.
    meta: EXIF metadata and other useful processing parameters. Includes per
      image exposure information that can be passed into the NeRF model with
      each ray: the set of unique exposure times is determined and each image
      assigned a corresponding exposure index (mapping to an exposure value).
      These are keys 'unique_shutters', 'exposure_idx', and 'exposure_value' in
      the `meta` dictionary.
      We rescale so the maximum `exposure_value` is 1 for convenience.
    testscene: True when dataset includes ground truth test image, else False.
  """

  image_dir = os.path.join(data_dir, 'raw')

  testimg_file = os.path.join(data_dir, 'hdrplus_test/merged.dng')
  testscene = utils.file_exists(testimg_file)
  if testscene:
    # Test scenes have train/ and test/ split subdirectories inside raw/.
    image_dir = os.path.join(image_dir, split.value)
    if split == utils.DataSplit.TEST:
      # COLMAP image names not valid for test split of test scene.
      image_names = None
    else:
      # Discard the first COLMAP image name as it is a copy of the test image.
      image_names = image_names[1:]

  raws, exifs = load_raw_images(image_dir, image_names)
  meta = process_exif(exifs)

  if testscene and split == utils.DataSplit.TEST:
    # Test split for test scene must load the "ground truth" HDR+ merged image.
    with utils.open_file(testimg_file, 'rb') as imgin:
      testraw = rawpy.imread(imgin).raw_image
    # HDR+ output has 2 extra bits of fixed precision, need to divide by 4.
    testraw = testraw.astype(np.float32) / 4.0
    # Need to rescale long exposure test image by fast:slow shutter speed ratio.
    fast_shutter = meta['ShutterSpeed'][0]
    slow_shutter = meta['ShutterSpeed'][-1]
    shutter_ratio = fast_shutter / slow_shutter
    # Replace loaded raws with the "ground truth" test image.
    raws = testraw[None]
    # Test image shares metadata with the first loaded image (fast exposure).
    meta = {k: meta[k][:1] for k in meta}
  else:
    shutter_ratio = 1.0

  # Next we determine an index for each unique shutter speed in the data.
  shutter_speeds = meta['ShutterSpeed']
  # Sort the shutter speeds from slowest (largest) to fastest (smallest).
  # This way index 0 will always correspond to the brightest image.
  unique_shutters = np.sort(np.unique(shutter_speeds))[::-1]
  exposure_idx = np.zeros_like(shutter_speeds, dtype=np.int32)
  for i, shutter in enumerate(unique_shutters):
    # Assign index `i` to all images with shutter speed `shutter`.
    exposure_idx[shutter_speeds == shutter] = i
  meta['exposure_idx'] = exposure_idx
  meta['unique_shutters'] = unique_shutters
  # Rescale to use relative shutter speeds, where 1. is the brightest.
  # This way the NeRF output with exposure=1 will always be reasonable.
  meta['exposure_values'] = shutter_speeds / unique_shutters[0]

  # Rescale raw sensor measurements to [0, 1] (plus noise).
  blacklevel = meta['BlackLevel'].reshape(-1, 1, 1)
  whitelevel = meta['WhiteLevel'].reshape(-1, 1, 1)
  images = (raws - blacklevel) / (whitelevel - blacklevel) * shutter_ratio

  # Calculate value for exposure level when gamma mapping, defaults to 97%.
  # Always based on full resolution image 0 (for consistency).
  image0_raw_demosaic = np.array(bilinear_demosaic_jax(images[0]))
  image0_rgb = image0_raw_demosaic @ meta['cam2rgb'][0].T
  exposure = np.percentile(image0_rgb, exposure_percentile)
  meta['exposure'] = exposure
  # Sweep over various exposure percentiles to visualize in training logs.
  exposure_levels = {p: np.percentile(image0_rgb, p) for p in _PERCENTILE_LIST}
  meta['exposure_levels'] = exposure_levels

  # Create postprocessing function mapping raw images to tonemapped sRGB space.
  cam2rgb0 = meta['cam2rgb'][0]
  meta['postprocess_fn'] = lambda z, x=exposure: postprocess_raw(z, cam2rgb0, x)

  # Demosaic Bayer images (preserves the measured RGGB values) and downsample
  # if needed. Moving array to device + running processing function in Jax +
  # copying back to CPU is faster than running directly on CPU.
  def processing_fn(x):
    x_jax = jnp.array(x)
    x_demosaic_jax = bilinear_demosaic_jax(x_jax)
    if n_downsample > 1:
      x_demosaic_jax = lib_image.downsample(x_demosaic_jax, n_downsample)
    return np.array(x_demosaic_jax)

  images = np.stack([processing_fn(im) for im in images], axis=0)

  return images, meta, testscene


def best_fit_affine(x, y, axis):
  """Computes best fit a, b such that a * x + b = y, in a least square sense."""
  x_m = x.mean(axis=axis)
  y_m = y.mean(axis=axis)
  xy_m = (x * y).mean(axis=axis)
  xx_m = (x * x).mean(axis=axis)
  # slope a = Cov(x, y) / Cov(x, x).
  a = (xy_m - x_m * y_m) / (xx_m - x_m * x_m)
  b = y_m - a * x_m
  return a, b  # pytype: disable=bad-return-type  # jax-ndarray


def match_images_affine(
    est, gt, axis = (0, 1)
):
  """Computes affine best fit of gt->est, then maps est back to match gt."""
  # Mapping is computed gt->est to be robust since `est` may be very noisy.
  a, b = best_fit_affine(gt, est, axis=axis)
  # Inverse mapping back to gt ensures we use a consistent space for metrics.
  est_matched = (est - b) / a
  return est_matched
