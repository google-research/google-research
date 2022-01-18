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
"""Utility functions."""
import collections
import os
from os import path
from absl import flags
import flax
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from PIL import Image
import yaml
from jaxnerf.nerf import datasets

BASE_DIR = "jaxnerf"
INTERNAL = False


@flax.struct.dataclass
class TrainState:
  optimizer: flax.optim.Optimizer


@flax.struct.dataclass
class Stats:
  loss: float
  psnr: float
  loss_c: float
  psnr_c: float
  weight_l2: float


Rays = collections.namedtuple("Rays", ("origins", "directions", "viewdirs"))


def namedtuple_map(fn, tup):
  """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
  return type(tup)(*map(fn, tup))


def define_flags():
  """Define flags for both training and evaluation modes."""
  flags.DEFINE_string("train_dir", None, "where to store ckpts and logs")
  flags.DEFINE_string("data_dir", None, "input data directory.")
  flags.DEFINE_string("config", None,
                      "using config files to set hyperparameters.")

  # Dataset Flags
  # TODO(pratuls): rename to dataset_loader and consider cleaning up
  flags.DEFINE_enum("dataset", "blender",
                    list(k for k in datasets.dataset_dict.keys()),
                    "The type of dataset feed to nerf.")
  flags.DEFINE_enum(
      "batching", "single_image", ["single_image", "all_images"],
      "source of ray sampling when collecting training batch,"
      "single_image for sampling from only one image in a batch,"
      "all_images for sampling from all the training images.")
  flags.DEFINE_bool(
      "white_bkgd", True, "using white color as default background."
      "(used in the blender dataset only)")
  flags.DEFINE_integer("batch_size", 1024,
                       "the number of rays in a mini-batch (for training).")
  flags.DEFINE_integer("factor", 4,
                       "the downsample factor of images, 0 for no downsample.")
  flags.DEFINE_bool("spherify", False, "set for spherical 360 scenes.")
  flags.DEFINE_bool(
      "render_path", False, "render generated path if set true."
      "(used in the llff dataset only)")
  flags.DEFINE_integer(
      "llffhold", 8, "will take every 1/N images as LLFF test set."
      "(used in the llff dataset only)")
  flags.DEFINE_bool(
      "use_pixel_centers", False,
      "If True, generate rays through the center of each pixel. Note: While "
      "this is the correct way to handle rays, it is not the way rays are "
      "handled in the original NeRF paper. Setting this TRUE yields ~ +1 PSNR "
      "compared to Vanilla NeRF.")

  # Model Flags
  flags.DEFINE_string("model", "nerf", "name of model to use.")
  flags.DEFINE_float("near", 2., "near clip of volumetric rendering.")
  flags.DEFINE_float("far", 6., "far clip of volumentric rendering.")
  flags.DEFINE_integer("net_depth", 8, "depth of the first part of MLP.")
  flags.DEFINE_integer("net_width", 256, "width of the first part of MLP.")
  flags.DEFINE_integer("net_depth_condition", 1,
                       "depth of the second part of MLP.")
  flags.DEFINE_integer("net_width_condition", 128,
                       "width of the second part of MLP.")
  flags.DEFINE_float("weight_decay_mult", 0, "The multiplier on weight decay")
  flags.DEFINE_integer(
      "skip_layer", 4, "add a skip connection to the output vector of every"
      "skip_layer layers.")
  flags.DEFINE_integer("num_rgb_channels", 3, "the number of RGB channels.")
  flags.DEFINE_integer("num_sigma_channels", 1,
                       "the number of density channels.")
  flags.DEFINE_bool("randomized", True, "use randomized stratified sampling.")
  flags.DEFINE_integer("min_deg_point", 0,
                       "Minimum degree of positional encoding for points.")
  flags.DEFINE_integer("max_deg_point", 10,
                       "Maximum degree of positional encoding for points.")
  flags.DEFINE_integer("deg_view", 4,
                       "Degree of positional encoding for viewdirs.")
  flags.DEFINE_integer(
      "num_coarse_samples", 64,
      "the number of samples on each ray for the coarse model.")
  flags.DEFINE_integer("num_fine_samples", 128,
                       "the number of samples on each ray for the fine model.")
  flags.DEFINE_bool("use_viewdirs", True, "use view directions as a condition.")
  flags.DEFINE_float(
      "noise_std", None, "std dev of noise added to regularize sigma output."
      "(used in the llff dataset only)")
  flags.DEFINE_bool("lindisp", False,
                    "sampling linearly in disparity rather than depth.")
  flags.DEFINE_string("net_activation", "relu",
                      "activation function used within the MLP.")
  flags.DEFINE_string("rgb_activation", "sigmoid",
                      "activation function used to produce RGB.")
  flags.DEFINE_string("sigma_activation", "relu",
                      "activation function used to produce density.")
  flags.DEFINE_bool(
      "legacy_posenc_order", False,
      "If True, revert the positional encoding feature order to an older version of this codebase."
  )

  # Train Flags
  flags.DEFINE_float("lr_init", 5e-4, "The initial learning rate.")
  flags.DEFINE_float("lr_final", 5e-6, "The final learning rate.")
  flags.DEFINE_integer(
      "lr_delay_steps", 0, "The number of steps at the beginning of "
      "training to reduce the learning rate by lr_delay_mult")
  flags.DEFINE_float(
      "lr_delay_mult", 1., "A multiplier on the learning rate when the step "
      "is < lr_delay_steps")
  flags.DEFINE_float("grad_max_norm", 0.,
                     "The gradient clipping magnitude (disabled if == 0).")
  flags.DEFINE_float("grad_max_val", 0.,
                     "The gradient clipping value (disabled if == 0).")

  flags.DEFINE_integer("max_steps", 1000000,
                       "the number of optimization steps.")
  flags.DEFINE_integer("save_every", 10000,
                       "the number of steps to save a checkpoint.")
  flags.DEFINE_integer("print_every", 100,
                       "the number of steps between reports to tensorboard.")
  flags.DEFINE_integer(
      "render_every", 5000, "the number of steps to render a test image,"
      "better to be x00 for accurate step time record.")
  flags.DEFINE_integer("gc_every", 10000,
                       "the number of steps to run python garbage collection.")

  # Eval Flags
  flags.DEFINE_bool(
      "eval_once", True,
      "evaluate the model only once if true, otherwise keeping evaluating new"
      "checkpoints if there's any.")
  flags.DEFINE_bool("save_output", True,
                    "save predicted images to disk if True.")
  flags.DEFINE_integer(
      "chunk", 8192,
      "the size of chunks for evaluation inferences, set to the value that"
      "fits your GPU/TPU memory.")


def update_flags(args):
  """Update the flags in `args` with the contents of the config YAML file."""
  pth = path.join(BASE_DIR, args.config + ".yaml")
  with open_file(pth, "r") as fin:
    configs = yaml.load(fin, Loader=yaml.FullLoader)
  # Only allow args to be updated if they already exist.
  invalid_args = list(set(configs.keys()) - set(dir(args)))
  if invalid_args:
    raise ValueError(f"Invalid args {invalid_args} in {pth}.")
  args.__dict__.update(configs)


def open_file(pth, mode="r"):
  if not INTERNAL:
    return open(pth, mode=mode)


def file_exists(pth):
  if not INTERNAL:
    return path.exists(pth)


def listdir(pth):
  if not INTERNAL:
    return os.listdir(pth)


def isdir(pth):
  if not INTERNAL:
    return path.isdir(pth)


def makedirs(pth):
  if not INTERNAL:
    os.makedirs(pth)


def render_image(render_fn, rays, rng, normalize_disp, chunk=8192):
  """Render all the pixels of an image (in test mode).

  Args:
    render_fn: function, jit-ed render function.
    rays: a `Rays` namedtuple, the rays to be rendered.
    rng: jnp.ndarray, random number generator (used in training mode only).
    normalize_disp: bool, if true then normalize `disp` to [0, 1].
    chunk: int, the size of chunks to render sequentially.

  Returns:
    rgb: jnp.ndarray, rendered color image.
    disp: jnp.ndarray, rendered disparity image.
    acc: jnp.ndarray, rendered accumulated weights per pixel.
  """
  height, width = rays[0].shape[:2]
  num_rays = height * width
  rays = namedtuple_map(lambda r: r.reshape((num_rays, -1)), rays)

  unused_rng, key_0, key_1 = jax.random.split(rng, 3)
  host_id = jax.host_id()
  results = []
  for i in range(0, num_rays, chunk):
    # pylint: disable=cell-var-from-loop
    chunk_rays = namedtuple_map(lambda r: r[i:i + chunk], rays)
    chunk_size = chunk_rays[0].shape[0]
    rays_remaining = chunk_size % jax.device_count()
    if rays_remaining != 0:
      padding = jax.device_count() - rays_remaining
      chunk_rays = namedtuple_map(
          lambda r: jnp.pad(r, ((0, padding), (0, 0)), mode="edge"), chunk_rays)
    else:
      padding = 0
    # After padding the number of chunk_rays is always divisible by
    # host_count.
    rays_per_host = chunk_rays[0].shape[0] // jax.host_count()
    start, stop = host_id * rays_per_host, (host_id + 1) * rays_per_host
    chunk_rays = namedtuple_map(lambda r: shard(r[start:stop]), chunk_rays)
    chunk_results = render_fn(key_0, key_1, chunk_rays)[-1]
    results.append([unshard(x[0], padding) for x in chunk_results])
    # pylint: enable=cell-var-from-loop
  rgb, disp, acc = [jnp.concatenate(r, axis=0) for r in zip(*results)]
  # Normalize disp for visualization for ndc_rays in llff front-facing scenes.
  if normalize_disp:
    disp = (disp - disp.min()) / (disp.max() - disp.min())
  return (rgb.reshape((height, width, -1)), disp.reshape(
      (height, width, -1)), acc.reshape((height, width, -1)))


def compute_psnr(mse):
  """Compute psnr value given mse (we assume the maximum pixel value is 1).

  Args:
    mse: float, mean square error of pixels.

  Returns:
    psnr: float, the psnr value.
  """
  return -10. * jnp.log(mse) / jnp.log(10.)


def compute_ssim(img0,
                 img1,
                 max_val,
                 filter_size=11,
                 filter_sigma=1.5,
                 k1=0.01,
                 k2=0.03,
                 return_map=False):
  """Computes SSIM from two images.

  This function was modeled after tf.image.ssim, and should produce comparable
  output.

  Args:
    img0: array. An image of size [..., width, height, num_channels].
    img1: array. An image of size [..., width, height, num_channels].
    max_val: float > 0. The maximum magnitude that `img0` or `img1` can have.
    filter_size: int >= 1. Window size.
    filter_sigma: float > 0. The bandwidth of the Gaussian used for filtering.
    k1: float > 0. One of the SSIM dampening parameters.
    k2: float > 0. One of the SSIM dampening parameters.
    return_map: Bool. If True, will cause the per-pixel SSIM "map" to returned

  Returns:
    Each image's mean SSIM, or a tensor of individual values if `return_map`.
  """
  # Construct a 1D Gaussian blur filter.
  hw = filter_size // 2
  shift = (2 * hw - filter_size + 1) / 2
  f_i = ((jnp.arange(filter_size) - hw + shift) / filter_sigma)**2
  filt = jnp.exp(-0.5 * f_i)
  filt /= jnp.sum(filt)

  # Blur in x and y (faster than the 2D convolution).
  filt_fn1 = lambda z: jsp.signal.convolve2d(z, filt[:, None], mode="valid")
  filt_fn2 = lambda z: jsp.signal.convolve2d(z, filt[None, :], mode="valid")

  # Vmap the blurs to the tensor size, and then compose them.
  num_dims = len(img0.shape)
  map_axes = tuple(list(range(num_dims - 3)) + [num_dims - 1])
  for d in map_axes:
    filt_fn1 = jax.vmap(filt_fn1, in_axes=d, out_axes=d)
    filt_fn2 = jax.vmap(filt_fn2, in_axes=d, out_axes=d)
  filt_fn = lambda z: filt_fn1(filt_fn2(z))

  mu0 = filt_fn(img0)
  mu1 = filt_fn(img1)
  mu00 = mu0 * mu0
  mu11 = mu1 * mu1
  mu01 = mu0 * mu1
  sigma00 = filt_fn(img0**2) - mu00
  sigma11 = filt_fn(img1**2) - mu11
  sigma01 = filt_fn(img0 * img1) - mu01

  # Clip the variances and covariances to valid values.
  # Variance must be non-negative:
  sigma00 = jnp.maximum(0., sigma00)
  sigma11 = jnp.maximum(0., sigma11)
  sigma01 = jnp.sign(sigma01) * jnp.minimum(
      jnp.sqrt(sigma00 * sigma11), jnp.abs(sigma01))

  c1 = (k1 * max_val)**2
  c2 = (k2 * max_val)**2
  numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
  denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
  ssim_map = numer / denom
  ssim = jnp.mean(ssim_map, list(range(num_dims - 3, num_dims)))
  return ssim_map if return_map else ssim


def save_img(img, pth):
  """Save an image to disk.

  Args:
    img: jnp.ndarry, [height, width, channels], img will be clipped to [0, 1]
      before saved to pth.
    pth: string, path to save the image to.
  """
  with open_file(pth, "wb") as imgout:
    Image.fromarray(np.array(
        (np.clip(img, 0., 1.) * 255.).astype(jnp.uint8))).save(imgout, "PNG")


def learning_rate_decay(step,
                        lr_init,
                        lr_final,
                        max_steps,
                        lr_delay_steps=0,
                        lr_delay_mult=1):
  """Continuous learning rate decay function.

  The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
  is log-linearly interpolated elsewhere (equivalent to exponential decay).
  If lr_delay_steps>0 then the learning rate will be scaled by some smooth
  function of lr_delay_mult, such that the initial learning rate is
  lr_init*lr_delay_mult at the beginning of optimization but will be eased back
  to the normal learning rate when steps>lr_delay_steps.

  Args:
    step: int, the current optimization step.
    lr_init: float, the initial learning rate.
    lr_final: float, the final learning rate.
    max_steps: int, the number of steps during optimization.
    lr_delay_steps: int, the number of steps to delay the full learning rate.
    lr_delay_mult: float, the multiplier on the rate when delaying it.

  Returns:
    lr: the learning for current step 'step'.
  """
  if lr_delay_steps > 0:
    # A kind of reverse cosine decay.
    delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
        0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1))
  else:
    delay_rate = 1.
  t = np.clip(step / max_steps, 0, 1)
  log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
  return delay_rate * log_lerp


def shard(xs):
  """Split data into shards for multiple devices along the first dimension."""
  return jax.tree_map(
      lambda x: x.reshape((jax.local_device_count(), -1) + x.shape[1:]), xs)


def to_device(xs):
  """Transfer data to devices (GPU/TPU)."""
  return jax.tree_map(jnp.array, xs)


def unshard(x, padding=0):
  """Collect the sharded tensor to the shape before sharding."""
  y = x.reshape([x.shape[0] * x.shape[1]] + list(x.shape[2:]))
  if padding > 0:
    y = y[:-padding]
  return y
