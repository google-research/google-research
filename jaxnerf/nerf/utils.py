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

# Lint as: python3
"""Non-differentiable utility functions."""
import os
from os import path
from absl import flags
import flax
from flax import nn
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import yaml
from jaxnerf.nerf import datasets
from jaxnerf.nerf import models

BASE_DIR = "jaxnerf"
INTERNAL = False


@flax.struct.dataclass
class Stats:
  loss: float
  psnr: float


def define_flags():
  """Define flags for both training and evaluation modes."""
  flags.DEFINE_string("train_dir", None, "where to store ckpts and logs")
  flags.DEFINE_string("data_dir", None, "input data directory.")
  flags.DEFINE_string("config", None,
                      "using config files to set hyperparameters.")

  # Dataset Flags
  flags.DEFINE_enum("dataset", "blender",
                    list(k for k in datasets.dataset_dict.keys()),
                    "The type of dataset feed to nerf.")
  flags.DEFINE_bool("image_batching", False,
                    "sample rays in a batch from different images.")
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

  # Model Flags
  flags.DEFINE_enum("model", "nerf", list(k for k in models.model_dict.keys()),
                    "name of model to use.")
  flags.DEFINE_float("near", 2., "near clip of volumetric rendering.")
  flags.DEFINE_float("far", 6., "far clip of volumentric rendering.")
  flags.DEFINE_integer("net_depth", 8, "depth of the first part of MLP.")
  flags.DEFINE_integer("net_width", 256, "width of the first part of MLP.")
  flags.DEFINE_integer("net_depth_condition", 1,
                       "depth of the second part of MLP.")
  flags.DEFINE_integer("net_width_condition", 128,
                       "width of the second part of MLP.")
  flags.DEFINE_integer(
      "skip_layer", 4, "add a skip connection to the output vector of every"
      "skip_layer layers.")
  flags.DEFINE_integer("num_rgb_channels", 3, "the number of RGB channels.")
  flags.DEFINE_integer("num_sigma_channels", 1,
                       "the number of density channels.")
  flags.DEFINE_bool("randomized", True, "use randomized stratified sampling.")
  flags.DEFINE_integer("deg_point", 10,
                       "Degree of positional encoding for points.")
  flags.DEFINE_integer("deg_view", 4,
                       "degree of positional encoding for viewdirs.")
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

  # Train Flags
  flags.DEFINE_float("lr", 5e-4, "Learning rate for training.")
  flags.DEFINE_integer(
      "lr_decay", 500, "the number of steps (in 1000s) for exponential"
      "learning rate decay.")
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
  pth = path.join(BASE_DIR, args.config + ".yaml")
  with open_file(pth, "r") as fin:
    configs = yaml.load(fin, Loader=yaml.FullLoader)
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


def render_image(state, rays, render_fn, rng, normalize_disp, chunk=8192):
  """Render all the pixels of an image (in test mode).

  Args:
    state: model_utils.TrainState.
    rays: a `Rays` namedtuple, the rays to be rendered.
    render_fn: function, jit-ed render function.
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
  rays = datasets.ray_fn(lambda r: r.reshape((num_rays, -1)), rays)

  unused_rng, key_0, key_1 = jax.random.split(rng, 3)
  model = state.optimizer.target
  model_state = state.model_state
  host_id = jax.host_id()
  results = []
  with nn.stateful(model_state, mutable=False):
    for i in range(0, num_rays, chunk):
      # pylint: disable=cell-var-from-loop
      print("  " + "X" * int((i / num_rays) * 78), end="\r")
      chunk_rays = datasets.ray_fn(lambda r: r[i:i + chunk], rays)
      chunk_size = chunk_rays[0].shape[0]
      rays_remaining = chunk_size % jax.device_count()
      rays_per_host = chunk_size // jax.host_count()
      if rays_remaining != 0:
        padding = jax.device_count() - rays_remaining
        chunk_rays = datasets.ray_fn(
            lambda r: jnp.pad(r, ((0, padding), (0, 0)), mode="edge"),
            chunk_rays)
      else:
        padding = 0
      # After padding the number of chunk_rays is always divisible by
      # host_count.
      start, stop = host_id * rays_per_host, (host_id + 1) * rays_per_host
      chunk_rays = datasets.ray_fn(lambda r: shard(r[start:stop]), chunk_rays)
      chunk_results = render_fn(key_0, key_1, model, chunk_rays)[-1]
      results.append([unshard(x, padding) for x in chunk_results])
      # pylint: enable=cell-var-from-loop
    print("")
  rgb, disp, acc = [jnp.concatenate(r, axis=1)[0] for r in zip(*results)]
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


def learning_rate_decay(step, init_lr=5e-4, decay_steps=100000, decay_rate=0.1):
  """Continuous learning rate decay function.

  The computation for learning rate is lr = (init_lr * decay_rate**(step /
  decay_steps))

  Args:
    step: int, the global optimization step.
    init_lr: float, the initial learning rate.
    decay_steps: int, the decay steps, please see the learning rate computation
      above.
    decay_rate: float, the decay rate, please see the learning rate computation
      above.

  Returns:
    lr: the learning for global step 'step'.
  """
  power = step / decay_steps
  return init_lr * (decay_rate**power)


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
