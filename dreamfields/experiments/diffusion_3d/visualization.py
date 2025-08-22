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

"""Visualization utilities."""

import glob
import math
import os
from typing import Any, Iterable, Mapping, Tuple

import matplotlib
matplotlib.use("Agg")
# pylint: disable=g-import-not-at-top
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
import torch
import torchvision
import tqdm.auto as tqdm
# pylint: enable=g-import-not-at-top


def grid(a, **kwargs):
  if a.shape[-1] == 3 or a.shape[-1] == 1:
    a = a.movedim(-1, 1)  # NHWC to NCHW.
  a = torchvision.utils.make_grid(a, **kwargs)  # NCHW to CH'W'.
  return a.movedim(0, 2)  # Return in HWC order.


@torch.no_grad()
def viz_grid(a, **kwargs):
  g = grid(a.detach(), **kwargs)
  return g.clamp(0., 1.).cpu()


def get_arrays(history,
               keys):
  keys = [k for k in keys if (k and k in history[0].keys())]
  mapping = {}
  for k in keys:
    iters, vals = zip(*[(m["iteration"], m[k]) for m in history if k in m])
    iters = np.array(iters)
    vals = np.stack(vals)
    mapping[k] = (iters, vals)
  return mapping


def plot_keys(metric_series,
              output_dir, filename):
  """Plots and saves metrics as image given a dictionary of series."""
  filename = os.path.join(output_dir, filename)
  with torch.inference_mode():
    fig, axes = plt.subplots(
        3, math.ceil(len(metric_series) / 3), figsize=(len(metric_series), 9))
    for i, (plot_key, (loss_iters,
                       loss_values)) in enumerate(metric_series.items()):
      ax = axes.ravel()[i]
      if "eikonal" in plot_key or "mse" in plot_key or plot_key in [
          "diffusion/diffusion_loss", "loss/total_loss"
      ]:
        ax.semilogy(loss_iters, loss_values)
      elif loss_values.ndim == 2:
        # Plot individual series, e.g. for cameras.
        ax.plot(loss_iters[:, None], loss_values)
      else:
        ax.plot(loss_iters, loss_values)
      ax.set_title(plot_key)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)


def plot_images(visualizations, iteration,
                output_dir, filename, plot_config):
  """Plot images in a grid."""
  filename = os.path.join(output_dir, filename)

  # Configure subplot dimensions.
  ncol = 2
  nrow = len(plot_config) / ncol
  if nrow > int(nrow):
    nrow = int(nrow) + 1
  else:
    nrow = int(nrow)

  h, w = next(iter(visualizations.values())).shape[:2]
  aspect = w / h

  fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.5 * aspect, nrow * 3))
  for ax, (key, title, plot_kwargs) in zip(axes.ravel(), plot_config):
    im = visualizations[key]
    if im.ndim == 3 and im.shape[-1] == 1:
      # Remove channel if of shape 1, so colormapping works.
      im = im.squeeze(-1)

    # Color map image manually.
    if "cmap" in plot_kwargs:
      cm = plt.get_cmap(plot_kwargs["cmap"])
      im = cm(im.mean(axis=-1))[Ellipsis, :3]
      plot_kwargs = {k: v for k, v in plot_kwargs.items() if k != "cmap"}

    ax.imshow(im, **plot_kwargs)
    ax.set_title(title)
  fig.suptitle(f"Iteration {iteration:5}", x=0.48, y=0.98, fontweight="bold")
  fig.tight_layout()
  fig.savefig(filename, bbox_inches="tight")
  plt.close(fig)


def make_optimization_video(directory, pattern="im_*.png", return_frames=False):
  """Concatenate images into a video."""
  paths = glob.glob(os.path.join(directory, pattern))
  paths.sort(
      key=lambda fname: int(os.path.basename(fname)[len("im_"):-len(".png")]))
  images = [media.read_image(path) for path in tqdm.tqdm(paths)]
  if images:
    # Pad frames to the same shape.
    heights, widths = zip(*[im.shape[:2] for im in images])
    max_height, max_width = max(heights), max(widths)
    for i, im in enumerate(images):
      assert im.ndim == 3
      pad_height = max_height - im.shape[0]
      pad_width = max_width - im.shape[1]
      if pad_height != 0 or pad_width != 0:
        images[i] = np.pad(im, ((0, pad_height), (0, pad_width), (0, 0)))

    # Stack and write video.
    images = np.stack(images)[Ellipsis, :3]

    if return_frames:
      return images

    media.write_video(
        os.path.join(directory, "optimization.mp4"), images, fps=30)
  else:
    print(
        f"WARNING: No images to combine in {directory} for optimization history."
    )


def write_video(video_frames, output_dir, filename, pbar=None, fps=30):
  """Write video to disk."""
  filename = os.path.join(output_dir, filename)
  media.write_video(filename, video_frames, fps=fps)
  if pbar:
    pbar.write(f"Wrote {filename}.")
