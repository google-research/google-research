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

"""Function to render a video for the predicted layers of the video.

This script also renders different combination of the foreground layers with
background layer.
"""

import functools
import os
from typing import Dict

from absl import logging

from clu import metrics
import flax
import flax.jax_utils as flax_utils
import flax.linen as nn
from flax.training import checkpoints
from flax.training import train_state
import jax
import jax.numpy as jnp
import mediapy
import ml_collections
import numpy as np
import tensorflow as tf

from omnimatte3D.src import datasets
from omnimatte3D.src import models
from omnimatte3D.src.utils import file_utils
from omnimatte3D.src.utils import train_utils


def eval_step(
    model,
    state,
    batch,
):
  """Compute the metrics for the given model in inference mode.

  The model is applied to the inputs with train=False using all devices on the
  host. Afterwards metrics are averaged across *all* devices (of all hosts).

  Args:
    model: Flax module for the model. The apply method must take input images
      and a boolean argument indicating whether to use training or inference
      mode.
    state: Replicate model state.
    batch: Inputs that should be evaluated.

  Returns:
    Dictionary of the replicated metrics.
  """
  logging.info("eval_step(batch=%s)", batch)
  variables = {"params": state.params}

  out = model.apply(variables, batch)
  return out


def evaluate(config, workdir):
  """Evalution function."""

  # Hide the GPUs and TPUs from TF so it does not reserve memory on them for
  # LPIPS computation or dataset loading.
  tf.config.experimental.set_visible_devices([], "GPU")
  tf.config.experimental.set_visible_devices([], "TPU")

  rng = jax.random.PRNGKey(config.seed)

  # Currently only supports single image batch eval.
  config.dataset.batch_size = 1
  assert (
      jax.local_device_count() == 1
  ), "Currently does not support multi device training."

  # ----------------------------------------------------------------------------
  # Build input pipeline.
  rng, data_rng = jax.random.split(rng)
  data_rng = jax.random.fold_in(data_rng, jax.process_index())

  train_iter, eval_iter, _ = datasets.create_datasets(config)
  eval_ds = eval_iter
  example_batch = train_iter.peek()
  train_iter = flax.jax_utils.prefetch_to_device(train_iter, 6)

  config.dataset.image_height = example_batch["src_rgb"].shape[-3]
  config.dataset.image_width = example_batch["src_rgb"].shape[-2]
  # ----------------------------------------------------------------------------
  # Initialize model.
  learning_rate_fn = train_utils.create_learning_rate_fn(config)
  rng, model_rng = jax.random.split(rng)
  model, state, _ = models.create_train_state(
      config,
      model_rng,
      learning_rate_fn=learning_rate_fn,
      example_batch=example_batch,
  )

  # ----------------------------------------------------------------------------
  # Get the eval step.
  p_eval_step = jax.pmap(
      functools.partial(eval_step, model=model),
      axis_name="batch",
  )

  last_step = 0
  out_dir = os.path.join(workdir, "renders")

  while True:
    state = checkpoints.restore_checkpoint(workdir, state)
    step = int(state.step)

    if step <= last_step:
      # Check if a new checkpoint has been saved. If not, load the latest
      # checkpoint again. Repeat until a new checkpoint is available.
      continue

    if config.eval.save_output and (not file_utils.isdir(out_dir)):
      file_utils.makedirs(out_dir)

    out_dir_step = os.path.join(out_dir, f"{step}")
    if config.eval.save_output and (not file_utils.isdir(out_dir_step)):
      file_utils.makedirs(out_dir_step)

    state = flax_utils.replicate(state)

    video_dict = None
    for idx in range(eval_ds.cardinality()):
      batch = next(eval_iter)
      if config.dev_run:
        if idx == 10:
          break
      logging.info("Evaluating [%d / %d].", idx, eval_ds.cardinality())
      batch = jax.tree.map(np.asarray, batch)
      out_pred = flax_utils.unreplicate(p_eval_step(state=state, batch=batch))

      viz_batch = jax.tree_util.tree_map(lambda x: x[0], batch)
      video_frame_dict = out_pred.get_video_dict(viz_batch)
      # Add time dimension for concat.
      video_frame_dict = jax.tree_util.tree_map(
          lambda x: x[None],
          video_frame_dict,
      )
      if video_dict is None:
        video_dict = video_frame_dict
      else:
        # stack to the current
        for key, val in video_frame_dict.items():
          video_dict[key] = np.concatenate([video_dict[key], val], axis=0)  # pylint: disable=unsupported-assignment-operation

    for key, val in video_dict.items():
      mediapy.write_video(os.path.join(out_dir, f"{key}.mp4"), val, fps=20)
      mediapy.write_video(os.path.join(out_dir_step, f"{key}.mp4"), val, fps=20)
    if config.eval.eval_once:
      break
    if int(step) >= config.train.max_steps:
      break
    last_step = step

  logging.info("Finishing evaluation at step %d", last_step)
