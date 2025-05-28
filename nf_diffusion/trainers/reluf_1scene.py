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

"""Training ReLU Field with SRN dataset for ONE SCENE."""

import functools
import os
from typing import Any, Callable, Dict, Optional, Tuple, Union

from absl import logging
from clu import metrics
from clu import parameter_overview
import flax
import flax.jax_utils as flax_utils
import flax.linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import mediapy as media
import ml_collections
import numpy as np
import optax
import tensorflow as tf

from nf_diffusion.models.utils import instant_ngp_utils
from nf_diffusion.trainers.metrics import image as image_metrics
from nf_diffusion.trainers.utils import trainer_utils as utils

Metric = metrics.Metric


def lr_fn(step, max_steps, lr0, lr1, lr_delay_steps=100, lr_delay_mult=0.1):
  """Learning rate schedule from instant_NGP."""

  def log_lerp(t, v0, v1):
    """Interpolate log-linearly from `v0` (t=0) to `v1` (t=1)."""
    if v0 <= 0 or v1 <= 0:
      raise ValueError(f"Interpolants {v0} and {v1} must be positive.")
    lv0 = jnp.log(v0)
    lv1 = jnp.log(v1)
    return jnp.exp(jnp.clip(t, 0, 1) * (lv1 - lv0) + lv0)

  if lr_delay_steps > 0:
    # A kind of reverse cosine decay.
    delay_rate = lr_delay_mult + (1 - lr_delay_mult) * jnp.sin(
        0.5 * jnp.pi * jnp.clip(step / lr_delay_steps, 0, 1)
    )
  else:
    delay_rate = 1.0
  return delay_rate * log_lerp(step / max_steps, lr0, lr1)


def get_learning_rate_scheduler(
    config, data_info
):
  return functools.partial(
      lr_fn,
      max_steps=data_info.num_train_steps,
      lr0=config.opt.lr0,
      lr1=config.opt.lr1,
  )


def create_train_state(
    config,
    rng,
    data_info = None,  # data_info
):
  """Create and initialize the model."""
  grid_size = config.model.grid_size
  num_features = config.model.num_features
  features_color = jax.random.uniform(
      rng,
      (grid_size, grid_size, grid_size, num_features - 1),
      dtype=jnp.float32,
  )
  features_color = (features_color - 0.5) * 2.0
  features_color = features_color * config.model.pyramid_init_noise_level
  features_density = jnp.zeros(
      (grid_size, grid_size, grid_size, 1), dtype=jnp.float32
  )
  params = jnp.concatenate([features_density, features_color], axis=-1)
  parameter_overview.log_parameter_overview({"params": params})

  adam_kwargs = dict(config.opt.args)
  adam_kwargs["learning_rate"] = get_learning_rate_scheduler(config, data_info)
  optimizer = optax.adamw(**adam_kwargs)

  return (
      None,
      optimizer,
      train_state.TrainState.create(
          apply_fn=None,
          params=params,
          tx=optimizer,
      ),
  )


@flax.struct.dataclass
class EvalMetrics(metrics.Collection):
  psnr: metrics.Average.from_output("psnr")


@flax.struct.dataclass
class TrainMetrics(metrics.Collection):
  learning_rate: metrics.LastValue.from_output("learning_rate")
  loss: metrics.Average.from_output("loss")
  loss_std: metrics.Std.from_output("loss")
  loss_color_l2: metrics.Average.from_output("loss_color_l2")
  loss_color_huber: metrics.Average.from_output("loss_color_huber")
  loss_density: metrics.Average.from_output("loss_density")
  loss_distortion: metrics.Average.from_output("loss_distortion")
  psnr: metrics.Average.from_output("psnr")


# Losses.
def lossfun_distortion(x, w):
  """Compute iint w_i w_j |x_i - x_j| d_i d_j."""
  # The loss incurred between all pairs of intervals.
  ux = (x[Ellipsis, 1:] + x[Ellipsis, :-1]) / 2
  dux = jnp.abs(ux[Ellipsis, :, None] - ux[Ellipsis, None, :])
  losses_cross = jnp.sum(w * jnp.sum(w[Ellipsis, None, :] * dux, axis=-1), axis=-1)

  # The loss incurred within each individual interval with itself.
  losses_self = jnp.sum(w**2 * (x[Ellipsis, 1:] - x[Ellipsis, :-1]), axis=-1) / 3

  return losses_cross + losses_self


def huber_loss(x, y, delta=0.1):
  abs_errors = jnp.abs(x - y)
  quadratic = jnp.minimum(abs_errors, delta)
  linear = abs_errors - quadratic
  return 0.5 * quadratic**2 + delta * linear


def train_step(
    config,
    unused_model,  # NOT USED
    state,
    unused_opt,
    learning_rate_fn,
    batch,
    rng,
    *unused_args,
    **unused_kwargs,
):
  """Perform a single training step."""
  step = state.step + 1
  lr = learning_rate_fn(step)

  key, rng = jax.random.split(rng)
  rays, pixels = instant_ngp_utils.random_ray_batch(
      key, (config.trainer.per_device_num_rays,), batch
  )

  def loss_fn(vox):
    rgb_est, _, _, coarse_den, _, weights, t = instant_ngp_utils.render_rays(
        rays, vox, rng, config
    )
    loss_color_l2 = jnp.mean(jnp.square(rgb_est - pixels))
    loss_color_huber = jnp.mean(huber_loss(rgb_est, pixels))
    loss_distortion = config.trainer.distortion_loss_strength * jnp.mean(
        lossfun_distortion(t, weights)
    )
    loss_density = config.trainer.density_regularization * jnp.mean(
        jnp.square(coarse_den)
    )
    loss = loss_color_huber + loss_density + loss_distortion
    stats = {
        "loss_color_l2": loss_color_l2,
        "loss_color_huber": loss_color_huber,
        "loss_density": loss_density,
        "loss_distortion": loss_distortion,
        "loss": loss,
    }
    return loss, stats

  # Get gradient function, then evaluate it with current parameters
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, output), grad = grad_fn(state.params)
  if config.get("multi"):
    # Compute average gradient across multiple workers.
    grad = jax.lax.pmean(grad, axis_name="batch")
  state = state.apply_gradients(grads=grad)

  mse = output["loss_color_l2"]
  if config.get("multi"):
    grad = jax.lax.pmean(mse, axis_name="batch")
  psnr = image_metrics.compute_psnr(mse=mse)
  if config.get("multi"):
    stats = {k: jax.lax.pmean(v, axis_name="batch") for k, v in output.items()}
    metrics_update = TrainMetrics.gather_from_model_output(
        **stats, learning_rate=lr, psnr=psnr
    )
  else:
    metrics_update = TrainMetrics.single_from_model_output(
        **output, learning_rate=lr, psnr=psnr
    )
  return state, metrics_update, {}


def evaluate(
    config,
    _,
    pstate,
    eval_ds,
    rng,
    unused_num_eval_steps = -1,
):
  """Evaluate the model on the given dataset."""
  logging.info("Starting evaluation.")
  eval_metrics = None
  state = flax_utils.unreplicate(pstate)

  render_loop = instant_ngp_utils.make_render_loop(state.params, config)
  with utils.StepTraceContextHelper("eval", 0) as trace_context:
    for step, batch in enumerate(eval_ds):  # pytype: disable=wrong-arg-types
      data = jax.tree.map(jnp.asarray, batch)
      render_poses = data["c2w"]
      hwf = data["hwf"]
      rng = jax.random.fold_in(rng, step)

      frames = []
      for pose in render_poses:
        frames.append(
            render_loop(instant_ngp_utils.camera_ray_batch(pose, hwf), rng)[0]
        )
      psnrs_test = [
          -10 * jnp.log10(jnp.mean(jnp.square(rgb - gt)))
          for (rgb, gt) in zip(frames, data["images"])
      ]
      psnr_test = np.array(psnrs_test).mean()
      eval_metrics = EvalMetrics.single_from_model_output(psnr=psnr_test)
    trace_context.next_step()
  eval_info = {
      "out": jnp.concatenate([x[None, Ellipsis] for x in frames], axis=0),
      "gtr": data["images"],
  }
  return eval_metrics, eval_info


def eval_visualize(
    unused_config,
    writer,
    step,
    _,  # model: nn.Module
    unused_state,
    eval_info,
    visdir,
):
  """Eval visualization routine."""
  eval_info = {"eval/{0}".format(key): val for key, val in eval_info.items()}
  writer.write_images(step, eval_info)
  print(eval_info["eval/out"].shape)
  out = jax.device_get(eval_info["eval/out"])
  print(out.shape, out.max(), out.min())
  gtr = jax.device_get(eval_info["eval/gtr"])
  assert out.shape[0] == gtr.shape[0]
  curr_dir = os.path.join(visdir, "step_%d" % step)
  tf.io.gfile.MakeDirs(curr_dir)
  for i in range(out.shape[0]):
    out_fpath = os.path.join(curr_dir, "out_%d.png" % i)
    media.write_image(out_fpath, (out[i] * 255).astype(np.uint8))

    gtr_fpath = os.path.join(curr_dir, "gtr_%d.png" % i)
    media.write_image(gtr_fpath, (gtr[i] * 255).astype(np.uint8))
