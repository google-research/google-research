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

from typing import Any, Callable, Dict, Tuple, Union

from absl import logging
from clu import metrics
import flax
import flax.jax_utils as flax_utils
from flax.training import train_state
import jax
import jax.numpy as jnp
import mediapy as media
import ml_collections
import numpy as np
import tensorflow as tf

from nf_diffusion.models.utils import reluf_utils
from nf_diffusion.trainers.metrics import image as image_metrics
from nf_diffusion.trainers.reluf_1scene_coarse2fine import huber_loss
from nf_diffusion.trainers.reluf_1scene_coarse2fine import lossfun_distortion
from nf_diffusion.trainers.utils import trainer_utils as utils


Metric = metrics.Metric


@flax.struct.dataclass
class EvalMetrics(metrics.Collection):
  psnr: metrics.Average.from_output("psnr")
  ssim: metrics.Average.from_output("ssim")


@flax.struct.dataclass
class TrainMetrics(metrics.Collection):
  learning_rate: metrics.LastValue.from_output("learning_rate")
  loss: metrics.Average.from_output("loss")
  loss_std: metrics.Std.from_output("loss")
  loss_color_l2: metrics.Average.from_output("loss_color_l2")
  loss_color_huber: metrics.Average.from_output("loss_color_huber")
  loss_distortion: metrics.Average.from_output("loss_distortion")
  loss_density: metrics.Average.from_output("loss_density")
  loss_den_ttv: metrics.Average.from_output("loss_den_ttv")
  loss_color_ttv: metrics.Average.from_output("loss_color_ttv")
  psnr: metrics.Average.from_output("psnr")


def cauchy_density_regularization(vox, preconditioner, density_offset,
                                  density_c):
  sig = vox[Ellipsis, 0] * preconditioner + density_offset
  sig = reluf_utils.safe_exp(sig)
  # return jnp.log(1 + sig ** 2 / config.trainer.density_c).mean()
  return jnp.log(1 + sig ** 2 / density_c).mean()


def color_total_var_regularization(vox):
  colors = vox[Ellipsis, 1:]
  dx = huber_loss(jnp.roll(colors, 1, axis=-4), colors).mean()
  dy = huber_loss(jnp.roll(colors, 1, axis=-3), colors).mean()
  dz = huber_loss(jnp.roll(colors, 1, axis=-2), colors).mean()
  return (dx + dy + dz) / 3.


def density_total_var_regularization(vox):
  density = vox[Ellipsis, :1]
  dx = huber_loss(jnp.roll(density, 1, axis=-4), density).mean()
  dy = huber_loss(jnp.roll(density, 1, axis=-3), density).mean()
  dz = huber_loss(jnp.roll(density, 1, axis=-2), density).mean()
  return (dx + dy + dz) / 3.


def density_l1_regularization(vox,
                              min_density=-6):
  return huber_loss(
      vox[Ellipsis, 0], jnp.ones_like(vox[Ellipsis, 0]) * min_density).mean()


def binary_density_regularization(vox, max_density=100, min_density=-100):
  sig = vox[Ellipsis, 0]
  sig_max_loss = huber_loss(sig, jnp.ones_like(sig) * max_density)
  sig_min_loss = huber_loss(sig, jnp.ones_like(sig) * min_density)
  return jnp.minimum(sig_min_loss, sig_max_loss).mean()


def color_const_regularization(vox, const=0):
  colors = vox[Ellipsis, 1:]
  return huber_loss(colors, jnp.ones_like(colors) * const).mean()


def make_loss_fn(model, config):
  def loss_fn(vox, rays, pixels, rng):
    rgb_est, _, _, _, _, weights, t = reluf_utils.render_rays(
        rays, vox, rng, config)
    loss_color_l2 = jnp.mean(jnp.square(rgb_est - pixels))
    loss_color_huber = jnp.mean(huber_loss(rgb_est, pixels))
    loss_distortion = config.trainer.distortion_loss_strength * jnp.mean(
        lossfun_distortion(t, weights))

    # Regularization for density
    loss_density = cauchy_density_regularization(
        vox,
        preconditioner=config.model.preconditioner,
        density_offset=config.model.density_offset,
        density_c=config.trainer.density_c
    ) * config.trainer.cauchy_density_regularization_weight
    loss_den_ttv = density_total_var_regularization(
        vox) * config.trainer.density_ttv_weight
    loss_den_l1 = density_l1_regularization(
        vox,
        min_density=config.trainer.get("min_density", -6)
        ) * config.trainer.get("density_l1_weight", 0)
    loss_den_binary = binary_density_regularization(
        vox,
        max_density=config.trainer.get("max_density", 6),
        min_density=config.trainer.get("min_density", -6),
    ) * config.trainer.get("binary_density_weight", 0)

    # Regularization for color
    loss_color_ttv = color_total_var_regularization(
        vox) * config.trainer.color_ttv_weight
    loss_color_const = color_const_regularization(
        vox, const=config.trainer.color_const
    ) * config.trainer.color_const_weight

    loss = (
        loss_distortion
        + loss_density
        + loss_den_ttv
        + loss_den_l1
        + loss_den_binary
        + loss_color_ttv
        + loss_color_huber
        + loss_color_const
    )
    stats = {
        "loss_color_l2": loss_color_l2,
        "loss_color_huber": loss_color_huber,
        "loss_distortion": loss_distortion,
        "loss_density": loss_density,
        "loss_den_ttv": loss_den_ttv,
        "loss_den_l1": loss_den_l1,
        "loss_color_ttv": loss_color_ttv,
        "loss_color_const": loss_color_const,
        "loss": loss
    }
    return loss, stats

  return loss_fn

def train_step(
    config,
    model,  # NOT USED
    state,
    opt,
    learning_rate_fn,
    batch,
    rng,
    *args,
    **kwargs
):
  """Perform a single training step.

  Args:
    config: the configuration of the training, passed in to configurate train
      step.
    model: Flax module for the model. The apply method must take input images
      and a boolean argument indicating whether to use training or inference
      mode.
    state: State of the model (optimizer and state).
    batch: Training inputs for this step.
    learning_rate_fn: Function that computes the learning rate given the step
      number.

  Returns:
    The new model state and dictionary with metrics.
  """
  step = state.step + 1
  lr = learning_rate_fn(step)

  # Get rays and colors
  rays, pixels = batch
  rays = (rays["ray_origins"], rays["ray_directions"])
  pixels = pixels["ray_colors"]

  # Get gradient function, then evaluate it with current parameters
  loss_fn = make_loss_fn(model, config)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, output), grad = grad_fn(state.params, rays, pixels, rng)
  if config.get("multi"):
    # Compute average gradient across multiple workers.
    grad = jax.lax.pmean(grad, axis_name="batch")
  state = state.apply_gradients(grads=grad)

  # Clip
  if config.trainer.get("clip_value", False):
    def clip_value(vox):
      return jnp.concatenate([
          jnp.clip(
              vox[Ellipsis, :1], config.model.den_min, config.model.den_max),
          jnp.clip(
              vox[Ellipsis, 1:], config.model.rgb_min, config.model.rgb_max),
      ], axis=-1)
    state = state.replace(params=clip_value(state.params))

  mse = output["loss_color_l2"]
  if config.get("multi"):
    grad = jax.lax.pmean(mse, axis_name="batch")
  psnr = image_metrics.compute_psnr(mse=mse)
  if config.get("multi"):
    stats = {k: jax.lax.pmean(v, axis_name="batch") for k, v in output.items()}
    metrics_update = TrainMetrics.gather_from_model_output(
        **stats, learning_rate=lr, psnr=psnr)
  else:
    metrics_update = TrainMetrics.single_from_model_output(
        **output, learning_rate=lr, psnr=psnr)
  return state, metrics_update, {}


def evaluate(
    config,
    _,
    pstate,
    eval_ds,
    rng,
    num_eval_steps = -1
):
  """Evaluate the model on the given dataset."""
  logging.info("Starting evaluation.")
  eval_metrics = None
  state = flax_utils.unreplicate(pstate)

  render_loop = reluf_utils.make_render_loop(state.params, config)
  frames = []
  depths = []
  alphas = []
  gtrlst = []
  with utils.StepTraceContextHelper("eval", 0) as trace_context:
    for step, batch in enumerate(eval_ds):  # pytype: disable=wrong-arg-types
      logging.info("Eval step: %d", step)
      data = jax.tree.map(jnp.asarray, batch)
      rays, colors = data
      rays = (rays["ray_origins"], rays["ray_directions"])
      gt = colors["ray_colors"]
      gtrlst.append(gt)

      frame, depth, acc = render_loop(rays, rng)[:3]
      frames.append(frame[0])
      depth_image = jnp.asarray(
          media.to_rgb(
              depth[0, Ellipsis, 0],
              vmin=config.trainer.near,
              vmax=config.trainer.far,
              cmap="jet",
          )
      )
      depths.append(depth_image)
      alphas.append(acc[0])
      curr_psnr_test = image_metrics.compute_psnr(img0=frame, img1=gt)
      curr_ssim_test = image_metrics.compute_ssim(
          img0=frame, img1=gt, max_val=1)
      eval_metrics_curr = EvalMetrics.single_from_model_output(
          psnr=curr_psnr_test, ssim=curr_ssim_test)
      if eval_metrics is None:
        eval_metrics = eval_metrics_curr
      else:
        eval_metrics.merge(eval_metrics_curr)

      if step > num_eval_steps:
        break
      trace_context.next_step()
  eval_info = {
      "out":
          np.array(jax.device_get(
              jnp.concatenate([x[None, Ellipsis] for x in frames], axis=0))),
      "depths":
          np.array(jax.device_get(
              jnp.concatenate([x[None, Ellipsis] for x in depths], axis=0))),
      "alphas":
          np.array(jax.device_get(
              jnp.concatenate([x[None, Ellipsis] for x in alphas], axis=0))),
      "gtr":
          np.array(jax.device_get(jnp.concatenate(gtrlst, axis=0))),
  }
  return eval_metrics, eval_info
