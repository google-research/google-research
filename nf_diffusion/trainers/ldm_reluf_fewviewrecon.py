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
from clu import checkpoint
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

import nf_diffusion.models.ldm_reluf as ldm_model
from nf_diffusion.models.utils import reluf_utils
from nf_diffusion.trainers import vdm
from nf_diffusion.trainers.metrics import image as image_metrics
from nf_diffusion.trainers.reluf_1scene_coarse2fine import huber_loss
from nf_diffusion.trainers.reluf_1scene_coarse2fine import lossfun_distortion
from nf_diffusion.trainers.utils import trainer_utils as utils


get_opt = vdm.get_opt


def get_learning_rate_scheduler(
    config,
    unused_data_info,
):
  return lambda _: config.opt.learning_rate


def create_train_state(
    config, rng, data_info
):
  """Initialize training state."""
  logging.warning("=== Initializing model ===")
  model = ldm_model.LDM(**config.model)

  # Get initialization voxel data for input to the diffusion model
  resolution = config.data.get("vox_res", 64)
  channels = config.data.get("vox_dim", 4)
  if config.data.get("sigma_only", False):
    channels = 1
  batch_size = config.data.per_device_batch_size
  cond = jnp.ones((batch_size,), "int32")

  inputs = {
      "vox": jnp.zeros(
          (batch_size, resolution, resolution, resolution, channels), "float32"
      ),
      "cond": cond,
  }

  rng1, rng2 = jax.random.split(rng)
  variables = model.init({"params": rng1, "sample": rng2}, **inputs)
  parameter_overview.log_parameter_overview(variables)
  mutable_state, params = variables.pop("params")

  # Whether load pretrained models
  if config.get("resume", None) is not None:
    if config.resume.get("score_model_den", None) is not None:
      den_cfg = config.resume.score_model_den
      logging.info("=== Resume density score model from: %s ===", den_cfg.path)
      all_params = checkpoint.load_state_dict(den_cfg.path)
      all_params = all_params["params"]
      assert all_params is not None
      score_model_den_params = all_params
      for k in den_cfg.param_keys:
        score_model_den_params = score_model_den_params[k]
      params = flax.core.frozen_dict.unfreeze(params)
      params["score_model_den"] = score_model_den_params
      params = flax.core.frozen_dict.freeze(params)

  logging.info("=== Initializing train state (optimizer)===")
  opt = get_opt(config, data_info)
  opt_state = opt.init(params)
  state = vdm.TrainState.create(
      params=params, mutable_state=mutable_state, opt_state=opt_state
  )

  return model, opt, state


Metric = metrics.Metric


@flax.struct.dataclass
class TrainMetrics(metrics.Collection):
  learning_rate: metrics.LastValue.from_output("learning_rate")
  loss: metrics.Average.from_output("loss")
  loss_std: metrics.Std.from_output("loss")
  loss_color_l2: metrics.Average.from_output("loss_color_l2")
  loss_color_huber: metrics.Average.from_output("loss_color_huber")
  loss_distortion: metrics.Average.from_output("loss_distortion")
  psnr: metrics.Average.from_output("psnr")
  ssim: metrics.Average.from_output("ssim")


# Direct gradients using volume rendering.
# @jax.jit
def make_direct_recon_guidance(config):
  """Return function that computes the gradient of directly guide the voxel."""
  render_config = config.render_config
  distortion_loss_strength = config.tester.distortion_loss_strength

  def guidance(z_t, ro, rd, rgb, rng):
    """Compute gradient of reconstruction guidance on input [z_t]."""

    def loss_fn(vox):
      rgb_est, _, _, _, _, weights, t = reluf_utils.render_rays(
          (ro, rd), vox[0], rng, render_config
      )
      loss_color_l2 = jnp.mean(jnp.square(rgb_est - rgb))
      loss_color_huber = jnp.mean(huber_loss(rgb_est, rgb))
      loss_distortion = distortion_loss_strength * jnp.mean(
          lossfun_distortion(t, weights)
      )
      loss = loss_color_huber + loss_distortion
      stats = {
          "loss_color_l2": loss_color_l2,
          "loss_color_huber": loss_color_huber,
          "loss_distortion": loss_distortion,
          "loss": loss,
      }
      return loss, stats

    return jax.value_and_grad(loss_fn, has_aux=True)(z_t)

  return guidance


def make_denoise_step(model, variables):
  """DDPM denoising step."""

  def step(z_t, t, cond, d, rng):
    """Entirely denoise a voxel with time t using DDPM.

    Args:
      z_t: (..., 1, 64, 64, 64, 4), jnp.float32
      t: int, range [0, model.timesteps], t=0 means it's clean.
      cond: (..., 1), jnp.int32
      d: int, step size.
      rng: Random keys.

    Returns:
      z_t: cleaned up voxel.
      s: the cleaned-up voxel's noise.
    """
    return model.apply(
        variables,
        i=(model.timesteps - t),
        d=d,
        T=model.timesteps,
        z_t=z_t,
        cond=cond,
        rng=rng,
        mutable=False,
        method=model.sample_step_i_d,
    )

  return step


# Reconstruction guidance : denoised the image then compute the guidance
def make_recon_guidance(model, variables, config):
  """Return function that denoise z0 and denormalize it."""

  def get_z0_denormed(z_t, t, cond, rng):
    """Entirely denoise a voxel with time t using DDPM.

    Args:
      z_t: (1, res, res, res, 4).
      t: int (from 0 to model.timesteps), NOTE: t=0 means it's super clean.
      cond: (1,) tensor, jnp.int32
      rng: random key.

    Returns:
      z_0 (1, res, res, res, 4)
    """
    z0 = model.apply(
        variables,
        i=(model.timesteps - t),
        d=t,
        T=model.timesteps,
        z_t=z_t,
        cond=cond,
        rng=rng,
        mutable=False,
        method=model.sample_step_i_d,
    )[0]
    z0 = model.apply(variables, z0, method=model.rescale_z0)
    return model.denormalize_vox(z0)

  direct_recon_guidance = make_direct_recon_guidance(config)

  def denoise_recon_guidance(z_t, cond, t, ro, rd, rgb, rng):
    z_0 = get_z0_denormed(z_t, t, cond, rng)
    return direct_recon_guidance(z_0, ro, rd, rgb, rng)

  return denoise_recon_guidance


def train_step(
    config,
    unused_model,  # NOT USED
    state,
    unused_opt,
    learning_rate_fn,
    batch,
    rng,
    *unused_args,
    **unused_kwargs
):
  """Perform a single training step.

  Args:
    config: the configuration of the training, passed in to configurate train
      step.
    unused_model: Flax module for the model. The apply method must take input
      images and a boolean argument indicating whether to use training or
      inference mode.
    state: State of the model (optimizer and state).
    learning_rate_fn: Function that computes the learning rate given the step
      number.
    batch: Training inputs for this step.
    rng: random key.

  Returns:
    The new model state and dictionary with metrics.
  """
  step = state.step + 1
  lr = learning_rate_fn(step)

  # Get rays and colors
  rays, pixels = batch
  rays = (rays["ray_origins"], rays["ray_directions"])
  pixels = pixels["ray_colors"]

  def loss_fn(vox):
    rgb_est, _, _, _, _, weights, t = reluf_utils.render_rays(
        rays, vox, rng, config
    )
    loss_color_l2 = jnp.mean(jnp.square(rgb_est - pixels))
    loss_color_huber = jnp.mean(huber_loss(rgb_est, pixels))
    loss_distortion = config.trainer.distortion_loss_strength * jnp.mean(
        lossfun_distortion(t, weights)
    )
    loss = loss_color_huber + loss_distortion
    stats = {
        "loss_color_l2": loss_color_l2,
        "loss_color_huber": loss_color_huber,
        "loss_distortion": loss_distortion,
        "loss": loss,
    }
    return loss, stats

  # Get gradient function, then evaluate it with current parameters
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, output), grad = grad_fn(state.params)
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


@flax.struct.dataclass
class EvalMetrics(metrics.Collection):
  psnr: metrics.Average.from_output("psnr")
  ssim: metrics.Average.from_output("ssim")


def evaluate(
    config,
    _,
    state,  # NOTE: this is replicated
    eval_ds,
    rng,
    num_eval_steps = -1,
):
  """Evaluate the model on the given dataset."""
  logging.info("Starting evaluation.")
  eval_metrics = None
  state = flax_utils.unreplicate(state)

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
          img0=frame, img1=gt, max_val=1
      )
      eval_metrics_curr = EvalMetrics.single_from_model_output(
          psnr=curr_psnr_test, ssim=curr_ssim_test
      )
      if eval_metrics is None:
        eval_metrics = eval_metrics_curr
      else:
        eval_metrics.merge(eval_metrics_curr)

      if step > num_eval_steps:
        break
      trace_context.next_step()
  eval_info = {
      "out": np.array(
          jax.device_get(
              jnp.concatenate([x[None, Ellipsis] for x in frames], axis=0)
          )
      ),
      "depths": np.array(
          jax.device_get(
              jnp.concatenate([x[None, Ellipsis] for x in depths], axis=0)
          )
      ),
      "alphas": np.array(
          jax.device_get(
              jnp.concatenate([x[None, Ellipsis] for x in alphas], axis=0)
          )
      ),
      "gtr": np.array(jax.device_get(jnp.concatenate(gtrlst, axis=0))),
  }
  return eval_metrics, eval_info
