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

"""Methods for training Variational Diffusion Model."""

import copy
import functools
import os
from typing import Any, Dict, Optional, Tuple, Union

from absl import logging
from clu import checkpoint
from clu import metrics
from clu import parameter_overview
import flax
import flax.jax_utils as flax_utils
import flax.linen as nn

import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

import tensorflow as tf
import nf_diffusion.models.ldm_reluf_dual_v2 as ldm_model
from nf_diffusion.trainers import vdm
from nf_diffusion.trainers.utils import image_vis_utils
from nf_diffusion.trainers.utils import reluf_vis_utils
from nf_diffusion.trainers.utils import trainer_utils as utils


get_learning_rate_scheduler = vdm.get_learning_rate_scheduler
ema_apply_gradients = vdm.ema_apply_gradients
get_opt = vdm.get_opt


class TrainState(flax.struct.PyTreeNode):
  """Simple train state for the common case with a single Optax optimizer."""

  step: int
  params: Union[flax.core.frozen_dict.FrozenDict[str, Any], Dict[str, Any]]
  ema_params: Union[flax.core.frozen_dict.FrozenDict[str, Any], Dict[str, Any]]
  mutable_state: Any
  opt_state: optax.OptState

  @classmethod
  def create(cls, *, params, mutable_state, opt_state, **kwargs):
    """Creates a new instance with `step=0` and initialized `opt_state`."""
    ema_params = copy.deepcopy(params)
    return cls(
        step=0,
        params=params,
        ema_params=ema_params,
        opt_state=opt_state,
        mutable_state=mutable_state,
        **kwargs,
    )


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
      "vox":
          jnp.zeros((batch_size, resolution, resolution, resolution, channels),
                    "float32"),
      "cond":
          cond,
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
      resume_params_key = den_cfg.get("resume_params_key", "params")
      all_params = all_params[resume_params_key]
      assert all_params is not None
      score_model_den_params = all_params
      for k in den_cfg.param_keys:
        score_model_den_params = score_model_den_params[k]
      params = flax.core.frozen_dict.unfreeze(params)
      params["score_model_den"] = score_model_den_params
      params = flax.core.frozen_dict.freeze(params)

    if config.resume.get("score_model_rgb", None) is not None:
      rgb_cfg = config.resume.score_model_rgb
      logging.info("=== Resume RGB score model from: %s ===", rgb_cfg.path)
      all_params = checkpoint.load_state_dict(rgb_cfg.path)
      resume_params_key = rgb_cfg.get("resume_params_key", "params")
      all_params = all_params[resume_params_key]
      assert all_params is not None
      score_model_rgb_params = all_params
      for k in rgb_cfg.param_keys:
        score_model_rgb_params = score_model_rgb_params[k]
      params = flax.core.frozen_dict.unfreeze(params)
      params["score_model_rgb"] = score_model_rgb_params
      params = flax.core.frozen_dict.freeze(params)

  logging.info("=== Initializing train state (optimizer)===")
  opt = get_opt(config, data_info)
  opt_state = opt.init(params)
  train_state = TrainState.create(
      params=params,
      mutable_state=mutable_state,
      opt_state=opt_state)

  return model, opt, train_state


@flax.struct.dataclass
class TrainMetrics(metrics.Collection):
  """Triain metrics."""

  learning_rate: metrics.LastValue.from_output("learning_rate")
  loss: metrics.Average.from_output("loss")
  loss_std: metrics.Std.from_output("loss")

  # From loss function
  bpd: metrics.LastValue.from_output("bpd")
  bpd_latent: metrics.LastValue.from_output("bpd_latent")
  bpd_diff: metrics.LastValue.from_output("bpd_diff")

  # From other input
  loss_klz_den: metrics.LastValue.from_output("loss_klz_den")
  loss_klz_rgb: metrics.LastValue.from_output("loss_klz_rgb")
  loss_klz: metrics.LastValue.from_output("loss_klz")
  loss_diff: metrics.LastValue.from_output("loss_diff")
  loss_diff_den: metrics.LastValue.from_output("loss_diff_den")
  loss_diff_rgb: metrics.LastValue.from_output("loss_diff_rgb")
  loss_diff_vis: metrics.LastValue.from_output("loss_diff_vis")
  loss_diff_den_vis: metrics.LastValue.from_output("loss_diff_den_vis")
  loss_diff_rgb_vis: metrics.LastValue.from_output("loss_diff_rgb_vis")
  var_0_den: metrics.LastValue.from_output("var_0_den")
  var_0_rgb: metrics.LastValue.from_output("var_0_rgb")
  var_1_den: metrics.LastValue.from_output("var_1_den")
  var_1_rgb: metrics.LastValue.from_output("var_1_rgb")

  # gradient norm
  grad_norm: metrics.LastValue.from_output("grad_norm")


def make_loss_fn(config, model):
  """Make VDM loss function."""
  latent_loss_weight = config.trainer.get("latent_loss_weight", 1.)
  diff_loss_weight = config.trainer.get("diff_loss_weight", 1.)
  diff_loss_rgb_weight = config.trainer.get("diff_loss_rgb_weight", 1.)
  diff_loss_den_weight = config.trainer.get("diff_loss_den_weight", 1.)
  diff_vis_loss_weight = config.trainer.get("diff_vis_loss_weight", 1.)
  diff_vis_loss_rgb_weight = config.trainer.get("diff_vis_loss_rgb_weight", 1.)
  diff_vis_loss_den_weight = config.trainer.get("diff_vis_loss_den_weight", 1.)

  def loss_fn(params, pm_state, batch, rng, is_train):
    inputs = {"vox": batch["vox"], "cond": batch["label"]}
    rng, sample_rng = jax.random.split(rng)
    rngs = {"sample": sample_rng}
    if is_train:
      rng, dropout_rng = jax.random.split(rng)
      rngs["dropout"] = dropout_rng

    # sample time steps, with antithetic sampling
    if is_train:
      outputs, new_pm_state = model.apply(
          variables={"params": params, **pm_state},
          **inputs,
          rngs=rngs,
          train=True,
          mutable=list(pm_state.keys()),
      )
    else:
      outputs = model.apply(
          variables={"params": params, **pm_state},
          **inputs,
          rngs=rngs,
          train=is_train,
          mutable=False
      )
      new_pm_state = pm_state

    rescale_to_bpd = 1. / (np.prod(inputs["vox"].shape[1:]) * np.log(2.))
    bpd_latent = jnp.mean(
        outputs["loss_klz"]) * rescale_to_bpd * latent_loss_weight

    bpd_diff_den = jnp.mean(
        outputs["loss_diff_den"]) * rescale_to_bpd * diff_loss_den_weight
    bpd_diff_rgb = jnp.mean(
        outputs["loss_diff_rgb"]) * rescale_to_bpd * diff_loss_rgb_weight
    bpd_diff = (bpd_diff_den + bpd_diff_rgb) * diff_loss_weight
    # bpd_diff = jnp.mean(
    #     outputs["loss_diff"]) * rescale_to_bpd * diff_loss_weight

    bpd_diff_den_vis = jnp.mean(
        outputs["loss_diff_den_vis"]
        ) * rescale_to_bpd * diff_vis_loss_den_weight
    bpd_diff_rgb_vis = jnp.mean(
        outputs["loss_diff_rgb_vis"]
        ) * rescale_to_bpd * diff_vis_loss_rgb_weight
    bpd_diff_vis = (bpd_diff_den_vis + bpd_diff_rgb_vis) * diff_vis_loss_weight
    # loss_diff = loss_diff_den + loss_diff_rgb
    # bpd_diff_vis = jnp.mean(
    #     outputs["loss_diff_vis"]) * rescale_to_bpd * diff_vis_loss_weight
    bpd = bpd_latent + bpd_diff + bpd_diff_vis
    scalar_dict = {
        "bpd": bpd,
        "bpd_latent": bpd_latent,
        # Original diffusion loss
        "bpd_diff": bpd_diff,
        "bpd_diff_rgb": bpd_diff_rgb,
        "bpd_diff_den": bpd_diff_den,
        # Visibility masked diffusion loss
        "bpd_diff_vis": bpd_diff_vis,
        "bpd_diff_vis_rgb": bpd_diff_rgb_vis,
        "bpd_diff_vis_den": bpd_diff_den_vis,
    }
    scalar_dict.update(outputs)
    output = {"scalars": scalar_dict,
              "mutable_state": new_pm_state}

    return bpd, output
  return loss_fn


def train_step(
    config,
    model,
    state,
    opt,
    # opt,
    learning_rate_fn,
    batch,
    rng
):
  """Train step."""
  assert config.get("multi")
  rng = jax.random.fold_in(rng, jax.lax.axis_index("batch"))
  rng = jax.random.fold_in(rng, state.step)

  if config.trainer.get("add_label", True) and "label" not in batch:
    batch["label"] = jnp.zeros(batch["vox"].shape[0], jnp.int32)

  # TODO(guandao) this can be slow since compile it every step
  with utils.StepTraceContextHelper("train_make_loss", 0) as trace_context:
    curr_loss_fn = make_loss_fn(config, model)
    grad_fn = jax.value_and_grad(curr_loss_fn, has_aux=True)
    trace_context.next_step()
  (loss_val, metrics_dict), grads = grad_fn(
      state.params, state.mutable_state, batch,
      rng=rng,
      is_train=True)
  grads = jax.lax.pmean(grads, "batch")
  new_state = ema_apply_gradients(
      state, opt=opt, grads=grads, ema_rate=config.opt.ema_rate)
  new_mutable_state = metrics_dict["mutable_state"]
  new_state.replace(mutable_state=new_mutable_state)

  metrics_dict["scalars"] = jax.tree.map(
      lambda x: jax.lax.pmean(x, axis_name="batch"),
      metrics_dict["scalars"])

  lr = learning_rate_fn(state.step)
  metrics_dict["scalars"]["grad_norm"] = utils.compute_grad_norm(grads)
  if config.get("multi"):
    metrics_update = TrainMetrics.gather_from_model_output(
        loss=loss_val, learning_rate=lr, **metrics_dict["scalars"])
  else:
    metrics_update = TrainMetrics.single_from_model_output(
        loss=loss_val, learning_rate=lr, **metrics_dict["scalars"])

  train_info = {
      "hist/grad": grads,
      "hist/grad": state.params,
  }
  return new_state, metrics_update, train_info


@flax.struct.dataclass
class EvalMetrics(metrics.Collection):
  """Eval metrics."""

  # Loss
  bpd: metrics.Average.from_output("bpd")
  bpd_latent: metrics.Average.from_output("bpd_latent")
  bpd_diff: metrics.Average.from_output("bpd_diff")

  # From other input
  loss_klz_den: metrics.Average.from_output("loss_klz_den")
  loss_klz_rgb: metrics.Average.from_output("loss_klz_rgb")
  loss_klz: metrics.Average.from_output("loss_klz")
  loss_diff: metrics.Average.from_output("loss_diff")
  loss_diff_den: metrics.Average.from_output("loss_diff_den")
  loss_diff_rgb: metrics.Average.from_output("loss_diff_rgb")
  var_0_den: metrics.Average.from_output("var_0_den")
  var_0_rgb: metrics.Average.from_output("var_0_rgb")
  var_1_den: metrics.Average.from_output("var_1_den")
  var_1_rgb: metrics.Average.from_output("var_1_rgb")


def evaluate(
    config,
    model,
    pstate,
    eval_ds,
    rng,
    num_eval_steps = -1,
    verbose = False
):
  """Perform one evaluation."""
  logging.info("=== Experiment.evaluate() ===")
  assert config.get("multi")
  rng = jax.random.fold_in(rng, jax.process_index())

  state = flax_utils.unreplicate(pstate)
  p_create_slice = jax.pmap(reluf_vis_utils.create_center_slices, "batch")
  visualize_ortho = functools.partial(
      reluf_vis_utils.visualize_orthographic_projection,
      scene_scale=config.render_config.trainer.scene_grid_scale,
      white_bkgd=config.render_config.data.white_bkgd,
      preconditioner=config.render_config.model.preconditioner,
      offset=config.render_config.model.density_offset
  )
  p_visualize_ortho = jax.pmap(visualize_ortho, axis_name="batch")

  # Evaluating test time losses
  logging.info("=== Eval set loss ===")
  eval_metrics = EvalMetrics.empty()
  curr_loss_fn = make_loss_fn(config, model)
  p_curr_loss_fn = jax.pmap(
      functools.partial(curr_loss_fn, is_train=False), axis_name="batch")
  batch = None
  for curr_eval_step, batch in enumerate(eval_ds):
    logging.info("\t\tEval step [%d/%d]", curr_eval_step, num_eval_steps)
    if verbose:
      print("[%d/%d]" % (curr_eval_step, num_eval_steps))
    del batch["sid"]
    if config.trainer.get("add_label", True) and "label" not in batch:
      batch["label"] = jnp.zeros(batch["vox"].shape[0], jnp.int32)
    rng_i = jax.random.fold_in(rng, curr_eval_step)
    batch = jax.tree.map(jnp.asarray, batch)
    _, outputs = p_curr_loss_fn(
        pstate.params,
        pstate.mutable_state,
        batch,
        rng=flax_utils.replicate(rng_i))
    scalars = {k: jnp.mean(v) for k, v in outputs["scalars"].items()}
    curr_eval_metric = EvalMetrics.single_from_model_output(**scalars)
    eval_metrics.merge(curr_eval_metric)
    if curr_eval_step >= num_eval_steps:
      break

  eval_info = {}
  sample_fn = ldm_model.make_sample_fn_even_d(
      model,
      config,
      config.trainer.get("skip_every_d_sample_steps", 1),
      multi=True)
  # p_sample_fn = jax.pmap(sample_fn, axis_name="batch")
  if batch is not None:
    uncond_cond = jnp.zeros((1,), dtype=jnp.int32)
    logging.info("Sample.")
    eval_info["images/inputs"] = batch["images"]

    # Signature: sample_fn(params, mstates, rng, cond)
    # pvariables = {
    # "params": pstate.params,
    # **pstate.mutable_state
    # }
    # # z0 will be a shared device array with (#devices, ...)
    # z0 = p_sample_fn(pvariables, flax_utils.replicate(rng_i),
    # flax_utils.replicate(uncond_cond))

    variables = {
        "params": state.params,
        **state.mutable_state
    }
    # z0 will be a shared device array with (#devices, ...)
    z0 = sample_fn(variables, rng_i, uncond_cond)

    # Slices: {key: shared_device_array(#device, ...), ...}
    logging.info("Creating sigma slices.")
    sig_slices = p_create_slice(z0[Ellipsis, :1])
    logging.info("==Saving sigma slice images.")
    for k, s in sig_slices.items():
      # transfer to host
      s = np.array(jax.device_get(s))
      eval_info["images/slice_sig_%s" % k] = s

    logging.info("Creating center slices.")
    if z0.shape[-1] > 1:
      rgb_slices = p_create_slice(z0[Ellipsis, 1:4])
      logging.info("==Saving center slice images.")
      for k, s in rgb_slices.items():
        # transfer to host
        s = np.array(jax.device_get(s))
        eval_info["images/slice_rgb_%s" % k] = s
    else:
      logging.info("==Skip.")

    # Orthographic project, it can be pmapped now
    logging.info("Creating orthographic projection.")
    if z0.shape[-1] > 1:
      print(z0.dtype, z0.shape)
      otho_imgs = p_visualize_ortho(z0)  # (#divice, H, W, 3)
      logging.info("==>saving ortho projection images.")
      eval_info["images/ortho_res"] = otho_imgs
    else:
      logging.info("==>Skip.")

    # Orthographic project, but with color cube
    logging.info("Creating orthographic projection with color cube.")
    bss, res = z0.shape[:-4], z0.shape[-2]
    color_cube = jnp.concatenate([
        x[Ellipsis, None] for x in jnp.meshgrid(
            jnp.arange(res), jnp.arange(res), jnp.arange(res))
    ], axis=-1) / float(res - 1)
    color_cube = color_cube.reshape(*([1] * len(bss) + list(z0.shape[-4:-1] +
                                                            (3,))))
    color_cube = color_cube * jnp.ones(z0.shape[:-1] + (1,))
    z0_colorcube = jnp.concatenate([z0[Ellipsis, :1], color_cube], axis=-1)
    sig_color_ortho_imgs = p_visualize_ortho(z0_colorcube)  # (#divice, H, W, 3)
    eval_info["images/sig_colorcube_ortho"] = sig_color_ortho_imgs

    # Orthographic projection, but with constant color (gray)
    logging.info("Creating orthographic projection with gray color.")
    z0_gray = jnp.concatenate(
        [z0[Ellipsis, :1], jnp.ones(tuple(z0.shape[:-1]) + (3,)) * 0.5], axis=-1)
    sig_gray_ortho_imgs = p_visualize_ortho(z0_gray) # (#divice, H, W, 3)
    eval_info["images/sig_gray_ortho"] = sig_gray_ortho_imgs

    # Finally save the z0, push it to CPU (host) memory.
    eval_info["tensor/z0"] = np.array(jax.device_get(z0))


  return eval_metrics, eval_info


def eval_visualize(config,
                   writer,
                   step,
                   model,
                   state,
                   eval_info,
                   eval_dir):
  curr_dir = os.path.join(eval_dir, "step_%d" % step)
  tf.io.gfile.MakeDirs(curr_dir)
  max_num_images = config.get("eval_vis_num_imgs", 16)
  images_dict = {}
  for k, v in eval_info.items():
    if k.startswith("images/"):
      name = k[len("images/"):]
      img = v[:max_num_images]
      v_imgs = np.array(image_vis_utils.create_images(img))
      images_dict["eval_%s" % name] = v_imgs
      curr_fpath = os.path.join(curr_dir, "%s.png" % name)
      image_vis_utils.save_images(v_imgs, curr_fpath)

  writer.write_images(step, images_dict)
  hist_dict = {("hist_" + k): np.array(jax.device_get(v))
               for k, v in images_dict.items()}
  writer.write_histograms(step, hist_dict)


def train_visualize(config,
                    writer,
                    step,
                    model,
                    state,
                    train_info,
                    visdir):
  train_info_img = {
      "train/{0}".format(key): image_vis_utils.create_images(val)
      for key, val in train_info.items()
      if key.startswith("images/")}
  writer.write_images(step, train_info_img)

  curr_dir = os.path.join(visdir, "step_%d" % step)
  tf.io.gfile.MakeDirs(curr_dir)
  for k, out in train_info_img.items():
    out_fpath = os.path.join(curr_dir, "%s.png" % (k[len("train/"):]))
    image_vis_utils.save_images(out, out_fpath)

  # Visualizing the histograms
  train_info_hist = {
      ("hist_image_" + k): np.array(jax.device_get(v))
      for k, v in train_info_img.items()}
  writer.write_histograms(step, train_info_hist)

  # Now visualize the historgam
  hist_dict = {}
  for key, val in train_info.items():
    if key.startswith("hist/"):
      name = key[len("hist/"):]
      hist_dict.update({
          "/".join([name] + list(k)): np.array(jax.device_get(v))
          for k, v in flax.traverse_util.flatten_dict(val).items()
      })
  writer.write_histograms(step, hist_dict)
