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

import functools
import os
from typing import Any, Dict, Optional, Tuple

from absl import logging
# from clu import metric_writers
# from flax.training import checkpoints
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
import nf_diffusion.models.ldm_reluf_dual_v2 as ldm_model_v2
import nf_diffusion.models.ldm_reluf_dual_v3 as ldm_model_v3
from nf_diffusion.trainers import ldm_reluf_dual_v2
from nf_diffusion.trainers import vdm
from nf_diffusion.trainers.utils import image_vis_utils
from nf_diffusion.trainers.utils import trainer_utils as utils


get_learning_rate_scheduler = vdm.get_learning_rate_scheduler
ema_apply_gradients = vdm.ema_apply_gradients
get_opt = vdm.get_opt
TrainState = ldm_reluf_dual_v2.TrainState


def create_train_state(
    config, rng, data_info
):
  """Initialize training state."""
  logging.warning("=== Initializing model ===")
  model_name = config.trainer.get("model_name", "v2")
  if model_name == "v2":
    model = ldm_model_v2.LDM(**config.model)
  elif model_name == "v3":
    model = ldm_model_v3.LDM(**config.model)
  else:
    raise ValueError

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
    verbose = False,
):
  """Perform one evaluation."""
  logging.info("=== Experiment.evaluate() ===")
  assert config.get("multi")
  rng = jax.random.fold_in(rng, jax.process_index())

  # Evaluating test time losses
  logging.info("=== Eval set loss ===")
  eval_metrics = EvalMetrics.empty()
  curr_loss_fn = make_loss_fn(config, model)
  p_curr_loss_fn = jax.pmap(
      functools.partial(curr_loss_fn, is_train=False), axis_name="batch")
  for curr_eval_step, batch in enumerate(eval_ds):
    logging.info("\t\tEval step [%d/%d]", curr_eval_step, num_eval_steps)
    if verbose:
      print("[%d/%d]" % (curr_eval_step, num_eval_steps))
    batch = {"vox": batch["vox"]}
    if config.trainer.get("add_label", True) and "label" not in batch:
      batch["label"] = jnp.zeros(batch["vox"].shape[0], jnp.int32)
    rng_i = jax.random.fold_in(rng, curr_eval_step)
    batch = jax.tree.map(jnp.asarray, batch)
    eval_with_ema_params = config.trainer.get("eval_with_ema_params", True)
    _, outputs = p_curr_loss_fn(
        (pstate.ema_params if eval_with_ema_params else pstate.params),
        pstate.mutable_state,
        batch,
        rng=flax_utils.replicate(rng_i))
    scalars = {k: jnp.mean(v) for k, v in outputs["scalars"].items()}
    curr_eval_metric = EvalMetrics.single_from_model_output(**scalars)
    eval_metrics = eval_metrics.merge(curr_eval_metric)
    if curr_eval_step >= num_eval_steps:
      break

  return eval_metrics, {}


def eval_visualize(config,
                   writer,
                   step,
                   model,
                   state,
                   eval_info,
                   eval_dir):
  # NOTE: this is none since I don't want to evaluate anything
  pass

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
