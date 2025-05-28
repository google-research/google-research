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

"""GLIDE diffusion model utilities."""

# pylint: disable=g-bad-import-order
import functools

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import create_model_and_diffusion
from glide_text2im.model_creation import model_and_diffusion_defaults
from glide_text2im.model_creation import model_and_diffusion_defaults_upsampler
# pylint: enable=g-bad-import-order


def load_diffusion(name,
                   t_respace,
                   device,
                   has_cuda,
                   classifier_free_guidance_scale,
                   enable_parallel=True):
  """Create base model."""
  if name == "base" or name == "base-inpaint":
    options = model_and_diffusion_defaults()
  elif name == "upsample":
    options = model_and_diffusion_defaults_upsampler()
  else:
    raise NotImplementedError

  options["inpaint"] = "inpaint" in name
  options["use_fp16"] = has_cuda
  options["timestep_respacing"] = str(
      t_respace)  # Use few diffusion steps for fast sampling.
  model, diffusion = create_model_and_diffusion(**options)

  # Copy some properties to GPU.
  diffusion.log_betas = np.log(diffusion.betas)
  for attr in [
      "log_betas", "posterior_log_variance_clipped", "alphas_cumprod_prev",
      "alphas_cumprod", "sqrt_recip_alphas_cumprod",
      "sqrt_recipm1_alphas_cumprod", "posterior_variance",
      "posterior_mean_coef1", "posterior_mean_coef2",
      "posterior_log_variance_clipped"
  ]:
    value_np = getattr(diffusion, attr)
    value_th = torch.from_numpy(value_np).to(device)
    setattr(diffusion, f"{attr}_th", value_th)

  # Overwrite methods with faster GPU versions.
  diffusion.q_posterior_mean_variance = functools.partial(
      q_posterior_mean_variance, diffusion)
  # pylint: disable=protected-access
  diffusion._predict_xstart_from_eps = functools.partial(
      predict_xstart_from_eps, diffusion)
  # pylint: disable=g-long-lambda
  diffusion.p_mean_variance = lambda model, *args, **kwargs: p_mean_variance(
      diffusion, diffusion._wrap_model(model), *args, **kwargs)
  # pylint: enable=g-long-lambda
  # pylint: enable=protected-access

  model.load_state_dict(load_checkpoint(name, torch.device("cpu")))
  model.eval()
  if has_cuda:
    model.convert_to_fp16()
  model.to(device)
  print("total base parameters", sum(x.numel() for x in model.parameters()))

  if has_cuda and enable_parallel:
    parallel_model = nn.DataParallel(model)
  else:
    parallel_model = model

  assert classifier_free_guidance_scale >= 0
  if classifier_free_guidance_scale > 0:
    # Create a classifier-free guidance sampling function.
    def model_fn(x_t, ts, **kwargs):
      half = x_t[:len(x_t) // 2]
      combined = torch.cat([half, half], dim=0)
      model_out = parallel_model(combined, ts, **kwargs)
      eps, rest = model_out[:, :3], model_out[:, 3:]
      cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
      half_eps = (1 + classifier_free_guidance_scale
                 ) * cond_eps - classifier_free_guidance_scale * uncond_eps
      eps = torch.cat([half_eps, half_eps], dim=0)
      return torch.cat([eps, rest], dim=1)
  else:
    model_fn = parallel_model

  return model, parallel_model, model_fn, diffusion, options


@torch.no_grad()
def embed_queries_glide(queries, glide_model, glide_options, device):
  """Embed textual queries with the GLIDE text encoder."""

  tokens = []
  mask = []
  for query_mod in queries:
    # Create the caption tokens.
    view_tokens = glide_model.tokenizer.encode(query_mod)
    view_tokens, view_mask = glide_model.tokenizer.padded_tokens_and_mask(
        view_tokens, glide_options["text_ctx"])
    tokens.append(view_tokens)
    mask.append(view_mask)

  # Create the classifier-free guidance tokens (empty).
  uncond_tokens, uncond_mask = glide_model.tokenizer.padded_tokens_and_mask(
      [], glide_options["text_ctx"])
  uncond_tokens = [uncond_tokens] * len(queries)
  uncond_mask = [uncond_mask] * len(queries)

  # Pack the tokens together into model kwargs.
  model_kwargs = dict(
      tokens=torch.tensor(tokens + uncond_tokens, device=device),
      mask=torch.tensor(mask + uncond_mask, dtype=torch.bool, device=device),
  )
  return model_kwargs


def extract_into_tensor(arr, timesteps, broadcast_shape):
  """Extract values from a 1-D numpy array for a batch of indices.

  Args:
    arr: the 1-D numpy array.
    timesteps: a tensor of indices into the array to extract.
    broadcast_shape: a larger shape of K dimensions with the batch dimension
      equal to the length of timesteps.

  Returns:
    res: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
  """
  device = timesteps.device
  assert arr.device == device
  res = arr[timesteps].float()
  new_dims = [1] * (len(broadcast_shape) - res.ndim)
  res = res.view(*res.shape, *new_dims)
  return torch.broadcast_to(res, broadcast_shape)


def predict_xstart_from_eps(diffusion, x_t, t, eps):
  assert x_t.shape == eps.shape
  return (extract_into_tensor(diffusion.sqrt_recip_alphas_cumprod_th, t,
                              x_t.shape) * x_t -
          extract_into_tensor(diffusion.sqrt_recipm1_alphas_cumprod_th, t,
                              x_t.shape) * eps)


def q_posterior_mean_variance(diffusion, x_start, x_t, t):
  """Compute mean and variance of diffusion posterior q(z_{t-1} | z_t, x_0)."""
  assert x_start.shape == x_t.shape
  mean_coef1 = extract_into_tensor(diffusion.posterior_mean_coef1_th, t,
                                   x_t.shape)
  mean_coef2 = extract_into_tensor(diffusion.posterior_mean_coef2_th, t,
                                   x_t.shape)
  posterior_mean = mean_coef1 * x_start + mean_coef2 * x_t
  posterior_variance = extract_into_tensor(diffusion.posterior_variance_th, t,
                                           x_t.shape)
  posterior_log_variance_clipped = extract_into_tensor(
      diffusion.posterior_log_variance_clipped_th, t, x_t.shape)
  assert (posterior_mean.shape[0] == posterior_variance.shape[0] ==
          posterior_log_variance_clipped.shape[0] == x_start.shape[0])
  return posterior_mean, posterior_variance, posterior_log_variance_clipped


def p_mean_variance(self,
                    model,
                    x,
                    t,
                    clip_denoised=True,
                    denoised_fn=None,
                    model_kwargs=None):
  """Apply the model to get p(x_{t-1} | x_t), as well as a prediction of the initial x, x_0.

  Args:
    self: diffusion class.
    model: the model, which takes a signal and a batch of timesteps as input.
    x: the [N x C x ...] tensor at time t.
    t: a 1-D Tensor of timesteps.
    clip_denoised: if True, clip the denoised signal into [-1, 1].
    denoised_fn: if not None, a function which applies to the x_start prediction
      before it is used to sample. Applies before clip_denoised.
    model_kwargs: if not None, a dict of extra keyword arguments to pass to the
      model. This can be used for conditioning.

  Returns:
    a dict with the following keys:
      - 'mean': the model mean output.
      - 'variance': the model variance output.
      - 'log_variance': the log of 'variance'.
      - 'pred_xstart': the prediction for x_0.
  """
  if model_kwargs is None:
    model_kwargs = {}

  batch, channels = x.shape[:2]
  assert t.shape == (batch,)
  model_output = model(x, t, **model_kwargs)
  if isinstance(model_output, tuple):
    model_output, extra = model_output
  else:
    extra = None

  assert model_output.shape == (batch, channels * 2, *x.shape[2:])
  model_output, model_var_values = torch.split(model_output, channels, dim=1)
  min_log = extract_into_tensor(self.posterior_log_variance_clipped_th, t,
                                x.shape)
  max_log = extract_into_tensor(self.log_betas_th, t, x.shape)
  # The model_var_values is [-1, 1] for [min_var, max_var].
  frac = (model_var_values + 1) / 2
  model_log_variance = frac * max_log + (1 - frac) * min_log
  model_variance = torch.exp(model_log_variance)

  def process_xstart(x):
    if denoised_fn is not None:
      x = denoised_fn(x)
    if clip_denoised:
      return x.clamp(-1, 1)
    return x

  # pylint: disable=protected-access
  pred_xstart = process_xstart(
      self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output))
  # pylint: enable=protected-access
  model_mean, _, _ = self.q_posterior_mean_variance(
      x_start=pred_xstart, x_t=x, t=t)

  assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
  return {
      "mean": model_mean,
      "variance": model_variance,
      "log_variance": model_log_variance,
      "pred_xstart": pred_xstart,
      "extra": extra,
  }


def scale_glide(x):
  return x * 2 - 1  # Scale from [0, 1] to [-1, 1].


def preprocess_glide(x, order="NHWC", resize=True):
  if order == "NHWC":
    # x is [NHWC]. Reshape to NCHW.
    x = x.movedim(-1, 1)
  x = scale_glide(x)
  if resize and x.shape[-2:] != (64, 64):
    x = F.interpolate(x, (64, 64), mode="bilinear")
  return x


def unscale_glide(x):
  return (x + 1) / 2  # Scale from [-1, 1] to [0, 1].
