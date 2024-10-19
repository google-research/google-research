# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Extension of diffusers.DDIMScheduler."""

from typing import Optional, Tuple, Union
from diffusers import DDIMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput
from diffusers.utils import randn_tensor
import torch
from torch.distributions import Normal


class DDIMSchedulerExtended(DDIMScheduler):
  """Extension of diffusers.DDIMScheduler."""

  def _get_variance_logprob(self, timestep, prev_timestep):
    alpha_prod_t = self.alphas_cumprod[timestep].to(timestep.device)
    mask_a = (prev_timestep >= 0).int().to(timestep.device)
    mask_b = 1 - mask_a
    alpha_prod_t_prev = (
        self.alphas_cumprod[prev_timestep].to(timestep.device) * mask_a
        + self.final_alpha_cumprod.to(timestep.device) * mask_b
    )
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    variance = (beta_prod_t_prev / beta_prod_t) * (
        1 - alpha_prod_t / alpha_prod_t_prev
    )

    return variance

  # new step function that can take multiple timesteps and middle step images as
  # input
  def step_logprob(
      self,
      model_output,
      timestep,
      sample,
      eta = 1.0,
      use_clipped_model_output = False,
      generator=None,
      variance_noise = None,
      return_dict = True,
  ):  # pylint: disable=g-bare-generic
    """Predict the sample at the previous timestep by reversing the SDE.

    Core function to propagate the diffusion process from the learned model
    outputs (most often the predicted noise).

    Args:
        model_output (`torch.FloatTensor`): direct output from learned diffusion
          model.
        timestep (`int`): current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`): current instance of sample being created
          by diffusion process.
        eta (`float`): weight of noise for added noise in diffusion step.
        use_clipped_model_output (`bool`): if `True`, compute "corrected"
          `model_output` from the clipped predicted original sample. Necessary
          because predicted original sample is clipped to [-1, 1] when
          `self.config.clip_sample` is `True`. If no clipping has happened,
          "corrected" `model_output` would coincide with the one provided as
          input and `use_clipped_model_output` will have not effect.
        generator: random number generator.
        variance_noise (`torch.FloatTensor`): instead of generating noise for
          the variance using `generator`, we can directly provide the noise for
          the variance itself. This is useful for methods such as
          CycleDiffusion. (https://arxiv.org/abs/2210.05559)
        return_dict (`bool`): option for returning tuple rather than
          DDIMSchedulerOutput class

    Returns:
        [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] or `tuple`:
        [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] if `return_dict` is
        True, otherwise a `tuple`. When
        returning a tuple, the first element is the sample tensor.

        log_prob (`torch.FloatTensor`): log probability of the sample.
    """
    if self.num_inference_steps is None:
      raise ValueError(
          "Number of inference steps is 'None', you need to run 'set_timesteps'"
          " after creating the scheduler"
      )

    # pylint: disable=line-too-long
    # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
    # Ideally, read DDIM paper in-detail understanding

    # Notation (<variable name> -> <name in paper>
    # - pred_noise_t -> e_theta(x_t, t)
    # - pred_original_sample -> f_theta(x_t, t) or x_0
    # - std_dev_t -> sigma_t
    # - eta -> η
    # - pred_sample_direction -> "direction pointing to x_t"
    # - pred_prev_sample -> "x_t-1"

    # 1. get previous step value (=t-1)
    prev_timestep = (
        timestep - self.config.num_train_timesteps // self.num_inference_steps
    )

    # 2. compute alphas, betas
    alpha_prod_t = self.alphas_cumprod[timestep].to(timestep.device)
    # alpha_prod_t = alpha_prod_t.to(torch.float16)
    mask_a = (prev_timestep >= 0).int().to(timestep.device)
    mask_b = 1 - mask_a
    alpha_prod_t_prev = (
        self.alphas_cumprod[prev_timestep].to(timestep.device) * mask_a
        + self.final_alpha_cumprod.to(timestep.device) * mask_b
    )
    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    if self.config.prediction_type == "epsilon":
      pred_original_sample = (
          sample - beta_prod_t ** (0.5) * model_output
      ) / alpha_prod_t ** (0.5)
    elif self.config.prediction_type == "sample":
      pred_original_sample = model_output
    elif self.config.prediction_type == "v_prediction":
      pred_original_sample = (alpha_prod_t**0.5) * sample - (
          beta_prod_t**0.5
      ) * model_output
      # predict V
      model_output = (alpha_prod_t**0.5) * model_output + (
          beta_prod_t**0.5
      ) * sample
    else:
      raise ValueError(
          f"prediction_type given as {self.config.prediction_type} must be one"
          " of `epsilon`, `sample`, or `v_prediction`"
      )

    # 4. Clip "predicted x_0"
    if self.config.clip_sample:
      pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    variance = self._get_variance_logprob(timestep, prev_timestep).to(
        dtype=sample.dtype
    )
    std_dev_t = (eta * variance ** (0.5)).to(dtype=sample.dtype)

    if use_clipped_model_output:
      # the model_output is always re-derived from the clipped x_0 in Glide
      model_output = (
          sample - alpha_prod_t ** (0.5) * pred_original_sample
      ) / beta_prod_t ** (0.5)

    # pylint: disable=line-too-long
    # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (
        0.5
    ) * model_output

    # pylint: disable=line-too-long
    # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    prev_sample = (
        alpha_prod_t_prev ** (0.5) * pred_original_sample
        + pred_sample_direction
    )

    if eta > 0:
      device = model_output.device
      if variance_noise is not None and generator is not None:
        raise ValueError(
            "Cannot pass both generator and variance_noise. Please make sure"
            " that either `generator` or `variance_noise` stays `None`."
        )

      if variance_noise is None:
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=device,
            dtype=model_output.dtype,
        )
      variance = std_dev_t * variance_noise
      dist = Normal(prev_sample, std_dev_t)
      prev_sample = prev_sample.detach().clone() + variance
      log_prob = (
          dist.log_prob(prev_sample.detach().clone())
          .mean(dim=-1)
          .mean(dim=-1)
          .mean(dim=-1)
          .detach()
          .cpu()
      )
    if not return_dict:
      return (prev_sample,)

    return (
        DDIMSchedulerOutput(
            prev_sample=prev_sample, pred_original_sample=pred_original_sample
        ),
        log_prob,
    )

  def step_forward_logprob(
      self,
      model_output,
      timestep,
      sample,
      next_sample,
      eta = 1.0,
      use_clipped_model_output = False,
      generator=None,
      variance_noise = None,
      return_dict = True,
  ):  # pylint: disable=g-bare-generic
    """Predict the sample at the previous timestep by reversing the SDE.

    Core function to propagate the diffusion process from the learned model
    outputs (most often the predicted noise).

    Args:
        model_output (`torch.FloatTensor`): direct output from learned diffusion
          model.
        timestep (`int`): current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`): current instance of sample (x_t) being
          created by diffusion process.
        next_sample (`torch.FloatTensor`): instance of next sample (x_t-1) being
          created by diffusion process. RL sampling is the backward process,
          therefore, x_t-1 is the "next" sample of x_t.
        eta (`float`): weight of noise for added noise in diffusion step.
        use_clipped_model_output (`bool`): if `True`, compute "corrected"
          `model_output` from the clipped predicted original sample. Necessary
          because predicted original sample is clipped to [-1, 1] when
          `self.config.clip_sample` is `True`. If no clipping has happened,
          "corrected" `model_output` would coincide with the one provided as
          input and `use_clipped_model_output` will have not effect.
        generator: random number generator.
        variance_noise (`torch.FloatTensor`): instead of generating noise for
          the variance using `generator`, we can directly provide the noise for
          the variance itself. This is useful for methods such as
          CycleDiffusion. (https://arxiv.org/abs/2210.05559)
        return_dict (`bool`): option for returning tuple rather than
          DDIMSchedulerOutput class

    Returns:
        log probability.
    """
    if self.num_inference_steps is None:
      raise ValueError(
          "Number of inference steps is 'None', you need to run 'set_timesteps'"
          " after creating the scheduler"
      )

    # pylint: disable=line-too-long
    # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
    # Ideally, read DDIM paper in-detail understanding

    # Notation (<variable name> -> <name in paper>
    # - pred_noise_t -> e_theta(x_t, t)
    # - pred_original_sample -> f_theta(x_t, t) or x_0
    # - std_dev_t -> sigma_t
    # - eta -> η
    # - pred_sample_direction -> "direction pointing to x_t"
    # - pred_prev_sample -> "x_t-1"

    # 1. get previous step value (=t-1)
    prev_timestep = (
        timestep - self.config.num_train_timesteps // self.num_inference_steps
    )

    # 2. compute alphas, betas
    alpha_prod_t = self.alphas_cumprod[timestep].to(timestep.device)
    mask_a = (prev_timestep >= 0).int().to(timestep.device)
    mask_b = 1 - mask_a
    alpha_prod_t_prev = (
        self.alphas_cumprod[prev_timestep].to(timestep.device) * mask_a
        + self.final_alpha_cumprod.to(timestep.device) * mask_b
    )
    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    if self.config.prediction_type == "epsilon":
      pred_original_sample = (
          sample - beta_prod_t ** (0.5) * model_output
      ) / alpha_prod_t ** (0.5)
    elif self.config.prediction_type == "sample":
      pred_original_sample = model_output
    elif self.config.prediction_type == "v_prediction":
      pred_original_sample = (alpha_prod_t**0.5) * sample - (
          beta_prod_t**0.5
      ) * model_output
      # predict V
      model_output = (alpha_prod_t**0.5) * model_output + (
          beta_prod_t**0.5
      ) * sample
    else:
      raise ValueError(
          f"prediction_type given as {self.config.prediction_type} must be one"
          " of `epsilon`, `sample`, or `v_prediction`"
      )

    # 4. Clip "predicted x_0"
    if self.config.clip_sample:
      pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    variance = self._get_variance_logprob(timestep, prev_timestep).to(
        dtype=sample.dtype
    )
    std_dev_t = (eta * variance ** (0.5)).to(dtype=sample.dtype)

    if use_clipped_model_output:
      # the model_output is always re-derived from the clipped x_0 in Glide
      model_output = (
          sample - alpha_prod_t ** (0.5) * pred_original_sample
      ) / beta_prod_t ** (0.5)

    # pylint: disable=line-too-long
    # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (
        0.5
    ) * model_output

    # pylint: disable=line-too-long
    # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    prev_sample = (
        alpha_prod_t_prev ** (0.5) * pred_original_sample
        + pred_sample_direction
    )

    if eta > 0:
      device = model_output.device
      if variance_noise is not None and generator is not None:
        raise ValueError(
            "Cannot pass both generator and variance_noise. Please make sure"
            " that either `generator` or `variance_noise` stays `None`."
        )

      if variance_noise is None:
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=device,
            dtype=model_output.dtype,
        )
      variance = std_dev_t * variance_noise
      dist = Normal(prev_sample, std_dev_t)
      log_prob = (
          dist.log_prob(next_sample.detach().clone())
          .mean(dim=-1)
          .mean(dim=-1)
          .mean(dim=-1)
      )

    return log_prob
