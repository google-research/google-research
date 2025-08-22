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

"""Perceptual loss function."""
import abc
from typing import Optional
import flax.linen as nn
from gvt.utils import pretrained_model_utils
import jax.numpy as jnp


def l2_loss(y_true, y_pred):
  diff = y_true - y_pred
  diff = jnp.asarray(diff, jnp.float32)
  return jnp.mean(jnp.square(diff))


class PerceptualLoss(abc.ABC):
  """Perceptual loss function."""

  model: Optional[nn.Module] = None
  state: Optional[pretrained_model_utils.ModelState] = None

  def __call__(
      self,
      real_images,
      fake_images,
      perceptual_loss_on_logit = False):
    """Calculates perceptual loss on pre-trained model."""
    if self.model is None or self.state is None:
      self.model, self.state = pretrained_model_utils.get_pretrained_model()
    real_pools, real_outputs = pretrained_model_utils.get_pretrained_embs(
        self.state, self.model, images=real_images)
    fake_pools, fake_outputs = pretrained_model_utils.get_pretrained_embs(
        self.state, self.model, images=fake_images)
    if perceptual_loss_on_logit:
      loss = l2_loss(real_outputs, fake_outputs)
    else:
      loss = l2_loss(real_pools, fake_pools)
    return loss


def perceptual_loss(
    state,
    model,
    real_images,
    fake_images,
    perceptual_loss_on_logit = False):
  """Calculates perceptual loss on pre-trained model."""
  real_pools, real_outputs = pretrained_model_utils.get_pretrained_embs(
      state, model, images=real_images)
  fake_pools, fake_outputs = pretrained_model_utils.get_pretrained_embs(
      state, model, images=fake_images)
  if perceptual_loss_on_logit:
    loss = l2_loss(real_outputs, fake_outputs)
  else:
    loss = l2_loss(real_pools, fake_pools)
  return loss


def get_model_state():
  model, state = pretrained_model_utils.get_pretrained_model()
  return model, state
