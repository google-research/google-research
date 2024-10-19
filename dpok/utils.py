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

"""Helper functions."""

from ImageReward import ImageReward
import torch


def image_reward_get_reward(
    model, pil_image, prompt, weight_dtype
):
  """Gets rewards using ImageReward model."""
  image = (
      model.preprocess(pil_image).unsqueeze(0).to(weight_dtype).to(model.device)
  )
  image_embeds = model.blip.visual_encoder(image)
  image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
      model.device
  )

  text_input = model.blip.tokenizer(
      prompt,
      padding="max_length",
      truncation=True,
      max_length=35,
      return_tensors="pt",
  ).to(model.device)
  text_output = model.blip.text_encoder(
      text_input.input_ids,
      attention_mask=text_input.attention_mask,
      encoder_hidden_states=image_embeds,
      encoder_attention_mask=image_atts,
      return_dict=True,
  )
  txt_features = text_output.last_hidden_state[:, 0, :]
  rewards = model.mlp(txt_features)
  rewards = (rewards - model.mean) / model.std
  return rewards, txt_features
