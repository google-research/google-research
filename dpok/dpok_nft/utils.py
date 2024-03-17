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
