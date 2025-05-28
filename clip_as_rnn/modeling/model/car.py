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

"""Implementation of CaR."""

import os

import clip
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# pylint: disable=g-importing-member
# pylint: disable=g-bad-import-order
from modeling.model.clip_wrapper import CLIPWrapper
from modeling.model.clip_wrapper import forward_clip
from modeling.model.clipcam import CLIPCAM
from modeling.model.crf import PostProcess
from modeling.model.utils import apply_visual_prompts
from utils.visualize import viz_attn


class CaR(nn.Module):
  """CaR module."""

  def __init__(
      self,
      cfg,
      device="cpu",
      visualize=False,
      confidence_threshold=0.45,
      save_path="save_path",
      seg_mode="refer",
      semantic_clip_model_name=None,
      semantic_pretrained_data=None,
      semantic_templates=None,
      text_template=None,
      visual_prompt_type="circle",
      clipes_threshold=0.4,
      cam_text_template="a clean origami {}.",
      bg_cls=None,
      iom_thres=0.6,
      min_pred_threshold=0.01,
      bg_factor=1.0,
      mask_threshold=0.5,
  ):
    """CaR model for image segmentation.

    Args:
        cfg: the config file.
        device: the device to run the model.
        visualize: whether to visualize the intermediate results
        confidence_threshold: the confidence threshold for semantic
          segmentation. If the confidence score is lower than the threshold, the
          mask will be discarded.
        save_path: the path to save the intermediate results
        seg_mode: the segmentation mode, can be 'refer' or 'semantic'
        semantic_clip_model_name: the name of the semantic segmentation model.
        semantic_pretrained_data: the path to the pretrained semantic
          segmentation model.
        semantic_templates: the templates for semantic segmentation.
        text_template: the template for visual prompting.
        visual_prompt_type: the type of visual prompting.
        clipes_threshold: the threshold for CLIPES.
        cam_text_template: the template for CAM.
        bg_cls: background classes.
        iom_thres: IoM threshold.
        min_pred_threshold: Prediction threshold.
        bg_factor: Background factor.
        mask_threshold: Mask threshold.
    """
    super(CaR, self).__init__()
    # CLIP parameters
    self.confidence_threshold = confidence_threshold
    self.device = device
    self.visualize = visualize
    self.save_path = save_path
    self.seg_mode = seg_mode
    self.semantic_clip_model_name = semantic_clip_model_name
    self.semantic_pretrained_data = semantic_pretrained_data
    self.semantic_templates = semantic_templates
    self.text_template = text_template
    self.visual_prompt_type = visual_prompt_type
    self.clipes_threshold = clipes_threshold
    self.cam_text_template = cam_text_template
    self.iom_thres = iom_thres
    self.min_pred_threshold = min_pred_threshold
    self.bg_cls = bg_cls
    self.bg_factor = bg_factor
    self.mask_threshold = mask_threshold

    if not hasattr(cfg, "clip"):
      raise ValueError("The config file should contain the CLIP parameters.")

    if not hasattr(cfg, "car"):
      raise ValueError("The config file should contain the car parameters.")

    if hasattr(cfg, "cam"):
      raise ValueError("cfg.cam is deprecated, please use cfg.car ")

    for k, v in vars(cfg.clip).items():
      setattr(self, k, v)

    for k, v in vars(cfg.car).items():
      setattr(self, k, v)

    if hasattr(cfg, "sam"):
      for k, v in vars(cfg.sam).items():
        setattr(self, k, v)
    if not self.bg_cls:
      self.bg_cls = None
    print(f"The model is running on {self.device}")
    self.clip_model, self.preprocess = clip.load(
        self.clip_model_name, device=self.device
    )
    self.clip_model = CLIPWrapper(self.clip_model)
    self.post_process = PostProcess(device=self.device)
    self.mask_generator = CLIPCAM(
        self.clip_model,
        device=self.device,
        text_template=self.text_template,
        threshold=self.clipes_threshold,
        bg_cls=self.bg_cls,
    )
    self.semantic_clip_model, self.semantic_preprocess = clip.load(
        self.semantic_clip_model_name, device=self.device
    )
    self.semantic_clip_model = CLIPWrapper(self.semantic_clip_model)

  def get_confidence(self, cam_map, binary_cam_map):
    confidence_map = torch.sum(cam_map * binary_cam_map[None], dim=[2, 3])
    confidence_map = confidence_map / torch.sum(binary_cam_map, dim=[1, 2])
    confidence_score = confidence_map.squeeze()
    return confidence_score

  def set_visual_prompt_type(self, visual_prompt_type):
    self.visual_prompt_type = visual_prompt_type

  def set_bg_factor(self, bg_factor):
    self.bg_factor = bg_factor

  def set_confidence_threshold(self, confidence_threshold):
    self.confidence_threshold = confidence_threshold

  def set_mask_threshold(self, mask_threshold):
    self.mask_threshold = mask_threshold

  def apply_visual_prompts(self, image, mask):
    if torch.sum(mask).item() <= 1:
      return image
    image_array = np.array(image)
    img_h, img_w = image_array.shape[0:2]
    mask = (
        F.interpolate(mask[None][None], size=(img_h, img_w), mode="nearest")
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )
    mask = (mask > self.mask_threshold).astype(np.uint8)
    prompted_image = apply_visual_prompts(
        image_array, mask, self.visual_prompt_type, self.visualize
    )
    return prompted_image

  def get_mask_confidence(self, prompted_images, prompt_text):
    """Get the confidene for each mask with visual prompting."""
    # get the center, width and height of the mask
    prompted_tensor = torch.stack(
        [self.semantic_preprocess(img) for img in prompted_images], dim=0
    )
    prompted_tensor = prompted_tensor.to(self.device)
    h, w = prompted_tensor.shape[-2:]
    text_prediction = forward_clip(
        self.semantic_clip_model, prompted_tensor, prompt_text, h, w
    )
    return text_prediction

  def _filter_texts(
      self, ori_mask_id, sem_scores, prompt_text, semantic_prompt_text
  ):
    """Remove false positive masks by score filtering and recall the backbone to get the CAM maps for the filtered texts."""
    if not ori_mask_id:
      max_id = np.argmax(sem_scores)
      ori_mask_id.append(max_id)
    filtered_text = [prompt_text[i] for i in ori_mask_id]
    filtered_clip_text = [
        [spt[i] for i in ori_mask_id] for spt in semantic_prompt_text
    ]
    return filtered_text, filtered_clip_text

  def _forward_stage(self, ori_img, cam_text, clip_text, semantic_prompt_text):
    mask_proposals = self.get_mask_proposals(ori_img, cam_text)
    num_texts = len(cam_text)
    ori_mask_id = []
    sem_scores = torch.zeros((num_texts,), device=self.device).float()
    prompted_imgs = [
        self.apply_visual_prompts(ori_img, cam_map)
        for cam_map in mask_proposals
    ]
    text_scores = self.get_mask_confidence(prompted_imgs, semantic_prompt_text)
    mask_scores = torch.diagonal(text_scores)
    for mask_idx, mask_score in enumerate(mask_scores):
      # record mask idx
      if mask_score > self.confidence_threshold:
        ori_mask_id.append(mask_idx)
      sem_scores[mask_idx] = mask_score
    sem_scores = sem_scores.cpu().detach().numpy()
    filtered_texts, semantic_prompt_text = self._filter_texts(
        ori_mask_id,
        sem_scores,
        clip_text,
        semantic_prompt_text=semantic_prompt_text,
    )

    all_scores = torch.zeros((num_texts,), device=self.device).float()
    sem_scores = torch.from_numpy(sem_scores).to(self.device)
    for new_id, ori_id in enumerate(ori_mask_id):
      if new_id >= len(mask_proposals):
        # the mask is filtered out.
        continue
      all_scores[ori_id] = sem_scores[ori_id]
    return filtered_texts, all_scores, mask_proposals, semantic_prompt_text

  def _get_save_path(self, text):
    folder_name = "_".join([t.replace(" ", "_") for t in text])
    if len(folder_name) > 20:
      folder_name = folder_name[:20]
    output_path = os.path.join(self.save_path, folder_name)
    sub_output_path = [
        os.path.join(output_path, t.replace(" ", "_")) for t in text
    ]
    return output_path, sub_output_path

  def get_mask_proposals(self, img, text):
    if self.seg_mode == "refer":
      if isinstance(img, list):
        cam_map_list = [self.mask_generator(i, t)[0] for i, t in zip(img, text)]
      else:
        cam_map_list = [self.mask_generator(img, t)[0] for t in text]
      return torch.cat(cam_map_list, dim=0)
    elif self.seg_mode == "semantic":
      return self.mask_generator(img, text)[0]
    else:
      raise ValueError(
          "Unknown segmentation mode. Only refer and semantic segmentation are"
          " supported."
      )

  def _forward_car(self, ori_img, text):
    if isinstance(text, str):
      text = [text]
    _, sub_output_path = self._get_save_path(text)
    image_array = np.array(ori_img)
    clip_text = [self.cam_text_template.format(t) for t in text]
    cam_text = text
    init_clip_text = clip_text  # the text prompts of CLIP is different.
    semantic_prompt_text = clip_text
    # Apply semantic prompting augmentation.
    if self.semantic_templates is not None:
      semantic_prompt_text = []
      for template in self.semantic_templates:
        templated_text = [template.format(t) for t in text]
        semantic_prompt_text.append(templated_text)

    num_positive_last = 0
    run = 0
    prev_mask_proposals = None
    while True:
      run += 1
      cur_texts, all_scores, mask_proposals, semantic_prompt_text = (
          self._forward_stage(
              ori_img, cam_text, clip_text, semantic_prompt_text
          )
      )
      if cur_texts:  # if there is no text, skip the refinement
        cam_text = cur_texts
        clip_text = cur_texts

      if run > 0 and (torch.max(mask_proposals) == 0).item():
        mask_proposals = prev_mask_proposals

      prev_mask_proposals = mask_proposals
      num_positive = (all_scores > 0).sum().item()
      if num_positive == num_positive_last:
        # stop the refinement if the number of positive masks
        # does not change.
        break
      num_positive_last = num_positive
    # Apply densecrf for refinement.
    # SAM is optional and is applied outside the model.
    refined_masks = self.post_process(
        ori_img,
        mask_proposals,
        separate=self.seg_mode == "refer",
        bg_factor=self.bg_factor,
    )
    predicted_class_idx = [init_clip_text.index(t) for t in cur_texts]
    if self.visualize:
      _ = [
          viz_attn(
              image_array,
              attn,
              prefix=sub_output_path[aid],
              img_name="semantic_mask",
          )
          for aid, attn in enumerate(refined_masks)
      ]
    final_predicted_masks = torch.zeros(len(text), *refined_masks[0].shape)
    final_all_scores = torch.zeros(len(text))
    for idx, mask, score in zip(predicted_class_idx, refined_masks, all_scores):
      final_predicted_masks[idx] = mask
      final_all_scores[idx] = score
    return final_predicted_masks, final_all_scores

  def forward(self, im_ori, text):
    # raw_image_np is the padded image input with shape (512, 512, 3)
    pseudo_masks, conf_scores = self._forward_car(im_ori, text)
    return pseudo_masks, conf_scores
