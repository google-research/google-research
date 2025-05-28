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

"""Calculate CAM with CLIP model."""

import warnings

import clip
import cv2
import numpy as np
import torch

# pylint: disable=g-importing-member
# pylint: disable=g-bad-import-order
from modeling.model.cam import CAM
from modeling.model.cam import scale_cam_image
from modeling.model.utils import img_ms_and_flip
from modeling.model.utils import reshape_transform
from modeling.model.utils import scoremap2bbox

warnings.filterwarnings("ignore")


class ClipOutputTarget:

  def __init__(self, category):
    self.category = category

  def __call__(self, model_output):
    if len(model_output.shape) == 1:
      return model_output[self.category]
    return model_output[:, self.category]


def zeroshot_classifier(classnames, templates, model, device):
  """Zeroshot classifier."""
  with torch.no_grad():
    zeroshot_weights = []
    for classname in classnames:
      if templates is None:
        texts = [classname]
      else:
        # format with class
        texts = [template.format(classname) for template in templates]
      texts = clip.tokenize(texts).to(device)  # tokenize
      class_embeddings = model.encode_text(texts)  # embed with text encoder
      class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
      class_embedding = class_embeddings.mean(dim=0)
      class_embedding /= class_embedding.norm()
      zeroshot_weights.append(class_embedding)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
  return zeroshot_weights.t()


class CLIPCAM:
  """Generate CAM with CLIP model."""

  def __init__(
      self,
      clip_model,
      device,
      text_template=None,
      threshold=0.4,
      bg_cls=None,
  ):
    self.device = device
    self.clip_model = clip_model.to(device)
    self.text_template = text_template
    self.threshold = threshold
    self.stride = self.clip_model.visual.patch_size

    # if self.dataset_name == 'voc' else BACKGROUND_CATEGORY_COCO
    self.bg_cls = bg_cls
    self.bg_text_features = None
    if self.bg_cls is not None:
      self.bg_text_features = zeroshot_classifier(
          self.bg_cls,
          ("a clean origami {}.",),
          self.clip_model,
          self.device,
      ).to(self.device)
    self.target_layers = [self.clip_model.visual.transformer.resblocks[-1].ln_1]
    self.cam = CAM(
        model=self.clip_model,
        target_layers=self.target_layers,
        reshape_transform=reshape_transform,
        use_cuda="cuda" in device,
        stride=self.stride,
    )

  def set_bg_cls(self, bg_cls):
    # if len(bg_cls) == 0:
    if not bg_cls:
      self.bg_cls = None
      self.bg_text_features = None
    else:
      self.bg_cls = bg_cls
      self.bg_text_features = zeroshot_classifier(
          self.bg_cls,
          ("a clean origami {}.",),
          self.clip_model,
          self.device,
      ).to(self.device)

  def __call__(self, ori_img, text, scale=1.0):
    """Get CAM masks and features.

    Args:
      ori_img(Image): image to be searched.
      text (str): text to be searched.
      scale (float): image scale.
    Returns:
      CAM masks and features.
    """
    ori_width = ori_img.size[0]
    ori_height = ori_img.size[1]
    if isinstance(text, str):
      text = [text]

    # convert image to bgr channel
    ms_imgs = img_ms_and_flip(ori_img, ori_height, ori_width, scales=[scale])
    image = ms_imgs[0]

    image = image.unsqueeze(0)
    h, w = image.shape[-2], image.shape[-1]
    image = image.to(self.device)
    image_features, attn_weight_list = self.clip_model.encode_image(image, h, w)

    highres_cam_to_save = []
    refined_cam_to_save = []
    # keys = []

    # [bg_id_for_each_image[im_idx]].to(device_id)
    bg_features_temp = None
    if self.bg_text_features is not None:
      bg_features_temp = self.bg_text_features.to(self.device)
    fg_features_temp = zeroshot_classifier(
        text, self.text_template, self.clip_model, self.device
    ).to(self.device)
    if bg_features_temp is None:
      text_features_temp = fg_features_temp
    else:
      text_features_temp = torch.cat(
          [fg_features_temp, bg_features_temp], dim=0
      )
    input_tensor = [
        image_features,
        text_features_temp.to(self.device),
        h,
        w,
    ]

    # for idx, label in enumerate(label_list):
    # keys.append(new_class_names.index(label))
    for idx, _ in enumerate(text):
      targets = [ClipOutputTarget(idx)]

      # torch.cuda.empty_cache()
      grayscale_cam, _, attn_weight_last = self.cam(
          input_tensor=input_tensor, targets=targets, target_size=None
      )  # (ori_width, ori_height))

      grayscale_cam = grayscale_cam[0, :]
      if grayscale_cam.max() == 0:
        input_tensor_fg = (
            image_features,
            fg_features_temp.to(self.device),
            h,
            w,
        )
        grayscale_cam, _, attn_weight_last = self.cam(
            input_tensor=input_tensor_fg,
            targets=targets,
            target_size=None,
        )
        grayscale_cam = grayscale_cam[0, :]

      grayscale_cam_highres = cv2.resize(grayscale_cam, (ori_width, ori_height))
      highres_cam_to_save.append(torch.tensor(grayscale_cam_highres))

      if idx == 0:
        attn_weight_list.append(attn_weight_last)
        attn_weight = [
            aw[:, 1:, 1:] for aw in attn_weight_list
        ]  # (b, hxw, hxw)
        attn_weight = torch.stack(attn_weight, dim=0)[-8:]
        attn_weight = torch.mean(attn_weight, dim=0)
        attn_weight = attn_weight[0].cpu().detach()
      attn_weight = attn_weight.float()

      box, cnt = scoremap2bbox(
          scoremap=grayscale_cam,
          threshold=self.threshold,
          multi_contour_eval=True,
      )
      aff_mask = torch.zeros((grayscale_cam.shape[0], grayscale_cam.shape[1]))
      for i_ in range(cnt):
        x0_, y0_, x1_, y1_ = box[i_]
        aff_mask[y0_:y1_, x0_:x1_] = 1

      aff_mask = aff_mask.view(
          1, grayscale_cam.shape[0] * grayscale_cam.shape[1]
      )
      aff_mat = attn_weight

      trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
      trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)

      for _ in range(2):
        trans_mat = trans_mat / torch.sum(trans_mat, dim=0, keepdim=True)
        trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)
      trans_mat = (trans_mat + trans_mat.transpose(1, 0)) / 2

      # This is copied from CLIP-ES
      for _ in range(1):
        trans_mat = torch.matmul(trans_mat, trans_mat)

      trans_mat = trans_mat * aff_mask

      cam_to_refine = torch.FloatTensor(grayscale_cam)
      cam_to_refine = cam_to_refine.view(-1, 1)

      # (n,n) * (n,1)->(n,1)
      cam_refined = torch.matmul(trans_mat, cam_to_refine).reshape(
          h // self.stride, w // self.stride
      )
      cam_refined = cam_refined.cpu().numpy().astype(np.float32)
      cam_refined_highres = scale_cam_image(
          [cam_refined], (ori_width, ori_height)
      )[0]
      refined_cam_to_save.append(torch.tensor(cam_refined_highres))

      # post process the cam map
      # label = process(raw_image, refined_cam, postprocessor)
      # vis_img = vis_mask(np.asarray(raw_image), label, [0, 255, 0])
      # vis_img.save(f'clip_es_crf_{idx}.jpg')

    # keys = torch.tensor(keys)
    # cam_all_scales.append(torch.stack(cam_to_save,dim=0))

    cam_masks = torch.stack(refined_cam_to_save, dim=0)

    return cam_masks.to(self.device), fg_features_temp.to(self.device)
