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

"""DenseCRF."""

import numpy as np
from pydensecrf import densecrf as dcrf
from pydensecrf import utils
import torch
import torch.nn.functional as F


class DenseCRF(object):
  """DenseCRF class."""

  def __init__(self, iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std):
    self.iter_max = iter_max
    self.pos_w = pos_w
    self.pos_xy_std = pos_xy_std
    self.bi_w = bi_w
    self.bi_xy_std = bi_xy_std
    self.bi_rgb_std = bi_rgb_std

  def __call__(self, image, probmap):
    c, h, w = probmap.shape

    u = utils.unary_from_softmax(probmap)
    u = np.ascontiguousarray(u)

    image = np.ascontiguousarray(image)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(u)
    d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
    d.addPairwiseBilateral(
        sxy=self.bi_xy_std,
        srgb=self.bi_rgb_std,
        rgbim=image,
        compat=self.bi_w,
    )

    q = d.inference(self.iter_max)
    q = np.array(q).reshape((c, h, w))

    return q


class PostProcess:
  """Post processing with dense CRF."""

  def __init__(self, device):
    self.device = device
    self.postprocessor = DenseCRF(
        iter_max=10,
        pos_xy_std=1,
        pos_w=3,
        bi_xy_std=67,
        bi_rgb_std=3,
        bi_w=4,
    )

  def apply_crf(self, image, cams, bg_factor=1.0):
    """Apply dense CRF."""
    bg_score = np.power(1 - np.max(cams, axis=0, keepdims=True), bg_factor)
    cams = np.concatenate((bg_score, cams), axis=0)
    prob = cams

    image = image.astype(np.uint8).transpose(1, 2, 0)
    prob = self.postprocessor(image, prob)

    label = np.argmax(prob, axis=0)

    label_tensor = torch.from_numpy(label).long()
    refined_mask = F.one_hot(label_tensor).to(device=self.device)
    refined_mask = refined_mask.permute(2, 0, 1)
    refined_mask = refined_mask[1:].float()
    return refined_mask

  def __call__(self, image, cams, separate=False, bg_factor=1.0):
    mean_bgr = (104.008, 116.669, 122.675)
    # covert Image to numpy array
    image = np.array(image).astype(np.float32)

    # RGB -> BGR
    image = image[:, :, ::-1]
    # Mean subtraction
    image -= mean_bgr
    # HWC -> CHW
    image = image.transpose(2, 0, 1)

    if isinstance(cams, torch.Tensor):
      cams = cams.cpu().detach().numpy()
    if separate:
      refined_mask = [
          self.apply_crf(image, cam[None], bg_factor) for cam in cams
      ]
      refined_mask = torch.cat(refined_mask, dim=0)
    else:
      refined_mask = self.apply_crf(image, cams, bg_factor)

    return refined_mask
