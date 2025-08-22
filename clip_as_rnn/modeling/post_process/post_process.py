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

"""Post processing."""

import torch
import torch.nn.functional as F

# pylint: disable=g-bad-import-order
# pylint: disable=g-importing-member
from modeling.post_process.object_discovery import get_instances
from utils.metrics import IoM


# This should be a abstract function to generate masks for the input image.
# However, we first hack it due to the time limit.
def generate_masks_from_sam(
    image_path, save_path, pipeline, img_sam=None, visualize=True
):
  """Generate masks from SAM."""
  masks, _, mask_list = pipeline.segment_automask(
      image_path=image_path,
      visualize=visualize,
      save_path=save_path,
      image=img_sam,
  )
  mask_tensor = torch.from_numpy(masks)
  mask_tensor = mask_tensor.float()
  return mask_tensor, mask_list


def match_masks(
    mask_tensor, attn_map, mask_list, iom_thres=0.0, min_pred_threshold=0.2
):
  """Match masks with the attention map according to the IoU.

  Args:
      mask_tensor: A torch.Tensor for the masks with shape [num_masks, height,
        width].
      attn_map: A torch.Tensor for the attention map with shape [1, 1, height,
        width].
      mask_list: A list of masks with shape [num_masks, height, width]
      iom_thres: A float for the threshold to apply to the attention map.
      min_pred_threshold: The prediction score threshold.

  Returns:
      A list of matched_masks with shape [num_masks, height, width],
      len(matched_masks) = number of captions
  """
  predictions = attn_map.squeeze(1).detach()
  iom = IoM(predictions, mask_tensor, min_pred_threshold=min_pred_threshold)
  keep_mask = iom > iom_thres
  # mask_tensor = mask_tensor[keep_mask]
  new_list = []
  for mid, m_dict in enumerate(mask_list):
    if keep_mask[mid]:
      new_list.append(m_dict)
  #  if not len(new_list):
  if not new_list:
    max_id = torch.argmax(iom)
    new_list.append(mask_list[max_id])
  return new_list


def post_process_mask(attn_masks, pad=None, min_area_ratio=0.15):
  """Post process attention masks."""
  if pad is not None:
    left, top, width, height = pad
    attn_masks = attn_masks[Ellipsis, top : top + height, left : left + width]
  else:
    height = None
    width = None
  mask_area = attn_masks.sum(dim=(1, 2))
  total_area = mask_area.sum()
  keep_mask = mask_area / total_area > min_area_ratio
  if torch.sum(keep_mask) == 0:
    if keep_mask.shape[0] == 0:
      return torch.zeros(
          (1, height, width), device=attn_masks.device, dtype=attn_masks.dtype
      )
    keep_mask[torch.argmax(mask_area)] = True
  attn_masks = attn_masks[keep_mask]
  return attn_masks


def filter_masks(
    attn_masks,
    pad=None,
    mask_threshold=0.3,
    min_area_ratio=0.15,
    return_largest=False,
    device=None,
    return_instances=False,
):
  """Filter attention mask below the threshold."""
  attn_masks[attn_masks < mask_threshold] = 0
  # get_instances will be operated on cpu
  ins_masks = get_instances(attn_masks, return_largest=return_largest)
  ins_masks = [post_process_mask(m, pad, min_area_ratio) for m in ins_masks]
  ins_masks = list(filter(lambda x: x is not None, ins_masks))
  ins_masks = [m.to(device) for m in ins_masks]
  if not return_instances:
    return [torch.any(m, dim=0, keepdim=True).to(m.dtype) for m in ins_masks]
  return ins_masks


def post_process(
    input_array,
    attn_masks,
    pad=None,
    mask_threshold=0.3,
    return_largest=False,
    min_area_ratio=0.15,
    return_instances=False,
):
  """post process the input tensor with the attention masks.

  Args:
      input_array: A np.ndarray input array to be post processed with shape
        [width, height, 3, batch_size]
      attn_masks: A torch.Tensor for the attention masks with shape [1,
        num_texts, width, height]
      pad: A list of padding: [pad_left, pad_top, width, height], where
        pad_left, pad_top and width, height are int values.
      mask_threshold: The threshold to binarize the mask.
      return_largest: If true, return the largest connected component.
      min_area_ratio: Keep the mask if its area is larger than this threshold.
      return_instances: Whether to return instances or not.

  Returns:
      attn_masks: A list of tensors with shape [num_instances, height, width]
          x num_texts, where len(attn_masks) = num_texts.
      NOTE: the number_instances for each text (class) may vary.
      The output is a binary tensor.
  """
  if len(attn_masks.shape) == 3:
    attn_masks = attn_masks[None]
  img_width, img_height = input_array.shape[:2]
  attn_masks = F.interpolate(
      attn_masks, size=(img_height, img_width), mode='bicubic'
  ).squeeze(0)
  device = attn_masks.device
  output_masks = filter_masks(
      attn_masks,
      pad=pad,
      mask_threshold=mask_threshold,
      min_area_ratio=min_area_ratio,
      return_largest=return_largest,
      device=device,
      return_instances=return_instances,
  )
  if pad is not None:
    left, top, width, height = pad
    input_array = input_array[top : top + height, left : left + width]
  return input_array, output_masks
