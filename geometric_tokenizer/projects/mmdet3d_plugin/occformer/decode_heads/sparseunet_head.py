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

# pylint: skip-file
from typing import List
import numpy as np
from projects.mmdet3d_plugin.utils import fast_hist_crop, per_class_iu
import spconv.pytorch as spconv
from timm.models.layers import trunc_normal_
import torch
from torch import nn as nn
from torch import Tensor
import torch.nn.functional as F


class SparseUNetHead(nn.Module):

  def __init__(
      self,
      channels,
      num_classes,
      train_cfg,
      test_cfg,
      classifier=False,
      **kwargs,
  ):
    super().__init__(channels, num_classes, **kwargs)
    self.conv_seg = None
    self.classifier = classifier
    if self.classifier:
      # self.classifier = spconv.SubMConv3d(channels, num_classes, kernel_size=1, padding=1, bias=True)
      self.classifier = nn.Linear(channels, num_classes, bias=True)
      trunc_normal_(self.classifier.weight, std=0.02)
      if self.classifier.bias is not None:
        nn.init.constant_(self.classifier.bias, 0)
    else:
      self.classifier = nn.Identity()

  def predict(self, inputs, batch_data_samples):
    seg_logits = self.forward(inputs)

    batch_idx = torch.cat(
        [data_samples.batch_idx for data_samples in batch_data_samples]
    )
    seg_logit_list = []
    for i, data_sample in enumerate(batch_data_samples):
      seg_logit = seg_logits[batch_idx == i]
      seg_logit = seg_logit[data_sample.point2voxel_map]
      seg_logit_list.append(seg_logit)

    return seg_logit_list

  def forward(self, x):
    return self.classifier(x)

  def forward_train(
      self,
      voxel_feats,
      img_metas,
      gt_occ,
      points=None,
      scatter=None,
      voxel_coors=None,
      addition_dict=None,
      **kwargs,
  ):
    seg_logit = self(voxel_feats[0])
    batch_size = voxel_coors[-1, 0] + 1


    losses = dict()
    losses['loss_sem_seg'] = self.loss_decode(
        seg_logit, gt_occ, ignore_index=self.ignore_index
    )

    # get the lidar segmentation metric
    # we have 2 options
    # 1. use grid prediction and then interplate; 2. use 'point2voxel_map'

    # 1. forward_lidarseg
    # losses_lidarseg = self.forward_lidarseg(seg_logit_grid, points, img_metas)

    # 2. use 'point2voxel_map'
    losses_lidarseg = self.forward_lidarseg_idxmap(
        seg_logit,
        gt_occ,
        points,
        img_metas=img_metas,
        pts2voxel_map=addition_dict['point2voxel_map'],
        range_mask=addition_dict['range_mask'],
    )
    # print(losses_lidarseg)

    losses.update(losses_lidarseg)

    return losses

  def simple_test(
      self,
      voxel_feats,
      voxel_labels,
      img_metas,
      gt_occ,
      points=None,
      scatter=None,
      voxel_coors=None,
      addition_dict=None,
      **kwargs,
  ):
    seg_logit = self(voxel_feats[0])
    batch_size = voxel_coors[-1, 0] + 1




    # res = {
    #         'output_voxels': [seg_logit_grid],
    #         'output_points': None,
    #         }
    res = {
        'output_voxels': None,
        'output_points': None,
    }

    # res['output_points'] = self.forward_lidarseg(
    #     voxel_preds=seg_logit_grid,
    #     points=points,
    #     img_metas=img_metas,
    # )

    res['output_points'] = self.forward_lidarseg_idxmap(
        voxel_preds=seg_logit,
        voxel_labels=voxel_labels,
        points=points,
        img_metas=img_metas,
        pts2voxel_map=addition_dict['point2voxel_map'],
        range_mask=addition_dict['range_mask'],
    )
    # res['output_points'] = self.forward_lidarseg_test2(
    #     voxel_preds=seg_logit,
    #     voxel_labels=voxel_labels,
    #     points=points,
    #     img_metas=img_metas,
    #     pts2voxel_map=addition_dict['point2voxel_map'], range_mask=addition_dict['range_mask']
    # )

    # print(res['output_points_test'])

    return res

  def compute_miou_training(self, logits, target):
    # compute the lidarseg metric
    assert logits.shape[0] == target.shape[0]
    output_clses = torch.argmax(logits[:, 1:], dim=1) + 1
    target_np = target.cpu().numpy()
    output_clses_np = output_clses.cpu().numpy()

    unique_label = np.arange(16)
    hist = fast_hist_crop(output_clses_np, target_np, unique_label)
    iou = per_class_iu(hist)
    return iou

  def forward_lidarseg(
      self,
      voxel_preds,
      points,
      img_metas=None,
      pts2voxel_map=None,
      seg_logit=None,
  ):
    pc_range = torch.tensor(img_metas[0]['pc_range']).type_as(voxel_preds)
    pc_range_min = pc_range[:3]
    pc_range = pc_range[3:] - pc_range_min

    # point_logits_test = seg_logit[pts2voxel_map]

    point_logits = []
    for batch_index, points_i in enumerate(points):
      points_i = (points_i[:, :3].float() - pc_range_min) / pc_range
      points_i = (points_i * 2) - 1
      points_i = points_i[Ellipsis, [2, 1, 0]]
      # points_i = points_i[..., [2, 0, 1]]

      out_of_range_mask = (points_i < -1) | (points_i > 1)
      out_of_range_mask = out_of_range_mask.any(dim=1)
      points_i = points_i.view(1, 1, 1, -1, 3)

      # # transfer voxel_preds to probability
      # voxel_preds = torch.softmax(voxel_preds, dim=1)


      point_logits_i = F.grid_sample(
          voxel_preds[batch_index : batch_index + 1],
          points_i,
          mode='bilinear',
          padding_mode=self.padding_mode,
          align_corners=False,
      )


      point_logits_i = point_logits_i.squeeze().t().contiguous()  # [b, n, c]
      point_logits.append(point_logits_i)

    point_logits = torch.cat(point_logits, dim=0)

    if self.training:
      point_labels = torch.cat([x[:, -1] for x in points]).long()
      # compute the lidarseg metric
      output_clses = torch.argmax(point_logits[:, 1:], dim=1) + 1
      target_points_np = point_labels.cpu().numpy()
      output_clses_np = output_clses.cpu().numpy()

      unique_label = np.arange(16)
      hist = fast_hist_crop(output_clses_np, target_points_np, unique_label)
      iou = per_class_iu(hist)
      loss_dict = {}
      loss_dict['point_mean_iou'] = torch.tensor(np.nanmean(iou)).cuda()
      return loss_dict
    else:
      return torch.softmax(point_logits, dim=1)

  def forward_lidarseg_test(
      self,
      voxel_preds,
      points,
      img_metas=None,
      pts2voxel_map=None,
      seg_logit=None,
  ):
    pc_range = torch.tensor(img_metas[0]['pc_range']).type_as(voxel_preds)
    pc_range_min = pc_range[:3]
    pc_range = pc_range[3:] - pc_range_min

    # point_logits_test = seg_logit[pts2voxel_map]

    point_logits = []
    for batch_index, points_i in enumerate(points):
      points_i = (points_i[:, :3].float() - pc_range_min) / pc_range
      points_i = (points_i * 2) - 1
      points_i = points_i[Ellipsis, [2, 1, 0]]

      out_of_range_mask = (points_i < -1) | (points_i > 1)
      out_of_range_mask = out_of_range_mask.any(dim=1)
      points_i = points_i.view(1, 1, 1, -1, 3)
      point_logits_i = F.grid_sample(
          voxel_preds[batch_index : batch_index + 1],
          points_i,
          mode='bilinear',
          padding_mode=self.padding_mode,
          align_corners=False,
      )
      point_logits_i = point_logits_i.squeeze().t().contiguous()  # [b, n, c]
      point_logits.append(point_logits_i)

    point_logits = torch.cat(point_logits, dim=0)

    point_labels = torch.cat([x[:, -1] for x in points]).long()
    # compute the lidarseg metric
    output_clses = torch.argmax(point_logits[:, 1:], dim=1) + 1
    target_points_np = point_labels.cpu().numpy()
    output_clses_np = output_clses.cpu().numpy()

    unique_label = np.arange(16)
    hist = fast_hist_crop(output_clses_np, target_points_np, unique_label)
    iou = per_class_iu(hist)
    loss_dict = {}
    loss_dict['point_mean_iou'] = torch.tensor(np.nanmean(iou)).cuda()
    return loss_dict

  def forward_lidarseg_test2(
      self,
      voxel_preds,
      voxel_labels,
      points,
      img_metas=None,
      pts2voxel_map=None,
      range_mask=None,
  ):
    point_logits = voxel_preds[pts2voxel_map]
    point_labels = torch.cat([x[:, -1] for x in points]).long()

    if range_mask is not None:
      assert point_labels.shape[0] == range_mask.shape[0]
      point_labels = point_labels[range_mask]
      assert point_labels.shape[0] == point_logits.shape[0]
      # point_labels_mask = point_labels < 255
      # point_logits, point_labels = point_logits[point_labels_mask], point_labels[point_labels_mask]

    # compute the lidarseg metric
    voxel_miou = self.compute_miou_training(voxel_preds, voxel_labels)

    if point_logits.shape[1] == 17:
      output_clses = torch.argmax(point_logits[:, 1:], dim=1) + 1
    else:
      output_clses = torch.argmax(point_logits, dim=1)
    target_points_np = point_labels.cpu().numpy()
    output_clses_np = output_clses.cpu().numpy()

    unique_label = np.arange(16)
    hist = fast_hist_crop(output_clses_np, target_points_np, unique_label)
    iou = per_class_iu(hist)
    loss_dict = {}
    loss_dict['point_mean_iou'] = torch.tensor(np.nanmean(iou)).cuda()
    loss_dict['voxel_mean_iou'] = torch.tensor(np.nanmean(voxel_miou)).cuda()
    print(
        'point mean iou: {} ;  voxel_mean_iou: {}'.format(
            loss_dict['point_mean_iou'], loss_dict['voxel_mean_iou']
        )
    )
    # print((output_clses_np==target_points_np).sum()/output_clses_np.shape[0])
    return torch.softmax(point_logits, dim=1)

  def forward_lidarseg_idxmap(
      self,
      voxel_preds,
      voxel_labels,
      points,
      img_metas=None,
      pts2voxel_map=None,
      range_mask=None,
  ):
    point_logits = voxel_preds[pts2voxel_map]
    point_labels = torch.cat([x[:, -1] for x in points]).long()

    if range_mask is not None:
      assert point_labels.shape[0] == range_mask.shape[0]
      point_labels = point_labels[range_mask]
      assert point_labels.shape[0] == point_logits.shape[0]
      # point_labels_mask = point_labels < 255
      # point_logits, point_labels = point_logits[point_labels_mask], point_labels[point_labels_mask]

    if self.training:
      # compute the lidarseg metric
      voxel_miou = self.compute_miou_training(voxel_preds, voxel_labels)

      if point_logits.shape[1] == 17:
        output_clses = torch.argmax(point_logits[:, 1:], dim=1) + 1
      else:
        output_clses = torch.argmax(point_logits, dim=1)
      target_points_np = point_labels.cpu().numpy()
      output_clses_np = output_clses.cpu().numpy()

      unique_label = np.arange(16)
      hist = fast_hist_crop(output_clses_np, target_points_np, unique_label)
      iou = per_class_iu(hist)
      loss_dict = {}
      loss_dict['point_mean_iou'] = torch.tensor(np.nanmean(iou)).cuda()
      loss_dict['voxel_mean_iou'] = torch.tensor(np.nanmean(voxel_miou)).cuda()

      # print((output_clses_np==target_points_np).sum()/output_clses_np.shape[0])
      return loss_dict
    else:
      return torch.softmax(point_logits, dim=1)

  def forward_lidarseg_idxmap_combine(
      self,
      voxel_preds_grid,
      voxel_preds,
      points,
      img_metas=None,
      pts2voxel_map=None,
      range_mask=None,
  ):
    pc_range = torch.tensor(img_metas[0]['pc_range']).type_as(voxel_preds)
    pc_range_min = pc_range[:3]
    pc_range = pc_range[3:] - pc_range_min

    point_logits_withinrange = voxel_preds[pts2voxel_map]
    range_mask_out = ~range_mask
    points_len = sum([len(points_i) for points_i in points])
    point_logits = torch.zeros(points_len, voxel_preds.size(1)).type_as(
        voxel_preds
    )
    point_logits_out = []

    for batch_index, points_i in enumerate(points):
      points_i = points_i[range_mask_out]  # only choose the points out-of-range
      points_i = (points_i[:, :3].float() - pc_range_min) / pc_range
      points_i = (points_i * 2) - 1
      points_i = points_i[Ellipsis, [2, 1, 0]]
      # points_i = points_i[..., [2, 0, 1]]

      # out_of_range_mask = (points_i < -1) | (points_i > 1)
      # out_of_range_mask = out_of_range_mask.any(dim=1)
      points_i = points_i.view(1, 1, 1, -1, 3)

      # # transfer voxel_preds to probability
      point_logits_i = F.grid_sample(
          voxel_preds_grid[batch_index : batch_index + 1],
          points_i,
          mode='bilinear',
          padding_mode=self.padding_mode,
          align_corners=self.align_corners,
      )




      point_logits_i = point_logits_i.squeeze().t().contiguous()  # [b, n, c]
      point_logits_out.append(point_logits_i)

    point_logits_out = torch.cat(point_logits_out, dim=0)
    point_logits[range_mask] = point_logits_withinrange
    point_logits[range_mask_out] = point_logits_out

    if self.training:
      point_labels = torch.cat([x[:, -1] for x in points]).long()
      # if range_mask is not None:
      #     point_labels = point_labels[range_mask]

      # compute the lidarseg metric
      output_clses = torch.argmax(point_logits[:, 1:], dim=1) + 1
      target_points_np = point_labels.cpu().numpy()
      output_clses_np = output_clses.cpu().numpy()

      unique_label = np.arange(16)
      hist = fast_hist_crop(output_clses_np, target_points_np, unique_label)
      iou = per_class_iu(hist)
      loss_dict = {}
      loss_dict['point_mean_iou'] = torch.tensor(np.nanmean(iou)).cuda()

      # print((output_clses_np==target_points_np).sum()/output_clses_np.shape[0])
      return loss_dict
    else:
      return torch.softmax(point_logits, dim=1)
