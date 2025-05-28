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

"""Task heads for multi-task modeling API.
"""

import jax.numpy as jnp

from findit import base
from findit import fpn
from findit import generate_detections
from findit import heads
from findit import roi_ops
from findit import spatial_transform_ops


class FasterRCNNHead(base.BaseModel):
  """Faster R-CNN head function.

  Attributes:
    use_image_text_feat: Whether to use image/text fused features as inputs.
    dtype: A jax data type.
    batch_norm_group_size: The batch norm group size.
    backbone: A nn.Module specifying which backbone to use, e.g. ResNet.
    feature_pyramid: A nn.Module specifying which multilevel feature to use.
    region_proposal_head: A nn.Module specifying the architecture of region
      proposal prediction head.
    fastrcnn_head: A nn.Module specifying the architecture of detection head.
  """
  use_image_text_feat: bool = True
  dtype = jnp.float32
  batch_norm_group_size: int = 0
  feature_pyramid = fpn.Fpn
  region_proposal_head = heads.RpnHead
  fastrcnn_head = heads.FastrcnnHead

  def setup(self):
    module_attrs = {
        'train': (self.mode == base.ExecutionMode.TRAIN),
        'mode': self.mode,
        'dtype': self.dtype,
        'batch_norm_group_size': self.batch_norm_group_size
    }
    if self.feature_pyramid:
      self.feature_pyramid_fn = self.feature_pyramid(
          **base.filter_attrs(self.feature_pyramid, module_attrs))
    self.region_proposal_head_fn = self.region_proposal_head(
        **base.filter_attrs(self.region_proposal_head, module_attrs))
    self.fastrcnn_head_fn = self.fastrcnn_head(
        **base.filter_attrs(self.fastrcnn_head, module_attrs))

  def __call__(self,
               vision_features,
               text_features,
               image_text_features,
               paddings,
               labels,
               return_rois = False):
    """Faster RCNN model.

    Args:
      vision_features: A dictionary of level features of shape [N, H, W, C]. The
        keys are the feature levels, e.g. 2-6.
      text_features: An array of text features. Unused when use_text_feat=False.
      image_text_features: A dictionary of level features of shape [N, H, W, C].
        The keys are the feature levels, e.g. 2-6.
      paddings: Optional paddings. Not used here.
      labels: A dictionary with the following key-value pairs:
        'image_info': An array that encodes the information of the image and the
          applied preprocessing. The shape is [batch_size, 4, 2]. It is in the
          format of: [[original_height, original_width], [desired_height,
            desired_width], [y_scale, x_scale], [y_offset, x_offset]], where
            [desired_height, desired_width] is the actual scaled image size, and
            [y_scale, x_scale] is the scaling factor, which is the ratio of
            scaled dimension / original dimension.
        'anchor_boxes': ordered dictionary with keys [min_level, min_level+1,
          ..., max_level]. The values are arrays with shape [batch, height_l *
          width_l, 4] representing anchor boxes at each level.
        'gt_boxes': Groundtruth bounding box annotations. The box is represented
          in [y1, x1, y2, x2] format. The coordinates are w.r.t the scaled image
          that is fed to the network. The array is padded with -1 to the fixed
          dimension [batch_size, max_num_instances, 4] along axis 1.
        'gt_classes': Groundtruth classes annotations. The array is padded with
          -1 to the fixed dimension [batch_size, max_num_instances] along axis
          1. max_num_instances is defined in the input parser.
      return_rois: A boolean if True also returns the roi features as part of
        the outputs.

    Returns:
      model_outputs: A dictionary with the following key-value pairs:
        'rpn_score_outputs': A dictionary with keys representing levels and
          values representing scores in [batch_size, height, width,
          num_anchors].
        'rpn_box_outputs': A dictionary with keys representing levels and values
          representing box regression targets in
          [batch_size, height, width, num_anchors * 4].
        'class_outputs': An array representing the class prediction for each
          box with a shape of [batch_size, num_boxes, num_classes].
        'box_outputs': An array representing the box prediction for each box
          with a shape of [batch_size, num_boxes, num_classes * 4].
        'class_targets': An array representing the class label for each box
          with a shape of [batch_size, num_boxes].
        'box_targets': An array representing the box label for each box
          with a shape of [batch_size, num_boxes, 4].
        'roi_features': if return_rois is True, returns a
          [batch_size, num_rois, output_size, output_size, features] array of
          the selected roi features.
    """
    del paddings
    del text_features
    feature_map = (
        image_text_features if self.use_image_text_feat else vision_features)
    if feature_map is None:
      raise ValueError('Feature map must not be None!')

    if self.feature_pyramid:
      feature_map = self.feature_pyramid_fn(feature_map)

    rpn_score_outputs, rpn_box_outputs = self.region_proposal_head_fn(
        feature_map)
    rpn_rois, _ = roi_ops.multilevel_propose_rois(
        rpn_box_outputs,
        rpn_score_outputs,
        labels['anchor_boxes'],
        labels['image_info'][:, 1, :])
    model_outputs = {'rpn_score_outputs': rpn_score_outputs,
                     'rpn_box_outputs': rpn_box_outputs}
    roi_features = spatial_transform_ops.multilevel_crop_and_resize(
        feature_map, rpn_rois, 7)
    class_outputs, box_outputs = self.fastrcnn_head_fn(roi_features)
    model_outputs.update({
        'class_outputs': class_outputs,
        'box_outputs': box_outputs,
    })
    detection_results = generate_detections.process_and_generate_detections(
        box_outputs, class_outputs, rpn_rois,
        labels['image_info'][:, 1:2, :])
    model_outputs.update(detection_results)
    return model_outputs

