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

"""Mask RCNN model with multi-task learning in JAX."""

# See issue #620.
# pytype: disable=wrong-arg-count

from typing import Any, Callable, Dict, Tuple, Union, Optional
from absl import logging

import gin
import jax
from jax import lax
import jax.numpy as jnp


from moe_mtl.modeling import cls_heads
from moe_mtl.modeling import uvit_vmoe_mtl

ModuleDef = Any
Array = jnp.ndarray
LevelArray = Dict[int, Array]
NamedLevelArray = Dict[str, Optional[Union[Array, LevelArray]]]


@gin.register
class MaskrcnnModelMTL(base.BaseModel):
  """Mask R-CNN MTL model function.

  Attributes:
    dtype: A jax data type.
    batch_norm_group_size: The batch norm group size.
    backbone: A nn.Module specifying which backbone to use, e.g. ResNet.
    feature_pyramid: A nn.Module specifying which multilevel feature to use.
    region_proposal_head: A nn.Module specifying the architecture of region
      proposal prediction head.
    fastrcnn_head: A nn.Module specifying the architecture of detection head.
    maskrcnn_head: A nn.Module specifying the architecture of instance mask
      prediction head.
    roi_feature_fn: A function to crop and resize multilevel region features.
    generate_rois_fn: A function to generate the regions from region proposal
      head predictions.
    generate_detections_fn: A function to generate box detections from the
      classification and box regression outputs of fastrcnn head via non-max
      suppression and score thresholding.
    sample_rois_fn: A function to sample ROIs and provides training targets for
      fast rcnn head.
    sample_masks_fn: A function to sample ROIs and provides training targets for
      mask rcnn head.
    classification_head: A classification head
  """
  dtype: jnp.dtype = jnp.float32
  batch_norm_group_size: int = 0
  backbone: ModuleDef = uvit_vmoe_mtl.UViTVMoEMTL
  feature_pyramid: ModuleDef = fpn.Fpn
  classification_head = cls_heads.ClassificationHead
  region_proposal_head: ModuleDef = heads.RpnHead
  fastrcnn_head: ModuleDef = heads.FastrcnnHead
  maskrcnn_head: ModuleDef = heads.MaskrcnnHead
  include_mask: bool = True
  roi_feature_fn: Callable[Ellipsis, Array] = (
      spatial_transform_ops.multilevel_crop_and_resize)
  generate_rois_fn: Callable[Ellipsis, Tuple[Array, Array]] = (
      roi_ops.multilevel_propose_rois)
  generate_detections_fn: Callable[Ellipsis, Dict[str, Array]] = (
      generate_detections.process_and_generate_detections)
  sample_rois_fn: Callable[Ellipsis, Tuple[Array, Ellipsis]] = (
      target_ops.sample_box_targets)
  sample_masks_fn: Callable[Ellipsis, Tuple[Array, Ellipsis]] = (
      target_ops.sample_mask_targets)
  pool_type_cls: str = 'gap'
  pool_type_det: str = 'gap'

  def setup(self):
    module_fn_and_self_names = [
        (self.backbone, 'backbone_fn'),
        (self.feature_pyramid, 'feature_pyramid_fn'),
        (self.region_proposal_head, 'region_proposal_head_fn'),
        (self.fastrcnn_head, 'fastrcnn_head_fn'),
        (self.maskrcnn_head, 'maskrcnn_head_fn'),
        (self.classification_head, 'classification_head_fn')
    ]

    attrs = {
        'train': (self.mode == base.ExecutionMode.TRAIN),
        'mode': self.mode,
        'dtype': self.dtype,
        'batch_norm_group_size': self.batch_norm_group_size
    }

    for module_fn, self_name in module_fn_and_self_names:
      module = module_fn(
          mode=self.mode) if self_name == 'backbone_fn' else module_fn()
      kwargs = {a: v for a, v in attrs.items() if hasattr(module, a)}
      setattr(self, self_name, module_fn(**kwargs))

  @gin_utils.allow_remapping
  def __call__(
      self,
      images_det,
      images_cls,
      labels_det,
      labels_cls,
      second_stage = False,
      return_rois = False):
    """Mask RCNN model.

    Args:
      images_det: An array of shape [batch_size, height, width, channels].
      images_cls: An array of shape [batch_size, height, width, channels].
      labels_det: A dictionary with the following key-value pairs: 'image_info':
        An array that encodes the information of the image and the applied
        preprocessing. The shape is [batch_size, 4, 2]. It is in the format of:
        [[original_height, original_width], [desired_height, desired_width],
        [y_scale, x_scale], [y_offset, x_offset]], where [desired_height,
        desired_width] is the actual scaled image size, and [y_scale, x_scale]
        is the scaling factor, which is the ratio of scaled dimension / original
        dimension. 'anchor_boxes': ordered dictionary with keys [min_level,
        min_level+1, ..., max_level]. The values are arrays with shape [batch,
        height_l * width_l, 4] representing anchor boxes at each level.
        'gt_boxes': Groundtruth bounding box annotations. The box is represented
        in [y1, x1, y2, x2] format. The coordinates are w.r.t the scaled image
        that is fed to the network. The array is padded with -1 to the fixed
        dimension [batch_size, max_num_instances, 4] along axis 1. 'gt_classes':
        Groundtruth classes annotations. The array is padded with -1 to the
        fixed dimension [batch_size, max_num_instances] along axis 1.
        max_num_instances is defined in the input parser. 'gt_masks':
        Groundtrugh masks cropped by the bounding box and resized to a fixed
        size determined by mask_target_size. The array is padded with -1 to the
        fixed dimension [batch_size, max_num_instances, mask_target_size,
        mask_target_size] along axis 1.
      labels_cls: An array of shape [batch_size, num_of_classes].
      second_stage: Whether enable the Stable Router or not
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
        'mask_outputs': An array representing the prediction for each mask,
          with a shape of
          [batch_size, num_masks, mask_height, mask_width].
        'mask_targets': An array representing the binary mask of ground truth
          labels for each mask with a shape of
          [batch_size, num_masks, mask_height, mask_width].
        'mask_class_targets': An array with a shape of [batch_size, num_masks],
          representing the classes of mask targets.
        'roi_features': if return_rois is True, returns a
          [batch_size, num_rois, output_size, output_size, features] array of
          the selected roi features.
    """
    key = self.make_rng('rng')
    model_outputs = {}
    backbone_features = self.backbone_fn(
        images_det, images_cls, second_stage=second_stage)
    if isinstance(backbone_features, tuple):
      backbone_features_det, backbone_features_cls, metrics = backbone_features
    else:
      backbone_features_det, backbone_features_cls = backbone_features
      metrics = None

    if self.pool_type_cls == 'gap':
      backbone_features_cls = backbone_features_cls.mean(
          axis=tuple(range(1, backbone_features_cls.ndim - 1)))
    elif self.pool_type_cls == 'tok':
      backbone_features_cls = backbone_features_cls[:, 0]
    else:
      raise NotImplementedError

    fpn_features = self.feature_pyramid_fn(backbone_features_det)
    rpn_score_outputs, rpn_box_outputs = self.region_proposal_head_fn(
        fpn_features)
    classification_features = self.classification_head_fn(backbone_features_cls)
    logging.info(jax.tree.map(lambda x: x.shape, labels_det))
    rpn_rois, _ = self.generate_rois_fn(
        rpn_boxes=rpn_box_outputs,
        rpn_scores=rpn_score_outputs,
        anchor_boxes=labels_det['anchor_boxes'],
        image_shape=labels_det['image_info'][:, 1, :])

    if self.mode == base.ExecutionMode.TRAIN:
      rpn_rois = lax.stop_gradient(rpn_rois)
      rpn_rois, matched_gt_boxes, matched_gt_classes, matched_gt_indices = (
          self.sample_rois_fn(rpn_rois, labels_det['gt_boxes'],
                              labels_det['gt_classes'], key))
      # Create bounding box training targets.
      box_targets = box_utils.encode_boxes(
          matched_gt_boxes, rpn_rois, weights=[10.0, 10.0, 5.0, 5.0])
      # If the target is background, the box target is set to all 0s.
      box_targets = jnp.where(
          jnp.tile(matched_gt_classes[Ellipsis, None] == 0, [1, 1, 4]),
          jnp.zeros_like(box_targets), box_targets)
      model_outputs.update({
          'class_targets': matched_gt_classes,
          'box_targets': box_targets,
      })
      if self.include_mask:
        mask_rpn_rois, mask_classes, mask_targets = self.sample_masks_fn(
            rpn_rois, matched_gt_boxes, matched_gt_classes, matched_gt_indices,
            labels_det['gt_masks'])
        model_outputs.update({
            'mask_targets': lax.stop_gradient(mask_targets),
            'mask_class_targets': mask_classes.astype(jnp.int32),
        })

    roi_features = self.roi_feature_fn(
        fpn_features, rpn_rois, output_size=7, use_einsum_gather=True)
    class_outputs, box_outputs = self.fastrcnn_head_fn(roi_features)

    if self.mode != base.ExecutionMode.TRAIN:
      detection_results = self.generate_detections_fn(
          box_outputs, class_outputs, rpn_rois,
          labels_det['image_info'][:, 1:2, :])
      mask_rpn_rois = detection_results['detection_boxes']
      mask_classes = detection_results['detection_classes'].astype(jnp.int32)
      model_outputs.update(detection_results)

    model_outputs.update({
        'rpn_score_outputs': rpn_score_outputs,
        'rpn_box_outputs': rpn_box_outputs,
        'class_outputs': class_outputs,
        'box_outputs': box_outputs,
    })

    if self.include_mask or return_rois:
      mask_roi_features = self.roi_feature_fn(
          fpn_features, mask_rpn_rois, output_size=14, use_einsum_gather=True)

      if self.include_mask:
        mask_outputs = self.maskrcnn_head_fn(mask_roi_features, mask_classes)
        if self.mode == base.ExecutionMode.TRAIN:
          mask_outputs = {'mask_outputs': mask_outputs}
        else:
          mask_outputs = {'detection_masks': jax.nn.sigmoid(mask_outputs)}
        model_outputs.update(mask_outputs)

      if return_rois:
        model_outputs.update({
            'roi_features': mask_roi_features,
            'roi_boxes': mask_rpn_rois,
        })

    model_outputs['metrics'] = metrics
    model_outputs['logits'] = classification_features
    return model_outputs
