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
import abc
from typing import Callable, Dict, Optional, Tuple

from flax.linen import initializers
import gin
import jax
from jax import lax
import jax.numpy as jnp

from modeling import base
from modeling import clip_models
from modeling import fpn
from modeling import heads
from ops import generate_detections
from ops import roi_ops
from ops import spatial_transform_ops
from ops import target_ops
from utils import box_utils
from utils.types import Array
from utils.types import DictArray
from utils.types import DType
from utils.types import ModelFn
from utils.types import MultilevelFeature
from utils.types import NestedDictArray

_EPSILON = 1e-7


def safe_divide(x,
                y,
                rtol = 1e-5,
                atol = 1e-8):
  """Computes a safe divide which returns 0 if the denominator is zero.

  Reference:
  https://www.tensorflow.org/api_docs/python/tf/math/divide_no_nan
  Args:
    x: A float of numerator.
    y: A float of denominator.
    rtol: The relative tolerance parameter. See numpy.isclose for more info.
    atol: The absolute tolerance parameter. See numpy.isclose for more info.

  Returns:
    z: output x / y or 0.
  """
  is_zero = jnp.isclose(y, 0.0, rtol=rtol, atol=atol)
  safe_y = jnp.where(is_zero, jnp.ones_like(y), y)
  return jnp.where(is_zero, jnp.zeros_like(x), x / safe_y)


class AbstractTaskHead(metaclass=abc.ABCMeta):
  """Abstract class for task heads in the multi-task models.

  The task head takes vision_features, text_features, the fused_features from
  the fusion model, and the task specific labels/target_outputs to generate the
  predictions and logits for the task specific losses and metrics.
  """

  @abc.abstractmethod
  def __call__(self, vision_features, text_features,
               fused_features, paddings,
               labels):
    """Abstract method that defines the protocol for the subclasses.

    The task head takes all of the vision, text, fused features and the task
    specific output/targets as the input and generates the logits and
    predictions.

    Args:
      vision_features: Vision features from the image encoders. It could be a a
        dictionary of level features of shape [batch, height, width,
        feature_dim]. Or just an image level ndarray features with shape [batch,
        height, width, feature_dim]. It could also be none if the task head only
        relies on the fused_features.
      text_features: Text features from the text_encoders in shape [batch,
        seq_len, feature_dim]. It could be none if the task head only relies on
        the fused_features.
      fused_features: Encoded features for both input images and input texts.
        The dimension for the fused features are [batch, len, feature_dim].
      paddings: Optional paddings that indicate which values are valid (1) and
        which are padded (0) in the fused_features. Paddings is in shape [batch,
        fused_feature_len].
      labels: The nested dict array for task-specific labels.

    Returns:
      A nested dictionary that contains the task specific output logits and
      predictions that will be used in losses and metrics computations.
    """


class BaseTaskHead(AbstractTaskHead, base.BaseModel):
  """Module interface of task heads."""


@gin.register
class ClipFasterRCNNHead(BaseTaskHead):
  """CLIP Faster R-CNN head function.

  Attributes:
    dtype: A jax data type.
    batch_norm_group_size: The batch norm group size.
    temperature_scale: The scale of the learnable temperature. This is only
      useful when use_text_feat is True.
    base_vlm_weight: A float of VLM score weight for the base classes.
    novel_vlm_weight: A float of VLM score weight for the novel classes. When
      the base_indicator is not given, we use novel_vlm_weight by default.
    use_frozen_vlm: A bool whether to use frozen backbone inference.
    objectness_weight: A float of objectenss score weight. The final detection
      score is the combination of objectness score and VLM scores.
    obj_rpn_nms_threshold_train: A float of NMS threhsold in the RPN when using
      objectness prediction. Defaults to 0.7.
    obj_rpn_nms_threshold_test: A float of NMS threhsold in the RPN when using
      objectness prediction. Defaults to 1.0.
    roi_output_size: An integer of the ROI-ALIGN crop size.
    include_mask: Whether to predict mask.
    output_decoded_boxes: Whether to output decoded boxes as prediction.
    backbone: A nn.Module specifying which backbone to use, e.g. ResNet.
    feature_pyramid: A nn.Module specifying which multilevel feature to use.
    region_proposal_head: A nn.Module specifying the architecture of region
      proposal prediction head.
    fastrcnn_head: A nn.Module specifying the architecture of detection head.
    maskrcnn_head: A nn.Module specifying the architecture of instance mask
      prediction head.
    roi_head_fn: A function to pool the ROI features into class embeddings.
    roi_feature_fn: A function to crop and resize multilevel region features.
    roi_scale_factor: A float scale of ROI crop size.
    generate_rois_fn: A function to generate the regions from region proposal
      head predictions.
    generate_rois_decoded_boxes_fn: A function to generate the regions and the
      decoded boxes from region proposalhead predictions.
    generate_detections_fn: A function to generate box detections from the
      classification and box regression outputs of fastrcnn head via non-max
      suppression and score thresholding.
    sample_rois_fn: A function to sample ROIs and provides training targets for
      fast rcnn head.
    sample_masks_fn: A function to sample ROIs and provides training targets for
      mask rcnn head.

  """
  dtype: DType = jnp.float32
  batch_norm_group_size: int = 0
  temperature_scale: float = 0.1
  clip_sim_temp: float = 0.01
  base_vlm_weight: float = 0.35
  novel_vlm_weight: float = 0.65
  use_frozen_vlm: bool = False
  objectness_weight: float = 0.0
  obj_rpn_nms_threshold_train: float = 0.7
  obj_rpn_nms_threshold_test: float = 1.0
  roi_output_size: int = 7
  include_mask: bool = False
  output_decoded_boxes: bool = False
  feature_pyramid: Optional[ModelFn] = fpn.Fpn
  region_proposal_head: ModelFn = heads.RpnHead
  fastrcnn_head: ModelFn = heads.FastrcnnHead
  maskrcnn_head: ModelFn = heads.MaskrcnnHead
  roi_head_fn: Callable[Ellipsis, Array] = clip_models.AttentionPool
  roi_feature_fn: Callable[Ellipsis, Array] = (
      spatial_transform_ops.multilevel_crop_and_resize)
  roi_scale_factor: Optional[float] = None
  generate_rois_fn: Callable[Ellipsis, Tuple[Array, Array]] = (
      roi_ops.multilevel_propose_rois)
  generate_rois_and_decoded_boxes_fn: Callable[
      Ellipsis, Tuple[Array, Array, Dict[int, Array]]] = (
          roi_ops.multilevel_propose_rois)
  generate_detections_fn: Callable[Ellipsis, Dict[str, Array]] = (
      generate_detections.process_and_generate_detections)
  sample_rois_fn: Callable[Ellipsis, Tuple[Array, Ellipsis]] = (
      target_ops.sample_box_targets)
  sample_masks_fn: Callable[Ellipsis, Tuple[Array, Ellipsis]] = (
      target_ops.sample_mask_targets)

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
    self.maskrcnn_head_fn = self.maskrcnn_head(
        **base.filter_attrs(self.maskrcnn_head, module_attrs))
    self.temperature = self.param(
        'temperature', initializers.ones, (1,))
    self.roi_head = self.roi_head_fn(name='attnpool')

  def compute_region_text_similarity(self,
                                     box_features,
                                     text_features,
                                     temp = None):
    """Compute ROI-text cosine similarity.

    Args:
      box_features: An array with shape (batch, num_rois, fc_dims).
      text_features: An array with shape (batch, num_classes, text_dims).
      temp: A float to control the scale of logits.

    Returns:
      class_outputs: An array with (batch, num_rois, num_classes).
    """
    box_norm = jnp.linalg.norm(box_features, axis=-1, keepdims=True)
    box_features = safe_divide(box_features, box_norm)
    text_norm = jnp.linalg.norm(text_features, axis=-1, keepdims=True)
    text_features = safe_divide(text_features, text_norm)
    class_outputs = jnp.einsum(
        'ijk,ilk->ijl', box_features, text_features).astype(jnp.float32)
    if temp is None:
      return class_outputs / (self.temperature_scale * self.temperature)
    else:
      return class_outputs / temp

  def ensemble_class_outputs(self,
                             class_outputs,
                             pretrained_class_outputs,
                             base_indicator = None):
    """Ensemble the Faster R-CNN class outputs with pretrained ones.

    Background probability comes from the Faster R-CNN classifier.

    Args:
      class_outputs: Predicted class probabilities in shape
        (batch, num_rois, num_classes).
      pretrained_class_outputs: Pretrained class probabilities in shape
        (batch, num_rois, num_classes).
      base_indicator: Indicator of which classes are seen in training time in
        shape (num_classes,). It's a float array with 0/1 values.

    Returns:
      ensembled_probs: Ensembled class probabilities in shape
        (batch, num_rois, num_classes).
    """
    # Convert to probabilities.
    class_probs = jax.nn.softmax(class_outputs, axis=-1)
    pretrained_probs = jax.nn.softmax(pretrained_class_outputs, axis=-1)
    if base_indicator is None:
      pretrained_weight = jnp.array([self.novel_vlm_weight])
      pretrained_weight = pretrained_weight.reshape(1, 1, 1)
      ensembled_probs = jnp.power(
          class_probs, (1.0 - pretrained_weight)) * jnp.power(
              pretrained_probs, pretrained_weight)
    else:
      base_indicator = base_indicator[:, None, :]  # Add num_rois dim.
      base_probs = jnp.power(class_probs,
                             1.0 - self.base_vlm_weight) * jnp.power(
                                 pretrained_probs, self.base_vlm_weight)
      novel_probs = jnp.power(class_probs,
                              1.0 - self.novel_vlm_weight) * jnp.power(
                                  pretrained_probs, self.novel_vlm_weight)
      ensembled_probs = jnp.where(base_indicator, base_probs, novel_probs)

    # Use detector background score.
    ensembled_probs = jnp.concatenate(
        [class_probs[:, :, :1], ensembled_probs[:, :, 1:]], axis=-1)
    # Renormalize the probability to 1.
    ensembled_probs /= jnp.sum(ensembled_probs, axis=-1, keepdims=True)
    return ensembled_probs

  def __call__(self,
               vision_features,
               text_features,
               image_text_features,
               paddings,
               labels,
               frozen_vision_features = None
               ):
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
      frozen_vision_features: Optional, a dictionary of level features of shape
        [N, H, W, C]. The keys are the feature levels, e.g. 2-6.

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
    del image_text_features
    max_level = max(vision_features.keys())
    key = self.make_rng('rng')
    if self.feature_pyramid:
      feature_map = self.feature_pyramid_fn(vision_features)
    else:
      feature_map = vision_features

    rpn_score_outputs, rpn_box_outputs = self.region_proposal_head_fn(
        feature_map)
    if self.output_decoded_boxes:
      obj_rpn_nms_threshold = (self.obj_rpn_nms_threshold_train
                               if self.mode == base.ExecutionMode.TRAIN
                               else self.obj_rpn_nms_threshold_test)
      rpn_rois, rpn_rois_scores, rpn_box_outputs = (
          self.generate_rois_and_decoded_boxes_fn(
              rpn_boxes=rpn_box_outputs,
              rpn_scores=rpn_score_outputs,
              anchor_boxes=labels['anchor_boxes'],
              image_shape=labels['image_info'][:, 1, :],
              rpn_nms_threshold=obj_rpn_nms_threshold,
              use_lrtb_boxes=True,
              output_decoded_boxes=True))
    else:
      rpn_rois, _ = self.generate_rois_fn(
          rpn_boxes=rpn_box_outputs,
          rpn_scores=rpn_score_outputs,
          anchor_boxes=labels['anchor_boxes'],
          image_shape=labels['image_info'][:, 1, :])
    model_outputs = {'rpn_score_outputs': rpn_score_outputs,
                     'rpn_box_outputs': rpn_box_outputs}

    if self.mode == base.ExecutionMode.TRAIN:
      rpn_rois = lax.stop_gradient(rpn_rois)
      rpn_rois, matched_gt_boxes, matched_gt_classes, matched_gt_indices = (
          self.sample_rois_fn(rpn_rois, labels['gt_boxes'],
                              labels['gt_classes'], key))
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
            labels['gt_masks'])
        model_outputs.update({
            'mask_targets': lax.stop_gradient(mask_targets),
            'mask_class_targets': mask_classes.astype(jnp.int32),
        })

    roi_features = self.roi_feature_fn(feature_map, rpn_rois, output_size=7)
    box_features, box_outputs = self.fastrcnn_head_fn(roi_features)
    class_outputs = self.compute_region_text_similarity(
        box_features, text_features)
    # Instantiate ROI head at training time by passing dummy ROI features.
    if self.mode == base.ExecutionMode.TRAIN:
      _ = self.roi_head(
          jnp.zeros((1, self.roi_output_size, self.roi_output_size,
                     vision_features[max_level].shape[-1])))
    else:
      if self.use_frozen_vlm:
        if frozen_vision_features is None:
          raise ValueError('frozen_vision_features is None.')
        top_feature_map = {max_level: frozen_vision_features[max_level]}
      else:
        top_feature_map = {max_level: vision_features[max_level]}

      if self.roi_scale_factor:
        scale_factor = self.roi_scale_factor
        scale_rpn_rois = box_utils.rescale_boxes(
            rpn_rois, labels['image_info'][:, 1], scale=scale_factor)
        pretrained_roi_features = self.roi_feature_fn(
            top_feature_map, scale_rpn_rois, output_size=self.roi_output_size)
      else:
        pretrained_roi_features = self.roi_feature_fn(
            top_feature_map, rpn_rois, output_size=self.roi_output_size)

      pretrained_box_features = self.roi_head(pretrained_roi_features)
      pretrained_class_outputs = self.compute_region_text_similarity(
          pretrained_box_features, text_features, self.clip_sim_temp)
      base_indicator = labels.get('base_category_indicator', None)
      class_outputs = self.ensemble_class_outputs(class_outputs,
                                                  pretrained_class_outputs,
                                                  base_indicator)
      if self.objectness_weight > 0.0:
        if not self.output_decoded_boxes:
          raise ValueError(
              'objectness score is used with output_decoded_boxes (lrtb).')
        batch, num_rois = rpn_rois_scores.shape
        rpn_rois_scores = jnp.reshape(rpn_rois_scores, [batch, num_rois, 1])
        class_outputs = jnp.power(
            rpn_rois_scores, self.objectness_weight) * class_outputs
      detection_results = self.generate_detections_fn(
          box_outputs, class_outputs, rpn_rois,
          labels['image_info'][:, 1:2, :])
      model_outputs.update(detection_results)
      # Add these for the host evaluator.
      if 'source_id' in labels and 'image_info' in labels:
        model_outputs.update({
            'source_id': labels['source_id'],
            'image_info': labels['image_info'],
        })
      if self.include_mask:
        mask_rpn_rois = detection_results['detection_boxes']
        mask_classes = detection_results['detection_classes'].astype(jnp.int32)

    model_outputs.update({
        'class_outputs': class_outputs,
        'box_outputs': box_outputs,
    })

    if self.include_mask:
      mask_roi_features = self.roi_feature_fn(
          feature_map, mask_rpn_rois, output_size=14)
      mask_outputs = self.maskrcnn_head_fn(mask_roi_features, mask_classes)
      if self.mode == base.ExecutionMode.TRAIN:
        mask_outputs = {'mask_outputs': mask_outputs}
      else:
        mask_outputs = {'detection_masks': jax.nn.sigmoid(mask_outputs)}
      model_outputs.update(mask_outputs)

    return model_outputs
