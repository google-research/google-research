# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""SSD300 Model Configuration.

References:
  Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed,
  Cheng-Yang Fu, Alexander C. Berg
  SSD: Single Shot MultiBox Detector
  arXiv:1512.02325

Ported from MLPerf reference implementation:
  https://github.com/mlperf/reference/tree/ssd/single_stage_detector/ssd

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os
import re
import threading
import tensorflow.compat.v1 as tf

from cnn_quantization.tf_cnn_benchmarks import constants
from cnn_quantization.tf_cnn_benchmarks import mlperf
from cnn_quantization.tf_cnn_benchmarks import ssd_constants
from cnn_quantization.tf_cnn_benchmarks.cnn_util import log_fn
from cnn_quantization.tf_cnn_benchmarks.models import model as model_lib
from cnn_quantization.tf_cnn_benchmarks.models import resnet_model
from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib import layers as contrib_layers

BACKBONE_MODEL_SCOPE_NAME = 'resnet34_backbone'


class SSD300Model(model_lib.CNNModel):
  """Single Shot Multibox Detection (SSD) model for 300x300 image datasets."""

  def __init__(self, label_num=ssd_constants.NUM_CLASSES, batch_size=32,
               learning_rate=1e-3, backbone='resnet34', params=None):
    super(SSD300Model, self).__init__('ssd300', 300, batch_size, learning_rate,
                                      params=params)
    # For COCO dataset, 80 categories + 1 background = 81 labels
    self.label_num = label_num

    # Currently only support ResNet-34 as backbone model
    if backbone != 'resnet34':
      raise ValueError('Invalid backbone model %s for SSD.' % backbone)
    mlperf.logger.log(key=mlperf.tags.BACKBONE, value=backbone)

    # Number of channels and default boxes associated with the following layers:
    #   ResNet34 layer, Conv7, Conv8_2, Conv9_2, Conv10_2, Conv11_2
    self.out_chan = [256, 512, 512, 256, 256, 256]
    mlperf.logger.log(key=mlperf.tags.LOC_CONF_OUT_CHANNELS,
                      value=self.out_chan)

    # Number of default boxes from layers of different scales
    #   38x38x4, 19x19x6, 10x10x6, 5x5x6, 3x3x4, 1x1x4
    self.num_dboxes = [4, 6, 6, 6, 4, 4]
    mlperf.logger.log(key=mlperf.tags.NUM_DEFAULTS_PER_CELL,
                      value=self.num_dboxes)

    # TODO(haoyuzhang): in order to correctly restore in replicated mode, need
    # to create a saver for each tower before graph is finalized. Use variable
    # manager for better efficiency.
    self.backbone_savers = []

    # Collected predictions for eval stage. It maps each image id in eval
    # dataset to a dict containing the following information:
    #   source_id: raw ID of image
    #   raw_shape: raw shape of image
    #   pred_box: encoded box coordinates of prediction
    #   pred_scores: scores of classes in prediction
    self.predictions = {}

    # Global step when predictions are collected.
    self.eval_global_step = 0

    # Average precision. In asynchronous eval mode, this is the latest AP we
    # get so far and may not be the results at current eval step.
    self.eval_coco_ap = 0

    # Process, queues, and thread for asynchronous evaluation. When enabled,
    # create a separte process (async_eval_process) that continously pull
    # intermediate results from the predictions queue (a multiprocessing queue),
    # process them, and push final results into results queue (another
    # multiprocessing queue). The main thread is responsible to push message
    # into predictions queue, and start a separate thread to continuously pull
    # messages from results queue to update final results.
    # Message in predictions queue should be a tuple of two elements:
    #    (evaluation step, predictions)
    # Message in results queue should be a tuple of two elements:
    #    (evaluation step, final results)
    self.async_eval_process = None
    self.async_eval_predictions_queue = None
    self.async_eval_results_queue = None
    self.async_eval_results_getter_thread = None

    # The MLPerf reference uses a starting lr of 1e-3 at bs=32.
    self.base_lr_batch_size = 32

  def skip_final_affine_layer(self):
    return True

  def add_backbone_model(self, cnn):
    # --------------------------------------------------------------------------
    # Resnet-34 backbone model -- modified for SSD
    # --------------------------------------------------------------------------

    # Input 300x300, output 150x150
    cnn.conv(64, 7, 7, 2, 2, mode='SAME_RESNET', use_batch_norm=True)
    cnn.mpool(3, 3, 2, 2, mode='SAME')

    resnet34_layers = [3, 4, 6, 3]
    version = 'v1'

    # ResNet-34 block group 1
    # Input 150x150, output 75x75
    for i in range(resnet34_layers[0]):
      # Last argument forces residual_block to use projection shortcut, even
      # though the numbers of input and output channels are equal
      resnet_model.residual_block(cnn, 64, 1, version)

    # ResNet-34 block group 2
    # Input 75x75, output 38x38
    for i in range(resnet34_layers[1]):
      stride = 2 if i == 0 else 1
      resnet_model.residual_block(cnn, 128, stride, version, i == 0)

    # ResNet-34 block group 3
    # This block group is modified: first layer uses stride=1 so that the image
    # size does not change in group of layers
    # Input 38x38, output 38x38
    for i in range(resnet34_layers[2]):
      # The following line is intentionally commented out to differentiate from
      # the original ResNet-34 model
      # stride = 2 if i == 0 else 1
      resnet_model.residual_block(cnn, 256, stride, version, i == 0)

    # ResNet-34 block group 4: removed final block group
    # The following 3 lines are intentially commented out to differentiate from
    # the original ResNet-34 model
    # for i in range(resnet34_layers[3]):
    #   stride = 2 if i == 0 else 1
    #   resnet_model.residual_block(cnn, 512, stride, version, i == 0)

  def add_inference(self, cnn):
    cnn.use_batch_norm = True
    cnn.batch_norm_config = {'decay': ssd_constants.BATCH_NORM_DECAY,
                             'epsilon': ssd_constants.BATCH_NORM_EPSILON,
                             'scale': True}

    with tf.variable_scope(BACKBONE_MODEL_SCOPE_NAME):
      self.add_backbone_model(cnn)

    # --------------------------------------------------------------------------
    # SSD additional layers
    # --------------------------------------------------------------------------

    def add_ssd_layer(cnn, depth, k_size, stride, mode):
      return cnn.conv(
          depth,
          k_size,
          k_size,
          stride,
          stride,
          mode=mode,
          use_batch_norm=False,
          kernel_initializer=contrib_layers.xavier_initializer())

    # Activations for feature maps of different layers
    self.activations = [cnn.top_layer]
    # Conv7_1, Conv7_2
    # Input 38x38, output 19x19
    add_ssd_layer(cnn, 256, 1, 1, 'valid')
    self.activations.append(add_ssd_layer(cnn, 512, 3, 2, 'same'))

    # Conv8_1, Conv8_2
    # Input 19x19, output 10x10
    add_ssd_layer(cnn, 256, 1, 1, 'valid')
    self.activations.append(add_ssd_layer(cnn, 512, 3, 2, 'same'))

    # Conv9_1, Conv9_2
    # Input 10x10, output 5x5
    add_ssd_layer(cnn, 128, 1, 1, 'valid')
    self.activations.append(add_ssd_layer(cnn, 256, 3, 2, 'same'))

    # Conv10_1, Conv10_2
    # Input 5x5, output 3x3
    add_ssd_layer(cnn, 128, 1, 1, 'valid')
    self.activations.append(add_ssd_layer(cnn, 256, 3, 1, 'valid'))

    # Conv11_1, Conv11_2
    # Input 3x3, output 1x1
    add_ssd_layer(cnn, 128, 1, 1, 'valid')
    self.activations.append(add_ssd_layer(cnn, 256, 3, 1, 'valid'))

    self.loc = []
    self.conf = []

    for nd, ac, oc in zip(self.num_dboxes, self.activations, self.out_chan):
      l = cnn.conv(
          nd * 4,
          3,
          3,
          1,
          1,
          input_layer=ac,
          num_channels_in=oc,
          activation=None,
          use_batch_norm=False,
          kernel_initializer=contrib_layers.xavier_initializer())
      scale = l.get_shape()[-1]
      # shape = [batch_size, nd * 4, scale, scale]
      l = tf.reshape(l, [self.batch_size, nd, 4, scale, scale])
      # shape = [batch_size, nd, 4, scale, scale]
      l = tf.transpose(l, [0, 1, 3, 4, 2])
      # shape = [batch_size, nd, scale, scale, 4]
      self.loc.append(tf.reshape(l, [self.batch_size, -1, 4]))
      # shape = [batch_size, nd * scale * scale, 4]

      c = cnn.conv(
          nd * self.label_num,
          3,
          3,
          1,
          1,
          input_layer=ac,
          num_channels_in=oc,
          activation=None,
          use_batch_norm=False,
          kernel_initializer=contrib_layers.xavier_initializer())
      # shape = [batch_size, nd * label_num, scale, scale]
      c = tf.reshape(c, [self.batch_size, nd, self.label_num, scale, scale])
      # shape = [batch_size, nd, label_num, scale, scale]
      c = tf.transpose(c, [0, 1, 3, 4, 2])
      # shape = [batch_size, nd, scale, scale, label_num]
      self.conf.append(tf.reshape(c, [self.batch_size, -1, self.label_num]))
      # shape = [batch_size, nd * scale * scale, label_num]

    # Shape of locs: [batch_size, NUM_SSD_BOXES, 4]
    # Shape of confs: [batch_size, NUM_SSD_BOXES, label_num]
    locs, confs = tf.concat(self.loc, 1), tf.concat(self.conf, 1)

    # Pack location and confidence outputs into a single output layer
    # Shape of logits: [batch_size, NUM_SSD_BOXES, 4+label_num]
    logits = tf.concat([locs, confs], 2)

    cnn.top_layer = logits
    cnn.top_size = 4 + self.label_num

    return cnn.top_layer

  def get_learning_rate(self, global_step, batch_size):
    rescaled_lr = self.get_scaled_base_learning_rate(batch_size)
    # Defined in MLPerf reference model
    boundaries = [160000, 200000]
    boundaries = [b * self.base_lr_batch_size // batch_size for b in boundaries]
    decays = [1, 0.1, 0.01]
    learning_rates = [rescaled_lr * d for d in decays]
    lr = tf.train.piecewise_constant(global_step, boundaries, learning_rates)
    warmup_steps = int(118287 / batch_size * 5)
    warmup_lr = (
        rescaled_lr * tf.cast(global_step, tf.float32) / tf.cast(
            warmup_steps, tf.float32))
    return tf.cond(global_step < warmup_steps, lambda: warmup_lr, lambda: lr)

  def get_scaled_base_learning_rate(self, batch_size):
    """Calculates base learning rate for creating lr schedule.

    In replicated mode, gradients are summed rather than averaged which, with
    the sgd and momentum optimizers, increases the effective learning rate by
    lr * num_gpus. Dividing the base lr by num_gpus negates the increase.

    Args:
      batch_size: Total batch-size.

    Returns:
      Base learning rate to use to create lr schedule.
    """
    base_lr = self.learning_rate
    if self.params.variable_update == 'replicated':
      base_lr = self.learning_rate / self.params.num_gpus
    scaled_lr = base_lr * (batch_size / self.base_lr_batch_size)
    return scaled_lr

  def _collect_backbone_vars(self):
    backbone_vars = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope='.*'+ BACKBONE_MODEL_SCOPE_NAME)
    var_list = {}

    # Assume variables in the checkpoint are following the naming convention of
    # a model checkpoint trained with TF official model
    # TODO(haoyuzhang): the following variable name parsing is hacky and easy
    # to break if there is change in naming convention of either benchmarks or
    # official models.
    for v in backbone_vars:
      # conv2d variable example (model <-- checkpoint):
      #   v/cg/conv24/conv2d/kernel:0 <-- conv2d_24/kernel
      if 'conv2d' in v.name:
        re_match = re.search(r'conv(\d+)/conv2d/(.+):', v.name)
        if re_match:
          layer_id = int(re_match.group(1))
          param_name = re_match.group(2)
          vname_in_ckpt = self._var_name_in_official_model_ckpt(
              'conv2d', layer_id, param_name)
          var_list[vname_in_ckpt] = v

      # batchnorm varariable example:
      #   v/cg/conv24/batchnorm25/gamma:0 <-- batch_normalization_25/gamma
      elif 'batchnorm' in v.name:
        re_match = re.search(r'batchnorm(\d+)/(.+):', v.name)
        if re_match:
          layer_id = int(re_match.group(1))
          param_name = re_match.group(2)
          vname_in_ckpt = self._var_name_in_official_model_ckpt(
              'batch_normalization', layer_id, param_name)
          var_list[vname_in_ckpt] = v

    return var_list

  def _var_name_in_official_model_ckpt(self, layer_name, layer_id, param_name):
    """Return variable names according to convention in TF official models."""
    vname_in_ckpt = layer_name
    if layer_id > 0:
      vname_in_ckpt += '_' + str(layer_id)
    vname_in_ckpt += '/' + param_name
    return vname_in_ckpt

  def loss_function(self, inputs, build_network_result):
    logits = build_network_result.logits

    # Unpack model output back to locations and confidence scores of predictions
    # Shape of pred_loc: [batch_size, NUM_SSD_BOXES, 4]
    # Shape of pred_label: [batch_size, NUM_SSD_BOXES, label_num]
    pred_loc, pred_label = tf.split(logits, [4, self.label_num], 2)

    # Shape of gt_loc: [batch_size, NUM_SSD_BOXES, 4]
    # Shape of gt_label: [batch_size, NUM_SSD_BOXES, 1]
    # Shape of num_gt: [batch_size]
    _, gt_loc, gt_label, num_gt = inputs
    gt_label = tf.cast(gt_label, tf.int32)

    box_loss = self._localization_loss(pred_loc, gt_loc, gt_label, num_gt)
    class_loss = self._classification_loss(pred_label, gt_label, num_gt)

    tf.summary.scalar('box_loss', tf.reduce_mean(box_loss))
    tf.summary.scalar('class_loss', tf.reduce_mean(class_loss))
    return class_loss + box_loss

  def _localization_loss(self, pred_loc, gt_loc, gt_label, num_matched_boxes):
    """Computes the localization loss.

    Computes the localization loss using smooth l1 loss.
    Args:
      pred_loc: a flatten tensor that includes all predicted locations. The
        shape is [batch_size, num_anchors, 4].
      gt_loc: a tensor representing box regression targets in
        [batch_size, num_anchors, 4].
      gt_label: a tensor that represents the classification groundtruth targets.
        The shape is [batch_size, num_anchors, 1].
      num_matched_boxes: the number of anchors that are matched to a groundtruth
        targets, used as the loss normalizater. The shape is [batch_size].
    Returns:
      box_loss: a float32 representing total box regression loss.
    """
    mask = tf.greater(tf.squeeze(gt_label), 0)
    float_mask = tf.cast(mask, tf.float32)

    smooth_l1 = tf.reduce_sum(tf.losses.huber_loss(
        gt_loc, pred_loc,
        reduction=tf.losses.Reduction.NONE
    ), axis=2)
    smooth_l1 = tf.multiply(smooth_l1, float_mask)
    box_loss = tf.reduce_sum(smooth_l1, axis=1)

    return tf.reduce_mean(box_loss / num_matched_boxes)

  def _classification_loss(self, pred_label, gt_label, num_matched_boxes):
    """Computes the classification loss.

    Computes the classification loss with hard negative mining.
    Args:
      pred_label: a flatten tensor that includes all predicted class. The shape
        is [batch_size, num_anchors, num_classes].
      gt_label: a tensor that represents the classification groundtruth targets.
        The shape is [batch_size, num_anchors, 1].
      num_matched_boxes: the number of anchors that are matched to a groundtruth
        targets. This is used as the loss normalizater.

    Returns:
      box_loss: a float32 representing total box regression loss.
    """
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        gt_label, pred_label, reduction=tf.losses.Reduction.NONE)

    mask = tf.greater(tf.squeeze(gt_label), 0)
    float_mask = tf.cast(mask, tf.float32)

    # Hard example mining
    neg_masked_cross_entropy = cross_entropy * (1 - float_mask)
    relative_position = contrib_framework.argsort(
        contrib_framework.argsort(
            neg_masked_cross_entropy, direction='DESCENDING'))
    num_neg_boxes = tf.minimum(
        tf.to_int32(num_matched_boxes) * ssd_constants.NEGS_PER_POSITIVE,
        ssd_constants.NUM_SSD_BOXES)
    top_k_neg_mask = tf.cast(tf.less(
        relative_position,
        tf.tile(num_neg_boxes[:, tf.newaxis], (1, ssd_constants.NUM_SSD_BOXES))
    ), tf.float32)

    class_loss = tf.reduce_sum(
        tf.multiply(cross_entropy, float_mask + top_k_neg_mask), axis=1)

    return tf.reduce_mean(class_loss / num_matched_boxes)

  def add_backbone_saver(self):
    # Create saver with mapping from variable names in checkpoint of backbone
    # model to variables in SSD model
    backbone_var_list = self._collect_backbone_vars()
    self.backbone_savers.append(tf.train.Saver(backbone_var_list))

  def load_backbone_model(self, sess, backbone_model_path):
    for saver in self.backbone_savers:
      saver.restore(sess, backbone_model_path)

  def get_input_data_types(self, subset):
    if subset == 'validation':
      return [self.data_type, tf.float32, tf.float32, tf.float32, tf.int32]
    return [self.data_type, tf.float32, tf.float32, tf.float32]

  def get_input_shapes(self, subset):
    """Return encoded tensor shapes for train and eval data respectively."""
    if subset == 'validation':
      # Validation data shapes:
      # 1. images
      # 2. ground truth locations of boxes
      # 3. ground truth classes of objects in boxes
      # 4. source image IDs
      # 5. raw image shapes
      return [
          [self.batch_size, self.image_size, self.image_size, self.depth],
          [self.batch_size, ssd_constants.MAX_NUM_EVAL_BOXES, 4],
          [self.batch_size, ssd_constants.MAX_NUM_EVAL_BOXES, 1],
          [self.batch_size],
          [self.batch_size, 3],
      ]

    # Training data shapes:
    # 1. images
    # 2. ground truth locations of boxes
    # 3. ground truth classes of objects in boxes
    # 4. numbers of objects in images
    return [
        [self.batch_size, self.image_size, self.image_size, self.depth],
        [self.batch_size, ssd_constants.NUM_SSD_BOXES, 4],
        [self.batch_size, ssd_constants.NUM_SSD_BOXES, 1],
        [self.batch_size]
    ]

  def accuracy_function(self, inputs, logits):
    """Returns the ops to measure the mean precision of the model."""
    try:
      from cnn_quantization.tf_cnn_benchmarks import ssd_dataloader  # pylint: disable=g-import-not-at-top
      from tensorflow_models.object_detection.box_coders import faster_rcnn_box_coder  # pylint: disable=g-import-not-at-top
      from tensorflow_models.object_detection.core import box_coder  # pylint: disable=g-import-not-at-top
      from tensorflow_models.object_detection.core import box_list  # pylint: disable=g-import-not-at-top
    except ImportError:
      raise ImportError('To use the COCO dataset, you must clone the '
                        'repo https://github.com/tensorflow/models and add '
                        'tensorflow/models and tensorflow/models/research to '
                        'the PYTHONPATH, and compile the protobufs; '
                        'To evaluate using COCO'
                        'metric, download and install Python COCO API from'
                        'https://github.com/cocodataset/cocoapi')

    # Unpack model output back to locations and confidence scores of predictions
    # pred_locs: relative locations (coordiates) of objects in all SSD boxes
    # shape: [batch_size, NUM_SSD_BOXES, 4]
    # pred_labels: confidence scores of objects being of all categories
    # shape: [batch_size, NUM_SSD_BOXES, label_num]
    pred_locs, pred_labels = tf.split(logits, [4, self.label_num], 2)

    ssd_box_coder = faster_rcnn_box_coder.FasterRcnnBoxCoder(
        scale_factors=ssd_constants.BOX_CODER_SCALES)
    anchors = box_list.BoxList(
        tf.convert_to_tensor(ssd_dataloader.DefaultBoxes()('ltrb')))
    pred_boxes = box_coder.batch_decode(
        encoded_boxes=pred_locs, box_coder=ssd_box_coder, anchors=anchors)

    pred_scores = tf.nn.softmax(pred_labels, axis=2)

    # TODO(haoyuzhang): maybe use `gt_boxes` and `gt_classes` for visualization.
    _, gt_boxes, gt_classes, source_id, raw_shape = inputs  # pylint: disable=unused-variable

    return {
        (constants.UNREDUCED_ACCURACY_OP_PREFIX +
         ssd_constants.PRED_BOXES): pred_boxes,
        (constants.UNREDUCED_ACCURACY_OP_PREFIX +
         ssd_constants.PRED_SCORES): pred_scores,
        # TODO(haoyuzhang): maybe use these values for visualization.
        # constants.UNREDUCED_ACCURACY_OP_PREFIX+'gt_boxes': gt_boxes,
        # constants.UNREDUCED_ACCURACY_OP_PREFIX+'gt_classes': gt_classes,
        (constants.UNREDUCED_ACCURACY_OP_PREFIX +
         ssd_constants.SOURCE_ID): source_id,
        (constants.UNREDUCED_ACCURACY_OP_PREFIX +
         ssd_constants.RAW_SHAPE): raw_shape
    }

  def postprocess(self, results):
    """Postprocess results returned from model."""
    try:
      from cnn_quantization.tf_cnn_benchmarks import coco_metric  # pylint: disable=g-import-not-at-top
    except ImportError:
      raise ImportError('To use the COCO dataset, you must clone the '
                        'repo https://github.com/tensorflow/models and add '
                        'tensorflow/models and tensorflow/models/research to '
                        'the PYTHONPATH, and compile the protobufs; '
                        'To evaluate using COCO'
                        'metric, download and install Python COCO API from'
                        'https://github.com/cocodataset/cocoapi')

    pred_boxes = results[ssd_constants.PRED_BOXES]
    pred_scores = results[ssd_constants.PRED_SCORES]
    # TODO(haoyuzhang): maybe use these values for visualization.
    # gt_boxes = results['gt_boxes']
    # gt_classes = results['gt_classes']
    source_id = results[ssd_constants.SOURCE_ID]
    raw_shape = results[ssd_constants.RAW_SHAPE]

    # COCO evaluation requires processing COCO_NUM_VAL_IMAGES exactly once. Due
    # to rounding errors (i.e., COCO_NUM_VAL_IMAGES % batch_size != 0), setting
    # `num_eval_epochs` to 1 is not enough and will often miss some images. We
    # expect user to set `num_eval_epochs` to >1, which will leave some unused
    # images from previous steps in `predictions`. Here we check if we are doing
    # eval at a new global step.
    if results['global_step'] > self.eval_global_step:
      self.eval_global_step = results['global_step']
      self.predictions.clear()

    for i, sid in enumerate(source_id):
      self.predictions[int(sid)] = {
          ssd_constants.PRED_BOXES: pred_boxes[i],
          ssd_constants.PRED_SCORES: pred_scores[i],
          ssd_constants.SOURCE_ID: source_id[i],
          ssd_constants.RAW_SHAPE: raw_shape[i]
      }

    # COCO metric calculates mAP only after a full epoch of evaluation. Return
    # dummy results for top_N_accuracy to be compatible with benchmar_cnn.py.
    if len(self.predictions) >= ssd_constants.COCO_NUM_VAL_IMAGES:
      log_fn('Got results for all {:d} eval examples. Calculate mAP...'.format(
          ssd_constants.COCO_NUM_VAL_IMAGES))

      annotation_file = os.path.join(self.params.data_dir,
                                     ssd_constants.ANNOTATION_FILE)
      # Size of predictions before decoding about 15--30GB, while size after
      # decoding is 100--200MB. When using async eval mode, decoding takes
      # 20--30 seconds of main thread time but is necessary to avoid OOM during
      # inter-process communication.
      decoded_preds = coco_metric.decode_predictions(self.predictions.values())
      self.predictions.clear()

      if self.params.collect_eval_results_async:
        def _eval_results_getter():
          """Iteratively get eval results from async eval process."""
          while True:
            step, eval_results = self.async_eval_results_queue.get()
            self.eval_coco_ap = eval_results['COCO/AP']
            mlperf.logger.log_eval_accuracy(
                self.eval_coco_ap, step, self.batch_size * self.params.num_gpus,
                ssd_constants.COCO_NUM_TRAIN_IMAGES)
            if self.reached_target():
              # Reached target, clear all pending messages in predictions queue
              # and insert poison pill to stop the async eval process.
              while not self.async_eval_predictions_queue.empty():
                self.async_eval_predictions_queue.get()
              self.async_eval_predictions_queue.put('STOP')
              break

        if not self.async_eval_process:
          # Limiting the number of messages in predictions queue to prevent OOM.
          # Each message (predictions data) can potentially consume a lot of
          # memory, and normally there should only be few messages in the queue.
          # If often blocked on this, consider reducing eval frequency.
          self.async_eval_predictions_queue = multiprocessing.Queue(2)
          self.async_eval_results_queue = multiprocessing.Queue()

          # Reason to use a Process as opposed to Thread is mainly the
          # computationally intensive eval runner. Python multithreading is not
          # truly running in parallel, a runner thread would get significantly
          # delayed (or alternatively delay the main thread).
          self.async_eval_process = multiprocessing.Process(
              target=coco_metric.async_eval_runner,
              args=(self.async_eval_predictions_queue,
                    self.async_eval_results_queue,
                    annotation_file))
          self.async_eval_process.daemon = True
          self.async_eval_process.start()

          self.async_eval_results_getter_thread = threading.Thread(
              target=_eval_results_getter, args=())
          self.async_eval_results_getter_thread.daemon = True
          self.async_eval_results_getter_thread.start()

        self.async_eval_predictions_queue.put(
            (self.eval_global_step, decoded_preds))
        return {'top_1_accuracy': 0, 'top_5_accuracy': 0.}

      eval_results = coco_metric.compute_map(decoded_preds, annotation_file)
      self.eval_coco_ap = eval_results['COCO/AP']
      ret = {'top_1_accuracy': self.eval_coco_ap, 'top_5_accuracy': 0.}
      for metric_key, metric_value in eval_results.items():
        ret[constants.SIMPLE_VALUE_RESULT_PREFIX + metric_key] = metric_value
      mlperf.logger.log_eval_accuracy(self.eval_coco_ap, self.eval_global_step,
                                      self.batch_size * self.params.num_gpus,
                                      ssd_constants.COCO_NUM_TRAIN_IMAGES)
      return ret
    log_fn('Got {:d} out of {:d} eval examples.'
           ' Waiting for the remaining to calculate mAP...'.format(
               len(self.predictions), ssd_constants.COCO_NUM_VAL_IMAGES))
    return {'top_1_accuracy': self.eval_coco_ap, 'top_5_accuracy': 0.}

  def get_synthetic_inputs(self, input_name, nclass):
    """Generating synthetic data matching real data shape and type."""
    inputs = tf.random_uniform(
        self.get_input_shapes('train')[0], dtype=self.data_type)
    inputs = contrib_framework.local_variable(inputs, name=input_name)
    boxes = tf.random_uniform(
        [self.batch_size, ssd_constants.NUM_SSD_BOXES, 4], dtype=tf.float32)
    classes = tf.random_uniform(
        [self.batch_size, ssd_constants.NUM_SSD_BOXES, 1], dtype=tf.float32)
    nboxes = tf.random_uniform(
        [self.batch_size], minval=1, maxval=10, dtype=tf.float32)
    return (inputs, boxes, classes, nboxes)

  def reached_target(self):
    return (self.params.stop_at_top_1_accuracy and
            self.eval_coco_ap >= self.params.stop_at_top_1_accuracy)
