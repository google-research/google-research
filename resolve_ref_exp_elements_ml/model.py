# coding=utf-8
# Copyright 2018 The Google Research Authors.
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

r"""Model to segment a region by refering expression.

Uses the DeepLab and Kona models. Bassed on Segmentation from Natural Language
Expressions (https://arxiv.org/pdf/1603.06180.pdf)
This code was modified from DeepLab at
https://github.com/tensorflow/models/tree/master/research/deeplab.
"""
from deeplab import feature_extractor
import elements_embeddings
import model_input
import tensorflow as tf
import tensorflow.contrib.slim as slim


_LOGITS_SCOPE_NAME = 'logits'
_MERGED_LOGITS_SCOPE = 'merged_logits'
_IMAGE_POOLING_SCOPE = 'image_pooling'
_ASPP_SCOPE = 'aspp'
_CONCAT_PROJECTION_SCOPE = 'concat_projection'
_DECODER_SCOPE = 'decoder'


def get_extra_layer_scopes():
  """Gets the scopes for extra layers.

  Returns:
    A list of scopes for extra layers.
  """
  return [
      _LOGITS_SCOPE_NAME,
      _IMAGE_POOLING_SCOPE,
      _ASPP_SCOPE,
      _CONCAT_PROJECTION_SCOPE,
      _DECODER_SCOPE,
  ]


def predict_labels_multi_scale(images,
                               feature_map,
                               flags,
                               outputs_to_num_classes,
                               eval_scales=None,
                               add_flipped_images=False,
                               merge_method='max',
                               atrous_rates=None,
                               add_image_level_feature=False,
                               aspp_with_batch_norm=False,
                               aspp_with_separable_conv=False,
                               multi_grid=None,
                               output_stride=8,
                               decoder_output_stride=None,
                               decoder_use_separable_conv=False,
                               crop_size=None,
                               logits_kernel_size=1,
                               depth_multiplier=1.0,
                               model_variant=None):
  """Predicts segmentation labels.

  Args:
    images: A tensor of size [batch, height, width, channels].
    feature_map: Features used by the model.
    flags: The input Flags.
    outputs_to_num_classes: A map from output type to the number of classes.
      For example, for the task of semantic segmentation with 21 semantic
      classes, we would have outputs_to_num_classes['semantic'] = 21.
    eval_scales: The scales to resize images for evaluation.
    add_flipped_images: Add flipped images for evaluation or not.
    merge_method: Method to merge multi scale features.
    atrous_rates: A list of atrous convolution rates for last layer.
    add_image_level_feature: Add image level feature.
    aspp_with_batch_norm: Use batch normalization layers for ASPP.
    aspp_with_separable_conv: Use separable convolution for ASPP.
    multi_grid: Employ a hierarchy of different atrous rates within network.
    output_stride: The ratio of input to output spatial resolution.
    decoder_output_stride: The ratio of input to output spatial resolution when
      employing decoder to refine segmentation results.
    decoder_use_separable_conv: Employ separable convolution for decoder or not.
    crop_size: A tuple [crop_height, crop_width].
    logits_kernel_size: The kernel size for computing logits.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops used in MobileNet.
    model_variant: Model variant for feature extraction.

  Returns:
    A dictionary with keys specifying the output_type (e.g., semantic
      prediction) and values storing Tensors representing predictions (argmax
      over channels). Each prediction has size [batch, height, width].
  """
  if eval_scales is None:
    eval_scales = [1.0]

  outputs_to_predictions = {output: [] for output in outputs_to_num_classes}

  for i, image_scale in enumerate(eval_scales):
    with tf.variable_scope(
        tf.get_variable_scope(), reuse=True if i > 0 else None):
      outputs_to_scales_to_logits = multi_scale_logits(
          images,
          feature_map,
          flags,
          outputs_to_num_classes,
          image_pyramid=[image_scale],
          merge_method=merge_method,
          atrous_rates=atrous_rates,
          add_image_level_feature=add_image_level_feature,
          aspp_with_batch_norm=aspp_with_batch_norm,
          aspp_with_separable_conv=aspp_with_separable_conv,
          multi_grid=multi_grid,
          output_stride=output_stride,
          decoder_output_stride=decoder_output_stride,
          decoder_use_separable_conv=decoder_use_separable_conv,
          logits_kernel_size=logits_kernel_size,
          crop_size=crop_size,
          depth_multiplier=depth_multiplier,
          model_variant=model_variant,
          is_training=False,
          fine_tune_batch_norm=False)

    if add_flipped_images:
      with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        outputs_to_scales_to_logits_reversed = multi_scale_logits(
            tf.reverse_v2(images, [2]),
            feature_map,
            flags,
            outputs_to_num_classes,
            image_pyramid=[image_scale],
            merge_method=merge_method,
            atrous_rates=atrous_rates,
            add_image_level_feature=add_image_level_feature,
            aspp_with_batch_norm=aspp_with_batch_norm,
            aspp_with_separable_conv=aspp_with_separable_conv,
            multi_grid=multi_grid,
            output_stride=output_stride,
            decoder_output_stride=decoder_output_stride,
            decoder_use_separable_conv=decoder_use_separable_conv,
            logits_kernel_size=logits_kernel_size,
            crop_size=crop_size,
            depth_multiplier=depth_multiplier,
            model_variant=model_variant,
            is_training=False,
            fine_tune_batch_norm=False)

    for output in sorted(outputs_to_scales_to_logits):
      scales_to_logits = outputs_to_scales_to_logits[output]
      logits = tf.image.resize_bilinear(
          scales_to_logits[_MERGED_LOGITS_SCOPE],
          tf.shape(images)[1:3],
          align_corners=True)
      outputs_to_predictions[output].append(
          tf.expand_dims(tf.nn.softmax(logits), 4))

      if add_flipped_images:
        scales_to_logits_reversed = (
            outputs_to_scales_to_logits_reversed[output])
        logits_reversed = tf.image.resize_bilinear(
            tf.reverse_v2(
                scales_to_logits_reversed[_MERGED_LOGITS_SCOPE], [2]),
            tf.shape(images)[1:3],
            align_corners=True)
        outputs_to_predictions[output].append(
            tf.expand_dims(tf.nn.softmax(logits_reversed), 4))

  outputs_to_probs = {}

  # For each pixel in the output, get the output categorization
  for output in sorted(outputs_to_predictions):
    predictions = outputs_to_predictions[output]
    # Compute average prediction across different scales and flipped images.
    predictions = tf.reduce_mean(tf.concat(predictions, 4), axis=4)
    outputs_to_predictions[output] = tf.argmax(predictions, 3)
    outputs_to_probs[output] = predictions

  return outputs_to_predictions, outputs_to_probs


def predict_labels(images,
                   feature_map,
                   flags,
                   outputs_to_num_classes,
                   image_pyramid=None,
                   merge_method='max',
                   atrous_rates=None,
                   add_image_level_feature=False,
                   aspp_with_batch_norm=False,
                   aspp_with_separable_conv=False,
                   multi_grid=None,
                   output_stride=8,
                   decoder_output_stride=None,
                   decoder_use_separable_conv=False,
                   crop_size=None,
                   logits_kernel_size=1,
                   depth_multiplier=1.0,
                   model_variant=None):
  """Predicts segmentation labels.

  Args:
    images: A tensor of size [batch, height, width, channels].
    feature_map: Features used by the model.
    flags: The input Flags.
    outputs_to_num_classes: A map from output type to the number of classes.
      For example, for the task of semantic segmentation with 21 semantic
      classes, we would have outputs_to_num_classes['semantic'] = 21.
    image_pyramid: Input image scales for multi-scale feature extraction.
    merge_method: Method to merge multi scale features.
    atrous_rates: A list of atrous convolution rates for last layer.
    add_image_level_feature: Add image level feature.
    aspp_with_batch_norm: Use batch normalization layers for ASPP.
    aspp_with_separable_conv: Use separable convolution for ASPP.
    multi_grid: Employ a hierarchy of different atrous rates within network.
    output_stride: The ratio of input to output spatial resolution.
    decoder_output_stride: The ratio of input to output spatial resolution when
      employing decoder to refine segmentation results.
    decoder_use_separable_conv: Employ separable convolution for decoder or not.
    crop_size: A tuple [crop_height, crop_width].
    logits_kernel_size: The kernel size for computing logits.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops used in MobileNet.
    model_variant: Model variant for feature extraction.

  Returns:
    A dictionary with keys specifying the output_type (e.g., semantic
      prediction) and values storing Tensors representing predictions (argmax
      over channels). Each prediction has size [batch, height, width].
  """
  outputs_to_scales_to_logits = multi_scale_logits(
      images,
      feature_map,
      flags,
      outputs_to_num_classes,
      image_pyramid=image_pyramid,
      merge_method=merge_method,
      atrous_rates=atrous_rates,
      add_image_level_feature=add_image_level_feature,
      aspp_with_batch_norm=aspp_with_batch_norm,
      aspp_with_separable_conv=aspp_with_separable_conv,
      multi_grid=multi_grid,
      output_stride=output_stride,
      decoder_output_stride=decoder_output_stride,
      decoder_use_separable_conv=decoder_use_separable_conv,
      logits_kernel_size=logits_kernel_size,
      crop_size=crop_size,
      depth_multiplier=depth_multiplier,
      model_variant=model_variant,
      is_training=False,
      fine_tune_batch_norm=False)

  predictions = {}
  outputs_to_probs = {}
  #   Resize the output of the vis/va model to match
  #   original input (after model architecture)
  for output in sorted(outputs_to_scales_to_logits):
    scales_to_logits = outputs_to_scales_to_logits[output]
    logits = scales_to_logits[_MERGED_LOGITS_SCOPE]
    if output == 'segment':
      logits = tf.image.resize_bilinear(
          logits, tf.shape(images)[1:3], align_corners=True)
      outputs_to_probs[output] = tf.nn.softmax(logits)
      predictions[output] = tf.argmax(logits, 3)
    elif output == 'regression':
      outputs_to_probs[output] = logits
      predictions[output] = logits

  return predictions, outputs_to_probs


def scale_dimension(dim, scale):
  """Scales the input dimension.

  Args:
    dim: Input dimension (a scalar).
    scale: The amount of scaling applied to the input.

  Returns:
    Scaled dimension.
  """
  if isinstance(dim, tf.Tensor):
    return tf.cast((tf.to_float(dim) - 1.0) * scale + 1.0, dtype=tf.int32)
  else:
    return int((float(dim) - 1.0) * scale + 1.0)


def multi_scale_logits(images,
                       feature_map,
                       flags,
                       outputs_to_num_classes,
                       image_pyramid=None,
                       merge_method='max',
                       atrous_rates=None,
                       add_image_level_feature=False,
                       aspp_with_batch_norm=False,
                       aspp_with_separable_conv=False,
                       multi_grid=None,
                       output_stride=8,
                       decoder_output_stride=None,
                       decoder_use_separable_conv=False,
                       logits_kernel_size=1,
                       crop_size=None,
                       depth_multiplier=1.0,
                       model_variant=None,
                       weight_decay=0.0001,
                       is_training=False,
                       fine_tune_batch_norm=False):
  """Gets the logits for multi-scale inputs.

  The returned logits are all downsampled (due to max-pooling layers)
  for both training and evaluation.

  Args:
    images: A tensor of size [batch, height, width, channels].
    feature_map: Features used by the model.
    flags: The input Flags.
    outputs_to_num_classes: A map from output type to the number of classes.
      For example, for the task of semantic segmentation with 21 semantic
      classes, we would have outputs_to_num_classes['semantic'] = 21.
    image_pyramid: Input image scales for multi-scale feature extraction.
    merge_method: Method to merge multi scale features.
    atrous_rates: A list of atrous convolution rates for ASPP.
    add_image_level_feature: Add image-level feature.
    aspp_with_batch_norm: Use batch normalization layers for ASPP.
    aspp_with_separable_conv: Use separable convolution for ASPP.
    multi_grid: Employ a hierarchy of different atrous rates within network.
    output_stride: The ratio of input to output spatial resolution.
    decoder_output_stride: The ratio of input to output spatial resolution when
      employing decoder to refine segmentation results.
    decoder_use_separable_conv: Employ separable convolution for decoder or not.
    logits_kernel_size: The kernel size for computing logits.
    crop_size: A tuple (crop_height, crop_width).
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops used in MobileNet.
    model_variant: Model variant for feature extraction.
    weight_decay: The weight decay for model variables.
    is_training: Is training or not.
    fine_tune_batch_norm: Fine-tune the batch norm parameters or not.

  Returns:
    outputs_to_scales_to_logits: A map of maps from output_type (e.g.,
      semantic prediction) to a dictionary of multi-scale logits names to
      logits. For each output_type, the dictionary has keys which
      correspond to the scales and values which correspond to the logits.
      For example, if `scales` equals [1.0, 1.5], then the keys would
      include 'merged_logits', 'logits_1.00' and 'logits_1.50'.

  Raises:
    ValueError: If crop_size = None and add_image_level_feature = True,
      since add_image_level_feature requires crop_size information.
  """
  # Setup default values.
  if not image_pyramid:
    image_pyramid = [1.0]
  if crop_size is None and add_image_level_feature:
    raise ValueError(
        'Crop size must be specified for using image-level feature.')
  crop_height = crop_size[0] if crop_size else tf.shape(images)[1]
  crop_width = crop_size[1] if crop_size else tf.shape(images)[2]

  # Compute the height, width for the output logits.
  if decoder_output_stride is None:
    logits_output_stride = output_stride
  else:
    logits_output_stride = decoder_output_stride
  logits_height = scale_dimension(
      crop_height,
      max(1.0, max(image_pyramid)) / logits_output_stride)
  logits_width = scale_dimension(
      crop_width,
      max(1.0, max(image_pyramid)) / logits_output_stride)

  # Compute the logits for each scale in the image pyramid.
  outputs_to_scales_to_logits = {k: {} for k in outputs_to_num_classes}

  for count, image_scale in enumerate(image_pyramid):
    if image_scale != 1.0:
      scaled_height = scale_dimension(crop_height, image_scale)
      scaled_width = scale_dimension(crop_width, image_scale)
      scaled_crop_size = [scaled_height, scaled_width]
      scaled_images = tf.image.resize_bilinear(
          images, scaled_crop_size, align_corners=True)
      if crop_size:
        scaled_images.set_shape([None, scaled_height, scaled_width, 3])
    else:
      scaled_crop_size = crop_size
      scaled_images = images

    outputs_to_logits = _get_logits(
        scaled_images,
        feature_map,
        flags,
        outputs_to_num_classes,
        atrous_rates=atrous_rates,
        add_image_level_feature=add_image_level_feature,
        aspp_with_batch_norm=aspp_with_batch_norm,
        aspp_with_separable_conv=aspp_with_separable_conv,
        multi_grid=multi_grid,
        output_stride=output_stride,
        decoder_output_stride=decoder_output_stride,
        decoder_use_separable_conv=decoder_use_separable_conv,
        logits_kernel_size=logits_kernel_size,
        crop_size=scaled_crop_size,
        depth_multiplier=depth_multiplier,
        model_variant=model_variant,
        weight_decay=weight_decay,
        reuse=True if count > 0 else None,
        is_training=is_training,
        fine_tune_batch_norm=fine_tune_batch_norm)
    if 'segment' in outputs_to_logits:
      # Resize the logits to have the same dimension before merging.
      outputs_to_logits['segment'] = tf.image.resize_bilinear(
          outputs_to_logits['segment'], [logits_height, logits_width],
          align_corners=True)
    # Return when only one input scale.
    if len(image_pyramid) == 1:
      for output in sorted(outputs_to_num_classes):
        outputs_to_scales_to_logits[output][
            _MERGED_LOGITS_SCOPE] = outputs_to_logits[output]
      return outputs_to_scales_to_logits

    # Save logits to the output map.
    for output in sorted(outputs_to_num_classes):
      outputs_to_scales_to_logits[output][
          'logits_%.2f' % image_scale] = outputs_to_logits[output]

  # Merge the logits from all the multi-scale inputs.
  for output in sorted(outputs_to_num_classes):
    if output in ['segment', 'regression']:
      merge_axis = 4
    else:
      merge_axis = 2
    # Concatenate the multi-scale logits for each output type.
    all_logits = [
        tf.expand_dims(logits, axis=merge_axis)
        for logits in outputs_to_scales_to_logits[output].values()
    ]
    all_logits = tf.concat(all_logits, merge_axis)
    merge_fn = tf.reduce_max if merge_method == 'max' else tf.reduce_mean
    outputs_to_scales_to_logits[output][_MERGED_LOGITS_SCOPE] = merge_fn(
        all_logits, axis=merge_axis)

  return outputs_to_scales_to_logits


def foreground_iou(labels, predictions, flags):
  """Calculates the iou of class 1 ignoring class 0."""
  weights = tf.to_float(
      tf.not_equal(labels,
                   model_input.dataset_descriptors[flags.dataset].ignore_label))

  predictions = tf.to_float(predictions)
  labels = tf.to_float(labels)
  labels *= weights
  predictions *= weights

  # Use this instead of built in mean_iou,
  # because we want to ignnore the background label.
  # This gives the IOU the same meaning as it does in our baseline
  # https://arxiv.org/pdf/1603.06180.pdf
  intersect = labels * predictions
  union = labels + predictions - intersect

  intersect_sum = tf.reduce_sum(intersect, [1, 2, 3])
  union_sum = tf.reduce_sum(union, [1, 2, 3])
  # Make sure we don't divide by 0.
  iou = tf.where(
      tf.less(union_sum, 1e-20), tf.ones_like(union_sum),
      intersect_sum / union_sum)

  return iou, weights


def extract_features(images,
                     feature_map,
                     flags,
                     atrous_rates=None,
                     add_image_level_feature=False,
                     aspp_with_batch_norm=False,
                     aspp_with_separable_conv=False,
                     output_stride=8,
                     multi_grid=None,
                     crop_size=None,
                     depth_multiplier=1.0,
                     model_variant=None,
                     weight_decay=0.0001,
                     reuse=None,
                     is_training=False,
                     fine_tune_batch_norm=False):
  """Extracts features by the particular model_variant.

  Args:
    images: A tensor of size [batch, height, width, channels].
    feature_map: Features used by the model.
    flags: The input Flags.
    atrous_rates: A list of atrous convolution rates for ASPP.
    add_image_level_feature: Add image-level feature.
    aspp_with_batch_norm: Use batch normalization layers for ASPP.
    aspp_with_separable_conv: Use separable convolution for ASPP.
    output_stride: The ratio of input to output spatial resolution.
    multi_grid: Employ a hierarchy of different atrous rates within network.
    crop_size: A tuple [crop_height, crop_width].
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops used in MobileNet.
    model_variant: Model variant for feature extraction.
    weight_decay: The weight decay for model variables.
    reuse: Reuse the model variables or not.
    is_training: Is training or not.
    fine_tune_batch_norm: Fine-tune the batch norm parameters or not.

  Returns:
    concat_logits: A tensor of size [batch, feature_height, feature_width,
      feature_channels], where feature_height/feature_width are determined by
      the images height/width and output_stride.
    end_points: A dictionary from components of the network to the corresponding
      activation.
  """
  #  end_points refers to the layers within the backbone CNN
  features, end_points = feature_extractor.extract_features(
      images,
      output_stride=output_stride,
      multi_grid=multi_grid,
      depth_multiplier=depth_multiplier,
      model_variant=model_variant,
      weight_decay=weight_decay,
      reuse=reuse,
      is_training=is_training,
      fine_tune_batch_norm=fine_tune_batch_norm)

  if flags.image_keep_prob < 1.0:
    # Trailing / so tensorflow reuses the scope instead of creating a new one.
    with tf.variable_scope('MobilenetV2/'):
      for dropout_layer_name in [
          'layer_3', 'layer_6', 'layer_9', 'layer_12', 'layer_15'
      ]:
        add_dropout_layer = end_points[dropout_layer_name]

        add_dropout_layer_out = tf.contrib.graph_editor.get_consuming_ops(
            [add_dropout_layer])[0]
        updated_add_dropout_layer = tf.nn.dropout(
            add_dropout_layer, keep_prob=flags.image_keep_prob)
        tf.contrib.graph_editor.connect(
            updated_add_dropout_layer,
            add_dropout_layer_out,
            disconnect_first=True)

  #  Determines to which layer of backbone CNN to inject
  #  the elements (OCR) model
  if flags.add_ref_elements_layer:
    add_ref_elements_layer = end_points[flags.add_ref_elements_layer]
  else:
    #  Add at top if not specified
    add_ref_elements_layer = features

  add_ref_elements_layer = tf.nn.dropout(
      add_ref_elements_layer, keep_prob=flags.image_keep_prob)

  add_ref_elements_layer_shape = add_ref_elements_layer.get_shape().as_list()[
      1:]
  elements_enc_size = add_ref_elements_layer_shape
  if flags.elements_enc_size > 0:
    elements_enc_size[2] = flags.elements_enc_size

  flags.elements_3d_output = True
  #  Elements Model + Referring Expression
  enc_elements, enc_ref_exp, _ = (
      elements_embeddings.ref_elements_model(feature_map, elements_enc_size,
                                             flags))

  if flags.elements_img_comb_cnn:
    if flags.use_image:
      enc_ref_exp_elements = tf.concat(
          [enc_elements, enc_ref_exp, add_ref_elements_layer], 3)
      tf.summary.histogram('enc_ref_exp_elements1', enc_ref_exp_elements)

    else:
      enc_ref_exp_elements = tf.concat([enc_elements, enc_ref_exp], 3)

    # Convolution layer after concat of image, ref exp, and elements information
    for _ in range(flags.elements_img_comb_cnn_layers - 1):
      enc_ref_exp_elements = tf.layers.conv2d(
          enc_ref_exp_elements,
          enc_ref_exp_elements.shape[3] / 2,
          3,
          padding='SAME',
          activation=tf.nn.relu,
          strides=1)
    enc_ref_exp_elements = tf.nn.dropout(
        enc_ref_exp_elements, keep_prob=flags.comb_dropout_keep_prob)
    enc_ref_exp_elements = tf.layers.conv2d(
        enc_ref_exp_elements,
        add_ref_elements_layer.shape[3],
        3,
        padding='SAME',
        activation=None,
        strides=1)
  else:
    enc_ref_exp_elements = enc_elements + enc_ref_exp

  tf.summary.histogram('enc_ref_exp_elements', enc_ref_exp_elements)

  no_img_updated_add_ref_elements_layer = enc_ref_exp_elements

  #   Summation between image and rest of model
  if flags.use_image:
    updated_add_ref_elements_layer = (
        add_ref_elements_layer + enc_ref_exp_elements)
  else:
    updated_add_ref_elements_layer = no_img_updated_add_ref_elements_layer

  tf.summary.histogram('updated_add_ref_elements_layer',
                       updated_add_ref_elements_layer)
  tf.summary.histogram('features_after', features)

  if flags.add_ref_elements_layer:
    features, end_points2 = feature_extractor.extract_features(
        images,
        output_stride=output_stride,
        multi_grid=multi_grid,
        depth_multiplier=depth_multiplier,
        model_variant=model_variant,
        weight_decay=weight_decay,
        reuse=tf.AUTO_REUSE,
        is_training=is_training,
        fine_tune_batch_norm=fine_tune_batch_norm)

    add_ref_elements_layer = end_points2[flags.add_ref_elements_layer]

    # This connects the output of the addition to the appropriate
    # layer of the backbone network.
    # It's done this way to avoid modifying the code of the backbone network.
    # tf.contrib.graph_editor.swap_outputs(
    #     tf.contrib.graph_editor.sgv(updated_add_ref_elements_layer.op),
    #     tf.contrib.graph_editor.sgv(add_ref_elements_layer.op))
    add_ref_elements_layer_out = tf.contrib.graph_editor.get_consuming_ops(
        [add_ref_elements_layer])[0]
    updated_add_ref_elements_layer = tf.identity(
        updated_add_ref_elements_layer, name='updated_add_ref_elements_layer')
    tf.contrib.graph_editor.connect(
        updated_add_ref_elements_layer,
        add_ref_elements_layer_out,
        disconnect_first=True)
  else:
    features = updated_add_ref_elements_layer

  if not aspp_with_batch_norm:
    return features, end_points
  else:
    batch_norm_params = {
        'is_training': is_training and fine_tune_batch_norm,
        'decay': 0.9997,
        'epsilon': 1e-5,
        'scale': True,
    }

    with slim.arg_scope(
        [slim.conv2d, slim.separable_conv2d],
        weights_regularizer=slim.l2_regularizer(weight_decay),
        activation_fn=tf.nn.relu,
        normalizer_fn=slim.batch_norm,
        padding='SAME',
        stride=1,
        reuse=reuse):
      with slim.arg_scope([slim.batch_norm], **batch_norm_params):
        depth = 256
        branch_logits = []

        if add_image_level_feature:
          pool_height = scale_dimension(crop_size[0], 1. / output_stride)
          pool_width = scale_dimension(crop_size[1], 1. / output_stride)
          image_feature = slim.avg_pool2d(features,
                                          [pool_height, pool_width],
                                          [pool_height, pool_width],
                                          padding='VALID')
          image_feature = slim.conv2d(image_feature,
                                      depth,
                                      1,
                                      scope=_IMAGE_POOLING_SCOPE)
          image_feature = tf.image.resize_bilinear(
              image_feature, [pool_height, pool_width], align_corners=True)
          image_feature.set_shape([None, pool_height, pool_width, depth])
          branch_logits.append(image_feature)

        # Employ a 1x1 convolution.
        branch_logits.append(slim.conv2d(features, depth, 1,
                                         scope=_ASPP_SCOPE + str(0)))

        if atrous_rates:
          # Employ 3x3 convolutions with different atrous rates.
          for i, rate in enumerate(atrous_rates, 1):
            scope = _ASPP_SCOPE + str(i)
            if aspp_with_separable_conv:
              aspp_features = _split_separable_conv2d(
                  features,
                  filters=depth,
                  rate=rate,
                  weight_decay=weight_decay,
                  scope=scope)
            else:
              aspp_features = slim.conv2d(
                  features, depth, 3, rate=rate, scope=scope)
            branch_logits.append(aspp_features)

        # Merge branch logits.
        concat_logits = tf.concat(branch_logits, 3)
        concat_logits = slim.conv2d(
            concat_logits, depth, 1, scope=_CONCAT_PROJECTION_SCOPE)
        concat_logits = tf.nn.dropout(
            concat_logits, keep_prob=flags.comb_dropout_keep_prob)

        return concat_logits, end_points


def _get_logits(images,
                feature_map,
                flags,
                outputs_to_num_classes,
                atrous_rates=None,
                add_image_level_feature=False,
                aspp_with_batch_norm=False,
                aspp_with_separable_conv=False,
                multi_grid=None,
                output_stride=8,
                decoder_output_stride=None,
                decoder_use_separable_conv=False,
                logits_kernel_size=1,
                crop_size=None,
                depth_multiplier=1.0,
                model_variant=None,
                weight_decay=0.0001,
                reuse=None,
                is_training=False,
                fine_tune_batch_norm=False):
  """Gets the logits by atrous/image spatial pyramid pooling.

  Args:
    images: A tensor of size [batch, height, width, channels].
    feature_map: Features used by the model.
    flags: The input Flags.
    outputs_to_num_classes: A map from output type to the number of classes.
      For example, for the task of semantic segmentation with 21 semantic
      classes, we would have outputs_to_num_classes['semantic'] = 21.
    atrous_rates: A list of atrous convolution rates for ASPP.
    add_image_level_feature: Add image-level feature.
    aspp_with_batch_norm: Use batch normalization layers for ASPP.
    aspp_with_separable_conv: Use separable convolution for ASPP.
    multi_grid: Employ a hierarchy of different atrous rates within network.
    output_stride: The ratio of input to output spatial resolution.
    decoder_output_stride: The ratio of input to output spatial resolution when
      employing decoder to refine segmentation results.
    decoder_use_separable_conv: Employ separable convolution for decoder or not.
    logits_kernel_size: The kernel size for computing logits.
    crop_size: A tuple [crop_height, crop_width].
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops used in MobileNet.
    model_variant: Model variant for feature extraction.
    weight_decay: The weight decay for model variables.
    reuse: Reuse the model variables or not.
    is_training: Is training or not.
    fine_tune_batch_norm: Fine-tune the batch norm parameters or not.

  Returns:
    outputs_to_logits: A map from output_type to logits.
  """
  features, end_points = extract_features(
      images,
      feature_map,
      flags,
      atrous_rates=atrous_rates,
      add_image_level_feature=add_image_level_feature,
      aspp_with_batch_norm=aspp_with_batch_norm,
      aspp_with_separable_conv=aspp_with_separable_conv,
      output_stride=output_stride,
      multi_grid=multi_grid,
      crop_size=crop_size,
      depth_multiplier=depth_multiplier,
      model_variant=model_variant,
      weight_decay=weight_decay,
      reuse=reuse,
      is_training=is_training,
      fine_tune_batch_norm=fine_tune_batch_norm)

  if flags.output_mode in ['segment', 'regression', 'combined'
                          ] and (decoder_output_stride is not None):
    decoder_height = scale_dimension(crop_size[0], 1.0 / decoder_output_stride)
    decoder_width = scale_dimension(crop_size[1], 1.0 / decoder_output_stride)
    features = refine_by_decoder(
        features,
        end_points,
        decoder_height=decoder_height,
        decoder_width=decoder_width,
        decoder_use_separable_conv=decoder_use_separable_conv,
        model_variant=model_variant,
        weight_decay=weight_decay,
        reuse=reuse,
        is_training=is_training,
        fine_tune_batch_norm=fine_tune_batch_norm)

  outputs_to_logits = {}
  for output in sorted(outputs_to_num_classes):
    if output == 'segment':
      outputs_to_logits[output] = _get_branch_logits(
          features,
          outputs_to_num_classes[output],
          atrous_rates,
          aspp_with_batch_norm=aspp_with_batch_norm,
          kernel_size=logits_kernel_size,
          weight_decay=weight_decay,
          reuse=reuse,
          scope_suffix=output)
    elif output == 'regression':
      outputs_to_logits[output] = _get_coordinate_logits(features, flags)

  return outputs_to_logits


def refine_by_decoder(features,
                      end_points,
                      decoder_height,
                      decoder_width,
                      decoder_use_separable_conv=False,
                      model_variant=None,
                      weight_decay=0.0001,
                      reuse=None,
                      is_training=False,
                      fine_tune_batch_norm=False):
  """Adds the decoder to obtain sharper segmentation results.

  Args:
    features: A tensor of size [batch, features_height, features_width,
      features_channels].
    end_points: A dictionary from components of the network to the corresponding
      activation.
    decoder_height: The height of decoder feature maps.
    decoder_width: The width of decoder feature maps.
    decoder_use_separable_conv: Employ separable convolution for decoder or not.
    model_variant: Model variant for feature extraction.
    weight_decay: The weight decay for model variables.
    reuse: Reuse the model variables or not.
    is_training: Is training or not.
    fine_tune_batch_norm: Fine-tune the batch norm parameters or not.

  Returns:
    Decoder output with size [batch, decoder_height, decoder_width,
      decoder_channels].
  """
  batch_norm_params = {
      'is_training': is_training and fine_tune_batch_norm,
      'decay': 0.9997,
      'epsilon': 1e-5,
      'scale': True,
  }

  with slim.arg_scope(
      [slim.conv2d, slim.separable_conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      padding='SAME',
      stride=1,
      reuse=reuse):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with tf.variable_scope(_DECODER_SCOPE, _DECODER_SCOPE, [features]):
        feature_list = feature_extractor.networks_to_feature_maps[
            model_variant][feature_extractor.DECODER_END_POINTS]
        if feature_list is None:
          tf.logging.info('Not found any decoder end points.')
          return features
        else:
          decoder_features = features
          for i, name in enumerate(feature_list):
            decoder_features_list = [decoder_features]

            # MobileNet variants use different naming convention.
            if 'mobilenet' in model_variant:
              feature_name = name
            else:
              feature_name = '{}/{}'.format(
                  feature_extractor.name_scope[model_variant], name)
            decoder_features_list.append(
                slim.conv2d(end_points[feature_name],
                            48,
                            1,
                            scope='feature_projection'+str(i)))
            # Resize to decoder_height/decoder_width.
            for j, feature in enumerate(decoder_features_list):
              decoder_features_list[j] = tf.image.resize_bilinear(
                  feature,
                  [decoder_height, decoder_width],
                  align_corners=True)
              decoder_features_list[j].set_shape(
                  [None, decoder_height, decoder_width, None])
            decoder_depth = 256
            if decoder_use_separable_conv:
              decoder_features = _split_separable_conv2d(
                  tf.concat(decoder_features_list, 3),
                  filters=decoder_depth,
                  rate=1,
                  weight_decay=weight_decay,
                  scope='decoder_conv0')
              decoder_features = _split_separable_conv2d(
                  decoder_features,
                  filters=decoder_depth,
                  rate=1,
                  weight_decay=weight_decay,
                  scope='decoder_conv1')
            else:
              num_convs = 2
              decoder_features = slim.repeat(
                  tf.concat(decoder_features_list, 3),
                  num_convs,
                  slim.conv2d,
                  decoder_depth,
                  3,
                  scope='decoder_conv'+str(i))
          return decoder_features


def cr_conv(inp, feature_size, kernel_size, activation, flags, padding='valid'):
  """Create a 2d convolution layer for ClickRegression.

  Args:
    inp: A float tensor of shape [batch, height, width, channels]
    feature_size: Output feature size
    kernel_size: Size of kernel as specified by tf.layers.conv2d
    activation: Activation function as specified by tf.layers.conv2d
    flags: The input flags
    padding: Convolution padding type as specified by tf.layers.conv2d

  Returns:
    Convolution output of shape [batch, height, width, channels]
  """
  conv = tf.layers.conv2d(
      inp, feature_size, kernel_size, activation=activation, padding=padding)
  if flags.regression_batch_norm:
    conv = tf.layers.batch_normalization(
        conv,
        momentum=0.9997,
        epsilon=1e-5,
        training=flags.train_mode,
        renorm=True)
  return conv


def _get_coordinate_logits(features, flags):
  """Get the regression output of the model.

  Args:
    features: A float tensor of shape [batch, height, width, channels].
    flags: The input flags

  Returns:
    logits with shape [batch, x_coord, y_coord].
  """
  conv1_1 = cr_conv(
      features, 512, 3, activation=tf.nn.relu, padding='same', flags=flags)
  conv1_2 = cr_conv(conv1_1, 256, 1, activation=tf.nn.relu, flags=flags)
  conv1_3 = cr_conv(
      conv1_2, 512, 3, activation=tf.nn.relu, padding='same', flags=flags)
  max_pool1 = tf.layers.max_pooling2d(conv1_3, 2, 2)

  conv2_1 = cr_conv(
      max_pool1, 512, 3, activation=tf.nn.relu, padding='same', flags=flags)
  conv2_2 = cr_conv(conv2_1, 256, 1, activation=tf.nn.relu, flags=flags)
  conv2_3 = cr_conv(
      conv2_2, 512, 3, activation=tf.nn.relu, padding='same', flags=flags)
  conv2_4 = cr_conv(conv2_3, 256, 1, activation=tf.nn.relu, flags=flags)
  conv2_5 = cr_conv(
      conv2_4, 512, 3, activation=tf.nn.relu, padding='same', flags=flags)
  max_pool2 = tf.layers.max_pooling2d(conv2_5, 2, 2)

  conv3_1 = cr_conv(
      max_pool2, 1024, 3, activation=tf.nn.relu, padding='same', flags=flags)
  conv3_2 = cr_conv(conv3_1, 512, 1, activation=tf.nn.relu, flags=flags)
  conv3_3 = cr_conv(
      conv3_2, 1024, 3, activation=tf.nn.relu, padding='same', flags=flags)
  conv3_4 = cr_conv(conv3_3, 512, 1, activation=tf.nn.relu, flags=flags)
  conv3_5 = cr_conv(
      conv3_4, 1024, 3, activation=tf.nn.relu, padding='same', flags=flags)

  # Global average pool to create single 1024 sized tensor
  average_pool = tf.layers.average_pooling2d(
      conv3_5, [conv3_5.shape[1], conv3_5.shape[2]], 1)

  flatten = tf.layers.flatten(average_pool)
  if flags.coord_softmax:
    dense = tf.layers.dense(flatten, 750, activation=tf.nn.relu)

    x_proj = tf.layers.dense(dense, flags.image_size, activation=tf.nn.relu)
    y_proj = tf.layers.dense(dense, flags.image_size, activation=tf.nn.relu)

    x_softmax = tf.nn.softmax(x_proj)
    y_softmax = tf.nn.softmax(y_proj)

    coords = tf.to_float(tf.range(flags.image_size))
    x_pred = tf.reduce_sum(x_softmax * coords, axis=-1)
    y_pred = tf.reduce_sum(y_softmax * coords, axis=-1)

    predictions = tf.stack([x_pred, y_pred], axis=-1)
  else:
    hidden = tf.layers.dense(flatten, 513, activation=tf.nn.relu)
    predictions = tf.layers.dense(hidden, 2)

  return predictions


def _get_branch_logits(features,
                       num_classes,
                       atrous_rates=None,
                       aspp_with_batch_norm=False,
                       kernel_size=1,
                       weight_decay=0.0001,
                       reuse=None,
                       scope_suffix=''):
  """Gets the logits from each model's branch.

  The underlying model is branched out in the last layer when atrous
  spatial pyramid pooling is employed, and all branches are sum-merged
  to form the final logits.

  Args:
    features: A float tensor of shape [batch, height, width, channels].
    num_classes: Number of classes to predict.
    atrous_rates: A list of atrous convolution rates for last layer.
    aspp_with_batch_norm: Use batch normalization layers for ASPP.
    kernel_size: Kernel size for convolution.
    weight_decay: Weight decay for the model variables.
    reuse: Reuse model variables or not.
    scope_suffix: Scope suffix for the model variables.

  Returns:
    Merged logits with shape [batch, height, width, num_classes].
  """
  # When using batch normalization with ASPP, ASPP has been applied before
  # in extract_features, and thus we simply apply 1x1 convolution here.
  if aspp_with_batch_norm or atrous_rates is None:
    atrous_rates = [1]
    assert kernel_size == 1

  with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
      reuse=reuse):
    with tf.variable_scope(
        _LOGITS_SCOPE_NAME, _LOGITS_SCOPE_NAME, [features]):
      branch_logits = []
      for i, rate in enumerate(atrous_rates):
        scope = scope_suffix
        if i != 0:
          scope += '_' + str(i)

        branch_logits.append(slim.conv2d(
            features, num_classes, kernel_size=kernel_size,
            rate=rate, activation_fn=None, normalizer_fn=None,
            scope=scope))

      return tf.add_n(branch_logits)


def _split_separable_conv2d(inputs,
                            filters,
                            rate=1,
                            weight_decay=0.00004,
                            depthwise_weights_initializer_stddev=0.33,
                            pointwise_weights_initializer_stddev=0.06,
                            scope=None):
  """Splits separable conv2d into depthwise and pointwise conv2d.

  Args:
    inputs: Input tensor with shape [batch, height, width, channels].
    filters: Number of filters in the 1x1 pointwise convolution.
    rate: Atrous convolution rate for the depthwise convolution.
    weight_decay: The weight decay to use for regularizing the model.
    depthwise_weights_initializer_stddev: The standard deviation of the
      truncated normal weight initializer for depthwise convolution.
    pointwise_weights_initializer_stddev: The standard deviation of the
      truncated normal weight initializer for pointwise convolution.
    scope: Optional scope for the operation.

  Returns:
    Computed features after split separable conv2d.
  """
  outputs = slim.separable_conv2d(
      inputs,
      None,
      3,
      depth_multiplier=1,
      rate=rate,
      weights_initializer=tf.truncated_normal_initializer(
          stddev=depthwise_weights_initializer_stddev),
      weights_regularizer=None,
      scope=scope + '_depthwise')
  return slim.conv2d(
      outputs,
      filters,
      1,
      weights_initializer=tf.truncated_normal_initializer(
          stddev=pointwise_weights_initializer_stddev),
      weights_regularizer=slim.l2_regularizer(weight_decay),
      scope=scope + '_pointwise')


def get_ignore_mask(labels, flags):
  """Get the mask of elements to ignore.

  Args:
    labels: Ground truth label of shape [batch, height, width].
    flags: The input flags

  Returns:
    3D mask of same dimensionality as labels. 0 for elements to ignore, 1 for
    elements to keep.
  """
  mask = tf.to_int32(
      tf.not_equal(labels,
                   model_input.dataset_descriptors[flags.dataset].ignore_label))
  return mask


def get_output_to_num_classes(flags):
  """Get the output_to_num_classes dictionary to define the learning problem.

  Args:
    flags: The input flags

  Returns:
    This dictionary defines which part of the model to run given the output
    mode.

  """
  # TODO(ahah): Add support for elements model
  output_to_num_classes = {}
  if flags.output_mode in ['segment', 'combined']:
    output_to_num_classes['segment'] = model_input.dataset_descriptors[
        flags.dataset].num_classes
  if flags.output_mode in ['regression', 'combined']:
    output_to_num_classes['regression'] = model_input.dataset_descriptors[
        flags.dataset].num_classes

  return output_to_num_classes
