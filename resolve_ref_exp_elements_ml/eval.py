# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Evaluation script for the VIS model.
"""
import math
import common  # pylint: disable=unused-import
import model
import model_input
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.platform import app

flags = tf.app.flags
FLAGS = flags.FLAGS

# Settings for log directories.

flags.DEFINE_string('eval_logdir', None, 'Where to write the event logs.')

flags.DEFINE_string('checkpoint_dir', None, 'Directory of model checkpoints.')

# Settings for evaluating the model.

flags.DEFINE_integer('batch_size', 8,
                     'The number of images in each batch during evaluation.')

flags.DEFINE_integer('eval_interval_secs', 300,
                     'How often (in seconds) to run evaluation.')

flags.DEFINE_multi_float('eval_scales', [1.0],
                         'The scales to resize images for evaluation.')

flags.DEFINE_bool('add_flipped_images', False,
                  'Add flipped images for evaluation or not.')

flags.DEFINE_string('split', 'val',
                    'Which split of the dataset used for evaluation')

flags.DEFINE_integer('num_samples', 2048, '')


def calc_high_prob_overlaps(labels, probs, weights):
  """Calculates if the highest probability prediction is correct."""
  labels = tf.to_float(labels)
  probs *= weights
  labels *= weights
  probs = probs[:, :, :, 1]
  labels = labels[:, :, :, 0]

  overlap_probs = probs * labels
  non_overlap_probs = probs - overlap_probs

  return tf.greater(
      tf.reduce_max(overlap_probs, [1, 2]),
      tf.reduce_max(non_overlap_probs, [1, 2]))


def calc_accuracy_in_box(labels, coords, weights):
  """Calculate if coordinate lies within the ground truth.

  Args:
    labels: Ground truth label of shape [batch, height, width].
    coords: Regression output of shape [batch, 1, 1]
    weights: Padding mask for label

  Returns:
    Merged logits with shape [batch, height, width, num_classes].
  """
  labels *= weights
  labels = labels[:, :, :, 0]

  coords = tf.cast(coords, tf.int32)
  coords = tf.clip_by_value(coords, 0, FLAGS.image_size - 1)

  # Create the indices for the batch dimension
  batch_ind = tf.expand_dims(tf.range(tf.shape(coords)[0]), -1)

  # Concatenate the indices of the batch dimension with coords
  # to create a matrix of prediction indices
  coords = tf.cast(tf.concat([batch_ind, coords], -1), tf.int32)

  # Grab the elements prediced by index
  predicted_locations = tf.gather_nd(labels, coords)

  # Check if the elements selected are part of the ground truth bounding box
  accurate = tf.to_float(tf.greater(predicted_locations, 0))
  return accurate


def main(unused_argv):
  FLAGS.comb_dropout_keep_prob = 1.0
  FLAGS.image_keep_prob = 1.0
  FLAGS.elements_keep_prob = 1.0

  # Get dataset-dependent information.

  tf.gfile.MakeDirs(FLAGS.eval_logdir)
  tf.logging.info('Evaluating on %s set', FLAGS.split)

  with tf.Graph().as_default():
    samples = model_input.get_input_fn(FLAGS)()

    # Get model segmentation predictions.
    num_classes = model_input.dataset_descriptors[FLAGS.dataset].num_classes
    output_to_num_classes = model.get_output_to_num_classes(FLAGS)

    if tuple(FLAGS.eval_scales) == (1.0,):
      tf.logging.info('Performing single-scale test.')
      predictions, probs = model.predict_labels(
          samples['image'],
          samples,
          FLAGS,
          outputs_to_num_classes=output_to_num_classes,
          image_pyramid=FLAGS.image_pyramid,
          merge_method=FLAGS.merge_method,
          atrous_rates=FLAGS.atrous_rates,
          add_image_level_feature=FLAGS.add_image_level_feature,
          aspp_with_batch_norm=FLAGS.aspp_with_batch_norm,
          aspp_with_separable_conv=FLAGS.aspp_with_separable_conv,
          multi_grid=FLAGS.multi_grid,
          depth_multiplier=FLAGS.depth_multiplier,
          output_stride=FLAGS.output_stride,
          decoder_output_stride=FLAGS.decoder_output_stride,
          decoder_use_separable_conv=FLAGS.decoder_use_separable_conv,
          crop_size=[FLAGS.image_size, FLAGS.image_size],
          logits_kernel_size=FLAGS.logits_kernel_size,
          model_variant=FLAGS.model_variant)
    else:
      tf.logging.info('Performing multi-scale test.')
      predictions, probs = model.predict_labels_multi_scale(
          samples['image'],
          samples,
          FLAGS,
          outputs_to_num_classes=output_to_num_classes,
          eval_scales=FLAGS.eval_scales,
          add_flipped_images=FLAGS.add_flipped_images,
          merge_method=FLAGS.merge_method,
          atrous_rates=FLAGS.atrous_rates,
          add_image_level_feature=FLAGS.add_image_level_feature,
          aspp_with_batch_norm=FLAGS.aspp_with_batch_norm,
          aspp_with_separable_conv=FLAGS.aspp_with_separable_conv,
          multi_grid=FLAGS.multi_grid,
          depth_multiplier=FLAGS.depth_multiplier,
          output_stride=FLAGS.output_stride,
          decoder_output_stride=FLAGS.decoder_output_stride,
          decoder_use_separable_conv=FLAGS.decoder_use_separable_conv,
          crop_size=[FLAGS.image_size, FLAGS.image_size],
          logits_kernel_size=FLAGS.logits_kernel_size,
          model_variant=FLAGS.model_variant)

    metric_map = {}
    for output in output_to_num_classes:
      output_predictions = predictions[output]
      output_probs = probs[output]
      if output == 'segment':
        output_predictions = tf.expand_dims(output_predictions, 3)
        if num_classes == 2:
          labels = samples['label']

          iou, weights = model.foreground_iou(labels, output_predictions, FLAGS)
          soft_iou, _ = model.foreground_iou(labels, output_probs[:, :, :, 1:2],
                                             FLAGS)

          metric_map['mIOU'] = tf.metrics.mean(iou)
          metric_map['soft_mIOU'] = tf.metrics.mean(soft_iou)

          high_prob_overlaps = calc_high_prob_overlaps(labels, output_probs,
                                                       weights)
          metric_map['highestOverlaps'] = tf.metrics.mean(high_prob_overlaps)

          output_probs *= weights

        else:
          output_predictions = tf.reshape(output_predictions, shape=[-1])
          labels = tf.reshape(samples['label'], shape=[-1])
          weights = tf.to_float(
              tf.not_equal(
                  labels,
                  model_input.dataset_descriptors[FLAGS.dataset].ignore_label))

          # Set ignore_label regions to label 0, because metrics.mean_iou
          # requires range of labels=[0, dataset.num_classes).
          # Note the ignore_label regions are not evaluated since
          # the corresponding regions contain weights=0.
          labels = tf.where(
              tf.equal(
                  labels,
                  model_input.dataset_descriptors[FLAGS.dataset].ignore_label),
              tf.zeros_like(labels), labels)

          predictions_tag = 'mIOU'
          for eval_scale in FLAGS.eval_scales:
            predictions_tag += '_' + str(eval_scale)
          if FLAGS.add_flipped_images:
            predictions_tag += '_flipped'

          # Define the evaluation metric.
          metric_map[predictions_tag] = slim.metrics.mean_iou(
              output_predictions, labels, num_classes, weights=weights)

        def label_summary(labels, weights, name):
          tf.summary.image(
              name,
              tf.reshape(
                  tf.cast(
                      tf.to_float(labels * 255) / tf.to_float(num_classes),
                      tf.uint8) * tf.cast(weights, tf.uint8),
                  [-1, FLAGS.image_size, FLAGS.image_size, 1]), 8)

        label_summary(labels, weights, 'label')
        label_summary(output_predictions, weights, 'output_predictions')
        tf.summary.image('logits', tf.expand_dims(output_probs[:, :, :, 1], 3))

      elif output == 'regression':
        labels = samples['label']
        ignore_mask = model.get_ignore_mask(labels, FLAGS)

        accurate = calc_accuracy_in_box(labels, output_probs, ignore_mask)
        metric_map['inBoxAccuracy'] = tf.metrics.mean(accurate)

    tf.summary.image('image', samples['image'], 8)

    metrics_to_values, metrics_to_updates = slim.metrics.aggregate_metric_map(
        metric_map)

    for metric_name, metric_value in metrics_to_values.iteritems():
      metric_value = tf.Print(metric_value, [metric_value], metric_name)
      tf.summary.scalar(metric_name, metric_value)

    num_batches = int(math.ceil(FLAGS.num_samples / float(FLAGS.batch_size)))

    tf.logging.info('Eval num images %d', FLAGS.num_samples)
    tf.logging.info('Eval batch size %d and num batch %d', FLAGS.batch_size,
                    num_batches)

    slim.evaluation.evaluation_loop(
        master='',
        checkpoint_dir=FLAGS.checkpoint_dir,
        logdir=FLAGS.eval_logdir,
        num_evals=num_batches,
        eval_op=metrics_to_updates.values(),
        summary_op=tf.summary.merge_all(),
        max_number_of_evaluations=None,
        eval_interval_secs=FLAGS.eval_interval_secs)


if __name__ == '__main__':
  flags.mark_flag_as_required('checkpoint_dir')
  flags.mark_flag_as_required('eval_logdir')
  app.run()
