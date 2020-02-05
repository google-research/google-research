# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Segmentation results visualization on a given set of images.

See model.py for more details and usage.
"""

import math
import os.path
import time
import common  # pylint: disable=unused-import
from deeplab import save_annotation
import model
import model_input
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.contrib.slim as slim
from tensorflow.compat.v1.python.platform import app
from tensorflow.contrib import slim as contrib_slim

flags = tf.app.flags
FLAGS = flags.FLAGS

# Settings for log directories.

flags.DEFINE_string('vis_logdir', None, 'Where to write the event logs.')

flags.DEFINE_string('checkpoint_dir', None, 'Directory of model checkpoints.')

# Settings for evaluating the model.

flags.DEFINE_integer('batch_size', 32,
                     'The number of images in each batch during evaluation.')

flags.DEFINE_integer('num_vis_examples', 32,
                     'Number of examples for visualization.')

flags.DEFINE_integer('eval_interval_secs', 60,
                     'How often (in seconds) to run evaluation.')

flags.DEFINE_multi_float('eval_scales', [1.0],
                         'The scales to resize images for evaluation.')

flags.DEFINE_bool('add_flipped_images', False,
                  'Add flipped images for evaluation or not.')

flags.DEFINE_string('split', 'val',
                    'Which split of the dataset used for evaluation')

flags.DEFINE_string('colormap_type', 'pascal', 'Visualization colormap type.')

# The html template directory.
_HTML_TEMPLATE_DIR = '.'

# The folder where semantic segmentation predictions are saved.
_SEMANTIC_PREDICTION_SAVE_FOLDER = 'segmentation_results'

# The folder where raw semantic segmentation predictions are saved.
_RAW_SEMANTIC_PREDICTION_SAVE_FOLDER = 'raw_segmentation_results'

# The format to save image.
_IMAGE_FORMAT = '%06d_image'

# The format to save prediction
_PREDICTION_FORMAT = '%06d_prediction'

_LABEL_FORMAT = '%06d_label'


def _process_batch(sess, samples, semantic_predictions, labels, image_id_offset,
                   save_dir):
  """Evaluates one single batch qualitatively.

  Args:
    sess: TensorFlow session.
    samples: The input features.
    semantic_predictions: Model predictions.
    labels: Ground truth labels.
    image_id_offset: Image id offset for indexing images.
    save_dir: The directory where the predictions will be saved.
  Returns:
    The referring expressions.
  """
  (original_images, new_refs, semantic_predictions, labels) = sess.run(
      [samples['image'], samples['ref_exp'], semantic_predictions, labels])

  num_image = semantic_predictions.shape[0]
  for i in range(num_image):
    original_image = np.squeeze(original_images[i])
    semantic_prediction = np.squeeze(semantic_predictions[i])
    label = np.squeeze(labels[i])

    # Save image.
    save_annotation.save_annotation(
        original_image,
        save_dir,
        _IMAGE_FORMAT % (image_id_offset + i),
        add_colormap=False)

    # Save prediction.
    save_annotation.save_annotation(
        semantic_prediction,
        save_dir,
        _PREDICTION_FORMAT % (image_id_offset + i),
        add_colormap=True,
        colormap_type=FLAGS.colormap_type)

    save_annotation.save_annotation(
        label,
        save_dir,
        _LABEL_FORMAT % (image_id_offset + i),
        add_colormap=True,
        colormap_type=FLAGS.colormap_type)

  return new_refs.tolist()


def main(unused_argv):
  # Get dataset-dependent information.
  # Prepare for visualization.
  tf.gfile.MakeDirs(FLAGS.vis_logdir)
  save_dir = os.path.join(FLAGS.vis_logdir, _SEMANTIC_PREDICTION_SAVE_FOLDER)
  tf.gfile.MakeDirs(save_dir)
  raw_save_dir = os.path.join(FLAGS.vis_logdir,
                              _RAW_SEMANTIC_PREDICTION_SAVE_FOLDER)
  tf.gfile.MakeDirs(raw_save_dir)
  num_vis_examples = FLAGS.num_vis_examples

  print('Visualizing on set', FLAGS.split)

  g = tf.Graph()
  with g.as_default():
    samples = model_input.get_input_fn(FLAGS)()
    outputs_to_num_classes = model.get_output_to_num_classes(FLAGS)

    # Get model segmentation predictions.
    if tuple(FLAGS.eval_scales) == (1.0,):
      tf.logging.info('Performing single-scale test.')
      predictions, probs = model.predict_labels(
          samples['image'],
          samples,
          FLAGS,
          outputs_to_num_classes=outputs_to_num_classes,
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
          outputs_to_num_classes=outputs_to_num_classes,
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

    if FLAGS.output_mode == 'segment':
      predictions = tf.squeeze(
          tf.cast(predictions[FLAGS.output_mode], tf.int32))
      probs = probs[FLAGS.output_mode]

      labels = tf.squeeze(tf.cast(samples['label'], tf.int32))
      weights = tf.cast(
          tf.not_equal(
              labels,
              model_input.dataset_descriptors[FLAGS.dataset].ignore_label),
          tf.int32)

      labels *= weights
      predictions *= weights

      tf.train.get_or_create_global_step()
      saver = tf.train.Saver(contrib_slim.get_variables_to_restore())
      sv = tf.train.Supervisor(
          graph=g,
          logdir=FLAGS.vis_logdir,
          init_op=tf.global_variables_initializer(),
          summary_op=None,
          summary_writer=None,
          global_step=None,
          saver=saver)
      num_batches = int(math.ceil(num_vis_examples / float(FLAGS.batch_size)))
      last_checkpoint = None

      # Infinite loop to visualize the results when new checkpoint is created.
      while True:
        last_checkpoint = contrib_slim.evaluation.wait_for_new_checkpoint(
            FLAGS.checkpoint_dir, last_checkpoint)
        start = time.time()
        print('Starting visualization at ' +
              time.strftime('%Y-%m-%d-%H:%M:%S', time.gmtime()))
        print('Visualizing with model %s', last_checkpoint)

        print('Visualizing with model ', last_checkpoint)

        with sv.managed_session(
            FLAGS.master, start_standard_services=False) as sess:
          # sv.start_queue_runners(sess)
          sv.saver.restore(sess, last_checkpoint)

          image_id_offset = 0
          refs = []
          for batch in range(num_batches):
            print('Visualizing batch', batch + 1, num_batches)
            refs.extend(
                _process_batch(
                    sess=sess,
                    samples=samples,
                    semantic_predictions=predictions,
                    labels=labels,
                    image_id_offset=image_id_offset,
                    save_dir=save_dir))
            image_id_offset += FLAGS.batch_size

      print('Finished visualization at ' +
            time.strftime('%Y-%m-%d-%H:%M:%S', time.gmtime()))
      time_to_next_eval = start + FLAGS.eval_interval_secs - time.time()
      if time_to_next_eval > 0:
        time.sleep(time_to_next_eval)


if __name__ == '__main__':
  flags.mark_flag_as_required('checkpoint_dir')
  flags.mark_flag_as_required('vis_logdir')
  app.run()
