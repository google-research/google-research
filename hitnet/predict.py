# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""HITNet prediction main file.

This script processes pairs of images with a frozen HITNet model and saves the
predictions as 16bit PNG or fp32 PFM files.
"""
import glob
import io

from absl import app
from absl import flags
import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf

_PRINT_SUMMARY_EVERY_X_LINES = 20

flags.DEFINE_string('data_pattern', 'data', 'Input sstable file pattern')
flags.DEFINE_string('model_path', 'model_path', 'Path to frozen model file')
flags.DEFINE_float('png_disparity_factor', 256, 'Disparity multiplcation factor'
                   ' for output.')
flags.DEFINE_string('iml_pattern', 'left*.png', 'Input left image pattern')
flags.DEFINE_string('imr_pattern', 'right*.png', 'Input right image pattern')
flags.DEFINE_string('gtl_pattern', 'gtl*.png', 'Input left gt pattern')
flags.DEFINE_string('gtr_pattern', 'gtr*.png', 'Input left gt pattern')
flags.DEFINE_integer('max_test_number', 0, 'Max number of images to test.')
flags.DEFINE_integer('input_channels', 3,
                     'Number of input channels for the model.')
flags.DEFINE_boolean('evaluate', False, 'Compute metrics.')
flags.DEFINE_boolean(
    'predict_right', False,
    'Whether to query and save disparity for secondary image.')
flags.DEFINE_boolean('save_png', True, 'Whether to save output as pfm.')
flags.DEFINE_boolean('save_pfm', False, 'Whether to save output as pfm.')
flags.DEFINE_integer('crop_left', 0, 'Crop input right after load.')
flags.DEFINE_integer('crop_right', 0, 'Crop input right after load.')
flags.DEFINE_integer('crop_top', 0, 'Crop input right after load.')
flags.DEFINE_integer('crop_bottom', 0, 'Crop input right after load.')
FLAGS = flags.FLAGS


def image_as_array(filename):
  """Reads an image file and converts it to fp32."""
  with open(filename, 'rb') as f:
    image = np.array(Image.open(f)).astype(np.float32)
  return image


def pfm_as_bytes(filename):
  """Reads a disparity map groundtruth file (.pfm)."""
  with open(filename, 'rb') as f:
    header = f.readline().strip()
    width, height = [int(x) for x in f.readline().split()]
    scale = float(f.readline().strip())
    endian = '<' if scale < 0 else '>'
    shape = (height, width, 3) if header == 'PF' else (height, width, 1)
    data = np.frombuffer(f.read(), endian + 'f')
    data = data.reshape(shape)[::-1]  # PFM stores data upside down

  return data


def load_images(file_names,
                input_channels=3,
                crop_left=0,
                crop_right=0,
                crop_top=0,
                crop_bottom=0):
  """Load an image pair and optionally a GT pair from files.

  Optionally crops the inputs before they are seen by the model and model
  preprocessor.

  Args:
    file_names: Tuple with input and GT filenames.
    input_channels: number of input channels required by the frozen model.
    crop_left: Left crop amount.
    crop_right: Right crop amount.
    crop_top: Top crop amount.
    crop_bottom: Bottom crop amount.

  Returns:
    An np array with left and right images and optionall left and right GT.
  """
  left = image_as_array(file_names[0])
  right = image_as_array(file_names[1])
  gt = None
  if len(file_names) > 2:
    gt = pfm_as_bytes(file_names[2])
  if len(file_names) > 3:
    gtr = pfm_as_bytes(file_names[3])
    gt = np.concatenate((gt, gtr), axis=-1)
  num_dims = len(left.shape)
  # Make sure input images have 3-dim shape and 3 channels.
  if num_dims < 3:
    left = np.expand_dims(left, axis=-1)
    right = np.expand_dims(right, axis=-1)
    left = np.tile(left, (1, 1, input_channels))
    right = np.tile(right, (1, 1, input_channels))
  else:
    _, _, channels = left.shape
    if channels > input_channels:
      left = left[:, :, :input_channels]
      right = right[:, :, :input_channels]
  left = left[crop_top:, crop_left:, :]
  right = right[crop_top:, crop_left:, :]
  if gt is not None:
    gt = gt[crop_top:, crop_left:, :]
  if crop_bottom > 0:
    left = left[:-crop_bottom, :, :]
    right = right[:-crop_bottom, :, :]
    if gt is not None:
      gt = gt[:-crop_bottom, :, :]
  if crop_right > 0:
    left = left[:, :-crop_right, :]
    right = right[:, :-crop_right, :]
    if gt is not None:
      gt = gt[:, :-crop_right, :]

  np_images = np.concatenate((left, right), axis=-1) / 255.0
  return np_images, gt


def encode_image_as_16bit_png(data, filename):
  with io.BytesIO() as im_bytesio:
    height, width = data.shape
    array_bytes = data.astype(np.uint16).tobytes()
    array_img = Image.new('I', (width, height))
    array_img.frombytes(array_bytes, 'raw', 'I;16')
    array_img.save(im_bytesio, format='png')
    with open(filename, 'wb') as f:
      f.write(im_bytesio.getvalue())


def encode_image_as_pfm(data, filename):
  with open(filename, 'wb') as f:
    f.write(bytes('Pf\n', 'ascii'))
    f.write(bytes('%d %d\n' % (data.shape[1], data.shape[0]), 'ascii'))
    f.write(bytes('-1.0\n', 'ascii'))
    f.write(data[::-1].tobytes())  # PFM stores data upside down


def evaluate(disparity, gt, psm_threshold=192, max_disparity=1e6):
  """Computes metrics for predicted disparity against GT.

  Computes:
    PSM EPE: average disparity error for pixels with less than psm_threshold GT
    disparity value.
    bad_X: percent of pixels with disparity error larger than X. The divisor is
    the number of pixels with valid GT in the image.

  Args:
    disparity: Predicted disparity.
    gt: GT disparity.
    psm_threshold: Disparity threshold to compute PSM EPE.
    max_disparity: Maximum valid GT disparity.

  Returns:
    An np array with example metrics.
    [psm_epe, bad_0.1, bad_0.5, bad_1.0, bad_2.0, bad_3.0].
  """
  gt_mask = np.where((gt > 0) & (gt < max_disparity), np.ones_like(gt),
                     np.zeros_like(gt))
  gt_diff = np.where(gt_mask > 0, np.abs(gt - disparity), np.zeros_like(gt))
  psm_mask = np.where(gt < psm_threshold, gt_mask, np.zeros_like(gt))
  gt_mask_count = np.sum(gt_mask) + 1e-5
  psm_mask_count = np.sum(psm_mask) + 1e-5
  bad01 = np.where(gt_diff > 0.1, np.ones_like(gt_diff), np.zeros_like(gt_diff))
  bad05 = np.where(gt_diff > 0.5, np.ones_like(gt_diff), np.zeros_like(gt_diff))
  bad1 = np.where(gt_diff > 1.0, np.ones_like(gt_diff), np.zeros_like(gt_diff))
  bad2 = np.where(gt_diff > 2.0, np.ones_like(gt_diff), np.zeros_like(gt_diff))
  bad3 = np.where(gt_diff > 3.0, np.ones_like(gt_diff), np.zeros_like(gt_diff))

  bad01 = 100.0 * np.sum(bad01 * gt_mask) / gt_mask_count
  bad05 = 100.0 * np.sum(bad05 * gt_mask) / gt_mask_count
  bad1 = 100.0 * np.sum(bad1 * gt_mask) / gt_mask_count
  bad2 = 100.0 * np.sum(bad2 * gt_mask) / gt_mask_count
  bad3 = 100.0 * np.sum(bad3 * gt_mask) / gt_mask_count
  psm_epe = np.sum(gt_diff * psm_mask) / psm_mask_count
  return np.array([psm_epe, bad01, bad05, bad1, bad2, bad3])


def main(argv):
  del argv  # Unused

  np_aggregates = None
  test_count = 0
  with tf.device('/cpu:0'):
    graph_def = tf.GraphDef()
    with open(FLAGS.model_path, 'rb') as f:
      graph_def.ParseFromString(f.read())

    # generate lists of images to process
    iml_files = sorted(glob.glob(FLAGS.data_pattern + FLAGS.iml_pattern))
    imr_files = sorted(glob.glob(FLAGS.data_pattern + FLAGS.imr_pattern))
    if FLAGS.evaluate:
      gtl_files = sorted(glob.glob(FLAGS.data_pattern + FLAGS.gtl_pattern))
      if not FLAGS.predict_right:
        all_files = zip(iml_files, imr_files, gtl_files)
      else:
        gtr_files = sorted(glob.glob(FLAGS.data_pattern + FLAGS.gtr_pattern))
        all_files = zip(iml_files, imr_files, gtl_files, gtr_files)
    else:
      all_files = zip(iml_files, imr_files)

    for file_names in all_files:
      if test_count > FLAGS.max_test_number:
        break
      print(file_names)
      # Load new pair of images to process.
      np_images, np_gt = load_images(file_names, FLAGS.input_channels,
                                     FLAGS.crop_left, FLAGS.crop_right,
                                     FLAGS.crop_top, FLAGS.crop_bottom)

      filename = file_names[0].replace('.png', '')
      with tf.Graph().as_default() as default_graph:
        tf.import_graph_def(graph_def, name='graph')
        # Setup input-output tensors for the frozen model.
        xl = default_graph.get_tensor_by_name('graph/input:0')
        reference = default_graph.get_tensor_by_name(
            'graph/reference_output_disparity:0')
        if FLAGS.predict_right:
          secondary = default_graph.get_tensor_by_name(
              'graph/secondary_output_disparity:0')

        # Run the model.
        with tf.Session(graph=default_graph) as sess:

          feed_dict = {xl: np.expand_dims(np_images, 0)}
          if FLAGS.predict_right:
            (reference_disparity, secondary_disparity) = sess.run(
                (reference, secondary), feed_dict=feed_dict)
          else:
            reference_disparity = sess.run(reference, feed_dict=feed_dict)
        if FLAGS.evaluate:
          if FLAGS.predict_right:
            # Treat left and predictions as separate examples, same as in PSMNet
            # code.
            np_result_reference = evaluate(reference_disparity, np_gt[:, :, :1])
            np_result_secondary = evaluate(secondary_disparity, np_gt[:, :, 1:])
            np_result = np_result_reference + np_result_secondary
            np_result = 0.5 * np_result
          else:
            np_result = evaluate(reference_disparity, np_gt)
          if np_aggregates is not None:
            np_aggregates += np_result
          else:
            np_aggregates = np_result
          if not test_count % _PRINT_SUMMARY_EVERY_X_LINES:
            to_print = np_aggregates / ((float)(test_count + 1))
            print(test_count, to_print)
        # Save output disparity.
        if FLAGS.save_png:
          encode_image_as_16bit_png(
              reference_disparity[0, :, :, 0] * FLAGS.png_disparity_factor,
              filename + '_reference.png')
          if FLAGS.predict_right:
            encode_image_as_16bit_png(
                secondary_disparity[0, :, :, 0] * FLAGS.png_disparity_factor,
                filename + '_secondary.png')
        if FLAGS.save_pfm:
          encode_image_as_pfm(
              reference_disparity[0, :, :, 0] * FLAGS.png_disparity_factor,
              filename + '_reference.pfm')
          if FLAGS.predict_right:
            encode_image_as_16bit_png(
                secondary_disparity[0, :, :, 0] * FLAGS.png_disparity_factor,
                filename + '_secondary.pfm')
        test_count += 1

  if FLAGS.evaluate:
    print('Images processed:')
    print(test_count)
    print('psm_epe bad_0.1 bad_0.5 bad_1.0 bad_2.0 bad_3.0')
    to_print = np_aggregates / ((float)(test_count))
    print(to_print)


if __name__ == '__main__':
  app.run(main)
