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

"""Pascal3D+ eval data loading."""

from absl import app
import numpy as np
import tensorflow as tf
import tensorflow_graphics.geometry.transformation as transformation


all_pascal_classes = ['aeroplane', 'bicycle', 'boat', 'bottle',
                      'bus', 'car', 'chair', 'diningtable', 'motorbike',
                      'sofa', 'train', 'tvmonitor']


def load_pascal_eval(
    # pylint: disable=dangerous-default-value
    input_filename, img_dims=[224, 224], object_classes='all',
    serialized_example_dataset=None):
  # pylint: enable=dangerous-default-value
  """Create a tf.data.Dataset for Pascal3D+ evaluation.

  Args:
    input_filename: path to TFRecord file containing test set records.
      See parse_example below to see the expected format of the records.
    img_dims: [H, W] resolution of the cropped output image.
    object_classes: string containing a single class name from
      all_pascal_classes above. Dataset will be filtered to contain only images
      from that class. Can be 'all' (default), to return all classes.
    serialized_example_dataset: optional, a tf.data.Dataset that returns
      serialized tf.train.Example protos. If not None, this dataset will be used
      in place of the input_filename.

  Returns:
    tf.data.Dataset over Pascal3D+ test set.
  """

  def parse_example_real(example, context=1.166666):
    """Parse a single test example stored in a serialized tf.train.Example.

    See feature_description dict below for the expected fields.

    Args:
      example: a serialized tf.train.Example.
      context: factor for enlarging bounding box for cropping.

    Returns:
      im_crop: cropped image.
      R: ground truth rotation matrix.
      az: ground truth azimuth.
      el: ground truth elevation.
      th: ground truth tile angle.
      valid_box: boolean, true if input bounding box was valid.
      easy: boolean, true if the example is classified as easy (e.g. not
        occluded/truncated).
      class_one_hot: one hot vector of class label.
    """
    feature_description = {
        'image_buffer': tf.io.FixedLenFeature([], tf.string),
        'left': tf.io.FixedLenFeature([], tf.float32),
        'top': tf.io.FixedLenFeature([], tf.float32),
        'right': tf.io.FixedLenFeature([], tf.float32),
        'bottom': tf.io.FixedLenFeature([], tf.float32),
        'azimuth': tf.io.FixedLenFeature([], tf.float32),
        'elevation': tf.io.FixedLenFeature([], tf.float32),
        'theta': tf.io.FixedLenFeature([], tf.float32),
        'easy': tf.io.FixedLenFeature([], tf.int64),
        'class_name': tf.io.FixedLenFeature([], tf.string),
        'class_num': tf.io.FixedLenFeature([], tf.int64),
    }
    fd = tf.io.parse_single_example(
        serialized=example, features=feature_description)
    im_enc = fd['image_buffer']
    im = tf.io.decode_jpeg(im_enc, channels=3)

    im = tf.image.convert_image_dtype(im, tf.float32)

    easy = fd['easy']
    class_num = tf.reshape(fd['class_num'], ())

    # Input bounding box pixel coordinates should be 1-based so subtract here.
    left = fd['left'] - 1.0
    top = fd['top'] - 1.0
    right = fd['right'] - 1.0
    bottom = fd['bottom'] - 1.0

    # Bounding box can be invalid at two points, input or after clip.
    valid_box = left < right and top < bottom
    im_crop = im
    if valid_box:
      mid_left_right = (right + left) / 2.0
      mid_top_bottom = (bottom + top) / 2.0
      # Add context.
      left = mid_left_right - context * (mid_left_right - left)
      right = mid_left_right + context * (right - mid_left_right)
      top = mid_top_bottom - context * (mid_top_bottom - top)
      bottom = mid_top_bottom + context * (bottom - mid_top_bottom)
      # Crop takes normalized coordinates.
      im_shape = tf.cast(tf.shape(im), tf.float32)
      y1 = tf.cast(top, tf.float32) / (im_shape[0] - 1.0)
      y2 = tf.cast(bottom, tf.float32) / (im_shape[0] - 1.0)
      x1 = tf.cast(left, tf.float32) / (im_shape[1] - 1.0)
      x2 = tf.cast(right, tf.float32) / (im_shape[1] - 1.0)
      y1 = tf.clip_by_value(y1, 0.0, 1.0)
      y2 = tf.clip_by_value(y2, 0.0, 1.0)
      x1 = tf.clip_by_value(x1, 0.0, 1.0)
      x2 = tf.clip_by_value(x2, 0.0, 1.0)
      valid_box = y1 < y2 and x1 < x2
      if valid_box:
        bbox = tf.reshape(tf.stack([y1, x1, y2, x2]), (1, 4))
        imb = tf.expand_dims(im, 0)
        im_crop = tf.image.crop_and_resize(
            image=imb, boxes=bbox, box_indices=[0], crop_size=img_dims)[0]

    # Inputs are in degrees, convert to rad.
    az = tf.reshape(fd['azimuth'], (1, 1)) * np.pi / 180.0
    el = tf.reshape(fd['elevation'], (1, 1)) * np.pi / 180.0
    th = tf.reshape(fd['theta'], (1, 1)) * np.pi / 180.0

    # R = R_z(th) * R_x(el−pi/2) * R_z(−az).
    rot1 = transformation.rotation_matrix_3d.from_euler(
        tf.concat([tf.zeros_like(az), tf.zeros_like(az), -az], -1))
    rot2 = transformation.rotation_matrix_3d.from_euler(
        tf.concat([el-np.pi/2.0, tf.zeros_like(el), th], -1))
    rot = tf.matmul(rot2, rot1)
    rot = tf.reshape(rot, (3, 3))

    class_one_hot = tf.one_hot(class_num, len(all_pascal_classes))

    return im_crop, rot, az, el, th, valid_box, tf.cast(easy,
                                                        tf.int32), class_one_hot

  def is_easy(im, rot, az, el, th, valid_box, easy, class_one_hot):
    del im, rot, az, el, th, valid_box, class_one_hot
    return tf.cond(easy > 0, lambda: True, lambda: False)

  def is_valid_box(im, rot, az, el, th, valid_box, easy, class_one_hot):
    del im, rot, az, el, th, easy, class_one_hot
    return valid_box

  if serialized_example_dataset is not None:
    data_real = serialized_example_dataset
  else:
    data_real = tf.data.TFRecordDataset(input_filename)

  data_real = data_real.map(parse_example_real, num_parallel_calls=4)
  data_real = data_real.filter(is_valid_box)
  # The common setting is to test only on the easy examples.
  data_real = data_real.filter(is_easy)
  # No augmentation since this is a data loader for the test set.

  # Filter by object class if all classes are not specified.
  if object_classes != 'all':
    class_ind = all_pascal_classes.index(object_classes)
    def _filter_by_class(im, rot, az, el, th, valid_box, easy, class_one_hot):
      del im, rot, az, el, th, valid_box, easy
      return tf.reduce_all(
          class_one_hot == tf.one_hot(class_ind, len(all_pascal_classes)))
    data_real = data_real.filter(_filter_by_class)

  data_real = data_real.map(lambda im, rot, *args: (im, rot))

  return data_real


