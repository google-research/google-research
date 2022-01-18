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

"""Evaluates trained pose estimation models on pascal3d images."""
import os

from absl import app
from absl import flags
import numpy as np
from scipy.spatial.transform import Rotation
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string('model_dir',
                    'car_regression/',
                    'The path to the keras model.')
flags.DEFINE_string('images_dir',
                    'PASCAL3D+_release1.1/PASCAL/VOCdevkit/VOC2012/JPEGImages/',
                    'The directory of the test images.')
flags.DEFINE_string('dict_dir', 'cars_with_keypoints/',
                    'The directory with the tfrecords of images to use as a '
                    'dictionary in the lookup method.')
flags.DEFINE_string('object_class', 'cars',
                    'The object category to evaluate on: cars or chairs.')
flags.DEFINE_string('mode', 'regression',
                    'The mode of obtaining a pose for evaluation: regression or'
                    ' lookup.')
flags.DEFINE_integer('dict_size', 1800, 'The size of the dictionary to use for '
                     'the lookup method.')


@tf.function
def tightcrop_transparency_image(image, final_image_dims=(128, 128),
                                 alpha_channel_threshold=1e-6):
  """Uses the alpha channel of an image to tight-crop the foreground.

  Args:
    image: [H, W, 4] tensor.
    final_image_dims: The height H' and width W' of the returned image.
    alpha_channel_threshold: The value below which is considered background.
  Returns:
    [H', W', 4] tensor, image after being tight cropped and resized.
  """

  height = tf.cast(tf.shape(image)[0], tf.float32)
  width = tf.cast(tf.shape(image)[1], tf.float32)

  xymin = tf.reduce_min(
      tf.where(tf.math.greater(image[:, :, 3], alpha_channel_threshold)),
      axis=0)
  xymax = tf.reduce_max(
      tf.where(tf.math.greater(image[:, :, 3], alpha_channel_threshold)),
      axis=0)

  bounding_box = tf.cast(tf.concat([xymin, xymax], 0), tf.float32)
  bounding_box = tf.stack([
      bounding_box[0] / height, bounding_box[1] / width,
      bounding_box[2] / height, bounding_box[3] / width
  ], 0)

  cropped_image = tf.image.crop_and_resize(
      tf.expand_dims(image, 0), [bounding_box], [0], final_image_dims)[0]

  return cropped_image


def load_shapenet(dict_dir, object_class='car'):
  """Load shapenet renderings into a tf.data.Dataset.

  Args:
    dict_dir: The path to the tfrecords to use as a dictionary.
    object_class: car or chair (string).

  Returns:
    tf.data.Dataset of tuples of images and their rotation matrices, shapes
    ([128, 128, 3], [3, 3])
  """

  def parse_feature_dict(example):
    images = tf.stack([tf.image.decode_png(example['img0'], channels=4),
                       tf.image.decode_png(example['img1'], channels=4)], 0)
    camera_mats = tf.stack([tf.reshape(example['mv0'], [4, 4]),
                            tf.reshape(example['mv1'], [4, 4])], 0)
    images = tf.image.convert_image_dtype(images, tf.float32)
    camera_mats = camera_mats[:, :3, :3]
    return images, camera_mats

  features_dict = {
      'img0': tf.io.FixedLenFeature([], tf.string),
      'img1': tf.io.FixedLenFeature([], tf.string),
      'mv0': tf.io.FixedLenFeature([16], tf.float32),
      'mv1': tf.io.FixedLenFeature([16], tf.float32),
  }
  tfrecord_fname = os.path.join(dict_dir, '{:04d}.tfrecord')
  class_id = ['chair', 'car'].index(object_class)
  test_rec_num, max_latitude = [[47, 1.3], [182, 0.5]][class_id]
  dataset = tf.data.TFRecordDataset([tfrecord_fname.format(test_rec_num),
                                     tfrecord_fname.format(test_rec_num+1)])

  dataset = dataset.map(
      lambda example: tf.io.parse_example(example, features_dict))
  dataset = dataset.map(parse_feature_dict)

  dataset = dataset.unbatch()
  # Filters out rendered images from rarely seen elevations in real images
  dataset = dataset.filter(
      lambda _, rotation_matrix: rotation_matrix[2, 1] < np.sin(max_latitude))

  def tightcrop_and_correct(image, rotation_matrix):
    image = tightcrop_transparency_image(image)[Ellipsis, :3]
    # The rendering axes are different from the assumed axes
    correction_mat = np.float32([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
    rotation_matrix = tf.matmul(rotation_matrix, correction_mat)
    return image, rotation_matrix

  dataset = dataset.map(tightcrop_and_correct)

  return dataset


def convert_euler_angles_to_rotmat(euler_angles):
  euler_angles = np.reshape(euler_angles, [-1, 3])
  azimuth, elevation, tilt = np.split(euler_angles, 3, axis=-1)
  rotation_matrix = Rotation.from_euler(
      'zxz',
      np.concatenate([-azimuth, elevation - np.pi / 2., tilt], 1)).as_matrix()
  return rotation_matrix


def geodesic_dist_rotmats(rotation_matrices1, rotation_matrices2):
  """Computes the geodesic distance between two sets of rotation matrices.

  Args:
    rotation_matrices1: [N, 3, 3] tensor of rotation matrices.
    rotation_matrices2: [M, 3, 3] tensor of rotation matrices.
  Returns:
    geodesic_dists: [N, M] tensor of distances (in radians).
  """

  rotation_matrices1 = tf.cast(rotation_matrices1, tf.float32)
  rotation_matrices2 = tf.cast(rotation_matrices2, tf.float32)
  product = tf.matmul(rotation_matrices1, rotation_matrices2, transpose_b=True)
  geodesic_dists = tf.math.acos(
      tf.clip_by_value((tf.linalg.trace(product)-1.)/2., -1., 1.))
  return geodesic_dists


def parse_image(annotation_line, final_image_dims=(128, 128)):
  """Loads an image, tight-crops it, and computes rotation matrix.

  Args:
    annotation_line: string containing the filename, bounding box coordinates,
      and Euler angles.
    final_image_dims: The final (H', W') of the returned images.

  Returns:
    image: [128, 128, 3] image of the tight-cropped object.
    rotation_matrix: [3, 3] tensor.
  """
  entries = annotation_line.split()
  image_fname, left, top, right, bottom, azimuth, elevation, tilt = entries

  image_path = os.path.join(FLAGS.images_dir, image_fname)
  image = tf.io.decode_image(tf.io.read_file(image_path), dtype=tf.float32)

  left, top, right, bottom = np.float32([left, top, right, bottom]) - 1
  azimuth, elevation, tilt = np.float32([azimuth, elevation, tilt])

  image_shape = tf.cast(tf.shape(image), tf.float32)
  y1 = top / (image_shape[0] - 1.)
  y2 = bottom / (image_shape[0] - 1.)
  x1 = left / (image_shape[1] - 1.)
  x2 = right / (image_shape[1] - 1.)
  bounding_box = tf.stack([y1, x1, y2, x2])
  bounding_box = tf.clip_by_value(bounding_box, 0., 1.)

  image = tf.image.crop_and_resize(image[tf.newaxis], [bounding_box], [0],
                                   final_image_dims)[0]
  # Inputs are in degrees, convert to radians
  azimuth = tf.reshape(azimuth, [1]) * np.pi / 180.
  elevation = tf.reshape(elevation, [1]) * np.pi / 180.
  tilt = tf.reshape(tilt, [1]) * np.pi / 180.
  tilt = -tilt

  rotation_matrix = convert_euler_angles_to_rotmat([azimuth, elevation, tilt])

  return image, rotation_matrix


def main(_):
  model = tf.keras.models.load_model(FLAGS.model_dir)
  mode = FLAGS.mode
  object_class = FLAGS.object_class

  # Load the accompanying txt file with the test images and annotations
  test_fname = f'isolating_factors/pose_estimation/{object_class}_test.txt'
  with open(test_fname, 'r') as f:
    data = f.readlines()

  test_images, test_rotation_matrices = [[], []]
  for image_line in data:
    image, rotation_matrix = parse_image(image_line)
    test_images.append(image)
    test_rotation_matrices.append(rotation_matrix)
  test_rotation_matrices = tf.concat(test_rotation_matrices, 0)

  if mode == 'lookup':
    dataset_dict = load_shapenet(FLAGS.dict_dir, object_class=object_class)
    dict_size = FLAGS.dict_size

    dict_embeddings, dict_rotation_matrices = [[], []]
    chunk_size = 64  # To chunk up the process of embedding the dict
    for images, rotation_matrices in dataset_dict.shuffle(4000).batch(
        chunk_size).take(1 + dict_size // chunk_size):
      embeddings = model(images, training=False)
      dict_embeddings.append(embeddings)
      dict_rotation_matrices.append(rotation_matrices)
    dict_embeddings = tf.concat(dict_embeddings, 0)[:dict_size]
    dict_rotation_matrices = tf.concat(dict_rotation_matrices, 0)[:dict_size]

    test_embeddings = []
    for image in test_images:
      embedding = model(tf.expand_dims(image, 0), training=False)
      test_embeddings.append(embedding)
    test_embeddings = tf.concat(test_embeddings, 0)

    # For each embedding, get the nearest neighbor using cosine similarity
    test_embeddings, _ = tf.linalg.normalize(test_embeddings, ord=2, axis=-1)
    dict_embeddings, _ = tf.linalg.normalize(dict_embeddings, ord=2, axis=-1)
    similarity_matrix = tf.matmul(test_embeddings, dict_embeddings,
                                  transpose_b=True)
    closest_across_dict = tf.math.argmax(similarity_matrix, axis=1)
    predicted_rotation_matrices = tf.gather(dict_rotation_matrices,
                                            closest_across_dict)

  else:  # Regression
    predicted_rotation_matrices = []
    for image in test_images:
      pred_euler_angles = model(tf.expand_dims(image, 0), training=False)
      predicted_rotation_matrices.append(
          convert_euler_angles_to_rotmat(pred_euler_angles))
    predicted_rotation_matrices = tf.concat(predicted_rotation_matrices, 0)

  errors = geodesic_dist_rotmats(test_rotation_matrices,
                                 predicted_rotation_matrices)
  errors = np.rad2deg(errors)
  median_angular_error = np.median(errors)
  accuracy15 = np.average(errors < 15)
  accuracy30 = np.average(errors < 30)
  print('Median angular error: {:.3f} deg.'.format(median_angular_error))
  print('Accuracy at 15 deg: {:.3f}.'.format(accuracy15))
  print('Accuracy at 30 deg: {:.3f}.'.format(accuracy30))


if __name__ == '__main__':
  app.run(main)
