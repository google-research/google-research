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

# Lint as: python2, python3
"""Utility functions for deep homography estimation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range
import tensorflow as tf
from tensorflow.contrib import image as contrib_image


def get_two_dummy_frames():
  """Returns two random RGB frames."""

  return tf.random.uniform((2, 600, 800, 3))


def get_batchpairs_ava(unused_source,
                       max_shift,
                       batch_size=2,
                       queue_size=60,
                       num_threads=3,
                       train_height=128,
                       train_width=128,
                       unused_frame_gap=0,
                       pixel_noise=0.0,
                       mix=True,
                       screen=False,
                       mode='train',
                       to_gray=True):
  """Prepares training image batches from AVA dataset.

  Args:
    unused_source: pattern of input data containing source images from AVA
      dataset. Unused now.
    max_shift: the range each image corner point can move
    batch_size: the size of training or testing batches
    queue_size: the queue size of the shuffle buffer
    num_threads: the number of threads of the shuffle buffer
    train_height: the height of the training/testing images
    train_width: the width of the training/testing images
    unused_frame_gap: the temporal gap between two selected frames. Unused now.
    pixel_noise: the magnitude of additive noises
    mix: whether mix the magnitude of corner point shifts
    screen: whether remove highly distorted homographies
    mode: 'train' or 'eval', specifying whether preparing images for training or
      testing
    to_gray: whether prepare color or gray scale training images
  Returns:
    a batch of training images and the corresponding ground-truth homographies
  """
  raw_frames = get_two_dummy_frames()
  return augment_seqs_ava(raw_frames, 2, max_shift,
                          batch_size=batch_size, queue_size=queue_size,
                          num_threads=num_threads, train_height=train_height,
                          train_width=train_width, pixel_noise=pixel_noise,
                          mix=mix, screen=screen, mode=mode, to_gray=to_gray)


def get_batchseqs_ava(unused_source,
                      num_frame,
                      max_shift,
                      batch_size=2,
                      queue_size=60,
                      num_threads=3,
                      train_height=128,
                      train_width=128,
                      pixel_noise=0.0,
                      mix=True,
                      screen=False,
                      mode='train',
                      to_gray=True):
  """Prepares training sequence batches from AVA dataset.

  Note currently this function only generates dummy data.

  Args:
    unused_source: pattern of input data containing source images from AVA
      dataset. Not used now.
    num_frame: the number of frames in a sequence
    max_shift: the range each image corner point can move
    batch_size: the size of training or testing batches
    queue_size: the queue size of the shuffle buffer
    num_threads: the number of threads of the shuffle buffer
    train_height: the height of the training/testing images
    train_width: the width of the training/testing images
    pixel_noise: the magnitude of additive noises
    mix: whether mix the magnitude of corner point shifts
    screen: whether remove highly distorted homographies
    mode: 'train' or 'eval', specifying whether preparing images for training or
      testing
    to_gray: whether prepare color or gray scale training images
  Returns:
    a batch of training images and the corresponding ground-truth homographies
  """
  raw_frames = get_two_dummy_frames()
  return augment_seqs_ava(raw_frames, num_frame, max_shift,
                          batch_size=batch_size, queue_size=queue_size,
                          num_threads=num_threads, train_height=train_height,
                          train_width=train_width, pixel_noise=pixel_noise,
                          mix=mix, screen=screen, mode=mode, to_gray=to_gray)


def augment_seqs_ava(raw_frames, num_frame, max_shift, batch_size=2,
                     queue_size=60, num_threads=3, train_height=128,
                     train_width=128, pixel_noise=0.0, mix=True, screen=False,
                     mode='train', to_gray=True):
  """Prepares training sequence batches from AVA dataset.

  Args:
    raw_frames: input video frames from AVA dataset
    num_frame: the number of frames in a sequence
    max_shift: the range each image corner point can move
    batch_size: the size of training or testing batches
    queue_size: the queue size of the shuffle buffer
    num_threads: the number of threads of the shuffle buffer
    train_height: the height of the training/testing images
    train_width: the width of the training/testing images
    pixel_noise: the magnitude of additive noises
    mix: whether mix the magnitude of corner point shifts
    screen: whether remove highly distorted homographies
    mode: 'train' or 'eval', specifying whether preparing images for training or
      testing
    to_gray: whether prepare color or gray scale training images
  Returns:
    a batch of training images and the corresponding ground-truth homographies
  """
  if to_gray:
    output_frames = tf.image.rgb_to_grayscale(raw_frames)
    num_channel = 1
  else:
    output_frames = raw_frames
    num_channel = 3

  frame_height = tf.to_float(tf.shape(output_frames)[1])
  frame_width = tf.to_float(tf.shape(output_frames)[2])

  if mix:
    p = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32)
    scale = (tf.to_float(tf.greater(p, 0.1)) + tf.to_float(tf.greater(p, 0.2))
             + tf.to_float(tf.greater(p, 0.3))) / 3
  else:
    scale = 1.0
  new_max_shift = max_shift * scale
  rand_shift_base = tf.random_uniform([num_frame, 8], minval=-new_max_shift,
                                      maxval=new_max_shift, dtype=tf.float32)
  crop_width = frame_width - 2 * new_max_shift - 1
  crop_height = frame_height - 2 * new_max_shift - 1
  ref_window = tf.to_float(tf.stack([0, 0, 0, crop_height - 1, crop_width - 1,
                                     0, crop_width - 1, crop_height - 1]))
  if screen:
    new_shift_list = []
    flag_list = []
    hmg_list = []
    src_points = tf.reshape(ref_window, [4, 2])
    for i in range(num_frame):
      dst_points = tf.reshape(rand_shift_base[i] + ref_window + new_max_shift,
                              [4, 2])
      hmg = calc_homography_from_points(src_points, dst_points)
      hmg_list.append(hmg)
    for i in range(num_frame - 1):
      hmg = tf.matmul(tf.matrix_inverse(hmg_list[i + 1]), hmg_list[i])
      shift = homography_to_shifts(hmg, crop_width, crop_height)
      angles = calc_homography_distortion(crop_width, crop_height, shift)
      max_angle = tf.reduce_min(angles)
      flag = tf.to_float(max_angle >= -0.707)
      flag_list.append(flag)
      if i > 0:
        new_shift = rand_shift_base[i] * flag * flag_list[i - 1]
      else:
        new_shift = rand_shift_base[i] * flag
      new_shift_list.append(new_shift)
    new_shift_list.append(rand_shift_base[num_frame - 1]
                          * flag_list[num_frame - 2])
    rand_shift = tf.stack(new_shift_list)
  else:
    rand_shift = rand_shift_base

  mat_scale = tf.diag(tf.stack([crop_width / train_width,
                                crop_height / train_height, 1.0]))
  inv_mat_scale = tf.matrix_inverse(mat_scale)
  hmg_list = []
  frame_list = []
  for i in range(num_frame):
    src_points = tf.reshape(ref_window, [4, 2])
    dst_points = tf.reshape(rand_shift[i] + ref_window + new_max_shift, [4, 2])
    hmg = calc_homography_from_points(src_points, dst_points)
    hmg_list.append(hmg)
    transform = tf.reshape(hmg, [9]) / hmg[2, 2]
    warped = contrib_image.transform(output_frames[i], transform[:8],
                                     'bilinear')
    crop_window = tf.expand_dims(tf.stack(
        [0, 0, (crop_height - 1) / (frame_height - 1),
         (crop_width - 1) / (frame_width - 1)]), 0)
    resized_base = tf.image.crop_and_resize(
        tf.expand_dims(warped, 0), crop_window, [0],
        [train_height, train_width])
    resized = tf.squeeze(resized_base, [0])

    noise_im = tf.truncated_normal(shape=tf.shape(resized), mean=0.0,
                                   stddev=pixel_noise, dtype=tf.float32)
    noise_frame = normalize_image(tf.to_float(resized) + noise_im)
    frame_list.append(noise_frame)
  noise_frames = tf.reshape(
      tf.stack(frame_list, 2),
      (train_height, train_width, num_frame * num_channel))

  label_list = []
  for i in range(num_frame - 1):
    hmg_combine = tf.matmul(tf.matrix_inverse(hmg_list[i + 1]), hmg_list[i])
    hmg_final = tf.matmul(inv_mat_scale, tf.matmul(hmg_combine, mat_scale))
    label = homography_to_shifts(hmg_final, train_width, train_height)
    label_list.append(label)
  labels = tf.reshape(tf.stack(label_list, 0), [(num_frame - 1) * 8])

  if mode == 'train':
    min_after_dequeue = int(queue_size / 3)
  else:
    min_after_dequeue = batch_size * 3
  batch_frames, batch_labels = tf.train.shuffle_batch(
      [noise_frames, labels], batch_size=batch_size,
      num_threads=num_threads, capacity=queue_size,
      min_after_dequeue=min_after_dequeue, enqueue_many=False)

  return tf.cast(batch_frames, tf.float32), tf.cast(batch_labels, tf.float32)


def get_batchpairs_coco(unused_source,
                        max_shift,
                        batch_size=2,
                        queue_size=60,
                        num_threads=3,
                        train_height=128,
                        train_width=128,
                        pixel_noise=0.0,
                        mix=True,
                        screen=False,
                        to_gray=True,
                        mode='train'):
  """Prepares training image batches from MS COCO dataset.

  Note currently this function only generates dummy data.

  Args:
    unused_source: pattern of input data containing source images from MS COCO
      dataset. Not used now.
    max_shift: the range each image corner point can move
    batch_size: the size of training or testing batches
    queue_size: the queue size of the shuffle buffer
    num_threads: the number of threads of the shuffle buffer
    train_height: the height of the training/testing images
    train_width: the width of the training/testing images
    pixel_noise: the magnitude of additive noises
    mix: whether mix the magnitude of corner point shifts
    screen: whether remove highly distorted homographies
    to_gray: whether prepare color or gray scale training images
    mode: 'train' or 'eval', specifying whether preparing images for training or
      testing
  Returns:
    a batch of training images and the corresponding ground-truth homographies
  """
  frames = get_two_dummy_frames()
  if to_gray:
    output_frames = tf.image.rgb_to_grayscale(frames)
    num_channel = 1
  else:
    output_frames = frames
    num_channel = 3

  frame_height = tf.shape(output_frames)[1]
  frame_width = tf.shape(output_frames)[2]

  max_crop_shift_x = tf.cast(frame_width - train_width, tf.float32)
  max_crop_shift_y = tf.cast(frame_height - train_height, tf.float32)

  crop_shift_x = tf.random_uniform([], minval=max_shift+1,
                                   maxval=max_crop_shift_x-max_shift-1,
                                   dtype=tf.float32)
  crop_shift_y = tf.random_uniform([], minval=max_shift+1,
                                   maxval=max_crop_shift_y-max_shift-1,
                                   dtype=tf.float32)

  rand_shift_base = tf.random_uniform([8], minval=-max_shift, maxval=max_shift,
                                      dtype=tf.float32)
  if mix:
    p = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32)
    scale = (tf.to_float(tf.greater(p, 0.1)) + tf.to_float(tf.greater(p, 0.2))
             + tf.to_float(tf.greater(p, 0.3))) / 3
  else:
    scale = 1.0

  if screen:
    angles = calc_homography_distortion(train_width, train_height,
                                        scale * rand_shift_base)
    max_angle = tf.reduce_min(angles)
    rand_shift = tf.to_float(max_angle >= -0.707) * scale * rand_shift_base
  else:
    rand_shift = scale * rand_shift_base

  dy1 = crop_shift_y + rand_shift[1]
  dx1 = crop_shift_x + rand_shift[0]
  dy2 = crop_shift_y + train_height - 1 + rand_shift[3]
  dx2 = crop_shift_x + rand_shift[2]
  dy3 = crop_shift_y + rand_shift[5]
  dx3 = crop_shift_x + train_width - 1 + rand_shift[4]
  dy4 = crop_shift_y + train_height - 1 + rand_shift[7]
  dx4 = crop_shift_x + train_width - 1 + rand_shift[6]
  cropped_frame1 = subpixel_homography(output_frames[0],
                                       train_height, train_width,
                                       dy1, dx1, dy2, dx2, dy3, dx3, dy4, dx4)
  cropped_frame2 = subpixel_crop(output_frames[0], crop_shift_y,
                                 crop_shift_x, train_height, train_width)
  noise_im1 = tf.truncated_normal(shape=tf.shape(cropped_frame1), mean=0.0,
                                  stddev=pixel_noise, dtype=tf.float32)
  noise_im2 = tf.truncated_normal(shape=tf.shape(cropped_frame2), mean=0.0,
                                  stddev=pixel_noise, dtype=tf.float32)
  normalized_im1 = normalize_image(tf.cast(cropped_frame1, tf.float32)
                                   + noise_im1)
  normalized_im2 = normalize_image(tf.cast(cropped_frame2, tf.float32)
                                   + noise_im2)
  cropped_pair = tf.reshape(
      tf.stack((normalized_im1, normalized_im2), 2),
      (train_height, train_width, 2 * num_channel))
  label = rand_shift

  if mode == 'train':
    min_after_dequeue = int(queue_size / 3)
  else:
    min_after_dequeue = batch_size * 3
  batch_frames, batch_labels = tf.train.shuffle_batch(
      [cropped_pair, label], batch_size=batch_size,
      num_threads=num_threads, capacity=queue_size,
      min_after_dequeue=min_after_dequeue, enqueue_many=False)

  return tf.cast(batch_frames, tf.float32), tf.cast(batch_labels, tf.float32)


def subpixel_crop(image, y, x, height, width):
  """Crops out a region [x, y, x + width, y + height] from an image.

  Args:
    image: input image of shape [input_height, input_width, channels] and of
      data type uint8 or float32
    y: the y coordinate of the top left corner of the cropping window
    x: the x coordinate of the top left corner of the cropping window
    height: the height of the cropping window
    width: the width of the cropping window
  Returns:
    the cropping result of shape [height, width, channels] with the same type
    as image
  """
  transformation = tf.cast(tf.stack([1, 0, x, 0, 1, y, 0, 0]), tf.float32)
  translated = contrib_image.transform(image, transformation, 'bilinear')
  cropped = tf.image.crop_to_bounding_box(translated, 0, 0, height, width)
  return cropped


def subpixel_homography(image, height, width, dy1, dx1, dy2, dx2, dy3, dx3, dy4,
                        dx4):
  """Applies a homography to an image.

  Args:
    image: input image of shape [input_height, input_width, channels] and of
      data type uint8 or float32
    height: the output image height
    width: the output image width
    dy1: the vertical shift of the top left corner
    dx1: the horizontal shift of the top left corner
    dy2: the vertical shift of the bottom left corner
    dx2: the horizontal shift of the bottom left corner
    dy3: the vertical shift of the top right corner
    dx3: the horizontal shift of the top right corner
    dy4: the vertical shift of the bottom right corner
    dx4: the horizontal shift of the bottom right corner
  Returns:
    the warping result of shape [height, width, channels] with the same data
    type as image
  """
  rx1 = tf.cast(tf.stack([0, 0, 1, 0, 0, 0, 0, 0]), tf.float32)
  ry1 = tf.cast(tf.stack([0, 0, 0, 0, 0, 1, 0, 0]), tf.float32)
  rx2 = tf.cast(tf.stack([0, height - 1, 1, 0, 0, 0, 0, -(height - 1) * dx2]),
                tf.float32)
  ry2 = tf.cast(tf.stack([0, 0, 0, 0, height - 1, 1, 0, -(height - 1) * dy2]),
                tf.float32)
  rx3 = tf.cast(tf.stack([width - 1, 0, 1, 0, 0, 0, -(width - 1) * dx3, 0]),
                tf.float32)
  ry3 = tf.cast(tf.stack([0, 0, 0, width - 1, 0, 1, -(width - 1) * dy3, 0]),
                tf.float32)
  rx4 = tf.cast(tf.stack([width - 1, height - 1, 1, 0, 0, 0, -(width - 1) * dx4,
                          -(height - 1) * dx4]), tf.float32)
  ry4 = tf.cast(tf.stack([0, 0, 0, width - 1, height - 1, 1, -(width - 1) * dy4,
                          -(height - 1) * dy4]), tf.float32)
  mat = tf.stack([rx1, ry1, rx2, ry2, rx3, ry3, rx4, ry4])
  b = tf.reshape(tf.cast(tf.stack([dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4]),
                         tf.float32), [8, 1])
  inv_mat = tf.matrix_inverse(mat)
  transformation = tf.reshape(tf.matmul(inv_mat, b), [8])
  warped = contrib_image.transform(image, transformation, 'bilinear')
  cropped = tf.image.crop_to_bounding_box(warped, 0, 0, height, width)
  return cropped


def normalize_image(image):
  """Normalizes pixel values to the range of [-0.5, 0.5).

  Args:
    image: input image of shape [input_height, input_width, channels] and of
      data type uint8 or float32
  Returns:
    the normalized image of the same shape as image and of data type float32
  """

  return tf.to_float(image) / 256.0 - 0.5


def homography_warp_per_image(image, width, height, corner_shifts):
  """Transforms an input image using a specified homography.

  Args:
    image: input image of shape [input_height, input_width, channels] and of
      data type uint8 or float32
    width: the homograph is parameterized using the displacements of four
      image corners. width is the width of the image that the corner
      displacement is computed from.
    height: the image height
    corner_shifts: the displacements of the four image corner points of data
      type float32 and of shape [8]
  Returns:
    the warped result of the same shape as image and of data type float32
  """
  transform = shifts_to_homography(width, height, corner_shifts,
                                   is_forward=False, is_matrix=False)
  warped = contrib_image.transform(image, transform, 'bilinear')
  return warped, transform


def homography_warp_per_batch(batch_images, batch_shifts):
  """Transforms a batch of images using specified homographies.

  Args:
    batch_images: input images of shape [batch_size, input_height, input_width,
      channels] and of data type uint8 or float32
    batch_shifts: input batch of homographies parameterized using the
      displacements of four image corners
  Returns:
    batch_warped_image: the warped result of the same shape and data type as
      batch_images
    transform_list: the batch of homographies. Each homography is parameterized
      as an 8 x 1 vector that is the first 8 elements of the 3 x 3 homography
      matrix
  """
  batch_size = batch_images.get_shape().as_list()[0]
  dheight = tf.to_float(tf.shape(batch_images)[1])
  dwidth = tf.to_float(tf.shape(batch_images)[2])
  warped_image_list = []
  transform_list = []
  for i in range(0, batch_size):
    warped_image, transform = homography_warp_per_image(
        batch_images[i], dwidth, dheight, batch_shifts[i])
    warped_image_list.append(warped_image)
    transform_list.append(transform)
  batch_warped_image = tf.stack(warped_image_list, 0)
  return batch_warped_image, transform_list


def homography_scale_warp_per_image(image, width, height, ref_width, ref_height,
                                    corner_shifts):
  """Transforms an input image using a specified homography.

  Args:
    image: input image of shape [height, width, channels] and of data type uint8
      or float32
    width: the width of the input image
    height: the height of the input image
    ref_width: the homograph is parameterized using the displacements of four
      image corners. ref_width is the width of the original image that the
      corner displacement is computed from
    ref_height: the height of the original image that the corner displacement
      is computed from
    corner_shifts: the displacements of the four image corner points of data
      type float32 and of shape [8]
  Returns:
    the warped result of the same shape and data type as image
  """
  hmg_base = shifts_to_homography(ref_width, ref_height, corner_shifts,
                                  is_forward=False, is_matrix=False)
  sx = tf.to_float(ref_width) / tf.to_float(width)
  sy = tf.to_float(ref_height) / tf.to_float(height)
  vec_scale = tf.stack([1, sy / sx, 1 / sx, sx / sy, 1, 1 / sy, sx, sy])
  transform = tf.multiply(hmg_base, vec_scale)
  warped = contrib_image.transform(image, transform, 'bilinear')
  return warped, transform


def homography_scale_warp_per_batch(batch_images, ref_width, ref_height,
                                    batch_shifts):
  """Transforms a batch of input images.

  Args:
    batch_images: input images of shape [batch_size, height, width, channels]
      and of data type uint8 or float32
    ref_width: the homograph is parameterized using the displacements of four
      image corners. ref_width is the width of the original image that the
      corner displacement is computed from
    ref_height: the height of the original image that the corner displacement
      is computed from
    batch_shifts: the displacements of the four image corner points of data
      type float32 and of shape [batch_size, 8]
  Returns:
    the warped results of the same shape and data type as batch_images
  """
  batch_size = batch_images.get_shape().as_list()[0]
  height = tf.shape(batch_images)[1]
  width = tf.shape(batch_images)[2]
  warped_list = []
  hmg_list = []
  for i in range(0, batch_size):
    warped, hmg = homography_scale_warp_per_image(
        batch_images[i], width, height, ref_width, ref_height, batch_shifts[i])
    warped_list.append(warped)
    hmg_list.append(hmg)
  batch_warped_images = tf.reshape(tf.stack(warped_list, 0),
                                   (batch_size, height, width))
  return batch_warped_images, hmg_list


def calc_homography_from_points(src_points, dst_points, is_matrix=True):
  """Computes a homography from four pairs of corresponding points.

  Args:
    src_points: source points of shape [4, 2] and of data type float32 or int32
    dst_points: target points of shape [4, 2] and of data type float32 or int32
    is_matrix: whether represent the final homography using matrix or vector
  Returns:
    the output homography of data type float32. If is_matrix is True, it is of
      shape [3, 3]; otherwise [8]
  """
  mat_elements = []
  r_vec_elements = []
  for i in range(0, 4):
    rx = tf.to_float(tf.stack([src_points[i, 0], src_points[i, 1], 1, 0, 0, 0,
                               -dst_points[i, 0] * src_points[i, 0],
                               -dst_points[i, 0] * src_points[i, 1]]))
    ry = tf.to_float(tf.stack([0, 0, 0, src_points[i, 0], src_points[i, 1], 1,
                               -dst_points[i, 1] * src_points[i, 0],
                               -dst_points[i, 1] * src_points[i, 1]]))
    mat_elements.append(rx)
    mat_elements.append(ry)
    r_vec_elements.append(dst_points[i, 0])
    r_vec_elements.append(dst_points[i, 1])
  mat = tf.stack(mat_elements)
  r_vec = tf.reshape(tf.to_float(tf.stack(r_vec_elements)), [8, 1])
  inv_mat = tf.matrix_inverse(mat)
  transform = tf.reshape(tf.matmul(inv_mat, r_vec), [8])
  if is_matrix:
    hmg = tf.reshape(tf.concat([transform, [1.0]], 0), [3, 3])
  else:
    hmg = transform
  return hmg


def shifts_to_homography(width, height, corner_shifts, is_forward=True,
                         is_matrix=True):
  """Computes a homography from from displacements.

  Args:
    width: the homograph is parameterized using the displacements of four
      image corners. width is the width of the image that the corner
      displacement is computed from.
    height: the image height
    corner_shifts: the displacements of the four image corner points of data
      type float32 and of shape [8]
    is_forward: whether map from the input image to the shifted one
    is_matrix: whether represent the final homography using matrix or vector
  Returns:
    the output homography of data type float32. If is_matrix is True, it is of
      shape [3, 3]; otherwise [8]
  """
  ref_window = tf.to_float(tf.stack([0, 0, 0, height - 1, width - 1, 0,
                                     width - 1, height - 1]))
  src_points = tf.reshape(ref_window, [4, 2])
  dst_points = tf.reshape(ref_window + corner_shifts, [4, 2])
  if is_forward:
    return calc_homography_from_points(src_points, dst_points, is_matrix)
  else:
    return calc_homography_from_points(dst_points, src_points, is_matrix)


def apply_homography_to_point(hmg, x, y, is_matrix=True):
  """Transforms a point (x, y) with a homography hmg.

  Args:
    hmg: a homography transformation of data type float32. If mode is 'matrix',
      it is of shape [3, 3]; otherwise its shape is [8].
    x: the x coordinate of the point
    y: the y coordinate of the point
    is_matrix: whether the parameterization of the homography is matrix or not
  Returns:
    the transformed coordinate of the input point
  """
  if is_matrix:
    z1 = hmg[2, 0] * x + hmg[2, 1] * y + hmg[2, 2]
    x1 = (hmg[0, 0] * x + hmg[0, 1] * y + hmg[0, 2]) / z1
    y1 = (hmg[1, 0] * x + hmg[1, 1] * y + hmg[1, 2]) / z1
  else:
    z1 = hmg[6] * x + hmg[7] * y + 1.0
    x1 = (hmg[0] * x + hmg[1] * y + hmg[2]) / z1
    y1 = (hmg[3] * x + hmg[4] * y + hmg[5]) / z1
  return x1, y1


def homography_shift_mult_batch(batch_shift1, w1, h1, batch_shift2,
                                w2, h2, w, h):
  """Multiplies two homographies.

  Args:
    batch_shift1: a batch of homographies parameterized as the displacement of
      four corner points, with data type float32 and shape [batch_size, 8]
    w1: the width of the image where batch_shift1 is computed from
    h1: the height of the image where batch_shift1 is computed from
    batch_shift2: a batch of homographies parameterized as the displacement of
      four corner points, with the same data type and shape as batch_shift1
    w2: the width of the image where batch_shift2 is computed from
    h2: the height of the image where batch_shift2 is computed from
    w: the width of the image where the output corner_shift is computed from
    h: the height of the image where the output corner_shift is computed from
  Returns:
    the batch of homography multiplication results, with the same shape and data
    type as batch_shift1
  """
  batch_size = batch_shift1.get_shape().as_list()[0]
  corner_shifts_list = []
  for i in range(0, batch_size):
    corner_shifts = homography_shift_mult(batch_shift1[i], w1, h1,
                                          batch_shift2[i], w2, h2, w, h)
    corner_shifts_list.append(corner_shifts)
  return tf.stack(corner_shifts_list)


def homography_shift_mult(corner_shift1, w1, h1, corner_shift2, w2, h2, w, h):
  """Multiplies two homographies.

  Args:
    corner_shift1: a homography transformation parameterized as the displacement
      of four corner points. It is of data type float32 and of shape [8]
    w1: the width of the image where corner_shift1 is computed from
    h1: the height of the image where corner_shift1 is computed from
    corner_shift2: a homography transformation parameterized as the displacement
      of four corner points, with the same data type and shape as corner_shift1
    w2: the width of the image where corner_shift2 is computed from
    h2: the height of the image where corner_shift2 is computed from
    w: the width of the image where the output corner_shift is computed from
    h: the height of the image where the output corner_shift is computed from
  Returns:
    the product of the two homographies of the same shape and data type as
      corner_shift1
  """
  hmg1 = shifts_to_homography(w1, h1, corner_shift1, is_forward=False,
                              is_matrix=True)
  mat_scale1 = tf.reshape(tf.stack(
      [tf.to_float(w1) / tf.to_float(w), 0, 0, 0,
       tf.to_float(h1) / tf.to_float(h), 0, 0, 0, 1]), [3, 3])
  mat1 = tf.matmul(tf.matrix_inverse(mat_scale1), tf.matmul(hmg1, mat_scale1))

  hmg2 = shifts_to_homography(w2, h2, corner_shift2, is_forward=False,
                              is_matrix=True)
  mat_scale2 = tf.reshape(tf.stack(
      [tf.to_float(w2) / tf.to_float(w), 0, 0, 0,
       tf.to_float(h2) / tf.to_float(h), 0, 0, 0, 1]), [3, 3])
  mat2 = tf.matmul(tf.matrix_inverse(mat_scale2), tf.matmul(hmg2, mat_scale2))

  hmg = tf.matrix_inverse(tf.matmul(mat1, mat2))
  return homography_to_shifts(hmg, w, h, is_matrix=True)


def homography_to_shifts(hmg, width, height, is_matrix=True):
  """Converts a homography from matrix parameterization to corner shifts.

  Args:
    hmg: a homography transformation data type float32. If is_matrix is True, it
      has shape [3, 3]; otherwise [8]
    width: the width of the image where corner_shift is computed from
    height: the height of the image where corner_shift is computed from
    is_matrix: whether the homography is of shape [3, 3] or [8]
  Returns:
    the corner displacements of data type float32 and shape [8]
  """

  x1, y1 = apply_homography_to_point(hmg, 0, 0, is_matrix)
  x2, y2 = apply_homography_to_point(hmg, 0, height - 1, is_matrix)
  x3, y3 = apply_homography_to_point(hmg, width - 1, 0, is_matrix)
  x4, y4 = apply_homography_to_point(hmg, width - 1, height - 1, is_matrix)
  corner_shifts = tf.stack((x1, y1, x2, y2 - height + 1, x3 - width + 1, y3,
                            x4 - width + 1, y4 - height + 1), axis=-1)
  return corner_shifts


def calc_homography_distortion(width, height, corner_shift):
  """Calculates the distortion to a rectangle caused by a homography.

  Args:
    width: the width of the image where corner_shift is computed from
    height: the height of the image where corner_shift is computed from
    corner_shift: the homography parameterized using the corner displacement, of
      shape [8] and data type float32
  Returns:
    a list of four angles after applying the homography to a rectanglar image of
      size [width, height]
  """
  corners = tf.to_float(tf.stack([0, 0, 0, height-1, width-1, height-1, width-1
                                  , 0]))
  new_corners = tf.reshape(corners + corner_shift, [4, 2])
  angles = []
  for i in range(0, 4):
    a = new_corners[(i + 1) % 4] - new_corners[i]
    b = new_corners[(i + 3) % 4] - new_corners[i]
    angle = tf.tensordot(a, b, 1) / (tf.norm(a) * tf.norm(b))
    angles.append(angle)
  return angles
