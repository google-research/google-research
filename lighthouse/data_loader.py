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

"""Functions for InteriorNet data loading.

Modified version of code written by Arthur (Kefan) Chen.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import numpy as np
import tensorflow.compat.v1 as tf


def dataset_to_tensors(dataset, capacity, map_fn=None, parallelism=None):
  """Return a tensor with all elements of the dataset in one batch.

  Args:
    dataset: A Tensorflow dataset.
    capacity: (int) The size of the dataset.
    map_fn: A mapping function applied to the dataset.
    parallelism: (int) How many sequences to process in parallel

  Returns:
    A tensor containing all elements of the dataset in one batch.
  """
  with tf.name_scope(None, 'dataset_to_tensors',
                     [dataset, capacity, map_fn, parallelism]):
    if map_fn is not None:
      dataset = dataset.map(map_fn, num_parallel_calls=parallelism)
    return tf.contrib.data.get_single_element(dataset.batch(capacity))


class ViewTrip(
    collections.namedtuple('ViewTrip', [
        'scene_id', 'sequence_id', 'timestamp', 'rgb', 'pano', 'depth',
        'normal', 'mask', 'pose', 'intrinsics', 'resolution'
    ])):
  """A class for handling a trip of views."""

  def overlap_mask(self):
    intrinsics = self.intrinsics * tf.constant([[1., 1., 1.], [1., -1., 1.],
                                                [1., 1., 1.]])
    mask1_in_2, mask2_in_1 = image_overlap(self.depth[0], self.pose[0],
                                           self.depth[1], self.pose[1],
                                           intrinsics)
    masks = tf.stack([mask1_in_2, mask2_in_1], 0)
    return ViewTrip(self.scene_id, self.sequence_id, self.timestamp, self.rgb,
                    self.pano, self.depth, self.normal, masks, self.pose,
                    self.intrinsics, self.resolution)

  def reverse(self):
    """Returns the reverse of the sequence."""
    return ViewTrip(self.scene_id, self.sequence_id,
                    tf.reverse(self.timestamp, [0]), tf.reverse(self.rgb, [0]),
                    tf.reverse(self.pano, [0]), tf.reverse(self.depth, [0]),
                    tf.reverse(self.normal, [0]), tf.reverse(self.mask, [0]),
                    tf.reverse(self.pose, [0]), self.intrinsics,
                    self.resolution)

  def random_reverse(self):
    """Returns either the sequence or its reverse, with equal probability."""
    uniform_random = tf.random_uniform([], 0, 1.0)
    condition = tf.less(uniform_random, 0.5)
    return tf.cond(condition, lambda: self, lambda: self.reverse())  # pylint: disable=unnecessary-lambda

  def deterministic_reverse(self):
    """Returns either the sequence or its reverse, based on the sequence id."""
    return tf.cond(
        self.hash_in_range(2, 0, 1), lambda: self, lambda: self.reverse())  # pylint: disable=unnecessary-lambda

  def hash_in_range(self, buckets, base, limit):
    """Return true if the hashing key falls in the range [base, limit)."""
    hash_bucket = tf.string_to_hash_bucket_fast(self.scene_id, buckets)
    return tf.logical_and(
        tf.greater_equal(hash_bucket, base), tf.less(hash_bucket, limit))


class ViewSequence(
    collections.namedtuple('ViewSequence', [
        'scene_id', 'sequence_id', 'timestamp', 'rgb', 'pano', 'depth',
        'normal', 'pose', 'intrinsics', 'resolution'
    ])):
  """A class for handling a sequence of views."""

  def subsequence(self, stride):
    return ViewSequence(
        self.scene_id, self.sequence_id,
        tf.strided_slice(
            self.timestamp, [0], [self.length()], strides=[stride]),
        tf.strided_slice(self.rgb, [0], [self.length()], strides=[stride]),
        tf.strided_slice(self.pano, [0], [self.length()], strides=[stride]),
        tf.strided_slice(self.depth, [0], [self.length()], strides=[stride]),
        tf.strided_slice(self.normal, [0], [self.length()], strides=[stride]),
        tf.strided_slice(self.pose, [0], [self.length()], strides=[stride]),
        tf.strided_slice(
            self.intrinsics, [0], [self.length()], strides=[stride]),
        tf.strided_slice(
            self.resolution, [0], [self.length()], strides=[stride]))

  def random_subsequence(self, min_stride, max_stride):
    random_stride = tf.random_uniform([],
                                      minval=min_stride,
                                      maxval=max_stride,
                                      dtype=tf.int32)
    return self.subsequence(random_stride)

  def generate_trips(self, min_gap=1, max_gap=5):
    """Generate a tf Dataset of training triplets with an offset between three frames.

    Args:
      min_gap: (int) the minimum offset between two frames of a sampled triplet.
      max_gap: (int) the maximum offset between two frames of a sampled triplet.

    Returns:
      A tf.data.Dataset of ViewSequences without images, consisting of
      triplets from the input sequence separated by the given offset.
    """

    def mapper(timestamp_trips, rgb_trips, pano_trips, depth_trips,
               normal_trips, pose_trips):
      """A function mapping a data tuple to ViewTrip."""
      return ViewTrip(self.scene_id, self.sequence_id, timestamp_trips,
                      rgb_trips, pano_trips, depth_trips, normal_trips,
                      tf.zeros([1]), pose_trips, self.intrinsics[0],
                      self.resolution[0])

    with tf.control_dependencies(
        [tf.Assert(tf.less(max_gap, self.length()),
                   [max_gap, self.length()])]):
      timestamp_trips = []
      rgb_trips = []
      pano_trips = []
      depth_trips = []
      normal_trips = []
      pose_trips = []
      # generate triplets with an offset that ranges
      # from 'min_gap' to 'max_gap'.
      for stride in range(min_gap, max_gap + 1):
        inds = tf.range(stride, self.length() - stride)
        inds_jitter = tf.random.uniform(
            minval=-40,
            maxval=40,
            shape=[self.length() - 2 * stride],
            dtype=tf.int32)
        rand_inds = tf.minimum(
            tf.maximum(inds + inds_jitter, 0),
            self.length() - 1)
        timestamp = tf.stack([
            self.timestamp[:-2 * stride], self.timestamp[2 * stride:],
            self.timestamp[stride:-stride],
            tf.gather(self.timestamp, rand_inds)
        ],
                             axis=1)
        rgb = tf.stack([
            self.rgb[:-2 * stride], self.rgb[2 * stride:],
            self.rgb[stride:-stride],
            tf.gather(self.rgb, rand_inds)
        ],
                       axis=1)
        pano = tf.stack([
            self.pano[:-2 * stride], self.pano[2 * stride:],
            self.pano[stride:-stride],
            tf.gather(self.pano, rand_inds)
        ],
                        axis=1)
        depth = tf.stack([
            self.depth[:-2 * stride], self.depth[2 * stride:],
            self.depth[stride:-stride],
            tf.gather(self.depth, rand_inds)
        ],
                         axis=1)
        normal = tf.stack([
            self.normal[:-2 * stride], self.normal[2 * stride:],
            self.normal[stride:-stride],
            tf.gather(self.normal, rand_inds)
        ],
                          axis=1)
        pose = tf.stack([
            self.pose[:-2 * stride], self.pose[2 * stride:],
            self.pose[stride:-stride],
            tf.gather(self.pose, rand_inds)
        ],
                        axis=1)
        timestamp_trips.append(timestamp)
        rgb_trips.append(rgb)
        pano_trips.append(pano)
        depth_trips.append(depth)
        normal_trips.append(normal)
        pose_trips.append(pose)

      timestamp_trips = tf.concat(timestamp_trips, 0)
      rgb_trips = tf.concat(rgb_trips, 0)
      pano_trips = tf.concat(pano_trips, 0)
      depth_trips = tf.concat(depth_trips, 0)
      normal_trips = tf.concat(normal_trips, 0)
      pose_trips = tf.concat(pose_trips, 0)
      dataset = tf.data.Dataset.from_tensor_slices(
          (timestamp_trips, rgb_trips, pano_trips, depth_trips, normal_trips,
           pose_trips))
      return dataset.map(mapper)

  def length(self):
    """Returns the length of the sequence."""
    return tf.shape(self.timestamp)[0]

  def reverse(self):
    """Returns the reverse of the sequence."""
    return ViewSequence(self.scene_id, self.sequence_id,
                        tf.reverse(self.timestamp, [0]),
                        tf.reverse(self.rgb, [0]), tf.reverse(self.pano, [0]),
                        tf.reverse(self.depth,
                                   [0]), tf.reverse(self.normal, [0]),
                        tf.reverse(self.pose, [0]),
                        tf.reverse(self.intrinsics, [0]),
                        tf.reverse(self.resolution, [0]))

  def random_reverse(self):
    """Returns either the sequence or its reverse, with equal probability."""
    uniform_random = tf.random_uniform([], 0, 1.0)
    condition = tf.less(uniform_random, 0.5)
    return tf.cond(condition, lambda: self, lambda: self.reverse())  # pylint: disable=unnecessary-lambda

  def deterministic_reverse(self):
    """Returns either the sequence or its reverse, based on the sequence id."""
    return tf.cond(
        self.hash_in_range(2, 0, 1), lambda: self, lambda: self.reverse())  # pylint: disable=unnecessary-lambda

  def hash_in_range(self, buckets, base, limit):
    """Return true if the hashing key falls in the range [base, limit)."""
    hash_bucket = tf.string_to_hash_bucket_fast(self.scene_id, buckets)
    return tf.logical_and(
        tf.greater_equal(hash_bucket, base), tf.less(hash_bucket, limit))


def check_cam_coherence(path):
  """Check the coherence of a camera path."""
  cam_gt = path + 'cam0_gt.visim'
  cam_render = path + 'cam0.render'
  lines = tf.string_split([tf.read_file(cam_render)], '\n').values
  lines = lines[3:]
  lines = tf.strided_slice(lines, [0], [lines.shape_as_list()[0]], [2])
  fields = tf.reshape(tf.string_split(lines, ' ').values, [-1, 10])
  timestamp_from_render, numbers = tf.split(fields, [1, 9], -1)
  numbers = tf.strings.to_number(numbers)
  eye, lookat, up = tf.split(numbers, [3, 3, 3], -1)
  up_vector = tf.nn.l2_normalize(up - eye)
  lookat_vector = tf.nn.l2_normalize(lookat - eye)
  rotation_from_lookat = lookat_matrix(up_vector, lookat_vector)

  lines = tf.string_split([tf.read_file(cam_gt)], '\n').values
  lines = lines[1:]
  fields = tf.reshape(tf.string_split(lines, ',').values, [-1, 8])
  timestamp_from_gt, numbers = tf.split(fields, [1, 7], -1)
  numbers = tf.strings.to_number(numbers)
  position, quaternion = tf.split(numbers, [3, 4], -1)
  rotation_from_quaternion = from_quaternion(quaternion)

  assert tf.reduce_all(tf.equal(timestamp_from_render, timestamp_from_gt))
  assert tf.reduce_all(tf.equal(eye, position))
  so3_diff = (tf.trace(
      tf.matmul(
          rotation_from_lookat, rotation_from_quaternion, transpose_a=True)) -
              1) / 2
  tf.assert_near(so3_diff, tf.ones_like(so3_diff))


def lookat_matrix(up, lookat_direction):
  """Construct a matrix that "looks at" a direction."""
  # lookat_direction [Batch, 3]
  # return [Batch, 3, 3] colomn major cam2world lookat matrix.
  # z is the forward direction. x is the right vector. y is the up vector.
  # Stack x, y, z vectors by colomn to get the lookat matrix.
  # [[x.x y.x z.x]
  #  [x.y y.y z.y]
  #  [x.z y.z z.z]]
  z = tf.linalg.l2_normalize(-lookat_direction, axis=-1)
  x = tf.linalg.l2_normalize(tf.cross(up, z), axis=-1)
  y = tf.cross(z, x)
  lookat = tf.stack([x, y, z], axis=-1)
  return lookat


def load_sequence(sequence_dir, data_dir, parallelism=10):
  """Load a sequence."""
  n_timestamp = 1000
  v = tf.string_split([sequence_dir], '/').values
  scene_id, sequence_id = v[-2], v[-1]
  camera_dir = data_dir + 'GroundTruth_HD1-HD6/' + scene_id + '/'
  trajectory_name = 'velocity_angular' + tf.strings.substr(v[-1], -4, -4) + '/'
  camera_dir = camera_dir + trajectory_name
  camera_timestamp_path = camera_dir + 'cam0.timestamp'
  timestamp, img_name = read_timestamp(camera_timestamp_path)

  rgb_paths = sequence_dir + '/cam0/data/' + img_name
  pano_paths = sequence_dir + '/cam0_pano/data/' + img_name
  depth_paths = sequence_dir + '/depth0/data/' + img_name
  normal_paths = sequence_dir + '/normal0/data/' + img_name

  camera_parameters_path = camera_dir + 'cam0.ccam'
  pose_matrix, intrinsic_matrix, resolution = read_camera_parameters(
      camera_parameters_path, n_timestamp, parallel_camera_process=parallelism)
  return ViewSequence(scene_id, sequence_id, timestamp, rgb_paths, pano_paths,
                      depth_paths, normal_paths, pose_matrix, intrinsic_matrix,
                      resolution)


def read_timestamp(path):
  """Read a path's timestamp."""
  # parse the lines
  lines = tf.string_split([tf.read_file(path)], '\n').values
  # ignore the header
  lines = lines[1:]
  # parse the columns
  fields = tf.reshape(tf.string_split(lines, ',').values, [-1, 2])
  timestamp, img_name = tf.split(fields, [1, 1], -1)
  timestamp = tf.squeeze(timestamp, -1)
  img_name = tf.squeeze(img_name, -1)
  return timestamp, img_name


def read_camera_parameters(path, n_timestamp, parallel_camera_process=10):
  """Read a camera's parameters."""
  # parse the lines
  lines = tf.string_split([tf.read_file(path)], '\n').values
  # ignore the header
  lines = lines[6:]
  # parse the columns
  fields = tf.reshape(tf.string_split(lines, ' ').values, [-1, 15])
  # convert string to float32
  fields = tf.strings.to_number(fields)
  # <camera info: f, cx, cy, dist.coeff[0],dist.coeff[1],dist.coeff[2]>
  # <orientation: w,x,y,z> <position: x,y,z> <image resolution: width, height>
  camera_info, orientation, position, resolution = tf.split(
      fields, [6, 4, 3, 2], -1)
  camera_ds = tf.data.Dataset.from_tensor_slices(
      (camera_info, orientation, position, resolution))

  def process_camera_parameters(camera_info, orientation, position, resolution):
    # convert quaternion to 3x3 matrix
    rotation_matrix = from_quaternion(orientation)
    # 3x4 pose matrix [R_3x3 |t_3x1]
    pose_matrix = tf.concat([rotation_matrix, tf.expand_dims(position, -1)], -1)
    intrinsic_matrix = build_intrinsic_matrix(camera_info[0], camera_info[1],
                                              camera_info[2])
    return (pose_matrix, intrinsic_matrix, resolution)

  return dataset_to_tensors(
      camera_ds,
      capacity=n_timestamp,
      map_fn=process_camera_parameters,
      parallelism=parallel_camera_process)


def build_intrinsic_matrix(f, cx, cy):
  # camera instrinsics [[f 0 cx]
  #                     [0 f cy]
  #                     [0 0  1]] (f is focal length in pixels.)
  return tf.stack(
      [tf.stack([f, 0., cx]),
       tf.stack([0., f, cy]),
       tf.constant([0., 0., 1.])])


def load_image_data(trip):
  """Load empty ViewTrip with images."""

  def load_single_image(filename, shape):
    """Load a single image given the filename."""
    image = tf.image.decode_png(tf.read_file(filename), 3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image.set_shape(shape)
    return image

  def load_depth(filename, shape):
    """Load the 16-bit png depth map in milimeters given the filename."""
    depth = tf.image.decode_png(tf.read_file(filename), 3, tf.dtypes.uint16)
    depth = tf.cast(depth, tf.float32) / 1000
    depth.set_shape(shape)
    return depth

  def load_surface_normal(filename, shape):
    """Load the surface normal given the filename."""
    normal = tf.image.decode_png(tf.read_file(filename), 3, tf.dtypes.uint16)
    normal = 2 * tf.cast(normal, tf.float32) / (2**16 - 1) - 1
    normal.set_shape(shape)
    return normal

  trip_length = 4  # triplet plus more distant camera for pano supervision
  rgb = dataset_to_tensors(
      tf.data.Dataset.from_tensor_slices(trip.rgb),
      trip_length,
      lambda filename: load_single_image(filename, [480, 640, 3]),
      parallelism=4)
  pano = dataset_to_tensors(
      tf.data.Dataset.from_tensor_slices(trip.pano),
      trip_length,
      lambda filename: load_single_image(filename, [1500, 3000, 3]),
      parallelism=4)
  depth = dataset_to_tensors(
      tf.data.Dataset.from_tensor_slices(trip.depth),
      trip_length,
      lambda filename: load_depth(filename, [480, 640, 3]),
      parallelism=4)
  # depth: [N, height, width, 3] all channels are identical.
  depth = depth[:, :, :, :1]
  normal = dataset_to_tensors(
      tf.data.Dataset.from_tensor_slices(trip.normal),
      trip_length,
      lambda filename: load_surface_normal(filename, [480, 640, 3]),
      parallelism=4)

  return ViewTrip(trip.scene_id, trip.sequence_id, trip.timestamp, rgb, pano,
                  depth, normal, trip.mask, trip.pose, trip.intrinsics,
                  trip.resolution)


def small_translation_condition(trip, translation_threshold):
  # trip.pose: [N, 3, 4]
  positions = trip.pose[:, :, -1]
  t_norm = tf.norm(positions[0] - positions[1], axis=-1)
  return tf.greater(t_norm, translation_threshold)


def too_close_condition(trip, depth_threshold=0.1):
  depths = trip.depth[:3, :, :, 0]
  depthmax = tf.reduce_max(depths)
  depths = tf.where(
      tf.equal(depths, 0.0), depthmax * tf.ones_like(depths), depths)
  return tf.greater(tf.reduce_min(depths), depth_threshold)


def pano_forwards_condition(trip):
  """Checks if a pano is in a forward condition."""
  ref_pose = trip.pose[1, :, :]
  pano_pose = trip.pose[3, :, :]
  ref_twds = -1.0 * ref_pose[:, 2]

  # make sure max_depth>forward motion>median_depth
  t_vec = pano_pose[:, 3] - ref_pose[:, 3]
  ref_depth = trip.depth[1, :, :, 0]
  ref_depth = tf.where(
      tf.equal(ref_depth, 0.0),
      tf.reduce_max(ref_depth) * tf.ones_like(ref_depth), ref_depth)
  max_depth = tf.reduce_max(ref_depth)
  median_depth = tf.contrib.distributions.percentile(ref_depth, 0.5)

  min_depth_cond = tf.greater(tf.reduce_sum(ref_twds * t_vec), median_depth)
  max_depth_cond = tf.less(tf.reduce_sum(ref_twds * t_vec), max_depth)

  return tf.logical_and(min_depth_cond, max_depth_cond)


def dark_trip_condition(trip, threshold=0.1):
  cond = tf.math.greater(image_brightness(trip.rgb), threshold)
  return tf.math.reduce_all(cond)


def image_brightness(image):
  r, g, b = tf.split(image, [1, 1, 1], -1)
  brightness = tf.sqrt(0.299 * (r**2) + 0.587 * (g**2) + 0.114 * (b**2))
  avg_brightness = tf.reduce_mean(brightness, axis=[1, 2, 3])
  return avg_brightness


def filter_random_lighting(sequence_dir):
  sequence_name = tf.string_split([sequence_dir], '/').values[-1]
  lighting = tf.substr(sequence_name, 0, 6)
  return tf.not_equal(lighting, 'random')


def filter_seq_length(sequence_dir):
  img_files = tf.data.Dataset.list_files(sequence_dir + '/cam0/data/*.png')
  pano_files = tf.data.Dataset.list_files(sequence_dir +
                                          '/cam0_pano/data/*.png')
  num_imgs = tf.data.experimental.cardinality(img_files)
  num_panos = tf.data.experimental.cardinality(pano_files)
  return tf.logical_and(tf.equal(num_imgs, 1000), tf.equal(num_panos, 1000))


def prepare_training_set(
    dataset,
    min_gap,
    max_gap,
    min_stride,
    max_stride,
    batch_size,
    epochs,
    min_overlap,  # pylint: disable=unused-argument
    max_overlap,  # pylint: disable=unused-argument
    translation_threshold,
    luminence_threshold,
    depth_threshold,
    parallel_image_reads,
    prefetch_buffer,
    filter_envmap=True):
  """Prepare the training set."""
  dataset = dataset.map(
      lambda sequence: sequence.random_subsequence(min_stride, max_stride))
  dataset = dataset.flat_map(
      lambda sequence: sequence.generate_trips(min_gap, max_gap))
  dataset = dataset.shuffle(1000000).repeat(epochs)
  # filter small translations
  dataset = dataset.filter(
      lambda trip: small_translation_condition(trip, translation_threshold))
  # load images
  dataset = dataset.map(load_image_data, parallel_image_reads).apply(
      tf.data.experimental.ignore_errors())
  # filter dark pairs
  dataset = dataset.filter(
      lambda trip: dark_trip_condition(trip, luminence_threshold))
  # filter out target panos that move backwards instead of forwards
  if filter_envmap:
    dataset = dataset.filter(pano_forwards_condition)
  # filter out examples that are too close to scene
  dataset = dataset.filter(
      lambda trip: too_close_condition(trip, depth_threshold))
  dataset = dataset.batch(
      batch_size, drop_remainder=True).prefetch(prefetch_buffer)
  return dataset


def prepare_eval_set(
    dataset,
    min_gap,
    max_gap,
    min_stride,
    max_stride,
    batch_size,
    min_overlap,  # pylint: disable=unused-argument
    max_overlap,  # pylint: disable=unused-argument
    translation_threshold,
    luminence_threshold,
    depth_threshold,
    parallel_image_reads,
    prefetch_buffer):
  """Prepare the eval set."""

  stride = (min_stride + max_stride) // 2
  dataset = dataset.map(lambda sequence: sequence.subsequence(stride))
  dataset = dataset.flat_map(
      lambda sequence: sequence.generate_trips(min_gap, max_gap))
  # filter small translations
  dataset = dataset.filter(
      lambda trip: small_translation_condition(trip, translation_threshold))
  # load images
  dataset = dataset.map(load_image_data, parallel_image_reads).apply(
      tf.data.experimental.ignore_errors())
  # filter dark trips
  dataset = dataset.filter(
      lambda trip: dark_trip_condition(trip, luminence_threshold))
  # filter target panos that move backwards instead of forwards
  dataset = dataset.filter(pano_forwards_condition)
  # filter out examples that are too close to scene
  dataset = dataset.filter(
      lambda trip: too_close_condition(trip, depth_threshold))

  dataset = dataset.batch(
      batch_size, drop_remainder=True).prefetch(prefetch_buffer)
  return dataset


def world_to_camera_projection(p_world, intrinsics, world_to_camera):
  """Project world coordinates to camera coordinates."""
  shape = p_world.shape.as_list()
  height, width = shape[0], shape[1]
  p_world_homogeneous = tf.concat([p_world, tf.ones([height, width, 1])], -1)
  intrinsics = tf.tile(intrinsics[tf.newaxis, tf.newaxis, :],
                       [height, width, 1, 1])
  world_to_camera = tf.tile(world_to_camera[tf.newaxis, tf.newaxis, :],
                            [height, width, 1, 1])
  p_camera = tf.squeeze(
      tf.matmul(world_to_camera, tf.expand_dims(p_world_homogeneous, -1)), -1)
  p_camera_z = p_camera * tf.constant([1., 1., -1.], shape=[1, 1, 3])
  p_image = tf.squeeze(
      tf.matmul(intrinsics, tf.expand_dims(p_camera_z, -1)), -1)
  return p_image[:, :, :2] / (p_image[:, :, -1:] + 1e-8), p_image[:, :, -1]


def camera_to_world_projection(depth, intrinsics, camera_to_world):
  """Project camera coordinates to world coordinates."""
  # p_pixel: batch, w, h, 3 principal_point, fov 2-d list
  # r: batch, 3, 3 camera to world rotation
  # t: batch, 3 camera to world translation, depth: batch, w, h, 1
  shape = depth.shape.as_list()
  height, width = shape[0], shape[1]
  xx, yy = tf.meshgrid(
      tf.lin_space(0., width - 1., width), tf.lin_space(0., height - 1.,
                                                        height))
  p_pixel = tf.stack([xx, yy], axis=-1)
  p_pixel_homogeneous = tf.concat([p_pixel, tf.ones([height, width, 1])], -1)

  camera_to_world = tf.tile(camera_to_world[tf.newaxis, tf.newaxis, :],
                            [height, width, 1, 1])
  intrinsics = tf.tile(intrinsics[tf.newaxis, tf.newaxis, :],
                       [height, width, 1, 1])
  # Convert pixels coordinates (u, v, 1) to camera coordinates (x_c, y_c, f)
  # on the image plane.
  p_image = tf.squeeze(
      tf.matmul(
          tf.matrix_inverse(intrinsics), tf.expand_dims(p_pixel_homogeneous,
                                                        -1)), -1)

  lookat_axis = tf.tile(
      tf.constant([0., 0., 1.], shape=[1, 1, 3]), [height, width, 1])
  z = depth * tf.reduce_sum(
      tf.math.l2_normalize(p_image, axis=-1) * lookat_axis,
      axis=-1,
      keepdims=True)
  p_camera = z * p_image
  # convert from OpenCV convention to OpenGL
  p_camera = p_camera * tf.constant([1., 1., -1.], shape=[1, 1, 3])
  p_camera_homogeneous = tf.concat(
      [p_camera, tf.ones(shape=[height, width, 1])], -1)
  # Convert camera coordinates to world coordinates.
  p_world = tf.squeeze(
      tf.matmul(camera_to_world, tf.expand_dims(p_camera_homogeneous, -1)), -1)
  return p_world


def image_overlap(depth1, pose1_c2w, depth2, pose2_c2w, intrinsics):
  """Determines the overlap of two images."""

  pose1_w2c = tf.matrix_inverse(
      tf.concat([pose1_c2w, tf.constant([[0., 0., 0., 1.]])], 0))[:3]
  pose2_w2c = tf.matrix_inverse(
      tf.concat([pose2_c2w, tf.constant([[0., 0., 0., 1.]])], 0))[:3]

  p_world1 = camera_to_world_projection(depth1, intrinsics, pose1_c2w)
  p_image1_in_2, z1_c2 = world_to_camera_projection(p_world1, intrinsics,
                                                    pose2_w2c)

  p_world2 = camera_to_world_projection(depth2, intrinsics, pose2_c2w)
  p_image2_in_1, z2_c1 = world_to_camera_projection(p_world2, intrinsics,
                                                    pose1_w2c)

  shape = depth1.shape.as_list()
  height, width = shape[0], shape[1]
  height = tf.cast(height, tf.float32)
  width = tf.cast(width, tf.float32)
  mask_h2_in_1 = tf.logical_and(
      tf.less_equal(p_image2_in_1[:, :, 1], height),
      tf.greater_equal(p_image2_in_1[:, :, 1], 0.))
  mask_w2_in_1 = tf.logical_and(
      tf.less_equal(p_image2_in_1[:, :, 0], width),
      tf.greater_equal(p_image2_in_1[:, :, 0], 0.))
  mask2_in_1 = tf.logical_and(
      tf.logical_and(mask_h2_in_1, mask_w2_in_1), z2_c1 > 0)

  mask_h1_in_2 = tf.logical_and(
      tf.less_equal(p_image1_in_2[:, :, 1], height),
      tf.greater_equal(p_image1_in_2[:, :, 1], 0.))
  mask_w1_in_2 = tf.logical_and(
      tf.less_equal(p_image1_in_2[:, :, 0], width),
      tf.greater_equal(p_image1_in_2[:, :, 0], 0.))
  mask1_in_2 = tf.logical_and(
      tf.logical_and(mask_h1_in_2, mask_w1_in_2), z1_c2 > 0)

  return mask1_in_2, mask2_in_1


def images_have_overlap(trip, min_ratio, max_ratio):
  """Checks if images have any overlap."""
  # the y axis in image coordinates increases from top to bottom.
  mask1_in_2, mask2_in_1 = trip.mask[0], trip.mask[1]
  shape = mask1_in_2.shape.as_list()
  height, width = shape[0], shape[1]
  ratio1 = tf.reduce_sum(tf.cast(mask1_in_2, tf.float32)) / (height * width)
  ratio2 = tf.reduce_sum(tf.cast(mask2_in_1, tf.float32)) / (height * width)
  cond1 = tf.logical_and(
      tf.less_equal(ratio1, max_ratio), tf.less_equal(ratio2, max_ratio))
  cond2 = tf.logical_and(
      tf.greater_equal(ratio1, min_ratio), tf.greater_equal(ratio2, min_ratio))
  return tf.logical_and(cond1, cond2)


def data_loader(parent_dir='',
                dataset_list=('HD1', 'HD2', 'HD3', 'HD4', 'HD5', 'HD6'),
                min_gap=1,
                max_gap=4,
                min_stride=1,
                max_stride=2,
                epochs=-1,
                batch_size=1,
                random_lighting=False,
                luminence_threshold=0.1,
                depth_threshold=0.1,
                min_overlap=0.3,
                max_overlap=1.0,
                min_translation=0.05,
                validation_percentage=0,
                test_percentage=10,
                parallelism=20,
                parallel_image_reads=100,
                prefetch_buffer=20,
                filter_envmap=True):
  """Loads data."""

  datasets = collections.namedtuple('datasets',
                                    ['training', 'validation', 'test'])

  test_start = 100 - test_percentage
  val_start = test_start - validation_percentage

  data_dir = os.path.join(parent_dir, dataset_list[0])
  scenes = tf.data.Dataset.list_files(os.path.join(data_dir, '*'))
  for dataset in dataset_list[1:]:
    data_dir = os.path.join(parent_dir, dataset)
    scenes = scenes.concatenate(
        tf.data.Dataset.list_files(os.path.join(data_dir, '*')))

  sequences = scenes.flat_map(
      lambda scene_dir: tf.data.Dataset.list_files(scene_dir + '/*')).apply(
          tf.data.experimental.ignore_errors())
  if not random_lighting:
    sequences = sequences.filter(filter_random_lighting)

  sequences = sequences.filter(filter_seq_length).apply(
      tf.data.experimental.ignore_errors())

  sequences = sequences.map(
      lambda sequence_dir: load_sequence(sequence_dir, parent_dir, parallelism),
      num_parallel_calls=parallelism)

  training = sequences.filter(
      lambda sequence: sequence.hash_in_range(100, 0, val_start))
  validation = sequences.filter(
      lambda sequence: sequence.hash_in_range(100, val_start, test_start))
  test = sequences.filter(
      lambda sequence: sequence.hash_in_range(100, test_start, 100))

  training = prepare_training_set(training, min_gap, max_gap, min_stride,
                                  max_stride, batch_size, epochs, min_overlap,
                                  max_overlap, min_translation,
                                  luminence_threshold, depth_threshold,
                                  parallel_image_reads, prefetch_buffer,
                                  filter_envmap)
  validation = prepare_eval_set(validation, min_gap, max_gap, min_stride,
                                max_stride, batch_size, min_overlap,
                                max_overlap, min_translation,
                                luminence_threshold, depth_threshold,
                                parallel_image_reads, prefetch_buffer)
  test = prepare_eval_set(test, min_gap, max_gap, min_stride, max_stride,
                          batch_size, min_overlap, max_overlap, min_translation,
                          luminence_threshold, depth_threshold,
                          parallel_image_reads, prefetch_buffer)

  return datasets(training, validation, test)


def relative_pose(element):
  r1_c2w, t1_world = tf.split(element.pose[:, 0], [3, 1], -1)
  r2_c2w, t2_world = tf.split(element.pose[:, 1], [3, 1], -1)
  relative_rotation_c2toc1 = tf.matmul(r1_c2w, r2_c2w, transpose_a=True)
  # [batch, 3, 1]
  translation_c1 = tf.matmul(r1_c2w, t2_world - t1_world, transpose_a=True)
  # [batch, 3]
  translation_c1 = tf.math.l2_normalize(tf.squeeze(translation_c1, -1), axis=-1)
  return relative_rotation_c2toc1, translation_c1


def quaternion_to_matrix(quaternion):
  quaternion = tf.nn.l2_normalize(quaternion, axis=-1)
  w, x, y, z = tf.unstack(quaternion, axis=-1)
  return tf.stack([
      tf.stack([
          1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w
      ], -1),
      tf.stack([
          2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w
      ], -1),
      tf.stack([
          2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2
      ], -1)
  ], 1)


def from_quaternion(quaternion):
  """Convert from a quaternion."""
  quaternion = tf.convert_to_tensor(value=quaternion)
  w, x, y, z = tf.unstack(quaternion, axis=-1)
  tx = 2.0 * x
  ty = 2.0 * y
  tz = 2.0 * z
  twx = tx * w
  twy = ty * w
  twz = tz * w
  txx = tx * x
  txy = ty * x
  txz = tz * x
  tyy = ty * y
  tyz = tz * y
  tzz = tz * z
  matrix = tf.stack((1.0 - (tyy + tzz), txy - twz, txz + twy,
                     txy + twz, 1.0 - (txx + tzz), tyz - twx,
                     txz - twy, tyz + twx, 1.0 - (txx + tyy)),
                    axis=-1)  # pyformat: disable
  output_shape = tf.concat((tf.shape(input=quaternion)[:-1], (3, 3)), axis=-1)
  return tf.reshape(matrix, shape=output_shape)


def format_pose(pose_c2w, do_flip=False):
  flip_val = -1.0 if do_flip else 1.0
  pose_z_flip = tf.concat([
      pose_c2w[:, :3, 0:1], pose_c2w[:, :3, 1:2],
      flip_val * pose_c2w[:, :3, 2:3], pose_c2w[:, :3, 3:]
  ],
                          axis=2)
  filler = np.array([0.0, 0.0, 0.0, 1.0])[tf.newaxis, tf.newaxis, :]
  return tf.concat([pose_z_flip, filler], axis=1)


def format_inputs(s, height, width, env_height, env_width):
  """Package an example from the dataset iterator."""

  batch = {}

  num_imgs = 3
  randrange = tf.random.shuffle(tf.range(num_imgs))
  batch['ordering'] = randrange

  batch['ref_image'] = tf.image.resize_area(
      s.rgb[:, randrange[0], :, :, :], size=[height, width])
  batch['ref_pose'] = format_pose(s.pose[:, randrange[0], :, :], do_flip=True)

  ref_depths = s.depth[:, randrange[0], :, :, :]
  ref_depths = tf.where(
      tf.equal(ref_depths, 0.0),
      tf.reduce_max(ref_depths) * tf.ones_like(ref_depths), ref_depths)
  ref_depths = tf.nn.pool(
      ref_depths, window_shape=[3, 3], pooling_type='MAX', padding='SAME')
  batch['ref_depth'] = tf.image.resize_area(
      ref_depths, size=[height, width])[Ellipsis, 0]

  batch['tgt_image'] = tf.image.resize_area(
      s.rgb[:, randrange[1], :, :, :], size=[height, width])
  batch['tgt_pose'] = format_pose(s.pose[:, randrange[1], :, :], do_flip=True)

  src_images = []
  src_poses = []
  for i in range(2, num_imgs):
    src_images.append(
        tf.image.resize_area(
            s.rgb[:, randrange[i], :, :, :], size=[height, width]))
    src_poses.append(format_pose(s.pose[:, randrange[i], :, :], do_flip=True))
  src_images = tf.concat(src_images, axis=3)
  src_poses = tf.stack(src_poses, axis=3)
  batch['src_images'] = src_images
  batch['src_poses'] = src_poses

  intrinsics = tf.cast(s.intrinsics, tf.float32)
  ds = [s.rgb.shape[2] // height, s.rgb.shape[3] // height]
  intrinsics = tf.concat([
      intrinsics[:, 0:1, :] / tf.to_float(ds[1]),
      intrinsics[:, 1:2, :] / tf.to_float(ds[0]), intrinsics[:, 2:3, :]
  ],
                         axis=1)
  batch['intrinsics'] = intrinsics

  env_img = tf.image.resize_area(
      s.pano[:, num_imgs, :, :, :], size=[env_height, env_width])
  batch['env_image'] = env_img
  env_pose = format_pose(s.pose[:, num_imgs, :, :], do_flip=True)
  batch['env_pose'] = env_pose

  return batch
