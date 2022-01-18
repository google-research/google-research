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

"""Generate the wide baseline stereo image dataset from the Matterport3D.

We generate the data by randomly sample different perspective views from
panoramic images in Matterport3D to create a large scale dataset with a large
varieties of motion. The dataset contains a pair of perspective images labeled
with the relative rotation from camera 2 to camera 1, and the relative
translation direction in the frame of camera 1.

Matterport3D: https://niessner.github.io/Matterport/
  https://arxiv.org/pdf/1709.06158.pdf
"""
import collections
import math

import numpy as np
from pano_utils import math_utils
from pano_utils import transformation
import tensorflow.compat.v1 as tf


def world_to_image_projection(p_world, intrinsics, pose_w2c):
  """Project points in the world frame to the image plane.

  Args:
    p_world: [HEIGHT, WIDTH, 3] points in the world's coordinate frame.
    intrinsics: [3, 3] camera's intrinsic matrix.
    pose_w2c: [3, 4] camera pose matrix (world to camera).

  Returns:
    [HEIGHT, WIDTH, 2] points in the image coordinate.
    [HEIGHT, WIDTH, 1] the z depth.
  """
  shape = p_world.shape.as_list()
  height, width = shape[0], shape[1]
  p_world_homogeneous = tf.concat([p_world, tf.ones([height, width, 1])], -1)
  p_camera = tf.squeeze(
      tf.matmul(pose_w2c[tf.newaxis, tf.newaxis, :],
                tf.expand_dims(p_world_homogeneous, -1)), -1)
  p_camera = p_camera*tf.constant([1., 1., -1.], shape=[1, 1, 3])
  p_image = tf.squeeze(tf.matmul(intrinsics[tf.newaxis, tf.newaxis, :],
                                 tf.expand_dims(p_camera, -1)), -1)
  z = p_image[:, :, -1:]
  return tf.math.divide_no_nan(p_image[:, :, :2], z), z


def image_to_world_projection(depth, intrinsics, pose_c2w):
  """Project points on the image to the world frame.

  Args:
    depth: [HEIGHT, WIDTH, 1] the depth map contains the radial distance from
      the camera eye to each point corresponding to each pixel.
    intrinsics: [3, 3] camera's intrinsic matrix.
    pose_c2w: [3, 4] camera pose matrix (camera to world).

  Returns:
    [HEIGHT, WIDTH, 3] points in the world's coordinate frame.
  """
  shape = depth.shape.as_list()
  height, width = shape[0], shape[1]
  xx, yy = tf.meshgrid(tf.lin_space(0., width-1., width),
                       tf.lin_space(0., height-1., height))
  p_pixel_homogeneous = tf.concat([tf.stack([xx, yy], axis=-1),
                                   tf.ones([height, width, 1])], -1)

  p_image = tf.squeeze(tf.matmul(
      tf.matrix_inverse(intrinsics[tf.newaxis, tf.newaxis, :]),
      tf.expand_dims(p_pixel_homogeneous, -1)), -1)

  z = depth*tf.reduce_sum(
      tf.math.l2_normalize(p_image, axis=-1)*tf.constant([[[0., 0., 1.]]]),
      axis=-1,
      keepdims=True)
  p_camera = z*p_image
  # convert to OpenGL coordinate system.
  p_camera = p_camera*tf.constant([1., 1., -1.], shape=[1, 1, 3])
  p_camera_homogeneous = tf.concat(
      [p_camera, tf.ones(shape=[height, width, 1])], -1)
  # Convert camera coordinates to world coordinates.
  p_world = tf.squeeze(
      tf.matmul(pose_c2w[tf.newaxis, tf.newaxis, :],
                tf.expand_dims(p_camera_homogeneous, -1)), -1)
  return p_world


def overlap_mask(depth1,
                 pose1_c2w,
                 depth2,
                 pose2_c2w,
                 intrinsics):
  """Compute the overlap masks of two views using triangulation.

  The masks have the same shape of the input images. A pixel value is true if it
  can be seen by both cameras.

  Args:
    depth1: [HEIGHT, WIDTH, 1] the depth map of the first view.
    pose1_c2w: [3, 4] camera pose matrix (camera to world) of the first view.
      pose1_c2w[:, :3] is the rotation and pose1_c2w[:, -1] is the translation.
    depth2: [HEIGHT, WIDTH, 1] the depth map of the second view.
    pose2_c2w: [3, 4] camera pose matrix (camera to world) of the second view.
      pose1_c2w[:, :3] is the rotation and pose1_c2w[:, -1] is the translation.
    intrinsics: [3, 3] camera's intrinsic matrix.

  Returns:
    [HEIGHT, WIDTH] two overlap masks of the two inputs respectively.
  """

  pose1_w2c = tf.matrix_inverse(
      tf.concat([pose1_c2w, tf.constant([[0., 0., 0., 1.]])], 0))[:3]
  pose2_w2c = tf.matrix_inverse(
      tf.concat([pose2_c2w, tf.constant([[0., 0., 0., 1.]])], 0))[:3]

  p_world1 = image_to_world_projection(depth1, intrinsics, pose1_c2w)
  p_image1_in_2, z1_c2 = world_to_image_projection(
      p_world1, intrinsics, pose2_w2c)

  p_world2 = image_to_world_projection(depth2, intrinsics, pose2_c2w)
  p_image2_in_1, z2_c1 = world_to_image_projection(
      p_world2, intrinsics, pose1_w2c)

  shape = depth1.shape.as_list()
  height, width = shape[0], shape[1]
  height = tf.cast(height, tf.float32)
  width = tf.cast(width, tf.float32)
  # Error tolerance.
  eps = 1e-4
  # check the object seen by camera 2 is also projected to camera 1's image
  # plane and in front of the camera 1.
  mask_h2_in_1 = tf.logical_and(
      tf.less_equal(p_image2_in_1[:, :, 1], height+eps),
      tf.greater_equal(p_image2_in_1[:, :, 1], 0.-eps))
  mask_w2_in_1 = tf.logical_and(
      tf.less_equal(p_image2_in_1[:, :, 0], width+eps),
      tf.greater_equal(p_image2_in_1[:, :, 0], 0.-eps))
  # check the projected points are within the image boundaries and in front of
  # the camera.
  mask2_in_1 = tf.logical_and(
      tf.logical_and(mask_h2_in_1, mask_w2_in_1), tf.squeeze(z2_c1, -1) > 0)

  # check the object seen by camera 1 is also projected to camera 2's image
  # plane and in front of the camera 2.
  mask_h1_in_2 = tf.logical_and(
      tf.less_equal(p_image1_in_2[:, :, 1], height+eps),
      tf.greater_equal(p_image1_in_2[:, :, 1], 0.-eps))
  mask_w1_in_2 = tf.logical_and(
      tf.less_equal(p_image1_in_2[:, :, 0], width+eps),
      tf.greater_equal(p_image1_in_2[:, :, 0], 0.-eps))
  # check the projected points are within the image boundaries and in front of
  # the camera.
  mask1_in_2 = tf.logical_and(
      tf.logical_and(mask_h1_in_2, mask_w1_in_2), tf.squeeze(z1_c2, -1) > 0)

  return mask1_in_2, mask2_in_1


def overlap_ratio(mask1, mask2):
  """Check if the overlapping ratio of the input is within given limits.

  The overlap ratio is measured by the minimum of the ratio between the area
  seen by both cameras and the image size. This function returns a ViewPair
  object containing the perspective images, the masks that shows the common area
  seen by both cameras, the camera's field of view (FoV), the relative rotation
  from camera 2 to camera 1, and the relative translation direction in the frame
  of camera 1.

  Args:
    mask1: [HEIGHT, WIDTH] overlapping mask.
    mask2: [HEIGHT, WIDTH] overlapping mask.

  Returns:
    A tf.float32 tensor.
  """
  shape = mask1.shape.as_list()
  height, width = shape[0], shape[1]
  return tf.min(tf.reduce_sum(tf.cast(mask1, tf.float32))/(height * width),
                tf.reduce_sum(tf.cast(mask2, tf.float32))/(height * width))


# This is written for Matterport3D's directory structure.
def generate_from_meta(meta_data_path,
                       pano_data_dir,
                       pano_height=1024,
                       pano_width=2048,
                       output_height=512,
                       output_width=512):
  """Generate the stereo image dataset from Matterport3D using the meta data.

  Example call:
    ds = generate_from_meta(
      meta_data_path='matterport3d/saved_meta/R90_fov90/test_meta/',
      pano_data_dir='matterport3d/pano/')

  Args:
    meta_data_path: (string) the path to the meta data files.
    pano_data_dir: (string) the path to the panorama images of the Matterport3D.
    pano_height: (int) the height dimension of the panorama images.
    pano_width: (int) the width dimension of the panorama images.
    output_height: (int) the height dimension of the output perspective images.
    output_width: (int) the width dimension of the output perspective images.

  Returns:
    Tensorflow Dataset.
  """

  def load_text(file_path, n_lines=200):
    """Load text data from a file."""
    return tf.data.Dataset.from_tensor_slices(
        tf.data.experimental.get_single_element(
            tf.data.TextLineDataset(file_path).batch(n_lines)))

  def load_single_image(filename):
    """Load a single image given the filename."""
    image = tf.image.decode_jpeg(tf.read_file(filename), 3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image.set_shape([pano_height, pano_width, 3])
    return image

  def string_to_matrix(s, shape):
    """Decode strings to matrices tensor."""
    m = tf.reshape(
        tf.stack([tf.decode_csv(s, [0.0] * np.prod(shape))], 0), shape)
    m.set_shape(shape)
    return m

  def decode_line(line):
    """Decode text lines."""
    DataPair = collections.namedtuple(
        'DataPair', ['src_img', 'trt_img', 'fov', 'rotation', 'translation'])

    splitted = tf.decode_csv(line, ['']*10, field_delim=' ')

    img1 = load_single_image(pano_data_dir+splitted[0]+'/'+splitted[1]+'.jpeg')
    img2 = load_single_image(pano_data_dir+splitted[0]+'/'+splitted[2]+'.jpeg')
    fov = string_to_matrix(splitted[3], [1])
    r1 = string_to_matrix(splitted[4], [3, 3])
    t1 = string_to_matrix(splitted[5], [3])
    r2 = string_to_matrix(splitted[6], [3, 3])
    t2 = string_to_matrix(splitted[7], [3])
    sampled_r1 = string_to_matrix(splitted[8], [3, 3])
    sampled_r2 = string_to_matrix(splitted[9], [3, 3])

    r_c2_to_c1 = tf.matmul(sampled_r1, sampled_r2, transpose_a=True)
    t_c1 = tf.squeeze(tf.matmul(sampled_r1,
                                tf.expand_dims(tf.nn.l2_normalize(t2-t1), -1),
                                transpose_a=True))

    sampled_rotation = tf.matmul(tf.stack([sampled_r1, sampled_r2], 0),
                                 tf.stack([r1, r2], 0), transpose_a=True)

    sampled_views = transformation.rectilinear_projection(
        tf.stack([img1, img2], 0),
        [output_height, output_width],
        fov,
        tf.matrix_transpose(sampled_rotation))
    src_img, trt_img = sampled_views[0], sampled_views[1]
    return DataPair(src_img, trt_img, fov, r_c2_to_c1, t_c1)

  # meta_data_path has slash '/' at the end.
  ds = tf.data.Dataset.list_files(meta_data_path+'*')
  ds = ds.flat_map(load_text)
  ds = ds.map(decode_line)
  return ds


def generate_random_views(pano1_rgb,
                          pano2_rgb,
                          r1, t1, r2, t2,
                          max_rotation=90.,
                          max_tilt=5.,
                          output_fov=90.,
                          output_height=512,
                          output_width=512,
                          pano1_depth=None,
                          pano2_depth=None):
  """Generate stereo image pairs by randomly sampling the panoramic images.

  We randomly sample camera lookat directions and project the panorama to
  perspective images. We also compute the overlaping area between the pair given
  the depth map if depthmaps are provided. The overlap is measured by the
  minimum of the ratio between the area seen by both cameras and the image size.
  This function returns a ViewPair object containing the perspective images,
  the masks that shows the common area seen by both cameras, the camera's field
  of view (FoV), the relative rotation from camera 2 to camera 1, and the
  relative translation direction in the frame of camera 1.


  Args:
    pano1_rgb: [HEIGHT, WIDTH, 3] the input RGB panoramic image.
    pano2_rgb: [HEIGHT, WIDTH, 3] the input RGB panoramic image.
    r1: [3, 3] the camera to world rotation of camera 1.
    t1: [3] the world location of camera 1.
    r2: [3, 3] the camera to world rotation of camera 2.
    t2: [3] the world location of camera 2.
    max_rotation: (float) maximum relative rotation between the output image
      pair in degrees.
    max_tilt: (float) maximum tilt angle of the up vector in degrees.
    output_fov: (float) output images' horizontal field of view in degrees.
    output_height: (int) the height dimension of the output perspective images.
    output_width: (int) the width dimension of the output perspective images.
    pano1_depth: [HEIGHT, WIDTH, 1] the panoramic depth map of pano1_rgb.
    pano2_depth: [HEIGHT, WIDTH, 1] the panoramic depth map of pano2_rgb.

  Returns:
    ViewPair
  """
  ViewPair = collections.namedtuple(
      'ViewPair', ['img1', 'img2', 'mask1', 'mask2', 'fov', 'r', 't'])

  swap_yz = tf.constant([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]],
                        shape=[1, 3, 3])
  lookat_direction1 = math_utils.random_vector_on_sphere(
      1, [[-math.sin(math.pi/3), math.sin(math.pi/3)], [0., 2*math.pi]])
  lookat_direction1 = tf.squeeze(
      tf.matmul(swap_yz, tf.expand_dims(lookat_direction1, -1)), -1)

  lookat_direction2 = math_utils.uniform_sampled_vector_within_cone(
      lookat_direction1, math_utils.degrees_to_radians(max_rotation))
  lookat_directions = tf.concat([lookat_direction1, lookat_direction2], 0)
  up1 = math_utils.uniform_sampled_vector_within_cone(
      tf.constant([[0., 0., 1.]]), math_utils.degrees_to_radians(max_tilt))
  up2 = math_utils.uniform_sampled_vector_within_cone(
      tf.constant([[0., 0., 1.]]), math_utils.degrees_to_radians(max_tilt))
  lookat_rotations = math_utils.lookat_matrix(
      tf.concat([up1, up2], 0), lookat_directions)
  sample_rotations = tf.matmul(
      tf.concat([r1, r2], 0), lookat_rotations, transpose_a=True)

  sampled_views = transformation.rectilinear_projection(
      tf.stack([pano1_rgb, pano2_rgb], 0),
      [output_height, output_width],
      output_fov,
      sample_rotations)

  r_c2_to_c1 = tf.matmul(
      lookat_rotations[0], lookat_rotations[1], transpose_a=True)
  t_c1 = tf.squeeze(tf.matmul(lookat_rotations[0],
                              tf.expand_dims(tf.nn.l2_normalize(t2-t1), -1),
                              transpose_a=True))

  if pano1_depth is not None and pano2_depth is not None:
    sampled_depth = transformation.rectilinear_projection(
        tf.stack([pano1_depth, pano2_depth], 0),
        [output_height, output_width],
        output_fov,
        sample_rotations)

    fx = output_width*0.5/math.tan(math_utils.degrees_to_radians(output_fov)/2)
    intrinsics = tf.constant([[fx, 0., output_width*0.5],
                              [0., -fx, output_height*0.5],
                              [0., 0., 1.]])
    pose1_c2w = tf.concat([lookat_rotations[0], tf.expand_dims(t1, -1)], 1)
    pose2_c2w = tf.concat([lookat_rotations[1], tf.expand_dims(t2, -1)], 1)
    mask1, mask2 = overlap_mask(sampled_depth[0],
                                pose1_c2w,
                                sampled_depth[1],
                                pose2_c2w,
                                intrinsics)
  else:
    mask1 = None
    mask2 = None

  return ViewPair(sampled_views[0],
                  sampled_views[1],
                  mask1,
                  mask2,
                  output_fov,
                  r_c2_to_c1,
                  t_c1)
