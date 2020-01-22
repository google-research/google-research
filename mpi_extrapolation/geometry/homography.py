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

"""TensorFlow utils for image transformations via homographies.

Modified from code written by Shubham Tulsiani and Tinghui Zhou.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from mpi_extrapolation.geometry import sampling


def divide_safe(num, den, name=None):
  eps = 1e-8
  den += eps*tf.cast(tf.equal(den, 0), 'float32')
  return tf.divide(num, den, name=name)


def _transpose(rot):
  """Transposes last two dimnesions.

  Args:
      rot: relative rotation, are [...] X M X N matrices
  Returns:
      rot_t: [...] X N X M matrices
  """
  with tf.name_scope('transpose'):
    n_inp_dim = len(rot.get_shape())
    perm = list(range(n_inp_dim))
    perm[-1] = n_inp_dim - 2
    perm[-2] = n_inp_dim - 1
    rot_t = tf.transpose(rot, perm=perm)
    return rot_t


def inv_homography(k_s, k_t, rot, t, n_hat, a):
  """Computes inverse homography matrix.

  Args:
      k_s: intrinsics for source cameras, are [...] X 3 X 3 matrices
      k_t: intrinsics for target cameras, are [...] X 3 X 3 matrices
      rot: relative rotation, are [...] X 3 X 3 matrices
      t: [...] X 3 X 1, translations from source to target camera
      n_hat: [...] X 1 X 3, plane normal w.r.t source camera frame
      a: [...] X 1 X 1, plane equation displacement
  Returns:
      homography: [...] X 3 X 3 inverse homography matrices
  """
  with tf.name_scope('inv_homography'):
    rot_t = _transpose(rot)
    k_t_inv = tf.matrix_inverse(k_t, name='k_t_inv')

    denom = a - tf.matmul(tf.matmul(n_hat, rot_t), t)
    numerator = tf.matmul(tf.matmul(tf.matmul(rot_t, t), n_hat), rot_t)
    inv_hom = tf.matmul(
        tf.matmul(k_s, rot_t + divide_safe(numerator, denom)),
        k_t_inv, name='inv_hom')
    return inv_hom


def inv_homography_dmat(k_t, rot, t, n_hat, a):
  """Computes M where M*(u,v,1) = d_t.

  Args:
      k_t: intrinsics for target cameras, are [...] X 3 X 3 matrices
      rot: relative rotation, are [...] X 3 X 3 matrices
      t: [...] X 3 X 1, translations from source to target camera
      n_hat: [...] X 1 X 3, plane normal w.r.t source camera frame
      a: [...] X 1 X 1, plane equation displacement
  Returns:
      d_mat: [...] X 1 X 3 matrices
  """
  with tf.name_scope('inv_homography'):
    rot_t = _transpose(rot)
    k_t_inv = tf.matrix_inverse(k_t, name='k_t_inv')

    denom = a - tf.matmul(tf.matmul(n_hat, rot_t), t)
    d_mat = divide_safe(
        -1 * tf.matmul(tf.matmul(n_hat, rot_t), k_t_inv), denom, name='dmat')
    return d_mat


def transform_pts(pts_coords_init, h_mat):
  """Transforms input points according to homography.

  Args:
      pts_coords_init : [...] X H X W X 3; pixel (u,v,1) coordinates.
      h_mat : [...] X 3 X 3; desired matrix transformation
  Returns:
      pts_coords : [...] X H X W X 3; transformed (u, v, w) coordinates.
  """
  with tf.name_scope('transform_pts'):
    h_mat_size = tf.shape(h_mat)
    pts_init_size = tf.shape(pts_coords_init)
    pts_transform_size = [h_mat_size[0], h_mat_size[1], -1, h_mat_size[3]]

    pts_coords_init_reshape = tf.reshape(pts_coords_init, pts_transform_size)

    h_mat_transpose = _transpose(h_mat)
    pts_mul = tf.matmul(pts_coords_init_reshape, h_mat_transpose)
    pts_coords_transformed = tf.reshape(pts_mul, pts_init_size)
    return pts_coords_transformed


def normalize_homogeneous(pts_coords):
  """Converts homogeneous coordinates to regular coordinates.

  Args:
      pts_coords : [...] X n_dims_coords+1; Homogeneous coordinates.
  Returns:
      pts_coords_uv_norm : [...] X n_dims_coords;
          normal coordinates after dividing by the last entry.
  """
  with tf.name_scope('normalize_homogeneous'):
    pts_size = tf.shape(pts_coords)
    n_dims = tf.rank(pts_coords)
    n_dims_coords = pts_size[-1] - 1

    pts_coords_uv, pts_coords_norm = tf.split(
        pts_coords, [n_dims_coords, 1], axis=n_dims - 1)

    return divide_safe(pts_coords_uv, pts_coords_norm)


def transform_plane_imgs(imgs, pixel_coords_trg, k_s, k_t, rot, t, n_hat, a):
  """Transforms input imgs via homographies for corresponding planes.

  Args:
    imgs: are L X B X H_s X W_s X C
    pixel_coords_trg: B X H_t X W_t X 3; pixel (u,v,1) coordinates.
    k_s: intrinsics for source cameras, are B X 3 X 3 matrices
    k_t: intrinsics for target cameras, are B X 3 X 3 matrices
    rot: relative rotation, are B X 3 X 3 matrices
    t: B X 3 X 1, translations from source to target camera
    n_hat: L X B X 1 X 3, plane normal w.r.t source camera frame
    a: L X B X 1 X 1, plane equation displacement
  Returns:
    [...] X H_t X W_t X C images after bilinear sampling from input.
      Coordinates outside the image are sampled as 0.
  """
  with tf.name_scope('transform_plane_imgs'):

    hom_t2s_planes = inv_homography(k_s, k_t, rot, t, n_hat, a)
    pixel_coords_t2s = transform_pts(pixel_coords_trg, hom_t2s_planes)
    pixel_coords_t2s = normalize_homogeneous(pixel_coords_t2s)
    imgs_s2t = sampling.bilinear_wrapper(imgs, pixel_coords_t2s)

    return imgs_s2t


def planar_transform(imgs, pixel_coords_trg, k_s, k_t, rot, t, n_hat, a):
  """transforms imgs, masks and computes dmaps according to planar transform.

  Args:
    imgs: are [L, B, H, W, C], typically RGB images per layer
    pixel_coords_trg: B X H_t X W_t X 3;
        pixel (u,v,1) coordinates of target image pixels. (typically meshgrid)
    k_s: intrinsics for source cameras, are B X 3 X 3 matrices
    k_t: intrinsics for target cameras, are B X 3 X 3 matrices
    rot: relative rotation, are B X 3 X 3 matrices
    t: B X 3 X 1, translations from source to target camera
       (R*p_src + t = p_tgt)
    n_hat: L X B X 1 X 3, plane normal w.r.t source camera frame
      (typically [0 0 1])
    a: L X B X 1 X 1, plane equation displacement (n_hat * p_src + a = 0)
  Returns:
    imgs_transformed: L X [...] X C images in trg frame
  Assumes the first dimension corresponds to layers.
  """
  with tf.name_scope('planar_transform'):
    n_layers = tf.shape(imgs)[0]
    rot_rep_dims = [n_layers, 1, 1, 1]

    cds_rep_dims = [n_layers, 1, 1, 1, 1]

    k_s = tf.tile(tf.expand_dims(k_s, axis=0), rot_rep_dims)
    k_t = tf.tile(tf.expand_dims(k_t, axis=0), rot_rep_dims)
    t = tf.tile(tf.expand_dims(t, axis=0), rot_rep_dims)
    rot = tf.tile(tf.expand_dims(rot, axis=0), rot_rep_dims)
    pixel_coords_trg = tf.tile(tf.expand_dims(
        pixel_coords_trg, axis=0), cds_rep_dims)

    imgs_trg = transform_plane_imgs(
        imgs, pixel_coords_trg, k_s, k_t, rot, t, n_hat, a)
    return imgs_trg


def transform_plane_eqns(rot, t, n_hat, a):
  """Transforms plane euqations according to frame transformation.

  Args:
    rot: relative rotation, are [...] X 3 X 3 matrices
    t: [...] X 3 X 1, translations from source to target camera
    n_hat: [...] X 1 X 3, plane normal w.r.t source camera frame
    a: [...] X 1 X 1, plane equation displacement
  Returns:
    n_hat_t: [...] X 1 X 3, plane normal w.r.t target camera frame
    a_t: [...] X 1 X 1, plane plane equation displacement
  """
  with tf.name_scope('transform_plane_eqns'):
    rot_t = _transpose(rot)
    n_hat_t = tf.matmul(n_hat, rot_t)
    a_t = a - tf.matmul(n_hat, tf.matmul(rot_t, t))
    return n_hat_t, a_t


def shift_plane_eqns(plane_shift, pred_planes):
  """For every p on the original plane, p+plane_shift lies on the new plane.

  Args:
    plane_shift: relative rotation, are B X 1 X 3 points
    pred_planes: [n_hat, a]
        n_hat is [...] X 1 X 3, plane normal w.r.t source camera frame
        a is [...] X 1 X 1, plane equation displacement
  Returns:
    shifted_planes: [n_hat_t, a_t] where
        n_hat_t: [...] X 1 X 3, plane normal after shifting
        a_t: [...] X 1 X 1, plane plane equation displacement after plane shift
  """
  n_hat = pred_planes[0]
  a = pred_planes[1]
  n_hat_t = n_hat
  a_t = a - tf.reduce_sum(plane_shift*n_hat, axis=-1, keep_dims=True)
  return [n_hat_t, a_t]


def trg_disp_maps(pixel_coords_trg, k_t, rot, t, n_hat, a):
  """Computes pixelwise inverse depth for target pixels via plane equations.

  Args:
    pixel_coords_trg: [...] X H_t X W_t X 3; pixel (u,v,1) coordinates.
    k_t: intrinsics for target cameras, are [...] X 3 X 3 matrices
    rot: relative rotation, are [...] X 3 X 3 matrices
    t: [...] X 3 X 1, translations from source to target camera
    n_hat: [...] X 1 X 3, plane normal w.r.t source camera frame
    a: [...] X 1 X 1, plane equation displacement
  Returns:
    [...] X H_t X W_t X 1 images corresponding to inverse depth at each pixel
  """
  with tf.name_scope('trg_disp_maps'):
    dmats_t = inv_homography_dmat(k_t, rot, t, n_hat, a)  # size: [...] X 1 X 3
    disp_t = tf.reduce_sum(
        tf.expand_dims(dmats_t, -2)*pixel_coords_trg, axis=-1, keep_dims=True)
    return disp_t
