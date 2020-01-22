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

"""Renders depth maps from EuRoC MAV point clouds.

https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets

Some of the rooms of the EuRoC MAV dataset have point clouds. This script
renders depth maps from the point clouds.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

from absl import app
from absl import flags
import matplotlib.image
import numpy as np


flags.DEFINE_string('room_path', '', 'Path to the EuRoC data for one of the '
                    'rooms ')

flags.DEFINE_string('output_path', '', 'Path where to store the outputs.')


FLAGS = flags.FLAGS


# A 4D transform that connects the Cam0 to the body of the MAV. This is taken
# from the sensor.yaml file. To project the point cloud on Cam1, please replace
# with the respective extrinsic matrix. This is constant across all the rooms in
# the dataset.
CAM0_TO_BODY = np.array(
    [[0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
     [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
     [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
     [0.0, 0.0, 0.0, 1.0]])

# Intrinsics of Cam0. This is taken from cam0/sensor.yaml and is the same for
# all rooms.
FX = 458.654
FY = 457.296
X0 = 367.215
Y0 = 248.375
K1 = -0.28340811
K2 = 0.07395907
H = 480
W = 752


def get_camera_view_pointcloud(transform, xyz, greyscale_color):
  """Transform point cloud to camera view, prune points outside of the view.

  Args:
    transform: 4x4 transform matrix representing position and orientation of
      the body of the MAV.
    xyz: A 4xN matrix, point cloud in homogeneous coordinates. The k-th column
      is (x, y, z, 1), where x, y, z are the coordinates of the k-th point.
    greyscale_color: N-vector, vertex grayscale value. The k-th entry is the
      greyscale color of the k-th point.

  Returns:
    3xM (M < N) matrix representing the point cloud in the camera view.
    M vector, vertex grayscale value.
    Only points that fall within the camera viweing angle and are in front of
    the camera are kept.
  """

  overall_transform = np.linalg.inv(CAM0_TO_BODY).dot(np.linalg.inv(transform))
  transformed_xyz = xyz.dot(overall_transform.transpose())
  x, y, z, _ = _split(transformed_xyz)
  u, v = _project_and_distort(x, y, z)
  # Remove points that are out of frame. Keep some margin (1.05), to make sure
  # occlusions are addressed correctly at the edges of the field of view. For
  # example a point that is just slightly out of frame can occlude a neighboring
  # point inside the frame.
  valid_mask = np.logical_and.reduce(
      (z > 0.0, u > -0.05 * W, u < W * 1.05, v > -0.05 * H, v < H * 1.05),
      axis=0)
  valid_points = valid_mask.nonzero()[0]
  return transformed_xyz[valid_points, :3], greyscale_color[valid_points]


def get_occluded_points(xyz, neighborhood_radius, z_threshold):
  """Remove points that are occluded by others from a camera-view point cloud.

  Args:
    xyz: A 3xN matrix representing the point cloud in the camera view.
    neighborhood_radius: The radius around each point in which it occludes
      others.
    z_threshold: Minimum z distance betweem two points for them considered to
      be occluding each other. If two points are verty close in z, they likely
      belong to the same surface and thus do not occlude each other.

  Returns:
    A list of indices in xyz corresponding to points that are occluded.
  """

  def get_bin(xz, yz):
    xbin = int(round(xz / neighborhood_radius))
    ybin = int(round(yz / neighborhood_radius))
    return xbin, ybin

  xs, ys, zs = _split(xyz)
  xzs = xs / zs
  yzs = ys / zs
  grid = collections.defaultdict(lambda: np.inf)
  for ind in range(xyz.shape[0]):
    # Place each point in the bin where it belongs, and in the neighboring bins.
    # Keep only the closest point to the camera in each bin.
    xbin, ybin = get_bin(xzs[ind], yzs[ind])
    for i in range(-1, 2):
      for j in range(-1, 2):
        grid[(xbin + i, ybin + j)] = min(grid[(xbin + i, ybin + j)], zs[ind])

  occluded_indices = []
  for ind in range(xyz.shape[0]):
    # Loop over all points and see if they are occluded, by finding the closest
    # point to the camera within the same bin and testing for the occlusion
    # condition. A point is occluded if there is another point in the same bin
    # that is far enough in z, so that it cannot belong to the same surface,
    zmin = grid[get_bin(xzs[ind], yzs[ind])]
    if zmin < (1 - z_threshold) * zs[ind]:
      occluded_indices.append(ind)
  return occluded_indices


def render_rgb(xyz, c):
  """Given a colored cloud in camera coordinates, render an image.

  This function is useful for visualization / debugging.

  Args:
    xyz: A 3xN matrix representing the point cloud in the camera view.
    c: A N-long vector containing (greyscale) colors of the points.

  Returns:
    A rendered image.
  """
  x, y, z = _split(xyz)
  u, v = _project_and_distort(x, y, z)
  u = np.floor(0.5 * u).astype(int)
  v = np.floor(0.5 * v).astype(int)

  rendered_c = np.full((int(H / 2), int(W / 2)), 0.0)
  rendered_c[v, u] = c
  rendered_c = np.stack([rendered_c] * 3, axis=2)
  return rendered_c


def render_z(xyz):
  """Given a colored cloud in camera coordinates, render a depth map.

  This function is useful for visualization / debugging.

  Args:
    xyz: A 3xN matrix representing the point cloud in the camera view.

  Returns:
    A rendered depth map.
  """
  x, y, z = _split(xyz)
  u, v = _project_and_distort(x, y, z)
  u = np.floor(0.5 * u).astype(int)
  v = np.floor(0.5 * v).astype(int)
  rendered_z = np.full((int(H / 2), int(W / 2)), -np.inf)
  rendered_z[v, u] = z
  maxz = np.max(rendered_z)
  rendered_z = np.where(rendered_z == -np.inf, np.nan, rendered_z)
  rendered_z /= maxz
  return rendered_z


class GroundTruthInterpolator(object):
  """Interpolates MAV position and orientation groundtruth to a timestamp."""

  def __init__(self, filename):
    """Creates an instance.

    Args:
      filename: A string, filepath of the state_groundtruth_estimate0.csv file.
    """
    with open(filename) as f:
      lines = f.readlines()
    lines = lines[1:]  # skip the first line

    gt = []
    for l in lines:
      tokens = l.split(',')
      gt.append([float(t) for t in tokens[:8]])
    self._gt = np.array(gt)
    self._mint = np.min(self._gt[:, 0])
    self._maxt = np.max(self._gt[:, 0])

  def get_transform(self, timestamp):
    """Interpolates the MAV's transform matrix at a timestamp."""
    if timestamp < self._mint or timestamp > self._maxt:
      return None

    # self._gt[:, 0], the 0th column, is the timestamp. Columns 1-3 are x, y, z,
    # and columns 4-7 are quaternion components describing the rotation.
    timestamps = self._gt[:, 0]
    x = np.interp(timestamp, timestamps, self._gt[:, 1])
    y = np.interp(timestamp, timestamps, self._gt[:, 2])
    z = np.interp(timestamp, timestamps, self._gt[:, 3])

    qw = np.interp(timestamp, timestamps, self._gt[:, 4])
    qx = np.interp(timestamp, timestamps, self._gt[:, 5])
    qy = np.interp(timestamp, timestamps, self._gt[:, 6])
    qz = np.interp(timestamp, timestamps, self._gt[:, 7])

    # Creates a matrix
    transform = np.array([[
        1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw,
        2 * qx * qz + 2 * qy * qw, x
                          ],  # pylint: disable=bad-continuation
                          [
                              2 * qx * qy + 2 * qz * qw,
                              1 - 2 * qx * qx - 2 * qz * qz,
                              2 * qy * qz - 2 * qx * qw, y
                          ],
                          [
                              2 * qx * qz - 2 * qy * qw,
                              2 * qy * qz + 2 * qx * qw,
                              1 - 2 * qx * qx - 2 * qy * qy, z
                          ], [0.0, 0.0, 0.0, 1.0]])

    return transform


def read_ply(filename):
  """Reads a PLY file representing EuRoc's point cloud."""
  with open(filename) as f:
    lines = f.readlines()
  lines = lines[11:]
  xyz = []
  c = []  # The color channel (just one, it's greyscale)
  for l in lines:
    tokens = l.split(' ')
    xyz.append([float(t) for t in tokens[:3]])
    c.append(float(tokens[3]))

  return np.array(xyz), np.array(c)


def filter_out_ot_frame_points(xyz, c):
  """Remove all points in a camera-view pointcloud that are out of frame.

  Args:
    xyz: A 3xN matrix representing the point cloud in the camera view.
    c: A N-long vector containing (greyscale) colors of the points.

  Returns:
    A 3xM matrix and a M-long vector representing the filtered colored point
    cloud.
  """
  x, y, z = _split(xyz)
  u, v = _project_and_distort(x, y, z)
  u = np.floor(u).astype(int)
  v = np.floor(v).astype(int)
  valid_mask = np.logical_and.reduce((u >= 0, u < W, v >= 0, v < H), axis=0)
  valid_points = valid_mask.nonzero()[0]
  return xyz[valid_points, :], c[valid_points]


def sample_uniform(xyz, bin_size):
  """subsamples a point cloud to be more uniform in perspective coordinates.

  Args:
    xyz: A 3xN matrix representing the point cloud in the camera view.
    bin_size: Size of a square in which we allow only a single point.

  Returns:
    A list of indices, corresponding to a subset of the original `xyz`, to keep.
  """
  x, y, z = _split(xyz)
  xbins = (x / z / bin_size)
  ybins = (y / z / bin_size)
  xbins_rounded = np.round(xbins)
  ybins_rounded = np.round(ybins)
  xbins_diff = xbins_rounded - xbins
  ybins_diff = ybins_rounded - ybins
  diff_sq = xbins_diff**2 + ybins_diff**2

  bin_to_ind = {}
  for ind in range(len(diff_sq)):
    bin_ = (xbins_rounded[ind], ybins_rounded[ind])
    if bin_ not in bin_to_ind or diff_sq[ind] < bin_to_ind[bin_][1]:
      bin_to_ind[bin_] = (ind, diff_sq[ind])

  inds_to_keep = sorted([i[0] for i in bin_to_ind.values()])
  return inds_to_keep


def main(argv):
  del argv  # unused
  gti = GroundTruthInterpolator(
      os.path.join(FLAGS.room_path, 'state_groundtruth_estimate0/data.csv'))
  print('Groundtruth loaded.')
  xyz, c = read_ply(os.path.join(FLAGS.room_path, 'pointcloud0/data.ply'))
  print('PLY loaded.')
  xyz_homogeneous = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)

  imagesto_render = sorted(
      os.listdir(os.path.join(FLAGS.room_path, 'cam0/data')))

  imagesto_render = imagesto_render[0::5]  # render every fifth image

  for imfile in imagesto_render:
    timestamp = float(imfile.split('.')[0])
    transform = gti.get_transform(timestamp)
    if transform is None:
      print ('Timestamp %d has no groundtruth.' % int(timestamp))
      continue
    else:
      print ('Rendering timestamp %d...' % int(timestamp))

    xyz_view, c_view = get_camera_view_pointcloud(transform, xyz_homogeneous, c)
    print ('View pointcloud generated, %d points.' % xyz_view.shape[0])
    occluded_inds = get_occluded_points(xyz_view, 0.02, 0.08)
    occluded_inds = set(occluded_inds)
    visible_indices = [
        i for i in range(xyz_view.shape[0]) if i not in occluded_inds
    ]
    print ('%d visible points found.' % len(visible_indices))
    visible_xyz = xyz_view[visible_indices, :]
    visible_c = c_view[visible_indices]
    visible_xyz, visible_c = filter_out_ot_frame_points(visible_xyz, visible_c)

    inds_to_keep = sample_uniform(visible_xyz, 1e-2)
    visible_xyz = visible_xyz[inds_to_keep]
    visible_c = visible_c[inds_to_keep]

    rgb_image = render_rgb(visible_xyz, visible_c)
    z_image = render_z(visible_xyz)
    matplotlib.image.imsave(
        os.path.join(FLAGS.output_path, '%dgrayscale.png' % int(timestamp)),
        rgb_image)
    matplotlib.image.imsave(
        os.path.join(FLAGS.output_path, '%ddepth.png' % int(timestamp)),
        z_image)
    np.save(
        os.path.join(FLAGS.output_path, '%d.npy' % int(timestamp)), visible_xyz)


def _split(matrix):
  return [
      np.squeeze(v, axis=1) for v in np.split(matrix, matrix.shape[1], axis=1)
  ]


def _project_and_distort(x, y, z):
  """Apply perspective projection and distortion on a point cloud.

  Args:
    x: A vector containing the x coordinates of the points.
    y: A vector containing the y coordinates of the points, same length as x.
    z: A vector containing the z coordinates of the points, same length as x.

  Returns:
    A tuple of two vectors of the same length as x, containing the image-plane
    coordinates (u, v) of the point cloud.
  """
  xz = (x / z)
  yz = (y / z)
  # 2. Apply radial camera distortion:
  rr = xz**2 + yz**2
  distortion = (1 + K1 * rr + K2 * rr * rr)
  xz *= distortion
  yz *= distortion
  # 3. Apply intrinsic matrix to get image coordinates:
  u = FX * xz + X0
  v = FY * yz + Y0
  return u, v


if __name__ == '__main__':
  app.run(main)
