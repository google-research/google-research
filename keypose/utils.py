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

"""Utilities for reading images, parsing protobufs, etc."""

import os

import matplotlib as plt
import matplotlib.cm  # pylint: disable=unused-import
import numpy as np
from skimage.transform import resize
import tensorflow as tf
import yaml

from google.protobuf import text_format
from keypose import data_pb2 as pb

try:
  import cv2  # pylint: disable=g-import-not-at-top
except ImportError as e:
  print(e)


# Read image, including .exr images.
def read_image(fname):
  ext = os.path.splitext(fname)[1]
  if ext == '.exr':
    print('reading exr file')
    image = cv2.imread(fname, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
  else:
    image = cv2.imread(fname)
  assert image is not None, 'Cannot open %s' % fname
  return image


def k_matrix_from_camera(camera):
  return np.array([
      [camera.fx, 0, camera.cx],  # fx, cx
      [0, camera.fy, camera.cy],  # fy, cy
      [0, 0, 1]
  ])


def q_matrix_from_camera(camera):
  return np.array([
      [1.0, 0, 0, -camera.cx],  # cx
      [0, 1.0, 0, -camera.cy],  # cy
      [0, 0, 0, camera.fx],  # fx
      [0, 0, 1.0 / camera.baseline, 0]
  ])  # baseline


def p_matrix_from_camera(camera):
  return np.array([
      [camera.fx, 0, camera.cx, 0],  # fx, cx
      [0, camera.fy, camera.cy, 0],  # fy, cy
      [0, 0, 0, camera.fx * camera.baseline],  # fx*baseline
      [0, 0, 1.0, 0]
  ])


# Parse a transform protobuf.
def get_transform(trans_pb):
  transform = np.array(trans_pb.element)
  if transform.size == 16:
    return transform.reshape((4, 4))
  return None


def get_keypoints(targ_pb):
  """Parse a keypoint protobuf."""
  uvds = []
  visible = []
  focal_length = targ_pb.camera.fx
  baseline = targ_pb.camera.baseline
  for kp_pb in targ_pb.keypoints:
    uvds.append(
        np.array([kp_pb.u, kp_pb.v, focal_length * baseline / kp_pb.z, 1]))
    visible.append(kp_pb.visible)
  transform = get_transform(targ_pb.transform)
  if transform is None:
    world_t_uvd = None
  else:
    world_t_uvd = np.linalg.inv(transform).dot(
        q_matrix_from_camera(targ_pb.camera))
  return np.array(uvds), world_t_uvd, np.array(visible)


# NOTE: xyzs are NOT world coords.  xy are percent image coords, z is depth (m).
def get_contents_pb(targ_pb):
  uvds, xyzs, visible = get_keypoints(targ_pb)
  cam = targ_pb.camera
  transform = get_transform(targ_pb.transform)
  if transform is None:
    return cam, None, None, uvds, xyzs, visible, transform
  uvd_t_world = p_matrix_from_camera(cam).dot(transform)
  world_t_uvd = np.linalg.inv(transform).dot(q_matrix_from_camera(cam))
  return cam, uvd_t_world, world_t_uvd, uvds, xyzs, visible, transform


# Reading text-formatted protobuf files
def read_from_text_file(path, proto):
  with open(path, 'r') as file:
    text_format.Parse(file.read(), proto)
  return proto


def read_target_pb(path):
  target_pb = pb.KeyTargets()
  read_from_text_file(path, target_pb)
  return target_pb


def read_keys_pb(path):
  target_pb = pb.KeyTargets()
  read_from_text_file(path, target_pb)
  return target_pb.kp_target


# Read the contents of a target protobuf.
def read_contents_pb(path):
  return get_contents_pb(read_target_pb(path).kp_target)


# Read contents of a camera protobuf.
def read_data_params(path):
  data_pb = pb.DataParams()
  read_from_text_file(path, data_pb)
  cam = data_pb.camera
  return data_pb.resx, data_pb.resy, data_pb.num_kp, cam


def project_np(mat, vec):
  """Projects homogeneous 3D XYZ coordinates to image uvd coordinates."""
  # <vec> has shape [4, N].
  # <mat> has shape [4, 4]
  # Return has shape [4, N].

  p = mat.dot(vec)
  # Using <3:4> instead of <3> preserves shape.
  p = p / (p[3:4, :] + 1.0e-10)
  return p


def get_params(param_file=None, cam_file=None, cam_image_file=None):
  """Returns default or overridden user-specified parameters, and cam params."""

  # param_file points to a yaml string.
  # cam_file points to a camera pbtxt.
  # cam_image_file points to a camera pbtxt.

  params = ConfigParams()
  if param_file:
    params.merge_yaml(param_file)
  mparams = params.model_params

  if cam_file:
    print('Model camera params file name is: %s' % cam_file)
    resx, resy, _, cam = read_data_params(cam_file)

  mparams.resx = resx
  mparams.resy = resy
  mparams.modelx = resx
  mparams.modely = resy

  if mparams.crop:
    mparams.modelx = int(mparams.crop[0])
    mparams.modely = int(mparams.crop[1])

  mparams.disp_default /= resx  # Convert to fraction of image.

  if cam_image_file:
    print('Image camera params file name is: %s' % cam_image_file)
    _, _, _, cam_image = read_data_params(cam_image_file)

  print('MParams:', mparams.make_dict())
  return params, cam, cam_image


# General configuration class for referencing parameters using dot notation.
class ConfigParams:
  """General configuration class for referencing params using dot notation."""

  def __init__(self, init=None):
    if init:
      self.merge_yaml(init)

  def make_dict(self):
    ret = {}
    for k in self.__dict__:
      val = self.__dict__[k]
      if isinstance(val, self.__class__):
        ret[k] = val.make_dict()
      else:
        ret[k] = val
    return ret

  # No nesting.
  def make_shallow_dict(self):
    ret = {}
    for k in self.__dict__:
      val = self.__dict__[k]
      if not isinstance(val, self.__class__):
        ret[k] = val
    return ret

  def merge_dict(self, p_dict):
    """Merges a dictionary into this params."""
    for k in p_dict:
      val = p_dict[k]
      if isinstance(val, dict):
        if getattr(self, k, None):
          sub_params = getattr(self, k)
          sub_params.merge_dict(val)
        else:
          sub_params = self.__class__()
          sub_params.merge_dict(val)
          setattr(self, k, sub_params)
      else:
        setattr(self, k, val)

  def merge_yaml(self, yaml_str):
    try:
      ret = yaml.safe_load(yaml_str)
    except yaml.YAMLError as exc:
      print('Error in loading yaml string')
      print(exc)
      return False
    self.merge_dict(ret)
    return True

  # Takes a string of the form 'a:1,b:2,c:[1,2],d:abcd', etc.
  # No nesting.
  def merge_str(self, p_str):
    items = p_str.split('=')
    assert len(items) >= 2
    # pylint: disable=g-complex-comprehension
    items = items[:1] + [
        item for v in items[1:-1] for item in v.rsplit(',', 1)
    ] + items[-1:]
    y_list = [items[i] + ': ' + items[i + 1] for i in range(0, len(items), 2)]
    y_str = '\n'.join(y_list)
    self.merge_yaml(y_str)

  # Returns a string suitable for reading back in.  Does not allow
  # nested ConfigParams.
  def __repr__(self):
    # Checks float repr for not having a decimal point in scientific notation.
    # Checks for None and returns empty string.
    def yaml_str(obj):
      if isinstance(obj, type(None)):
        return ''
      if isinstance(obj, list):
        return '[' + ','.join([yaml_str(x) for x in obj]) + ']'
      ret = str(obj)
      if isinstance(obj, float):
        if 'e' in ret and '.' not in ret:
          return '.0e'.join(ret.split('e'))
      return ret

    ivars = self.make_shallow_dict()
    s = ','.join([k + '=' + yaml_str(ivars[k]) for k in ivars])
    return s.replace(' ', '')


DEFAULT_PARAMS = """
model_params:
  disp_default: 30  # Default offset in disparity.
  sym: [0]  # Symmetry in the loss function.
  input_sym: [0]  # Mix up symmetric ground truth values.
  use_stereo: true  # Use stereo input.
  visible: true  # Use only samples with visible keypoints.
  crop: [180, 120, 30]  # Crop patch size, WxH, plus disp offset.
  dither: 0.0  # Amount to dither the crop, in pixels.
  gamma: []
"""


def resize_image(image, cam, cam_image, kps_pb):
  """Resize an image using scaling and cropping."""
  resx, resy = int(cam_image.resx), int(cam_image.resy)
  nxs, nys = int(cam.resx), int(cam.resy)
  fx = cam_image.fx
  nfx = cam.fx
  scale = fx / nfx

  crop = resy - nys * scale, resx - nxs * scale
  cropx, cropy = 0, 0
  if crop[0] > 1.5:
    cropy = int(round(crop[0] * 0.5))
  if crop[1] > 1.5:
    cropx = int(round(crop[1] * 0.5))

  # Resize image.
  image = image[cropy:resy - cropy, cropx:resx - cropx, :]
  image = resize(image, (nys, nxs), mode='constant')  # Converts to float.
  image = image.astype(np.float32)

  def scale_cam(cam_pb):
    cam_pb.fx /= scale
    cam_pb.fy /= scale
    cam_pb.cx = (cam_pb.cx - cropx) / scale
    cam_pb.cy = (cam_pb.cy - cropy) / scale
    cam_pb.resx = nxs
    cam_pb.resy = nys

  def resize_target(targ_pb):
    scale_cam(kps_pb.camera)
    for kp in targ_pb.keypoints:
      kp.u = (kp.u - cropx) / scale
      kp.v = (kp.v - cropy) / scale
      kp.x = kp.u / nxs
      kp.y = kp.v / nys
      if kp.u < 0 or kp.u >= nxs or kp.v < 0 or kp.v >= nys:
        kp.visible = 0.0

  resize_target(kps_pb)
  return image


def do_occlude_crop(image,
                    image2,
                    key_pts,
                    key_pts_r,
                    crop,
                    visible,
                    dither,
                    var_offset=False):
  """Crop area around the object."""
  # Crop is [W, H, R]', where 'R' is right-disparity offset; or else [].
  # Images can be either floating-point or uint8.

  offset = np.array([0, 0, 0], dtype=np.float32)
  crop = np.array(crop)
  if crop.size == 0:
    return image, image2, offset, visible
  nxs, nys = crop[0], crop[1]

  def do_crop(im, left_x, top_y, margin=10.0):
    y, x, _ = im.shape
    x -= margin
    y -= margin
    right_x = left_x + nxs
    bot_y = top_y + nys
    if (left_x < margin or left_x > x or right_x < margin or right_x > x or
        top_y < margin or top_y > y or bot_y < margin or bot_y > y):
      visible[:] = 0.0
      return im[0:nys, 0:nxs, :]
    return im[top_y:bot_y, left_x:right_x, :]

  centroid = np.mean(key_pts, axis=0)[0:2]
  centroid += np.random.uniform(low=-dither, high=dither, size=(2))
  off_x = int(centroid[0] - nxs / 2 - (nxs - nys) / 2)
  off_y = int(centroid[1] - nys / 2)
  image = do_crop(image, off_x, off_y)
  off_d = crop[2]
  if var_offset:
    off_d = int(centroid[0] - np.mean(key_pts_r, axis=0)[0])
  image2 = do_crop(image2, off_x - off_d, off_y)
  offset = np.array([off_x, off_y, off_d], dtype=np.float32)
  return image, image2, offset, visible


# Meshing functions.
def farthest_point_sampling(point_set, k):
  """Find the k most spread-out poses of a set of poses; translation only."""
  num_pts = point_set.shape[0]
  start_idx = np.random.randint(num_pts)
  existing_set = np.expand_dims(point_set[start_idx], axis=0)
  rest_set = np.copy(point_set)
  np.delete(rest_set, start_idx, 0)

  existing_indices = [start_idx]
  rest_indices = np.arange(num_pts)
  np.delete(rest_indices, start_idx)

  for _ in range(k - 1):
    dist = np.sum(np.square(existing_set), axis=1, keepdims=True) + \
           np.sum(np.square(rest_set.T), axis=0, keepdims=True) - \
           np.dot(existing_set, rest_set.T) * 2
    min_dist = dist.min(axis=0)
    max_idx = min_dist.argmax()
    existing_set = np.concatenate(
        [existing_set, np.expand_dims(rest_set[max_idx], axis=0)], axis=0)
    existing_indices.append(rest_indices[max_idx])
    np.delete(rest_set, max_idx, 0)
    np.delete(rest_indices, max_idx)

  return existing_set, existing_indices


# Class for holding a CAD object and its keypoints.
class MeshObj:
  """Class for holding a CAD object and its keypoints."""

  def __init__(self):
    self.keypoints = np.array([])
    self.vertices = np.array([])
    self.large_points = True

  def read_obj(self, path, num=300):
    """Read in a .obj file, parse into vertices and keypoints."""
    # Vertices are in label "o mesh".
    # Keypoints are in label "o kp.NNN".
    # Just read vertices, not other elements of the .obj file.
    kps = {}
    with open(path, 'r') as f:
      line = f.readline()
      while True:
        if not line:
          break
        if len(line) > 2 and line[:2] == 'o ':
          # New mesh.
          name = line[2:-1]
          vertices, line = self._read_sub_obj(f)
          if 'mesh' in name:
            self.vertices = vertices
          if 'kp' in name:  # Keypoint object.
            kp_num = int(name.split('.')[1])
            kp_val = np.mean(vertices, axis=0)
            kps[kp_num] = kp_val
        else:
          line = f.readline()
    keypoints = [kps[i] for i in range(len(kps))]
    if not keypoints:
      print('Did not find any keypoints')
      return
    self.keypoints = np.array(keypoints)
    self.keypoints = np.concatenate(
        [self.keypoints, np.ones((self.keypoints.shape[0], 1))], axis=-1)
    self.xyzw = np.concatenate(
        [self.vertices, np.ones((self.vertices.shape[0], 1))], axis=-1)
    self.make_reduced(num)

  def make_reduced(self, num):
    self._make_colored_subset(num)
    self.xyzw_reduced = np.concatenate(
        [self.vertices_reduced,
         np.ones((self.vertices_reduced.shape[0], 1))],
        axis=-1)

  def _read_sub_obj(self, f):
    """Read in all the vertices."""
    # First read to the beginning of vertices.
    line = ''
    vertices = []
    while True:
      line = f.readline()
      if not line:
        return None
      if 'v ' in line:
        break

    # Now process all vertices.
    while True:
      elems = line[:-1].split(' ')
      if elems[0] != 'v':
        break
      vertices.append(np.array([float(x) for x in elems[1:]]))
      line = f.readline()
      if not line:
        break
    return np.array(vertices), line

  # Pick a subset of colored points for display.
  def _make_colored_subset(self, num):
    self.vertices_reduced, _ = farthest_point_sampling(self.vertices, num)
    colors = plt.cm.get_cmap('spring')
    zs = self.vertices_reduced[:, 2]
    self.reduced_colors = (255 *
                           colors(np.interp(zs, (zs.min(), zs.max()),
                                            (0, 1)))[:, :3]).astype(np.uint8)

  def project_to_uvd(self, xyzs, p_matrix):
    """Does a transform from CAD coords to kps coords, then projects to uvd."""
    kps_t_mesh = ortho_procrustes(self.keypoints, xyzs.T[:, :3])
    uvd_t_mesh = p_matrix.dot(kps_t_mesh)
    self.uvds = project_np(uvd_t_mesh, self.xyzw.T).T
    self.uvds_reduced = project_np(uvd_t_mesh, self.xyzw_reduced.T).T

  def draw_points(self, image, offsets=(0, 0)):
    """Draws u,v points on an image as circles."""
    for i, pt in enumerate(self.uvds_reduced):
      u = int(pt[0] - offsets[0])
      v = int(pt[1] - offsets[1])
      if u < 0 or u >= image.shape[1] or v < 0 or v >= image.shape[0]:
        continue
      image[v, u, :] = self.reduced_colors[i, :]
      if self.large_points:
        if u > 0 and v > 0 and u < image.shape[1] - 1 and v < image.shape[0] - 1:
          for ui in range(-1, 2):
            for vi in range(-1, 2):
              image[v + vi, u + ui, :] = self.reduced_colors[i, :]
    return image

  def segmentation(self, size):
    """Create segmentation image using morphology operations."""
    mask = np.zeros(size)
    for pt in self.uvds:
      u = int(pt[0])
      v = int(pt[1])
      if u < 0 or u >= size[1] or v < 0 or v >= size[0]:
        continue
      mask[v, u] = 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Converts to float.
    mask_dilated = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    border = mask_dilated - mask
    return mask.astype(np.uint8), border.astype(np.uint8)


# Orthogonal procrustes method for two sets of 3D points.
# Also does the initial translation to centroid.
# Should work in degenerate cases: 1 or 2 points.
def ortho_procrustes(p_c, p_s):
  """Return R,t of the best estimate transform on point clouds p_c and p_s."""
  # No scaling. Transform is from p_c to p_s, i.e., T * p_c ~= p_s.
  # Format of args is numpy array, nx3 or nx4,d, each row a 3D point

  p_c = p_c[:, :3]
  p_s = p_s[:, :3]
  cm = np.mean(p_c, 0)
  pcn = p_c - cm  # get mean of each dimension, subtract
  sm = np.mean(p_s, 0)
  psn = p_s - sm
  a_mat = psn.transpose().dot(pcn)
  u, _, vt = np.linalg.svd(a_mat)
  dd = np.eye(3)
  dd[2, 2] = np.linalg.det(u.dot(vt))
  rot = u.dot(dd.dot(vt))  # Should check for orthogonality.
  t = sm - rot.dot(cm)
  tfm = np.eye(4)
  tfm[0:3, 0:3] = rot
  tfm[0:3, 3] = t
  return tfm


# Read in CAD model from a .obj file.
# <path> is a file path from the data_tools/objects/ directory.
# <num> is the number of points to use in the reduced set.
def read_mesh(path, num=300):
  obj = MeshObj()
  print('Reading obj file %s' % path)
  obj.read_obj(path, num)
  print('Obj file has %d vertices and %d keypoints' %
        (len(obj.vertices), len(obj.keypoints)))
  return obj


# Error functions.
def project(tmat, tvec, tvec_transpose=False):
  """Projects homogeneous 3D XYZ coordinates to image uvd coordinates."""
  # <tvec> has shape [[N,] batch_size, 4, num_kp] or [batch_size, num_kp, 4].
  # <tmat> has shape [[N,] batch_size, 4, 4]
  # Return has shape [[N,] batch_size, 4, num_kp]

  tp = tf.matmul(tmat, tvec, transpose_b=tvec_transpose)
  # Using <3:4> instead of <3> preserves shape.
  tp = tp / (tp[Ellipsis, 3:4, :] + 1.0e-10)
  return tp


def world_error(labels, xyzw):
  xyzw = tf.transpose(xyzw, [0, 2, 1])  # [batch, 4, num_kp]
  # [batch, 4, num_kp]
  gt_world_coords = project(labels['to_world_L'], labels['keys_uvd_L'], True)
  sub = xyzw[:, :3, :] - gt_world_coords[:, :3, :]
  wd = tf.square(sub)
  wd = tf.reduce_sum(wd, axis=[-2])  # [batch, num_kp] result.
  wd = tf.sqrt(wd)
  return wd  # [batch, num_kp]
