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

"""Utilities for reading images, parsing protobufs, etc."""

import math
import os
import random

from google.protobuf import text_format
import matplotlib as plt
import matplotlib.cm  # pylint: disable=unused-import
import numpy as np
from skimage.draw import ellipse
from skimage.exposure import adjust_gamma
from skimage.filters import gaussian
from skimage.transform import ProjectiveTransform
from skimage.transform import resize
from skimage.transform import warp
import tensorflow as tf
import yaml

from keypose import data_pb2 as pb

try:
  import cv2  # pylint: disable=g-import-not-at-top
except ImportError as e:
  print(e)

# Top level keypose directory.
KEYPOSE_PATH = os.path.join(os.getcwd(), 'keypose')


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


# Camera array to camera protobuf
def cam_array_to_pb(cam):
  cam_pb = pb.Camera()
  cam_pb.fx = cam[0]
  cam_pb.fy = cam[1]
  cam_pb.cx = cam[2]
  cam_pb.cy = cam[3]
  cam_pb.baseline = cam[4]
  cam_pb.resx = cam[5]
  cam_pb.resy = cam[6]
  return cam_pb


# Camera protobuf to camera array
def cam_pb_to_array(cam_pb):
  return [
      cam_pb.fx, cam_pb.fy, cam_pb.cx, cam_pb.cy, cam_pb.baseline, cam_pb.resx,
      cam_pb.resy
  ]


# Dummy transform that always produces 0 vector as result.
def dummy_to_uvd():
  to_uvd = np.zeros((4, 4))
  to_uvd[3, 3] = 1.0
  return to_uvd


# Zero uvd stack.
def dummy_keys_uvd(num):
  uvd = np.zeros(4)
  uvd[3] = 1.0
  return [uvd for _ in range(num)]


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


# Returns a dict of the filename strings in a TfSet.
def read_tfset(path):
  data_pb = pb.TfSet()
  read_from_text_file(path, data_pb)
  return data_pb


def make_tfset(train_names, val_names, name):
  tfset_pb = pb.TfSet()
  tfset_pb.train[:] = train_names
  tfset_pb.val[:] = val_names
  tfset_pb.name = name
  return tfset_pb


# Read the contents of a target protobuf.
def read_contents_pb(path):
  return get_contents_pb(read_target_pb(path).kp_target)


# Make a DataParams protobuf.
def make_data_params(resx, resy, num_kp, cam):
  data_pb = pb.DataParams()
  data_pb.camera.CopyFrom(cam)
  data_pb.resx = resx
  data_pb.resy = resy
  data_pb.num_kp = num_kp
  return data_pb


# Read contents of a camera protobuf.
def read_data_params(path):
  data_pb = pb.DataParams()
  read_from_text_file(path, data_pb)
  cam = data_pb.camera
  return data_pb.resx, data_pb.resy, data_pb.num_kp, cam


def write_to_text_file(path, proto):
  with open(path, 'w') as file:
    file.write(str(proto))


def project_np(mat, vec):
  """Projects homogeneous 3D XYZ coordinates to image uvd coordinates."""
  # <vec> has shape [4, N].
  # <mat> has shape [4, 4]
  # Return has shape [4, N].

  p = mat.dot(vec)
  # Using <3:4> instead of <3> preserves shape.
  p = p / (p[3:4, :] + 1.0e-10)
  return p


# Maximum number of projection frames for losses.
MAX_TARGET_FRAMES = 5

# Standard image input parameters for the model.
DEFAULT_PARAMS = """
batch_size: 32
dset_dir: ''
model_dir: ''
steps: 80000

model_params:
model_params:
  use_regress: True  # Use regression or integration.
  batchnorm: [0.999, 1.0e-8, False]
  num_filters: 48  # Number of filters across DCNN.
  filter_size: 3
  max_dilation: 32
  dilation_rep: 1
  dropout: 0.0
  disp_default: 30
  sym: [0]
  input_sym: [0]  # Mix up symmetric ground truth values.
  use_stereo: true
  visible: true  # Use only samples with visible keypoints.

  crop: [180, 120, 30]  # Crop patch size, WxH, plus disp offset.
  dither: 20.0  # Amount to dither the crop, in pixels.

  loss_kp: 1.0
  loss_kp_step: [0, 200]
  loss_prob: 0.001
  loss_proj: 2.5
  loss_proj_step: [10000, 20000]
  loss_reg: 0.01
  loss_dispmap: 1.0
  loss_dispmap_step: [5000, 10000]

  noise: 4.0
  occ_fraction: 0.2
  kp_occ_radius: 0.0   # Radius for occlusion for real-world keypoints.
  blur: [1.0, 4.0]  # Min, max in pixels.
  motion: [0, 0, 0, 0]   # Motion blur, min/max in pixels, min/max in angle (deg).
  gamma: [0.8, 1.2]

  rot: [0.0, 0.0, 0.0]  # In degrees, [X-axis, Y-axis, Z-axis]

  # Homography parameters [X-axis, Y-axis]
  shear: []  # Use X-axis only for stereo [10,0].
  scale: []  # [min, max] applied only on the Y axis.
  flip: []   # Use Y-axis only for stereo [False,True].
"""


def get_params(param_file=None,
               cam_file=None,
               cam_image_file=None,
               defaults=DEFAULT_PARAMS):
  """Returns default or overridden user-specified parameters, and cam params."""

  # param_file points to a yaml string.
  # cam_file points to a camera pbtxt.
  # cam_image_file points to a camera pbtxt.

  params = ConfigParams(defaults)
  if param_file:
    params.merge_yaml(param_file)
  mparams = params.model_params

  cam = None
  if cam_file:
    print('Model camera params file name is: %s' % cam_file)
    resx, resy, num_kp, cam = read_data_params(cam_file)

    mparams.resx = resx
    mparams.resy = resy
    mparams.num_kp = num_kp
    mparams.modelx = resx
    mparams.modely = resy
    mparams.disp_default /= mparams.resx  # Convert to fraction of image.

  if mparams.crop:
    mparams.modelx = int(mparams.crop[0])
    mparams.modely = int(mparams.crop[1])

  cam_image = None
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

  def read_yaml(self, fname):
    with open(fname, 'r') as f:
      ret = self.merge_yaml(f)
    return ret

  def write_yaml(self, fname):
    with open(fname, 'w') as f:
      yaml.dump(self.make_dict(), f, default_flow_style=None)

  def merge_dict(self, p_dict):
    """Merges a dictionary into this params."""
    for k in p_dict:
      val = p_dict[k]
      if k == 'default_file':
        print('Default file:', val)
        fname = os.path.join(KEYPOSE_PATH, val + '.yaml')
        print('Default file fname:', fname)
        self.read_yaml(fname)
        continue
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


# uint8 image [0,255] to float [0,1]
def image_uint8_to_float(image):
  if image.dtype == np.float32:
    return image
  image = image.astype(np.float32) * (1.0 / 255.0)
  image_np = np.clip(image, 0.0, 1.0)
  return image_np


def resize_image(image, cam, cam_image, targs_pb):
  """Resize an image using scaling and cropping; changes kps_pb to correspond."""
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
    scale_cam(targ_pb.camera)
    for kp in targ_pb.keypoints:
      kp.u = (kp.u - cropx) / scale
      kp.v = (kp.v - cropy) / scale
      kp.x = kp.u / nxs
      kp.y = kp.v / nys
      if kp.u < 0 or kp.u >= nxs or kp.v < 0 or kp.v >= nys:
        kp.visible = 0.0

  resize_target(targs_pb.kp_target)
  for targ_pb in targs_pb.proj_targets:
    resize_target(targ_pb)

  return image


def rotation_transform(x_rot, y_rot, z_rot):
  """Creates a rotation transform with rotations around the three axes.

  Args:
    x_rot: rotate around X axis (degrees).
    y_rot: rotate around Y axis (degrees).
    z_rot: rotate around Z axis (degrees).

  Returns:
    4x4 transform.
  """
  x_rot = np.pi * x_rot / 180.0
  xt = np.array([[1.0, 0, 0, 0], [0, np.cos(x_rot), -np.sin(x_rot), 0.0],
                 [0, np.sin(x_rot), np.cos(x_rot), 0.0], [0, 0, 0, 1]])
  y_rot = np.pi * y_rot / 180.0
  yt = np.array([[np.cos(y_rot), 0, np.sin(y_rot), 0], [0, 1, 0, 0],
                 [-np.sin(y_rot), 0, np.cos(y_rot), 0], [0, 0, 0, 1]])
  z_rot = np.pi * z_rot / 180.0
  zt = np.array([[np.cos(z_rot), -np.sin(z_rot), 0, 0],
                 [np.sin(z_rot), np.cos(z_rot), 0, 0], [0, 0, 1, 0],
                 [0, 0, 0, 1]])
  return xt.dot(yt).dot(zt)


# rotation of the camera
def rotate_camera(rotation, image, camera, transform, key_pts):
  """Rotates a camera around its optical axis.

  Args:
    rotation: 4x4 rotation transform, c'_R_c
    image: Camera image, h x w x 4 (or 3).
    camera: 7-element camera parameters (fx, fy, cx, cy, baseline, resx, resy)
    transform: 4x4 transform matrix, w_T_c (camera-to-world).
    key_pts: Nx4 u,v,d,w image keypoints, in pixel coordinates.

  Returns:
    Rotated image, zero-filled
    Updated transform w_T_c' = w_T_c * P * c_R_c' * Q
    updated keypoints u,v,d,w, Nx4
    visibility vector for keypoint, N

  Keypoints are converted by P * c'_R_c * Q
  """

  def in_bounds(pt, bounds):
    if (pt[0] >= 0 and pt[0] <= bounds[0] and pt[1] >= 0 and
        pt[1] <= bounds[1]):
      return 1.0
    else:
      return 0.0

  cam_pb = cam_array_to_pb(camera)
  pmat = p_matrix_from_camera(cam_pb)
  qmat = q_matrix_from_camera(cam_pb)
  tp = np.dot(transform, np.dot(pmat, np.dot(np.linalg.inv(rotation), qmat)))
  key_pts_p = project_np(
      np.dot(pmat, np.dot(rotation, qmat)), np.transpose(key_pts))
  kmat = k_matrix_from_camera(cam_pb)
  hmat = kmat.dot(rotation[:3, :3].transpose()).dot(np.linalg.inv(kmat))
  image_np = np.clip(image, 0.0, 0.9999)
  warped = warp(image_np, ProjectiveTransform(matrix=hmat))
  visible = np.array([
      in_bounds(key_pts_p[:2, i], camera[5:7]) for i in range(key_pts.shape[0])
  ])
  return warped, tp, key_pts_p.transpose(), visible


def warp_homography(res, scale, shear, flip):
  """Returns a homography for image scaling, shear and flip.

  Args:
    res: resolution of the image, [x_res, y_res].
    scale: scale factor [x_scale, y_scale].
    shear: shear in [x_deg, y_deg].
    flip: boolean [x_flip, y_flip].
  """
  center_mat = np.eye(3)
  center_mat[0, 2] = -res[0] / 2.0
  center_mat[1, 2] = -res[1] / 2.0
  cmat_inv = np.linalg.inv(center_mat)

  flip_mat = np.eye(3)
  if flip[0]:
    flip_mat[0, 0] = -1
  if flip[1]:
    flip_mat[1, 1] = -1

  shear_mat = np.eye(3)
  shear_mat[0, 1] = math.tan(math.radians(shear[0]))
  shear_mat[1, 0] = math.tan(math.radians(shear[1]))

  scale_mat = np.eye(3)
  scale_mat[0, 0] = scale[0]
  scale_mat[1, 1] = scale[1]

  return cmat_inv.dot(scale_mat.dot(shear_mat.dot(flip_mat.dot(center_mat))))


def do_rotation(image, image2, transform, camera, key_pts, visible, rotation):
  """Add a random rotation about the camera centerpoint.

  Args:
    image: HxWx4 image, left image if stereo; can be uint8 or float.
    image2: HxWx4 image, right image if stereo, or None
    transform: 4x4 to_world transform.
    camera: 7-element camera parameters.
    key_pts: Nx4 uvdw keypoints.
    visible: Visibility prediate for keypoints.
    rotation: Rotation as 3-tuple, XYZ axes.

  Returns:
    image: Warped by random rotation, float32.
    transform: Updated to_world transform.
    key_pts: updated uvdw keypoints.
    visible: visibility vector for the keypoints.
  """
  image = image_uint8_to_float(image)
  image2 = image_uint8_to_float(image2)
  area, _ = get_area(image)
  if area < 10:  # Something is wrong here.
    return image, image2, transform, key_pts, visible

  rotation = (float(rotation[0]), float(rotation[1]), float(rotation[2]))
  while True:
    rot = rotation_transform(
        random.uniform(-rotation[0], rotation[0]),
        random.uniform(-rotation[1], rotation[1]),
        random.uniform(-rotation[2], rotation[2]))
    image_p, transform_p, key_pts_p, visible_p = rotate_camera(
        rot, image, camera, transform, key_pts)
    area_p, _ = get_area(image)
    if float(area_p) / area > 0.6:
      if image2 is not None:
        image2_p, _, _, _ = rotate_camera(rot, image2, camera, transform,
                                          key_pts)
      else:
        image2_p = image_p
      break

  # Warp function converts images to float64, this converts back.
  return (image_p.astype(np.float32), image2_p.astype(np.float32),
          transform_p.astype(np.float32), key_pts_p.astype(np.float32),
          visible_p.astype(np.float32))


def do_2d_homography(image, image2, scale, shear, flip, mirrored, split):
  """Add random 2D transforms to input images.

  Images are warped according to the 2D transforms of scaling,
  shear and flip.  The 2D homogenous transform inverse is returned,
  so that keypoints can be adjusted after they are predicted.
  Transforms that preserve horizontal epipolar lines are vertical flip,
  X-axis shear, mirroring, and scaling.
  TODO: visibility analysis.

  Args:
    image: HxWx4 image, left image if stereo; can be uint8 or float.
    image2: HxWx4 image, right image if stereo, or None
    scale: floating point bounds, uniform random scale, e.g., [0.8, 1.2].
    shear: x,y shear max bounds for uniform random shear, in degrees.
    flip: [True, True] if images are randomly flipped horizontal, vertical.
    mirrored: True if images are mirrored.
    split: train / eval split.

  Returns:
    image: Warped by random transform, float32.
    image2: Warped by same random transform, float32.
    homography: 3x3 homography matrix for the warp.
  """
  if (not mirrored) and not (split == 'train' and (scale or shear or flip)):
    return image, image2, np.eye(3, dtype=np.float32)

  if not scale:
    scale = [1.0, 1.0]
  if not shear:
    shear = [0.0, 0.0]
  if not flip:
    flip = [False, False]
  image = image_uint8_to_float(image)
  image2 = image_uint8_to_float(image2)
  if mirrored:
    flip = [False, True]
  else:
    flip = [random.choice([False, flip[0]]), random.choice([False, flip[1]])]
  hom = warp_homography((image.shape[1], image.shape[0]), [
      1.0, random.uniform(scale[0], scale[1])
  ], [random.uniform(-shear[0], shear[0]),
      random.uniform(-shear[1], shear[1])], flip)
  if np.allclose(hom, np.eye(3)):
    return image, image2, np.eye(3, dtype=np.float32)
  hom_inv = np.linalg.inv(hom)
  image_p = warp_2d(image, hom_inv)
  image2_p = warp_2d(image2, hom_inv)
  # Warp function converts images to float64, this converts back.
  return (image_p.astype(np.float32), image2_p.astype(np.float32),
          hom_inv.astype(np.float32))


def warp_2d(image, hom):
  image_np = np.clip(image, 0.0, 0.9999)
  warped = warp(image_np, ProjectiveTransform(matrix=hom))
  return warped


# Returns a 2D gaussian centered on mean (pix coords, float) with
# variance var (pixels, float).
def gauss2d(mean, var, size):
  x = np.arange(0, size[0])
  y = np.arange(0, size[1])
  x, y = np.meshgrid(x, y)
  mx, my = mean
  vx, vy = var
  return np.float32(1. / (2. * np.pi * vx * vy) *
                    np.exp(-((x - mx)**2. / (2. * vx**2.) + (y - my)**2. /
                             (2. * vy**2.))))


# Normalize so the peak is 1.0.
def norm_gauss2d(mean, var, size):
  g2d = gauss2d(mean, var, size)
  m = np.max(g2d)
  if m <= 0.0:
    return g2d
  return g2d * (1.0 / np.max(g2d))


# Make an inverse gaussian for exclusion of a prob field.
def inv_gauss(mean, var, size):
  g = gauss2d(mean, var, size)
  m = np.max(g)
  g_inv = (m - g) * (1.0 / m)
  return g_inv


def project_uvd(mat, uvd, offset):
  uvw = np.array([uvd[0] - offset[0], uvd[1] - offset[1], 1.0])
  uvwt = mat.dot(uvw)
  return uvwt[:2] / (uvwt[2] + 1e-10)


def do_vertical_flip(image, image2, mirrored):
  """Flip image vertically.

  The 2D homogenous transform inverse is returned,
  so that keypoints can be adjusted after they are predicted.

  Args:
    image: HxWx4 image, left image if stereo; can be uint8 or float.
    image2: HxWx4 image, right image if stereo, or None.
    mirrored: True if image is mirrored.

  Returns:
    image: flipped vertically.
    image2: flipped vertically.
    homography: 3x3 homography matrix for vertical flip.
  """
  if not mirrored:
    return image, image2, np.eye(3, dtype=np.float32)
  image = image_uint8_to_float(image)
  image2 = image_uint8_to_float(image2)
  image_p = np.flipud(image)
  image2_p = np.flipud(image2)
  hom = warp_homography(
      (image.shape[1], image.shape[0]),
      [1.0, 1.0],  # Scale (none).
      [0.0, 0.0],  # Shear (none).
      [False, True])
  return image_p, image2_p, np.linalg.inv(hom).astype(np.float32)


# Returns gaussian 2Ds with shape [num_kpts, h, w].
# keys_uvd has shape [num_kps, 4]
def do_spatial_prob(keys_uvd, hom, offset, var, size):
  hom_inv = np.linalg.inv(hom)
  uvs = [project_uvd(hom_inv, uvd, offset) for uvd in keys_uvd]
  probs = [inv_gauss(uv, [var, var], size) for uv in uvs]
  return np.array(probs, dtype=np.float32)


# Largest fraction of object that can be occluded.
def do_occlude(image, image2, occ_fraction=0.0):
  """Add an elliptical occlusion to the RGBA image.

  Args:
    image: RGBA images with A channel @ 255 for valid object.
    image2: RGBA images, right stereo.
    occ_fraction: fraction of image to be occluded.

  Returns:
    Modified image.
    Modifies image in place.
  """
  area, inds = get_area(image)
  if area < 50:
    return image, image2
  radius = 2.0 * np.sqrt(area / np.pi) * occ_fraction

  for _ in range(0, 1):
    i = random.randint(0, area - 1)
    ind = (inds[0][i], inds[1][i])
    rr, cc = ellipse(
        ind[0],
        ind[1],
        random.uniform(1.0, radius),
        random.uniform(1.0, radius),
        shape=image.shape[0:2],
        rotation=random.uniform(-1.0, 1.0) * np.pi)
    image[rr, cc, 3] = 0
    image2[rr, cc, 3] = 0

  return image, image2


def do_motion_blur(image, distance, angle):
  """Random fake motion blur by compositing in the x,y plane of the image.

  Args:
    image: tensor of images, floating point [0,1], 3 or 4 channels.
    distance: how far to move, in pixels <min, max>.
    angle: how much to rotate, in degrees <min, max>.

  Returns:
    Updated image with motion blur.
  """
  if not distance[1]:  # No blur.
    return image
  dist = random.uniform(*distance) * 0.5
  ang = math.radians(random.uniform(*angle))
  x, y = dist * math.cos(ang), dist * math.sin(ang)
  rows, cols = image.shape[:2]
  im = np.array([[1, 0, x], [0, 1, y]])
  im1 = cv2.warpAffine(image, im, (cols, rows))
  im = np.array([[1, 0, 2 * x], [0, 1, 2 * y]])
  im2 = cv2.warpAffine(image, im, (cols, rows))
  im = np.array([[1, 0, -x], [0, 1, -y]])
  im3 = cv2.warpAffine(image, im, (cols, rows))
  im = np.array([[1, 0, -2 * x], [0, 1, -2 * y]])
  im4 = cv2.warpAffine(image, im, (cols, rows))
  im = (image + im1 + im2 + im3 + im4) * 0.2
  np.clip(im, 0.0, 1.0, im)
  return im


def do_composite(image, bg_fname, sigma, motion, noise, gamma):
  """Composite a background image onto the foreground.

  Args:
    image: original image, floating point [0,1].
    bg_fname: background image file name, None or empty string if none.
    sigma: blur in pixels; single value or range.
    motion: 4-tuple <min_pix_move, max_pix_move, min_deg_angle, max_deg angle>.
    noise: pixel noise in the range 0-255, either single value or range.
    gamma: gamma correction to be applied to image, 0 for none.

  Returns:
    Updated image.
  """

  def make_random(x):
    # Arg x can be list, tuple, numpy.ndarray
    if isinstance(x, list) or isinstance(x, tuple):
      x = np.array(x)
    assert isinstance(
        x, np.ndarray), 'Argument to do_composite must be list or array'
    if x.size == 0:
      return None
    elif x.size == 1:
      return x[0]
    else:
      return random.uniform(x[0], x[1])

  if motion[1]:
    image = do_motion_blur(image, motion[:2], motion[2:])

  ys, xs, _ = image.shape
  if bg_fname:
    scene = read_image(bg_fname) * (1.0 / 255.0)
    if random.choice([True, False]):
      scene = np.flipud(scene)
    if random.choice([True, False]):
      scene = np.fliplr(scene)
    yss, xss = scene.shape[0], scene.shape[1]
    assert yss > ys, 'Background image must be larger than training image'
    assert xss > xs, 'Background image must be larger than training image'

  # Adjust gamma of image.
  gamma = make_random(gamma)
  if gamma is not None:
    image[:, :, :3] = adjust_gamma(image[:, :, :3], gamma)
  # Add noise to object.
  noise = make_random(noise)
  if noise is not None:
    image[:, :, :3] += np.random.randn(ys, xs, 3) * noise * 0.5 / 255.
  np.clip(image, 0.0, 1.0, image)

  # Cut out ellipse where image alpha is 0.
  if bg_fname:
    ul_y = random.randint(0, yss - ys - 1)
    ul_x = random.randint(0, xss - xs - 1)
    scene_crop = scene[ul_y:ul_y + ys, ul_x:ul_x + xs, :]
    mask = image[:, :, 3]
    rgbmask = np.stack([mask, mask, mask], axis=2)
    image[:, :, :3] = scene_crop * (1.0 - rgbmask) + image[:, :, :3] * rgbmask
  else:
    image[image[:, :, 3] == 0, 0:3] = 0.5

  # Add gaussian blur and noise.
  sigma = make_random(sigma)
  im = np.copy(image[:, :, :3])  # Need copy to preserve float32
  gaussian(image[:, :, :3], sigma, multichannel=True, output=im)
  # Add noise to whole scene, after blur.
  if noise is not None:
    im[:, :, :3] += np.random.randn(ys, xs, 3) * noise * 0.5 / 255.
  np.clip(im, 0.0, 1.0, im)

  return im


# Returns area in pixels.
# Works for both uint8 and float32 images.
def get_area(image):
  inds = np.nonzero(image[:, :, 3] > 0)
  area = inds[0].shape[0]
  return area, inds


def do_occlude_crop(image,
                    image2,
                    key_pts,
                    key_pts_r,
                    crop,
                    visible,
                    dither,
                    var_offset=False):
  """Crop area around the object.

  Crop is [W, H, R]', where 'R' is right-disparity offset; or else [].
  Images can be either floating-point or uint8.

  Args:
    image: left image.
    image2: right image.
    key_pts: left keypoints.
    key_pts_r: right keypoints.
    crop: crop is [W, H, R]', where 'R' is right-disparity offset; or else [].
    visible: visibility status of keypoints, modified by function.
    dither: amount to dither the crop.
    var_offset: vary the offset between left and right images.

  Returns:
    image: cropped left image.
    image2: cropped right image.
    offset: offset of the crop in the original image.
    visible: visibility status of the keypoints.
  """

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
    dist = (
        np.sum(np.square(existing_set), axis=1, keepdims=True) +
        np.sum(np.square(rest_set.T), axis=0, keepdims=True) -
        np.dot(existing_set, rest_set.T) * 2)
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
