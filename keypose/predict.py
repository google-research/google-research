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

"""Pose estimation from keypoints using models.

Given a 2D image or stereo pair, predict a set of 3D keypoints that
match the target examples.

The <model_dir> should have a saved_model directory with
<saved_model.pb> and <variables>
"""

import glob
import os
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
import tensorflow as tf

from keypose import utils

try:
  import cv2  # pylint: disable=g-import-not-at-top
except ImportError as e:
  print(e)


def image_as_ubyte(image):
  with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    return ski.img_as_ubyte(image)


def draw_circle(image, uvd, color, size=2):
  # Filled color circle.
  cv2.circle(image, (int(uvd[0]), int(uvd[1])), size, tuple(color), -1)
  # White outline.
  cv2.circle(image, (int(uvd[0]), int(uvd[1])), size + 1, (255, 255, 255))


def predict(model_dir, image_dir, params, camera, camera_input, mesh_file=None):
  """Predicts keypoints on all images in dset_dir, evaluates 3D accuracy."""

  # Set up network for prediction.
  mparams = params.model_params
  num_kp = mparams.num_kp

  # Set up predictor from model directory.
  model = tf.saved_model.load(model_dir)
  predict_fn = model.signatures['serving_default']

  colors = plt.cm.get_cmap('rainbow')(np.linspace(0, 1.0, num_kp))[:, :3]
  colors = (colors * 255).tolist()

  # Set up mesh object, if the mesh file exists.
  obj = None
  if mesh_file and mesh_file != 'show':
    obj = utils.read_mesh(mesh_file, num=300)
    obj.large_points = False

  # Iterate over all images in image_dir.
  total_time = 0.0
  count = 0
  mae_list = []

  filenames = glob.glob(os.path.join(image_dir, '*_L.png'))
  filenames.sort()
  for fname in filenames:
    # Read in and resize images if necessary.
    print(fname)
    im_l = utils.read_image(fname)
    kps_pb = utils.read_keys_pb(fname.replace('_L.png', '_L.pbtxt'))
    im_l = utils.resize_image(im_l, camera, camera_input, kps_pb)
    keys_uvd_l, to_world_l, visible_l = utils.get_keypoints(kps_pb)

    im_r = utils.read_image(fname.replace('_L.png', '_R.png'))
    kps_pb = utils.read_keys_pb(fname.replace('_L.png', '_R.pbtxt'))
    im_r = utils.resize_image(im_r, camera, camera_input, kps_pb)
    keys_uvd_r, _, _ = utils.get_keypoints(kps_pb)

    # Do cropping if called out in mparams.
    if mparams.crop:
      img0, img1, offs, visible = utils.do_occlude_crop(
          im_l,
          im_r,
          keys_uvd_l,
          keys_uvd_r,
          mparams.crop,
          visible_l,
          dither=0.0,
          var_offset=False)
      if np.any(visible == 0.0):
        print('Could not crop')
        continue
      offsets = offs.astype(np.float32)
    hom = np.eye(3, dtype=np.float32)
    to_world_l = to_world_l.astype(np.float32)
    keys_uvd_l = keys_uvd_l.astype(np.float32)

    # Batch size of 1.
    img_l = tf.constant(np.expand_dims(img0[:, :, :3], 0))
    img_r = tf.constant(np.expand_dims(img1[:, :, :3], 0))
    to_world_l = tf.constant(np.expand_dims(to_world_l, 0))
    keys_uvd_l = tf.constant(np.expand_dims(keys_uvd_l, 0))
    offsets = tf.constant(np.expand_dims(offsets, 0))
    hom = tf.constant(np.expand_dims(hom, 0))
    labels = {}
    labels['keys_uvd_L'] = keys_uvd_l
    labels['to_world_L'] = to_world_l

    # Now do the magic.
    t0 = time.time()
    preds = predict_fn(
        img_L=img_l,
        img_R=img_r,
        to_world_L=to_world_l,
        offsets=offsets,
        hom=hom)

    if count > 0:  # Ignore first time, startup is long.
      total_time += time.time() - t0
    count += 1

    xyzw = tf.transpose(preds['xyzw'], [0, 2, 1])

    mae_3d = utils.world_error(labels, xyzw).numpy()
    mae_list.append(mae_3d)
    print('mae_3d:', mae_3d)
    uvdw = preds['uvdw'][0, Ellipsis].numpy()
    xyzw = xyzw[0, Ellipsis].numpy()
    offsets = offsets[0, Ellipsis].numpy()

    # uv_pix_raw is in the coords of the cropped image used by the model.
    uv_pix_raw = preds['uv_pix_raw'][0, Ellipsis].numpy()  # [num_kp, 3]

    img_l = image_as_ubyte(img_l[0, Ellipsis].numpy())
    img_r = image_as_ubyte(img_r[0, Ellipsis].numpy())
    iml_orig = cv2.resize(img_l, None, fx=2, fy=2)
    imr_orig = cv2.resize(img_r, None, fx=2, fy=2)
    if obj:
      p_matrix = utils.p_matrix_from_camera(camera)
      q_matrix = utils.q_matrix_from_camera(camera)
      xyzw_cam = utils.project_np(q_matrix, uvdw.T)
      obj.project_to_uvd(xyzw_cam, p_matrix)
      im_mesh = np.array(img_l)
      im_mesh = obj.draw_points(im_mesh, offsets)
      im_mesh = cv2.resize(im_mesh, None, fx=2, fy=2)
    else:
      im_mesh = np.zeros(iml_orig.shape, dtype=np.uint8)

    for i in range(uv_pix_raw.shape[0]):
      draw_circle(img_l, uv_pix_raw[i, :2], colors[i])
    im_kps = cv2.resize(img_l, None, fx=2, fy=2)
    im_large = cv2.vconcat(
        [cv2.hconcat([iml_orig, imr_orig]),
         cv2.hconcat([im_kps, im_mesh])])
    if mesh_file:
      cv2.imshow('Left and right images; keypoints and mesh', im_large)
      key = cv2.waitKey()
      if key == ord('q'):
        break

  print('Total time: %.1f for %d inferences, %.1f ms/inf' %
        (total_time, count, total_time * 1000.0 / count))
  mae_list = np.concatenate(mae_list).flatten()
  print('MAE 3D (m): %f' % np.mean(mae_list))


def main(argv):
  if not len(argv) >= 3:
    print('Usage: ./predict.py <model_dir> <image_dir> [show|<object>]')
    exit(0)
  model_dir = argv[1]
  image_dir = argv[2]
  mesh_file = None
  if len(argv) > 3:
    mesh_file = argv[3]
  fname = os.path.join(model_dir, 'params.yaml')
  with open(fname, 'r') as f:
    params, camera_model, camera_image = utils.get_params(
        param_file=f,
        cam_file=os.path.join(model_dir, 'data_params.pbtxt'),
        cam_image_file=os.path.join(image_dir, 'data_params.pbtxt'))
  predict(
      model_dir,
      image_dir,
      params,
      camera_model,
      camera_image,
      mesh_file=mesh_file)


if __name__ == '__main__':
  main(sys.argv)
