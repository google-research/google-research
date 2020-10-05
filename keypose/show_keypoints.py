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

"""From a directory of images and keypoint protobufs, shows images, with kps.

Invocation:
  $ python -m code/show_keypoints <image_dir> [<mesh_file>].
"""

import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from keypose import utils

try:
  import cv2  # pylint: disable=g-import-not-at-top
except ImportError as e:
  print(e)

colors = 255 * plt.cm.get_cmap('rainbow')(np.linspace(0, 1.0, 10))[:, :3]


def draw_circle(image, uvd, color, size=3):
  # Filled color circle.
  cv2.circle(image, (int(uvd[0]), int(uvd[1])), size, color, -1)
  # White outline.
  cv2.circle(image, (int(uvd[0]), int(uvd[1])), size + 1, (255, 255, 255))


def show_keypoints(image_dir, mesh_file):
  """Display keypoints in a keypose data file."""
  print('Looking for images in %s' % image_dir)
  filenames = glob.glob(os.path.join(image_dir, '*_L.png'))
  if not filenames:
    print("Couldn't find any PNG files in %s" % image_dir)
    exit(-1)
  filenames.sort()
  print('Found %d files in %s' % (len(filenames), image_dir))

  obj = None
  if mesh_file:
    obj = utils.read_mesh(mesh_file)

  for fname in filenames:
    im_l = utils.read_image(fname)
    im_r = utils.read_image(fname.replace('_L.png', '_R.png'))
    im_mask = utils.read_image(fname.replace('_L.png', '_mask.png'))
    im_border = utils.read_image(fname.replace('_L.png', '_border.png'))
    cam, _, _, uvds, _, _, transform = utils.read_contents_pb(
        fname.replace('_L.png', '_L.pbtxt'))
    print(fname)
    ret = show_kps(im_l, im_r, im_mask, im_border,
                   (cam, uvds, transform), obj)
    if ret:
      break


def show_kps(im_l, im_r, im_border, im_mask, kps, obj=None, size=3):
  """Draw left/right images and keypoints using OpenCV."""
  cam, uvds, _ = kps

  im_l = cv2.cvtColor(im_l, cv2.COLOR_BGR2RGB)
  im_r = cv2.cvtColor(im_r, cv2.COLOR_BGR2RGB)

  uvds = np.array(uvds)
  for i, uvd in enumerate(uvds):
    draw_circle(im_l, uvd, colors[i * 3], size)

  if obj:
    p_matrix = utils.p_matrix_from_camera(cam)
    q_matrix = utils.q_matrix_from_camera(cam)
    xyzs = utils.project_np(q_matrix, uvds.T)
    obj.project_to_uvd(xyzs, p_matrix)
    im_l = obj.draw_points(im_l)

  cv2.imshow('Image Left', im_l)
  cv2.imshow('Border', im_border)
  cv2.imshow('Mask', im_mask)

  key = cv2.waitKey()
  if key == ord('q'):
    return True
  return False


def main():
  image_dir = sys.argv[1]
  if len(sys.argv) < 3:
    mesh_file = None
  else:
    mesh_file = sys.argv[2]

  show_keypoints(image_dir, mesh_file)


if __name__ == '__main__':
  main()
