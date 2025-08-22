# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Dataset for forwarding facing scene in IBRNet/NeRF with reference view."""

import os
from os import path
import imageio

import numpy as np

from gen_patch_neural_rendering.src.datasets.ff_epipolar import FFEpipolar
from gen_patch_neural_rendering.src.utils import file_utils
from gen_patch_neural_rendering.src.utils import pose_utils


class EvalIBREpipolar(FFEpipolar):
  """Forward Facing epipolar dataset."""

  def _load_renderings(self, args):
    """Load images and camera information."""

    #-------------------------------------------
    # Load images.
    #-------------------------------------------
    basedir = path.join(args.dataset.eval_llff_dir, self.scene)
    img0 = [
        os.path.join(basedir, "images", f)
        for f in sorted(file_utils.listdir(os.path.join(basedir, "images")))
        if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
    ][0]
    with file_utils.open_file(img0) as f:
      sh = imageio.imread(f).shape
    if sh[0] / sh[
        1] != args.dataset.ff_image_height / args.dataset.ff_image_width:
      raise ValueError("not expected height width ratio")

    factor = sh[0] / args.dataset.ff_image_height

    sfx = "_4"
    imgdir = os.path.join(basedir, "images" + sfx)
    if not file_utils.file_exists(imgdir):
      imgdir = os.path.join(basedir, "images")
      if not file_utils.file_exists(imgdir):
        raise ValueError("{} does not exist".format(imgdir))

    images = self._load_images(imgdir, args.dataset.ff_image_width,
                               args.dataset.ff_image_height)

    #-------------------------------------------
    # Load poses and bds.
    #-------------------------------------------
    with file_utils.open_file(path.join(basedir, "poses_bounds.npy"),
                              "rb") as fp:
      poses_arr = np.load(fp)

    self.cam_transform = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0],
                                   [0, 0, 0, 1]])

    self.cam_transform_3x3 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    # poses_arr contains an array consisting of a 3x4 pose matrices and
    # 2 depth bounds for each image. The pose matrix contain [R t] as the
    # left 3x4 matrix
    # pose_arr has shape (...,14) {3x4 + 2}
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])

    # Convert R matrix from the form [down right back] to [right up back]
    poses = np.concatenate(
        [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)

    # Transpose such that the first dimension is number of images
    images = np.moveaxis(images, -1, 0)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    if args.dataset.normalize:
      scale = 1. / bds.max()
    else:
      scale = 1. / (bds.min() * .75)

    poses[:, :3, 3] *= scale
    bds *= scale
    poses_copy = poses.copy()
    poses_copy = pose_utils.recenter_poses(poses, None)
    poses = pose_utils.recenter_poses(poses, self.cam_transform)

    # Get the min and max depth of the scene
    self.min_depth = np.array([bds.min()])
    self.max_depth = np.array([bds.max()])

    # Use this to set the near and far plane
    args.model.near = self.min_depth.item()
    args.model.far = self.max_depth.item()

    if self.split == "test":
      self.render_poses = pose_utils.generate_spiral_poses(
          poses_copy, bds, self.cam_transform)

    i_test = np.arange(images.shape[0])[::args.dataset.llffhold]
    i_train = np.array(
        [i for i in np.arange(int(images.shape[0])) if i not in i_test])

    if self.split == "train":
      indices = i_train
    else:
      indices = i_test
    images = images[indices]
    poses = poses[indices]

    self.images = images
    self.camtoworlds = poses[:, :3, :4]

    # intrinsic arr has H, W, fx, fy, cx, cy
    self.focal = poses[0, -1, -1] * 1. / factor
    self.h, self.w = images.shape[1:3]
    self.resolution = self.h * self.w

    if args.dataset.render_path and self.split == "test":
      self.n_examples = self.render_poses.shape[0]
    else:
      self.n_examples = images.shape[0]

    self.intrinsic_matrix = np.array([[self.focal, 0, (self.w / 2), 0],
                                      [0, self.focal, (self.h / 2), 0],
                                      [0, 0, 1, 0]]).astype(np.float32)
