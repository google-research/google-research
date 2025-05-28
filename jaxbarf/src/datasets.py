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

"""Different datasets implementation plus a general port for all the datasets."""
import json
import os
from os import path
import pickle
import queue
import threading
from absl import logging
import cv2
import jax
import numpy as np
from PIL import Image
from jaxbarf.src import camera_numpy as camera
from jaxbarf.src import utils


def get_dataset(split, args, train_mode=True, calib_matrix=None):
  """Get dataset."""
  return dataset_dict[args.dataset](split, args, train_mode, calib_matrix)

class Dataset(threading.Thread):
  """Dataset Base Class."""

  def __init__(self,
               split,
               args,
               train_mode=True,
               calib_matrix=None):
    """train_mode: True: randomly sample rays; False: use the whole image"""
    super(Dataset, self).__init__()
    self.queue = queue.Queue(3)  # Set prefetch buffer to 3 batches.
    self.daemon = True
    self.use_pixel_centers = args.use_pixel_centers
    self.split = split
    self.train_mode = train_mode
    self.calib_matrix = calib_matrix  # calib the pose of test cameras
    if split == "train":
      if train_mode:
        self._train_init(args)
      else:  # test/eval mode on train set
        self._test_init(args)
    elif split == "test":
      self._test_init(args)
    else:
      raise ValueError(
          "the split argument should be either \"train\" or \"test\", set"
          "to {} here.".format(split))
    self.batch_size = args.batch_size // jax.host_count()
    self.batching = args.batching
    self.render_path = args.render_path
    self.start()

  def __iter__(self):
    """Iter."""
    return self

  def __next__(self):
    """Get the next training batch or test example.

    Returns:
      batch: dict, has "pixels" and "rays".
    """
    x = self.queue.get()
    if self.split == "train" and self.train_mode:
      return utils.shard(x)
    else:
      return utils.to_device(x)

  def peek(self):
    """Peek at the next training batch or test example without dequeuing it.

    Returns:
      batch: dict, has "pixels" and "rays".
    """
    x = self.queue.queue[0].copy()  # Make a copy of the front of the queue.
    if self.split == "train" and self.train_mode:
      return utils.shard(x)
    else:
      return utils.to_device(x)

  def run(self):
    """Run."""
    if self.split == "train" and self.train_mode:
      next_func = self._next_train
    else:
      next_func = self._next_test
    while True:
      self.queue.put(next_func())

  @property
  def size(self):
    """Size."""
    return self.n_examples

  def _train_init(self, args):
    """Initialize training."""
    self._load_renderings(args)
    self._generate_rays()

    if args.batching == "all_images":
      self.images = self.images.reshape([-1, 3])
      self.rays = utils.namedtuple_map(lambda r: r.reshape((-1,)+r.shape[3:]),
                                       self.rays)
    elif args.batching == "single_image" or args.batching == "every_image":
      self.images = self.images.reshape([-1, self.resolution, 3])
      self.rays = utils.namedtuple_map(
          lambda r: r.reshape((-1, self.resolution)+ r.shape[3:]), self.rays)
    else:
      raise NotImplementedError(
          f"{args.batching} batching strategy is not implemented.")

  def _test_init(self, args):
    """Test init."""
    self._load_renderings(args)
    self._generate_rays()
    self.it = 0

  def _next_train(self):
    """Sample next training batch."""
    if self.batching == "all_images":
      ray_indices = np.random.randint(0, self.rays[0].shape[0],
                                      (self.batch_size,))
      batch_pixels = self.images[ray_indices]
      batch_rays = utils.namedtuple_map(lambda r: r[ray_indices], self.rays)
    elif self.batching == "single_image":
      image_index = np.random.randint(0, self.n_examples, ())
      ray_indices = np.random.randint(0, self.rays[0][0].shape[0],
                                      (self.batch_size,))
      batch_pixels = self.images[image_index][ray_indices]
      batch_rays = utils.namedtuple_map(lambda r: r[image_index][ray_indices],
                                        self.rays)
    elif self.batching == "every_image":
      # Sample rays from all images, so that poses of all images are optimized
      # per iter.
      rays_per_example = self.batch_size // self.n_examples
      ray_indices = np.random.randint(0, self.rays[0][0].shape[0],
                                      (rays_per_example,))
      batch_pixels = self.images[:, ray_indices]
      batch_pixels = batch_pixels.reshape([-1, 3])
      batch_rays = utils.namedtuple_map(lambda r: r[:, ray_indices],
                                        self.rays)
      batch_rays = utils.namedtuple_map(lambda r: r.reshape((-1,)+r.shape[2:]),
                                       batch_rays)

      rays_remain = self.batch_size % self.n_examples
      image_index = np.random.randint(0, self.n_examples, ())
      ray_indices = np.random.randint(0, self.rays[0][0].shape[0],
                                      (rays_remain,))
      remain_pixels = self.images[image_index][ray_indices]
      remain_rays = utils.namedtuple_map(
          lambda r: r[image_index][ray_indices][None], self.rays)
      remain_rays = utils.namedtuple_map(
          lambda r: r.reshape((-1,)+r.shape[2:]), remain_rays)
      batch_pixels = np.concatenate((batch_pixels, remain_pixels), axis=0)
      batch_rays = utils.mergetuple(batch_rays, remain_rays)
    else:
      raise NotImplementedError(
          f"{self.batching} batching strategy is not implemented.")
    return {"pixels": batch_pixels, "rays": batch_rays}

  def _next_test(self):
    """Sample next test example."""
    idx = self.it
    self.it = (self.it + 1) % self.n_examples

    if self.render_path:
      return {"rays": utils.namedtuple_map(lambda r: r[idx], self.render_rays)}
    else:
      return {
          "pixels": self.images[idx],
          "rays": utils.namedtuple_map(lambda r: r[idx], self.rays)
      }

  def get_all_poses(self):
    return {"poses_gt": self.poses_gt, "poses_init": self.poses_init}

  def _generate_rays(self):
    """Generating rays for all images."""
    # pixel_center = 0.5 if self.use_pixel_centers else 0.0
    pixel_center = 0.5
    x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
        np.arange(self.w, dtype=np.float32) + pixel_center,  # X-Axis (columns)
        np.arange(self.h, dtype=np.float32) + pixel_center,  # Y-Axis (rows)
        indexing="xy")
    camera_dirs = np.stack([(x - self.w * 0.5) / self.focal,  # (W,H,3)
                            (y - self.h * 0.5) / self.focal, np.ones_like(x)],
                           axis=-1)
    directions = ((camera_dirs[None, Ellipsis, None, :] *  # (1,W,H,1,3)
                   self.camtoworlds[:, None, None, :3, :3]).sum(
                       axis=-1))  # (N,1,1,3,3)
    origins = np.broadcast_to(
        self.camtoworlds[:, None, None, :3, -1],  # (N,1,1,3)
        directions.shape)  # (N,W,H,3)
    viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

    # Populate the rays with (intr,camtoworlds, camera_dirs, imageid)
    n_camera_dirs = np.broadcast_to(
        camera_dirs[None], directions.shape)  # (N,W,H,3)
    imageids = np.arange(len(self.camtoworlds))  # (N,)
    n_image_ids = np.broadcast_to(
        imageids[:, None, None, None], directions[:, :, :, :1].shape)
    poses_gt = np.broadcast_to(self.poses_gt[:, None, None, Ellipsis], (
        len(self.poses_gt), self.w, self.h)+self.poses_gt.shape[-2:])
    poses_init = np.broadcast_to(self.poses_init[:, None, None, Ellipsis],
                                 (len(self.poses_init),
                                  self.w, self.h)+self.poses_init.shape[-2:])
    # Now every field in below has the shape (N,W,H,3)
    self.rays = utils.Rays(
        origins=origins, directions=directions, viewdirs=viewdirs,
        poses_gt=poses_gt, poses_init=poses_init,
        camera_dirs=n_camera_dirs, imageids=n_image_ids,
    )

class Blender(Dataset):
  """Blender Dataset."""

  def _load_renderings(self, args):
    """Load images from disk."""
    if args.render_path:
      raise ValueError("render_path cannot be used for the blender dataset.")

    if self.split == "test":
      json_file = "transforms_val.json"
    else:
      json_file = "transforms_train.json"
    with utils.open_file(path.join(args.data_dir, json_file), "r") as fp:
      meta = json.load(fp)
    images = []
    cams = []
    for i in range(len(meta["frames"])):
      frame = meta["frames"][i]
      fname = os.path.join(args.data_dir, frame["file_path"] + ".png")
      with utils.open_file(fname, "rb") as imgin:
        image = np.array(Image.open(imgin), dtype=np.float32) / 255.
        """
        if args.factor == 2:
          [halfres_h, halfres_w] = [hw // 2 for hw in image.shape[:2]]
          image = cv2.resize(
              image, (halfres_w, halfres_h), interpolation=cv2.INTER_AREA)
        elif args.factor > 0:
          raise ValueError("Blender dataset only supports factor=0 or 2, {} "
                           "set.".format(args.factor))
        """
        # hard code the image resolution to ensure consistency with prior works
        image = cv2.resize(image, (400, 400), interpolation=cv2.INTER_AREA)
      cams.append(np.array(frame["transform_matrix"], dtype=np.float32))
      images.append(image)
    self.images = np.stack(images, axis=0)
    if args.white_bkgd:
      self.images = (
          self.images[Ellipsis, :3] * self.images[Ellipsis, -1:] +
          (1. - self.images[Ellipsis, -1:]))
    else:
      self.images = self.images[Ellipsis, :3]
    self.h, self.w = self.images.shape[1:3]  # after resizing
    self.resolution = self.h * self.w  # after resizing
    self.camtoworlds = np.stack(cams, axis=0)[:, :3, :4]
    # flip and invert the camtoworlds to get the worldtocams poses following
    # pytorch-barf
    pose_flip = camera.define_pose(rot=np.diag([1, -1, -1]))
    self.camtoworlds = camera.compose([pose_flip, self.camtoworlds])
    self.poses_gt = camera.invert_pose(self.camtoworlds)
    if self.calib_matrix is not None:
      self.poses_gt=camera.refine_test_cameras(self.poses_gt, self.calib_matrix)
      self.camtoworlds = camera.invert_pose(self.poses_gt)
    camera_angle_x = float(meta["camera_angle_x"])
    self.focal = .5 * self.w / np.tan(.5 * camera_angle_x)
    self.n_examples = self.images.shape[0]
    assert len(self.poses_gt) == self.n_examples

    # Initialize camera poses.
    self.poses_init = None
    if args.init_poses_from_gt:
      logging.info("Initializing camera poses with ground truth.")
      # Initialize pose as ground truth (with noise).
      if args.camera_noise > 0.0:
        logging.info(
            "Perturbing camera poses with noise (std=%f).", args.camera_noise)
        self.se3_noise = camera.generate_camera_noise(
            num_poses=self.n_examples, noise_std=args.camera_noise)
        self.poses_init = camera.add_camera_noise(self.poses_gt, self.se3_noise)
      else:
        self.se3_noise = None
        self.poses_init = self.poses_gt.copy()
    elif args.init_poses_file:
      logging.info(
          "Initializing camera poses from file: %s", args.init_poses_file)
      with utils.open_file(args.init_poses_file, "rb") as f:
        self.poses_init = pickle.load(f)
    # convert from dict to np.array
    if self.poses_init is not None:
      poses_init = []
      for i in range(self.n_examples):
        poses_init.append(self.poses_init[i])
      self.poses_init = np.array(poses_init)
    else:
      self.poses_init = np.broadcast_to(
          np.eye(3, 4)[None, :, :], self.poses_gt.shape)

dataset_dict = {
    "blender": Blender,
}
