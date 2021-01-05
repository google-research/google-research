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

#!/usr/bin/env python
"""Form-2-fit Agent (https://form2fit.github.io/)."""

import os

import cv2
import numpy as np


from ravens import cameras
from ravens import utils
from ravens.models import Attention
from ravens.models import Matching

import tensorflow as tf


class Form2FitAgent:
  """Form-2-fit Agent (https://form2fit.github.io/)."""

  def __init__(self, name, task):
    self.name = name
    self.task = task
    self.total_iter = 0
    self.num_rotations = 24
    self.descriptor_dim = 16
    self.pixel_size = 0.003125
    self.input_shape = (320, 160, 6)
    self.camera_config = cameras.RealSenseD415.CONFIG
    self.models_dir = os.path.join('checkpoints', self.name)
    self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])

    self.pick_model = Attention(
        input_shape=self.input_shape,
        num_rotations=1,
        preprocess=self.preprocess,
        lite=True)
    self.place_model = Attention(
        input_shape=self.input_shape,
        num_rotations=1,
        preprocess=self.preprocess,
        lite=True)
    self.match_model = Matching(
        input_shape=self.input_shape,
        descriptor_dim=self.descriptor_dim,
        num_rotations=self.num_rotations,
        preprocess=self.preprocess,
        lite=True)

  def train(self, dataset, num_iter, writer, validation_dataset=None):
    """Train on dataset for a specific number of iterations."""
    del validation_dataset

    for i in range(num_iter):
      obs, act, _ = dataset.random_sample()

      # Get heightmap from RGB-D images.
      configs = act['camera_config']
      colormap, heightmap = self.get_heightmap(obs, configs)

      # Get training labels from data sample.
      pose0, pose1 = act['params']['pose0'], act['params']['pose1']
      p0_position, p0_rotation = pose0[0], pose0[1]
      p0 = utils.xyz_to_pix(p0_position, self.bounds, self.pixel_size)
      p0_theta = -np.float32(
          utils.quatXYZW_to_eulerXYZ(p0_rotation)[2])
      p1_position, p1_rotation = pose1[0], pose1[1]
      p1 = utils.xyz_to_pix(p1_position, self.bounds, self.pixel_size)
      p1_theta = -np.float32(
          utils.quatXYZW_to_eulerXYZ(p1_rotation)[2])
      p1_theta = p1_theta - p0_theta
      p0_theta = 0

      # Concatenate color with depth images.
      input_image = np.concatenate((colormap, heightmap[Ellipsis, None],
                                    heightmap[Ellipsis, None], heightmap[Ellipsis, None]),
                                   axis=2)

      # Do data augmentation (perturb rotation and translation).
      input_image, _, roundedpixels, _ = utils.perturb(input_image, [p0, p1])
      p0, p1 = roundedpixels

      # Compute training loss.
      loss0 = self.pick_model.train(input_image, p0, theta=0)
      loss1 = self.place_model.train(input_image, p1, theta=0)
      loss2 = self.match_model.train(input_image, p0, p1, theta=p1_theta)
      with writer.as_default():
        tf.summary.scalar(
            'pick_loss',
            self.pick_model.metric.result(),
            step=self.total_iter + i)
        tf.summary.scalar(
            'place_loss',
            self.place_model.metric.result(),
            step=self.total_iter + i)
        tf.summary.scalar(
            'match_loss',
            self.match_model.metric.result(),
            step=self.total_iter + i)
      print(
          f'Train Iter: {self.total_iter + i} Loss: {loss0:.4f} {loss1:.4f} {loss2:.4f}'
      )

    self.total_iter += num_iter
    self.save()

  def act(self, obs, info):
    """Run inference and return best action given visual observations."""
    del info

    act = {'camera_config': self.camera_config, 'primitive': None}
    if not obs:
      return act

    # Get heightmap from RGB-D images.
    colormap, heightmap = self.get_heightmap(obs, self.camera_config)

    # Concatenate color with depth images.
    input_image = np.concatenate(
        (colormap, heightmap[Ellipsis, None], heightmap[Ellipsis, None], heightmap[Ellipsis,
                                                                         None]),
        axis=2)

    # Get top-k pixels from pick and place heatmaps.
    k = 100
    pick_heatmap = self.pick_model.forward(
        input_image, apply_softmax=True).squeeze()
    place_heatmap = self.place_model.forward(
        input_image, apply_softmax=True).squeeze()
    descriptors = np.float32(self.match_model.forward(input_image))

    # V4
    pick_heatmap = cv2.GaussianBlur(pick_heatmap, (49, 49), 0)
    place_heatmap = cv2.GaussianBlur(place_heatmap, (49, 49), 0)
    pick_topk = np.int32(
        np.unravel_index(
            np.argsort(pick_heatmap.reshape(-1))[-k:], pick_heatmap.shape)).T
    pick_pixel = pick_topk[-1, :]
    from skimage.feature import peak_local_max  # pylint: disable=g-import-not-at-top
    place_peaks = peak_local_max(place_heatmap, num_peaks=1)
    distances = np.ones((place_peaks.shape[0], self.num_rotations)) * 10
    pick_descriptor = descriptors[0, pick_pixel[0],
                                  pick_pixel[1], :].reshape(1, -1)
    for i in range(place_peaks.shape[0]):
      peak = place_peaks[i, :]
      place_descriptors = descriptors[:, peak[0], peak[1], :]
      distances[i, :] = np.linalg.norm(
          place_descriptors - pick_descriptor, axis=1)
    ibest = np.unravel_index(np.argmin(distances), shape=distances.shape)
    p0_pixel = pick_pixel
    p0_theta = 0
    p1_pixel = place_peaks[ibest[0], :]
    p1_theta = ibest[1] * (2 * np.pi / self.num_rotations)

    # # V3
    # pick_heatmap = cv2.GaussianBlur(pick_heatmap, (49, 49), 0)
    # place_heatmap = cv2.GaussianBlur(place_heatmap, (49, 49), 0)
    # pick_topk = np.int32(
    #     np.unravel_index(
    #         np.argsort(pick_heatmap.reshape(-1))[-k:], pick_heatmap.shape)).T
    # place_topk = np.int32(
    #     np.unravel_index(
    #         np.argsort(place_heatmap.reshape(-1))[-k:],
    #         place_heatmap.shape)).T
    # pick_pixel = pick_topk[-1, :]
    # place_pixel = place_topk[-1, :]
    # pick_descriptor = descriptors[0, pick_pixel[0],
    #                               pick_pixel[1], :].reshape(1, -1)
    # place_descriptor = descriptors[:, place_pixel[0], place_pixel[1], :]
    # distances = np.linalg.norm(place_descriptor - pick_descriptor, axis=1)
    # irotation = np.argmin(distances)
    # p0_pixel = pick_pixel
    # p0_theta = 0
    # p1_pixel = place_pixel
    # p1_theta = irotation * (2 * np.pi / self.num_rotations)

    # # V2
    # pick_topk = np.int32(
    #     np.unravel_index(
    #         np.argsort(pick_heatmap.reshape(-1))[-k:], pick_heatmap.shape)).T
    # place_topk = np.int32(
    #     np.unravel_index(
    #         np.argsort(place_heatmap.reshape(-1))[-k:],
    #         place_heatmap.shape)).T
    # pick_pixel = pick_topk[-1, :]
    # pick_descriptor = descriptors[0, pick_pixel[0],
    #                               pick_pixel[1], :].reshape(1, 1, 1, -1)
    # distances = np.linalg.norm(descriptors - pick_descriptor, axis=3)
    # distances = np.transpose(distances, [1, 2, 0])
    # max_distance = int(np.round(np.max(distances)))
    # for i in range(self.num_rotations):
    #   distances[:, :, i] = cv2.circle(distances[:, :, i],
    #                                   (pick_pixel[1], pick_pixel[0]), 50,
    #                                   max_distance, -1)
    # ibest = np.unravel_index(np.argmin(distances), shape=distances.shape)
    # p0_pixel = pick_pixel
    # p0_theta = 0
    # p1_pixel = ibest[:2]
    # p1_theta = ibest[2] * (2 * np.pi / self.num_rotations)

    # # V1
    # pick_topk = np.int32(
    #     np.unravel_index(
    #         np.argsort(pick_heatmap.reshape(-1))[-k:], pick_heatmap.shape)).T
    # place_topk = np.int32(
    #     np.unravel_index(
    #         np.argsort(place_heatmap.reshape(-1))[-k:],
    #         place_heatmap.shape)).T
    # distances = np.zeros((k, k, self.num_rotations))
    # for ipick in range(k):
    #   pick_descriptor = descriptors[0, pick_topk[ipick, 0],
    #                                 pick_topk[ipick, 1], :].reshape(1, -1)
    #   for iplace in range(k):
    #     place_descriptors = descriptors[:, place_topk[iplace, 0],
    #                                     place_topk[iplace, 1], :]
    #     distances[ipick, iplace, :] = np.linalg.norm(
    #         place_descriptors - pick_descriptor, axis=1)
    # ibest = np.unravel_index(np.argmin(distances), shape=distances.shape)
    # p0_pixel = pick_topk[ibest[0], :]
    # p0_theta = 0
    # p1_pixel = place_topk[ibest[1], :]
    # p1_theta = ibest[2] * (2 * np.pi / self.num_rotations)

    # Pixels to end effector poses.
    p0_position = utils.pix_to_xyz(p0_pixel, heightmap, self.bounds,
                                   self.pixel_size)
    p1_position = utils.pix_to_xyz(p1_pixel, heightmap, self.bounds,
                                   self.pixel_size)
    p0_rotation = utils.eulerXYZ_to_quatXYZW((0, 0, -p0_theta))
    p1_rotation = utils.eulerXYZ_to_quatXYZW((0, 0, -p1_theta))

    act['primitive'] = 'pick_place'
    if self.task == 'sweeping':
      act['primitive'] = 'sweep'
    elif self.task == 'pushing':
      act['primitive'] = 'push'
    params = {
        'pose0': (p0_position, p0_rotation),
        'pose1': (p1_position, p1_rotation)
    }
    act['params'] = params
    return act

  #-------------------------------------------------------------------------
  # Helper Functions
  #-------------------------------------------------------------------------

  def preprocess(self, image):
    """Pre-process images (subtract mean, divide by std)."""
    color_mean = 0.18877631
    depth_mean = 0.00509261
    color_std = 0.07276466
    depth_std = 0.00903967
    image[:, :, :3] = (image[:, :, :3] / 255 - color_mean) / color_std
    image[:, :, 3:] = (image[:, :, 3:] - depth_mean) / depth_std
    return image

  def get_heightmap(self, obs, configs):
    """Reconstruct orthographic heightmaps with segmentation masks."""
    heightmaps, colormaps = utils.reconstruct_heightmaps(
        obs['color'], obs['depth'], configs, self.bounds, self.pixel_size)
    colormaps = np.float32(colormaps)
    heightmaps = np.float32(heightmaps)

    # Fuse maps from different views.
    valid = np.sum(colormaps, axis=3) > 0
    repeat = np.sum(valid, axis=0)
    repeat[repeat == 0] = 1
    colormap = np.sum(colormaps, axis=0) / repeat[Ellipsis, None]
    colormap = np.uint8(np.round(colormap))
    heightmap = np.sum(heightmaps, axis=0) / repeat
    return colormap, heightmap

  def load(self, num_iter):
    """Load pre-trained models."""
    pick_fname = 'pick-ckpt-%d.h5' % num_iter
    place_fname = 'place-ckpt-%d.h5' % num_iter
    match_fname = 'match-ckpt-%d.h5' % num_iter
    pick_fname = os.path.join(self.models_dir, pick_fname)
    place_fname = os.path.join(self.models_dir, place_fname)
    match_fname = os.path.join(self.models_dir, match_fname)
    self.pick_model.load(pick_fname)
    self.place_model.load(place_fname)
    self.match_model.load(match_fname)
    self.total_iter = num_iter

  def save(self):
    """Save models."""
    if not os.path.exists(self.models_dir):
      os.makedirs(self.models_dir)
    pick_fname = 'pick-ckpt-%d.h5' % self.total_iter
    place_fname = 'place-ckpt-%d.h5' % self.total_iter
    match_fname = 'match-ckpt-%d.h5' % self.total_iter
    pick_fname = os.path.join(self.models_dir, pick_fname)
    place_fname = os.path.join(self.models_dir, place_fname)
    match_fname = os.path.join(self.models_dir, match_fname)
    self.pick_model.save(pick_fname)
    self.place_model.save(place_fname)
    self.match_model.save(match_fname)
