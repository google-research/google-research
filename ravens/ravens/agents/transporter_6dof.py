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

#!/usr/bin/env python
"""Transporter Agent (6DoF Hybrid with Regression)."""

import numpy as np
from ravens import utils
from ravens.agents.transporter import TransporterAgent
from ravens.models import Attention
from ravens.models import Transport

import tensorflow as tf
import transformations


class Transporter6dAgent(TransporterAgent):
  """6D Transporter variant."""

  def __init__(self, name, task):
    super().__init__(name, task)

    self.attention_model = Attention(
        input_shape=self.input_shape,
        num_rotations=1,
        preprocess=self.preprocess)
    self.transport_model = Transport(
        input_shape=self.input_shape,
        num_rotations=self.num_rotations,
        crop_size=self.crop_size,
        preprocess=self.preprocess,
        per_pixel_loss=False,
        six_dof=False)

    self.rpz_model = Transport(
        input_shape=self.input_shape,
        num_rotations=self.num_rotations,
        crop_size=self.crop_size,
        preprocess=self.preprocess,
        per_pixel_loss=False,
        six_dof=True)

    self.transport_model.set_bounds_pixel_size(self.bounds, self.pixel_size)
    self.rpz_model.set_bounds_pixel_size(self.bounds, self.pixel_size)

    self.six_dof = True

    self.p0_pixel_error = tf.keras.metrics.Mean(name='p0_pixel_error')
    self.p1_pixel_error = tf.keras.metrics.Mean(name='p1_pixel_error')
    self.p0_theta_error = tf.keras.metrics.Mean(name='p0_theta_error')
    self.p1_theta_error = tf.keras.metrics.Mean(name='p1_theta_error')
    self.metrics = [
        self.p0_pixel_error, self.p1_pixel_error, self.p0_theta_error,
        self.p1_theta_error
    ]

  def get_six_dof(self,
                  transform_params,
                  heightmap,
                  pose0,
                  pose1,
                  augment=True):
    """Adjust SE(3) poses via the in-plane SE(2) augmentation transform."""
    debug_visualize = False

    p1_position, p1_rotation = pose1[0], pose1[1]
    p0_position, p0_rotation = pose0[0], pose0[1]

    if debug_visualize:
      self.vis = utils.create_visualizer()
      self.transport_model.vis = self.vis

    if augment:
      t_world_center, t_world_centernew = utils.get_se3_from_image_transform(
          *transform_params, heightmap, self.bounds, self.pixel_size)

      if debug_visualize:
        label = 't_world_center'
        utils.make_frame(self.vis, label, h=0.05, radius=0.0012, o=1.0)
        self.vis[label].set_transform(t_world_center)

        label = 't_world_centernew'
        utils.make_frame(self.vis, label, h=0.05, radius=0.0012, o=1.0)
        self.vis[label].set_transform(t_world_centernew)

      t_worldnew_world = t_world_centernew @ np.linalg.inv(t_world_center)
    else:
      t_worldnew_world = np.eye(4)

    p1_quat_wxyz = (p1_rotation[3], p1_rotation[0], p1_rotation[1],
                    p1_rotation[2])
    t_world_p1 = transformations.quaternion_matrix(p1_quat_wxyz)
    t_world_p1[0:3, 3] = np.array(p1_position)

    t_worldnew_p1 = t_worldnew_world @ t_world_p1

    p0_quat_wxyz = (p0_rotation[3], p0_rotation[0], p0_rotation[1],
                    p0_rotation[2])
    t_world_p0 = transformations.quaternion_matrix(p0_quat_wxyz)
    t_world_p0[0:3, 3] = np.array(p0_position)
    t_worldnew_p0 = t_worldnew_world @ t_world_p0

    if debug_visualize:
      label = 't_worldnew_p1'
      utils.make_frame(self.vis, label, h=0.05, radius=0.0012, o=1.0)
      self.vis[label].set_transform(t_worldnew_p1)

      label = 't_world_p1'
      utils.make_frame(self.vis, label, h=0.05, radius=0.0012, o=1.0)
      self.vis[label].set_transform(t_world_p1)

      label = 't_worldnew_p0-0thetaoriginally'
      utils.make_frame(self.vis, label, h=0.05, radius=0.0021, o=1.0)
      self.vis[label].set_transform(t_worldnew_p0)

    # PICK FRAME, using 0 rotation due to suction rotational symmetry
    t_worldnew_p0theta0 = t_worldnew_p0 * 1.0
    t_worldnew_p0theta0[0:3, 0:3] = np.eye(3)

    if debug_visualize:
      label = 'PICK'
      utils.make_frame(self.vis, label, h=0.05, radius=0.0021, o=1.0)
      self.vis[label].set_transform(t_worldnew_p0theta0)

    # PLACE FRAME, adjusted for this 0 rotation on pick
    t_p0_p0theta0 = np.linalg.inv(t_worldnew_p0) @ t_worldnew_p0theta0
    t_worldnew_p1theta0 = t_worldnew_p1 @ t_p0_p0theta0

    if debug_visualize:
      label = 'PLACE'
      utils.make_frame(self.vis, label, h=0.05, radius=0.0021, o=1.0)
      self.vis[label].set_transform(t_worldnew_p1theta0)

    # convert the above rotation to euler
    quatwxyz_worldnew_p1theta0 = transformations.quaternion_from_matrix(
        t_worldnew_p1theta0)
    q = quatwxyz_worldnew_p1theta0
    quatxyzw_worldnew_p1theta0 = (q[1], q[2], q[3], q[0])
    p1_rotation = quatxyzw_worldnew_p1theta0
    p1_euler = utils.quatXYZW_to_eulerXYZ(p1_rotation)
    roll = p1_euler[0]
    pitch = p1_euler[1]
    p1_theta = -p1_euler[2]

    p0_theta = 0
    z = p1_position[2]
    return p0_theta, p1_theta, z, roll, pitch

  def get_sample(self, dataset, augment=True):
    (obs, act, _, _), _ = dataset.sample()
    img = self.get_image(obs)

    # Get training labels from data sample.
    p0_xyz, p0_xyzw = act['pose0']
    p1_xyz, p1_xyzw = act['pose1']
    p0 = utils.xyz_to_pix(p0_xyz, self.bounds, self.pix_size)
    p0_theta = -np.float32(utils.quatXYZW_to_eulerXYZ(p0_xyzw)[2])
    p1 = utils.xyz_to_pix(p1_xyz, self.bounds, self.pix_size)
    p1_theta = -np.float32(utils.quatXYZW_to_eulerXYZ(p1_xyzw)[2])
    p1_theta = p1_theta - p0_theta
    p0_theta = 0

    if augment:
      img, _, (p0, p1), transforms = utils.perturb(img, [p0, p1])
      p0_theta, p1_theta, z, roll, pitch = self.get_six_dof(
          transforms, img[:, :, 3], (p0_xyz, p0_xyzw), (p1_xyz, p1_xyzw))

    return img, p0, p0_theta, p1, p1_theta, z, roll, pitch

  def train(self, dataset, num_iter, writer, validation_dataset):
    """Train on dataset for a specific number of iterations.

    Args:
      dataset: a ravens.Dataset.
      num_iter: int, number of iterations to train.
      writer: a TF summary writer (for tensorboard).
      validation_dataset: a ravens.Dataset.
    """

    validation_rate = 200

    for i in range(num_iter):

      tf.keras.backend.set_learning_phase(1)
      _, p0, p0_theta, p1, p1_theta, z, roll, pitch = self.get_sample(
          dataset)

      # Compute training losses.
      loss0 = self.attention_model.train(input_image, p0, p0_theta)

      loss1 = self.transport_model.train(input_image, p0, p1, p1_theta, z, roll,
                                         pitch)

      loss2 = self.rpz_model.train(input_image, p0, p1, p1_theta, z, roll,
                                   pitch)
      del loss2

      with writer.as_default():
        tf.summary.scalar(
            'attention_loss',
            self.attention_model.metric.result(),
            step=self.total_iter + i)

        tf.summary.scalar(
            'transport_loss',
            self.transport_model.metric.result(),
            step=self.total_iter + i)
        tf.summary.scalar(
            'z_loss',
            self.rpz_model.z_metric.result(),
            step=self.total_iter + i)
        tf.summary.scalar(
            'roll_loss',
            self.rpz_model.roll_metric.result(),
            step=self.total_iter + i)
        tf.summary.scalar(
            'pitch_loss',
            self.rpz_model.pitch_metric.result(),
            step=self.total_iter + i)

      print(f'Train Iter: {self.total_iter + i} Loss: {loss0:.4f} {loss1:.4f}')

      if (self.total_iter + i) % validation_rate == 0:
        print('Validating!')
        tf.keras.backend.set_learning_phase(0)
        input_image, p0, p0_theta, p1, p1_theta, z, roll, pitch = self.get_data_batch(
            validation_dataset, augment=False)

        loss1 = self.transport_model.train(
            input_image, p0, p1, p1_theta, z, roll, pitch, validate=True)

        loss2 = self.rpz_model.train(
            input_image, p0, p1, p1_theta, z, roll, pitch, validate=True)

        # compute pixel/theta metrics
        # [metric.reset_states() for metric in self.metrics]
        # for _ in range(30):
        #     obs, act, info = validation_dataset.random_sample()
        #     self.act(obs, act, info, compute_error=True)

        with writer.as_default():
          tf.summary.scalar(
              'val_transport_loss',
              self.transport_model.metric.result(),
              step=self.total_iter + i)
          tf.summary.scalar(
              'val_z_loss',
              self.rpz_model.z_metric.result(),
              step=self.total_iter + i)
          tf.summary.scalar(
              'val_roll_loss',
              self.rpz_model.roll_metric.result(),
              step=self.total_iter + i)
          tf.summary.scalar(
              'val_pitch_loss',
              self.rpz_model.pitch_metric.result(),
              step=self.total_iter + i)

          tf.summary.scalar(
              'p0_pixel_error',
              self.p0_pixel_error.result(),
              step=self.total_iter + i)
          tf.summary.scalar(
              'p1_pixel_error',
              self.p1_pixel_error.result(),
              step=self.total_iter + i)
          tf.summary.scalar(
              'p0_theta_error',
              self.p0_theta_error.result(),
              step=self.total_iter + i)
          tf.summary.scalar(
              'p1_theta_error',
              self.p1_theta_error.result(),
              step=self.total_iter + i)

        tf.keras.backend.set_learning_phase(1)

    self.total_iter += num_iter
    self.save()

  def act(self, obs, info, compute_error=False, gt_act=None):
    """Run inference and return best action given visual observations."""

    # Get heightmap from RGB-D images.
    colormap, heightmap = self.get_heightmap(obs, self.camera_config)

    # Concatenate color with depth images.
    input_image = np.concatenate(
        (colormap, heightmap[Ellipsis, None], heightmap[Ellipsis, None], heightmap[Ellipsis,
                                                                         None]),
        axis=2)

    # Attention model forward pass.
    attention = self.attention_model.forward(input_image)
    argmax = np.argmax(attention)
    argmax = np.unravel_index(argmax, shape=attention.shape)
    p0_pixel = argmax[:2]
    p0_theta = argmax[2] * (2 * np.pi / attention.shape[2])

    # Transport model forward pass.
    transport = self.transport_model.forward(input_image, p0_pixel)
    _, z, roll, pitch = self.rpz_model.forward(input_image, p0_pixel)

    argmax = np.argmax(transport)
    argmax = np.unravel_index(argmax, shape=transport.shape)

    # Index into 3D discrete tensor, grab z, roll, pitch activations
    z_best = z[:, argmax[0], argmax[1], argmax[2]][Ellipsis, None]
    roll_best = roll[:, argmax[0], argmax[1], argmax[2]][Ellipsis, None]
    pitch_best = pitch[:, argmax[0], argmax[1], argmax[2]][Ellipsis, None]

    # Send through regressors for each of z, roll, pitch
    z_best = self.rpz_model.z_regressor(z_best)[0, 0]
    roll_best = self.rpz_model.roll_regressor(roll_best)[0, 0]
    pitch_best = self.rpz_model.pitch_regressor(pitch_best)[0, 0]

    p1_pixel = argmax[:2]
    p1_theta = argmax[2] * (2 * np.pi / transport.shape[2])

    # Pixels to end effector poses.
    p0_position = utils.pix_to_xyz(p0_pixel, heightmap, self.bounds,
                                   self.pixel_size)
    p1_position = utils.pix_to_xyz(p1_pixel, heightmap, self.bounds,
                                   self.pixel_size)

    p1_position = (p1_position[0], p1_position[1], z_best)

    p0_rotation = utils.eulerXYZ_to_quatXYZW((0, 0, -p0_theta))
    p1_rotation = utils.eulerXYZ_to_quatXYZW(
        (roll_best, pitch_best, -p1_theta))

    if compute_error:
      gt_p0_position, gt_p0_rotation = gt_act['params']['pose0']
      gt_p1_position, gt_p1_rotation = gt_act['params']['pose1']

      gt_p0_pixel = np.array(
          utils.xyz_to_pix(gt_p0_position, self.bounds, self.pixel_size))
      gt_p1_pixel = np.array(
          utils.xyz_to_pix(gt_p1_position, self.bounds, self.pixel_size))

      self.p0_pixel_error(np.linalg.norm(gt_p0_pixel - np.array(p0_pixel)))
      self.p1_pixel_error(np.linalg.norm(gt_p1_pixel - np.array(p1_pixel)))

      gt_p0_theta = -np.float32(
          utils.quatXYZW_to_eulerXYZ(gt_p0_rotation)[2])
      gt_p1_theta = -np.float32(
          utils.quatXYZW_to_eulerXYZ(gt_p1_rotation)[2])

      self.p0_theta_error(
          abs((np.rad2deg(gt_p0_theta - p0_theta) + 180) % 360 - 180))
      self.p1_theta_error(
          abs((np.rad2deg(gt_p1_theta - p1_theta) + 180) % 360 - 180))

      return None

    return {'pose0': (p0_position, p0_rotation),
            'pose1': (p1_position, p1_rotation)}
