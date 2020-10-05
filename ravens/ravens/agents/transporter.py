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

"""Transporter Agent."""

import os

import numpy as np

from ravens import cameras
from ravens import utils
from ravens.models import Attention
from ravens.models import Transport
from ravens.models import TransportGoal

import tensorflow as tf
import transformations


class TransporterAgent:
  """Agent for 2D (translation-only) or 2D + rotation tasks."""

  def __init__(self,
               name,
               task,
               num_rotations=36,
               crop_bef_q=True,
               use_goal_image=False):
    self.name = name
    self.task = task
    self.total_iter = 0
    self.crop_size = 64
    self.num_rotations = num_rotations
    self.pixel_size = 0.003125
    self.input_shape = (320, 160, 6)
    self.camera_config = cameras.RealSenseD415.CONFIG
    self.models_dir = os.path.join('checkpoints', self.name)
    self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])
    self.six_dof = False
    self.crop_bef_q = crop_bef_q
    self.use_goal_image = use_goal_image

  def get_data_batch(self, dataset, augment=True):
    """Use dataset to extract and preprocess data.

    Supports adding a goal image, in which case the current and goal
    images are stacked together channel-wise (first 6 for current, last 6
    for goal) before doing data augmentation, to ensure consistency.

    Args:
      dataset: a ravens.Dataset (train or validation)
      augment: if True, perform data augmentation.

    Returns:
      tuple of data for training:
        (input_image, p0, p0_theta, p1, p1_theta)
      tuple additionally includes (z, roll, pitch) if self.six_dof
      if self.use_goal_image, then the goal image is stacked with the
      current image in `input_image`. If splitting up current and goal
      images is desired, it should be done outside this method.
    """
    if self.use_goal_image:
      obs, act, _, goal = dataset.random_sample(goal_images=True)
    else:
      obs, act, _ = dataset.random_sample()

    # Get heightmap from RGB-D images, including goal images if specified.
    configs = act['camera_config']
    colormap, heightmap = self.get_heightmap(obs, configs)
    if self.use_goal_image:
      colormap_g, heightmap_g = self.get_heightmap(goal, configs)

    # Get training labels from data sample.
    pose0, pose1 = act['params']['pose0'], act['params']['pose1']
    p0_position, p0_rotation = pose0[0], pose0[1]
    p0 = utils.position_to_pixel(p0_position, self.bounds, self.pixel_size)
    p0_theta = -np.float32(
        utils.get_rot_from_pybullet_quaternion(p0_rotation)[2])
    p1_position, p1_rotation = pose1[0], pose1[1]
    p1 = utils.position_to_pixel(p1_position, self.bounds, self.pixel_size)
    p1_theta = -np.float32(
        utils.get_rot_from_pybullet_quaternion(p1_rotation)[2])

    # Concatenate color with depth images.
    input_image = self.concatenate_c_h(colormap, heightmap)

    # If using goal image, stack _with_ input_image before data augmentation.
    if self.use_goal_image:
      goal_image = self.concatenate_c_h(colormap_g, heightmap_g)
      input_image = np.concatenate((input_image, goal_image), axis=2)
      assert input_image.shape[2] == 12, input_image.shape

    # Do data augmentation (perturb rotation and translation).
    if augment:
      input_image, _, rounded_pixels, transform_params = utils.perturb(
          input_image, [p0, p1])
      p0, p1 = rounded_pixels

    if self.six_dof:
      if not augment:
        transform_params = None
      p0_theta, p1_theta, z, roll, pitch = self.get_six_dof(
          transform_params, heightmap, pose0, pose1, augment=augment)
      return input_image, p0, p0_theta, p1, p1_theta, z, roll, pitch
    else:
      # If using a goal image, it is stacked with `input_image` and split later.
      p1_theta = p1_theta - p0_theta
      p0_theta = 0
      return input_image, p0, p0_theta, p1, p1_theta

  def train(self, dataset, num_iter, writer, validation_dataset):
    """Train on dataset for a specific number of iterations.

    Args:
      dataset: a ravens.Dataset.
      num_iter: int, number of iterations to train.
      writer: a TF summary writer (for tensorboard).
      validation_dataset: a ravens.Dataset.
    """

    validation_rate = 25
    for i in range(num_iter):

      assert not self.six_dof
      input_image, p0, p0_theta, p1, p1_theta = self.get_data_batch(dataset)  # pylint: disable=unbalanced-tuple-unpacking

      # Compute Attention training losses.
      if self.use_goal_image:
        half = int(input_image.shape[2] / 2)
        img_curr = input_image[:, :, :half]  # ignore goal portion
        loss0 = self.attention_model.train(img_curr, p0, p0_theta)
      else:
        loss0 = self.attention_model.train(input_image, p0, p0_theta)
      with writer.as_default():
        tf.summary.scalar(
            'attention_loss',
            self.attention_model.metric.result(),
            step=self.total_iter + i)

      # Compute Transport training losses.
      if isinstance(self.transport_model, Attention):
        loss1 = self.transport_model.train(input_image, p1, p1_theta)
      elif isinstance(self.transport_model, TransportGoal):
        half = int(input_image.shape[2] / 2)
        img_curr = input_image[:, :, :half]
        img_goal = input_image[:, :, half:]
        loss1 = self.transport_model.train(img_curr, img_goal, p0, p1, p1_theta)
      else:
        loss1 = self.transport_model.train(input_image, p0, p1, p1_theta)

        with writer.as_default():
          tf.summary.scalar(
              'transport_loss',
              self.transport_model.metric.result(),
              step=self.total_iter + i)

      print(f'Train Iter: {self.total_iter + i} Loss: {loss0:.4f} {loss1:.4f}')

      # Compute validation losses.
      if (self.total_iter + i) % validation_rate == 0:
        print('Validating!')
        tf.keras.backend.set_learning_phase(0)
        assert not self.six_dof
        input_image, p0, p0_theta, p1, p1_theta = self.get_data_batch(    # pylint: disable=unbalanced-tuple-unpacking
            validation_dataset)

        if isinstance(self.transport_model, Attention):
          print(
              'Validation error on Attention transport model not implemented....'
          )
          print("But don't want to error out to disrupt previous behavior.")
        elif isinstance(self.transport_model, TransportGoal):
          print('Validation error on TransportGoal model not implemented....')
        else:
          loss1 = self.transport_model.train(
              input_image, p0, p1, p1_theta, validate=True)

          with writer.as_default():
            tf.summary.scalar(
                'val_transport_loss',
                self.transport_model.metric.result(),
                step=self.total_iter + i)

        tf.keras.backend.set_learning_phase(1)

    self.total_iter += num_iter
    self.save()

  def act(self, obs, info, goal=None):
    """Run inference and return best action given visual observations."""
    del info

    act = {'camera_config': self.camera_config, 'primitive': None}
    if not obs:
      return act

    # Get heightmap from RGB-D images.
    colormap, heightmap = self.get_heightmap(obs, self.camera_config)
    if goal is not None:
      colormap_g, heightmap_g = self.get_heightmap(goal, self.camera_config)

    # Concatenate color with depth images.
    input_image = self.concatenate_c_h(colormap, heightmap)

    # Make a goal image if needed, and for consistency stack with input.
    if self.use_goal_image:
      goal_image = self.concatenate_c_h(colormap_g, heightmap_g)
      input_image = np.concatenate((input_image, goal_image), axis=2)
      assert input_image.shape[2] == 12, input_image.shape

    # Attention model forward pass.
    if self.use_goal_image:
      half = int(input_image.shape[2] / 2)
      input_only = input_image[:, :, :half]  # ignore goal portion
      attention = self.attention_model.forward(input_only)
    else:
      attention = self.attention_model.forward(input_image)
    argmax = np.argmax(attention)
    argmax = np.unravel_index(argmax, shape=attention.shape)
    p0_pixel = argmax[:2]
    p0_theta = argmax[2] * (2 * np.pi / attention.shape[2])

    # Transport model forward pass.
    if isinstance(self.transport_model, TransportGoal):
      half = int(input_image.shape[2] / 2)
      img_curr = input_image[:, :, :half]
      img_goal = input_image[:, :, half:]
      transport = self.transport_model.forward(img_curr, img_goal, p0_pixel)
    else:
      transport = self.transport_model.forward(input_image, p0_pixel)

    argmax = np.argmax(transport)
    argmax = np.unravel_index(argmax, shape=transport.shape)

    p1_pixel = argmax[:2]
    p1_theta = argmax[2] * (2 * np.pi / transport.shape[2])

    # Pixels to end effector poses.
    p0_position = utils.pixel_to_position(p0_pixel, heightmap, self.bounds,
                                          self.pixel_size)
    p1_position = utils.pixel_to_position(p1_pixel, heightmap, self.bounds,
                                          self.pixel_size)

    p0_rotation = utils.get_pybullet_quaternion_from_rot((0, 0, -p0_theta))
    p1_rotation = utils.get_pybullet_quaternion_from_rot((0, 0, -p1_theta))

    return self.p0_p1_position_rotations_to_act(act, p0_position, p0_rotation,
                                                p1_position, p1_rotation)

  #-------------------------------------------------------------------------
  # Helper Functions
  #-------------------------------------------------------------------------

  def concatenate_c_h(self, colormap, heightmap):
    """Concatenates color and height images to get a 6D image."""
    img = np.concatenate(
        (colormap, heightmap[Ellipsis, None], heightmap[Ellipsis, None], heightmap[Ellipsis,
                                                                         None]),
        axis=2)
    assert img.shape == self.input_shape, img.shape
    return img

  def p0_p1_position_rotations_to_act(self, act, p0_position, p0_rotation,
                                      p1_position, p1_rotation):
    """Calculate action from two poses."""
    act['primitive'] = 'pick_place'
    if self.task == 'insertion-sixdof':
      act['primitive'] = 'pick_place_6dof'
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
    # this line averages
    # heightmap = np.sum(heightmaps, axis=0) / repeat
    # this line does occlusion
    heightmap = np.max(heightmaps, axis=0)
    return colormap, heightmap

  def load(self, num_iter):
    """Load pre-trained models."""
    attention_fname = 'attention-ckpt-%d.h5' % num_iter
    transport_fname = 'transport-ckpt-%d.h5' % num_iter
    attention_fname = os.path.join(self.models_dir, attention_fname)
    transport_fname = os.path.join(self.models_dir, transport_fname)
    self.attention_model.load(attention_fname)
    self.transport_model.load(transport_fname)
    self.total_iter = num_iter

  def save(self):
    """Save models."""
    if not os.path.exists(self.models_dir):
      os.makedirs(self.models_dir)
    attention_fname = 'attention-ckpt-%d.h5' % self.total_iter
    transport_fname = 'transport-ckpt-%d.h5' % self.total_iter
    attention_fname = os.path.join(self.models_dir, attention_fname)
    transport_fname = os.path.join(self.models_dir, transport_fname)
    self.attention_model.save(attention_fname)
    self.transport_model.save(transport_fname)


class OriginalTransporterAgent(TransporterAgent):

  def __init__(self, name, task, num_rotations=36, crop_bef_q=True):
    super().__init__(name, task, num_rotations, crop_bef_q)

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
        crop_bef_q=self.crop_bef_q)


class NoTransportTransporterAgent(TransporterAgent):

  def __init__(self, name, task):
    super().__init__(name, task)

    self.attention_model = Attention(
        input_shape=self.input_shape,
        num_rotations=1,
        preprocess=self.preprocess)
    self.transport_model = Attention(
        input_shape=self.input_shape,
        num_rotations=self.num_rotations,
        preprocess=self.preprocess)


class PerPixelLossTransporterAgent(TransporterAgent):

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
        per_pixel_loss=True)


class GoalTransporterAgent(TransporterAgent):
  """Goal-Conditioned Transporters supporting a separate goal FCN."""

  def __init__(self, name, task, num_rotations=36):
    super().__init__(name, task, num_rotations, use_goal_image=True)

    self.attention_model = Attention(
        input_shape=self.input_shape,
        num_rotations=1,
        preprocess=self.preprocess)
    self.transport_model = TransportGoal(
        input_shape=self.input_shape,
        num_rotations=self.num_rotations,
        crop_size=self.crop_size,
        preprocess=self.preprocess)


class GoalNaiveTransporterAgent(TransporterAgent):
  """Naive version which stacks current and goal images through normal Transport."""

  def __init__(self, name, task, num_rotations=36):
    super().__init__(name, task, num_rotations, use_goal_image=True)

    # Stack the goal image for the vanilla Transport module.
    t_shape = (self.input_shape[0], self.input_shape[1],
               int(self.input_shape[2] * 2))

    self.attention_model = Attention(
        input_shape=self.input_shape,
        num_rotations=1,
        preprocess=self.preprocess)
    self.transport_model = Transport(
        input_shape=t_shape,
        num_rotations=self.num_rotations,
        crop_size=self.crop_size,
        preprocess=self.preprocess,
        per_pixel_loss=False,
        use_goal_image=True)


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
    p1_euler = utils.get_rot_from_pybullet_quaternion(p1_rotation)
    roll = p1_euler[0]
    pitch = p1_euler[1]
    p1_theta = -p1_euler[2]

    p0_theta = 0
    z = p1_position[2]
    return p0_theta, p1_theta, z, roll, pitch

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

      input_image, p0, p0_theta, p1, p1_theta, z, roll, pitch = self.get_data_batch(
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
    p0_position = utils.pixel_to_position(p0_pixel, heightmap, self.bounds,
                                          self.pixel_size)
    p1_position = utils.pixel_to_position(p1_pixel, heightmap, self.bounds,
                                          self.pixel_size)

    p1_position = (p1_position[0], p1_position[1], z_best)

    p0_rotation = utils.get_pybullet_quaternion_from_rot((0, 0, -p0_theta))
    p1_rotation = utils.get_pybullet_quaternion_from_rot(
        (roll_best, pitch_best, -p1_theta))

    if compute_error:
      gt_p0_position, gt_p0_rotation = gt_act['params']['pose0']
      gt_p1_position, gt_p1_rotation = gt_act['params']['pose1']

      gt_p0_pixel = np.array(
          utils.position_to_pixel(gt_p0_position, self.bounds, self.pixel_size))
      gt_p1_pixel = np.array(
          utils.position_to_pixel(gt_p1_position, self.bounds, self.pixel_size))

      self.p0_pixel_error(np.linalg.norm(gt_p0_pixel - np.array(p0_pixel)))
      self.p1_pixel_error(np.linalg.norm(gt_p1_pixel - np.array(p1_pixel)))

      gt_p0_theta = -np.float32(
          utils.get_rot_from_pybullet_quaternion(gt_p0_rotation)[2])
      gt_p1_theta = -np.float32(
          utils.get_rot_from_pybullet_quaternion(gt_p1_rotation)[2])

      self.p0_theta_error(
          abs((np.rad2deg(gt_p0_theta - p0_theta) + 180) % 360 - 180))
      self.p1_theta_error(
          abs((np.rad2deg(gt_p1_theta - p1_theta) + 180) % 360 - 180))

      return None

    return self.p0_p1_position_rotations_to_act(act, p0_position, p0_rotation,
                                                p1_position, p1_rotation)
