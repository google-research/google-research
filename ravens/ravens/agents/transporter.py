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

#!/usr/bin/env python
"""Transporter Agent."""

import os

import numpy as np

from ravens import cameras
from ravens import utils
from ravens.models import Attention
from ravens.models import Transport
from ravens.models import TransportGoal
from ravens.models import TransportPerPixelLoss
import tensorflow as tf


class TransporterAgent:
  """Agent that uses Transporter Networks."""

  def __init__(self, name, task, n_rotations=36):
    self.name = name
    self.task = task
    self.total_steps = 0
    self.crop_size = 64
    self.n_rotations = n_rotations
    self.pix_size = 0.003125
    self.in_shape = (320, 160, 6)
    self.cam_config = cameras.RealSenseD415.CONFIG
    self.models_dir = os.path.join('checkpoints', self.name)
    self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])

  def get_image(self, obs):
    """Stack color and height images image."""

    # if self.use_goal_image:
    #   colormap_g, heightmap_g = utils.get_fused_heightmap(goal, configs)
    #   goal_image = self.concatenate_c_h(colormap_g, heightmap_g)
    #   input_image = np.concatenate((input_image, goal_image), axis=2)
    #   assert input_image.shape[2] == 12, input_image.shape

    # Get color and height maps from RGB-D images.
    cmap, hmap = utils.get_fused_heightmap(
        obs, self.cam_config, self.bounds, self.pix_size)
    img = np.concatenate((cmap,
                          hmap[Ellipsis, None],
                          hmap[Ellipsis, None],
                          hmap[Ellipsis, None]), axis=2)
    assert img.shape == self.in_shape, img.shape
    return img

  def get_sample(self, dataset, augment=True):
    """Get a dataset sample.

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

    # Data augmentation.
    if augment:
      img, _, (p0, p1), _ = utils.perturb(img, [p0, p1])

    return img, p0, p0_theta, p1, p1_theta

  def train(self, dataset, writer=None):
    """Train on a dataset sample for 1 iteration.

    Args:
      dataset: a ravens.Dataset.
      writer: a TF summary writer (for tensorboard).
    """
    tf.keras.backend.set_learning_phase(1)
    img, p0, p0_theta, p1, p1_theta = self.get_sample(dataset)

    # Get training losses.
    step = self.total_steps + 1
    loss0 = self.attention.train(img, p0, p0_theta)
    if isinstance(self.transport, Attention):
      loss1 = self.transport.train(img, p1, p1_theta)
    else:
      loss1 = self.transport.train(img, p0, p1, p1_theta)
    with writer.as_default():
      sc = tf.summary.scalar
      sc('train_loss/attention', loss0, step)
      sc('train_loss/transport', loss1, step)
    print(f'Train Iter: {step} Loss: {loss0:.4f} {loss1:.4f}')
    self.total_steps = step

    # TODO(andyzeng) cleanup goal-conditioned model.

    # if self.use_goal_image:
    #   half = int(input_image.shape[2] / 2)
    #   img_curr = input_image[:, :, :half]  # ignore goal portion
    #   loss0 = self.attention.train(img_curr, p0, p0_theta)
    # else:
    #   loss0 = self.attention.train(input_image, p0, p0_theta)

    # if isinstance(self.transport, Attention):
    #   loss1 = self.transport.train(input_image, p1, p1_theta)
    # elif isinstance(self.transport, TransportGoal):
    #   half = int(input_image.shape[2] / 2)
    #   img_curr = input_image[:, :, :half]
    #   img_goal = input_image[:, :, half:]
    #   loss1 = self.transport.train(img_curr, img_goal, p0, p1, p1_theta)
    # else:
    #   loss1 = self.transport.train(input_image, p0, p1, p1_theta)

  def validate(self, dataset, writer=None):  # pylint: disable=unused-argument
    """Test on a validation dataset for 10 iterations."""
    print('Skipping validation.')
    # tf.keras.backend.set_learning_phase(0)
    # n_iter = 10
    # loss0, loss1 = 0, 0
    # for i in range(n_iter):
    #   img, p0, p0_theta, p1, p1_theta = self.get_sample(dataset, False)

    #   # Get validation losses. Do not backpropagate.
    #   loss0 += self.attention.train(img, p0, p0_theta, backprop=False)
    #   if isinstance(self.transport, Attention):
    #     loss1 += self.transport.train(img, p1, p1_theta, backprop=False)
    #   else:
    #     loss1 += self.transport.train(img, p0, p1, p1_theta, backprop=False)
    # loss0 /= n_iter
    # loss1 /= n_iter
    # with writer.as_default():
    #   sc = tf.summary.scalar
    #   sc('test_loss/attention', loss0, self.total_steps)
    #   sc('test_loss/transport', loss1, self.total_steps)
    # print(f'Validation Loss: {loss0:.4f} {loss1:.4f}')

  def act(self, obs, info=None, goal=None):  # pylint: disable=unused-argument
    """Run inference and return best action given visual observations."""
    tf.keras.backend.set_learning_phase(0)

    # Get heightmap from RGB-D images.
    img = self.get_image(obs)

    # Attention model forward pass.
    pick_conf = self.attention.forward(img)
    argmax = np.argmax(pick_conf)
    argmax = np.unravel_index(argmax, shape=pick_conf.shape)
    p0_pix = argmax[:2]
    p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])

    # Transport model forward pass.
    place_conf = self.transport.forward(img, p0_pix)
    argmax = np.argmax(place_conf)
    argmax = np.unravel_index(argmax, shape=place_conf.shape)
    p1_pix = argmax[:2]
    p1_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])

    # Pixels to end effector poses.
    hmap = img[:, :, 3]
    p0_xyz = utils.pix_to_xyz(p0_pix, hmap, self.bounds, self.pix_size)
    p1_xyz = utils.pix_to_xyz(p1_pix, hmap, self.bounds, self.pix_size)
    p0_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p0_theta))
    p1_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p1_theta))

    return {'pose0': (p0_xyz, p0_xyzw), 'pose1': (p1_xyz, p1_xyzw)}

    # TODO(andyzeng) cleanup goal-conditioned model.

    # Make a goal image if needed, and for consistency stack with input.
    # if self.use_goal_image:
    #   cmap_g, hmap_g = utils.get_fused_heightmap(goal, self.cam_config)
    #   goal_image = self.concatenate_c_h(colormap_g, heightmap_g)
    #   input_image = np.concatenate((input_image, goal_image), axis=2)
    #   assert input_image.shape[2] == 12, input_image.shape

    # if self.use_goal_image:
    #   half = int(input_image.shape[2] / 2)
    #   input_only = input_image[:, :, :half]  # ignore goal portion
    #   pick_conf = self.attention.forward(input_only)
    # else:
    # if isinstance(self.transport, TransportGoal):
    #   half = int(input_image.shape[2] / 2)
    #   img_curr = input_image[:, :, :half]
    #   img_goal = input_image[:, :, half:]
    #   place_conf = self.transport.forward(img_curr, img_goal, p0_pix)

  def load(self, n_iter):
    """Load pre-trained models."""
    print(f'Loading pre-trained model at {n_iter} iterations.')
    attention_fname = 'attention-ckpt-%d.h5' % n_iter
    transport_fname = 'transport-ckpt-%d.h5' % n_iter
    attention_fname = os.path.join(self.models_dir, attention_fname)
    transport_fname = os.path.join(self.models_dir, transport_fname)
    self.attention.load(attention_fname)
    self.transport.load(transport_fname)
    self.total_steps = n_iter

  def save(self):
    """Save models."""
    if not os.path.exists(self.models_dir):
      os.makedirs(self.models_dir)
    attention_fname = 'attention-ckpt-%d.h5' % self.total_steps
    transport_fname = 'transport-ckpt-%d.h5' % self.total_steps
    attention_fname = os.path.join(self.models_dir, attention_fname)
    transport_fname = os.path.join(self.models_dir, transport_fname)
    self.attention.save(attention_fname)
    self.transport.save(transport_fname)

#-----------------------------------------------------------------------------
# Other Transporter Variants
#-----------------------------------------------------------------------------


class OriginalTransporterAgent(TransporterAgent):

  def __init__(self, name, task, n_rotations=36):
    super().__init__(name, task, n_rotations)

    self.attention = Attention(
        in_shape=self.in_shape,
        n_rotations=1,
        preprocess=utils.preprocess)
    self.transport = Transport(
        in_shape=self.in_shape,
        n_rotations=self.n_rotations,
        crop_size=self.crop_size,
        preprocess=utils.preprocess)


class NoTransportTransporterAgent(TransporterAgent):

  def __init__(self, name, task, n_rotations=36):
    super().__init__(name, task, n_rotations)

    self.attention = Attention(
        in_shape=self.in_shape,
        n_rotations=1,
        preprocess=utils.preprocess)
    self.transport = Attention(
        in_shape=self.in_shape,
        n_rotations=self.n_rotations,
        preprocess=utils.preprocess)


class PerPixelLossTransporterAgent(TransporterAgent):

  def __init__(self, name, task, n_rotations=36):
    super().__init__(name, task, n_rotations)

    self.attention = Attention(
        in_shape=self.in_shape,
        n_rotations=1,
        preprocess=utils.preprocess)
    self.transport = TransportPerPixelLoss(
        in_shape=self.in_shape,
        n_rotations=self.n_rotations,
        crop_size=self.crop_size,
        preprocess=utils.preprocess)


class GoalTransporterAgent(TransporterAgent):
  """Goal-Conditioned Transporters supporting a separate goal FCN."""

  def __init__(self, name, task, n_rotations=36):
    super().__init__(name, task, n_rotations)

    self.attention = Attention(
        in_shape=self.in_shape,
        n_rotations=1,
        preprocess=utils.preprocess)
    self.transport = TransportGoal(
        in_shape=self.in_shape,
        n_rotations=self.n_rotations,
        crop_size=self.crop_size,
        preprocess=utils.preprocess)


class GoalNaiveTransporterAgent(TransporterAgent):
  """Naive version which stacks current and goal images through normal Transport."""

  def __init__(self, name, task, n_rotations=36):
    super().__init__(name, task, n_rotations)

    # Stack the goal image for the vanilla Transport module.
    t_shape = (self.in_shape[0], self.in_shape[1],
               int(self.in_shape[2] * 2))

    self.attention = Attention(
        in_shape=self.in_shape,
        n_rotations=1,
        preprocess=utils.preprocess)
    self.transport = Transport(
        in_shape=t_shape,
        n_rotations=self.n_rotations,
        crop_size=self.crop_size,
        preprocess=utils.preprocess,
        per_pixel_loss=False,
        use_goal_image=True)
