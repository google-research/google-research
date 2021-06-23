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

"""Defines the models.

The base of the network is a vision model that embeds an
RGB image to a flat vector which serves as a visual description.

The head of the network turns the visual description into a pose.  It must have
a method to compute the loss and to predict a pose given a visual description
vector.  For the implicit function based approach, the random sampling is all
contained within the class definition.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import re

from absl import logging
import numpy as np
import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfg


tfkl = tf.keras.layers
eps = 1e-9


def create_vision_model(image_dims=(224, 224, 3)):
  """Loads a pre-trained resnet for the vision model.

  Args:
    image_dims: The input shape.

  Returns:
    The vision model and the length of the visual description vector.
  """
  base_model = tf.keras.applications.ResNet101V2(
      weights='imagenet', include_top=False, input_shape=image_dims)
  inp = tf.keras.layers.Input(shape=image_dims)
  x = base_model(inp)
  x = tfkl.GlobalAveragePooling2D()(x)
  len_visual_description = x.shape[-1]
  vision_model = tf.keras.Model(inp, x)
  return vision_model, len_visual_description


class ImplicitSO3(tfkl.Layer):
  """Represents a distribution over SO(3) as an implicit function.

  Specifically, this is a fully connected network which takes as input
  a visual description vector and a query rotation, and outputs
  a single scalar which can be converted to a log likelihood via normalization.
  By querying many values we can build up a probability distribution over SO(3).

  The query format is a 3x3 rotation matrix, flattened and then positionally
  encoded in each component with multiple frequencies of sinusoids.

  Init args:
    len_visual_description: The length of the visual description vector, which
      is returned with the vision model at creation.
    number_fourier_components: The number of positional encoding frequencies for
      the rotation query.  If zero, positional encoding is not used.
    mlp_layer_sizes: A list of the number of units in each layer of the MLP.
    so3_sampling_mode: 'random' or 'grid'.  This is only efficacious during
     training, and determines how the normalization is computed.  'random'
     corresponds to sampling uniformly over the space of rotations, which is
     simple and less constrained but results in inexact normalization.  'grid'
     uses equivolumetric grids generated using healpix as a starting point.
     These grids are stored so each size only needs to be computed once per run.
    number_train_queries: The number of queries to use during training, which
      populate SO(3) to approximate the normalization of the likelihoods.  If
      so3_sampling_mode is 'grid', the nearest grid size in log space is used.
    number_eval_queries: The number of queries to use during evaluation, which
      is always a grid sampling (for proper normalization).

  """

  def __init__(self, len_visual_description,
               number_fourier_components, mlp_layer_sizes,
               so3_sampling_mode, number_train_queries,
               number_eval_queries, **kwargs):
    super().__init__(**kwargs)
    self.len_visual_description = len_visual_description
    # Rotations are represented as flattened 3x3 orthonormal matrices.
    self.len_rotation = 9
    self.number_fourier_components = number_fourier_components
    self.frequencies = tf.range(number_fourier_components, dtype=tf.float32)
    self.frequencies = tf.math.pow(2., self.frequencies)
    self.so3_sampling_mode = so3_sampling_mode
    self.number_train_queries = number_train_queries
    self.number_eval_queries = number_eval_queries
    if number_fourier_components == 0:
      self.len_query = self.len_rotation
    else:
      self.len_query = self.len_rotation * number_fourier_components * 2

    self.grids = {}

    # The grids will be populated on-demand.
    if self.so3_sampling_mode == 'grid':
      self.get_closest_available_grid(self.number_train_queries)
    self.get_closest_available_grid(self.number_eval_queries)

    input_visual = tfkl.Input(shape=(len_visual_description,))
    visual_embedding = tfkl.Dense(mlp_layer_sizes[0])(input_visual)
    input_query = tfkl.Input(shape=(None, self.len_query,))
    query_embedding = tfkl.Dense(mlp_layer_sizes[0])(input_query)

    # Broadcast sum to combine inputs.
    output = visual_embedding[:, tf.newaxis] + query_embedding

    output = tfkl.ReLU()(output)

    for num_units in mlp_layer_sizes[1:]:
      output = tfkl.Dense(num_units, 'relu')(output)
    output = tfkl.Dense(1)(output)
    self.implicit_model = tf.keras.models.Model(
        inputs=[input_visual, input_query],
        outputs=output)
    self.mlp_layer_sizes = mlp_layer_sizes

  def get_config(self):
    """For saving purposes."""
    config = super().get_config().copy()
    config.update({
        'len_visual_description': self.len_visual_description,
        'number_fourier_components': self.number_fourier_components,
        'mlp_layer_sizes': self.mlp_layer_sizes,
        'so3_sampling_mode': self.so3_sampling_mode,
        'number_train_queries': self.number_train_queries,
        'number_eval_queries': self.number_eval_queries,
    })
    return config

  def call(self, inp):
    return self.implicit_model(inp)

  def predict_probability(self, vision_description, rotation_matrix,
                          training=False):
    """Predict the probability of the rotation, given the vision description.

    Rotate the grids to make sure they contain the rotation matrix. It requires
    different queries per element in the batch, increasing the # of ops in the
    first MLP layer.

    Args:
      vision_description: The feature vectors of the batch of images.
      rotation_matrix: The query rotation matrices.
      training: True or False; determines how many queries to use for
        normalization and whether to use an equivolumetric grid.

    Returns:
      The probability density at the rotation given by rotation_matrix.
    """
    if training:
      query_rotations = self.generate_queries(
          self.number_train_queries, mode=self.so3_sampling_mode)
    else:
      query_rotations = self.generate_queries(
          self.number_eval_queries, mode='grid')
    delta_rot = tf.transpose(query_rotations[-1]) @ rotation_matrix
    query_rotations = tf.einsum('aij,bjk->baik', query_rotations, delta_rot)
    shape = query_rotations.shape
    query_rotations = tf.reshape(query_rotations,
                                 [shape[0], shape[1], self.len_rotation])
    query_rotations = self.positional_encoding(query_rotations)
    logits = self.implicit_model([vision_description, query_rotations],
                                 training=training)[Ellipsis, 0]

    probabilities = tf.nn.softmax(logits, axis=-1)
    # Scale by the volume per grid point.
    probabilities *= np.pi**2 / query_rotations.shape[1]
    # The final query rotation is the rotation we care about.
    return probabilities[:, -1]

  def predict_rotation(self, vision_descriptions, gradient_ascent=False):
    """Returns a predicted rotation given the vision description.

    The mode of the distribution is approximated, found either as the argmax
    over a set of samples, or by running gradient ascent on the probability with
    the sample argmax as the starting point.

    Args:
      vision_descriptions: The feature vectors for the images on which to run
        pose estimation.
      gradient_ascent: True or False; whether to perform gradient ascent after
        finding the argmax over the sample rotations, to squeeze out a little
        more performance.

    Returns:
      A tensor of rotations, shape [N, 3, 3] where N is the number of vision
      descriptions.
    """
    query_rotations = self.generate_queries(self.number_eval_queries,
                                            mode='grid')
    query_rotations_inp = tf.reshape(query_rotations, [-1, self.len_rotation])
    query_rotations_inp = self.positional_encoding(query_rotations_inp)
    logits = self.implicit_model([vision_descriptions, query_rotations_inp],
                                 training=False)[Ellipsis, 0]

    max_inds = tf.argmax(logits, axis=-1)
    max_rotations = tf.gather(query_rotations, max_inds)

    if not gradient_ascent:
      max_rotations = tf.reshape(max_rotations, [-1, 3, 3])
      return max_rotations

    # Perform gradient ascent, using the argmax rotations as starting seeds.
    # These parameters were found to work well for the scenarios tested, but
    # other cases may benefit from tuning these.
    update_step_size = 1e-4
    number_optimization_steps = 100
    # Convert to quaternions to enforce normalization.
    query_quaternions = tfg.quaternion.from_rotation_matrix(max_rotations)
    query_quaternions = tf.Variable(query_quaternions, trainable=True)

    @tf.function
    def gradient_ascent_step(query_quaternions):
      with tf.GradientTape() as tape:
        query_rotations_inp = tfg.rotation_matrix_3d.from_quaternion(
            query_quaternions)
        query_rotations_inp = tf.reshape(query_rotations_inp,
                                         [-1, self.len_rotation])
        query_rotations_inp = self.positional_encoding(query_rotations_inp)
        logits = self.implicit_model([vision_descriptions, query_rotations_inp],
                                     training=False)
        logits = tf.linalg.diag_part(logits[Ellipsis, 0])
        # Maximize the network output without caring about normalization
        loss = -tf.reduce_mean(logits)
      grads = tape.gradient(loss, query_quaternions)
      query_quaternions.assign_add(-grads * update_step_size)
      query_quaternions.assign(tfg.quaternion.normalize(query_quaternions))
      return

    for _ in range(number_optimization_steps):
      gradient_ascent_step(query_quaternions)
    max_rotations = tfg.rotation_matrix_3d.from_quaternion(query_quaternions)
    return max_rotations

  def compute_loss(self, vision_description, rotation_matrix_gt):
    """Return the negative log likelihood of the ground truth rotation matrix.

    Args:
      vision_description: The vector representations of a batch of images.
      rotation_matrix_gt: The ground truth rotation matrices associated with the
        batch of images.

    Returns:
      A scalar, the loss of the batch.
    """
    probs = self.predict_probability(
        vision_description, rotation_matrix_gt, training=True)
    loss = -tf.reduce_mean(tf.math.log(probs))
    return loss

  def get_closest_available_grid(self, number_queries=None):
    if not number_queries:
      number_queries = self.number_eval_queries
    # HEALPix-SO(3) is defined only on 72 * 8^N points; we find the closest
    # valid grid size (in log space) to the requested size.
    # The largest grid size we consider has 19M points.
    grid_sizes = 72*8**np.arange(7)
    size = grid_sizes[
        np.argmin(np.abs(np.log(number_queries) - np.log(grid_sizes)))
        ]
    if self.grids.get(size) is not None:
      return self.grids[size]
    else:
      logging.info('Using grid of size %d. Requested was %d.', size,
                   number_queries)
      grid_created = False

      if not grid_created:
        self.grids[size] = np.float32(generate_healpix_grid(size=size))

      return self.grids[size]

  def generate_queries(self, number_queries, mode='random'):
    """Generate query rotations from SO(3).

    Args:
      number_queries: The number of queries.
      mode: 'random' or 'grid'; determines whether to generate rotations from
        the uniform distribution over SO(3), or use an equivolumetric grid.

    Returns:
      A tensor of rotation matrices, shape [num_queries, 3, 3].
    """
    if mode == 'random':
      return self.generate_queries_random(number_queries)
    elif mode == 'grid':
      return self.get_closest_available_grid(number_queries)

  def generate_queries_random(self, number_queries):
    """Generate rotation matrices from SO(3) uniformly at random.

    Args:
      number_queries: The number of queries.
    Returns:
      A tensor of shape [number_queries, 3, 3].
    """
    random_quaternions = tfg.quaternion.normalized_random_uniform(
        (number_queries,))
    random_rotations = tfg.rotation_matrix_3d.from_quaternion(
        random_quaternions)
    random_rotations = tf.cast(random_rotations, tf.float32)
    return random_rotations

  def positional_encoding(self, query_rotations):
    """This handles the positional encoding.

    Args:
      query_rotations: tensor of shape [N, len_rotation] or
        [bs, N, len_rotation].

    Returns:
      Tensor of shape [N, len_query] or [bs, N, len_query].
    """
    if self.frequencies.shape[0] == 0:
      return query_rotations

    def _enc(freq):
      return tf.concat(
          [tf.math.sin(query_rotations * freq),
           tf.math.cos(query_rotations * freq)], -1)
    query_rotations = tf.map_fn(_enc, self.frequencies)
    query_shape = tf.shape(query_rotations)
    if len(query_shape) == 4:
      query_rotations = tf.transpose(query_rotations, [1, 2, 0, 3])
      shape = tf.concat([query_shape[1:3], [self.len_query]], axis=0)
      query_rotations = tf.reshape(query_rotations, shape)
    else:
      query_rotations = tf.transpose(query_rotations, [1, 0, 2])
      query_rotations = tf.reshape(query_rotations, [-1, self.len_query])
    return query_rotations

  def output_pdf(self, vision_description, num_queries=None,
                 query_rotations=None):
    """Returns a normalized distribution over pose, given a vision description.

    Args:
      vision_description: A batch of feature vectors, representing the images on
        which to estimate the pose.
      num_queries: The number of queries to evaluate the probability for.
      query_rotations: If supplied, these rotations will be used to evaluate the
        distribution and normalize it, instead using the kwarg num_queries.
    Returns:
      Both the rotations and their corresponding probabilities.
    """

    if num_queries is None:
      num_queries = self.number_eval_queries
    if query_rotations is None:
      query_rotations = self.get_closest_available_grid(num_queries)

    query_rotations_enc = tf.reshape(query_rotations, [-1, self.len_rotation])
    query_rotations_enc = self.positional_encoding(query_rotations_enc)
    log_probabilities = self.implicit_model(
        [vision_description, query_rotations_enc], training=False)[Ellipsis, 0]
    probabilities = tf.nn.softmax(log_probabilities, axis=-1)
    return query_rotations, probabilities


def generate_healpix_grid(recursion_level=None, size=None):
  """Generates an equivolumetric grid on SO(3) following Yershova et al. (2010).

  Uses a Healpix grid on the 2-sphere as a starting point and then tiles it
  along the 'tilt' direction 6*2**recursion_level times over 2pi.

  Args:
    recursion_level: An integer which determines the level of resolution of the
      grid.  The final number of points will be 72*8**recursion_level.  A
      recursion_level of 2 (4k points) was used for training and 5 (2.4M points)
      for evaluation.
    size: A number of rotations to be included in the grid.  The nearest grid
      size in log space is returned.

  Returns:
    (N, 3, 3) array of rotation matrices, where N=72*8**recursion_level.
  """
  import healpy as hp  # pylint: disable=g-import-not-at-top

  assert not(recursion_level is None and size is None)
  if size:
    recursion_level = max(int(np.round(np.log(size/72.)/np.log(8.))), 0)
  number_per_side = 2**recursion_level
  number_pix = hp.nside2npix(number_per_side)
  s2_points = hp.pix2vec(number_per_side, np.arange(number_pix))
  s2_points = np.stack([*s2_points], 1)

  # Take these points on the sphere and
  azimuths = np.arctan2(s2_points[:, 1], s2_points[:, 0])
  tilts = np.linspace(0, 2*np.pi, 6*2**recursion_level, endpoint=False)
  polars = np.arccos(s2_points[:, 2])
  grid_rots_mats = []
  for tilt in tilts:
    # Build up the rotations from Euler angles, zyz format
    rot_mats = tfg.rotation_matrix_3d.from_euler(
        np.stack([azimuths,
                  np.zeros(number_pix),
                  np.zeros(number_pix)], 1))
    rot_mats = rot_mats @ tfg.rotation_matrix_3d.from_euler(
        np.stack([np.zeros(number_pix),
                  np.zeros(number_pix),
                  polars], 1))
    rot_mats = rot_mats @ tf.expand_dims(
        tfg.rotation_matrix_3d.from_euler([tilt, 0., 0.]), 0)
    grid_rots_mats.append(rot_mats)

  grid_rots_mats = np.concatenate(grid_rots_mats, 0)
  return grid_rots_mats
