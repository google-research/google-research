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

"""Algorithms for pose representation learning."""

import tensorflow as tf

from poem.core import distance_utils
from poem.core import keypoint_utils
from poem.cv_mim import losses
from poem.cv_mim import models

# Model parameters.
MODEL_LINEAR_HIDDEN_DIM = 1024
MODEL_LINEAR_NUM_RESIDUAL_BLOCKS = 2
MODEL_LINEAR_NUM_LAYERS_PER_BLOCK = 2
MODEL_LINEAR_WEIGHT_MAX_NORM = 0.0
MODEL_LINEAR_DROPOUT_RATE = 0.25
MODEL_LINEAR_SHARED_LAYERS = ('flatten', 'fc0')

# Maximum 2D/3D MPJPE for two poses to be considered as positive match.
MAX_POSITIVE_KEYPOINT_MPJPE_2D = None
MAX_POSITIVE_KEYPOINT_MPJPE_3D = None

# Fusion operation type.
TYPE_FUSION_OP_CAT = 'CAT'  # Concatenate
TYPE_FUSION_OP_POE = 'POE'  # Product of Experts
TYPE_FUSION_OP_MOE = 'MOE'  # Mixture of Experts
SUPPORTED_FUSION_OP_TYPES = [
    TYPE_FUSION_OP_CAT, TYPE_FUSION_OP_POE, TYPE_FUSION_OP_MOE
]

# Algorithm type.
TYPE_ALGORITHM_ALIGN = 'ALIGN'
TYPE_ALGORITHM_DISENTANGLE = 'DISENTANGLE'
TYPE_ALGORITHM_AUTOENCIDER = 'AUTOENCODER'
TYPE_ALGORITHM_INFOMAX = 'INFOMAX'
SUPPORTED_ALGORITHM_TYPES = [
    TYPE_ALGORITHM_ALIGN, TYPE_ALGORITHM_DISENTANGLE,
    TYPE_ALGORITHM_AUTOENCIDER, TYPE_ALGORITHM_INFOMAX
]


def compute_positive_indicator_matrix(anchors, matches, distance_fn,
                                      max_positive_distance):
  """Computes all-pair positive indicator matrix.

  Args:
    anchors: A tensor for anchor points. Shape = [num_anchors, ...].
    matches: A tensor for match points. Shape = [num_matches, ...].
    distance_fn: A function handle for computing distance matrix.
    max_positive_distance: A float for the maximum positive distance threshold.

  Returns:
    A float tensor for positive indicator matrix. Shape = [num_anchors,
      num_matches].
  """
  distance_matrix = distance_utils.compute_distance_matrix(
      anchors, matches, distance_fn=distance_fn)
  distance_matrix = (distance_matrix + tf.transpose(distance_matrix)) / 2.0
  positive_indicator_matrix = distance_matrix <= max_positive_distance
  return tf.cast(positive_indicator_matrix, dtype=tf.dtypes.float32)


def get_algorithm(algorithm_type,
                  pose_embedding_dim,
                  view_embedding_dim=None,
                  fusion_op_type=TYPE_FUSION_OP_CAT,
                  view_loss_weight=5.0,
                  regularization_loss_weight=1.0,
                  disentangle_loss_weight=0.5,
                  **kwargs):
  """Gets algorithm.

  Args:
    algorithm_type: A string for algorithm type.
    pose_embedding_dim: An integer for dimension of pose embedding.
    view_embedding_dim: An integer for dimension of view embedding.
    fusion_op_type: A string the type of fusion operation.
    view_loss_weight: A float for the weight of view loss.
    regularization_loss_weight: A float for the weight of regularization loss.
    disentangle_loss_weight: A float for the weight of disentangle loss.
    **kwargs: A dictionary for additional arguments.

  Returns:
    An algorithm instance.
  """
  if algorithm_type == TYPE_ALGORITHM_ALIGN:
    return InfoMix(
        embedding_dim=pose_embedding_dim,
        view_loss_weight=view_loss_weight,
        regularization_loss_weight=regularization_loss_weight,
        **kwargs)
  elif algorithm_type == TYPE_ALGORITHM_DISENTANGLE:
    return InfoDisentangle(
        pose_embedding_dim=pose_embedding_dim,
        view_embedding_dim=view_embedding_dim,
        fusion_op_type=fusion_op_type,
        view_loss_weight=view_loss_weight,
        regularization_loss_weight=regularization_loss_weight,
        disentangle_loss_weight=disentangle_loss_weight,
        **kwargs)
  elif algorithm_type == TYPE_ALGORITHM_AUTOENCIDER:
    return AutoEncoder(
        pose_embedding_dim=pose_embedding_dim,
        view_embedding_dim=view_embedding_dim,
        regularization_loss_weight=regularization_loss_weight,
        **kwargs)
  elif algorithm_type == TYPE_ALGORITHM_INFOMAX:
    return InfoMax(
        pose_embedding_dim=pose_embedding_dim,
        view_embedding_dim=view_embedding_dim,
        fusion_op_type=fusion_op_type,
        view_loss_weight=view_loss_weight,
        regularization_loss_weight=regularization_loss_weight,
        **kwargs)
  else:
    raise ValueError('Unknown algorithm: {}'.format(algorithm_type))


def get_optimizers(algorithm_type, learning_rate):
  """Gets optimizers.

  Args:
    algorithm_type: A string for algorithm type.
    learning_rate: A float for learning rate.

  Returns:
    A dictionary of tf.keras.optimizers instances.
  """
  if algorithm_type == TYPE_ALGORITHM_ALIGN:
    optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    return dict(encoder_optimizer=optimizer)
  elif algorithm_type == TYPE_ALGORITHM_DISENTANGLE:
    encoder_optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    estimator_optimizer = tf.keras.optimizers.Adagrad(
        learning_rate=learning_rate)
    discriminator_optimizer = tf.keras.optimizers.Adagrad(
        learning_rate=learning_rate)
    return dict(
        encoder_optimizer=encoder_optimizer,
        estimator_optimizer=estimator_optimizer,
        discriminator_optimizer=discriminator_optimizer)
  elif algorithm_type == TYPE_ALGORITHM_AUTOENCIDER:
    optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    return dict(encoder_optimizer=optimizer)
  elif algorithm_type == TYPE_ALGORITHM_INFOMAX:
    optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    return dict(encoder_optimizer=optimizer)
  else:
    raise ValueError('Unknown algorithm: {}'.format(algorithm_type))


def get_encoder(embedding_dim, embedder_type=models.TYPE_EMBEDDER_POINT):
  """Gets default encoder for InfoMix.

  Args:
    embedding_dim: An integer for the dimension of the embedding.
    embedder_type: A string for the type of the embedder.

  Returns:
    A configured encoder.
  """
  return models.SimpleModel(
      output_shape=(embedding_dim,),
      embedder=embedder_type,
      hidden_dim=MODEL_LINEAR_HIDDEN_DIM,
      num_residual_linear_blocks=MODEL_LINEAR_NUM_RESIDUAL_BLOCKS,
      num_layers_per_block=MODEL_LINEAR_NUM_LAYERS_PER_BLOCK,
      dropout_rate=MODEL_LINEAR_DROPOUT_RATE,
      use_batch_norm=True,
      weight_max_norm=MODEL_LINEAR_WEIGHT_MAX_NORM,
      weight_initializer='he_normal')


class AutoEncoder(tf.keras.Model):
  """Model for auto-encoder."""

  def __init__(self,
               pose_embedding_dim,
               view_embedding_dim,
               embedder_type,
               regularization_loss_weight=1.0):
    """Initializer.

    Args:
      pose_embedding_dim: An integer for the dimension of the pose embedding.
      view_embedding_dim: An integer for the dimension of the view embedding.
      embedder_type: A string for the type of the embedder.
      regularization_loss_weight: A float for the weight of regularization loss.
    """
    super(AutoEncoder, self).__init__()
    self._regularization_loss_weight = regularization_loss_weight
    self._pose_embedding_dim = pose_embedding_dim
    self._view_embedding_dim = view_embedding_dim
    self._mse = tf.keras.losses.MeanSquaredError()

    embedding_dim = pose_embedding_dim + view_embedding_dim
    self.encoder = get_encoder(embedding_dim, embedder_type)

  def build(self, input_shape):
    """Builds the model.

    Args:
      input_shape: A TensorShape for the shape of the input.
    """
    self.decoder = models.SimpleModel(
        output_shape=input_shape[1:],
        embedder=models.TYPE_EMBEDDER_POINT,
        hidden_dim=MODEL_LINEAR_HIDDEN_DIM,
        num_residual_linear_blocks=MODEL_LINEAR_NUM_RESIDUAL_BLOCKS,
        num_layers_per_block=MODEL_LINEAR_NUM_LAYERS_PER_BLOCK,
        dropout_rate=MODEL_LINEAR_DROPOUT_RATE,
        use_batch_norm=True,
        weight_max_norm=MODEL_LINEAR_WEIGHT_MAX_NORM,
        weight_initializer='he_normal')

  def call(self, inputs, training=False):
    """Computes a forward pass.

    Args:
      inputs: An input tensor. Shape = [batch_size, num_joints, 2]
      training: A boolean indicating whether the call is for training or not.

    Returns:
      pose_embeddings: An output tensor for the pose embeddings. Shape =
        [batch_size, pose_embedding_dim].
      view_embeddings: An output tensor for the view embeddings. Shape =
        [batch_size, view_embedding_dim].
    """
    outputs, _ = self.encoder(inputs, training=training)
    pose_embeddings, view_embeddings = tf.split(
        outputs,
        num_or_size_splits=[self._pose_embedding_dim, self._view_embedding_dim],
        axis=-1)
    return pose_embeddings, view_embeddings

  def compute_reconstruction_loss(self, pose_embeddings, view_embeddings,
                                  targets):
    """Computes the reconstruction loss.

    Args:
      pose_embeddings: A tensor for the pose embeddings. Shape = [batch_size,
        pose_embedding_dim].
      view_embeddings: A tensor for the view embeddings. Shape = [batch_size,
        view_embedding_dim].
      targets: A tensor for the targets to be reconstructed.

    Returns:
      A scalar for the reconstruction loss.
    """
    input_embeddings = tf.concat([pose_embeddings, view_embeddings], axis=-1)
    outputs, _ = self.decoder(input_embeddings, training=True)
    return self._mse(targets, outputs)

  def train(self, inputs, encoder_optimizer):
    """Trains the model for one step.

    Args:
      inputs: A list of input tensors containing 2D and 3D keypoints. Shape = [
        batch_size, num_instances, num_joints, 2]
      encoder_optimizer: An optimizer object.

    Returns:
      A dictionary for all losses.
    """
    keypoints_2d, _ = inputs
    anchor_keypoints_2d, positive_keypoints_2d = tf.split(
        keypoints_2d, num_or_size_splits=[1, 1], axis=1)

    anchor_keypoints_2d = tf.squeeze(anchor_keypoints_2d, axis=1)
    positive_keypoints_2d = tf.squeeze(positive_keypoints_2d, axis=1)

    with tf.GradientTape() as tape:
      anchor_pose_embeddings, anchor_view_embeddings = self(
          anchor_keypoints_2d, training=True)
      anchor_regularization_loss = sum(self.encoder.losses)
      positive_pose_embeddings, positive_view_embeddings = self(
          positive_keypoints_2d, training=True)
      positive_regularization_loss = sum(self.encoder.losses)

      anchor_reconstruction_loss = self.compute_reconstruction_loss(
          anchor_pose_embeddings, anchor_view_embeddings, anchor_keypoints_2d)
      positive_reconstruction_loss = self.compute_reconstruction_loss(
          positive_pose_embeddings, positive_view_embeddings,
          positive_keypoints_2d)
      reconstruction_loss = (
          anchor_reconstruction_loss + positive_reconstruction_loss)

      anchor_cross_reconstruction_loss = self.compute_reconstruction_loss(
          positive_pose_embeddings, anchor_view_embeddings, anchor_keypoints_2d)
      positive_cross_reconstruction_loss = self.compute_reconstruction_loss(
          anchor_pose_embeddings, positive_view_embeddings,
          positive_keypoints_2d)
      cross_reconstruction_loss = (
          anchor_cross_reconstruction_loss + positive_cross_reconstruction_loss)

      regularization_loss = self._regularization_loss_weight * (
          anchor_regularization_loss + positive_regularization_loss)
      total_loss = (
          reconstruction_loss + cross_reconstruction_loss + regularization_loss)

    grads = tape.gradient(total_loss, self.trainable_variables)
    encoder_optimizer.apply_gradients(zip(grads, self.trainable_variables))
    encoder_losses = dict(
        total_loss=total_loss,
        reconstruction_loss=reconstruction_loss,
        cross_reconstruction_loss=cross_reconstruction_loss,
        regularization_loss=regularization_loss)

    return dict(encoder=encoder_losses)


class InfoMix(tf.keras.Model):
  """Model for InfoMix."""

  def __init__(self,
               embedding_dim,
               embedder_type,
               view_loss_weight=1.0,
               regularization_loss_weight=1.0):
    """Initializer.

    Args:
      embedding_dim: An integer for the dimension of the embedding.
      embedder_type: A string for the type of the embedder.
      view_loss_weight: A float for the weight of view loss.
      regularization_loss_weight: A float for the weight of regularization loss.
    """
    super(InfoMix, self).__init__()
    self.view_loss_weight = view_loss_weight
    self.regularization_loss_weight = regularization_loss_weight

    self.encoder = get_encoder(embedding_dim, embedder_type)
    self.subencoder = get_encoder(embedding_dim, models.TYPE_EMBEDDER_POINT)
    self.subencoder.blocks = [
        block for block in self.subencoder.blocks
        if block.name not in MODEL_LINEAR_SHARED_LAYERS
    ]

  def call(self, inputs, training=False):
    """Computes a forward pass.

    Args:
      inputs: An input tensor.
      training: A boolean indicating whether the call is for training or not.

    Returns:
      encoder_outputs: An output tensor of the encoder. Shape = [batch_size, 1,
        embedding_dim].
      subencoder_outputs: An output tensor of the subencoder. Shape =
        [batch_size, 1, embedding_dim].
    """
    encoder_outputs, encoder_activations = self.encoder(
        inputs, training=training)
    subencoder_outputs, _ = self.subencoder(
        encoder_activations['fc0'], training=training)

    encoder_outputs = tf.expand_dims(encoder_outputs, axis=1)
    subencoder_outputs = tf.expand_dims(subencoder_outputs, axis=1)
    return encoder_outputs, subencoder_outputs

  def compute_representation_loss(self, inputs, positive_indicator_matrix):
    """Computes the representation loss.

    Args:
      inputs: An input tensor. Shape = [batch_size, num_points, ...].
      positive_indicator_matrix: A tensor for positive indicator matrix. The
        positive correspondences will have value 1.0 and otherwise 0.0. Shape =
        [batch_size, batch_size].

    Returns:
      output_embeddings: A tensor for the embedding. Shape = [batch_size, 1,
        embedding_dim].
      representation_loss: A scalar for the representation loss.
      regularization_loss: A scalar for the regularization loss.
    """
    output_embeddings, subencoder_output_embeddings = self(
        inputs, training=True)
    representation_loss = losses.compute_fenchel_dual_loss(
        subencoder_output_embeddings, output_embeddings,
        losses.TYPE_MEASURE_JSD, positive_indicator_matrix)
    regularization_loss = sum(self.encoder.losses)
    return output_embeddings, representation_loss, regularization_loss

  def train(self, inputs, encoder_optimizer):
    """Trains the model for one step.

    Args:
      inputs: A list of input tensors containing 2D and 3D keypoints. Shape = [
        batch_size, num_instances, num_joints, {2|3}]
      encoder_optimizer: An optimizer object.

    Returns:
      A dictionary for all losses.
    """
    keypoints_2d, keypoints_3d = inputs
    anchor_keypoints_2d, positive_keypoints_2d = tf.split(
        keypoints_2d, num_or_size_splits=[1, 1], axis=1)
    anchor_keypoints_3d, positive_keypoints_3d = tf.split(
        keypoints_3d, num_or_size_splits=[1, 1], axis=1)

    anchor_keypoints_2d = tf.squeeze(anchor_keypoints_2d, axis=1)
    positive_keypoints_2d = tf.squeeze(positive_keypoints_2d, axis=1)
    anchor_keypoints_3d = tf.squeeze(anchor_keypoints_3d, axis=1)
    positive_keypoints_3d = tf.squeeze(positive_keypoints_3d, axis=1)

    if MAX_POSITIVE_KEYPOINT_MPJPE_2D is None:
      anchor_indicator_matrix = None
      positive_indicator_matrix = None
    else:
      anchor_indicator_matrix = compute_positive_indicator_matrix(
          anchor_keypoints_2d,
          anchor_keypoints_2d,
          distance_fn=keypoint_utils.compute_mpjpes,
          max_positive_distance=MAX_POSITIVE_KEYPOINT_MPJPE_2D)
      positive_indicator_matrix = compute_positive_indicator_matrix(
          positive_keypoints_2d,
          positive_keypoints_2d,
          distance_fn=keypoint_utils.compute_mpjpes,
          max_positive_distance=MAX_POSITIVE_KEYPOINT_MPJPE_2D)

    if MAX_POSITIVE_KEYPOINT_MPJPE_3D is None:
      view_indicator_matrix = None
    else:
      view_indicator_matrix = compute_positive_indicator_matrix(
          anchor_keypoints_3d,
          positive_keypoints_3d,
          distance_fn=keypoint_utils.compute_procrustes_aligned_mpjpes,
          max_positive_distance=MAX_POSITIVE_KEYPOINT_MPJPE_3D)

    with tf.GradientTape() as tape:
      (anchor_embeddings, anchor_representation_loss,
       anchor_regularization_loss) = self.compute_representation_loss(
           anchor_keypoints_2d, anchor_indicator_matrix)
      (positive_embeddings, positive_representation_loss,
       positive_regularization_loss) = self.compute_representation_loss(
           positive_keypoints_2d, positive_indicator_matrix)

      representation_loss = (
          anchor_representation_loss + positive_representation_loss)
      regularization_loss = self.regularization_loss_weight * (
          anchor_regularization_loss + positive_regularization_loss)
      total_loss = representation_loss + regularization_loss

      view_loss = self.view_loss_weight * losses.compute_fenchel_dual_loss(
          anchor_embeddings, positive_embeddings, losses.TYPE_MEASURE_JSD,
          view_indicator_matrix) * 2.0
      total_loss += view_loss

    grads = tape.gradient(total_loss, self.trainable_variables)
    encoder_optimizer.apply_gradients(zip(grads, self.trainable_variables))
    encoder_losses = dict(
        total_loss=total_loss,
        representation_loss=representation_loss,
        regularization_loss=regularization_loss,
        view_loss=view_loss)

    return dict(encoder=encoder_losses)


class InfoDisentangle(tf.keras.Model):
  """Model for InfoDisentangle."""

  def __init__(self,
               pose_embedding_dim,
               view_embedding_dim,
               fusion_op_type,
               embedder_type,
               view_loss_weight=1.0,
               disentangle_loss_weight=1.0,
               regularization_loss_weight=1.0):
    """Initializer.

    Args:
      pose_embedding_dim: An integer for the dimension of the pose embedding.
      view_embedding_dim: An integer for the dimension of the view embedding.
      fusion_op_type: A string the type of fusion operation.
      embedder_type: A string for the type of the embedder.
      view_loss_weight: A float for the weight of view loss.
      disentangle_loss_weight: A float for the weight of disentangle loss.
      regularization_loss_weight: A float for the weight of regularization loss.
    """
    super(InfoDisentangle, self).__init__()
    self.fusion_op_type = fusion_op_type
    self.view_loss_weight = view_loss_weight
    self.disentangle_loss_weight = disentangle_loss_weight
    self.regularization_loss_weight = regularization_loss_weight

    self.pose_embedding_dim = pose_embedding_dim
    self.view_embedding_dim = view_embedding_dim
    encoder_embedding_dim = pose_embedding_dim + view_embedding_dim

    if self.fusion_op_type == TYPE_FUSION_OP_CAT:
      subencoder_embedding_dim = encoder_embedding_dim
    else:
      if pose_embedding_dim != view_embedding_dim:
        raise ValueError('Eembedding dimensions are not equal.')
      subencoder_embedding_dim = pose_embedding_dim

    self.encoder = get_encoder(encoder_embedding_dim, embedder_type)
    self.subencoder = get_encoder(subencoder_embedding_dim,
                                  models.TYPE_EMBEDDER_POINT)
    self.subencoder.blocks = [
        block for block in self.subencoder.blocks
        if block.name not in MODEL_LINEAR_SHARED_LAYERS
    ]

    self.inter_likelihood_estimator = models.LikelihoodEstimator(
        self.view_embedding_dim)
    self.intra_likelihood_estimator = models.LikelihoodEstimator(
        self.view_embedding_dim)
    self.discriminator = tf.keras.Sequential([
        tf.keras.layers.Dense(encoder_embedding_dim, activation='relu'),
        tf.keras.layers.Dense(encoder_embedding_dim // 2, activation='relu'),
        tf.keras.layers.Dense(1),
    ])

  def call(self, inputs, training=False):
    """Computes a forward pass.

    Args:
      inputs: An input tensor.
      training: A boolean indicating whether the call is for training or not.

    Returns:
      encoder_outputs: An output tensor of the encoder. Shape = [batch_size, 1,
        embedding_dim].
      subencoder_outputs: An output tensor of the subencoder. Shape =
        [batch_size, 1, embedding_dim].
    """
    encoder_outputs, encoder_activations = self.encoder(
        inputs, training=training)
    subencoder_outputs, _ = self.subencoder(
        encoder_activations['fc0'], training=training)

    encoder_outputs = tf.expand_dims(encoder_outputs, axis=1)
    subencoder_outputs = tf.expand_dims(subencoder_outputs, axis=1)
    return encoder_outputs, subencoder_outputs

  def compute_representation_loss(self, inputs, positive_indicator_matrix):
    """Computes the representation loss.

    Args:
      inputs: An input tensor. Shape = [batch_size, num_points, ...].
      positive_indicator_matrix: A tensor for positive indicator matrix. The
        positive correspondences will have value 1.0 and otherwise 0.0. Shape =
        [batch_size, batch_size].

    Returns:
      output_embeddings: A tensor for the embedding. Shape = [batch_size, 1,
        embedding_dim].
      representation_loss: A scalar for the representation loss.
      regularization_loss: A scalar for the regularization loss.
    """
    output_embeddings, subencoder_output_embeddings = self(
        inputs, training=True)

    if self.fusion_op_type == TYPE_FUSION_OP_CAT:
      fusion_embeddings = output_embeddings
    else:
      pose_embeddings, view_embeddings = tf.split(
          output_embeddings,
          num_or_size_splits=[self.pose_embedding_dim, self.view_embedding_dim],
          axis=-1)
      if self.fusion_op_type == TYPE_FUSION_OP_POE:
        fusion_embeddings = pose_embeddings * view_embeddings
      elif self.fusion_op_type == TYPE_FUSION_OP_MOE:
        fusion_embeddings = 0.5 * (pose_embeddings + view_embeddings)
      else:
        raise ValueError('Unknown fusion operation: {}'.format(
            self.fusion_op_type))

    representation_loss = losses.compute_fenchel_dual_loss(
        subencoder_output_embeddings, fusion_embeddings,
        losses.TYPE_MEASURE_JSD, positive_indicator_matrix)
    regularization_loss = sum(self.encoder.losses)
    return output_embeddings, representation_loss, regularization_loss

  def compute_uniform_prior_loss(self, pose_embeddings, view_embeddings):
    """Computes the prior loss.

    We match the embeddnings to a uniform distribution U(0, 1) with adversarial
    learning. A discriminator is used for matching the embeddning distribution
    (fake) to a uniform one (real). We also compute the discriminator loss in
    this function.

    Args:
      pose_embeddings: A tensor for pose embeddings. Shape = [batch_size,
        embedding_dim].
      view_embeddings: A tensor for view embeddings. Shape = [batch_size,
        embedding_dim].

    Returns:
      prior_loss: A scalar for the prior loss.
      discriminator_loss: A scalar for the discriminator loss.
    """
    fake_inputs = tf.concat([pose_embeddings, view_embeddings], axis=-1)
    # Scale the range of embeddings to (0, 1).
    fake_inputs = tf.nn.sigmoid(fake_inputs)
    real_inputs = tf.random.uniform(shape=fake_inputs.shape)

    discriminator_loss, _, fake_outputs = losses.compute_discriminator_loss(
        self.discriminator, real_inputs, fake_inputs)
    prior_loss = losses.compute_generator_loss(fake_outputs,
                                               losses.TYPE_GENERATOR_LOSS_NS)
    return prior_loss, discriminator_loss

  def train(self, inputs, encoder_optimizer, estimator_optimizer,
            discriminator_optimizer):
    """Trains the model for one step.

    Args:
      inputs: A list of input tensors containing 2D and 3D keypoints. Shape = [
        batch_size, num_instances, num_joints, {2|3}]
      encoder_optimizer: An optimizer object for ecnoder.
      estimator_optimizer: An optimizer object for estimator.
      discriminator_optimizer: An optimizer object for discriminator.

    Returns:
      A dictionary for all losses.
    """
    keypoints_2d, keypoints_3d = inputs
    anchor_keypoints_2d, positive_keypoints_2d = tf.split(
        keypoints_2d, num_or_size_splits=[1, 1], axis=1)
    anchor_keypoints_3d, positive_keypoints_3d = tf.split(
        keypoints_3d, num_or_size_splits=[1, 1], axis=1)

    anchor_keypoints_2d = tf.squeeze(anchor_keypoints_2d, axis=1)
    positive_keypoints_2d = tf.squeeze(positive_keypoints_2d, axis=1)
    anchor_keypoints_3d = tf.squeeze(anchor_keypoints_3d, axis=1)
    positive_keypoints_3d = tf.squeeze(positive_keypoints_3d, axis=1)

    if MAX_POSITIVE_KEYPOINT_MPJPE_2D is None:
      anchor_indicator_matrix = None
      positive_indicator_matrix = None
    else:
      anchor_indicator_matrix = compute_positive_indicator_matrix(
          anchor_keypoints_2d,
          anchor_keypoints_2d,
          distance_fn=keypoint_utils.compute_mpjpes,
          max_positive_distance=MAX_POSITIVE_KEYPOINT_MPJPE_2D)
      positive_indicator_matrix = compute_positive_indicator_matrix(
          positive_keypoints_2d,
          positive_keypoints_2d,
          distance_fn=keypoint_utils.compute_mpjpes,
          max_positive_distance=MAX_POSITIVE_KEYPOINT_MPJPE_2D)

    if MAX_POSITIVE_KEYPOINT_MPJPE_3D is None:
      view_indicator_matrix = None
    else:
      view_indicator_matrix = compute_positive_indicator_matrix(
          anchor_keypoints_3d,
          positive_keypoints_3d,
          distance_fn=keypoint_utils.compute_procrustes_aligned_mpjpes,
          max_positive_distance=MAX_POSITIVE_KEYPOINT_MPJPE_3D)

    def compute_estimator_loss(estimator, x, y, positive_indicator_matrix):
      x_mean, x_logvar = estimator(x, training=True)
      likelihood = losses.compute_log_likelihood(x_mean, x_logvar, y)
      bound = losses.compute_contrastive_log_ratio(x_mean, x_logvar, y,
                                                   positive_indicator_matrix)
      return likelihood, bound

    with tf.GradientTape() as encoder_tape, tf.GradientTape(
    ) as estimator_tape, tf.GradientTape() as discriminator_tape:
      (anchor_embeddings, anchor_representation_loss,
       anchor_regularization_loss) = self.compute_representation_loss(
           anchor_keypoints_2d, anchor_indicator_matrix)
      (positive_embeddings, positive_representation_loss,
       positive_regularization_loss) = self.compute_representation_loss(
           positive_keypoints_2d, positive_indicator_matrix)

      representation_loss = (
          anchor_representation_loss + positive_representation_loss)
      regularization_loss = self.regularization_loss_weight * (
          anchor_regularization_loss + positive_regularization_loss)
      encoder_total_loss = representation_loss + regularization_loss

      anchor_pose_embeddings, anchor_view_embeddings = tf.split(
          anchor_embeddings,
          num_or_size_splits=[self.pose_embedding_dim, self.view_embedding_dim],
          axis=-1)
      positive_pose_embeddings, positive_view_embeddings = tf.split(
          positive_embeddings,
          num_or_size_splits=[self.pose_embedding_dim, self.view_embedding_dim],
          axis=-1)

      view_loss = self.view_loss_weight * losses.compute_fenchel_dual_loss(
          anchor_pose_embeddings, positive_pose_embeddings,
          losses.TYPE_MEASURE_JSD, view_indicator_matrix) * 2.0
      encoder_total_loss += view_loss

      anchor_view_embeddings = tf.squeeze(anchor_view_embeddings, axis=1)
      anchor_pose_embeddings = tf.squeeze(anchor_pose_embeddings, axis=1)
      positive_view_embeddings = tf.squeeze(positive_view_embeddings, axis=1)
      positive_pose_embeddings = tf.squeeze(positive_pose_embeddings, axis=1)

      (anchor_prior_loss,
       anchor_discriminator_loss) = self.compute_uniform_prior_loss(
           anchor_pose_embeddings, anchor_view_embeddings)
      (positive_prior_loss,
       positive_discriminator_loss) = self.compute_uniform_prior_loss(
           positive_pose_embeddings, positive_view_embeddings)
      prior_loss = anchor_prior_loss + positive_prior_loss
      encoder_total_loss += prior_loss
      discriminator_total_loss = (
          anchor_discriminator_loss + positive_discriminator_loss)

      inter_likelihood, inter_bound = compute_estimator_loss(
          self.inter_likelihood_estimator, anchor_view_embeddings,
          positive_view_embeddings, view_indicator_matrix)
      anchor_intra_likelihood, anchor_intra_bound = compute_estimator_loss(
          self.intra_likelihood_estimator, anchor_pose_embeddings,
          anchor_view_embeddings, anchor_indicator_matrix)
      positive_intra_likelihood, positive_intra_bound = compute_estimator_loss(
          self.intra_likelihood_estimator, positive_pose_embeddings,
          positive_view_embeddings, positive_indicator_matrix)

      intra_bound_loss = self.disentangle_loss_weight * (
          anchor_intra_bound + positive_intra_bound)
      inter_bound_loss = self.disentangle_loss_weight * inter_bound
      disentangle_loss = intra_bound_loss + inter_bound_loss * 2.0
      encoder_total_loss += disentangle_loss

      anchor_intra_likelihood_loss = -anchor_intra_likelihood
      positive_intra_likelihood_loss = -positive_intra_likelihood
      inter_likelihood_loss = -inter_likelihood
      estimator_total_loss = (
          anchor_intra_likelihood_loss + positive_intra_likelihood_loss +
          inter_likelihood_loss * 2.0)

    encoder_trainable_variables = (
        self.encoder.trainable_variables + self.subencoder.trainable_variables)
    encoder_grads = encoder_tape.gradient(encoder_total_loss,
                                          encoder_trainable_variables)
    estimator_trainable_variables = (
        self.intra_likelihood_estimator.trainable_variables +
        self.inter_likelihood_estimator.trainable_variables)
    estimator_grads = estimator_tape.gradient(estimator_total_loss,
                                              estimator_trainable_variables)

    discriminator_trainable_variables = self.discriminator.trainable_variables
    discriminator_grads = discriminator_tape.gradient(
        discriminator_total_loss, discriminator_trainable_variables)

    encoder_optimizer.apply_gradients(
        zip(encoder_grads, encoder_trainable_variables))
    estimator_optimizer.apply_gradients(
        zip(estimator_grads, estimator_trainable_variables))
    discriminator_optimizer.apply_gradients(
        zip(discriminator_grads, discriminator_trainable_variables))

    encoder_losses = dict(
        total_loss=encoder_total_loss,
        representation_loss=representation_loss,
        regularization_loss=regularization_loss,
        view_loss=view_loss,
        prior_loss=prior_loss,
        disentangle_loss=disentangle_loss,
        intra_bound_loss=intra_bound_loss,
        inter_bound_loss=inter_bound_loss)

    estimator_losses = dict(
        total_loss=estimator_total_loss,
        anchor_intra_likelihood_loss=anchor_intra_likelihood_loss,
        positive_intra_likelihood_loss=positive_intra_likelihood_loss,
        inter_likelihood_loss=inter_likelihood_loss)

    discriminator_losses = dict(total_loss=discriminator_total_loss)

    return dict(
        encoder=encoder_losses,
        estimator=estimator_losses,
        discriminator=discriminator_losses)


class InfoMax(tf.keras.Model):
  """Model for InfoMax."""

  def __init__(self,
               pose_embedding_dim,
               view_embedding_dim,
               fusion_op_type,
               embedder_type,
               view_loss_weight=1.0,
               regularization_loss_weight=1.0):
    """Initializer.

    Args:
      pose_embedding_dim: An integer for the dimension of the pose embedding.
      view_embedding_dim: An integer for the dimension of the view embedding.
      fusion_op_type: A string the type of fusion operation.
      embedder_type: A string for the type of the embedder.
      view_loss_weight: A float for the weight of view loss.
      regularization_loss_weight: A float for the weight of regularization loss.
    """
    super(InfoMax, self).__init__()
    self._fusion_op_type = fusion_op_type
    self._view_loss_weight = view_loss_weight
    self._regularization_loss_weight = regularization_loss_weight

    self._pose_embedding_dim = pose_embedding_dim
    self._view_embedding_dim = view_embedding_dim
    encoder_embedding_dim = pose_embedding_dim + view_embedding_dim

    if self._fusion_op_type == TYPE_FUSION_OP_CAT:
      subencoder_embedding_dim = encoder_embedding_dim
    else:
      if pose_embedding_dim != view_embedding_dim:
        raise ValueError('Eembedding dimensions are not equal.')
      subencoder_embedding_dim = pose_embedding_dim

    self.encoder = get_encoder(encoder_embedding_dim, embedder_type)
    self.subencoder = get_encoder(subencoder_embedding_dim,
                                  models.TYPE_EMBEDDER_POINT)
    self.subencoder.blocks = [
        block for block in self.subencoder.blocks
        if block.name not in MODEL_LINEAR_SHARED_LAYERS
    ]

  def call(self, inputs, training=False):
    """Computes a forward pass.

    Args:
      inputs: An input tensor.
      training: A boolean indicating whether the call is for training or not.

    Returns:
      pose_embeddings: A tensor of the pose embeddings. Shape = [batch_size, 1,
        embedding_dim].
      view_embeddings: A tensor of the view embeddings. Shape = [batch_size, 1,
        embedding_dim].
      subencoder_embeddings: An output tensor of the subencoder. Shape =
        [batch_size, 1, embedding_dim].
    """
    encoder_embeddings, encoder_activations = self.encoder(
        inputs, training=training)
    subencoder_embeddings, _ = self.subencoder(
        encoder_activations['fc0'], training=training)

    encoder_embeddings = tf.expand_dims(encoder_embeddings, axis=1)
    pose_embeddings, view_embeddings = tf.split(
        encoder_embeddings,
        num_or_size_splits=[self._pose_embedding_dim, self._view_embedding_dim],
        axis=-1)
    subencoder_embeddings = tf.expand_dims(subencoder_embeddings, axis=1)
    return pose_embeddings, view_embeddings, subencoder_embeddings

  def compute_representation_loss(self, pose_embeddings, view_embeddings,
                                  subencoder_embeddings,
                                  positive_indicator_matrix):
    """Computes the representation loss.

    Args:
      pose_embeddings: A tensor for the pose embeddings. Shape = [batch_size,
        1, pose_embedding_dim].
      view_embeddings: A tensor for the view embeddings. Shape = [batch_size,
        1, view_embedding_dim].
      subencoder_embeddings: A tensor for the embedding of the subencoder.
        Shape = [batch_size, 1, embedding_dim].
      positive_indicator_matrix: A tensor for positive indicator matrix. The
        positive correspondences will have value 1.0 and otherwise 0.0. Shape =
        [batch_size, batch_size].

    Returns:
      representation_loss: A scalar for the representation loss.
    """
    if self._fusion_op_type == TYPE_FUSION_OP_CAT:
      fusion_embeddings = tf.concat([pose_embeddings, view_embeddings], axis=-1)
    else:
      if self._fusion_op_type == TYPE_FUSION_OP_POE:
        fusion_embeddings = pose_embeddings * view_embeddings
      elif self._fusion_op_type == TYPE_FUSION_OP_MOE:
        fusion_embeddings = 0.5 * (pose_embeddings + view_embeddings)
      else:
        raise ValueError('Unknown fusion operation: {}'.format(
            self._fusion_op_type))

    representation_loss = losses.compute_fenchel_dual_loss(
        subencoder_embeddings, fusion_embeddings, losses.TYPE_MEASURE_JSD,
        positive_indicator_matrix)
    return representation_loss

  def train(self, inputs, encoder_optimizer):
    """Trains the model for one step.

    Args:
      inputs: A list of input tensors containing 2D and 3D keypoints. Shape = [
        batch_size, num_instances, num_joints, {2|3}]
      encoder_optimizer: An optimizer object for ecnoder.

    Returns:
      A dictionary for all losses.
    """
    keypoints_2d, keypoints_3d = inputs
    anchor_keypoints_2d, positive_keypoints_2d = tf.split(
        keypoints_2d, num_or_size_splits=[1, 1], axis=1)
    anchor_keypoints_3d, positive_keypoints_3d = tf.split(
        keypoints_3d, num_or_size_splits=[1, 1], axis=1)

    anchor_keypoints_2d = tf.squeeze(anchor_keypoints_2d, axis=1)
    positive_keypoints_2d = tf.squeeze(positive_keypoints_2d, axis=1)
    anchor_keypoints_3d = tf.squeeze(anchor_keypoints_3d, axis=1)
    positive_keypoints_3d = tf.squeeze(positive_keypoints_3d, axis=1)

    if MAX_POSITIVE_KEYPOINT_MPJPE_2D is None:
      anchor_indicator_matrix = None
      positive_indicator_matrix = None
    else:
      anchor_indicator_matrix = compute_positive_indicator_matrix(
          anchor_keypoints_2d,
          anchor_keypoints_2d,
          distance_fn=keypoint_utils.compute_mpjpes,
          max_positive_distance=MAX_POSITIVE_KEYPOINT_MPJPE_2D)
      positive_indicator_matrix = compute_positive_indicator_matrix(
          positive_keypoints_2d,
          positive_keypoints_2d,
          distance_fn=keypoint_utils.compute_mpjpes,
          max_positive_distance=MAX_POSITIVE_KEYPOINT_MPJPE_2D)

    if MAX_POSITIVE_KEYPOINT_MPJPE_3D is None:
      view_indicator_matrix = None
    else:
      view_indicator_matrix = compute_positive_indicator_matrix(
          anchor_keypoints_3d,
          positive_keypoints_3d,
          distance_fn=keypoint_utils.compute_procrustes_aligned_mpjpes,
          max_positive_distance=MAX_POSITIVE_KEYPOINT_MPJPE_3D)

    with tf.GradientTape() as encoder_tape:
      (anchor_pose_embeddings, anchor_view_embeddings,
       anchor_subencoder_embeddings) = self(
           anchor_keypoints_2d, training=True)
      anchor_regularization_loss = sum(self.encoder.losses)
      anchor_representation_loss = self.compute_representation_loss(
          anchor_pose_embeddings, anchor_view_embeddings,
          anchor_subencoder_embeddings, anchor_indicator_matrix)
      (positive_pose_embeddings, positive_view_embeddings,
       positive_subencoder_embeddings) = self(
           positive_keypoints_2d, training=True)
      positive_regularization_loss = sum(self.encoder.losses)
      positive_representation_loss = self.compute_representation_loss(
          positive_pose_embeddings, positive_view_embeddings,
          positive_subencoder_embeddings, positive_indicator_matrix)

      representation_loss = (
          anchor_representation_loss + positive_representation_loss)
      regularization_loss = self._regularization_loss_weight * (
          anchor_regularization_loss + positive_regularization_loss)
      encoder_total_loss = representation_loss + regularization_loss

      anchor_cross_representation_loss = self.compute_representation_loss(
          positive_pose_embeddings, anchor_view_embeddings,
          anchor_subencoder_embeddings, view_indicator_matrix)
      positive_cross_representation_loss = self.compute_representation_loss(
          anchor_pose_embeddings, positive_view_embeddings,
          positive_subencoder_embeddings, view_indicator_matrix)
      cross_representation_loss = self._view_loss_weight * (
          anchor_cross_representation_loss + positive_cross_representation_loss)
      encoder_total_loss += cross_representation_loss

    encoder_trainable_variables = (
        self.encoder.trainable_variables + self.subencoder.trainable_variables)
    encoder_grads = encoder_tape.gradient(encoder_total_loss,
                                          encoder_trainable_variables)
    encoder_optimizer.apply_gradients(
        zip(encoder_grads, encoder_trainable_variables))

    encoder_losses = dict(
        total_loss=encoder_total_loss,
        representation_loss=representation_loss,
        cross_representation_loss=cross_representation_loss,
        regularization_loss=regularization_loss)

    return dict(encoder=encoder_losses)
