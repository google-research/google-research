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

"""Smurf: Unsupervised Optical Flow.

This class provides functions for loading, restoring, computing loss,
and inference.
"""

# pylint:skip-file
import functools
import math
import gin
import tensorflow as tf

from smurf import smurf_utils
from smurf.smurf_models import pwc_model
from smurf.smurf_models import raft_model


@gin.configurable
class SMURFNet(object):
  """Simple interface with infer and train methods."""

  def __init__(
      self,
      checkpoint_dir='',
      optimizer='adam',
      learning_rate=0.0002,
      only_forward=False,
      dropout_rate=.25,
      selfsup_transform=None,
      fb_sigma_teacher=0.003,
      fb_sigma_student=0.03,
      train_mode='sequence-unsupervised',
      smoothness_edge_weighting='gaussian',
      smoothness_edge_constant=150,
      teacher_image_version='original',
      stop_gradient_mask=True,
      selfsup_mask='gaussian',
      feature_architecture='raft',
      flow_architecture='raft',
      size=(1, 640, 640),
      occlusion_estimation='wang',
      smoothness_at_level=2,
      use_float16=True,
  ):
    """Instantiate a SMURF model.

    Args:
      checkpoint_dir: str, location to checkpoint model
      optimizer: str, identifier of which optimizer to use
      learning_rate: float, learning rate to use for training
      only_forward: bool, if True, only infer flow in one direction
      dropout_rate: float, how much dropout to use with pwc net
      selfsup_transform: list of functions which transform the flow
        predicted from the raw images to be in the frame of images transformed
        by geometric_augmentation_fn
      fb_sigma_teacher: float, controls how much forward-backward flow
        consistency is needed by the teacher model in order to supervise the
        student
      fb_sigma_student: float, controls how much forward-backward
        consistency is needed by the student model in order to not receive
        supervision from the teacher model
      train_mode: str, controls what kind of training loss should be used. One
        of the following: 'unsupervised', 'sequence-unsupervised',
        'supervised', or 'sequence-supervised'
      smoothness_edge_weighting: str, controls how smoothness penalty is
        determined
      smoothness_edge_constant: float, a weighting on smoothness
      teacher_image_version: str, which image to give to teacher model
      stop_gradient_mask: bool, whether to stop the gradient of photometric
        loss through the occlusion mask.
      selfsup_mask: str, type of selfsupervision mask to use
      feature_architecture: str, which feature extractor architecture to use,
        either raft or pwc.
      flow_architecture: str, which flow model architecture to use, either raft
        or pwc.
      size: 3-tuple of batch size, height, width
      occlusion_estimation: str, a the type of occlusion estimation to use
      smoothness_at_level: int, the level to apply smoothness
      use_float16: bool, whether or not to use float16 inside the models. This
        improves memory usage and computation time and does not impact accuracy.
    Returns:
      Smurf model instance.
    """
    self._only_forward = only_forward
    self._selfsup_transform = selfsup_transform
    self._fb_sigma_teacher = fb_sigma_teacher
    self._fb_sigma_student = fb_sigma_student
    self._train_mode = train_mode
    self._smoothness_edge_weighting = smoothness_edge_weighting
    self._smoothness_edge_constant = smoothness_edge_constant
    self._smoothness_at_level = smoothness_at_level
    self._teacher_flow_model = None
    self._teacher_feature_model = None
    self._teacher_image_version = teacher_image_version
    self._stop_gradient_mask = stop_gradient_mask
    self._selfsup_mask = selfsup_mask
    self._size = size
    self._use_float16 = use_float16
    self._flow_architecture = flow_architecture

    # Models
    if feature_architecture == 'pwc':
      self._feature_model = pwc_model.PWCFeatureSiamese()
    elif feature_architecture == 'raft':
      self._feature_model = raft_model.RAFTFeatureSiamese()

    else:
      raise ValueError(
          'Unknown feature architecture {}'.format(feature_architecture))

    if flow_architecture == 'pwc':
      self._flow_model = pwc_model.PWCFlow(
          dropout_rate=dropout_rate)
    elif flow_architecture == 'raft':
      self._flow_model = raft_model.RAFT()
    else:
      raise ValueError('Unknown flow architecture {}'.format(flow_architecture))
    # By default, the teacher flow and featuure models are the same as
    # the student flow and feature models.
    self._teacher_flow_model = self._flow_model
    self._teacher_feature_model = self._feature_model

    self._learning_rate = learning_rate
    self._optimizer_type = optimizer
    self._make_or_reset_optimizer()
    # Set up checkpointing.
    self._make_or_reset_checkpoint()
    self.update_checkpoint_dir(checkpoint_dir)
    self._occlusion_estimation = occlusion_estimation

  def set_teacher_models(self, teacher_feature_model, teacher_flow_model):
    self._teacher_feature_model = teacher_feature_model
    self._teacher_flow_model = teacher_flow_model

  @property
  def feature_model(self):
    return self._feature_model

  @property
  def flow_model(self):
    return self._flow_model

  def update_checkpoint_dir(self, checkpoint_dir):
    """Changes the checkpoint directory for saving and restoring."""
    self._manager = tf.train.CheckpointManager(
        self._checkpoint, directory=checkpoint_dir, max_to_keep=1)

  def restore(self, reset_optimizer=False, reset_global_step=False):
    """Restores a saved model from a checkpoint."""
    status = self._checkpoint.restore(self._manager.latest_checkpoint)
    try:
      status.assert_existing_objects_matched()
    except AssertionError as e:
      print('Error while attempting to restore SMURF models:', e)
    if reset_optimizer:
      self._make_or_reset_optimizer()
      self._make_or_reset_checkpoint()
    if reset_global_step:
      tf.compat.v1.train.get_or_create_global_step().assign(0)

  def save(self):
    self._manager.save()

  def _make_or_reset_optimizer(self):
    """Creates the optimizer attribute if not created, else resets it."""
    # Force the models to initialize their variables
    fake_image = tf.ones((1, self._size[1], self._size[2], 3))
    feature_dict = self._feature_model(fake_image, fake_image)
    _ = self._flow_model(feature_dict)

    if self._optimizer_type == 'adam':
      self._optimizer = tf.keras.optimizers.Adam(self._learning_rate,
                                                 name='Optimizer')
    elif self._optimizer_type == 'sgd':
      self._optimizer = tf.keras.optimizers.SGD(
          self._learning_rate, name='Optimizer')
    else:
      raise ValueError('Optimizer "{}" not yet implemented.'.format(
          self._optimizer_type))
    if self._use_float16:
      self._optimizer = (
          tf.compat.v1.mixed_precision.enable_mixed_precision_graph_rewrite(
              self._optimizer))

  @property
  def optimizer(self):
    return self._optimizer

  def get_checkpoint(self, additional_variables):
    return tf.train.Checkpoint(
        optimizer=self._optimizer,
        feature_model=self._feature_model,
        flow_model=self._flow_model,
        optimizer_step=tf.compat.v1.train.get_or_create_global_step(),
        additional_variables=additional_variables,
    )

  def _make_or_reset_checkpoint(self):
    self._checkpoint = tf.train.Checkpoint(
        optimizer=self._optimizer,
        feature_model=self._feature_model,
        flow_model=self._flow_model,
        optimizer_step=tf.compat.v1.train.get_or_create_global_step())

  # Use of tf.function breaks exporting the model
  def infer_no_tf_function(self,
                           image1,
                           image2,
                           input_height=None,
                           input_width=None,
                           resize_flow_to_img_res=True,
                           infer_occlusion=False,
                           infer_bw=False):
    """Infer flow for two images.

    Args:
      image1: tf.tensor of shape [height, width, 3].
      image2: tf.tensor of shape [height, width, 3].
      input_height: height at which the model should be applied if different
        from image height.
      input_width: width at which the model should be applied if different from
        image width
      resize_flow_to_img_res: bool, if True, return the flow resized to the same
        resolution as (image1, image2). If False, return flow at the whatever
        resolution the model natively predicts it.
      infer_occlusion: bool, if True, return both flow and a soft occlusion
        mask, else return just flow.
      infer_bw: bool, if True, returns the flow in both the forward and reverse
        direction.

    Returns:
      Optical flow for each pixel in image1 pointing to image2.
    """

    results = self.batch_infer_no_tf_function(
        tf.stack([image1, image2])[None],
        input_height=input_height,
        input_width=input_width,
        resize_flow_to_img_res=resize_flow_to_img_res,
        infer_occlusion=infer_occlusion,
        infer_bw=infer_bw)

    # Remove batch dimension from all results.
    if type(results) in [tuple, list]:
      return [x[0] for x in results]
    else:
      return results[0]

  # Use of tf.function breaks exporting the model
  def batch_infer_no_tf_function(self,
                                 images,
                                 input_height=None,
                                 input_width=None,
                                 resize_flow_to_img_res=True,
                                 infer_occlusion=False,
                                 infer_bw=False):
    """Infer flow for two images.

    Args:
      images: tf.tensor of shape [batchsize, 2, height, width, 3].
      input_height: height at which the model should be applied if different
        from image height.
      input_width: width at which the model should be applied if different from
        image width
      resize_flow_to_img_res: bool, if True, return the flow resized to the same
        resolution as (image1, image2). If False, return flow at the whatever
        resolution the model natively predicts it.
      infer_occlusion: bool, if True, return both flow and a soft occlusion
        mask, else return just flow.
      infer_bw: bool, if True, return flow in the reverse direction

    Returns:
      Optical flow for each pixel in image1 pointing to image2.
    """
    orig_height, orig_width = images.shape[-3:-1]

    if input_height is None:
      input_height = orig_height
    if input_width is None:
      input_width = orig_width

    # Ensure a feasible computation resolution. If specified size is not
    # feasible with the model, change it to a slightly higher resolution.
    if self._flow_architecture == 'pwc':
      divisible_by_num = pow(2.0, self._num_levels)
    elif self._flow_architecture == 'raft':
      divisible_by_num = 8.0
    else:
      divisible_by_num = 1.

    if (input_height % divisible_by_num != 0 or
        input_width % divisible_by_num != 0):
      print('Cannot process images at a resolution of '+str(input_height)+
            'x'+str(input_width)+', since the height and/or width is not a '
            'multiple of '+str(divisible_by_num)+'.')
      # compute a feasible resolution
      input_height = int(
          math.ceil(float(input_height) / divisible_by_num) * divisible_by_num)
      input_width = int(
          math.ceil(float(input_width) / divisible_by_num) * divisible_by_num)
      print('Inference will be run at a resolution of '+str(input_height)+
            'x'+str(input_width)+'.')

    # Resize images to desired input height and width.
    if input_height != orig_height or input_width != orig_width:
      images = smurf_utils.resize(
          images, input_height, input_width, is_flow=False)

    feature_dict = self._feature_model(
        images[:, 0], images[:, 1], bidirectional=infer_occlusion)

    # Compute flow in frame of image1.
    # noinspection PyCallingNonCallable
    flow = self._flow_model(feature_dict, training=False)[0]

    if infer_occlusion or infer_bw:
      # noinspection PyCallingNonCallable
      flow_backward = self._flow_model(
          feature_dict, training=False, backward=True)[0]
      occlusion_mask = self.infer_occlusion(flow, flow_backward)
      occlusion_mask = smurf_utils.resize(
          occlusion_mask, orig_height, orig_width, is_flow=False)

    # Resize and rescale flow to original resolution. This always needs to be
    # done because flow is generated at a lower resolution.
    if resize_flow_to_img_res:
      flow = smurf_utils.resize(flow, orig_height, orig_width, is_flow=True)
      if infer_bw:
        flow_backward = smurf_utils.resize(flow_backward, orig_height,
                                           orig_width,
                                           is_flow=True)

    # TODO: A dictionary or object output here would be preferable to tuples.
    if infer_occlusion and infer_bw:
      return flow, occlusion_mask, flow_backward

    if infer_bw:
      return flow, flow_backward

    if infer_occlusion:
      return flow, occlusion_mask

    return flow

  @tf.function
  def infer(self,
            image1,
            image2,
            input_height=None,
            input_width=None,
            resize_flow_to_img_res=True,
            infer_occlusion=False,
            infer_bw=False):
    return self.infer_no_tf_function(image1, image2, input_height, input_width,
                                     resize_flow_to_img_res, infer_occlusion,
                                     infer_bw)

  @tf.function
  def batch_infer(self,
                  images,
                  input_height=None,
                  input_width=None,
                  resize_flow_to_img_res=True,
                  infer_occlusion=False,
                  infer_bw=False):
    return self.batch_infer_no_tf_function(images, input_height, input_width,
                                           resize_flow_to_img_res,
                                           infer_occlusion, infer_bw)

  def infer_occlusion(self, flow_forward, flow_backward):
    """Get a 'soft' occlusion mask from the forward and backward flow."""

    occlusion_mask = smurf_utils.compute_occlusions(flow_forward,
                                                    flow_backward,
                                                    self._occlusion_estimation,
                                                    occlusions_are_zeros=False)
    return occlusion_mask

  def loss_and_grad(self,
                    inputs,
                    weights,
                    occ_active=None):
    """Apply the model on the data in batch and compute the loss.

    Args:
      inputs: a dictionary of input tf.Tensors
      weights: dictionary with float entries per loss.
      occ_active: a dictionary describing how occlusions should be handled

    Returns:
      A tuple consisting of a tf.scalar that represents the total loss for the
      current batch, a list of gradients, and a list of the respective
      variables.
    """
    with tf.GradientTape() as tape:
      losses = self.compute_loss(
          inputs,
          weights,
          occ_active=occ_active)
      if self._use_float16:
        scaled_loss = self._optimizer.get_scaled_loss(losses['total-loss'])

    variables = (
        self._feature_model.trainable_variables +
        self._flow_model.trainable_variables)

    if self._use_float16:
      scaled_gradients = tape.gradient(scaled_loss, variables)
      grads = self._optimizer.get_unscaled_gradients(scaled_gradients)
    else:
      grads = tape.gradient(losses['total-loss'], variables)
    return losses, grads, variables

  def compute_loss(self,
                   inputs,
                   weights,
                   occ_active=None):
    """Apply models and compute losses for a batch of image sequences."""
    # Check if chosen train_mode is valid.
    if self._train_mode not in [
        'supervised', 'unsupervised', 'sequence-supervised',
        'sequence-unsupervised',]:
      raise NotImplementedError(
          'train_mode must be one of the following options: supervised, '
          'unsupervised, sequence-supervised.')
    # The 'images' here have been geometrically but not photometrically
    # augmented.
    images = inputs.get('images')
    augmented_images = inputs.get('augmented_images', images)
    ground_truth_flow = inputs.get('flow')
    ground_truth_valid = inputs.get('flow_valid')
    full_size_images = inputs.get('full_size_images')
    crop_h = inputs.get('crop_h')
    crop_w = inputs.get('crop_w')
    pad_h = inputs.get('pad_h')
    pad_w = inputs.get('pad_w')

    # Compute only a sequence loss.
    sequence_supervised_losses = {}
    if self._train_mode == 'sequence-supervised':
      flows = smurf_utils.compute_flow_for_sequence_loss(
          self._feature_model, self._flow_model, batch=augmented_images,
          training=True)
      sequence_supervised_losses = smurf_utils.supervised_sequence_loss(
          ground_truth_flow, ground_truth_valid, flows, weights)
      sequence_supervised_losses['total'] = sum(
          sequence_supervised_losses.values())
      sequence_supervised_losses = {
          key + '-loss': sequence_supervised_losses[key]
          for key in sequence_supervised_losses
      }
      return sequence_supervised_losses

    # Compute only a supervised loss.
    supervised_losses = {}
    if self._train_mode == 'supervised':
      if ground_truth_flow is None:
        raise ValueError('Need ground truth flow to compute supervised loss.')
      flows = smurf_utils.compute_flow_for_supervised_loss(
          self._feature_model, self._flow_model, batch=augmented_images,
          training=True)
      supervised_losses = smurf_utils.supervised_loss(
          ground_truth_flow, ground_truth_valid, flows, weights)
      supervised_losses['total'] = sum(supervised_losses.values())
      supervised_losses = {
          key + '-loss': supervised_losses[key] for key in supervised_losses
      }
      return supervised_losses

    # Compute all required flow fields.
    # TODO: Can't condition computation on this without breaking autograph.
    perform_selfsup = 'selfsup' in weights
    flows = smurf_utils.compute_flows_for_unsupervised_loss(
        feature_model=self._feature_model,
        flow_model=self._flow_model,
        batch=augmented_images,
        batch_without_aug=images,
        training=True,
        selfsup_transform_fn=self._selfsup_transform,
        return_sequence='sequence' in self._train_mode,
        perform_selfsup=perform_selfsup)

    # Prepare occlusion estimation function.
    occlusion_estimation_fn = functools.partial(
        smurf_utils.compute_occlusions,
        occlusion_estimation=self._occlusion_estimation,
        occlusions_are_zeros=True,
        occ_active=occ_active,
        boundaries_occluded=full_size_images is None)

    # Prepare a simplified call for the unsupervised loss function.
    unsupervised_loss_fn = functools.partial(
        smurf_utils.unsupervised_loss,
        weights=weights,
        occlusion_estimation_fn=occlusion_estimation_fn,
        only_forward=False,
        selfsup_transform_fn=self._selfsup_transform,
        fb_sigma_teacher=self._fb_sigma_teacher,
        fb_sigma_student=self._fb_sigma_student,
        smoothness_edge_weighting=self._smoothness_edge_weighting,
        smoothness_edge_constant=self._smoothness_edge_constant,
        stop_gradient_mask=self._stop_gradient_mask,
        selfsup_mask=self._selfsup_mask,
        smoothness_at_level=self._smoothness_at_level)

    losses = {}
    if self._train_mode == 'unsupervised':
      unsupervised_losses = unsupervised_loss_fn(
          images,
          flows,
          full_size_images=full_size_images,
          crop_h=crop_h,
          crop_w=crop_w,
          pad_h=pad_h,
          pad_w=pad_w)
      losses.update(unsupervised_losses)

    elif self._train_mode == 'sequence-unsupervised':
      sequence_unsupervised_losses = smurf_utils.unsupervised_sequence_loss(
          images=images,
          flows_sequence=flows,
          full_size_images=full_size_images,
          crop_h=crop_h,
          crop_w=crop_w,
          pad_h=pad_h,
          pad_w=pad_w,
          unsupervised_loss_fn=unsupervised_loss_fn)
      losses.update(sequence_unsupervised_losses)
    else:
      raise ValueError(f'Unknown mode {self._train_mode}')

    losses['total'] = sum(losses.values())
    losses = {key + '-loss': losses[key] for key in losses}
    return losses
