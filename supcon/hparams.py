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

"""Hyperparameters for contrastive learning."""

from supcon import enums
from supcon import hparam


@hparam.s
class Architecture:
  """Hyperparameters relating to the model architecture."""
  encoder_architecture = hparam.field(
      default=enums.EncoderArchitecture.RESNET_V1, abbrev='ea')
  encoder_depth = hparam.field(default=50, abbrev='d')
  # This is a multiplier on the number of channels in each conv layer. Width=1
  # uses 64 channels in the first block and doubles that in each subsequent
  # block. It is valid to use a non-integer value here so that we don't always
  # have to deal in multiples of 64.
  encoder_width = hparam.field(default=1., abbrev='w')
  first_conv_kernel_size = hparam.field(default=7, abbrev='fk')
  first_conv_stride = hparam.field(default=2, abbrev='fs')
  use_initial_max_pool = hparam.field(default=True, abbrev='imp')

  projection_head_layers = hparam.field(default=[2048, 128], abbrev='phl')
  projection_head_use_batch_norm = hparam.field(default=False, abbrev='up')
  projection_head_use_batch_norm_beta = hparam.field(default=True, abbrev='pb')
  normalize_projection_head_inputs = hparam.field(default=True, abbrev='phn')

  normalize_classifier_inputs = hparam.field(default=True, abbrev='chn')
  zero_initialize_classifier = hparam.field(default=False, abbrev='czi')
  stop_gradient_before_classification_head = hparam.field(
      default=True, abbrev='sgc')
  stop_gradient_before_projection_head = hparam.field(
      default=False, abbrev='sgp')

  use_global_batch_norm = hparam.field(default=True, abbrev='gbp')


@hparam.s
class ContrastiveLoss:
  """Hyperparameters relating to the contrastive loss."""
  use_labels = hparam.field(default=False, abbrev='ul')
  temperature = hparam.field(default=1.0, abbrev='t')
  contrast_mode = hparam.field(
      default=enums.LossContrastMode.ALL_VIEWS, abbrev='cm')
  summation_location = hparam.field(
      default=enums.LossSummationLocation.OUTSIDE, abbrev='sl')
  denominator_mode = hparam.field(
      default=enums.LossDenominatorMode.ALL, abbrev='dm')
  positives_cap = hparam.field(default=-1, abbrev='pc')
  scale_by_temperature = hparam.field(default=True, abbrev='sbt')


@hparam.s
class CrossEntropyLoss:
  """Hyperparameters relating to the cross-entropy loss."""
  label_smoothing = hparam.field(default=0., abbrev='ls')


@hparam.s
class LossStage:
  """Hyperparameters relating to the loss for a single training stage."""
  contrastive_weight = hparam.field(default=1.0, abbrev='cw')
  cross_entropy_weight = hparam.field(default=1.0, abbrev='xw')
  weight_decay_coeff = hparam.field(default=1e-4, abbrev='wd')
  use_encoder_weight_decay = hparam.field(default=True, abbrev='ed')
  use_projection_head_weight_decay = hparam.field(default=True, abbrev='pd')
  use_classification_head_weight_decay = hparam.field(default=True, abbrev='cd')


@hparam.s
class LossAllStages:
  """Hyperparameters relating to the loss.

  This only includes those parameters that don't vary based on training stage.
  """
  contrastive = hparam.nest(ContrastiveLoss)
  cross_entropy = hparam.nest(CrossEntropyLoss)
  include_bias_in_weight_decay = hparam.field(default=True, abbrev='bd')


@hparam.s
class TrainingStage:
  """Hyperparameters relating to a single stage of training."""
  train_epochs = hparam.field(default=700, abbrev='te')
  learning_rate_warmup_epochs = hparam.field(default=12.0, abbrev='wu')
  base_learning_rate = hparam.field(default=0.3, abbrev='lr')
  learning_rate_decay = hparam.field(
      default=enums.DecayType.COSINE, abbrev='de')
  # Only used for learning_rate_decay types PIECEWISE_LINEAR and EXPONENTIAL.
  decay_rate = hparam.field(default=0.1, abbrev='dr')
  # Only used for learning_rate_decay type PIECEWISE_LINEAR.
  decay_boundary_epochs = hparam.field(default=[30, 60, 80, 90], abbrev='be')
  # Only used for learning_rate_decay type EXPONENTIAL.
  epochs_per_decay = hparam.field(default=2.4, abbrev='epd')
  optimizer = hparam.field(default=enums.Optimizer.RMSPROP, abbrev='op')
  update_encoder_batch_norm = hparam.field(default=True, abbrev='ebn')
  rmsprop_epsilon = hparam.field(default=1.0, abbrev='rep')


@hparam.s
class Stage:
  """Hyperparameters relating to a single stage of training."""
  # Training is structured as having multiple stages, currently 2. The stages
  # are run sequentially, all using the same Graph. This enables using different
  # loss settings, learning rate schedules, and optimizers in each stage. The
  # primary use case for this is standard contrastive training, where stage 1
  # trains with just the contrastive loss applied to the projection head and the
  # encoder, and stage 2 trains with just the cross-entropy loss applied to just
  # the classification head. It's possible to train with just a single stage by
  # setting the `train_epochs` for an unused stage to 0. A single stage can use
  # multiple losses simultaneously by setting the appropriate weights in the
  # corresponding LossStage parameters for each stage.
  training = hparam.nest(TrainingStage)
  loss = hparam.nest(LossStage)


@hparam.s
class Eval:
  """Hyperparameters relating to evaluation."""
  batch_size = hparam.field(default=128, abbrev='ebs')


@hparam.s
class ImagePreprocessing:
  """Hyperparameters related to image preprocessing."""
  # Whether the preprocessed images should be tf.bfloat16 instead of tf.float32,
  # if the hardware supports it.
  allow_mixed_precision = hparam.field(default=False, abbrev='bf')
  # The side length, in pixels, of the preprocessing outputs, irrespective of
  # the natural size of the images.
  image_size = hparam.field(default=224, abbrev='is')
  # The type of augmentation to use.
  augmentation_type = hparam.field(
      default=enums.AugmentationType.SIMCLR, abbrev='au')
  augmentation_magnitude = hparam.field(default=0.5, abbrev='m')
  # The probability of warping.
  warp_probability = hparam.field(default=0.0, abbrev='wp')
  # The probability of blurring.
  blur_probability = hparam.field(default=0.5, abbrev='bp')
  # If True and `augmentation_type` applies blurring and `blur_probability is
  # greater than 0, the blur is applied in the model_fn rather than in the
  # preprocess_image call from the input_fn. This is useful because TPU training
  # runs the input_fn on CPU, but the blur operation is much faster when run on
  # TPU.
  defer_blurring = hparam.field(default=True, abbrev='db')
  # Whether to use a color jittering algorithm that aims to replicate
  # `torchvision.transforms.ColorJitter` rather than the standard TensorFlow
  # color jittering.
  use_pytorch_color_jitter = hparam.field(default=False, abbrev='pj')
  # Whether or not to apply whitening to datasets that specify whitening
  # parameters.
  apply_whitening = hparam.field(default=True, abbrev='wh')
  # The range of allowed values for the fraction of the original pixels that
  # need to be included in any crop.
  crop_area_range = hparam.field(default=(0.08, 1.0), abbrev='ar')
  # The strategy for obtaining the desired square crop in eval mode.
  eval_crop_method = hparam.field(
      default=enums.EvalCropMethod.RESIZE_THEN_CROP, abbrev='ec')
  # For eval_crop_methods other than IDENTITY, this value is used to determine
  # how to perform the central crop. See `enums.EvalCropMethod` for details.
  crop_padding = hparam.field(default=32, abbrev='cp')
  # The number of views to generate per input. Currently 2 is the only value
  # supported for training.
  # TODO(sarna): Generalize to arbitrary numbers of views.
  num_views = hparam.field(default=2, abbrev='nv')


@hparam.s
class InputData:
  """Hyperparameters relating to input data."""
  input_fn = hparam.field(default='imagenet', abbrev='ds')
  preprocessing = hparam.nest(ImagePreprocessing)
  max_samples = hparam.field(default=-1, abbrev='ms')
  label_noise_prob = hparam.field(default=0., abbrev='ln')
  # If True, during training the dataset is sharded per TPU host, rather than
  # each host independently iterating over the full dataset. This guarantees
  # that the same sample won't appear in the same global batch on different TPU
  # cores, and also saves memory when caching the dataset.
  shard_per_host = hparam.field(default=True, abbrev='hs')


@hparam.s
class WarmStart:
  """Hyperparameters relating to warm starting."""
  warm_start_classifier = hparam.field(default=False, abbrev='wsc')
  warm_start_projection_head = hparam.field(default=False, abbrev='wsp')
  warm_start_encoder = hparam.field(default=False, abbrev='wse')
  ignore_missing_checkpoint_vars = hparam.field(default=False, abbrev='wsi')
  batch_norm_in_train_mode = hparam.field(default=False, abbrev='wsb')


@hparam.s
class HParams:
  """Hyperparameters."""
  # This unfortunately needs to be in the root and be named the same as its
  # abbreviation so that Bootstrap's TPUEstimator wrapper can find it.
  bs = hparam.field(default=2048, abbrev='bs')
  architecture = hparam.nest(Architecture)
  loss_all_stages = hparam.nest(LossAllStages)
  stage_1 = hparam.nest(Stage, prefix='s1')
  stage_2 = hparam.nest(Stage, prefix='s2')
  eval = hparam.nest(Eval)
  input_data = hparam.nest(InputData)
  warm_start = hparam.nest(WarmStart)
