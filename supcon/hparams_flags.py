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

"""Defines flags for setting hparams."""

from absl import flags

from supcon import enums
from supcon import hparams

FLAGS = flags.FLAGS

# Train and eval batch sizes. For TPU execution, these are the total batch sizes
# for all TPUs together, not the smaller per-TPU ones.
flags.DEFINE_integer('batch_size', 2048, 'Train batch size')
flags.DEFINE_integer('eval_batch_size', 128, 'The batch size to use for eval.')

# Architecture
flags.DEFINE_float('resnet_width', 1., 'Width of the resnet to be'
                   'used in the model')
flags.DEFINE_enum_class(
    'resnet_architecture', enums.EncoderArchitecture.RESNET_V1,
    enums.EncoderArchitecture, 'Name of the resnet'
    ' architecture to be used in the model.')
flags.DEFINE_integer('resnet_depth', 50, 'Depth of the Resnet model')
flags.DEFINE_bool(
    'zero_initialize_classifier', False,
    'Whether or not to initialize parameters in the classification head '
    'to 0. Otherwise uses glorot uniform.')
flags.DEFINE_bool('use_projection_batch_norm', False,
                  'Whether to use batch norm in the projection head.')
flags.DEFINE_bool(
    'use_projection_batch_norm_beta', True,
    'Whether projection head batch norm layers should have a beta (bias) '
    'parameter.')
flags.DEFINE_bool(
    'normalize_embedding', True,
    'If the outputs of the encoder should be normalized before being input to '
    'the projections and classification heads.')
flags.DEFINE_integer(
    'first_conv_kernel_size', 7,
    'The kernel size of the first convolution in the Encoder network. '
    'Generally it is 7 for ImageNet and 3 for CIFAR10.')
flags.DEFINE_integer(
    'first_conv_stride', 2,
    'The stride of the first convolution in the Encoder network. '
    'Generally it is 2 for ImageNet and 1 for CIFAR10.')
flags.DEFINE_bool(
    'use_initial_max_pool', True,
    'Whether to include a max-pool layer between the initial convolution and '
    'the first residual block. in the Encoder network. Generally it is true '
    'for ImageNet and false for CIFAR10.')
flags.DEFINE_list(
    'projection_head_layers', [2048, 128],
    'The number of units for each dense layer in the projection head.')
flags.DEFINE_bool(
    'stop_gradient_before_classification_head', True,
    'Whether there is a stop gradient between the encoder and classification '
    'head.')
flags.DEFINE_bool(
    'stop_gradient_before_projection_head', False,
    'Whether there is a stop gradient between the encoder and projection head.')
flags.DEFINE_bool(
    'use_global_batch_norm', True,
    'Whether to use global batch norm, where statistics are aggregated across '
    'TPU cores, instead of local batch norm, where statistics are only '
    'computed on TPU core 0. This flag only has impact when running on TPU. '
    'Distributed GPU or CPU setups always use local batch norm.')

# Loss
flags.DEFINE_bool(
    'use_labels', True,
    'Whether to use labels for determining the positives in the contrastive '
    'loss (SupCon), rather than acting fully self-supervised (SimCLR).')
flags.DEFINE_float('temperature', 0.07,
                   'The temperature parameter of the contrastive loss.')
flags.DEFINE_bool(
    'scale_by_temperature', True,
    'Whether the contrastive loss gets scaled by the temperature, which helps '
    'normalize the loss gradient.')
flags.DEFINE_enum_class(
    'contrast_mode', enums.LossContrastMode.ALL_VIEWS, enums.LossContrastMode,
    'Contrast mode: ALL (contrast all views against all other views) or ONE '
    '(contrast one view against all other views).')
flags.DEFINE_enum_class('summation_location',
                        enums.LossSummationLocation.OUTSIDE,
                        enums.LossSummationLocation,
                        'Location of the summation: OUTSIDE or INSIDE.')
flags.DEFINE_enum_class('denominator_mode', enums.LossDenominatorMode.ALL,
                        enums.LossDenominatorMode,
                        'Denominator mode: ALL, ONE_POSITIVE, ONLY_NEGATIVES.')
flags.DEFINE_integer(
    'positives_cap', -1,
    'Integer maximum number of positives other than augmentations '
    'of anchor. Infinite if -1. Must be multiple of 2 (number of views).')
flags.DEFINE_float('label_smoothing', 0., 'The label smoothing rate')
flags.DEFINE_float(
    'stage_1_contrastive_loss_weight', 1.0,
    'The weight applied to the contrastive loss during stage 1 of training.')
flags.DEFINE_float(
    'stage_2_contrastive_loss_weight', 0.0,
    'The weight applied to the contrastive loss during stage 2 of training.')
flags.DEFINE_float(
    'stage_1_cross_entropy_loss_weight', 0.0,
    'The weight applied to the cross-entropy loss during stage 1 of training.')
flags.DEFINE_float(
    'stage_2_cross_entropy_loss_weight', 1.0,
    'The weight applied to the cross-entropy loss during stage 2 of training.')
flags.DEFINE_float(
    'stage_1_weight_decay', 0.0,
    'The weight applied to weight decay during stage 1 of training.')
flags.DEFINE_float(
    'stage_2_weight_decay', 0.0,
    'The weight applied to weight decay during stage 2 of training.')
flags.DEFINE_bool(
    'stage_1_use_encoder_weight_decay', True,
    'Whether to include the encoder in weight decay during stage 1 of training.'
)
flags.DEFINE_bool(
    'stage_2_use_encoder_weight_decay', True,
    'Whether to include the encoder in weight decay during stage 2 of training.'
)
flags.DEFINE_bool(
    'stage_1_use_projection_head_weight_decay', True,
    'Whether to include the projection head in weight decay during stage 1 of '
    'training.')
flags.DEFINE_bool(
    'stage_2_use_projection_head_weight_decay', True,
    'Whether to include the projection head in weight decay during stage 2 of '
    'training.')
flags.DEFINE_bool(
    'stage_1_use_classification_head_weight_decay', False,
    'Whether to include the classification head in weight decay during stage 1 '
    'of training.')
flags.DEFINE_bool(
    'stage_2_use_classification_head_weight_decay', False,
    'Whether to include the classification head in weight decay during stage 2 '
    'of training.')
flags.DEFINE_bool('use_bias_weight_decay', True,
                  'Whether to include bias variables in weight decay.')

# Training
flags.DEFINE_integer('stage_1_epochs', 700,
                     'The number of epochs for stage 1 of training.')
flags.DEFINE_integer('stage_2_epochs', 200,
                     'Number of epochs of for stage 2 of  training')
flags.DEFINE_float(
    'stage_1_warmup_epochs', 12.0,
    'The number of epochs over which warmup is applied to the learning rate '
    'for stage 1 of training.')
flags.DEFINE_float(
    'stage_2_warmup_epochs', 12.0,
    'The number of epochs over which warmup is applied to the learning rate '
    'for stage 2 of training.')
flags.DEFINE_float(
    'stage_1_base_learning_rate', 0.3,
    'The base learning rate used for stage 1 of training for a batch size of '
    '256. The true learning rate is '
    'multiplied by batch_size/256.')
flags.DEFINE_float(
    'stage_2_base_learning_rate', 0.05,
    'The base learning rate used for stage 2 of training for a batch size of '
    '256. The true learning rate is '
    'multiplied by batch_size/256.')
flags.DEFINE_enum_class('stage_1_optimizer', enums.Optimizer.RMSPROP,
                        enums.Optimizer,
                        'The optimizer to use for stage 1 of training.')
flags.DEFINE_enum_class('stage_2_optimizer', enums.Optimizer.RMSPROP,
                        enums.Optimizer,
                        'The optimizer to use for stage 2 of training.')
flags.DEFINE_enum_class(
    'stage_1_learning_rate_decay', enums.DecayType.COSINE, enums.DecayType,
    'The decay type for the stage 1 learning rate. COSINE, EXPONENTIAL, '
    'PIECEWISE_LINEAR or NONE')
flags.DEFINE_enum_class(
    'stage_2_learning_rate_decay', enums.DecayType.COSINE, enums.DecayType,
    'The decay type for the stage 2 learning rate. COSINE, EXPONENTIAL, '
    'PIECEWISE_LINEAR or NONE')
flags.DEFINE_float(
    'stage_1_decay_rate', 0.1,
    'The rate of learning rate decay. Only used for learning_rate_decay types '
    'p and e')
flags.DEFINE_float(
    'stage_2_decay_rate', 0.1,
    'The rate of learning rate decay. Only used for learning_rate_decay types '
    'p and e')
flags.DEFINE_list(
    'stage_1_decay_boundary_epochs', ['30', '60', '80', '90'],
    'The epochs at which decay occurs when using piecewise-linear learning '
    'rate decay.')
flags.DEFINE_list(
    'stage_2_decay_boundary_epochs', ['30', '60', '80', '90'],
    'The epochs at which decay occurs when using piecewise-linear learning '
    'rate decay.')
flags.DEFINE_float(
    'stage_1_epochs_per_decay', 2.4,
    'The learning rate decays by a factor of `decay_rate` every '
    '`epochs_per_decay` when using exponential learning rate decay.')
flags.DEFINE_float(
    'stage_2_epochs_per_decay', 2.4,
    'The learning rate decays by a factor of `decay_rate` every '
    '`epochs_per_decay` when using exponential learning rate decay.')
flags.DEFINE_bool(
    'stage_1_update_encoder_batch_norm', True,
    'Whether encoder batch norm statistics should be updated during stage 1 of '
    'training.')
flags.DEFINE_bool(
    'stage_2_update_encoder_batch_norm', True,
    'Whether encoder batch norm statistics should be updated during stage 2 of '
    'training.')
flags.DEFINE_float(
    'stage_1_rmsprop_epsilon', 1.0,
    'The epsilon parameter of the RMSProp optimizer. Only used if '
    '`stage_1_optimizer` is RMSPROP.')
flags.DEFINE_float(
    'stage_2_rmsprop_epsilon', 0.001,
    'The epsilon parameter of the RMSProp optimizer. Only used if '
    '`stage_2_optimizer` is RMSPROP.')

# Dataset handling
flags.DEFINE_string('input_fn', 'imagenet', 'The input_fn to use.')
flags.DEFINE_integer('num_images', 0,
                     'The maximum number of samples to use from the dataset.')
flags.DEFINE_float(
    'label_noise_prob', 0.,
    'The fraction of true labels to replace with random labels.')
flags.DEFINE_bool('shard_per_host', True,
                  'Whether to shard the dataset between TPU hosts.')

# Image preprocessing
flags.DEFINE_bool('allow_mixed_precision', False,
                  'Use bfloat16, if the hardware supports it.')
flags.DEFINE_integer(
    'image_size', 224,
    'The side length, in pixels, of the preprocessing outputs, irrespective of '
    'the natural size of the images.')
flags.DEFINE_enum_class('augmentation_type', enums.AugmentationType.AUTOAUGMENT,
                        enums.AugmentationType, 'Augmention type.')
flags.DEFINE_float(
    'augmentation_magnitude', 0.5, 'Strength parameter of the augmentation. '
    'Equivalent to m parameter of RandAugment or s parameter in SimCLR '
    'augmentation.')
flags.DEFINE_float(
    'blur_probability', 0.5,
    'The probability of blurring as part of the data augmentation, if '
    '`augmentation_type` includes a blur stage. 0 disables blurring.')
flags.DEFINE_bool(
    'defer_blurring', True,
    'If True and `augmentation_type` applies blurring and `blur_probability is '
    'greater than 0, the blur is applied in the model_fn rather than in the '
    'preprocess_image call from the input_fn. This is useful because TPU '
    'training runs the input_fn on CPU, but the blur operation is much faster '
    'when run on TPU.')
flags.DEFINE_bool(
    'use_pytorch_color_jitter', False,
    'Whether to use a color jittering algorithm that aims to replicate '
    '`torchvision.transforms.ColorJitter` rather than the standard TensorFlow '
    'color jittering. Only used for augmentation_types SIMCLR and '
    'STACKED_RANDAUGMENT.')
flags.DEFINE_bool(
    'apply_whitening', True,
    'Whether or not to apply whitening to datasets that specify whitening '
    'parameters.')
flags.DEFINE_list(
    'crop_area_range', ['0.08', '1.0'],
    'The range of allowed values for the fraction of the original pixels that '
    'need to be included in any crop.')
flags.DEFINE_enum_class(
    'eval_crop_method', enums.EvalCropMethod.RESIZE_THEN_CROP,
    enums.EvalCropMethod,
    'The strategy for obtaining the desired square crop in eval mode.')
flags.DEFINE_integer(
    'crop_padding', 32,
    'For eval_crop_methods other than IDENTITY, this value is used to '
    'determine how to perform the central crop. See `enums.EvalCropMethod` for '
    'details.')

# Flags for controling warm starting from a pretrained checkpoint.
flags.DEFINE_bool(
    'warm_start_classifier', False, 'Whether or not to load in the '
    'classification head weightsfrom reference checkpoint.')
flags.DEFINE_bool(
    'warm_start_projection_head', False, 'Whether or not to load in the '
    'projection head weights from reference checkpoint.')
flags.DEFINE_bool(
    'warm_start_encoder', False, 'Whether or not to load in the encoder '
    'weights from reference checkpoint.')
flags.DEFINE_bool(
    'batch_norm_in_train_mode', False, 'Whether to set batch norm to train '
    'mode when warm starting from a checkpoint.')
flags.DEFINE_bool(
    'ignore_missing_checkpoint_vars', False,
    'When warm-starting using `reference_ckpt`, determines whether or not to '
    'raise an error if the checkpoint does not contain all expected variables.')


def hparams_from_flags():
  return hparams.HParams(
      bs=FLAGS.batch_size,
      architecture=hparams.Architecture(
          encoder_architecture=FLAGS.resnet_architecture,
          encoder_depth=FLAGS.resnet_depth,
          encoder_width=FLAGS.resnet_width,
          first_conv_kernel_size=FLAGS.first_conv_kernel_size,
          first_conv_stride=FLAGS.first_conv_stride,
          use_initial_max_pool=FLAGS.use_initial_max_pool,
          projection_head_layers=tuple(map(int, FLAGS.projection_head_layers)),
          projection_head_use_batch_norm=FLAGS.use_projection_batch_norm,
          projection_head_use_batch_norm_beta=(
              FLAGS.use_projection_batch_norm_beta),
          normalize_projection_head_inputs=FLAGS.normalize_embedding,
          normalize_classifier_inputs=FLAGS.normalize_embedding,
          zero_initialize_classifier=FLAGS.zero_initialize_classifier,
          stop_gradient_before_classification_head=(
              FLAGS.stop_gradient_before_classification_head),
          stop_gradient_before_projection_head=(
              FLAGS.stop_gradient_before_projection_head),
          use_global_batch_norm=FLAGS.use_global_batch_norm),
      loss_all_stages=hparams.LossAllStages(
          contrastive=hparams.ContrastiveLoss(
              use_labels=FLAGS.use_labels,
              temperature=FLAGS.temperature,
              contrast_mode=FLAGS.contrast_mode,
              summation_location=FLAGS.summation_location,
              denominator_mode=FLAGS.denominator_mode,
              positives_cap=FLAGS.positives_cap,
              scale_by_temperature=FLAGS.scale_by_temperature),
          cross_entropy=hparams.CrossEntropyLoss(
              label_smoothing=FLAGS.label_smoothing),
          include_bias_in_weight_decay=FLAGS.use_bias_weight_decay),
      stage_1=hparams.Stage(
          training=hparams.TrainingStage(
              train_epochs=FLAGS.stage_1_epochs,
              learning_rate_warmup_epochs=FLAGS.stage_1_warmup_epochs,
              base_learning_rate=FLAGS.stage_1_base_learning_rate,
              learning_rate_decay=FLAGS.stage_1_learning_rate_decay,
              decay_rate=FLAGS.stage_1_decay_rate,
              decay_boundary_epochs=tuple(
                  map(int, FLAGS.stage_1_decay_boundary_epochs)),
              epochs_per_decay=FLAGS.stage_1_epochs_per_decay,
              optimizer=FLAGS.stage_1_optimizer,
              update_encoder_batch_norm=(
                  FLAGS.stage_1_update_encoder_batch_norm),
              rmsprop_epsilon=FLAGS.stage_1_rmsprop_epsilon),
          loss=hparams.LossStage(
              contrastive_weight=FLAGS.stage_1_contrastive_loss_weight,
              cross_entropy_weight=FLAGS.stage_1_cross_entropy_loss_weight,
              weight_decay_coeff=FLAGS.stage_1_weight_decay,
              use_encoder_weight_decay=FLAGS.stage_1_use_encoder_weight_decay,
              use_projection_head_weight_decay=(
                  FLAGS.stage_1_use_projection_head_weight_decay),
              use_classification_head_weight_decay=(
                  FLAGS.stage_1_use_classification_head_weight_decay)),
      ),
      stage_2=hparams.Stage(
          training=hparams.TrainingStage(
              train_epochs=FLAGS.stage_2_epochs,
              learning_rate_warmup_epochs=FLAGS.stage_2_warmup_epochs,
              base_learning_rate=FLAGS.stage_2_base_learning_rate,
              learning_rate_decay=FLAGS.stage_2_learning_rate_decay,
              decay_rate=FLAGS.stage_2_decay_rate,
              decay_boundary_epochs=tuple(
                  map(int, FLAGS.stage_2_decay_boundary_epochs)),
              epochs_per_decay=FLAGS.stage_2_epochs_per_decay,
              optimizer=FLAGS.stage_2_optimizer,
              update_encoder_batch_norm=(
                  FLAGS.stage_2_update_encoder_batch_norm),
              rmsprop_epsilon=FLAGS.stage_2_rmsprop_epsilon),
          loss=hparams.LossStage(
              contrastive_weight=FLAGS.stage_2_contrastive_loss_weight,
              cross_entropy_weight=FLAGS.stage_2_cross_entropy_loss_weight,
              weight_decay_coeff=FLAGS.stage_2_weight_decay,
              use_encoder_weight_decay=FLAGS.stage_2_use_encoder_weight_decay,
              use_projection_head_weight_decay=(
                  FLAGS.stage_2_use_projection_head_weight_decay),
              use_classification_head_weight_decay=(
                  FLAGS.stage_2_use_classification_head_weight_decay))),
      eval=hparams.Eval(batch_size=FLAGS.eval_batch_size),
      input_data=hparams.InputData(
          input_fn=FLAGS.input_fn,
          preprocessing=hparams.ImagePreprocessing(
              allow_mixed_precision=FLAGS.allow_mixed_precision,
              image_size=FLAGS.image_size,
              augmentation_type=FLAGS.augmentation_type,
              augmentation_magnitude=FLAGS.augmentation_magnitude,
              blur_probability=FLAGS.blur_probability,
              defer_blurring=FLAGS.defer_blurring,
              use_pytorch_color_jitter=FLAGS.use_pytorch_color_jitter,
              apply_whitening=FLAGS.apply_whitening,
              crop_area_range=tuple(map(float, FLAGS.crop_area_range)),
              eval_crop_method=FLAGS.eval_crop_method,
              crop_padding=FLAGS.crop_padding,
          ),
          max_samples=FLAGS.num_images,
          label_noise_prob=FLAGS.label_noise_prob,
          shard_per_host=FLAGS.shard_per_host),
      warm_start=hparams.WarmStart(
          warm_start_classifier=FLAGS.warm_start_classifier,
          ignore_missing_checkpoint_vars=FLAGS.ignore_missing_checkpoint_vars,
          warm_start_projection_head=FLAGS.warm_start_projection_head,
          warm_start_encoder=FLAGS.warm_start_encoder,
          batch_norm_in_train_mode=FLAGS.batch_norm_in_train_mode,
      ),
  )
