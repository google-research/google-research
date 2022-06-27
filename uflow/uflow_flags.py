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

"""Flags used by uflow training and evaluation."""
from absl import flags

FLAGS = flags.FLAGS

# General flags.
flags.DEFINE_bool(
    'no_tf_function', False, 'If True, run without'
    ' tf functions. This incurs a performance hit, but can'
    ' make debugging easier.')
flags.DEFINE_string('train_on', '',
                    '"format0:path0;format1:path1", e.g. "kitti:/usr/..."')
flags.DEFINE_string('eval_on', '',
                    '"format0:path0;format1:path1", e.g. "kitti:/usr/..."')
flags.DEFINE_string('plot_dir', '', 'Path to directory where plots are saved.')
flags.DEFINE_string('checkpoint_dir', '',
                    'Path to directory for saving and restoring checkpoints.')
flags.DEFINE_string('init_checkpoint_dir', '',
                    'Path to directory for initializing from a checkpoint.')
flags.DEFINE_bool(
    'plot_debug_info', False,
    'Flag to indicate whether to plot debug info during training.')
flags.DEFINE_bool(
    'use_tensorboard', False, 'Toggles logging to tensorboard.')
flags.DEFINE_string(
    'tensorboard_logdir', '', 'Where to log tensorboard summaries.')
flags.DEFINE_bool(
    'frozen_teacher', False, 'Whether or not to freeze the '
    'teacher model during distillation.')
flags.DEFINE_bool(
    'reset_global_step', True, 'Reset global step to 0 after '
    'loading from init_checkpoint')
flags.DEFINE_bool(
    'reset_optimizer', True, 'Reset optimizer internals after '
    'loading from init_checkpoint')

# Training flags.
flags.DEFINE_bool('evaluate_during_train', False,
                  'Whether or not to have the GPU train job perform evaluation '
                  'between epochs.')
flags.DEFINE_bool('from_scratch', False,
                  'Train from scratch. Do not restore the last checkpoint.')
flags.DEFINE_bool('no_checkpointing', False,
                  'Do not save model checkpoints during training.')
flags.DEFINE_integer('epoch_length', 1000,
                     'Number of gradient steps per epoch.')
flags.DEFINE_integer('num_train_steps', int(1e6),
                     'Number of gradient steps to train for.')
flags.DEFINE_integer('selfsup_after_num_steps', int(5e5),
                     'Number of gradient steps before self-supervision.')
flags.DEFINE_integer('selfsup_ramp_up_steps', int(1e5),
                     'Number of gradient steps for ramping up self-sup.')
flags.DEFINE_integer(
    'selfsup_step_cycle', int(1e10),
    'Number steps until the step counter for self-supervsion is reset.')
flags.DEFINE_integer('shuffle_buffer_size', 1024,
                     'Shuffle buffer size for training.')
flags.DEFINE_integer('height', 640, 'Image height for training and evaluation.')
flags.DEFINE_integer('width', 640, 'Image height for training and evaluation.')
flags.DEFINE_bool('crop_instead_of_resize', False, 'Crops images for training '
                  'instead of resizing the images.')
flags.DEFINE_integer('seq_len', 2, 'Sequence length for training flow.')
flags.DEFINE_integer('batch_size', 1, 'Batch size for training flow on '
                     'gpu.')
flags.DEFINE_string('optimizer', 'adam', 'One of "adam", "sgd"')
flags.DEFINE_float('gpu_learning_rate', 1e-4, 'Learning rate for training '
                   'UFlow on GPU.')
flags.DEFINE_integer('lr_decay_after_num_steps', 0, '')
flags.DEFINE_integer('lr_decay_steps', 0, '')
flags.DEFINE_string('lr_decay_type', 'none',
                    'One of ["none", "exponential", "linear", "gaussian"]')
flags.DEFINE_bool(
    'stop_gradient_mask', True, 'Whether or not to stop the '
    'gradient propagation through the occlusion mask.')
flags.DEFINE_integer('num_occlusion_iterations', 1,
                     'If occlusion estimation is "iterative"')
flags.DEFINE_bool('only_forward', False, '')
# Data augmentation (-> now gin configurable)
flags.DEFINE_string('teacher_image_version', 'original',
                    'one of original, augmented')
flags.DEFINE_float(
    'channel_multiplier', 1.,
    'Globally multiply the number of model convolution channels'
    'by this factor.')
flags.DEFINE_integer('num_levels', 5, 'The number of feature pyramid levels to '
                     'use.')
flags.DEFINE_bool('use_cost_volume', True, 'Whether or not to compute the '
                  'cost volume.')
flags.DEFINE_bool(
    'use_feature_warp', True, 'Whether or not to warp the '
    'model features when computing flow.')
flags.DEFINE_bool(
    'accumulate_flow', True, 'Whether or not to predict a flow '
    'adjustment on each feature pyramid level.')
flags.DEFINE_integer('level1_num_layers', 3, '')
flags.DEFINE_integer('level1_num_filters', 32, '')
flags.DEFINE_integer('level1_num_1x1', 0, '')
flags.DEFINE_float('dropout_rate', 0.1, 'Amount of level dropout.')
flags.DEFINE_bool('normalize_before_cost_volume', True, '')
flags.DEFINE_bool('original_layer_sizes', False, '')
flags.DEFINE_bool('shared_flow_decoder', False, '')
flags.DEFINE_bool('resize_selfsup', True, '')
flags.DEFINE_integer(
    'selfsup_crop_height', 64,
    'Number of pixels removed from the image at top and bottom'
    'for self-supervision.')
flags.DEFINE_integer(
    'selfsup_crop_width', 64,
    'Number of pixels removed from the image left and right'
    'for self-supervision.')
flags.DEFINE_integer(
    'selfsup_max_shift', 0,
    'Number of pixels removed from the image at top and bottom, left and right'
    'for self-supervision.')
flags.DEFINE_float(
    'fb_sigma_teacher', 0.003,
    'Forward-backward consistency scaling constant used for self-supervision.')
flags.DEFINE_float(
    'fb_sigma_student', 0.03,
    'Forward-backward consistency scaling constant used for self-supervision.')
flags.DEFINE_string('selfsup_mask', 'gaussian',
                    'One of [gaussian, ddflow, advection]')
flags.DEFINE_float('weight_photo', 0.0, 'Weight for photometric loss.')
flags.DEFINE_float('weight_ssim', 0.0, 'Weight for SSIM loss.')
flags.DEFINE_float('weight_census', 1.0, 'Weight for census loss.')
flags.DEFINE_float('weight_smooth1', 0.0, 'Weight for smoothness loss.')
flags.DEFINE_float('weight_smooth2', 2.0, 'Weight for smoothness loss.')
flags.DEFINE_float('smoothness_edge_constant', 150.,
                   'Edge constant for smoothness loss.')
flags.DEFINE_string('smoothness_edge_weighting', 'exponential',
                    'One of: gaussian, exponential')
flags.DEFINE_integer('smoothness_at_level', 2, '')

flags.DEFINE_float('weight_selfsup', 0.6, 'Weight for self-supervision loss.')
flags.DEFINE_float('weight_transl_consist', 0.0,
                   'Weight for loss enforcing uniform source usage.')

# Occlusion estimation parameters
flags.DEFINE_string('occlusion_estimation', 'wang',
                    'One of: none, brox, wang, uflow')
flags.DEFINE_integer('occ_after_num_steps_brox', 0, '')
flags.DEFINE_integer('occ_after_num_steps_wang', 0, '')
flags.DEFINE_integer('occ_after_num_steps_fb_abs', 0, '')
flags.DEFINE_integer('occ_after_num_steps_forward_collision', 0, '')
flags.DEFINE_integer('occ_after_num_steps_backward_zero', 0, '')
flags.DEFINE_float('occ_weights_fb_abs', 1000.0, '')
flags.DEFINE_float('occ_weights_forward_collision', 1000.0, '')
flags.DEFINE_float('occ_weights_backward_zero', 1000.0, '')
flags.DEFINE_float('occ_thresholds_fb_abs', 1.5, '')
flags.DEFINE_float('occ_thresholds_forward_collision', 0.4, '')
flags.DEFINE_float('occ_thresholds_backward_zero', 0.25, '')
flags.DEFINE_float('occ_clip_max_fb_abs', 10.0, '')
flags.DEFINE_float('occ_clip_max_forward_collision', 5.0, '')

flags.DEFINE_string(
    'distance_census', 'ddflow', 'Which type of distance '
    'metric to use when computing loss.')
flags.DEFINE_string(
    'distance_photo', 'robust_l1', 'Which type of distance '
    'metric to use when computing loss.')
flags.DEFINE_bool('use_supervision', False, 'Whether or not to train with '
                  'a supervised loss.')
flags.DEFINE_bool('resize_gt_flow_supervision', True, 'Whether or not to '
                  'resize ground truth flow for the supervised loss.')
flags.DEFINE_bool('use_gt_occlusions', False, 'Whether or not to train with '
                  'a ground trouth occlusion')
# Gin params are used to specify which augmentations to perform.
flags.DEFINE_multi_string(
    'config_file', None,
    'Path to a Gin config file. Can be specified multiple times. '
    'Order matters, later config files override former ones.')

flags.DEFINE_multi_string(
    'gin_bindings', None,
    'Newline separated list of Gin parameter bindings. Can be specified '
    'multiple times. Overrides config from --config_file.')
