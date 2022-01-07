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

"""Flags used by smurf training and evaluation."""
from absl import flags

FLAGS = flags.FLAGS

# General flags.
flags.DEFINE_bool(
    'no_tf_function', False, 'If True, run without'
    ' tf functions. This incurs a performance hit, but can'
    ' make debugging easier.')
flags.DEFINE_string('train_on', '',
                    '"format0:path0;format1:path1", e.g. "kitti:/tmp/..."')
flags.DEFINE_string('eval_on', '',
                    '"format0:path0;format1:path1", e.g. "kitti:/tmp/..."')
flags.DEFINE_string('plot_dir', '', 'Path to directory where plots are saved.')
flags.DEFINE_string('checkpoint_dir', '',
                    'Path to directory for saving and restoring checkpoints.')
flags.DEFINE_string('init_checkpoint_dir', '',
                    'Path to directory for initializing from a checkpoint.')
flags.DEFINE_bool('check_data', False,
                  'Flag to indicate whether to check the dataset.')
flags.DEFINE_bool(
    'plot_debug_info', False,
    'Flag to indicate whether to plot debug info during training.')
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
flags.DEFINE_integer('num_train_steps', int(75000),
                     'Number of gradient steps to train for.')
flags.DEFINE_integer('selfsup_after_num_steps', int(31250),
                     'Number of gradient steps before self-supervision.')
flags.DEFINE_integer('selfsup_ramp_up_steps', int(6250),
                     'Number of gradient steps for ramping up self-sup.')
flags.DEFINE_integer('shuffle_buffer_size', 1024,
                     'Shuffle buffer size for training.')
flags.DEFINE_integer('height', 296, 'Image height for training and evaluation.')
flags.DEFINE_integer('width', 696, 'Image height for training and evaluation.')
flags.DEFINE_bool('crop_instead_of_resize', True, 'Crops images for training '
                  'instead of resizing the images.')
flags.DEFINE_integer('seq_len', 2, 'Sequence length for training flow.')
flags.DEFINE_integer('virtual_gpus', 1, 'How many virtual GPUs to run with.')
flags.DEFINE_integer('global_gpu_batch_size', 1, 'Batch size for training flow '
                     'on gpu. If using multiple GPUs, this is the sum of the '
                     'batch size across all GPU replicas.')
flags.DEFINE_integer('num_gpus', 1, '')
flags.DEFINE_string('optimizer', 'adam', 'One of "adam", "sgd"')
flags.DEFINE_bool('gradient_clipping', True, 'Apply gradient clipping.')
flags.DEFINE_float('gradient_clipping_max_value', 1.0, 'Maximum value used '
                   'for the gradient clipping if active.')
flags.DEFINE_float('start_learning_rate', 1e-4, 'The initial learning rate '
                   'which will be warmed up into the final learning rate.')
flags.DEFINE_float('warm_up_steps', 0, 'Number of steps to warm up into the '
                   'final learning rate.')
flags.DEFINE_float('gpu_learning_rate', 2e-4, 'Learning rate for training '
                   'SMURF on GPU.')
flags.DEFINE_integer('lr_decay_after_num_steps', 62500, '')
flags.DEFINE_integer('lr_decay_steps', 2500, '')
flags.DEFINE_string('lr_decay_type', 'exponential',
                    'One of ["none", "exponential"]')
flags.DEFINE_bool(
    'stop_gradient_mask', True, 'Whether or not to stop the '
    'gradient propagation through the occlusion mask.')
flags.DEFINE_bool(
    'full_size_warp', True, 'Whether or not to perform the warp '
    'at full resolution.')
flags.DEFINE_integer('num_occlusion_iterations', 1,
                     'If occlusion estimation is "iterative"')
flags.DEFINE_bool('only_forward', False, 'Only compute loss in the forward '
                  'temporal direction.')
flags.DEFINE_string('teacher_image_version', 'original',
                    'one of original, augmented')
flags.DEFINE_bool('log_per_replica_values', False, 'Whether or not to log per '
                  'replica info.')
flags.DEFINE_float('dropout_rate', 0.1, 'Amount of level dropout.')
flags.DEFINE_bool(
    'resize_selfsup', True, 'Bilinearly resize the cropped image'
    'during self-supervision.')
flags.DEFINE_integer(
    'selfsup_crop_height', 64,
    'Number of pixels removed from the image at top and bottom'
    'for self-supervision.')
flags.DEFINE_integer(
    'selfsup_crop_width', 64,
    'Number of pixels removed from the image left and right'
    'for self-supervision.')
flags.DEFINE_float(
    'fb_sigma_teacher', 0.003,
    'Forward-backward consistency scaling constant used for self-supervision.')
flags.DEFINE_float(
    'fb_sigma_student', 0.03,
    'Forward-backward consistency scaling constant used for self-supervision.')
flags.DEFINE_string('selfsup_mask', 'gaussian',
                    'One of [gaussian, ddflow, advection, none]')

flags.DEFINE_float('weight_supervision', 0.1, 'Weight for the supervised-loss.')
flags.DEFINE_float('weight_census', 1.0, 'Weight for census loss.')
flags.DEFINE_float('weight_smooth1', 0.0, 'Weight for smoothness loss.')
flags.DEFINE_float('weight_smooth2', 2.0, 'Weight for smoothness loss.')
flags.DEFINE_float('smoothness_edge_constant', 150.,
                   'Edge constant for smoothness loss.')
flags.DEFINE_string('smoothness_edge_weighting', 'exponential',
                    'One of: gaussian, exponential')
flags.DEFINE_integer('smoothness_at_level', 2, 'Resolution level at which the '
                     'smoothness loss will be applied if active.')
flags.DEFINE_integer('smoothness_after_num_steps', -1,
                     'Number of steps to take before turning on smoothness '
                     'loss.')

flags.DEFINE_float('weight_selfsup', 0.3, 'Weight for self-supervision loss.')

# Occlusion estimation parameters
flags.DEFINE_string('occlusion_estimation', 'wang',
                    'One of: none, brox, wang')

flags.DEFINE_integer('occ_after_num_steps_brox', 25000, '')
flags.DEFINE_integer('occ_after_num_steps_wang', 0, '')
flags.DEFINE_integer('occ_after_num_steps_forward_collision', 0, '')

flags.DEFINE_string(
    'distance_census', 'ddflow', 'Which type of distance '
    'metric to use when computing loss.')
flags.DEFINE_string(
    'feature_architecture', 'raft',
    'Which type of feature architecture to use. '
    'Supported values are pwc or raft.')
flags.DEFINE_string(
    'flow_architecture', 'raft', 'Which type of flow architecture to use. '
    'Supported values are pwc or raft.')
flags.DEFINE_string(
    'train_mode', 'sequence-unsupervised',
    'Controls what kind of training loss '
    'should be used. Can be one of the following options: '
    'unsupervised, supervised, sequence-supervised, sequence-unsupervised.')
flags.DEFINE_bool(
    'resize_gt_flow_supervision', True, 'Whether or not to '
    'resize ground truth flow for the supervised loss.')
flags.DEFINE_multi_string(
    'config_file', None,
    'Path to a Gin config file. Can be specified multiple times. '
    'Order matters, later config files override former ones.')
flags.DEFINE_multi_string(
    'gin_bindings', None,
    'Newline separated list of Gin parameter bindings. Can be specified '
    'multiple times. Overrides config from --config_file.')
flags.DEFINE_bool('run_eval_once', False, 'If True, run the evaluator only one '
                  'time.')
