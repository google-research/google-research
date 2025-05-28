# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Hyper-Transformer flags."""
from absl import flags

flags.DEFINE_integer('samples_transformer', 32,
                     'Number of samples fed to the transformer.')
flags.DEFINE_integer('samples_cnn', 128,
                     'Number of samples fed to the CNN.')
flags.DEFINE_integer('use_labels', 8, 'First N labels to use for training '
                     '(overwritten by `train_dataset` label specification).')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate.')
flags.DEFINE_float('learning_rate_decay_steps', 10000,
                   'Learning rate decay steps.')
flags.DEFINE_float('learning_rate_decay_rate', 1.0,
                   'Learning rate decay rate.')
flags.DEFINE_integer('train_steps', 10000, 'Number of training steps.')
flags.DEFINE_string('data_numpy_dir', '', 'Location of the NumPy dataset '
                    'cache (can be useful for small datasets).')
flags.DEFINE_string('data_dir', None, 'Location of the cached tfds data.')
flags.DEFINE_integer('num_labels', 4, 'Number of labels to pick for each task.')
KERNEL_SIZE = flags.DEFINE_integer('kernel_size', 3,
                                   'Kernel size for the conv layers.')
STRIDE = flags.DEFINE_integer('stride', 2, 'Stride of the conv layers.')
flags.DEFINE_integer('image_size', 28, 'Image size.')
flags.DEFINE_integer('num_sample_repeats', 1,
                     'Number of repeats for each task.')
flags.DEFINE_integer('steps_between_saves', 50000, 'Number of steps between '
                     'model saves.')
flags.DEFINE_string('cnn_model_name', '3-layer', 'CNN model to train.')
flags.DEFINE_integer('embedding_dim', 8, 'Dimensionality of the embedding.')
flags.DEFINE_integer('query_key_dim', 256, 'Transformer key and query '
                     'dimensionality.')
flags.DEFINE_integer('value_dim', 256, 'Transformer value dimensionality.')
flags.DEFINE_integer('internal_dim', 512, 'Transformer internal depth.')
flags.DEFINE_integer('num_layers', 4, 'Number of Transformer layers.')
flags.DEFINE_integer('heads', 4, 'Number of Transformer heads.')
NUMBER_OF_TRAINED_CNN_LAYERS = flags.DEFINE_integer(
    'number_of_trained_cnn_layers', 0, 'Number of first layers'
    'of CNN that are trained using backpropagation instead of'
    'given by Transformer. If negative, this represents the number of last'
    'layers instead of the first.')
flags.DEFINE_float('dropout_rate', 0.0, 'Transformer dropout rate.')
flags.DEFINE_float('cnn_dropout_rate', 0.0, 'CNN dropout rate.')
flags.DEFINE_float('label_smoothing', 0.0, 'Cross-entropy label smoothing.')

flags.DEFINE_float('rotation_probability', 0.0, 'Probability of applying a '
                   'random rotation to an image.')
flags.DEFINE_float('boundary_probability', 0.0, 'Probability of extracting '
                   'image boundary.')
flags.DEFINE_float('smooth_probability', 0.5, 'Probability of smoothing an '
                   'image.')
flags.DEFINE_float('contrast_probability', 0.5, 'Probability of increasing '
                   'image contrast.')
flags.DEFINE_float('resize_probability', 0.0, 'Probability of resizing an '
                   'image.')
flags.DEFINE_float('negate_probability', 0.0, 'Probability of negating an '
                   'image.')
flags.DEFINE_float('roll_probability', 0.0, 'Probability of rolling the image.')
flags.DEFINE_float('angle_range', 180.0, 'Random angle variation.')
flags.DEFINE_bool('random_rotate_by_90', False, 'If set, the augmentation is a'
                  'random rotation by 0, 90, 180, or 270 degrees.')
flags.DEFINE_enum('layerwise_generator', 'joint', ['joint', 'separate'],
                  'Weight generator architecture to use: "joint" feeds both '
                  'input samples and weight placeholders to the Transformer, '
                  'while "separate" feeds samples into Encoder and '
                  'weights into Decoder.')
flags.DEFINE_string('num_layerwise_features', '', 'Number of features in the '
                    'layerwise feature extractor. If not specified, will be '
                    'set to the maximum of input and output channels. If set '
                    'to zero, will only use shared features.')
flags.DEFINE_float('lw_key_query_dim', 1.0, 'Dimensionality of the KQ '
                   'transformer self-attention module (fraction of the '
                   'embedding size if <3).')
flags.DEFINE_float('lw_value_dim', 1.0, 'Dimensionality of the value '
                   'transformer self-attention module (fraction of the '
                   'embedding size if <3).')
flags.DEFINE_float('lw_inner_dim', 1.0, 'Inner transformer dimension of '
                   'the self-attention module (fraction of the '
                   'embedding size if <3).')
flags.DEFINE_bool('lw_use_nonlinear_feature', False, 'Whether to use nonlinear '
                  'features produced by the feature extractor.')
flags.DEFINE_string('train_dataset', 'emnist', 'Training dataset '
                    'specification.')

# Currently, both approaches seem to produce very similar results, but
# 'output' weight allocation is more suitable for generating BN parameters
# (currently supported only with this weight allocation scheme).

flags.DEFINE_enum('lw_weight_allocation', 'spatial', ['spatial', 'output'],
                  'The approach to generating convolutional kernels: '
                  'either allocating Transformer weight embeddings to be '
                  'spatial kernel slices, or slices in the output dimension.')

flags.DEFINE_bool('lw_generate_bias', False, 'If set, convolutional layer '
                  'biases are also generated.')

flags.DEFINE_bool('lw_generate_bn', False,
                  'Whether to use generate batch normalization beta and gamma'
                  '(currently only supported for "output" weight allocation).')

flags.DEFINE_bool('per_label_augmentation', False,
                  'If set, we apply different augmentations to samples with '
                  'different labels (otherwise, the augmentation is the same '
                  'for all samples).')
flags.DEFINE_bool('use_decoder', False,
                  'If set, we use an Encoder-Decoder model (both getting the '
                  'input sequence) instead of an Encoder-based model.')
flags.DEFINE_bool('add_trainable_weights', False,
                  'If set, trainable weights are added to the generated '
                  'weights.')

flags.DEFINE_bool('balanced_batches', False,
                  'Whether to generate Transformer and CNN batches separately '
                  'thus providing a guarantee that each batch is balanced.')
flags.DEFINE_float('weight_variation_regularization', 0.0,
                   'Regularization weight used for regularizing a variation '
                   'of generated weights (works only when the '
                   '`add_trainable_weights` flag is set).')
flags.DEFINE_integer('shuffle_labels_seed', 0,
                     'Seed used for shuffling labels at initialization (0 '
                     'for no shuffling). Useful for datasets with many ordered '
                     'labels like Omniglot.')

flags.DEFINE_string('test_dataset', 'emnist:8-61', 'Test dataset '
                    'specification.')
flags.DEFINE_float('test_rotation_probability', -1, 'Probability of applying '
                   'a random rotation to a test image.')
flags.DEFINE_float('test_smooth_probability', -1, 'Probability of smoothing '
                   'a test image.')
flags.DEFINE_float('test_contrast_probability', -1, 'Probability of '
                   'increasing test image contrast.')
flags.DEFINE_float('test_resize_probability', -1, 'Probability of resizing '
                   'a test image.')
flags.DEFINE_float('test_negate_probability', -1, 'Probability of negating '
                   'a test image.')
flags.DEFINE_float('test_roll_probability', -1, 'Probability of rolling a '
                   'test image.')
flags.DEFINE_float('test_angle_range', -1, 'Random angle variation for test '
                   'images.')
flags.DEFINE_bool('test_random_rotate_by_90', False, 'Random rotation by 0, '
                  '90, 180, or 270 degrees for test images.')
flags.DEFINE_bool('test_per_label_augmentation', False,
                  'If set, we apply different augmentations to samples with '
                  'different labels for test samples.')
flags.DEFINE_string('test_split', 'train', 'Dataset split to use for testing '
                    '(since we usually use disjoint label sets, it can be set '
                    'to train)')

flags.DEFINE_enum('transformer_activation', 'softmax', ['softmax', 'sigmoid'],
                  'Activation function to use in the Transformer attention.')
flags.DEFINE_enum('transformer_nonlinearity', 'relu', ['gelu', 'relu', 'lrelu'],
                  'Default nonlinearity used for Transformer MLP layer '
                  '(also called "point-wise feed forward"). Vision Transformer '
                  'uses GELU by default.')
flags.DEFINE_enum('cnn_activation', 'relu', ['relu', 'lrelu'],
                  'CNN activation function.')
flags.DEFINE_bool('transformer_skip_last_nonlinearity', False,
                  'If set, we do not use output nonlinearity at the final '
                  'Transformer encoder layer.')

flags.DEFINE_enum('shared_feature_extractor', 'none',
                  ['none', '2-layer', '3-layer', '4-layer'],
                  'Shared feature extractor model.')
flags.DEFINE_integer('shared_features_dim', 32,
                     'Default number of channels in the shared feature '
                     'extractor network.')
flags.DEFINE_bool('separate_evaluation_bn_vars', False,
                  'If set, we use different CNN BN variables in the evaluation '
                  'model and the CNN backbone used in the weight generator.')
flags.DEFINE_enum('shared_feature_extractor_padding', 'valid',
                  ['valid', 'same'], 'Convolutional layer padding to use in '
                  'the shared feature extractor.')

flags.DEFINE_bool('apply_image_augmentations', False,
                  'If set, random typical image augmentation is applied to '
                  'each image.')
flags.DEFINE_integer('default_num_channels', 16,
                     'Default number of CNN channels for each layer.')

flags.DEFINE_integer('unlabeled_samples_per_class', 0,
                     'Number of unlabeled samples per label in the training '
                     '(or Transformer) batch.')

flags.DEFINE_integer('warmup_steps', 0, 'Number of steps for the warmup '
                     'when we train layer heads consecutively (0 disables '
                     'warmup).')

flags.DEFINE_float('max_prob_remove_unlabeled', 0.0,
                   'Maximum probability of randomly removing unlabeled '
                   'samples from the support set.')
flags.DEFINE_float('max_prob_remove_labeled', 0.0,
                   'Maximum probability of randomly removing labeled '
                   'samples from the support set.')

flags.DEFINE_float('shared_fe_dropout', 0.0, 'Shared feature extractor '
                   'dropout.')
flags.DEFINE_float('fe_dropout', 0.0, 'Local feature extractor dropout.')

flags.DEFINE_bool('augment_images_individually', False,
                  'If set, each image is augmented individually.')

flags.DEFINE_float('l2_reg_weight', 0.0, 'Regularization weight.')
flags.DEFINE_enum('logits_feature_extractor', 'default',
                  ['default', 'passthrough', 'mix'],
                  'Logits layer feature extractor to use (`default` applies '
                  'the same feature extractor as for conv, `passthrough` '
                  'just passes embeddings through and `mix` mixes both types).')

PRETRAIN_SHARED_FEATURE = flags.DEFINE_bool(
    'pretrain_shared_feature', False,
    'If set, we pre-train shared feature without HT components.')
SHARED_HEAD_WEIGHT = flags.DEFINE_float(
    'shared_head_weight', 0.0,
    'The weight of the loss for the head on top of the shared feature.')
RESTORE_SHARED_FEATURES_FROM = flags.DEFINE_string(
    'restore_shared_features_from', '',
    'Path to the checkpoint for restoring the shared feature from.')

flags.DEFINE_string('train_log_dir', '/tmp/experiment', 'Path for saving '
                    'checkpoints and summaries to.')
