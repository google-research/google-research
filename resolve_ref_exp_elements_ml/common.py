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

"""Functionality shared by train and eval."""

import tensorflow.compat.v1 as tf
flags = tf.app.flags

flags.DEFINE_enum(
    'output_mode', 'combined', ['segment', 'regression', 'combined'],
    '"segment", uses a model similar to DeepLab.'
    '"regression", uses the ClickRegression model'
    '"combined", uses the multitask learning architecture')

flags.DEFINE_string('master', '', 'name of the tensorflow server')

flags.DEFINE_integer('image_size', 513, '')

flags.DEFINE_integer(
    'logits_kernel_size', 1,
    'The kernel size for the convolutional kernel that '
    'generates logits.')

# Settings for model variants.

flags.DEFINE_string('model_variant', 'mobilenet_v2', 'DeepLab model variant.')

flags.DEFINE_string('pretrained_text_enc_name',
                    'https://tfhub.dev/google/universal-sentence-encoder/2',
                    'Text embedding to use for elements.')

flags.DEFINE_string(
    'pretrained_elements_ref_match_model',
    'https://tfhub.dev/google/nnlm-en-dim128-with-normalization/1', '')

flags.DEFINE_multi_float('image_pyramid', [1.0],
                         'Input scales for multi-scale feature extraction.')

flags.DEFINE_multi_integer('atrous_rates', [6, 12, 18],
                           'Atrous rates for atrous spatial pyramid pooling.')

flags.DEFINE_boolean('add_image_level_feature', True,
                     'Add image level feature.')

flags.DEFINE_boolean('aspp_with_batch_norm', True,
                     'Use batch norm parameters for ASPP or not.')

flags.DEFINE_boolean('aspp_with_separable_conv', True,
                     'Use separable convolution for ASPP or not.')

flags.DEFINE_multi_integer('multi_grid', [1, 1, 1],
                           'Employ a hierarchy of atrous rates for ResNet.')
flags.DEFINE_float('comb_dropout_keep_prob', 1.0, '')
flags.DEFINE_float('image_keep_prob', 1.0, '')
flags.DEFINE_float('elements_keep_prob', .75, '')

flags.DEFINE_float(
    'depth_multiplier', 1.0,
    'Multiplier for the depth (number of channels) for all '
    'convolution ops used in MobileNet.')

flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')

flags.DEFINE_integer(
    'decoder_output_stride', None,
    'The ratio of input to output spatial resolution when '
    'employing decoder to refine segmentation results.')

flags.DEFINE_boolean('decoder_use_separable_conv', True,
                     'Employ separable convolution for decoder or not.')

flags.DEFINE_enum('merge_method', 'max', ['max', 'avg'],
                  'Scheme to merge multi scale features.')

flags.DEFINE_boolean(
    'use_ref_exp', True,
    'Whether or not to use the referring expression in the model.')
flags.DEFINE_boolean(
    'use_elements_texts', True,
    'Whether or not to use the elements text in the model.'
    'Crash if this is true when use_elements_boxes is false.')
flags.DEFINE_boolean('use_elements_boxes', True,
                     'Whether or not to use the elements boxes in the model.')
flags.DEFINE_boolean(
    'use_image', True,
    'Whether or not to use the screenshot image in the model.'
    'Set to false for a baseline relying on elements.')
flags.DEFINE_boolean('use_elements_neighbors', False, '')
flags.DEFINE_boolean('use_elements_ref_match', False, '')

flags.DEFINE_enum(
    'merge_ref_elements_method', 'singDotAtten',
    ['', 'combine', 'singDotAtten', 'sepDotAtten', 'combAtten'],
    "'': Don't merge in elements model. 'combine': Concatenate the"
    ' representations and feed through a DNN.'
    " 'singDotAtten': Use the same DNN to calculate the representations"
    " of the items and expression to multiply. 'sepDotAtten' Use"
    ' separate networks to calculate the representations.'
    " 'combAtten': Use a network to directly output multiply values.")

flags.DEFINE_enum(
    'select_attention_method', 'singDotAtten',
    ['singDotAtten', 'sepDotAtten', 'combAtten'],
    'The attention method used to select a given item.'
    " 'singDotAtten': Use the same DNN to calculate the representations"
    " of the items and expression to multiply. 'sepDotAtten' Use"
    ' separate networks to calculate the representations.'
    " 'combAtten': Use a network to directly output multiply values.")

flags.DEFINE_enum(
    'elements_proj_mode', 'step', ['tile', 'step', 'cont'],
    'How to project the elements information onto the image feature.'
    " 'tile': blindly tile the feature over the image."
    " 'step': Tile only in the elements bounding box locations."
    " 'cont': Tile the values in bounding box locations and"
    ' increase magnitude near the center of the box.')
flags.DEFINE_boolean('incorrect_boxes_as_errors', True,
                     'Crash on incorrect box sizes.')
flags.DEFINE_string(
    'add_ref_elements_layer', '',
    'The layer to add the ref and elements representations to.')
flags.DEFINE_boolean(
    'proj_elements_memop', True,
    'Reduces elements projection mem by using a tf while loop.'
    'May be slower.')

flags.DEFINE_boolean('elements_3d_output', True, '')

flags.DEFINE_boolean('elements_cnn', True, '')

flags.DEFINE_boolean('elements_img_comb_cnn', True, '')
flags.DEFINE_integer('elements_img_comb_cnn_layers', 2, '')
flags.DEFINE_integer('elements_enc_size', 512, '')

# Dataset settings.

flags.DEFINE_string('dataset', 'default', 'Name of the segmentation dataset.')

flags.DEFINE_string('dataset_dir', '', 'Where the dataset reside.')

flags.DEFINE_integer('dataset_threads', 100, '')

flags.DEFINE_boolean('preprocess_divide_label', False, '')
flags.DEFINE_integer('shuffle_buffer_size', 10000, '')
flags.DEFINE_integer('file_shuffle_buffer_size', 100, '')

flags.DEFINE_boolean('use_labels', True,
                     'If True, include label in input pipeline.')
flags.DEFINE_boolean(
    'train_mode', True,
    'Specify whether we are in training mode. Used for model ops such as'
    'dropout and batch norm')
flags.DEFINE_boolean('coord_softmax', False,
                     'if True, use the coordinate softmax architecture.')

flags.DEFINE_boolean(
    'regression_batch_norm', False,
    'if True, apply batch normalization to all ClickRegression-specific'
    'convolutions')

flags.DEFINE_boolean(
    'use_groundtruth_box', False,
    'if True, use the xmin,xmax,ymin,ymax features of the ground truth box')
