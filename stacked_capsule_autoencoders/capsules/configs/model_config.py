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

"""Model config for MNIST."""

from absl import flags
from monty.collections import AttrDict
import sonnet as snt
import tensorflow.compat.v1 as tf
from stacked_capsule_autoencoders.capsules import primary
from stacked_capsule_autoencoders.capsules.attention import SetTransformer
from stacked_capsule_autoencoders.capsules.models.constellation import ConstellationAutoencoder
from stacked_capsule_autoencoders.capsules.models.constellation import ConstellationCapsule
from stacked_capsule_autoencoders.capsules.models.scae import ImageAutoencoder
from stacked_capsule_autoencoders.capsules.models.scae import ImageCapsule

flags.DEFINE_float('lr', 1e-4, 'Learning rate.')
flags.DEFINE_boolean('use_lr_schedule', True, 'Uses learning rate schedule'
                     ' if True.')

flags.DEFINE_integer('template_size', 11, 'Template size.')
flags.DEFINE_integer('n_part_caps', 16, 'Number of part capsules.')
flags.DEFINE_integer('n_part_caps_dims', 6, 'Part caps\' dimensionality.')
flags.DEFINE_integer('n_part_special_features', 16, 'Number of special '
                     'features.')

flags.DEFINE_integer('n_channels', 1, 'Number of input channels.')

flags.DEFINE_integer('n_obj_caps', 10, 'Number of object capsules.')
flags.DEFINE_integer('n_obj_caps_params', 32, 'Dimensionality of object caps '
                     'feature vector.')

flags.DEFINE_boolean('colorize_templates', False, 'Whether to infer template '
                     'color from input.')
flags.DEFINE_boolean('use_alpha_channel', False, 'Learns per-pixel mixing '
                     'proportions for every template; otherwise mixing '
                     'probabilities are constrained to have the same value as '
                     'image pixels.')

flags.DEFINE_string('template_nonlin', 'relu1', 'Nonlinearity used to normalize'
                    ' part templates.')
flags.DEFINE_string('color_nonlin', 'relu1', 'Nonlinearity used to normalize'
                    ' template color (intensity) value.')

flags.DEFINE_float('prior_within_example_sparsity_weight', 1., 'Loss weight.')
flags.DEFINE_float('prior_between_example_sparsity_weight', 1., 'Loss weight.')
flags.DEFINE_float('posterior_within_example_sparsity_weight', 10.,
                   'Loss weight.')
flags.DEFINE_float('posterior_between_example_sparsity_weight', 10.,
                   'Loss weight.')


def get(config):
  """Builds the model."""

  if config.model == 'scae':
    model = make_scae(config)

  elif config.model == 'constellation':
    model = make_constellation(config)

  else:
    raise ValueError('Unknown model type: "{}".'.format(config.model))

  lr = config.lr
  if config.use_lr_schedule:
    global_step = tf.train.get_or_create_global_step()
    lr = tf.train.exponential_decay(
        global_step=global_step,
        learning_rate=lr,
        decay_steps=1e4,
        decay_rate=.96)

  eps = 1e-2 / float(config.batch_size)  ** 2
  opt = tf.train.RMSPropOptimizer(config.lr, momentum=.9, epsilon=eps)

  return AttrDict(model=model, opt=opt, lr=config.lr)


def make_scae(config):
  """Builds the SCAE."""

  img_size = [config.canvas_size] * 2
  template_size = [config.template_size] * 2

  cnn_encoder = snt.nets.ConvNet2D(
      output_channels=[128] * 4,
      kernel_shapes=[3],
      strides=[2, 2, 1, 1],
      paddings=[snt.VALID],
      activate_final=True)

  part_encoder = primary.CapsuleImageEncoder(
      cnn_encoder,
      config.n_part_caps,
      config.n_part_caps_dims,
      n_features=config.n_part_special_features,
      similarity_transform=False,
      encoder_type='conv_att')

  part_decoder = primary.TemplateBasedImageDecoder(
      output_size=img_size,
      template_size=template_size,
      n_channels=config.n_channels,
      learn_output_scale=False,
      colorize_templates=config.colorize_templates,
      use_alpha_channel=config.use_alpha_channel,
      template_nonlin=config.template_nonlin,
      color_nonlin=config.color_nonlin,
  )

  obj_encoder = SetTransformer(
      n_layers=3,
      n_heads=1,
      n_dims=16,
      n_output_dims=256,
      n_outputs=config.n_obj_caps,
      layer_norm=True,
      dropout_rate=0.)

  obj_decoder = ImageCapsule(
      config.n_obj_caps,
      2,
      config.n_part_caps,
      n_caps_params=config.n_obj_caps_params,
      n_hiddens=128,
      learn_vote_scale=True,
      deformations=True,
      noise_type='uniform',
      noise_scale=4.,
      similarity_transform=False)

  model = ImageAutoencoder(
      primary_encoder=part_encoder,
      primary_decoder=part_decoder,
      encoder=obj_encoder,
      decoder=obj_decoder,
      input_key='image',
      label_key='label',
      n_classes=10,
      dynamic_l2_weight=10,
      caps_ll_weight=1.,
      vote_type='enc',
      pres_type='enc',
      stop_grad_caps_inpt=True,
      stop_grad_caps_target=True,
      prior_sparsity_loss_type='l2',
      prior_within_example_sparsity_weight=config.prior_within_example_sparsity_weight,  # pylint:disable=line-too-long
      prior_between_example_sparsity_weight=config.prior_between_example_sparsity_weight,  # pylint:disable=line-too-long
      posterior_sparsity_loss_type='entropy',
      posterior_within_example_sparsity_weight=config.posterior_within_example_sparsity_weight,  # pylint:disable=line-too-long
      posterior_between_example_sparsity_weight=config.posterior_between_example_sparsity_weight,  # pylint:disable=line-too-long
      )

  return model


flags.DEFINE_float('mixing_kl_weight', 0., '')
flags.DEFINE_float('sparsity_weight', 10., '')
flags.DEFINE_float('dynamic_l2_weight', 10., '')


def make_constellation(config):
  """Builds the constellation model."""

  n_caps = 3

  encoder = SetTransformer(
      n_layers=4,
      n_heads=4,
      n_dims=128,
      n_output_dims=32,
      n_outputs=n_caps,
      layer_norm=True,
      dropout_rate=0.,
  )

  decoder = ConstellationCapsule(
      n_caps=n_caps,
      n_caps_dims=2,  # only y, x
      n_caps_params=32,
      n_votes=4,
      n_hiddens=128,
      learn_vote_scale=True,
      deformations=True,
      noise_type='uniform',
      noise_scale=4,
      similarity_transform=True,
  )

  model = ConstellationAutoencoder(
      encoder=encoder,
      decoder=decoder,
      mixing_kl_weight=config.mixing_kl_weight,
      sparsity_weight=config.sparsity_weight,
      dynamic_l2_weight=config.dynamic_l2_weight,
      #
      prior_sparsity_loss_type='l2',
      prior_within_example_sparsity_weight=config.prior_within_example_sparsity_weight,  # pylint:disable=line-too-long
      prior_between_example_sparsity_weight=config.prior_within_example_sparsity_weight,  # pylint:disable=line-too-long
      prior_within_example_constant=0.,
      posterior_sparsity_loss_type='entropy',
      posterior_within_example_sparsity_weight=config.posterior_within_example_sparsity_weight,  # pylint:disable=line-too-long
      posterior_between_example_sparsity_weight=config.posterior_between_example_sparsity_weight,  # pylint:disable=line-too-long
  )

  return model
