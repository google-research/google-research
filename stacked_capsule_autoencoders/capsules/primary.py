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

"""Primary capsuls."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from monty.collections import AttrDict
import numpy as np

import sonnet as snt
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

from stacked_capsule_autoencoders.capsules import math_ops
from stacked_capsule_autoencoders.capsules import neural
from stacked_capsule_autoencoders.capsules import prob
from stacked_capsule_autoencoders.capsules.tensor_ops import make_brodcastable
from tensorflow.contrib import resampler as contrib_resampler

tfd = tfp.distributions


class CapsuleImageEncoder(snt.AbstractModule):
  """Primary capsule for images."""
  OutputTuple = collections.namedtuple(  # pylint:disable=invalid-name
      'PrimaryCapsuleTuple',
      'pose feature presence presence_logit '
      'img_embedding')

  def __init__(self,
               encoder,
               n_caps,
               n_caps_dims,
               n_features=0,
               noise_scale=4.,
               similarity_transform=False,
               encoder_type='linear',
               **encoder_kwargs):

    super(CapsuleImageEncoder, self).__init__()
    self._encoder = encoder
    self._n_caps = n_caps
    self._n_caps_dims = n_caps_dims
    self._n_features = n_features
    self._noise_scale = noise_scale
    self._similarity_transform = similarity_transform
    self._encoder_type = encoder_type
    self._encoder_kwargs = dict(
        n_layers=2, n_heads=4, n_dims=32, layer_norm=False)
    self._encoder_kwargs.update(encoder_kwargs)

  def _build(self, x):
    batch_size = x.shape[0]
    img_embedding = self._encoder(x)

    splits = [self._n_caps_dims, self._n_features, 1]  # 1 for presence
    n_dims = sum(splits)

    if self._encoder_type == 'linear':
      n_outputs = self._n_caps * n_dims

      h = snt.BatchFlatten()(img_embedding)
      h = snt.Linear(n_outputs)(h)

    else:
      h = snt.AddBias(bias_dims=[1, 2, 3])(img_embedding)

      if self._encoder_type == 'conv':
        h = snt.Conv2D(n_dims * self._n_caps, 1, 1)(h)
        h = tf.reduce_mean(h, (1, 2))
        h = tf.reshape(h, [batch_size, self._n_caps, n_dims])

      elif self._encoder_type == 'conv_att':
        h = snt.Conv2D(n_dims * self._n_caps + self._n_caps, 1, 1)(h)
        h = snt.MergeDims(1, 2)(h)
        h, a = tf.split(h, [n_dims * self._n_caps, self._n_caps], -1)

        h = tf.reshape(h, [batch_size, -1, n_dims, self._n_caps])
        a = tf.nn.softmax(a, 1)
        a = tf.reshape(a, [batch_size, -1, 1, self._n_caps])
        h = tf.reduce_sum(h * a, 1)

      else:
        raise ValueError('Invalid encoder type="{}".'.format(
            self._encoder_type))

    h = tf.reshape(h, [batch_size, self._n_caps, n_dims])

    pose, feature, pres_logit = tf.split(h, splits, -1)
    if self._n_features == 0:
      feature = None

    pres_logit = tf.squeeze(pres_logit, -1)
    if self._noise_scale > 0.:
      pres_logit += ((tf.random.uniform(pres_logit.shape) - .5)
                     * self._noise_scale)


    pres = tf.nn.sigmoid(pres_logit)
    pose = math_ops.geometric_transform(pose, self._similarity_transform)
    return self.OutputTuple(pose, feature, pres, pres_logit, img_embedding)


def choose_nonlinearity(name):
  nonlin = getattr(math_ops, name, getattr(tf.nn, name, None))

  if not nonlin:
    raise ValueError('Invalid nonlinearity: "{}".'.format(name))

  return nonlin


class TemplateBasedImageDecoder(snt.AbstractModule):
  """Template-based primary capsule decoder for images."""

  _templates = None

  def __init__(self,
               output_size,
               template_size,
               n_channels=1,
               learn_output_scale=False,
               colorize_templates=False,
               output_pdf_type='mixture',
               template_nonlin='relu1',
               color_nonlin='relu1',
               use_alpha_channel=False):

    super(TemplateBasedImageDecoder, self).__init__()
    self._output_size = output_size
    self._template_size = template_size
    self._n_channels = n_channels
    self._learn_output_scale = learn_output_scale
    self._colorize_templates = colorize_templates


    self._output_pdf_type = output_pdf_type
    self._template_nonlin = choose_nonlinearity(template_nonlin)
    self._color_nonlin = choose_nonlinearity(color_nonlin)
    self._use_alpha_channel = use_alpha_channel

  @property
  def templates(self):
    self._ensure_is_connected()
    return tf.squeeze(self._templates, 0)

  @snt.reuse_variables
  def make_templates(self, n_templates=None, template_feature=None):

    if self._templates is not None:
      if n_templates is not None and self._templates.shape[1] != n_templates:
        raise ValueError

    else:
      with self._enter_variable_scope():
        # create templates
        n_dims = self._n_channels

        template_shape = ([1, n_templates] + list(self._template_size) +
                          [n_dims])
        n_elems = np.prod(template_shape[2:])

        # make each templates orthogonal to each other at init
        n = max(n_templates, n_elems)
        q = np.random.uniform(size=[n, n])
        q = np.linalg.qr(q)[0]
        q = q[:n_templates, :n_elems].reshape(template_shape).astype(np.float32)

        q = (q - q.min()) / (q.max() - q.min())

        template_logits = tf.get_variable('templates', initializer=q)
        # prevent negative ink
        self._template_logits = template_logits
        self._templates = self._template_nonlin(template_logits)

        if self._use_alpha_channel:
          self._templates_alpha = tf.get_variable(
              'templates_alpha',
              shape=self._templates[Ellipsis, :1].shape,
              initializer=tf.zeros_initializer())

        self._n_templates = n_templates

    templates = self._templates
    if template_feature is not None:


      if self._colorize_templates:
        mlp = snt.BatchApply(snt.nets.MLP([32, self._n_channels]))
        template_color = mlp(template_feature)[:, :, tf.newaxis, tf.newaxis]

        if self._color_nonlin == math_ops.relu1:
          template_color += .99

        template_color = self._color_nonlin(template_color)
        templates = tf.identity(templates) * template_color

    return templates

  def _build(self,
             pose,
             presence=None,
             template_feature=None,
             bg_image=None,
             img_embedding=None):
    """Builds the module.

    Args:
      pose: [B, n_templates, 6] tensor.
      presence: [B, n_templates] tensor.
      template_feature: [B, n_templates, n_features] tensor; these features are
        used to change templates based on the input, if present.
      bg_image: [B, *output_size] tensor representing the background.
      img_embedding: [B, d] tensor containing image embeddings.

    Returns:
      [B, n_templates, *output_size, n_channels] tensor.
    """
    batch_size, n_templates = pose.shape[:2].as_list()
    templates = self.make_templates(n_templates, template_feature)

    if templates.shape[0] == 1:
      templates = snt.TileByDim([0], [batch_size])(templates)

    # it's easier for me to think in inverse coordinates
    warper = snt.AffineGridWarper(self._output_size, self._template_size)
    warper = warper.inverse()

    grid_coords = snt.BatchApply(warper)(pose)
    resampler = snt.BatchApply(contrib_resampler.resampler)
    transformed_templates = resampler(templates, grid_coords)

    if bg_image is not None:
      bg_image = tf.expand_dims(bg_image, axis=1)
    else:
      bg_image = tf.nn.sigmoid(tf.get_variable('bg_value', shape=[1]))
      bg_image = tf.zeros_like(transformed_templates[:, :1]) + bg_image

    transformed_templates = tf.concat([transformed_templates, bg_image], axis=1)

    if presence is not None:
      presence = tf.concat([presence, tf.ones([batch_size, 1])], axis=1)

    if True:  # pylint: disable=using-constant-test

      if self._use_alpha_channel:
        template_mixing_logits = snt.TileByDim([0], [batch_size])(
            self._templates_alpha)
        template_mixing_logits = resampler(template_mixing_logits, grid_coords)

        bg_mixing_logit = tf.nn.softplus(
            tf.get_variable('bg_mixing_logit', initializer=[0.]))

        bg_mixing_logit = (
            tf.zeros_like(template_mixing_logits[:, :1]) + bg_mixing_logit)

        template_mixing_logits = tf.concat(
            [template_mixing_logits, bg_mixing_logit], 1)

      else:
        temperature_logit = tf.get_variable('temperature_logit', shape=[1])
        temperature = tf.nn.softplus(temperature_logit + .5) + 1e-4
        template_mixing_logits = transformed_templates / temperature

    scale = 1.
    if self._learn_output_scale:
      scale = tf.get_variable('scale', shape=[1])
      scale = tf.nn.softplus(scale) + 1e-4

    if self._output_pdf_type == 'mixture':
      template_mixing_logits += make_brodcastable(
          math_ops.safe_log(presence), template_mixing_logits)

      rec_pdf = prob.MixtureDistribution(template_mixing_logits,
                                         [transformed_templates, scale],
                                         tfd.Normal)


    else:
      raise ValueError('Unknown pdf type: "{}".'.format(self._output_pdf_type))

    return AttrDict(
        raw_templates=tf.squeeze(self._templates, 0),
        transformed_templates=transformed_templates[:, :-1],
        mixing_logits=template_mixing_logits[:, :-1],
        pdf=rec_pdf)

