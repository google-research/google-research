# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Implement basic model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import sonnet as snt
from tensor2tensor.layers import common_layers
import tensorflow as tf
from qanet import optimizers
from qanet import squad_data
from qanet import squad_helper
from qanet.util import configurable
from qanet.util import misc_util
from tensorflow.contrib import learn as contrib_learn


MODE_KEYS = tf.estimator.ModeKeys


class Module(snt.AbstractModule, configurable.Configurable):
  """A configurable and templated neural net module.

  snt.AbstractModule remaps __call__ to tf.make_template(_build), allowing a
  single instantiated object to be used in many places to represent the same set
  of weights.
  """

  def __init__(self, mode, config=None, name=None, dataset=None):
    snt.AbstractModule.__init__(self, name=name or self.__class__.__name__)
    config = config or {}
    self.config = self.build_config(**config)
    self.mode = mode
    assert mode in [MODE_KEYS.TRAIN, MODE_KEYS.EVAL, MODE_KEYS.PREDICT]
    self.is_training = mode == MODE_KEYS.TRAIN
    self._dataset = dataset

    self.data_format = 'squad'
    if 'data_format' in config:
      self.data_format = config['data_format']

    self.loss_option = 'squad'
    if 'loss_option' in config:
      self.loss_option = config['loss_option']

  def init_submodule(self, subconfig, name=None):
    cls = configurable.Configurable.load(subconfig)
    return cls(
        mode=self.mode, config=subconfig, name=name, dataset=self._dataset)


def _update_partition_info(variable_map, nograd_var):
  """Update variable map with partition info, needed for translation ELMO."""
  enc_emb_vars = []
  dec_emb_vars = []
  enc_emb_keys = []
  dec_emb_keys = []
  if not nograd_var:  # every var is trainable
    for key in variable_map:
      tokens = key.split('/')
      if len(tokens) > 1:
        if 'part_' in tokens[-2]:
          if 'encoder' in tokens[-3]:
            enc_emb = '/'.join(tokens[:-2] + [tokens[-1]])
            enc_emb_keys.append(key)
            enc_emb_vars.append(variable_map[key])
          elif 'decoder' in tokens[-3]:
            dec_emb = '/'.join(tokens[:-2] + [tokens[-1]])
            dec_emb_keys.append(key)
            dec_emb_vars.append(variable_map[key])
  else:
    for key in variable_map:
      tokens = key.split('/')
      if 'part_' in tokens[-1]:
        if 'encoder' in tokens[-2]:
          enc_emb = '/'.join(tokens[:-1])
          enc_emb_keys.append(key)
          enc_emb_vars.append(variable_map[key])
        elif 'decoder' in tokens[-2]:
          dec_emb = '/'.join(tokens[:-1])
          dec_emb_keys.append(key)
          dec_emb_vars.append(variable_map[key])

  for key in enc_emb_keys + dec_emb_keys:
    del variable_map[key]

  variable_map[enc_emb] = enc_emb_vars
  variable_map[dec_emb] = dec_emb_vars


class Model(Module):
  """Base configurable type for all models. Usable from tf.learn interface."""

  @staticmethod
  def _config():
    return {
        'optimizer': optimizers.AdamOptimizer,
        'initializer': optimizers.XavierInit,
        'train_with_multi_answer_loss': False,
        'data_format': 'squad',
        'loss_option': 'squad',
        'init_checkpoint': '',  # Directory with checkpoint we restore from
    }

  @classmethod
  def get_model_fn(cls, train_steps, dataset, model_dir, use_estimator=False):

    def model_fn(features, labels, mode, params):
      """Applies the model to the input features to produce predictions."""
      if 'initializer' in params:
        initializer = configurable.Configurable.initialize(
            params['initializer'])
        tf.logging.info('Using %s initializer', params['initializer'])
        tf.get_variable_scope().set_initializer(initializer)
      else:
        tf.logging.info(
            'Not setting a global initializer. TF defaults to Xavier.')

      model_instance = cls(mode=mode, config=params, dataset=dataset)
      predictions = model_instance(features)  # pylint: disable=not-callable

      if labels:
        loss = model_instance.loss(
            predictions=predictions,
            targets=labels,
            multi_answer_loss=params['train_with_multi_answer_loss'])
      else:
        assert mode == MODE_KEYS.PREDICT
        loss = None

      # Always instantiate optimizer to (exponential moving averages at eval)
      optimizer = configurable.Configurable.load(params['optimizer'])
      optimizer_instance = optimizer(config=params['optimizer'])
      if mode == MODE_KEYS.TRAIN:
        train_op = optimizer_instance(loss, train_steps)
      else:
        train_op = None

      # Initialization
      if params['init_checkpoint']:  # Checkpoint is from a different model
        latest_checkpoint = tf.train.latest_checkpoint(model_dir)
        # only init if there's no saved checkpoint.
        if not latest_checkpoint:
          misc_util.init_from_checkpoint(
              params['init_checkpoint'], params['fn'])
        else:
          tf.logging.info('Latest checkpoint %s exists. No init from %s.' % (
              latest_checkpoint, params['init_checkpoint']))

      tf.logging.info('mode: %s' % mode)
      tf.logging.info('params: %s' % params)
      if params['optimizer']['ema_decay'] != 1.0 and mode != MODE_KEYS.TRAIN:
        ema = optimizer_instance.exponential_moving_average
        trainable_vars, _, has_partition = misc_util.get_trainable_vars(
            exclude_pattern=params['optimizer']['nograd_var'])

        # Restored variables
        variable_map = ema.variables_to_restore(trainable_vars)
        if has_partition:  # Update partition info
          _update_partition_info(
              variable_map, params['optimizer']['nograd_var'])
        saver = tf.train.Saver(variable_map)

        scaffold = tf.train.Scaffold(saver=saver)
      else:
        scaffold = None

      # Eval metrics
      eval_metric_ops = None
      if mode in [MODE_KEYS.TRAIN, MODE_KEYS.EVAL]:
        eval_metric_ops = model_instance.metrics(
            predictions=predictions, targets=labels)

      if use_estimator:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops,
            scaffold=scaffold)
      else:
        # Maintain backwards compatibility
        return contrib_learn.ModelFnOps(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops,
            scaffold=scaffold)

    return model_fn

  def metrics(self, predictions, targets):
    return None

  def _build(self, features):
    raise NotImplementedError

  def loss(self, predictions, targets, multi_answer_loss):
    raise NotImplementedError


class BaseQANetModel(Model):
  """Base QANet model for SQuAD v1.1. Implements loss and evaluation metrics."""

  def metrics(self, predictions, targets):
    """Metrics specific to this model (specifically F1, exact match)."""
    return squad_data.get_eval_metric_ops(
        targets=targets, predictions=predictions)

  def loss(self, predictions, targets, multi_answer_loss):
    """Wrap a common loss function."""
    losses = squad_helper.get_loss(
        targets['word_answer_starts'], targets['word_answer_ends'],
        targets['num_answers'],
        predictions['logits_start'], predictions['logits_end'],
        predictions['no_answer_bias'],
        multi_answer_loss=multi_answer_loss)
    return tf.reduce_mean(losses)

  def _build_embed(self, features, training):
    """Build embedding model and return document/question representations.

    Args:
      features: dict; input data.
      training: bool; whether we are under training mode.

    Returns:
      x: context Tensor of shape [batch, max_context_length, dim].
      q: question Tensor of shape [batch, max_question_length, dim].
    """
    config = self.config
    tf.logging.info('\n\n\n\n\n')
    tf.logging.info(config)
    tf.logging.info('\n\n\n\n\n')
    embedder = self.init_submodule(config['embedding'])
    xc, qc, xw, qw, _, _, _, _ = embedder(features)
    emb = embedder(features)
    # Concatenate word and character embeddings
    x = tf.concat([emb['xc'], emb['xw']], 2)
    q = tf.concat([emb['qc'], emb['qw']], 2)
    x = squad_helper.dropout_wrapper(x, config['input_keep_prob'], self.mode)
    q = squad_helper.dropout_wrapper(q, config['input_keep_prob'], self.mode)
    x = squad_helper.highway_layer(
        x, num_layers=2, activation='relu', training=training,
        dropout_rate=1-config['hw_keep_prob'])
    q = squad_helper.highway_layer(
        q, num_layers=2, activation='relu', training=training,
        dropout_rate=1-config['hw_keep_prob'], reuse=True)

    return x, q

  def _build_embed_encoder_and_attention(self, x, q,
                                         context_lengths, question_lengths):
    """Build modeling encoder and return start/end logits.

    Args:
      x: context Tensor of shape [batch, max_context_length, dim].
      q: question Tensor of shape [batch, max_question_length, dim].
      context_lengths: length Tensor of shape [batch].
      question_lengths: length Tensor of shape [batch].

    Returns:
      x_final: output Tensor of shape [batch, max_context_length, dim].
    """
    input_keep_prob = self.config['input_keep_prob']

    # Embedding encoder
    encoder_emb = self.init_submodule(
        self.config['encoder_emb'], name='xq_encoder')
    x = encoder_emb(x, context_lengths)['outputs']
    q = encoder_emb(q, question_lengths)['outputs']
    x = squad_helper.dropout_wrapper(x, input_keep_prob, self.mode)
    q = squad_helper.dropout_wrapper(q, input_keep_prob, self.mode)

    # Context-Question Attention
    with tf.variable_scope('attention'):
      xq, qx = squad_helper.bi_attention_memory_efficient_dcn(
          a=q,
          b=x,
          mask_a=question_lengths,
          mask_b=context_lengths)

    x_final = tf.concat([x, xq, x * xq, x * qx], 2)
    return x_final

  def _build_model_encoder(self, x, context_lengths, gpu_id=0, num_gpus=1):
    """Build modeling encoder and return start/end logits.

    Args:
      x: input Tensor of shape [batch, max_length, dim].
      context_lengths: length Tensor of shape [batch].
      gpu_id: start GPU id.
      num_gpus: number of GPUs available.

    Returns:
      logits_start: Tensor of shape [batch, max_length].
      logits_end: Tensor of shape [batch, max_length].
      modeling_layers: a list of modeling layers, from bottom to top,
        each has shape [batch, max_length, dim].
    """
    output_keep_prob = self.config['output_keep_prob']

    with tf.device(misc_util.get_device_str(gpu_id, num_gpus)):
      encoder_model = self.init_submodule(
          self.config['encoder_model'], name='encoder_model')
      x0 = encoder_model(x, context_lengths)['outputs']
      x0 = squad_helper.dropout_wrapper(x0, output_keep_prob, self.mode)
      x1 = encoder_model(x0, context_lengths)['outputs']
      x1 = squad_helper.dropout_wrapper(x1, output_keep_prob, self.mode)

    gpu_id += 1
    with tf.device(misc_util.get_device_str(gpu_id, num_gpus)):
      x2 = encoder_model(x1, context_lengths)['outputs']
      x2 = squad_helper.dropout_wrapper(x2, output_keep_prob, self.mode)
      x3 = encoder_model(x2, context_lengths)['outputs']
      x3 = squad_helper.dropout_wrapper(x3, output_keep_prob, self.mode)

      logits_start = squad_helper.exp_mask(
          tf.squeeze(
              tf.layers.dense(tf.concat([x1, x2], 2), 1, name='logits1'), 2),
          context_lengths)
      logits_end = squad_helper.exp_mask(
          tf.squeeze(
              tf.layers.dense(tf.concat([x1, x3], 2), 1, name='logits2'), 2),
          context_lengths)
    modeling_layers = [x0, x1, x2, x3]

    return logits_start, logits_end, modeling_layers

  def _build_no_answer_bias(self, modeling_layers, context_lengths):
    """Compute no-answer bias.

    Args:
      modeling_layers: a list of model encoding layers, from bottom to top.
      context_lengths: a Tensor of shape [batch_size].

    Returns:
      no_answer_bias: a Tensor of shape [batch_size].
    """
    del modeling_layers  # not use in the base model

    no_answer_bias = tf.get_variable('no_answer_bias', shape=[], dtype='float')
    no_answer_bias = tf.tile(
        tf.reshape(no_answer_bias, [1, 1]),
        [tf.shape(context_lengths)[0], 1])
    return no_answer_bias

  def _build_predictions(self, features, logits_start, logits_end,
                         no_answer_bias):
    """Build predictions dict."""
    return squad_helper.get_pred_ops(
        features,
        self.config['max_answer_size'],
        logits_start,
        logits_end,
        no_answer_bias,
        get_answer_op=self._dataset.get_answer_op)

  def _build(self, features):
    training = self.mode == tf.estimator.ModeKeys.TRAIN
    base_gpu = 0
    num_gpus = 4
    context_lengths = features['context_num_words']

    # Deep Embedding
    gpu_id = base_gpu
    with tf.device(misc_util.get_device_str(gpu_id, num_gpus)):
      x, q = self._build_embed(features, training)

    # Embed Encoder & Attention
    gpu_id += 1
    with tf.device(misc_util.get_device_str(gpu_id, num_gpus)):
      x_final = self._build_embed_encoder_and_attention(
          x, q, context_lengths, features['question_num_words'])

    # Modeling encoder
    gpu_id += 1
    logits_start, logits_end, modeling_layers = self._build_model_encoder(
        x_final, context_lengths,
        gpu_id=gpu_id, num_gpus=num_gpus)

    # Predict no_answer_bias
    no_answer_bias = self._build_no_answer_bias(
        modeling_layers, context_lengths)

    # Predictions
    predictions = self._build_predictions(features, logits_start, logits_end,
                                          no_answer_bias)

    misc_util.print_vars(label='All variables')
    return predictions


class QANetEncoder(Module):
  """
  Encoder class for QANet.

  Each building block is stack of
            (convolution * n + self-attention * m + feedforward)
  inside residuals.

  Each encoder is a stack of such building blocks
  """

  @staticmethod
  def _config():
    return {
        'layers': 2,  # TODO(thangluong): rename num_blocks
        'layers_conv': 2,  # TODO(thangluong): rename num_convs (per block)
        'structure': '',  # if not empty, ignore 'layers_conv'
        'layer_names': '',  # if not empty, ignore 'layers' & 'layers_conv'
        'hidden_size': 128,
        'attention_heads': 8,
        'attention_dropout': 0.0,
        'layer_output_dropout': 0.2,
        'kernel_size': 3,
        'ffn_hs_factor': 4,
        'activation': 'relu',
        'preactivate': True,
        'attention_type': 'dot_product',

        # Path dropout/stochastic depth
        'survival_rate_last_layer': 0.5,
        'survival_schedule': 'linear',  # linear or exp
    }

  def _build(self, inputs, length):
    config = copy.deepcopy(self.config)  # Make mutable copy

    # Turn off dropout at test time.
    if not self.is_training:
      for k in config:
        if 'dropout' in k:
          config[k] = 0.0

    net = squad_helper.preprocess_inputs(inputs, config['hidden_size'])
    if config['layer_names']:  # full encoder structure
      layer_names = config['layer_names'].split(',')
    elif config['structure']:  # structure of one block
      layer_names = config['structure'].split(',') * config['layers']
    else:
      layer_names = ['conv'] * config['layers_conv']
      layer_names.extend(['att', 'ffn'])
      layer_names *= config['layers']

    # Building layers
    n_layer = len(layer_names)
    survival_rate = None
    layer_input = net
    layer_type = None

    for layer_id, layer_type in enumerate(layer_names):
      var_scope = '%s_%d' % (layer_type, layer_id)
      with tf.variable_scope(var_scope):
        layer_input = net
        with tf.variable_scope('layer', reuse=tf.AUTO_REUSE):
          net_new = squad_helper.build_a_layer(
              layer_input, layer_type, self.is_training, config)

        if config['survival_rate_last_layer'] < 1.0:  # stochastic depth
          survival_rate = squad_helper.compute_survival_rate(
              layer_id, n_layer, config['survival_rate_last_layer'],
              config['survival_schedule'])
          net = squad_helper.stochastic_depth(
              net_new, layer_input, survival_rate, self.is_training)
        else:  # no stochastic depth, only residual connection
          net = net_new + layer_input

    net = common_layers.layer_norm(net)

    # Place into dictionary so it is usable in same places as an RNN layer
    return {'outputs': net}
