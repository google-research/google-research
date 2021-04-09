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

"""Masked language task with progressive training."""
# Lint as: python3
import copy
import os
from typing import List, Optional

from absl import logging
import dataclasses
import numpy
import orbit
import tensorflow as tf
import yaml

from grow_bert.lowcost.config import config_encoder as ecfg
from grow_bert.lowcost.models import bert_pretrain_model as small_pretrainer
from grow_bert.lowcost.models import pretrain_dataloader as small_dataloader
from grow_bert.lowcost.models import transformer_encoder as small_encoder_lib
from official.core import task_factory
from official.modeling import optimization
from official.modeling import tf_utils
from official.modeling.hyperparams import base_config
from official.modeling.hyperparams import config_definitions as cfg
from official.modeling.progressive import policies
from official.nlp.configs import bert
from official.nlp.configs import encoders
from official.nlp.data import pretrain_dataloader
from official.nlp.modeling import layers
from official.nlp.tasks import masked_lm


@dataclasses.dataclass
class MaskedLMConfig(cfg.TaskConfig):
  """The model config."""
  init_checkpoint: str = ''
  model: bert.PretrainerConfig = bert.PretrainerConfig(
      cls_heads=[
          bert.ClsHeadConfig(
              inner_dim=768,
              num_classes=2,
              dropout_rate=0.1,
              name='next_sentence')
      ],
      encoder=encoders.EncoderConfig(bert=encoders.BertEncoderConfig()))
  scale_loss: bool = False
  train_data: pretrain_dataloader.BertPretrainDataConfig = pretrain_dataloader.BertPretrainDataConfig(
  )
  small_train_data: pretrain_dataloader.BertPretrainDataConfig = pretrain_dataloader.BertPretrainDataConfig(
  )
  validation_data: pretrain_dataloader.BertPretrainDataConfig = pretrain_dataloader.BertPretrainDataConfig(
  )


@dataclasses.dataclass
class StackingStageConfig(base_config.Config):
  num_steps: int = 0
  warmup_steps: int = 10000
  initial_learning_rate: float = 1e-4
  end_learning_rate: float = 0.0
  decay_steps: int = 1000000
  override_num_layers: Optional[int] = None

  small_encoder_config: Optional[
      ecfg.SmallEncoderConfig] = ecfg.SmallEncoderConfig()
  override_train_data: Optional[
      pretrain_dataloader
      .BertPretrainDataConfig] = pretrain_dataloader.BertPretrainDataConfig()
  override_valid_data: Optional[
      pretrain_dataloader
      .BertPretrainDataConfig] = pretrain_dataloader.BertPretrainDataConfig()


@dataclasses.dataclass
class ProgStackingConfig(policies.ProgressiveConfig):
  stage_list: List[StackingStageConfig] = dataclasses.field(
      default_factory=lambda: [  # pylint: disable=g-long-lambda
          StackingStageConfig(
              num_steps=3000,
              warmup_steps=10000,
              initial_learning_rate=1e-4,
              end_learning_rate=1e-4,
              decay_steps=1000000),
          StackingStageConfig(
              num_steps=3000,
              warmup_steps=10000,
              initial_learning_rate=1e-4,
              end_learning_rate=1e-4,
              decay_steps=1000000)
      ])


@task_factory.register_task_cls(MaskedLMConfig)
class ProgressiveMaskedLM(policies.ProgressivePolicy, masked_lm.MaskedLMTask):
  """Mask language modeling with progressive policy."""

  def __init__(self,
               strategy,
               progressive_config,
               optimizer_config,
               train_data_config,
               small_train_data_config,
               task_config,
               logging_dir=None):
    """Initialize progressive training manager before the training loop starts.

    Arguments:
      strategy: A distribution strategy.
      progressive_config: ProgressiveConfig. Configuration for this class.
      optimizer_config: optimization_config.OptimizerConfig. Configuration for
        building the optimizer.
      train_data_config: config_definitions.DataConfig. Configuration for
        building the training dataset.
      task_config: TaskConfig. This is used in base_task.Task.
      logging_dir: a string pointing to where the model, summaries etc. will be
        saved. This is used in base_task.Task.
    """
    self._strategy = strategy
    self._progressive_config = progressive_config
    self._optimizer_config = optimizer_config
    self._train_data_config = train_data_config
    self._small_train_data_config = small_train_data_config
    self._model_config: bert.PretrainerConfig = task_config.model
    masked_lm.MaskedLMTask.__init__(
        self, params=task_config, logging_dir=logging_dir)
    policies.ProgressivePolicy.__init__(self)

  # Overrides policies.ProgressivePolicy
  def get_model(self, stage_id, old_model=None):
    """Build model for each stage."""
    stage_config: StackingStageConfig = self._progressive_config.stage_list[
        stage_id]
    if stage_config.small_encoder_config is not None:
      encoder_cfg: ecfg.TransformerEncoderConfig = ecfg.from_bert_encoder_config(
          self._model_config.encoder.bert, stage_config.small_encoder_config)
      model_cfg = copy.deepcopy(self._model_config)
      model_cfg.encoder = encoders.EncoderConfig(bert=encoder_cfg)
      model = self.build_small_model(model_cfg.as_dict())
    else:
      model_config = copy.deepcopy(self._model_config)
      if stage_config.override_num_layers is not None:
        model_config.encoder.bert.num_layers = stage_config.override_num_layers
      model = self.build_model(model_config)
      _ = model(model.inputs)

    if stage_id == 0:
      self.initialize(model)

    if stage_id > 0 and old_model is not None:
      logging.info('Stage %d copying weights.', stage_id)
      self.transform_model(small_model=old_model, model=model)
    return model

  # overrides policies.ProgressivePolicy
  def get_train_dataset(self, stage_id):
    stage_config = self._progressive_config.stage_list[stage_id]
    if stage_config.small_encoder_config is not None:
      train_data_config = self._small_train_data_config
      if stage_config.override_train_data is not None:
        logging.info('stage %d: override small train data to %s', stage_id,
                     stage_config.override_train_data)
        train_data_config = stage_config.override_train_data
      return orbit.utils.make_distributed_dataset(self._strategy,
                                                  self.build_small_inputs,
                                                  train_data_config)
    train_data_config = self._train_data_config
    if stage_config.override_train_data is not None:
      train_data_config = stage_config.override_train_data
      logging.info('stage %d: override full train data to %s', stage_id,
                   stage_config.override_train_data)
    return orbit.utils.make_distributed_dataset(self._strategy,
                                                self.build_inputs,
                                                train_data_config)

  # overrides policies.ProgressivePolicy
  def get_eval_dataset(self, stage_id):
    build_func = self.build_inputs
    stage_config = self._progressive_config.stage_list[stage_id]
    if stage_config.small_encoder_config is not None:
      build_func = self.build_small_inputs

    valid_data_config = self.task_config.validation_data
    if stage_config.override_valid_data is not None:
      valid_data_config = stage_config.override_valid_data
      logging.info('stage %d: override full valid data to %s', stage_id,
                   stage_config.override_valid_data)
    return orbit.utils.make_distributed_dataset(self._strategy, build_func,
                                                valid_data_config)

  def build_small_inputs(self, params, input_context=None):
    """Returns tf.data.Dataset for pretraining."""
    return small_dataloader.PretrainDataLoader(params).load(input_context)

  # Overrides policies.ProgressivePolicy
  def num_stages(self):
    return len(self._progressive_config.stage_list)

  # Overrides policies.ProgressivePolicy
  def num_steps(self, stage_id):
    return self._progressive_config.stage_list[stage_id].num_steps

  # Overrides policies.ProgressivePolicy
  def get_optimizer(self, stage_id):
    """Build optimizer for each stage."""
    params = self._optimizer_config.replace(
        learning_rate={
            'polynomial': {
                'decay_steps':
                    self._progressive_config.stage_list[stage_id].decay_steps,
                'initial_learning_rate':
                    self._progressive_config.stage_list[stage_id]
                    .initial_learning_rate,
                'end_learning_rate':
                    self._progressive_config.stage_list[stage_id]
                    .end_learning_rate,
            }
        },
        warmup={
            'polynomial': {
                'warmup_steps':
                    self._progressive_config.stage_list[stage_id].warmup_steps,
            }
        })
    opt_factory = optimization.OptimizerFactory(params)
    optimizer = opt_factory.build_optimizer(opt_factory.build_learning_rate())
    return optimizer

  def build_small_model(self, model_cfg):
    encoder_cfg = model_cfg['encoder']['bert']
    dataconf = self.task_config.train_data
    encoder_network = small_encoder_lib.TransformerEncoder(
        vocab_size=encoder_cfg['vocab_size'],
        hidden_size=encoder_cfg['hidden_size'],
        num_layers=encoder_cfg['num_layers'],
        num_attention_heads=encoder_cfg['num_attention_heads'],
        intermediate_size=encoder_cfg['intermediate_size'],
        activation=tf_utils.get_activation(encoder_cfg['hidden_activation']),
        dropout_rate=encoder_cfg['dropout_rate'],
        attention_dropout_rate=encoder_cfg['attention_dropout_rate'],
        max_sequence_length=encoder_cfg['max_position_embeddings'],
        type_vocab_size=encoder_cfg['type_vocab_size'],
        initializer=tf.keras.initializers.TruncatedNormal(
            stddev=encoder_cfg['initializer_range']),
        net2net_ratio=encoder_cfg['net2net_ratio'],
        net2net_layers=encoder_cfg['net2net_layers'],
        lightatt_layers=encoder_cfg['lightatt_layers'],
        input_pool_name=encoder_cfg['input_pool_name'],
        input_pool_size=encoder_cfg['input_pool_size'])
    sequence_length = dataconf.seq_length
    predict_length = dataconf.max_predictions_per_seq
    dummy_inputs = dict(
        input_mask=tf.zeros((1, sequence_length), dtype=tf.int32),
        input_positions=tf.zeros((1, sequence_length), dtype=tf.int32),
        input_type_ids=tf.zeros((1, sequence_length), dtype=tf.int32),
        input_word_ids=tf.zeros((1, sequence_length), dtype=tf.int32),
        masked_lm_positions=tf.zeros((1, predict_length), dtype=tf.int32),
        masked_input_ids=tf.zeros((1, predict_length), dtype=tf.int32),
        masked_segment_ids=tf.zeros((1, predict_length), dtype=tf.int32),
        masked_lm_weights=tf.zeros((1, predict_length), dtype=tf.float32))
    _ = encoder_network(dummy_inputs)

    if 'cls_heads' in model_cfg:
      classification_heads = [
          layers.ClassificationHead(**cfg) for cfg in model_cfg['cls_heads']
      ]
    else:
      classification_heads = []
    model = small_pretrainer.BertPretrainModel(
        mlm_initializer=tf.keras.initializers.TruncatedNormal(
            stddev=encoder_cfg['initializer_range']),
        mlm_activation=tf_utils.get_activation(
            encoder_cfg['hidden_activation']),
        encoder_network=encoder_network,
        classification_heads=classification_heads)
    _ = model(dummy_inputs)
    return model

  @staticmethod
  def transform_model(small_model, model):
    # copy variable weights
    # pylint: disable=protected-access
    encoder = model.encoder_network
    small_encoder = small_model.encoder_network
    model_embed_width = encoder.get_config()['embedding_width']
    model_hidden_size = encoder.get_config()['hidden_size']
    if model_embed_width is not None and model_embed_width != model_hidden_size:
      encoder._embedding_projection.set_weights(
          small_encoder._embedding_projection_layer.get_weights())
    encoder._embedding_layer.set_weights(
        small_encoder._embedding_layer.get_weights())
    encoder._type_embedding_layer.set_weights(
        small_encoder._type_embedding_layer.get_weights())
    encoder._position_embedding_layer.set_weights(
        small_encoder._position_embedding_layer.get_weights())
    encoder._embedding_norm_layer.set_weights(
        small_encoder._embedding_norm_layer.get_weights())
    encoder._pooler_layer.set_weights(small_encoder._pooler_layer.get_weights())

    model.masked_lm.bias.assign(small_model.masked_lm.bias)
    model.masked_lm.dense.set_weights(small_model.masked_lm.dense.get_weights())
    model.masked_lm.layer_norm.set_weights(
        small_model.masked_lm.layer_norm.get_weights())

    for i, cls_head in enumerate(model.classification_heads):
      cls_head.set_weights(small_model.classification_heads[i].get_weights())

    small_layers = small_encoder.transformer_layers
    small_num_layers = len(small_layers)
    num_layers = len(encoder.transformer_layers)
    logging.info('num_layers: %d, num_small_layers: %d', num_layers,
                 small_num_layers)

    if small_num_layers != num_layers:
      for i, layer in enumerate(encoder.transformer_layers):
        small_idx = i % small_num_layers
        logging.info('stack: %d -> %d', i, small_idx)
        small_layer = small_layers[small_idx]
        layer.set_weights(small_layer.get_weights())
    else:
      for i, layer in enumerate(encoder.transformer_layers):
        logging.info('!!! recover layer %d', i)
        small_layer = small_layers[i]

        # init attention layer
        attention_layer = layer._attention_layer
        small_attention_layer = small_layer._attention_layer
        attention_layer._value_dense.set_weights(
            small_attention_layer._value_dense.get_weights())
        attention_layer._output_dense.set_weights(
            small_attention_layer._output_dense.get_weights())
        if hasattr(
            small_layer, 'use_lightatt') and small_layer.use_lightatt and (
                not hasattr(layer, 'use_lightatt') or not layer.use_lightatt):
          logging.info('!!! recover lightatt')
          attention_layer._key_dense.set_weights(encoder.transformer_layers[
              i - 1]._attention_layer._key_dense.get_weights())
          attention_layer._query_dense.set_weights(encoder.transformer_layers[
              i - 1]._attention_layer._query_dense.get_weights())
        else:
          attention_layer._key_dense.set_weights(
              small_attention_layer._key_dense.get_weights())
          attention_layer._query_dense.set_weights(
              small_attention_layer._query_dense.get_weights())

        if hasattr(small_layer,
                   'net2net_ratio') and small_layer.net2net_ratio is not None:
          if hasattr(layer,
                     'net2net_ratio') and layer.net2net_ratio is not None:
            layer._output_dense_small.set_weights(
                small_layer._output_dense_small.get_weights())
            layer._intermediate_dense_small.set_weights(
                small_layer._intermediate_dense_small.get_weights())
          else:
            k = int(1 // small_layer.net2net_ratio)
            logging.info('!!! recover net2net %d', k)
            output_kernel, output_bias = layer._output_dense.get_weights()
            interm_kernel, interm_bias = layer._intermediate_dense.get_weights()
            output_small_kernel, output_small_bias = small_layer._output_dense_small.get_weights(
            )
            interm_small_kernel, interm_small_bias = small_layer._intermediate_dense_small.get_weights(
            )

            # check size
            small_interm_size = interm_small_kernel.shape[1]
            assert interm_kernel.shape[0] == output_kernel.shape[
                1] == output_bias.shape[0] == model_hidden_size
            error_message = (
                f'interm_kernel.shape1={interm_kernel.shape[1]}, '
                f'output_kernel.shape[0]={output_kernel.shape[0]}, '
                f'small_interm_size={small_interm_size}, k={k}')
            assert interm_kernel.shape[1] == output_kernel.shape[
                0] == interm_bias.shape[
                    0] == small_interm_size * k, error_message

            # restore
            new_output_bias = output_small_bias
            new_interm_bias = numpy.tile(interm_small_bias, k)
            new_interm_kernel = numpy.tile(interm_small_kernel, [1, k])
            new_output_kernel = numpy.tile(output_small_kernel, [k, 1]) / k
            layer._output_dense.set_weights(
                [new_output_kernel, new_output_bias])
            layer._intermediate_dense.set_weights(
                [new_interm_kernel, new_interm_bias])
        else:
          layer._output_dense.set_weights(
              small_layer._output_dense.get_weights())
          layer._intermediate_dense.set_weights(
              small_layer._intermediate_dense.get_weights())

        layer._output_layer_norm.set_weights(
            small_layer._output_layer_norm.get_weights())
        layer._attention_layer_norm.set_weights(
            small_layer._attention_layer_norm.get_weights())
      # pylint: enable=protected-access

  def initialize(self, model):
    init_dir_or_path = self.task_config.init_checkpoint
    logging.info('init dir_or_path: %s', init_dir_or_path)
    if not init_dir_or_path:
      return

    if tf.io.gfile.isdir(init_dir_or_path):
      init_dir = init_dir_or_path
      init_path = tf.train.latest_checkpoint(init_dir_or_path)
    else:
      init_path = init_dir_or_path
      init_dir = os.path.dirname(init_path)

    logging.info('init dir: %s', init_dir)
    logging.info('init path: %s', init_path)

    # restore from small model
    init_yaml_path = os.path.join(init_dir, 'params.yaml')
    if not tf.io.gfile.exists(init_yaml_path):
      init_yaml_path = os.path.join(os.path.dirname(init_dir), 'params.yaml')
    with tf.io.gfile.GFile(init_yaml_path, 'r') as rf:
      init_yaml_config = yaml.safe_load(rf)
    init_model_config = init_yaml_config['task']['model']
    if 'progressive' in init_yaml_config['trainer']:
      stage_list = init_yaml_config['trainer']['progressive']['stage_list']
      if stage_list:
        small_encoder_config = stage_list[-1]['small_encoder_config']
        if small_encoder_config is not None:
          small_encoder_config = ecfg.from_bert_encoder_config(
              init_model_config['encoder']['bert'], small_encoder_config)
          init_model_config['encoder']['bert'] = small_encoder_config.as_dict()

    # check if model size matches
    assert init_model_config['encoder']['bert'][
        'hidden_size'] == model.encoder_network.get_config()['hidden_size']

    # build small model
    small_model = self.build_small_model(init_model_config)
    ckpt = tf.train.Checkpoint(model=small_model)
    ckpt.restore(init_path).assert_existing_objects_matched()

    self.transform_model(small_model, model)
