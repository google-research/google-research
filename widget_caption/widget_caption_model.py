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

"""Widget captioning model."""
import collections
import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf

from widget_caption import widget_caption_config
from widget_caption import widget_caption_eval
from widget_caption import widget_caption_input as input_utils
from tensorflow_models.official.nlp.modeling import layers
from tensorflow_models.official.nlp.modeling import ops
from tensorflow_models.official.nlp.transformer import model_params
from tensorflow_models.official.nlp.transformer import model_utils
from tensorflow_models.official.nlp.transformer import optimizer
from tensorflow_models.official.nlp.transformer import transformer as nlp_transformer

flags.DEFINE_string('experiment', 'debug',
                    'Experiment name defined in widget_caption_config.py.')

flags.DEFINE_string('model_dir', None, 'Model dir')

flags.DEFINE_string('ckpt_filepath', None,
                    'Checkpoint path for saving weights of every epoch.')

FLAGS = flags.FLAGS


def create_hparams(experiment):
  """Creates the hyper parameters."""
  hparams = {}

  # General parameters.
  hparams['batch_size'] = 64
  hparams['eval_batch_size'] = 64
  hparams['learning_rate_warmup_steps'] = 2000
  hparams['learning_rate_constant'] = 1
  hparams['learning_rate'] = 0.001
  hparams['train_epoches'] = 20
  hparams['steps_per_epoch'] = 30
  hparams['train_steps'] = 100 * 1000
  hparams['eval_steps'] = 100
  hparams['caption_optimizer'] = 't2t'
  hparams['clip_norm'] = 5.0
  hparams['widget_encoder_checkpoint'] = ''
  hparams['train_files'] = ''
  hparams['eval_files'] = ''
  hparams['train_buffer_size'] = 2000
  hparams['eval_buffer_size'] = 500
  hparams['train_pixel_encoder'] = True
  hparams['debug'] = False
  hparams['distribution_strategy'] = 'mirrored'

  # Train model using decoding task, classification task, or both.
  hparams['decoding_task'] = True
  hparams['classification_task'] = False
  # Whether to use decoding for phrase classification: <START> phrase_id <EOS>.
  hparams['use_decoding_for_classification'] = False

  # Weight for the classification loss.
  hparams['classification_loss_weight'] = 1
  hparams['train_with_one_node'] = False

  # Embedding parameters.
  hparams['embedding_file'] = ''
  hparams['word_vocab_path'] = ''
  hparams['glove_trainable'] = True
  hparams['vocab_size'] = 10000
  hparams['phrase_vocab_size'] = 10000

  # View hierarchy encoder parameters.
  hparams['max_pixel_pos'] = 100
  hparams['max_dom_pos'] = 500
  hparams['screen_encoder'] = 'gcn'
  hparams['screen_embedding_feature'] = ['text', 'type', 'pos', 'click', 'dom']
  hparams['obj_text_aggregation'] = 'max'
  hparams['synthetic_screen_noise'] = 0.
  # Whether to add pixel encoding as input to view hierarchy encoder.
  hparams['encode_screen_with_context'] = False
  # Whether to add a residual link for pixel encoding.
  hparams['add_pixel_skip_link'] = False

  # General parameters.
  hparams['num_hidden_layers'] = 2
  hparams['hidden_size'] = 2
  hparams['filter_size'] = 2
  hparams['num_heads'] = 2
  hparams['dropout'] = 0.2
  hparams['layer_prepostprocess_dropout'] = 0.2
  hparams['attention_dropout'] = 0.2
  hparams['relu_dropout'] = 0.2

  transformer_hparams = model_params.BASE_PARAMS

  # Add parameters from transformer model.
  hparams.update(transformer_hparams)

  # Rewrite all the parameters from command-line flags.
  config = widget_caption_config.experiments[experiment]
  hparams.update(config)

  return hparams


def load_embed(file_name, vocab_size):
  """Loads a pre-trained embedding matrix.

  Args:
    file_name: the file name of the embedding file.
    vocab_size: if > 0, only load embedding weights for vocab_size words.

  Returns:
    vocab: a list of tokens.
    embeds: a numpy array of embeddings for each token plus an OOV embedding.
    depth: the depth of the embedding.
  Raises:
    ValueError: embeddings have different depths.
  """

  with tf.io.gfile.GFile(file_name, 'r') as embed_file:
    vocab = []
    embeds = []
    depth = -1
    for index, line in enumerate(embed_file):
      if vocab_size > 0 and index >= vocab_size:
        break
      line = line.strip()
      tokens = line.strip().split(' ')
      word = tokens[0]
      vocab.append(word)
      if depth == -1:
        embed = [float(token) for token in tokens[1:]]
      else:
        embed = [float(token) for token in tokens[-depth:]]
      d = len(embed)
      if depth == -1:
        depth = d
      if d != depth:
        raise ValueError('Inconsistent embedding sizes')
      embeds.append(embed)

    embeds = np.stack(embeds)

  return vocab, embeds, depth


def compute_score(predictions, references, vocab=None):
  """Computes the bleu score.

  Args:
    predictions: a numpy arrary in the shape of [batch_size, max_phrase_length]
    references: a numpy array in the shape of [batch_size, 7, 10]
    vocab: the vocabulary file.

  Returns:
    a scalar value for the corpus level bleu score.
  """
  assert np.rank(predictions) == 2
  assert predictions.shape[0] == references.shape[0]
  batch_size = predictions.shape[0]
  predictions = tf.make_ndarray(tf.make_tensor_proto(predictions)).tolist()
  references = tf.make_ndarray(tf.make_tensor_proto(references)).tolist()
  hypotheses_list = []
  references_list = []
  for index in range(batch_size):
    h = predictions[index]
    try:
      eos_index = h.index(input_utils.EOS)
    except ValueError:
      eos_index = len(h)
    hypotheses_list.append(h[:eos_index])

    ref = references[index].decode().split('|')
    ref_list = [r.strip().split(' ') for r in ref if r.strip()]
    references_list.append(ref_list)

  all_scores = collections.defaultdict(list)
  for hypothesis, references in zip(hypotheses_list, references_list):
    if vocab is not None and len(vocab):
      # Skip PADDING, UNK, EOS, START (0-3).
      hypothesis = [
          vocab[word_id].numpy().decode()
          for word_id in hypothesis
          if word_id > 3
      ]
      logging.info('hypothesis: %s', str(hypothesis))
      logging.info('references: %s', str(references))

    h_str = ' '.join(str(e) for e in hypothesis)
    r_str = [' '.join(str(e) for e in ref) for ref in references]

    scores = widget_caption_eval.coco_evaluate(r_str, h_str)
    for key, score in scores.items():
      all_scores[key].append(score)

  score_names = [
      'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'ROUGE-1-f1-mean',
      'ROUGE-1-f1-min', 'ROUGE-1-f1-max', 'ROUGE-2-f1-mean', 'ROUGE-2-f1-min',
      'ROUGE-2-f1-max', 'ROUGE-L-f1-mean', 'ROUGE-L-f1-min', 'ROUGE-L-f1-max'
  ]
  return [np.array(all_scores[name], dtype=np.float32) for name in score_names]


class EmbeddingLayer(tf.keras.layers.Layer):
  """Embedding layer."""

  def __init__(self,
               name,
               vocab_size,
               embedding_dim,
               embedding_file=None,
               hidden_dim=None,
               trainable=True):
    super(EmbeddingLayer, self).__init__(name=name)
    self._vocab_size = vocab_size
    self._hidden_dim = hidden_dim
    self._embedding_dim = embedding_dim
    self._embedding_file = embedding_file
    self._trainable = trainable

  def build(self, input_shape):
    if self._embedding_file:
      logging.info('Load embedding file for %s of vocab size %s: %s',
                   self._name, self._vocab_size, self._embedding_file)
      _, embedding_weights, depth = load_embed(
          file_name=self._embedding_file, vocab_size=self._vocab_size)
      self._embedding_dim = depth
      initializer = tf.constant_initializer(
          embedding_weights[:self._vocab_size, :])
    else:
      logging.info('Create random embedding matrix for %s of size %s',
                   self._name, self._vocab_size)
      initializer = tf.keras.initializers.RandomNormal(
          mean=0.0, stddev=0.1, seed=None)

    self.embeddings = self.add_weight(
        name='{}_weights'.format(self._name),
        shape=(self._vocab_size, self._embedding_dim),
        initializer=initializer,
        trainable=self._trainable,
        dtype='float32')

    if self._hidden_dim:
      self._project_layer = tf.keras.layers.Dense(self._hidden_dim)

  def call(self, inputs):
    embeddings = tf.nn.embedding_lookup(self.embeddings, inputs)
    if self._hidden_dim:
      embeddings = self._project_layer(embeddings)
    return embeddings


class PixelEncoderLayer(tf.keras.layers.Layer):
  """Pixel encoding layer (ResNet)."""

  def __init__(self, name, filters, kernel_sizes):
    super(PixelEncoderLayer, self).__init__(name=name)
    self._filters = filters
    self._kernel_sizes = kernel_sizes

  def build(self, input_shape):
    self._conv_layer_1 = tf.keras.layers.Conv2D(
        filters=self._filters[0],
        kernel_size=self._kernel_sizes[0],
        strides=1,
        padding='same')
    self._conv_layer_2 = tf.keras.layers.Conv2D(
        filters=self._filters[1],
        kernel_size=self._kernel_sizes[1],
        strides=1,
        padding='same')
    self._conv_layer_3 = tf.keras.layers.Conv2D(
        filters=self._filters[2],
        kernel_size=self._kernel_sizes[2],
        strides=2,
        padding='same')

    self._batch_norm_layer_1 = tf.keras.layers.BatchNormalization()
    self._batch_norm_layer_2 = tf.keras.layers.BatchNormalization()
    self._batch_norm_layer_3 = tf.keras.layers.BatchNormalization()

  def call(self, input_tensor, training, dropout=0.0):
    """Defines a single encoding layer."""
    x = input_tensor
    skip = x

    x = self._conv_layer_1(x)
    x = self._batch_norm_layer_1(x, training=training)
    x = tf.nn.relu(x)
    if training:
      x = tf.nn.dropout(x, rate=dropout)

    x = self._conv_layer_2(x)
    x = self._batch_norm_layer_2(x, training=training)
    x += skip
    x = tf.nn.relu(x)
    if training:
      x = tf.nn.dropout(x, rate=dropout)

    x = self._conv_layer_3(x)
    x = self._batch_norm_layer_3(x, training=training)
    x = tf.nn.relu(x)
    if training:
      x = tf.nn.dropout(x, rate=dropout)

    return x


class EncoderLayer(tf.keras.layers.Layer):
  """Generates encoder outputs for both the pixels and view hierarchy."""

  def __init__(self, hparams, word_embedding_layer):
    super(EncoderLayer, self).__init__(name='dual_encoder')
    self._hparams = hparams
    self._word_embedding_layer = word_embedding_layer

  def build(self, input_shape):
    self._type_embedding_layer = EmbeddingLayer(
        name='object_type',
        vocab_size=100,
        embedding_dim=self._hparams['hidden_size'])
    self._clickable_embedding_layer = EmbeddingLayer(
        name='object_clickable',
        vocab_size=2,
        embedding_dim=self._hparams['hidden_size'])
    self._pos_embedding_layers = [
        EmbeddingLayer(
            name='object_pos_0',
            vocab_size=self._hparams['max_pixel_pos'],
            embedding_dim=self._hparams['hidden_size']),
        EmbeddingLayer(
            name='object_pos_1',
            vocab_size=self._hparams['max_pixel_pos'],
            embedding_dim=self._hparams['hidden_size']),
        EmbeddingLayer(
            name='object_pos_2',
            vocab_size=self._hparams['max_pixel_pos'],
            embedding_dim=self._hparams['hidden_size']),
        EmbeddingLayer(
            name='object_pos_3',
            vocab_size=self._hparams['max_pixel_pos'],
            embedding_dim=self._hparams['hidden_size'],
        )
    ]
    self._dom_embedding_layers = [
        EmbeddingLayer(
            name='object_dom_pos_0',
            vocab_size=self._hparams['max_dom_pos'],
            embedding_dim=self._hparams['hidden_size']),
        EmbeddingLayer(
            name='object_dom_pos_1',
            vocab_size=self._hparams['max_dom_pos'],
            embedding_dim=self._hparams['hidden_size']),
        EmbeddingLayer(
            name='object_dom_pos_2',
            vocab_size=self._hparams['max_dom_pos'],
            embedding_dim=self._hparams['hidden_size'])
    ]

    self._final_layer = tf.keras.layers.Dense(
        self._hparams['hidden_size'], activation=None)
    self._vh_final_layer = tf.keras.layers.Dense(
        self._hparams['hidden_size'], activation=tf.nn.tanh)
    self._pixel_layers = self._get_encoder3(initial_channel_size=1)
    self._transformer_encoder = nlp_transformer.EncoderStack(self._hparams)

  def call(self, features, object_selector, training):
    # Compute encoding
    with tf.name_scope('encoder'):
      pixel_encoding = self._encode_pixel(features, object_selector, training)
      vh_encoding, obj_embedding = self._encode_view_hierarchy(
          features, object_selector, training)

      logging.info('Screen encoder: %s', self._hparams['screen_encoder'])
      if self._hparams['screen_encoder'] == 'pixel_only':
        combined_output = pixel_encoding
      elif self._hparams['screen_encoder'] == 'pixel_transformer':
        combined_output = tf.concat([pixel_encoding, vh_encoding], -1)
      elif self._hparams['screen_encoder'] == 'pixel_mlp':
        combined_output = tf.concat([pixel_encoding, obj_embedding], -1)
      else:
        raise ValueError

      # [valid_obj, hidden_size]
      logits = self._final_layer(combined_output)
      logits = tf.nn.relu(logits)
      if training:
        logits = tf.nn.dropout(logits, rate=self._hparams['dropout'])
      # Add the length dimension.
      logits = tf.expand_dims(logits, 1)
    return logits

  def _encode_pixel(self, features, object_selector, training):
    # Flatten object pixels.
    obj_pixels = tf.reshape(features['obj_pixels'], [-1, 64, 64, 1])
    # Otherwise, we just encode worker nodes' pixels.
    valid_obj_pixels = tf.gather(obj_pixels, object_selector)

    thumbnail_encoding = valid_obj_pixels
    for layer in self._pixel_layers:
      thumbnail_encoding = layer(
          thumbnail_encoding,
          training=training,
          dropout=self._hparams['dropout'])

    # [worker_node, 256]
    thumbnail_encoding = tf.reshape(thumbnail_encoding, [-1, 256])

    return thumbnail_encoding

  def _get_encoder3(self, initial_channel_size=3):
    """Defines the encoding model with a pre-defined filter/kernel sizes."""
    pixel_layers = []
    filter_groups = [[initial_channel_size, initial_channel_size, 4],
                     [4, 4, 16], [16, 16, 32], [32, 32, 64], [64, 64, 128],
                     [128, 128, 256]]
    kernel_size_groups = [[5, 3, 5], [5, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3],
                          [3, 3, 3]]

    for index, (filters, kernel_sizes) in enumerate(
        zip(filter_groups, kernel_size_groups)):
      assert len(filters) == len(kernel_sizes)
      name = 'pixel_encoder_{}'.format(index)
      layer = PixelEncoderLayer(name, filters, kernel_sizes)
      pixel_layers.append(layer)

    return pixel_layers

  def _embed_composite_feature(self, features, embedding_layers):
    """Embed a position feature."""
    embedding_list = []
    for i in range(len(embedding_layers)):
      embedding_list.append(embedding_layers[i](features[:, :, i]))
    embedding = tf.add_n(embedding_list)
    return embedding

  def _encode_view_hierarchy(self, features, object_selector, training):
    """Encodes view hierarchy."""
    logging.info('Using Transformer screen encoder')

    # obj_text only contain the first phrase if multiple exist.
    # [batch, node_num, 10, hidden_size]
    developer_embeddings = self._word_embedding_layer(
        features['developer_token_id'])
    resource_embeddings = self._word_embedding_layer(
        features['resource_token_id'])

    developer_embeddings = self._aggregate_text_embedding(
        features['developer_token_id'], developer_embeddings)
    resource_embeddings = self._aggregate_text_embedding(
        features['resource_token_id'], resource_embeddings)

    type_embedding = self._type_embedding_layer(
        tf.maximum(features['obj_type'], 0))
    clickable_embedding = self._clickable_embedding_layer(
        features['obj_clickable'])

    object_info = []
    if 'text' in self._hparams['screen_embedding_feature']:
      object_info.append(developer_embeddings)
      object_info.append(resource_embeddings)
    if 'type' in self._hparams['screen_embedding_feature']:
      object_info.append(type_embedding)
    if 'pos' in self._hparams['screen_embedding_feature']:
      pos_embedding = self._embed_composite_feature(features['obj_screen_pos'],
                                                    self._pos_embedding_layers)
      object_info.append(pos_embedding)
    if 'click' in self._hparams['screen_embedding_feature']:
      object_info.append(clickable_embedding)
    if 'dom' in self._hparams['screen_embedding_feature']:
      dom_embedding = self._embed_composite_feature(features['obj_dom_pos'],
                                                    self._dom_embedding_layers)
      object_info.append(dom_embedding)

    object_embed = tf.concat(object_info, -1)
    object_embed = self._vh_final_layer(object_embed)

    # [batch, obj_num]
    object_mask = tf.cast(tf.not_equal(features['obj_type'], -1), tf.float32)
    # [batch, obj_num, hidden_dim]
    object_embed = object_embed * tf.expand_dims(object_mask, -1)
    att_bias = model_utils.get_padding_bias(object_mask)

    if training:
      object_embed = tf.nn.dropout(object_embed, rate=self._hparams['dropout'])

    encoder_output = self._transformer_encoder(
        object_embed,
        attention_bias=att_bias,
        inputs_padding=None,  # not used in EncoderStack.
        training=training)

    object_embed = tf.reshape(object_embed, [-1, self._hparams['hidden_size']])
    encoder_output = tf.reshape(encoder_output,
                                [-1, self._hparams['hidden_size']])
    valid_object_embed = tf.gather(object_embed, object_selector)
    valid_screen_encoding = tf.gather(encoder_output, object_selector)
    return valid_screen_encoding, valid_object_embed

  def _aggregate_text_embedding(self, token_ids, embeddings):
    """Aggregate text embedding for a UI element."""
    if self._hparams['obj_text_aggregation'] == 'max':
      # Find valid tokens (not PADDING/EOS/UNK/START).
      valid_token_mask = tf.greater_equal(token_ids, 4)
      # Use large negative bias for invalid tokens.
      invalid_token_bias = tf.cast(
          tf.logical_not(valid_token_mask), tf.float32) * -1e9
      # [batch, node_num, word_num, hidden_size]
      embeddings = embeddings + tf.expand_dims(invalid_token_bias, axis=-1)
      # Max value for each dimension, [batch, node_num, hidden_size].
      embeddings = tf.reduce_max(embeddings, axis=-2)
      # For objects with no text, use 0.
      valid_object_mask = tf.cast(
          tf.reduce_any(valid_token_mask, axis=-1), tf.float32)
      embeddings = embeddings * tf.expand_dims(valid_object_mask, axis=-1)

    elif self._hparams['obj_text_aggregation'] == 'sum':
      # [batch, step, #max_obj, #max_token]  0 for padded tokens
      real_objects = tf.cast(tf.greater_equal(token_ids, 4), tf.float32)
      # [batch, step, #max_obj, hidden]   0s for padded objects
      embeddings = tf.reduce_sum(
          input_tensor=embeddings * tf.expand_dims(real_objects, 3), axis=-2)

    else:
      raise ValueError('Unrecognized token aggregation %s' %
                       (self._hparams['obj_text_aggregation']))
    return embeddings


class DecoderLayer(tf.keras.layers.Layer):
  """Captioning decoder layer."""

  def __init__(self, hparams, word_embedding_layer, position_embedding_layer):
    super(DecoderLayer, self).__init__(name='decoder')
    self._hparams = hparams
    self._word_embedding_layer = word_embedding_layer
    self._position_embedding_layer = position_embedding_layer

  def build(self, inputs):
    self._transformer_decoder = nlp_transformer.DecoderStack(self._hparams)

  def call(self,
           decoder_inputs,
           encoder_outputs,
           decoder_self_attention_bias,
           attention_bias,
           training,
           cache=None):
    """Return the output of the decoder layer stacks.

    Args:
      decoder_inputs: A tensor with shape [batch_size, target_length,
        hidden_size].
      encoder_outputs: A tensor with shape [batch_size, input_length,
        hidden_size]
      decoder_self_attention_bias: A tensor with shape [1, 1, target_len,
        target_length], the bias for decoder self-attention layer.
      attention_bias: A tensor with shape [batch_size, 1, 1, input_length], the
        bias for encoder-decoder attention layer.
      training: A bool, whether in training mode or not.
      cache: (Used for fast decoding) A nested dictionary storing previous
        decoder self-attention values. The items are:
          {layer_n: {"k": A tensor with shape [batch_size, i, key_channels],
                     "v": A tensor with shape [batch_size, i, value_channels]},
                       ...}

    Returns:
      Output of decoder layer stack.
      float32 tensor with shape [batch_size, target_length, hidden_size]
    """
    # Run values
    outputs = self._transformer_decoder(
        decoder_inputs,
        encoder_outputs,
        decoder_self_attention_bias,
        attention_bias,
        training=training,
        cache=cache)
    return outputs


class WidgetCaptionModel(tf.keras.Model):
  """Widget Captioning Model."""
  _SCORE_NAMES = [
      'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'ROUGE-1-f1-mean',
      'ROUGE-1-f1-min', 'ROUGE-1-f1-max', 'ROUGE-2-f1-mean', 'ROUGE-2-f1-min',
      'ROUGE-2-f1-max', 'ROUGE-L-f1-mean', 'ROUGE-L-f1-min', 'ROUGE-L-f1-max'
  ]

  # 10 words + EOS symbol.
  _MAX_DECODE_LENGTH = 11

  def __init__(self, hparams):
    super(WidgetCaptionModel, self).__init__()
    self._hparams = hparams
    with tf.name_scope('captioning'):
      self._word_embedding_layer = EmbeddingLayer(
          name='word',
          hidden_dim=self._hparams['hidden_size'],
          embedding_file=self._hparams['embedding_file'],
          vocab_size=self._hparams['vocab_size'],
          embedding_dim=self._hparams['hidden_size'],  # not used
          trainable=self._hparams['glove_trainable'])
      self._position_embedding_layer = layers.RelativePositionEmbedding(
          hidden_size=self._hparams['hidden_size'])

      self._encoder = EncoderLayer(self._hparams, self._word_embedding_layer)
      self._decoder = DecoderLayer(self._hparams, self._word_embedding_layer,
                                   self._position_embedding_layer)
      self._word_layer = tf.keras.layers.Dense(
          units=self._hparams['vocab_size'])

    self.model_metrics = {
        'loss': tf.keras.metrics.Mean(name='loss'),
        'global_norm': tf.keras.metrics.Mean(name='global_norm'),
    }

    self.caption_metrics = {}
    for score_name in self._SCORE_NAMES:
      scoped_name = 'COCO/{}'.format(score_name)
      self.caption_metrics[scoped_name] = tf.keras.metrics.Mean(
          name=scoped_name)

    self._word_vocab = []
    with tf.io.gfile.GFile(self._hparams['word_vocab_path']) as f:
      for index, line in enumerate(f):
        if index >= self._hparams['vocab_size']:
          break
        self._word_vocab.append(line.strip())

  def call(self, inputs, training):
    features, targets = inputs

    object_selector = self._caption_object_selector(features)

    encoder_outputs = self._encoder(features, object_selector, training)
    if self._hparams['decoding_task']:
      if targets is None:
        return self.predict(encoder_outputs, training)
      else:
        return self.decode(targets, encoder_outputs, training)

  def _caption_object_selector(self, features):
    worker = tf.reshape(tf.equal(features['label_flag'], 0), [-1])
    # [worker_node] indices into [BxN] vector for valid worker node.
    worker_position = tf.reshape(tf.where(worker), [-1])
    return worker_position

  def _caption_loss(self, targets, logits):
    per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=targets, logits=logits)

    # Create non-padding mask and only compute loss for non-padding positions.
    non_padding = tf.greater(targets, input_utils.PADDING)
    mask = tf.cast(non_padding, tf.float32)
    per_example_loss = per_example_loss * mask
    avg_loss = tf.reduce_sum(per_example_loss) / tf.reduce_sum(mask)
    avg_loss = tf.cond(tf.math.is_nan(avg_loss), lambda: 0.0, lambda: avg_loss)
    return avg_loss

  def train_step(self, data):
    targets, _ = self.compute_targets(data)

    with tf.GradientTape() as tape:
      logits = self([data, targets], training=True)

      if self._hparams['decoding_task']:
        avg_loss = self._caption_loss(targets, logits)

      trainable_vars = self.trainable_variables
      gradients = tape.gradient(avg_loss, trainable_vars)
      gradients, global_norm = tf.clip_by_global_norm(
          gradients, self._hparams['clip_norm'])

      # Update weights
      self.optimizer.apply_gradients(zip(gradients, trainable_vars))

    self.model_metrics['loss'].update_state(avg_loss)
    self.model_metrics['global_norm'].update_state(global_norm)
    train_metrics = ['loss', 'global_norm']
    return {m: self.model_metrics[m].result() for m in train_metrics}

  def test_step(self, data):
    targets, references = self.compute_targets(data)
    logits = self([data, targets], training=False)
    avg_loss = self._caption_loss(targets, logits)
    decoded = self([data, None], training=False)
    self.compute_caption_metrics(decoded, references)

    self.model_metrics['loss'].update_state(avg_loss)
    return {m.name: m.result() for m in self.model_metrics.values()}

  def compute_caption_metrics(self, predictions, references):
    """Computes the eval metrics for decoding."""
    py_types = [tf.float32] * len(self._SCORE_NAMES)
    scores = tf.py_function(compute_score,
                            (predictions, references, self._word_vocab),
                            py_types)
    for name, score in zip(self._SCORE_NAMES, scores):
      scoped_name = 'COCO/{}'.format(name)
      self.caption_metrics[scoped_name].update_state(score)
      self.model_metrics[scoped_name] = self.caption_metrics[scoped_name]

  def decode(self, targets, encoder_outputs, training):
    """Generate logits for each value in the target sequence.

    Args:
      targets: target values for the output sequence. int tensor with shape
        [batch_size, target_length]
      encoder_outputs: continuous representation of input sequence. float tensor
        with shape [batch_size, input_length, hidden_size]
      training: boolean, whether in training mode or not.

    Returns:
      float32 tensor with shape [batch_size, target_length, vocab_size]
    """
    with tf.name_scope('decode'):
      length = tf.shape(targets)[1]
      decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(
          length)
      encoder_shape = tf.shape(encoder_outputs)
      # [batch, 1] as there is only one object as input for decoding.
      mask = tf.ones([encoder_shape[0], encoder_shape[1]])
      # In mask, 1 = valid object, 0 = padding, attn_bias will have -NEG_INF for
      # paddings and 0 for valid objects.
      attention_bias = model_utils.get_padding_bias(mask)

      # Prepare inputs to decoder layers by shifting targets, adding positional
      # encoding and applying dropout.
      targets = tf.pad(
          targets, [[0, 0], [1, 0]], constant_values=input_utils.START)
      # Remove last element.
      targets = targets[:, :-1]
      decoder_inputs = self._word_embedding_layer(targets)

      # No need to shift, use START above to shift.
      # with tf.name_scope('shift_targets'):
      #   # Shift targets to the right, and remove the last element
      #   decoder_inputs = tf.pad(decoder_inputs,
      #                           [[0, 0], [1, 0], [0, 0]])[:, :-1, :]

      with tf.name_scope('add_pos_encoding'):
        pos_encoding = self._position_embedding_layer(decoder_inputs)
        decoder_inputs += pos_encoding

      if training:
        decoder_inputs = tf.nn.dropout(
            decoder_inputs, rate=self._hparams['layer_postprocess_dropout'])

      decoder_outputs = self._decoder(
          decoder_inputs,
          encoder_outputs,
          decoder_self_attention_bias,
          attention_bias,
          training=training)
      logits = self._word_layer(decoder_outputs)
      return logits

  def predict(self, encoder_outputs, training):
    """Return predicted sequence."""
    batch_size = tf.shape(encoder_outputs)[0]
    # input_length = tf.shape(encoder_outputs)[1]
    # 10 words + EOS symbol
    symbols_to_logits_fn = self._get_symbols_to_logits_fn(
        max_decode_length=self._MAX_DECODE_LENGTH, training=training)

    # Create initial set of IDs that will be passed into symbols_to_logits_fn.
    initial_ids = tf.ones([batch_size], dtype=tf.int32) * input_utils.START

    # Create cache storing decoder attention values for each layer.
    # pylint: disable=g-complex-comprehension
    init_decode_length = 0
    num_heads = self._hparams['num_heads']
    dim_per_head = self._hparams['hidden_size'] // num_heads
    cache = {
        'layer_%d' % layer: {
            'k':
                tf.zeros(
                    [batch_size, init_decode_length, num_heads, dim_per_head]),
            'v':
                tf.zeros(
                    [batch_size, init_decode_length, num_heads, dim_per_head])
        } for layer in range(self._hparams['num_hidden_layers'])
    }
    # pylint: enable=g-complex-comprehension

    # Add encoder output and attention bias to the cache.
    encoder_shape = tf.shape(encoder_outputs)
    # [batch, 1] as there is only one object as input for decoding.
    mask = tf.ones([encoder_shape[0], encoder_shape[1]])
    # In mask, 1 = valid object, 0 = padding, attn_bias will have -NEG_INF for
    # paddings and 0 for valid objects.
    attention_bias = model_utils.get_padding_bias(mask)

    cache['encoder_outputs'] = encoder_outputs
    cache['encoder_decoder_attention_bias'] = attention_bias

    # Use beam search to find the top beam_size sequences and scores.
    decoded_ids, _ = ops.beam_search.sequence_beam_search(
        symbols_to_logits_fn=symbols_to_logits_fn,
        initial_ids=initial_ids,
        initial_cache=cache,
        vocab_size=self._hparams['vocab_size'],
        beam_size=self._hparams['beam_size'],
        alpha=1,
        max_decode_length=self._MAX_DECODE_LENGTH,
        eos_id=input_utils.EOS)

    # Get the top sequence for each batch element and remove START symbol.
    top_decoded_ids = decoded_ids[:, 0, 1:]
    # top_scores = scores[:, 0]
    return top_decoded_ids

  def compute_targets(self, features):
    """Compute the target token ids and phrase ids."""
    batch_size = tf.shape(features['label_flag'])[0]
    num_objects = tf.shape(features['label_flag'])[1]

    worker_position = self._caption_object_selector(features)

    # [worker_node, 1]: retrieve the reference captions.
    valid_references = tf.gather(
        tf.reshape(features['reference'], [-1]), worker_position)

    # [worker_node, seq_len]: retrieve reference phrases.
    target_phrase = features['caption_token_id']
    target_phrase = tf.reshape(
        target_phrase, [batch_size * num_objects, self._MAX_DECODE_LENGTH])
    valid_target_phrase = tf.gather(target_phrase, worker_position)

    return valid_target_phrase, valid_references

  def _get_symbols_to_logits_fn(self, max_decode_length, training):
    """Returns a decoding function that calculates logits of the next tokens."""
    timing_signal = self._position_embedding_layer(
        inputs=None, length=max_decode_length)
    decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(
        max_decode_length)

    def symbols_to_logits_fn(ids, i, cache):
      """Generate logits for next potential IDs.

      Args:
        ids: Current decoded sequences. int tensor with shape [batch_size *
          beam_size, i + 1].
        i: Loop index.
        cache: dictionary of values storing the encoder output, encoder-decoder
          attention bias, and previous decoder attention values.

      Returns:
        Tuple of
          (logits with shape [batch_size * beam_size, vocab_size],
           updated cache values)
      """
      # Set decoder input to the last generated IDs. The previous ids attention
      # key/value are already stored in the cache.
      decoder_input = ids[:, -1:]

      # Preprocess decoder input by getting embeddings and adding timing signal.
      decoder_input = self._word_embedding_layer(decoder_input)
      decoder_input += timing_signal[i:i + 1]
      self_attention_bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]

      decoder_outputs = self._decoder(
          decoder_input,
          cache.get('encoder_outputs'),
          self_attention_bias,
          cache.get('encoder_decoder_attention_bias'),
          training=training,
          cache=cache)

      # Only use the last decoded state.
      decoder_outputs = decoder_outputs[:, -1, :]
      logits = self._word_layer(decoder_outputs)
      return logits, cache

    return symbols_to_logits_fn


class TensorBoardCallBack(tf.keras.callbacks.TensorBoard):
  """Learning rate log callback."""

  def on_train_batch_begin(self, batch, logs=None):
    super(TensorBoardCallBack, self).on_train_batch_begin(batch, logs)
    try:
      lr = self.model.optimizer.learning_rate(batch)
    except TypeError:
      lr = self.model.optimizer.learning_rate
    if batch % 100 == 0:
      try:
        with self.writer.as_default():
          tf.summary.scalar('learning rate', tensor=lr)
          self.writer.flush()
      except AttributeError:
        logging.info('TensorBoard not init yet')


def init_resnet(hparams, model):
  """Init resnet weights from a TF model if provided."""
  if not hparams['widget_encoder_checkpoint']:
    return

  reader = tf.train.load_checkpoint(hparams['widget_encoder_checkpoint'])

  # Initialize model weights.
  init_set = input_utils.input_fn(
      hparams['train_files'],
      1,
      hparams['vocab_size'],
      hparams['max_pixel_pos'],
      hparams['max_dom_pos'],
      epoches=1,
      buffer_size=1)
  init_features = next(iter(init_set))
  init_target = model.compute_targets(init_features)
  model([init_features, init_target[0]], training=True)

  weight_value_tuples = []
  for layer in model._encoder._pixel_layers:  # pylint: disable=protected-access
    for param in layer.weights:
      if 'batch_normalization' in param.name:
        continue
      sublayer, varname = param.name.replace(':0', '').split('/')[-2:]
      var_name = 'encoder/{}/{}'.format(sublayer, varname)
      if reader.has_tensor(var_name):
        logging.info('Found pretrained weights: %s %s, %s %s', param.name,
                     param.shape, var_name,
                     reader.get_tensor(var_name).shape)
        weight_value_tuples.append((param, reader.get_tensor(var_name)))
  logging.info('Load pretrained %s weights', len(weight_value_tuples))
  tf.keras.backend.batch_set_value(weight_value_tuples)


def main(argv=None):
  del argv

  hparams = create_hparams(FLAGS.experiment)

  if hparams['distribution_strategy'] == 'multi_worker_mirrored':
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
  elif hparams['distribution_strategy'] == 'mirrored':
    strategy = tf.distribute.MirroredStrategy()
  else:
    raise ValueError('Only `multi_worker_mirrored` is supported strategy '
                     'in Keras MNIST example at this time. Strategy passed '
                     'in is %s' % hparams['distribution_strategy'])

  # Build the train and eval datasets from the MNIST data.
  train_set = input_utils.input_fn(
      hparams['train_files'],
      hparams['batch_size'],
      hparams['vocab_size'],
      hparams['max_pixel_pos'],
      hparams['max_dom_pos'],
      epoches=1,
      buffer_size=hparams['train_buffer_size'])

  dev_set = input_utils.input_fn(
      hparams['eval_files'],
      hparams['eval_batch_size'],
      hparams['vocab_size'],
      hparams['max_pixel_pos'],
      hparams['max_dom_pos'],
      epoches=100,
      buffer_size=hparams['eval_buffer_size'])

  # Create and compile the model under Distribution strategy scope.
  # `fit`, `evaluate` and `predict` will be distributed based on the strategy
  # model was compiled with.
  with strategy.scope():
    model = WidgetCaptionModel(hparams)
    lr_schedule = optimizer.LearningRateSchedule(
        hparams['learning_rate_constant'], hparams['hidden_size'],
        hparams['learning_rate_warmup_steps'])
    opt = tf.keras.optimizers.Adam(
        lr_schedule,
        hparams['optimizer_adam_beta1'],
        hparams['optimizer_adam_beta2'],
        epsilon=hparams['optimizer_adam_epsilon'])
    model.compile(optimizer=opt)

    init_resnet(hparams, model)

  callbacks = [tf.keras.callbacks.TerminateOnNaN()]
  if FLAGS.model_dir:
    ckpt_filepath = os.path.join(FLAGS.model_dir, 'saved/{epoch:04d}')
    backup_dir = os.path.join(FLAGS.model_dir, 'backup')
    tensorboard_callback = TensorBoardCallBack(log_dir=FLAGS.model_dir)
    callbacks.append(tensorboard_callback)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_filepath, save_weights_only=True)
    callbacks.append(model_checkpoint_callback)
    if tf.executing_eagerly():
      callbacks.append(
          tf.keras.callbacks.experimental.BackupAndRestore(
              backup_dir=backup_dir))

  # Train the model with the train dataset.
  history = model.fit(
      x=train_set,
      epochs=hparams['train_epoches'],
      validation_data=dev_set,
      validation_steps=10,
      callbacks=callbacks)

  logging.info('Training ends successfully. `model.fit()` result: %s',
               history.history)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  app.run(main)
