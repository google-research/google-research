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

"""Common basic utilities and Transformer adapters."""
import functools
import glob
import os

from typing import Callable, Dict, List, Optional, Tuple, Union

import tensorflow as tf2
import tensorflow.compat.v1 as tf

from hypertransformer.tf.core import transformer


TransformerParamsFn = Callable[[int], transformer.TransformerParams]


def var_getter_wrapper(getter, name, *args, **kwargs):
  """Convenient wrapper around CNN variable getters."""
  weights = kwargs.pop('_weights')
  shape = kwargs.get('shape')
  cnn_var_getter = kwargs.pop('_cnn_var_getter')
  getter_dict = kwargs.pop('_getter_dict')
  add_trainable_weights = kwargs.pop('_add_trainable_weights', False)
  var_reg_weight = kwargs.pop('_var_reg_weight', 0.0)
  evaluation = kwargs.pop('_evaluation', False)
  separate_bn = kwargs.pop('_separate_bn', False)
  passed_regularizer = kwargs.pop('_kernel_regularizer', None)

  # Currently only support FC and conv kernel regularization.
  regularizer = passed_regularizer if name.endswith('/kernel') else None

  if 'offsets' in getter_dict:
    # Dictionary of offsets is used to allocate more than one tensor in the
    # same embedding.
    offsets = getter_dict['offsets']
    # These are actual generated weights.
    all_weights = getter_dict['all_weights']
  else:
    offsets, all_weights = {}, {}
    getter_dict['offsets'] = offsets
    getter_dict['all_weights'] = all_weights

  built_weights = cnn_var_getter(offsets, weights, shape, name)

  var_name = name
  if evaluation and separate_bn:
    var_name += '-eval'

  if built_weights is None:
    kwargs['regularizer'] = regularizer
    built_weights = getter(var_name, *args, **kwargs)
  else:
    if add_trainable_weights:
      var_weights = getter(var_name, *args, **kwargs)
      if var_reg_weight > 0.0:
        print(f'Adding weight variation regularization to {name}')
        built_weight_mag = tf.reduce_sum(tf.square(built_weights))
        var_weight_mag = tf.reduce_sum(tf.square(var_weights))
        reg_loss = var_reg_weight * built_weight_mag / (1e-8 + var_weight_mag)
        tf.losses.add_loss(tf.identity(reg_loss, name=f'reg_loss_{name}'),
                           loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)
      built_weights += var_weights

    if regularizer is not None and name not in all_weights:
      tf.losses.add_loss(regularizer(built_weights),
                         loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)

  all_weights[name] = built_weights
  return all_weights[name]


class TransformerIO(tf.Module):
  """Encoder and decoder interfacing with the Transformer."""

  def __init__(self,
               num_labels,
               num_weights,
               weight_block_size,
               embedding_dim = 8,
               weight_embedding_dim = None):
    self.num_labels = num_labels
    self.num_weights = num_weights
    self.embedding_dim = embedding_dim
    if weight_embedding_dim is None:
      weight_embedding_dim = embedding_dim
    self.weight_embedding_dim = weight_embedding_dim
    self.weight_block_size = weight_block_size
    self.label_embs = []
    self.weight_embs = []
    with tf.variable_scope(None, default_name='label_embeddings'):
      # 1 additional class is reserved for "no label" class
      for label_idx in range(self.num_labels + 1):
        self.label_embs.append(tf.get_variable(
            f'label_{label_idx}',
            shape=(embedding_dim,),
            initializer=tf.random_normal_initializer(stddev=1.0),
            dtype=tf.float32))
    self.label_embs = tf.stack(self.label_embs, axis=0)
    with tf.variable_scope(None, default_name='weight_embeddings'):
      for weight_idx in range(num_weights):
        self.weight_embs.append(tf.get_variable(
            f'weight_{weight_idx}',
            shape=(weight_embedding_dim,),
            initializer=tf.random_normal_initializer(stddev=1.0),
            dtype=tf.float32))

  def encode_labels(self, labels):
    """Generates label encodings."""
    return tf.gather(self.label_embs, labels, axis=0)


class SimpleTransformerIO(TransformerIO):
  """Encoder and decoder interfacing with the Transformer."""

  def encode_samples(self,
                     images,
                     labels):
    """Generates Transformer inputs from input samples."""
    with tf.name_scope(None, default_name='sample_encoder'):
      batch_size = images.shape[0]
      images = tf.reshape(images, [batch_size, -1])
      encoded_labels = self.encode_labels(labels)
      return tf.concat([encoded_labels, images], axis=-1)

  def extend_label_mask(self, label_mask):
    return label_mask

  def decode_weights(self, embeddings):
    """Generates weight patches from Transformer outputs."""
    with tf.name_scope(None, default_name='weight_decoder'):
      weights = []
      for weight_emb in self.weight_embs:
        weight_keys = embeddings[Ellipsis, :self.embedding_dim]
        weight_values = embeddings[Ellipsis, self.embedding_dim:]
        mixture = tf.einsum('j,ij->i', weight_emb, weight_keys)
        mixture = tf.nn.softmax(mixture)
        weights.append(tf.einsum('i,ij->j', mixture, weight_values))
      return weights


class SeparateTransformerIO(SimpleTransformerIO):
  """IO for feeding samples into Encoder and getting weights from Decoder."""

  def encode_weights(self):
    """Generates weight patches from Transformer outputs."""
    with tf.name_scope(None, default_name='weight_encoder'):
      tokens = []
      for i in range(self.num_weights):
        tensors = [self.weight_embs[i], tf.zeros(self.weight_block_size,)]
        tokens.append(tf.concat(tensors, axis=-1))
      return tf.stack(tokens, axis=0)

  def decode_weights(self, embeddings):
    """Generates weight patches from Transformer outputs."""
    with tf.name_scope(None, default_name='weight_decoder'):
      weights = []
      for i in range(self.num_weights):
        weights.append(embeddings[i, self.weight_embedding_dim:])
      return weights


class JointTransformerIO(TransformerIO):
  """Encoder and decoder interfacing with the Transformer."""

  def encode_samples(self,
                     images,
                     labels):
    """Generates Transformer inputs from input samples."""
    with tf.name_scope(None, default_name='sample_encoder'):
      batch_size = images.shape[0]
      images = tf.reshape(images, [batch_size, -1])
      encoded_labels = self.encode_labels(labels)
      sequence = tf.concat([encoded_labels, images], axis=-1)
    with tf.name_scope(None, default_name='weight_encoder'):
      weight_sequence = []
      for i in range(self.num_weights):
        weight_emb = self.weight_embs[i]
        zero_emb = tf.zeros(shape=images.shape[1:], dtype=tf.float32)
        weight_sequence.append(tf.concat([weight_emb, zero_emb], axis=-1))
      weight_sequence = tf.stack(weight_sequence, axis=0)
      sequence = tf.concat([weight_sequence, sequence], axis=0)
    return sequence

  def extend_label_mask(self, label_mask):
    weight_mask = tf.zeros((self.num_weights,), dtype=tf.float32)
    return tf.concat([weight_mask, label_mask], axis=-1)

  def decode_weights(self, embeddings):
    """Generates weight patches from Transformer outputs."""
    with tf.name_scope(None, default_name='weight_decoder'):
      weights = []
      for i in range(self.num_weights):
        weights.append(embeddings[i, self.embedding_dim:])
      return weights


def split_variable_name(name):
  """Extracts layer name and variable name from the full variable scope."""
  parts = name.split('/')
  return parts[-2], parts[-1]


def _parse_label_spec(label_spec):
  """Parses label specification."""
  labels = []
  parts = label_spec.split(',')
  for part in parts:
    subset = part.split('-')
    if len(subset) == 1:
      labels.append(int(subset[0]))
    elif len(subset) == 2:
      labels.extend(range(int(subset[0]), int(subset[1]) + 1))
    else:
      raise ValueError('Wrong label specification format.')
  return labels


def parse_dataset_spec(dataset_spec):
  """Parses the dataset specification."""
  if ':' not in dataset_spec:
    return dataset_spec, None
  parts = dataset_spec.split(':')
  if len(parts) > 2:
    raise ValueError(
        'Wrong dataset specification format: should be a dataset name, or be '
        'of the format "dataset_name:1-10,20-30,40".')
  dataset_spec, label_spec = parts
  return dataset_spec, _parse_label_spec(label_spec)


def nonlinearity(activation):
  if activation == 'relu':
    return tf.nn.relu
  elif activation == 'gelu':
    return tf2.nn.gelu
  elif activation == 'lrelu':
    return functools.partial(tf.nn.leaky_relu, alpha=0.1)
  else:
    raise ValueError(f'Unknown nonlinearity {activation}.')


def extract_checkpoint_step(s):
  # Files have format prefix-<step>.index
  return int(s.rsplit('.', 1)[0].rsplit('-', 1)[1])


def find_latest_checkpoint(ckpt_dir):
  all_checkpoints = glob.glob(os.path.join(ckpt_dir, '*.index'))
  if not all_checkpoints:
    return None
  latest = sorted(all_checkpoints, key=extract_checkpoint_step)[0]
  return latest.rsplit('.', 1)[0]


def latest_checkpoint(dir_or_checkpoint):
  # That's actual checkpoint prefix.
  if not os.path.isdir(dir_or_checkpoint) and os.path.exists(
      dir_or_checkpoint + '.index'):
    return dir_or_checkpoint
  # Rely on tensorflow latest_checkpoint but if no "checkpoint" is present just
  # scan the directory.
  latest = tf.train.latest_checkpoint(dir_or_checkpoint)
  return latest or find_latest_checkpoint(dir_or_checkpoint)


class MultiFileWriter(object):
  """Summary writer that supports writing to multiple files at once."""

  def __init__(self, logdir, *args, **kwargs):
    self.summary_args = list(args)
    self.summary_kwargs = dict(kwargs)
    self.logdir = logdir
    self.writers = {}

  def _get_writer(self, name=None):
    if name not in self.writers:
      self.writers[name] = tf.summary.FileWriter(
          os.path.join(self.logdir, name or ''),
          *self.summary_args, **self.summary_kwargs)
    return self.writers[name]

  def add_summary(self, summary, global_step=None):
    if isinstance(summary, dict):
      for k, v in summary.items():
        self._get_writer(k).add_summary(v, global_step=global_step)
    else:
      self._get_writer().add_summary(summary, global_step=global_step)

  def close(self):
    for each in self.writers.values():
      each.close()

  def flush(self):
    for each in self.writers.values():
      each.flush()


def _normalize_tag(tag):
  if '_' not in tag:
    return tag
  normalized_tag, value = tag.rsplit('_', 1)
  return normalized_tag if value.isdigit() else tag


def normalize_tags(s):
  """Normalizes tags inside serialized summary proto."""
  if isinstance(s, bytes):
    summary_proto = tf.summary.Summary.FromString(s)
    tags = set([v.tag for v in summary_proto.value])
    tag_map = {tag: tag for tag in tags}
    for each in tags:
      normalized_tag = _normalize_tag(each)
      if normalized_tag not in tag_map.values():
        tag_map[each] = normalized_tag
    for v in summary_proto.value:
      v.tag = tag_map[v.tag]
    return summary_proto.SerializeToString()
  elif isinstance(s, dict):
    for name in list(s.keys()):
      s[name] = normalize_tags(s[name])
    return s
  raise ValueError(f'Unexpected input {s}')


def load_variables(loc, var_list=None, step=None):
  """Returns variables from a given checkpoint."""
  path = latest_checkpoint(loc)
  if path is None:
    raise FileNotFoundError(f'No checkpoint available found at "{loc}"')
  loc = path
  if step is not None:
    base, _ = loc.rsplit('-', 1)
    loc = '-'.join([base, '%d' % step])

  tf.logging.info('Loading from %s ', loc)
  ckpt = tf.train.load_checkpoint(loc)

  result = {}
  for tensor in (var_list or ckpt.get_variable_to_shape_map()):
    result[tensor] = ckpt.get_tensor(tensor)
  return result
