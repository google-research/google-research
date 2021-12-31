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

"""Screen2Words input pipeline."""

from absl import flags
import tensorflow as tf

# Constants for embeddings.
PADDING = 0
EOS = 1
UKN = 2
START = 3

FLAGS = flags.FLAGS
MAX_TOKEN_PER_LABEL = 10


def _produce_target_phrase(phrases):
  """Randomly selects one phrase as the target phrase for training."""
  with tf.variable_scope('produce_output'):
    # Find indices for valid phrases with meaningful tokens.
    valid_phrase_indices = tf.reshape(
        tf.where(tf.reduce_any(tf.greater(phrases, EOS), -1)), [-1])
    # If indices is empty (no valid tokens/annotations), just use the index 0,

    # otherwise random shuffle the indices and select one.
    index = tf.cond(
        tf.greater(tf.shape(valid_phrase_indices)[0], 0),
        lambda: tf.cast(tf.random.shuffle(valid_phrase_indices)[0], tf.int32),
        lambda: 0)

    phrase = phrases[index]

    # Append EOS to the end of phrase.
    phrase = tf.boolean_mask(phrase, mask=tf.greater(phrase, PADDING))
    phrase = tf.concat([phrase, [EOS]], axis=0)

    # Pad the phrase to length of 11 (10 words + EOS).
    phrase = tf.pad(phrase,
                    [[0, MAX_TOKEN_PER_LABEL + 1 - tf.shape(phrase)[-1]]])
    return phrase, index


def _select_phrases(dense_features):
  """Selects phrases from the workers."""
  with tf.variable_scope('select_phrases'):
    # Sample one phrase for each node.
    output_phrase, ind = tf.map_fn(
        _produce_target_phrase,
        tf.expand_dims(dense_features['screen_caption_token_ids'], 0),
        fn_output_signature=(tf.int64, tf.int32))
    # Output shape: [N, seq_len]
    output_phrase = tf.reshape(output_phrase, [-1, MAX_TOKEN_PER_LABEL + 1])
    return output_phrase, ind


def _extract_image(dense_features, num_ui_objects, target_node=None):
  """Extracts image features."""
  with tf.variable_scope('extract_image'):
    visible = dense_features['visibility_seq'] * dense_features[
        'visibility_to_user_seq']
    obj_pixels = tf.reshape(dense_features['obj_img_mat'],
                            [num_ui_objects, 64, 64, 3])
    if target_node is not None:
      obj_pixels = tf.image.rgb_to_grayscale(tf.gather(obj_pixels, target_node))
    else:
      obj_pixels = tf.image.rgb_to_grayscale(obj_pixels)
      w = (
          dense_features['cord_x_seq'][:, 1] -
          dense_features['cord_x_seq'][:, 0])
      h = (
          dense_features['cord_y_seq'][:, 1] -
          dense_features['cord_y_seq'][:, 0])
      obj_visible = tf.logical_and(
          tf.equal(visible, 1),
          tf.logical_or(tf.greater(w, 0.005), tf.greater(h, 0.005)))
      obj_pixels = tf.where(obj_visible, obj_pixels, tf.zeros_like(obj_pixels))
    return tf.cast(obj_pixels, tf.float32) / 255.0, obj_visible


def parse_tf_example(serialized_example):
  """Parses a single tf example."""
  keys_to_features = {
      'developer_token_id': tf.VarLenFeature(tf.int64),
      'resource_token_id': tf.VarLenFeature(tf.int64),
      'screen_caption_token_ids': tf.VarLenFeature(tf.int64),
      'appdesc_token_id': tf.VarLenFeature(tf.int64),
      'clickable_seq': tf.VarLenFeature(tf.int64),
      'type_id_seq': tf.VarLenFeature(tf.int64),
      'cord_x_seq': tf.VarLenFeature(tf.float32),
      'cord_y_seq': tf.VarLenFeature(tf.float32),
      'visibility_to_user_seq': tf.VarLenFeature(tf.int64),
      'visibility_seq': tf.VarLenFeature(tf.int64),
      'attended_objects': tf.VarLenFeature(tf.int64),
      'label_flag': tf.VarLenFeature(tf.int64),  # 0: padding 1: node
      'obj_img_mat': tf.VarLenFeature(tf.int64),
      'obj_dom_pos': tf.VarLenFeature(tf.int64),
      'attention_boxes': tf.VarLenFeature(tf.float32),
      'gold_caption': tf.VarLenFeature(tf.string)
  }
  parsed = tf.parse_single_example(serialized_example, keys_to_features)
  dense_features = {}
  for key in keys_to_features:
    if key in ['gold_caption']:
      default_value = ''
    else:
      default_value = 0
    # Here we turn the features from id to one-hot vec.
    dense_features[key] = tf.sparse_tensor_to_dense(
        parsed[key], default_value=default_value)

  return dense_features


def create_parser(word_vocab_size,
                  max_pixel_pos=100,
                  max_dom_pos=500,
                  is_inference=False):
  """Creates a parser for tf.Example."""

  def process_tf_example(dense_features):
    """Parses a single tf example."""

    num_ui_objects = tf.shape(dense_features['clickable_seq'])[0]
    dense_features['screen_caption_token_ids'] = tf.reshape(
        dense_features['screen_caption_token_ids'], [5, MAX_TOKEN_PER_LABEL])

    dense_features['developer_token_id'] = tf.reshape(
        dense_features['developer_token_id'], [num_ui_objects, 10])

    dense_features['resource_token_id'] = tf.reshape(
        dense_features['resource_token_id'], [num_ui_objects, 10])
    dense_features['appdesc_token_id'] = tf.expand_dims(
        dense_features['appdesc_token_id'], 0)

    dense_features['screen_caption_token_ids'] = tf.where(
        tf.greater_equal(dense_features['screen_caption_token_ids'],
                         word_vocab_size),
        tf.cast(
            tf.fill(tf.shape(dense_features['screen_caption_token_ids']), UKN),
            dtype=tf.int64), dense_features['screen_caption_token_ids'])

    dense_features['developer_token_id'] = tf.where(
        tf.greater_equal(dense_features['developer_token_id'], word_vocab_size),
        tf.cast(
            tf.fill(tf.shape(dense_features['developer_token_id']), UKN),
            dtype=tf.int64), dense_features['developer_token_id'])

    dense_features['resource_token_id'] = tf.where(
        tf.greater_equal(dense_features['resource_token_id'], word_vocab_size),
        tf.cast(
            tf.fill(tf.shape(dense_features['resource_token_id']), UKN),
            dtype=tf.int64), dense_features['resource_token_id'])

    dense_features['appdesc_token_id'] = tf.where(
        tf.greater_equal(dense_features['appdesc_token_id'], word_vocab_size),
        tf.cast(
            tf.fill(tf.shape(dense_features['appdesc_token_id']), UKN),
            dtype=tf.int64), dense_features['appdesc_token_id'])

    dense_features['cord_x_seq'] = tf.reshape(
        dense_features['cord_x_seq'], [num_ui_objects, 2], name='cord_x_seq')

    dense_features['cord_y_seq'] = tf.reshape(
        dense_features['cord_y_seq'], [num_ui_objects, 2], name='cord_y_seq')

    dense_features['obj_dom_pos'] = tf.where(
        tf.greater_equal(dense_features['obj_dom_pos'], max_dom_pos),
        tf.cast(
            tf.fill(tf.shape(dense_features['obj_dom_pos']), 0),
            dtype=tf.int64), dense_features['obj_dom_pos'])

    dense_features['attention_boxes'] = tf.reshape(
        dense_features['attention_boxes'], [5, 5, 4], name='attention_boxes')
    dense_features['attended_objects'] = tf.reshape(
        dense_features['attended_objects'], [5, num_ui_objects],
        name='attended_objects')

    feature_dict = {}
    if not is_inference:
      output_phrase, ind = _select_phrases(dense_features)
      feature_dict['screen_caption_token_ids'] = output_phrase
      feature_dict['attention_boxes'] = tf.gather(
          dense_features['attention_boxes'], ind, axis=0)
      feature_dict['attended_objects'] = tf.gather(
          dense_features['attended_objects'], ind, axis=0)
    feature_dict['appdesc_token_id'] = dense_features['appdesc_token_id']
    feature_dict['developer_token_id'] = dense_features['developer_token_id']
    feature_dict['resource_token_id'] = dense_features['resource_token_id']
    feature_dict['label_flag'] = dense_features['label_flag']
    obj_pixels, obj_visible = _extract_image(dense_features, num_ui_objects)
    feature_dict['obj_pixels'] = obj_pixels
    feature_dict['obj_visible'] = obj_visible
    feature_dict['obj_screen_pos'] = tf.concat(
        [dense_features['cord_x_seq'], dense_features['cord_y_seq']], -1)
    feature_dict['obj_screen_pos'] = tf.to_int32(
        feature_dict['obj_screen_pos'] * (max_pixel_pos - 1))
    feature_dict['obj_clickable'] = dense_features['clickable_seq']
    feature_dict['obj_type'] = dense_features['type_id_seq']
    feature_dict['obj_dom_pos'] = tf.reshape(dense_features['obj_dom_pos'],
                                             [num_ui_objects, 3])
    feature_dict['references'] = dense_features['gold_caption']

    for key in [
        'screen_caption_token_ids', 'appdesc_token_id', 'developer_token_id',
        'resource_token_id', 'label_flag', 'obj_type', 'obj_clickable',
        'obj_visible', 'obj_dom_pos', 'attended_objects'
    ]:
      if key in feature_dict:
        feature_dict[key] = tf.cast(feature_dict[key], tf.int32)
    return feature_dict

  return process_tf_example


def input_fn(pattern_or_example,
             batch_size,
             word_vocab_size,
             max_pixel_pos=100,
             max_dom_pos=500,
             epoches=1,
             buffer_size=1,
             is_training=True):
  """Retrieves batches of data for training."""

  if isinstance(pattern_or_example, str):
    filepaths = tf.io.gfile.glob(pattern_or_example)
    dataset = tf.data.TFRecordDataset(filepaths)
  elif isinstance(pattern_or_example, tf.train.Example):
    dataset = tf.data.Dataset.from_tensors(
        pattern_or_example.SerializeToString())
  else:
    raise ValueError('Input must be a file path or tf.Example: %s' %
                     type(pattern_or_example))
  dataset = dataset.map(parse_tf_example)
  dataset = dataset.map(
      create_parser(word_vocab_size, max_pixel_pos, max_dom_pos, False))
  if is_training:
    dataset = dataset.shuffle(buffer_size=buffer_size)
  dataset = dataset.repeat(count=epoches)

  padding_info = [('screen_caption_token_ids', [1, None], 0),
                  ('appdesc_token_id', [1, None], 0),
                  ('developer_token_id', [None, MAX_TOKEN_PER_LABEL + 1], 0),
                  ('resource_token_id', [None, MAX_TOKEN_PER_LABEL + 1], 0),
                  ('label_flag', [None], 0),
                  ('obj_pixels', [None, 64, 64, 1], 0.0),
                  ('obj_type', [None], -1), ('obj_clickable', [None], 0),
                  ('obj_screen_pos', [None, 4], 0),
                  ('obj_dom_pos', [None, 3], 0), ('obj_visible', [None], 0),
                  ('attention_boxes', [1, 5, 4], 0.0),
                  ('attended_objects', [1, None], 0),
                  ('references', [None], tf.cast('', tf.string))]
  padded_shapes = {}
  padded_values = {}
  for (key, padding_shape, padding_value) in padding_info:
    padded_shapes[key] = padding_shape
    padded_values[key] = padding_value
  dataset = dataset.padded_batch(
      batch_size, padded_shapes=padded_shapes, padding_values=padded_values)
  dataset = dataset.prefetch(buffer_size=1024)
  return dataset
