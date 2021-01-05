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

"""Widget captioning input pipeline."""

from absl import flags
import tensorflow as tf

# Constants for embeddings.
PADDING = 0
EOS = 1
UKN = 2
START = 3

FLAGS = flags.FLAGS


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
    phrase = tf.pad(phrase, [[0, 11 - tf.shape(phrase)[-1]]])
    return phrase


def _select_phrases(dense_features):
  """Selects phrases from the workers."""
  with tf.variable_scope('select_phrases'):
    # Sample one phrase for each node.
    output_phrase = tf.map_fn(_produce_target_phrase,
                              dense_features['caption_token_id'])
    # Output shape: [N, seq_len]
    output_phrase = tf.reshape(output_phrase, [-1, 11])

    return output_phrase


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


def filter_empty_mturk():
  """Creates a filtering function."""

  def _has_mturk_captions(dense_features):
    """Check whether it has nodes with MTurk captions."""
    num_nodes = tf.shape(dense_features['label_flag'])[0]
    token_ids = tf.reshape(dense_features['caption_token_id'],
                           [num_nodes, 4, 10])
    nodes_with_annotations = tf.reduce_any(
        tf.reduce_any(tf.greater(token_ids, EOS), -1), -1)
    original_worker_node_mask = tf.equal(dense_features['label_flag'], 0)
    worker_node_mask = tf.logical_and(original_worker_node_mask,
                                      nodes_with_annotations)
    return tf.reduce_any(worker_node_mask)

  return _has_mturk_captions


def parse_tf_example(serialized_example):
  """Parses a single tf example."""
  keys_to_features = {
      'developer_token_id': tf.VarLenFeature(tf.int64),
      'resource_token_id': tf.VarLenFeature(tf.int64),
      'caption_token_id': tf.VarLenFeature(tf.int64),
      'caption_phrase_id': tf.VarLenFeature(tf.int64),
      'gold_caption': tf.VarLenFeature(tf.string),
      'clickable_seq': tf.VarLenFeature(tf.int64),
      'v_distance_seq': tf.VarLenFeature(tf.float32),
      'h_distance_seq': tf.VarLenFeature(tf.float32),
      'type_id_seq': tf.VarLenFeature(tf.int64),
      'cord_x_seq': tf.VarLenFeature(tf.float32),
      'cord_y_seq': tf.VarLenFeature(tf.float32),
      'visibility_to_user_seq': tf.VarLenFeature(tf.int64),
      'visibility_seq': tf.VarLenFeature(tf.int64),
      'label_flag': tf.VarLenFeature(tf.int64),  # 0: worker 1: developer
      'parent_child_seq': tf.VarLenFeature(tf.int64),
      'obj_img_mat': tf.VarLenFeature(tf.int64),
      'obj_dom_pos': tf.VarLenFeature(tf.int64),
      'is_leaf': tf.VarLenFeature(tf.int64),
  }
  parsed = tf.parse_single_example(serialized_example, keys_to_features)
  dense_features = {}
  for key in keys_to_features:
    if key in ['gold_caption']:
      default_value = ''
    else:
      default_value = 0
    dense_features[key] = tf.sparse_tensor_to_dense(
        parsed[key], default_value=default_value)

  return dense_features


def create_parser(word_vocab_size,
                  phrase_vocab_size,
                  max_pixel_pos=100,
                  max_dom_pos=500,
                  is_inference=False):
  """Creates a parser for tf.Example."""

  def process_tf_example(dense_features):
    """Parses a single tf example."""
    # Reshape the features

    num_ui_objects = tf.shape(dense_features['clickable_seq'])[0]

    dense_features['caption_token_id'] = tf.reshape(
        dense_features['caption_token_id'], [num_ui_objects, 4, 10])

    dense_features['developer_token_id'] = tf.reshape(
        dense_features['developer_token_id'], [num_ui_objects, 10])

    dense_features['resource_token_id'] = tf.reshape(
        dense_features['resource_token_id'], [num_ui_objects, 10])

    dense_features['caption_token_id'] = tf.where(
        tf.greater_equal(dense_features['caption_token_id'], word_vocab_size),
        tf.cast(
            tf.fill(tf.shape(dense_features['caption_token_id']), UKN),
            dtype=tf.int64), dense_features['caption_token_id'])

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

    dense_features['caption_phrase_id'] = tf.where(
        tf.greater_equal(dense_features['caption_phrase_id'],
                         phrase_vocab_size),
        tf.cast(
            tf.fill(tf.shape(dense_features['caption_phrase_id']), UKN),
            dtype=tf.int64), dense_features['caption_phrase_id'])

    dense_features['v_distance_seq'] = tf.reshape(
        dense_features['v_distance_seq'], [num_ui_objects, num_ui_objects],
        name='v_distance_seq')
    dense_features['h_distance_seq'] = tf.reshape(
        dense_features['h_distance_seq'], [num_ui_objects, num_ui_objects],
        name='h_distance_seq')
    dense_features['cord_x_seq'] = tf.reshape(
        dense_features['cord_x_seq'], [num_ui_objects, 2], name='cord_x_seq')
    dense_features['cord_y_seq'] = tf.reshape(
        dense_features['cord_y_seq'], [num_ui_objects, 2], name='cord_y_seq')
    dense_features['parent_child_seq'] = tf.reshape(
        tf.to_int32(dense_features['parent_child_seq']), [-1, num_ui_objects],
        name='parent_child_seq')

    dense_features['obj_dom_pos'] = tf.where(
        tf.greater_equal(dense_features['obj_dom_pos'], max_dom_pos),
        tf.cast(
            tf.fill(tf.shape(dense_features['obj_dom_pos']), 0),
            dtype=tf.int64), dense_features['obj_dom_pos'])

    feature_dict = {}
    if not is_inference:
      output_phrase = _select_phrases(dense_features)
      feature_dict['caption_token_id'] = output_phrase
      feature_dict['caption_phrase_id'] = dense_features['caption_phrase_id']

    feature_dict['developer_token_id'] = dense_features['developer_token_id']
    feature_dict['resource_token_id'] = dense_features['resource_token_id']
    feature_dict['reference'] = dense_features['gold_caption']
    # feature_dict['obj_str_seq'] = dense_features['obj_str_seq']

    feature_dict['label_flag'] = dense_features['label_flag']
    feature_dict['obj_is_leaf'] = dense_features['is_leaf']
    obj_pixels, obj_visible = _extract_image(dense_features, num_ui_objects)
    feature_dict['obj_pixels'] = obj_pixels
    feature_dict['obj_visible'] = obj_visible
    feature_dict['obj_screen_pos'] = tf.concat(
        [dense_features['cord_x_seq'], dense_features['cord_y_seq']], -1)
    feature_dict['obj_screen_pos'] = tf.to_int32(
        feature_dict['obj_screen_pos'] * (max_pixel_pos - 1))
    feature_dict['obj_clickable'] = dense_features['clickable_seq']
    feature_dict['obj_type'] = dense_features['type_id_seq']
    feature_dict['obj_adjacency'] = dense_features['parent_child_seq']
    feature_dict['obj_dom_pos'] = tf.reshape(dense_features['obj_dom_pos'],
                                             [num_ui_objects, 3])
    feature_dict['obj_is_padding'] = tf.zeros(tf.shape(num_ui_objects))
    for key in [
        'obj_adjacency',
        'obj_type',
        'obj_clickable',
        'obj_screen_pos',
        'obj_dom_pos',
        'developer_token_id',
        'resource_token_id',
    ]:
      # Add the auxiliary step dimension.
      feature_dict[key] = tf.expand_dims(feature_dict[key], 0)

    for key in [
        'caption_token_id',
        'caption_phrase_id',
        'developer_token_id',
        'resource_token_id',
        'label_flag',
        'obj_adjacency',
        'obj_type',
        'obj_clickable',
        'obj_visible',
        'obj_is_leaf',
        'icon_label',
        'obj_dom_pos',
        'obj_is_padding',
    ]:
      if key in feature_dict:
        feature_dict[key] = tf.cast(feature_dict[key], tf.int32)

    return feature_dict

  return process_tf_example


def input_fn(pattern,
             batch_size,
             word_vocab_size,
             phrase_vocab_size,
             max_pixel_pos=100,
             max_dom_pos=500,
             epoches=1,
             buffer_size=1):
  """Retrieves batches of data for training."""
  # files = tf.data.Dataset.list_files(pattern)
  filepaths = tf.io.gfile.glob(pattern)
  dataset = tf.data.TFRecordDataset([filepaths])
  dataset = dataset.map(
      parse_tf_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.filter(filter_empty_mturk())
  dataset = dataset.map(
      create_parser(word_vocab_size, phrase_vocab_size, max_pixel_pos,
                    max_dom_pos),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.shuffle(buffer_size=buffer_size)

  dataset = dataset.repeat(count=epoches)

  padding_value_int = tf.cast(0, tf.int32)
  anchor_padding_value_int = tf.cast(-1, tf.int32)
  padding_info = [
      ('caption_token_id', [None, 11], padding_value_int),
      ('caption_phrase_id', [None], padding_value_int),
      ('developer_token_id', [1, None, 10], padding_value_int),
      ('resource_token_id', [1, None, 10], padding_value_int),
      ('reference', [None], tf.cast('', tf.string)),
      ('label_flag', [None], anchor_padding_value_int),
      ('icon_label', [None], padding_value_int),
      ('icon_iou', [None], 0.0),
      ('obj_pixels', [None, 64, 64, 1], tf.cast(0, tf.float32)),
      ('obj_adjacency', [1, None, None], padding_value_int),
      ('obj_type', [1, None], anchor_padding_value_int),
      ('obj_clickable', [1, None], padding_value_int),
      ('obj_screen_pos', [1, None, 4], padding_value_int),
      ('obj_dom_pos', [1, None, 3], padding_value_int),
      ('obj_visible', [None], padding_value_int),
      ('obj_is_leaf', [None], padding_value_int),
      ('obj_is_padding', [None], 1),
  ]
  padded_shapes = {}
  padded_values = {}
  for (key, padding_shape, padding_value) in padding_info:
    padded_shapes[key] = padding_shape
    padded_values[key] = padding_value
  dataset = dataset.padded_batch(
      batch_size, padded_shapes=padded_shapes, padding_values=padded_values)
  dataset = dataset.prefetch(buffer_size=1024)
  return dataset
