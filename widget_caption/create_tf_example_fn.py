# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Functions to generate tf.Example for captioning model."""

import collections
import io
import itertools
import json
import re
from typing import Any, Dict, List, Generator

from absl import logging
import numpy as np
from PIL import Image
import tensorflow as tf


# Constant strings for word/phrase vocab.
UNKNOWN_WORD = '<UNK>'
PADDING_WORD = '<PAD>'
START_WORD = '<START>'
END_WORD = '<EOS>'

# Label flags to indicate a node is (or not) labeled by mturk worker.
_NODE_WITH_MTURK_CAPTION = 0
_NODE_WITHOUT_MTURK_CAPTION = 1

# Reserve 0 for padding.
_UI_OBJECT_TYPE = {
    'IMAGEVIEW': 1,
    'BUTTON': 2,
    'IMAGEBUTTON': 3,
    'VIEW': 4,
    'COMPOUNDBUTTON': 5,
    'CHECKBOX': 6,
    'RADIOBUTTON': 7,
    'FLOATINGACTIONBUTTON': 8,
    'TOGGLEBUTTON': 9,
    'SWITCH': 10,
    'UNKNOWN': 11
}

# Pattern to parse resource-id text (camel case or underscored).
_CAMEL_PATTERN = re.compile(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)')


def _get_node_type(ui_element):
  """Gets UI element type."""
  class_name = ui_element['class'].split('.')[-1]
  ancestors = ui_element['ancestors']
  for node_type in _UI_OBJECT_TYPE:
    if node_type == class_name.upper():
      return _UI_OBJECT_TYPE[node_type]
  for ancestor in ancestors:
    if ancestor.split('.')[-1].upper() in _UI_OBJECT_TYPE:
      return _UI_OBJECT_TYPE[ancestor.split('.')[-1].upper()]
  # As we use all the nodes from json tree, we might come across some nodes
  # which has node type not included in _ui_object_type. For those nodes,
  # we set their value type as 'UNKNOWN'
  return _UI_OBJECT_TYPE['UNKNOWN']


def _extract_features_from_node(node):
  """Casts UI element information to feature dictionary."""
  features = {}

  features['ui_obj_type_id'] = _get_node_type(node)
  features['label_flag'] = node['label_flag']

  if 'visibility' in node and node['visibility'] == 'visible':
    features['ui_obj_visibility'] = 1
  else:
    features['ui_obj_visibility'] = 0
  if 'visible-to-user' in node and node['visible-to-user']:
    features['ui_obj_visibility_to_user'] = 1
  else:
    features['ui_obj_visibility_to_user'] = 0

  features['ui_obj_clickable'] = 1 if node['clickable'] else 0

  # Scope into [0, 1].
  xmin = max(min(float(node['bounds'][0]) / 1440, 1), 0)
  xmax = max(min(float(node['bounds'][2]) / 1440, 1), 0)
  ymin = max(min(float(node['bounds'][1]) / 2560, 1), 0)
  ymax = max(min(float(node['bounds'][3]) / 2560, 1), 0)
  features['ui_obj_cord_x'] = [xmin, xmax]
  features['ui_obj_cord_y'] = [ymin, ymax]

  features['image/object/bbox/xmin'] = xmin
  features['image/object/bbox/ymin'] = ymin
  features['image/object/bbox/xmax'] = xmax
  features['image/object/bbox/ymax'] = ymax

  return features


def _adjust_bounds(width, height, bounds):
  """Adjusts image bounds w/ ratio of actual width/height and full screen."""
  width_ratio = width / 1440.
  height_ratio = height / 2560.
  top_x, top_y, bottom_x, bottom_y = bounds

  return [
      int(top_x * width_ratio),
      int(top_y * height_ratio),
      int(bottom_x * width_ratio),
      int(bottom_y * height_ratio)
  ]


def _extract_pixels(image, bounds):
  """Extracts image pixels and resizes it to 64*64*3.

  Args:
    image: A PIL.Image instance.
    bounds: <x1, y1, x2, y2> coordinates of the bounding box.

  Returns:
    A flatten list of pixel values (each pixel represented by 3 values).
  """
  try:
    cropped = image.crop(bounds)
    resized = cropped.resize((64, 64))
    pixels = resized.getdata()
  except Exception as e:  # pylint: disable=broad-except
    logging.warning('Error: %s', e)
    # Use all zero for image if exception.
    return [0] * (64 * 64 * 3)
  flatten = []
  for p in pixels:
    # PNG has 4 bands, JPEG has 3 bands, for PNG we use the first 3.
    flatten += p[:3]
  return flatten


def _extract_node_text(node):
  """Extracts `text` and `content-desc` attribute for a node."""
  text = node.get('text')
  content = node.get('content-desc', [])
  all_text = [text, content] if isinstance(content, str) else [text] + content
  # Remove None or string with only space.
  all_text = [t for t in all_text if t and t.strip()]
  return all_text


def _tokenize_node_text(node, tokenizer):
  """Tokenizes `text` and `content-desc` attribute for a node."""
  all_text = _extract_node_text(node)
  all_tokens = [tokenizer.tokenize(text) for text in all_text]
  # Remove empty token list.
  all_tokens = [t for t in all_tokens if t]
  return all_tokens


def _tokenize_resource_id_text(node):
  """Tokenize `resource-id` attribute for a node."""
  text = node.get('resource-id', '').strip()
  tokens = []
  if text:
    elements = text.split('/')
    assert len(elements) == 2
    resource = elements[1]
    # Tokenize it using camel pattern.
    tokens = _CAMEL_PATTERN.findall(resource)
    tokens = [t.lower() for t in tokens]
  return tokens


def _truncate_and_pad_token_ids(token_ids, max_length):
  """Truncates or pads the token id list to max length."""
  token_ids = token_ids[:max_length]
  padding_size = max_length - len(token_ids)
  if padding_size > 0:
    token_ids += [0] * padding_size
  return token_ids


def _extract_token(node, tokenizer):
  """Extracts tokens for a node."""
  all_developer_tokens = _tokenize_node_text(node, tokenizer)
  node['developer_token'] = []
  if all_developer_tokens:
    # Developer tokens, only keep the first token list.
    node['developer_token'] = all_developer_tokens[0]
    node['all_developer_token'] = all_developer_tokens

  # Tokens from `resource-id` attribute.
  resource_tokens = _tokenize_resource_id_text(node)
  node['resource_token'] = resource_tokens

  # MTurk caption tokens and phrase.
  captions = node.get('_caption_captions', [])
  caption_tokens = [c.split(' ') for c in captions]
  node['caption_token'] = caption_tokens


def _create_token_id(node, word_vocab, max_token_per_label, max_label_per_node):
  """Creates token ids for various tokens in the node."""
  # Developer token ids.
  developer_tokens = node.get('developer_token', [])
  developer_token_ids = [
      word_vocab[t] if t in word_vocab else word_vocab[UNKNOWN_WORD]
      for t in developer_tokens
  ]
  developer_token_ids = _truncate_and_pad_token_ids(developer_token_ids,
                                                    max_token_per_label)

  # Resource token ids.
  resource_tokens = node.get('resource_token', [])
  resource_token_ids = [
      word_vocab[t] if t in word_vocab else word_vocab[UNKNOWN_WORD]
      for t in resource_tokens
  ]
  resource_token_ids = _truncate_and_pad_token_ids(resource_token_ids,
                                                   max_token_per_label)

  # MTurk caption token ids.
  all_caption_token_ids = []
  caption_tokens = node.get('caption_token', [])
  for tokens in caption_tokens:
    # We don't use UNKNOWN for the target caption.
    token_ids = [word_vocab[t] for t in tokens if t in word_vocab]
    token_ids = _truncate_and_pad_token_ids(token_ids, max_token_per_label)
    all_caption_token_ids.append(token_ids)

  # Pad at node level.
  all_caption_token_ids = all_caption_token_ids[:max_label_per_node]
  padding_size = max_label_per_node - len(all_caption_token_ids)
  if padding_size > 0:
    token_padding = [[0] * max_token_per_label] * padding_size
    all_caption_token_ids += token_padding

  node['developer_token_id'] = developer_token_ids
  node['resource_token_id'] = resource_token_ids
  node['caption_token_id'] = all_caption_token_ids

  # If no valid tokens, set label_flag to 1 (node w/o mturk captions).
  token_sum = sum(sum(token_ids) for token_ids in all_caption_token_ids)
  if token_sum == 0:
    node['label_flag'] = _NODE_WITHOUT_MTURK_CAPTION
    node['caption_token_id'] = [[0] * max_token_per_label] * max_label_per_node


def _get_features_from_all_nodes(all_nodes, image):
  """Gets all feature dictionary from xml/json file_path and the image."""
  image_width, image_height = image.size
  all_features = collections.defaultdict(list)

  image_bytes = io.BytesIO()
  image.save(image_bytes, format='JPEG')
  image_bytes = image_bytes.getvalue()
  all_features['image/encoded'] = image_bytes

  for node in all_nodes:
    # node['bounds'] is based on [1440, 2560], need to resacle it to the actual
    # screenshot image szie.
    bounds = _adjust_bounds(image_width, image_height, node['bounds'])
    pixels = _extract_pixels(image, bounds)
    all_features['obj_dom_pos'] += [
        node['_caption_depth'], node['_caption_preorder_id'],
        node['_caption_postorder_id']
    ]
    all_features['obj_img_mat'] += pixels
    all_features['node_id'].append(node['_caption_node_id'].encode())
    all_features['is_leaf'].append(int(node['_is_leaf_node']))
    features = _extract_features_from_node(node)
    all_features['type_id_seq'].append(features['ui_obj_type_id'])
    all_features['visibility_seq'].append(features['ui_obj_visibility'])
    all_features['visibility_to_user_seq'].append(
        features['ui_obj_visibility_to_user'])
    all_features['clickable_seq'].append(features['ui_obj_clickable'])
    all_features['cord_x_seq'].append(features['ui_obj_cord_x'])
    all_features['cord_y_seq'].append(features['ui_obj_cord_y'])
    all_features['label_flag'].append(features['label_flag'])

    # Text features.
    all_features['developer_token_id'].append(node['developer_token_id'])
    all_features['resource_token_id'].append(node['resource_token_id'])
    all_features['caption_token_id'] += list(
        itertools.chain.from_iterable(node['caption_token_id']))

    if '_caption_captions' in node:
      gold_caption = '|'.join(node['_caption_captions']).encode('utf8')
      all_features['gold_caption'].append(gold_caption)
    else:
      all_features['gold_caption'].append(b'')

  return all_features


def _load_all_node(root, node_id_to_labels):
  """Loop through all the nodes to add meta info and captions."""
  all_nodes = []
  for node in iterate_nodes(root, False):
    node_id = node['_caption_node_id']
    if '_caption_node_type' in node:
      node['_caption_node_type'] = str(node['_caption_node_type'])

    if node_id in node_id_to_labels:
      # Add captions created by MTurk labeler.
      node['label_flag'] = _NODE_WITH_MTURK_CAPTION
      node['_caption_captions'] = node_id_to_labels[node_id]
    else:
      # Other nodes are used as context.
      node['label_flag'] = _NODE_WITHOUT_MTURK_CAPTION

    all_nodes.append(node)

  return all_nodes


def features_to_tf_example(features):
  """Converts feature dictionary into tf.Example protobuf.

  This function only supports to convert np.int and np.float array.

  Args:
    features: A feature dictionary. Keys are field names, values are np array.

  Returns:
    A tf.Example protobuf.

  Raises:
    ValueError: Feature dictionary's value field is not supported type.

  """
  new_features = {}
  for k, v in features.items():
    if not isinstance(v, np.ndarray):
      raise ValueError('Value field: %s is not numpy array' % str((k, v)))
    v = v.flatten()
    if np.issubdtype(v.dtype.type, np.string_):
      new_features[k] = tf.train.Feature(bytes_list=tf.train.BytesList(value=v))
    elif np.issubdtype(v.dtype.type, np.integer):
      new_features[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=v))
    elif np.issubdtype(v.dtype.type, np.floating):
      new_features[k] = tf.train.Feature(float_list=tf.train.FloatList(value=v))
    else:
      raise ValueError('Value for %s is not a recognized type; v: %s type: %s' %
                       (k, str(v[0]), str(type(v[0]))))
  return tf.train.Example(features=tf.train.Features(feature=new_features))


def _pre_order_travesal(root):
  """Traverses the tree in pre-order and assigns traversal index to nodes."""
  # Index 0 is reserved for padding when generating training data.
  index = 1
  nodes = [root]
  while nodes:
    # Get the last node in the list (order would be: root, left, right).
    node = nodes.pop()
    node['_caption_preorder_id'] = index
    index += 1

    children = node.get('children', [])  # type: List[Dict[str, Any]]
    # Skip None child as we don't use the traversal id to fetch child from list.
    children = [c for c in children if c]

    # Append the children from right to left.
    children.reverse()
    for child in children:
      nodes.append(child)


def _post_order_traversal(root):
  """Traverses the tree in post-order and assigns traversal index to nodes."""
  # Index 0 is reserved for padding when generating training data.
  index = 1
  nodes = [root]
  traverse = []

  while nodes:
    node = nodes.pop()
    # Build the post-order traversal.
    traverse.append(node)
    children = node.get('children', [])  # type: List[Dict[str, Any]]
    # Skip None child as we don't use the traversal id to fetch child from list.
    children = [c for c in children if c]
    children.reverse()
    for child in children:
      nodes.append(child)

  traverse.reverse()
  for node in traverse:
    # Order would be left, right, root.
    node['_caption_postorder_id'] = index
    index += 1


def iterate_nodes(
    root,
    only_leaf_node):
  """Iterates through nodes in the view hierarchy.

  Traverse the view tree in a depth-first manner.

  Args:
    root: the root node of the view.
    only_leaf_node: If True, only yield leaf nodes, otherwise yield all nodes.

  Yields:
    Leaf nodes with annotations.
  """
  # Traverse the tree in pre-order and post-order to get index ids for the
  # nodes. The index ids will be attached to each node and used as features to
  # indicate the node structural position in the tree for model training.
  _pre_order_travesal(root)
  _post_order_traversal(root)

  root['_caption_node_id'] = '0'
  node_list = [root]

  for node in node_list:
    children = node.get('children', [])

    if not children:
      # If it's a leaf node, annotate it.
      node['_is_leaf_node'] = True
    else:
      node['_is_leaf_node'] = False

    # Add children nodes to the list.
    for index, child in enumerate(children):
      if not child:
        # We don't remove None child beforehand as we want to keep the index
        # unchanged, so that we can use it to fetch a specific child in the list
        # directly.
        continue
      child['_caption_node_id'] = '{}.{}'.format(node['_caption_node_id'],
                                                 index)
      node_list.append(child)

    if only_leaf_node and not node['_is_leaf_node']:
      # Skip intermediate nodes if only leaf nodes are wanted.
      continue

    # Create node depth, defined as its depth in the json tree.
    node_id = node['_caption_node_id']  # type: str
    node['_caption_depth'] = len(node_id.split('.'))

    yield node


def create_tf_example(file_prefix, node_id_to_labels, tokenizer, word_vocab,
                      max_token_per_label, max_label_per_node):
  """Creates tf.Example for widget captioning model.

  Args:
    file_prefix: A path prefix for the json and jpg/png file.
    node_id_to_labels: A Dict from node id to its mturk labels.
    tokenizer: A tokenizer for tokenizing captions.
    word_vocab: Word vocab.
    max_token_per_label: Max number of tokens for each caption/label.
    max_label_per_node: Max number of captions/labels for each node.

  Returns:
    A tf.Example for the screen.
  """
  # Load view hierarchy.
  json_path = file_prefix + '.json'
  with tf.io.gfile.GFile(json_path) as f:
    view_hierarchy = json.load(f)

  root = view_hierarchy['activity']['root']
  all_nodes = _load_all_node(root, node_id_to_labels)

  for node in all_nodes:
    _extract_token(node, tokenizer)
    _create_token_id(node, word_vocab, max_token_per_label, max_label_per_node)

  # Load screenshot image, could be either JPEG or PNG image.
  jpg_path = file_prefix + '.jpg'
  png_path = file_prefix + '.png'
  if tf.io.gfile.exists(jpg_path):
    image_path = jpg_path
  else:
    image_path = png_path

  with tf.io.gfile.GFile(image_path, 'rb') as f:
    image = Image.open(f)
    image = image.convert('RGB')
    features = _get_features_from_all_nodes(all_nodes, image)
    features['file_prefix'] = [file_prefix.encode()]

  for key in features:
    features[key] = np.array(features[key])

  tf_example = features_to_tf_example(features)

  return tf_example


def extract_token(json_path, node_id_to_labels, tokenizer):
  """Extracts tokens from the view hierarchy and mturk labels.

  Args:
    json_path: File path to the view hierarchy json.
    node_id_to_labels: A Dict from node id to its mturk labels.
    tokenizer: A tokenizer for tokenizing captions.

  Yields:
    A tuple of <type, text>. Type can be caption_token, developer_token
  """
  # Load view hierarchy.
  with tf.io.gfile.GFile(json_path) as f:
    view_hierarchy = json.load(f)

  root = view_hierarchy['activity']['root']
  all_nodes = _load_all_node(root, node_id_to_labels)

  for node in all_nodes:
    _extract_token(node, tokenizer)

    # For worker text, we emit both token and phrase.
    for tokens in node['caption_token']:
      for token in tokens:
        yield '{}\t{}'.format('token', token)

    # For developer text, we only emit tokens.
    for token in node['developer_token']:
      yield '{}\t{}'.format('token', token)

    # Tokens for the resource-id attribute.
    for token in node['resource_token']:
      yield '{}\t{}'.format('token', token)


def extract_raw_text(json_path, node_id_to_labels):
  """Extracts raw text from the view hierarchy and mturk labels.

  Args:
    json_path: File path to the view hierarchy json.
    node_id_to_labels: A Dict from node id to its mturk labels.

  Yields:
    Raw text from view hierarchy and MTurk labels.
  """
  # Load view hierarchy.
  with tf.io.gfile.GFile(json_path) as f:
    view_hierarchy = json.load(f)

  root = view_hierarchy['activity']['root']
  all_nodes = _load_all_node(root, node_id_to_labels)

  for node in all_nodes:
    # Yield pre-existing text.
    for text in _extract_node_text(node):
      yield text

    # Yield MTurk captions.
    captions = node.get('_caption_captions', [])
    for caption in captions:
      yield caption
