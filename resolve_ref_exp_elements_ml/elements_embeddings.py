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

"""Machine learning models to calculate embeddings for screen elements.

These models are meant to be included as part of another model.
This model is not pretrained (except for the text_model)
and will be trained with the rest of the model.
"""

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub

REF_EXP_ID = 'ref_exp'
ELEMENTS_TEXT_ID = 'elements_text'
ELEMENTS_BOX_ID = 'elements_box'
ELEMENTS_EXIST_ID = 'elements_exists'
ELEMENTS_NEIGHBORS_ID = 'elements_neighbors'
ELEMENTS_REF_MATCH_ID = 'elements_ref_match'


def text_model(text_feature, pretrained_text_enc_name):
  """Builds the part of the model that processes the text.

  Args:
    text_feature: A batch tf.string to run through the model. Size: [batch_size]
    pretrained_text_enc_name: The text encoder to use.

  Returns:
    Encoding of the text. Size: [batch_size, text_embed_size] The
    text_embed_size depends on the specific tf hub model.
  """
  with tf.variable_scope('text_model'):
    text_encode_result = hub.Module(pretrained_text_enc_name)(text_feature)

    return text_encode_result


def undo_mask(x, mask, pad_val=0.0):
  """Converts the output of boolean_mask to the original input dimensions.

  The boolean_mask is usually used to condense items from multiple batches into
  one large 'batch' for faster processing. This function is used to convert
  back.
  Args:
    x: The input to reshape.
    mask: The mask used in boolean_mask.
    pad_val: value to pad with.

  Returns:
    x reshaped and padded.
  """
  with tf.variable_scope('undo_mask'):
    flat_x = tf.reshape(x, [-1])
    x_shape = tf.shape(x)[1:]
    expanded_mask = tf.tile(
        tf.reshape(
            mask, tf.concat([[-1, tf.shape(mask)[1]],
                             tf.ones_like(x_shape)], 0)),
        tf.concat([[1, 1], x_shape], 0))
    flat_mask = tf.reshape(expanded_mask, [-1])
    start_indices = tf.range(tf.shape(flat_mask)[0])
    condition_indices = tf.dynamic_partition(start_indices,
                                             tf.cast(flat_mask, tf.int32), 2)
    stitched = tf.dynamic_stitch(condition_indices, [
        tf.ones_like(condition_indices[0], tf.float32) * pad_val,
        tf.reshape(flat_x, [-1])
    ])
    final_shape = tf.shape(mask)
    out_shape = tf.concat([[final_shape[0], final_shape[1]], x_shape], 0)
    return tf.reshape(stitched, out_shape)


def tile_ref_enc_to_elements(ref_enc, elements_mask):
  """Utility to tile the ref_enc to the same shape as the elements."""
  with tf.variable_scope('tile_ref_enc_to_elements'):
    orig_shape = tf.shape(ref_enc)
    orig_shape_static = ref_enc.get_shape().as_list()
    ref_enc = tf.tile(
        tf.reshape(ref_enc, [orig_shape[0], 1, orig_shape[1]]),
        [1, tf.shape(elements_mask)[1], 1])
    ref_enc = tf.boolean_mask(ref_enc, elements_mask)
    ref_enc = tf.reshape(ref_enc, [-1, orig_shape_static[1]])
    return ref_enc


def tile_to_image(x, image_size):
  x = tf.reshape(x, [-1, 1, 1, x.get_shape()[1]])
  x = tf.tile(x, [1, image_size[0], image_size[1], 1])
  return x


def get_filled_rect(box, fill_vals, output_size, mode):
  """Returns a feature map with the values in the box filled in.

  This can either work with a single box or a batch of boxes.

  Args:
    box: The box to fill, in x,y,width,height format normalized
      between 0 and 1. shape: [num_boxes, 4]
    fill_vals: The vector to tile over each bounding box. This could be the
      embedding for the element.
    output_size: The length and width of the output. Assumes length and width
      are the same.
    mode: Method to compute values in the box. 'step': Pixels change immediately
      at box border. 'cont': Pixels change gradually at box border and increase
      towards the center.

  Returns:
    The tensor with the boxes filled.
  """

  with tf.variable_scope('get_filled_rect'):
    axis = tf.to_float(tf.range(output_size))

    disp_box = box
    disp_box *= tf.to_float(output_size)
    disp_box += [-1.5, -1.5, 3.0, 3.0]
    disp_box *= [1, 1, .5, .5]

    if mode == 'step':
      x_vals = tf.nn.relu(
          tf.nn.relu(axis - disp_box[:, 0:1]) / 2 -
          tf.nn.relu(axis - disp_box[:, 0:1] - disp_box[:, 2:3]))
      y_vals = tf.nn.relu(
          tf.nn.relu(axis - disp_box[:, 1:2]) / 2 -
          tf.nn.relu(axis - disp_box[:, 1:2] - disp_box[:, 3:4]))
    else:
      x_vals = tf.nn.relu(-tf.abs(axis - disp_box[:, 0:1] - disp_box[:, 2:3]) /
                          (disp_box[:, 2:3]) + 1)
      y_vals = tf.nn.relu(-tf.abs(axis - disp_box[:, 1:2] - disp_box[:, 3:4]) /
                          (disp_box[:, 3:4]) + 1)

    x_vals = tf.expand_dims(x_vals, 1)
    y_vals = tf.expand_dims(y_vals, 2)

    filled_rect = x_vals * y_vals
    if mode == 'step':
      filled_rect = tf.minimum(filled_rect, 1.0)

    fill_vals = tf.reshape(
        fill_vals, [tf.shape(fill_vals)[0], 1, 1,
                    tf.shape(fill_vals)[1]])
    filled_rect = tf.expand_dims(filled_rect, 3)
    filled_rect *= fill_vals

  return filled_rect


def atten_softmax(atten_mask, elements_mask):
  """Calculates the softmax between the values in each batch."""
  atten_mask = undo_mask(atten_mask, elements_mask, np.finfo(np.float32).min)

  atten_mask = tf.nn.softmax(atten_mask)

  atten_mask = tf.boolean_mask(atten_mask, elements_mask)

  return atten_mask


def atten_metric(elements_enc_attend, ref_enc_attend, elements_mask,
                 do_softmax):
  """Computes similarity metric to be used with attention."""
  with tf.variable_scope('atten_metric'):
    ref_enc_attend = tile_ref_enc_to_elements(ref_enc_attend, elements_mask)

    atten_mask = tf.multiply(elements_enc_attend, ref_enc_attend)
    atten_mask = tf.reduce_sum(atten_mask, axis=1)

    if do_softmax:
      atten_mask = atten_softmax(atten_mask, elements_mask)

    return atten_mask


def attention(query, attend_in, single_dot_in, elements_mask, do_softmax,
              attention_method, flags):
  """Returns the attention mask using the method described by attention_method.

  Args:
    query: Query vector. Shape: [batch_size, query_size]
    attend_in: Values for each item to use for attention. [batch_size *
      elements_per_query, attend_size]
    single_dot_in: Values for each item to use for attention in single dot mode.
      [batch_size * elements_per_query, single_dot_attend_size]
      single_dot_attend_size must be greater than query_size
    elements_mask: Mask for what elements items exist in the input.
    do_softmax: Whether to put the output through softmax.
    attention_method: The attention method to use.
    flags: The input Flags. (Currently unused)

  Returns:
    The attention mask.
  """
  del flags

  elements_item_size = attend_in.shape[1]
  # Use different weights for DNN ontop of Ref Exp, and Elements
  if 'sepDotAtten' == attention_method:
    elements_enc_attend = tf.layers.dense(attend_in, elements_item_size)
    query_attend = tf.layers.dense(query, elements_item_size)

    attention_mask = atten_metric(elements_enc_attend, query_attend,
                                  elements_mask, do_softmax)

  # Use the same weights for DNN ontop of Ref Exp, and Elements
  if 'singDotAtten' == attention_method:
    elements_enc_attend = single_dot_in

    query_attend = tf.concat([
        query,
        tf.zeros([
            tf.shape(query)[0],
            tf.shape(single_dot_in)[1] - tf.shape(query)[1]
        ])
    ], 1)

    # Concat along batch dim, so same weights used for each.
    all_attend = tf.concat([elements_enc_attend, query_attend], 0)
    all_attend = tf.layers.dense(all_attend, elements_item_size, tf.nn.relu)
    all_attend = tf.layers.dense(all_attend, elements_item_size)

    elements_enc_attend, query_attend = tf.split(
        all_attend,
        [tf.shape(elements_enc_attend)[0],
         tf.shape(query_attend)[0]])

    attention_mask = atten_metric(elements_enc_attend, query_attend,
                                  elements_mask, do_softmax)

  # Combine Ref Exp, and Elements before input to DNN
  if 'combAtten' == attention_method:
    query_tile = tile_ref_enc_to_elements(query, elements_mask)
    attention_mask = tf.concat([attend_in, query_tile], 1)
    attention_mask = tf.layers.dense(attention_mask, elements_item_size,
                                     tf.nn.relu)
    attention_mask = tf.layers.dense(attention_mask, 1)
    attention_mask = tf.squeeze(attention_mask, 1)

    if do_softmax:
      attention_mask = atten_softmax(attention_mask, elements_mask)

  tf.summary.histogram('attention_mask', attention_mask)

  return attention_mask


def filter_none(lst):
  """Removes None elements from the list."""
  lst = [el for el in lst if el is not None]
  return lst


def calc_neighbor_embed(elements_neighbors, elements_enc, elements_mask):
  """Calculates the sum of the embeddings of neighboring elements."""
  with tf.variable_scope('calc_neighbor_embed'):
    elements_enc_orig_shape = elements_enc.get_shape().as_list()

    elements_enc = undo_mask(elements_enc, elements_mask)

    elements_enc_shape = tf.shape(elements_enc)
    elements_enc_expand = tf.tile(elements_enc, [1, elements_enc_shape[1], 1])
    elements_enc_expand = tf.reshape(elements_enc_expand, [
        -1, elements_enc_shape[1], elements_enc_shape[1], elements_enc_shape[2]
    ])

    elements_neighbors = tf.cast(
        tf.expand_dims(elements_neighbors, 3), tf.float32)

    neighbor_embed = elements_enc_expand * elements_neighbors
    neighbor_embed = tf.reduce_mean(neighbor_embed, axis=2)

    neighbor_embed = tf.boolean_mask(neighbor_embed, elements_mask)

    neighbor_embed.set_shape(elements_enc_orig_shape)

    return neighbor_embed


def elements_model(elements_texts_enc, feature_map, output_size, elements_mask,
                   ref_enc, flags):
  """The part of the model that processes the elements text and boxes.

  This assumes that the text has already been preprocessed with the text_model.
  Even if you are only using the elements and not the referring expression, you
  should probably use the ref_elements_model since that also handles
  preprocessing with the text_model.

  Args:
    elements_texts_enc: The elements text encoded by the text_model. Size:
      [batch_size * elements_per_query, text_embed_size]
    feature_map: Features used by the model.
    output_size: Desired output size of the encoding. Format: [length, width,
      depth]
    elements_mask: Mask for what elements items exist in the input.
    ref_enc: The referring expression encoded by the text_model. [batch_size,
      text_embed_size]
    flags: The input Flags.

  Returns:
    The encoding of the elements data.
  """

  with tf.variable_scope('elements_model'):
    elements_item_size = output_size[2]

    if flags.use_elements_boxes:
      elements_boxes = tf.identity(feature_map[ELEMENTS_BOX_ID],
                                   ELEMENTS_BOX_ID)
      flat_elements_boxes = tf.boolean_mask(elements_boxes, elements_mask)
    else:
      elements_boxes = None
      flat_elements_boxes = None

    if ref_enc is not None:
      ref_enc_tile = tile_ref_enc_to_elements(ref_enc, elements_mask)

    elements_ref_match_enc = None
    if flags.use_elements_ref_match:
      elements_ref_match = tf.identity(feature_map[ELEMENTS_REF_MATCH_ID],
                                       ELEMENTS_REF_MATCH_ID)
      tf.summary.text('elements_ref_match', elements_ref_match)
      flat_elements_ref_match = tf.boolean_mask(elements_ref_match,
                                                elements_mask)

      elements_ref_match_enc = text_model(
          flat_elements_ref_match, flags.pretrained_elements_ref_match_model)

    # For combinding the element with the refering expression.
    if flags.merge_ref_elements_method == 'combine' and (ref_enc is not None):
      elements_enc = tf.concat(
          filter_none([
              elements_texts_enc, flat_elements_boxes, ref_enc_tile,
              elements_ref_match_enc
          ]), 1)
      elements_enc = tf.layers.dense(elements_enc, elements_item_size * 2,
                                     tf.nn.relu)
    else:
      # Paper results
      elements_enc = tf.concat(
          filter_none(
              [elements_texts_enc, flat_elements_boxes,
               elements_ref_match_enc]), 1)
      elements_enc = tf.layers.dense(elements_enc, elements_item_size,
                                     tf.nn.relu)

    neighbor_embed = None
    if flags.use_elements_neighbors:
      neighbor_embed = calc_neighbor_embed(feature_map[ELEMENTS_NEIGHBORS_ID],
                                           elements_enc, elements_mask)

    elements_enc = tf.concat(filter_none([elements_enc, neighbor_embed]), 1)

    elements_enc = tf.layers.dense(elements_enc, elements_item_size, tf.nn.relu)

    attend_in = elements_enc

    # "DNN"
    elements_enc = tf.nn.dropout(elements_enc, flags.elements_keep_prob)
    elements_enc = tf.layers.dense(elements_enc, elements_item_size, tf.nn.relu)
    elements_enc = tf.nn.dropout(elements_enc, flags.elements_keep_prob)
    elements_enc = tf.layers.dense(elements_enc, elements_item_size)

    elements_enc_pre_atten = elements_enc

    if 'Atten' in flags.merge_ref_elements_method and (ref_enc is not None):
      with tf.variable_scope('attention'):
        if elements_texts_enc is None:
          # Prepad with 0s so the box embedding won't overlap with the ref_enc.
          single_dot_concat = tf.zeros([
              tf.shape(flat_elements_boxes)[0],
              ref_enc.get_shape().as_list()[1]
          ])
        else:
          single_dot_concat = elements_texts_enc
        single_dot_in = tf.concat(
            filter_none([
                single_dot_concat,
                flat_elements_boxes,
                neighbor_embed,
                elements_ref_match_enc,
            ]), 1)
        single_dot_in = tf.concat(
            [single_dot_in,
             tf.ones([tf.shape(single_dot_in)[0], 1])], 1)

        attention_mask = attention(ref_enc, attend_in, single_dot_in,
                                   elements_mask, True,
                                   flags.merge_ref_elements_method, flags)

        attention_mask = tf.expand_dims(attention_mask, 1)

        elements_enc *= attention_mask

    # Projects the element embeddings into a 2d feature map.
    if flags.elements_proj_mode != 'tile':
      with tf.variable_scope('elements_proj'):
        # Projects the elements text onto the image feature map
        # on the corresponding bounding boxes.

        assert_op = tf.Assert(
            tf.equal(output_size[0], output_size[1]), [
                'Assumes height and width are the same.',
                feature_map[ELEMENTS_BOX_ID]
            ])
        with tf.control_dependencies([assert_op]):
          if flags.proj_elements_memop:
            # Iterate through all bounding boxes and embeddings to create
            # embedded bounding boxes and sum to result vector iterately
            elements_enc = undo_mask(elements_enc, elements_mask)

            fold_elms = tf.transpose(
                tf.concat([elements_enc, elements_boxes], 2), [1, 0, 2])

            initializer = tf.zeros([tf.shape(elements_mask)[0]] + output_size)

            def fold_fn(total, fold_elm):
              elements_enc_boxes = tf.split(
                  fold_elm,
                  [tf.shape(elements_enc)[2],
                   tf.shape(elements_boxes)[2]], 1)
              return total + get_filled_rect(
                  elements_enc_boxes[1], elements_enc_boxes[0], output_size[0],
                  flags.elements_proj_mode)

            elements_enc = tf.foldl(
                fold_fn,
                fold_elms,
                initializer=initializer,
                swap_memory=True,
                parallel_iterations=2)

          else:
            # Create embedding of all bb then reduce sum
            elements_enc = get_filled_rect(flat_elements_boxes, elements_enc,
                                           output_size[0],
                                           flags.elements_proj_mode)

            elements_enc = undo_mask(elements_enc, elements_mask)

            elements_enc = tf.reduce_sum(elements_enc, axis=1)

        # Turn sum into average.
        mask_sum = tf.cast(
            tf.reduce_sum(tf.cast(elements_mask, tf.uint8), 1), tf.float32)
        mask_sum = tf.reshape(mask_sum, [-1, 1, 1, 1])
        mask_sum = tf.where(
            tf.equal(mask_sum, 0), tf.ones_like(mask_sum), mask_sum)
        elements_enc /= mask_sum
        tf.summary.histogram('elements_enc', elements_enc)

        elements_enc_for_disp = tf.reduce_mean(elements_enc, 3, keepdims=True)
        tf.summary.image('elements_enc_for_disp', elements_enc_for_disp, 4)
    else:
      # Undo the mask for feature mapping
      sequence_elements_enc = undo_mask(elements_enc, elements_mask)

      elements_enc = tf.reduce_mean(sequence_elements_enc, axis=1)
      tf.summary.histogram('elements_enc', elements_enc)

      if flags.elements_3d_output:
        elements_enc = tile_to_image(elements_enc, output_size)

    if flags.elements_3d_output:
      elements_enc.set_shape(
          [None, output_size[0], output_size[1], elements_item_size])

    # Last CNN layer of elements model
    if flags.elements_3d_output and flags.elements_cnn:
      elements_enc = tf.layers.conv2d(
          elements_enc,
          elements_enc.shape[3],
          3,
          padding='SAME',
          activation=tf.nn.relu,
          strides=1)
      elements_enc = tf.nn.dropout(elements_enc, flags.elements_keep_prob)
      elements_enc = tf.layers.conv2d(
          elements_enc,
          elements_enc.shape[3],
          3,
          padding='SAME',
          activation=None,
          strides=1)

    return elements_enc, elements_enc_pre_atten


def ref_elements_model(feature_map, output_size, flags):
  """Calculates an embedding for the referring expression and screen elements.

  The input is defined in the feature_map, with a referring expression and also
  text and boxes for each screen element.
  The certain inputs can be ignored by the model by setting their corresponding
  use_ flags to false.

  The model produces a 3d output (length, width, depth) when:
  flags.elements_proj_mode != 'tile' or flags.elements_3d_output
  and a 1d output (embedding_size) otherwize.

  Args:
    feature_map: Dict of features used by the model. See
      research/socrates/vis_va_model/seg_model/model_input.py for how to
      construct from tf.Examples,
      REF_EXP_ID The referring expression as tf.string. Shape: [batch_size]
      ELEMENTS_TEXT_ID The text for each element as tf.string. Shape:
        [batch_size, num_elements] (If some batches have more elements, pad with
        empty string)
      ELEMENTS_BOX_ID The box for each element as tf.string. Shape:
        [batch_size, 4]. The last dimension is [x,y,width,height]. (If some
        batches have more elements, pad with empty 0s)
      ELEMENTS_EXIST_ID 1 for if an element is present and 0 otherwise. Used
        since the batches can have different numbers of elements.
    output_size: Desired output size of the encoding. Format: [length, width,
      depth] if the flags specify a 1d output, only the depth is considered.
    flags: The input Flags See
      research/socrates/screen_elements_embedding_model/notebook.ipynb for the
      optimal values for these.
      elements_3d_output Forces a 3d output even when elements_proj_mode==tile.
      use_ref_exp Whether or not to use the referring expression in the model.
      use_elements_texts Whether or not to use the elements information in the
        model. Crash if this is true when use_elements_boxes is false.
      use_elements_boxes:Whether or not to use the elements boxes in the model.
      merge_ref_elements_method options: '': Don't merge in elements model.
        'combine' Concatenate the representations and feed through a DNN.
        'singDotAtten' Use the same DNN to calculate the
        representations of the items and expression to multiply.
        'sepDotAtten' Use separate networks to calculate the
        representations. 'combAtten': Use a network to directly output
        multiply values.
      elements_proj_mode How to project the elements information onto the image
        feature.' 'tile' blindly tile the feature over the image." 'step': Tile
        only in the elements bounding box locations." 'cont': Tile the values in
        bounding box locations and" increase magnitude near the center of the
        box.'
      pretrained_text_enc_name - The text model to use.
      elements_keep_prob - Controls dropout.
      elements_cnn - True to use a CNN after the elements are processed.
      elements_enc_size - The size of the output encoding. -1
      to use the output_size.
     proj_elements_memop Reduces elements projection mem by using a tf while
       loop. May be slower.

  Returns:
    A tuple of elements_enc, ref_enc, elements_enc_for_select
    elements_enc - The embedding representing the elements
    ref_enc - The embedding representing the referring expression.
    elements_enc_for_select - only useful when building a model to select from
    the screen elements.
  """

  with tf.variable_scope('ref_elements_model'):
    # Checks if the referring elements model is needed. If not, return 0s
    if not (flags.use_ref_exp or flags.use_elements_texts or
            flags.use_elements_boxes or flags.use_elements_ref_match or
            flags.use_elements_neighbors):
      return (tf.zeros([
          tf.shape(feature_map[ELEMENTS_EXIST_ID])[0], output_size[0],
          output_size[1], output_size[2]
      ]),
              tf.zeros([
                  tf.shape(feature_map[ELEMENTS_EXIST_ID])[0], output_size[0],
                  output_size[1], output_size[2]
              ]), None)

    if flags.use_ref_exp:
      ref_exp = tf.identity(feature_map[REF_EXP_ID], REF_EXP_ID)
      tf.summary.text('ref_exp', ref_exp)
    else:
      ref_exp = []

    # Puts everything in the same "batch" so it can be processed
    # by the text_model. Ignores features with their use_ flags set to false.
    if (flags.use_elements_texts or flags.use_elements_boxes or
        flags.use_elements_ref_match or flags.use_elements_neighbors):
      elements_mask = tf.identity(feature_map[ELEMENTS_EXIST_ID],
                                  ELEMENTS_EXIST_ID)
    if flags.use_elements_texts:
      elements_texts = tf.identity(feature_map[ELEMENTS_TEXT_ID],
                                   ELEMENTS_TEXT_ID)
      tf.summary.text('elements_texts', elements_texts)

      # Use boolean mask to ignore empty padding elements
      flat_elements_texts = tf.boolean_mask(elements_texts, elements_mask)
    else:
      elements_texts = []
      flat_elements_texts = []

    if flags.use_ref_exp or flags.use_elements_texts:
      ref_elements = tf.concat([ref_exp, flat_elements_texts], 0)

      # Text model refers to embedding model for sentences
      ref_elements_concat_enc = text_model(ref_elements,
                                           flags.pretrained_text_enc_name)

      # Unpack the "batch".
      ref_enc, elements_texts_enc = tf.split(
          ref_elements_concat_enc,
          [tf.shape(ref_exp)[0],
           tf.shape(flat_elements_texts)[0]], 0)

    if not flags.use_ref_exp:
      ref_enc = None
    if not flags.use_elements_texts:
      elements_texts_enc = None

    if (flags.use_elements_texts or flags.use_elements_boxes or
        flags.use_elements_ref_match or flags.use_elements_neighbors):
      # Elements odel can process OCR infromation
      elements_enc, elements_enc_for_select = elements_model(
          elements_texts_enc, feature_map, output_size, elements_mask, ref_enc,
          flags)
    else:
      elements_enc = tf.zeros([
          tf.shape(ref_enc)[0], output_size[0], output_size[1], output_size[2]
      ])
      elements_enc_for_select = None

    if flags.use_ref_exp:
      ref_enc = tf.layers.dense(ref_enc, output_size[2], tf.nn.relu)
      tf.summary.histogram('ref_enc', ref_enc)
    else:
      ref_enc = tf.zeros(
          [tf.shape(feature_map[ELEMENTS_EXIST_ID])[0], output_size[2]])
    if flags.elements_proj_mode != 'tile' or flags.elements_3d_output:
      ref_enc = tile_to_image(ref_enc, output_size)

    return elements_enc, ref_enc, elements_enc_for_select
