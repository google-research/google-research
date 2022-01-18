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

"""Model layers in dual encoder SMITH model."""
from six.moves import range
import tensorflow.compat.v1 as tf

from smith import constants
from smith.bert import modeling


def get_doc_rep_with_masked_sent(input_sent_reps_doc,
                                 sent_mask_embedding,
                                 input_mask_doc_level,
                                 batch_size_static=32,
                                 max_masked_sent_per_doc=2,
                                 loop_sent_number_per_doc=32):
  """Get the document representations with masked sentences.

  Args:
      input_sent_reps_doc: float Tensor. The independent sentence embeddings
        without masks for the sentences in the current document. The shape is
        [batch, loop_sent_number_per_doc, hidden].
      sent_mask_embedding: float Tensor. The sentence embedding vector for the
        masked position. The shape is [hidden].
      input_mask_doc_level: int Tensor. The input masks on the document level to
        identify whether a location is a real sentence (mask = 1) or a padded
        sentence (mask = 0). The shape is [batch, loop_sent_number_per_doc].
      batch_size_static: scalar. The static batch size depending on the training
        or the evaluation mode.
      max_masked_sent_per_doc: scalar. The maximum number of masked sentences
        per document.
      loop_sent_number_per_doc: scalar. The number of looped sentences per
        document.

  Returns:
    The document representations with masked sentences and the positions/
    weights for each masked sentences. This masked sentence weight is 1 for the
    sampled real sentence position and 0 for the padded sentence position.
  """
  # We at least mask two sentences to build a candidate sentence pool for
  # negative sentence sampling. We generate the masked_sent_index and
  # masked_sent_weight for each document. Note that we do not add any word
  # or sentence level masks during prediction or inference stage.
  max_masked_sent_per_doc = max(max_masked_sent_per_doc, 2)
  input_sent_reps_doc_list = tf.unstack(
      input_sent_reps_doc, num=batch_size_static)
  real_sent_number_per_doc = tf.unstack(
      tf.reduce_sum(input_mask_doc_level, 1), num=batch_size_static)
  masked_sent_index_list = []
  masked_sent_weight_list = []

  # For each example in the current batch, we randomly sample
  # max_masked_sent_per_doc positions to mask the sentences. For each masked
  # sentence position, the sentence in the current position is the positive
  # example. The other co-masked sentences are the negative examples.
  # The sampled sentence indexes will not be duplicated.
  for batch_i in range(0, batch_size_static):
    # Since everything in TPU must have a fixed shape, here the max sampled
    # sentence index can be as large as loop_sent_number_per_doc. We will
    # generate the corresponding sentence LM weights to reduce the impact
    # on the final masked sentence LM loss following a similar way with the
    # handling of masked word LM loss and masked word LM weights.
    real_sent_number = real_sent_number_per_doc[batch_i]
    sampled_sent_index = tf.slice(
        tf.random_shuffle(tf.range(loop_sent_number_per_doc)), [0],
        [max_masked_sent_per_doc])
    sampled_sent_index = tf.sort(sampled_sent_index)
    masked_sent_index_list.append(sampled_sent_index)
    # Generates the corresponding sampled_sent_weight
    sample_sent_weight = tf.cast(
        tf.less(sampled_sent_index, real_sent_number), tf.float32)
    masked_sent_weight_list.append(sample_sent_weight)

    indices = tf.reshape(sampled_sent_index, [max_masked_sent_per_doc, -1])
    # Duplicates sent_mask_embedding for each masked position.
    updates = tf.reshape(
        tf.tile(
            sent_mask_embedding,
            [max_masked_sent_per_doc],
        ), [max_masked_sent_per_doc, -1])
    input_sent_reps_doc_list[batch_i] = tf.tensor_scatter_update(
        input_sent_reps_doc_list[batch_i], indices, updates)
  # Here masked_sent_index_list is a list a tensors, where each tensor stores
  # the masked sentence positions for each document in the current batch. The
  # shape of masked_sent_index_list is [batch, max_masked_sent_per_doc].
  # Here masked_sent_weight_list is a list a tensors, where each tensor stores
  # the masked sentence weights for each document in the current batch. The
  # shape of masked_sent_weight_list is [batch, max_masked_sent_per_doc].
  return (tf.stack(input_sent_reps_doc_list), tf.stack(masked_sent_index_list),
          tf.stack(masked_sent_weight_list))


def get_masked_sent_lm_output(bert_config,
                              input_tensor,
                              cur_sent_reps_doc_unmask,
                              sent_masked_positions,
                              sent_masked_weights,
                              debugging=False):
  """Get the sentence level masked LM loss.

  Args:
      bert_config: BertConfig object. The configuration file for the document
        level BERT model.
      input_tensor: float Tensor. The contextualized representations of all
        sentences learned by the document level BERT model. The shape is [batch,
        loop_sent_number_per_doc, hidden]. This is the model prediction.
      cur_sent_reps_doc_unmask: float Tensor. The unmasked sentence
        representations of the current document. The shape is [batch,
        loop_sent_number_per_doc, hidden]. This is the source of the ground
        truth and negative examples in the masked sentence prediction.
      sent_masked_positions: int Tensor. The masked sentence positions in the
        current document. The shape is [batch, max_masked_sent_per_doc].
      sent_masked_weights: float Tensor. The masked sentence weights in the
        current document. The shape is [batch, max_masked_sent_per_doc].
      debugging: bool. Whether it is in the debugging mode.

  Returns:
    The masked sentence LM loss and the mask sentence LM loss per example.

  """
  # The current method for masked sentence prediction: we approach this problem
  # as a multi-class classification problem similar to the masked word LM task.
  # For each masked sentence position, the sentence in the current position is
  # the positive example. The other co-masked sentences in the current document
  # and in the other documents of the same batch are the negative examples. We
  # compute the cross entropy loss over the sentence prediction task following
  # the implementation of the masked word LM loss in the BERT model.

  input_tensor_shape = modeling.get_shape_list(input_tensor)
  batch_size = input_tensor_shape[0]
  masked_position_shape = modeling.get_shape_list(sent_masked_positions)
  max_predictions_per_seq = masked_position_shape[1]

  # In the context of masked sentence prediction, the max_predictions_per_seq
  # is the same with max_masked_sent_per_doc.
  # Output Shape: [batch * max_predictions_per_seq, hidden].
  # Input_tensor is the model prediction for each position.
  input_tensor = gather_indexes(input_tensor, sent_masked_positions)
  # Independent_sent_embeddings is the ground truth input sentence embeddings
  # for the document level BERT model. The output shape is [batch *
  # max_predictions_per_seq, hidden].
  independent_sent_embeddings = gather_indexes(cur_sent_reps_doc_unmask,
                                               sent_masked_positions)

  with tf.variable_scope("cls/sent_predictions", reuse=tf.AUTO_REUSE):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      # Output Shape: [batch * max_predictions_per_seq, hidden].
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each predicted position.
    output_bias = tf.get_variable(
        "output_bias",
        shape=[batch_size * max_predictions_per_seq],
        initializer=tf.zeros_initializer())
    # Shape of input_tensor [batch * max_predictions_per_seq, hidden].
    # Shape of independent_sent_embeddings is [batch * max_predictions_per_seq,
    # hidden].
    # Shape of logits: [batch * max_predictions_per_seq,
    # batch * max_predictions_per_seq].
    logits = tf.matmul(
        input_tensor, independent_sent_embeddings, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    # Output Shape: [batch * max_predictions_per_seq,
    # batch * max_predictions_per_seq].
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    # Output Shape: [batch * max_predictions_per_seq].
    # Double checked the setting of label_ids here. The label_ids
    # should be the label index in the "sentence vocabulary". Thus if batch=32,
    # max_predictions_per_seq = 2, then label ids should be like
    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ..., 63]. For the ground truth one hot
    # label matrix, only the values in the diagonal positions are 1. All the
    # other positions should be 0.
    label_ids = tf.range(
        0, batch_size * max_predictions_per_seq, dtype=tf.int32)
    if debugging:
      label_ids = tf.Print(
          label_ids, [label_ids],
          message="label_ids in get_masked_sent_lm_output",
          summarize=30)
    # Output Shape: [batch * max_predictions_per_seq].
    # The label_weights is the flatten vector based on sent_masked_weights,
    # where the weight is 1.0 for sampled real sentences and 0.0 for sampled
    # masked sentences.
    label_weights = tf.reshape(sent_masked_weights, [-1])

    # Output Shape: [batch * max_predictions_per_seq,
    # batch * max_predictions_per_seq].
    one_hot_labels = tf.one_hot(
        label_ids, depth=batch_size * max_predictions_per_seq, dtype=tf.float32)

    # Output Shape: [batch * max_predictions_per_seq].
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    # Output Shape: [1].
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    # Output Shape: [1].
    denominator = tf.reduce_sum(label_weights) + 1e-5
    # Output Shape: [1].
    loss = numerator / denominator
    # Shape of loss [1].
    # Shape of per_example_loss is [batch * max_predictions_per_seq].
  return (loss, per_example_loss, log_probs)


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
  """Get loss and log probs for the masked LM."""
  # Output Shape: [batch * max_predictions_per_seq, hidden].
  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/word_predictions", reuse=tf.AUTO_REUSE):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      # Output Shape: [batch * max_predictions_per_seq, hidden].
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        shape=[bert_config.vocab_size],
        initializer=tf.zeros_initializer())
    # Shape of input_tensor [batch * max_predictions_per_seq, embedding_size].
    # Shape of output_weights (embed table) is [vocab_size, embedding_size].
    # In the current Bert implementation: embedding_size = hidden.
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    # Output Shape: [batch * max_predictions_per_seq, vocab_size].
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    # Output Shape: [batch * max_predictions_per_seq].
    label_ids = tf.reshape(label_ids, [-1])
    # Output Shape: [batch * max_predictions_per_seq].
    label_weights = tf.reshape(label_weights, [-1])

    # Output Shape: [batch * max_predictions_per_seq, vocab_size].
    one_hot_labels = tf.one_hot(
        label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    # Output Shape: [batch * max_predictions_per_seq].
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    # Output Shape: [1].
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    # Output Shape: [1].
    denominator = tf.reduce_sum(label_weights) + 1e-5
    # Output Shape: [1].
    loss = numerator / denominator
    # Shape of loss [1].
    # Shape of per_example_loss is [batch * max_predictions_per_seq].
  return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  # Shape of positions: [batch, max_mask_per_seq].
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  # Shape of flat_offsets: [batch, 1].
  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  # The shape of output_tensor [batch * max_mask_per_seq, hidden].
  return output_tensor


def get_attention_weighted_sum(input_tensor, bert_config, is_training,
                               attention_size):
  """Compute the attentive weighted sum of an input tensor.

  Args:
      input_tensor: The input tensor for attentive representation. The shape of
        input tensor is [batch, seq_length, hidden].
      bert_config: The model config file.
      is_training: If true, it is in training mode.
      attention_size: int. Dimension of contextual vector.

  Returns:
      The attentive representation of the input tensor. The shape of the output
        tensor is [batch, hidden].
  """
  with tf.variable_scope("combine_reps_attention", reuse=tf.AUTO_REUSE):
    context_vector = tf.get_variable(
        name="context_vector",
        shape=[attention_size],
        dtype=tf.float32)
    # Output Shape: [batch, seq_length, attention_size].
    projection = tf.layers.dense(
        input_tensor,
        attention_size,
        activation=tf.tanh,
        kernel_initializer=modeling.create_initializer(
            bert_config.initializer_range))
    # Output Shape: [batch, seq_length, 1].
    attention = tf.reduce_sum(
        tf.multiply(projection, context_vector), axis=2, keep_dims=True)
    # Output Shape: [batch, seq_length, 1].
    attention = tf.nn.softmax(attention, axis=1)
    # Output Shape: [batch, hidden].
    last_outputs = tf.reduce_sum(tf.multiply(input_tensor, attention), axis=1)
    if is_training:
      last_outputs = tf.layers.dropout(
          last_outputs, bert_config.attention_probs_dropout_prob, training=True)
  return last_outputs


def get_seq_rep_from_bert(bert_model):
  """Get the sequence represenation given a BERT encoder."""
  siamese_input_tensor = bert_model.get_pooled_output()
  hidden_size = siamese_input_tensor.shape[-1].value
  siamese_input_tensor = tf.layers.dense(
      siamese_input_tensor, units=hidden_size, activation=tf.nn.relu)
  normalized_siamese_input_tensor = tf.nn.l2_normalize(
      siamese_input_tensor, axis=1)
  return normalized_siamese_input_tensor


def get_sent_reps_masks_normal_loop(sent_index,
                                    input_sent_reps_doc,
                                    input_mask_doc_level,
                                    masked_lm_loss_doc,
                                    masked_lm_example_loss_doc,
                                    masked_lm_weights_doc,
                                    dual_encoder_config,
                                    is_training,
                                    train_mode,
                                    input_ids,
                                    input_mask,
                                    masked_lm_positions,
                                    masked_lm_ids,
                                    masked_lm_weights,
                                    use_one_hot_embeddings,
                                    debugging=False):
  """Get the sentence encodings, mask ids and masked word LM loss.

  Args:
      sent_index: The index of the current looped sentence.
      input_sent_reps_doc: The representations of all sentences in the doc
        learned by BERT.
      input_mask_doc_level: The document level input masks, which indicates
        whether a sentence is a real sentence or a padded sentence.
      masked_lm_loss_doc: The sum of all the masked word LM loss.
      masked_lm_example_loss_doc: The per example masked word LM loss.
      masked_lm_weights_doc: the weights of the maksed LM words. If the position
        is corresponding to a real masked word, it is 1.0; It is a padded mask,
        the weight is 0.
      dual_encoder_config: The config of the dual encoder.
      is_training: Whether it is in the training mode.
      train_mode: string. The train mode which can be finetune, joint_train, or
        pretrain.
      input_ids: The ids of the input tokens.
      input_mask: The mask of the input tokens.
      masked_lm_positions: The positions of the masked words in the language
        model training.
      masked_lm_ids: The ids of the masked words in LM model training.
      masked_lm_weights: The weights of the masked words in LM model training.
      use_one_hot_embeddings: Whether use one hot embedding. It should be true
        for the runs on TPUs.
      debugging: bool. Whether it is in the debugging mode.

  Returns:
    A list of tensors on the learned sentence representations and the masked
    word LM loss.
  """
  # Collect token information for the current sentence.
  bert_config = modeling.BertConfig.from_json_file(
      dual_encoder_config.encoder_config.bert_config_file)
  max_sent_length_by_word = dual_encoder_config.encoder_config.max_sent_length_by_word
  sent_bert_trainable = dual_encoder_config.encoder_config.sent_bert_trainable
  max_predictions_per_seq = dual_encoder_config.encoder_config.max_predictions_per_seq
  sent_start = sent_index * max_sent_length_by_word
  input_ids_cur_sent = tf.slice(input_ids, [0, sent_start],
                                [-1, max_sent_length_by_word])
  # Output shape: [batch, max_sent_length_by_word].
  input_mask_cur_sent = tf.slice(input_mask, [0, sent_start],
                                 [-1, max_sent_length_by_word])
  # Output Shape:  [batch].
  input_mask_cur_sent_max = tf.reduce_max(input_mask_cur_sent, 1)
  # Output Shape:  [loop_sent_number_per_doc, batch].
  input_mask_doc_level.append(input_mask_cur_sent_max)
  if debugging:
    input_ids_cur_sent = tf.Print(
        input_ids_cur_sent, [input_ids_cur_sent, input_mask_cur_sent],
        message="input_ids_cur_sent in get_sent_reps_masks_lm_loss",
        summarize=20)
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids_cur_sent,
      input_mask=input_mask_cur_sent,
      use_one_hot_embeddings=use_one_hot_embeddings,
      sent_bert_trainable=sent_bert_trainable)
  with tf.variable_scope("seq_rep_from_bert_sent_dense", reuse=tf.AUTO_REUSE):
    normalized_siamese_input_tensor = get_seq_rep_from_bert(model)
  input_sent_reps_doc.append(normalized_siamese_input_tensor)

  if (train_mode == constants.TRAIN_MODE_PRETRAIN or
      train_mode == constants.TRAIN_MODE_JOINT_TRAIN):
    # Collect masked token information for the current sentence.
    sent_mask_lm_token_start = sent_index * max_predictions_per_seq
    # Output shape: [batch, max_predictions_per_seq].
    masked_lm_positions_cur_sent = tf.slice(masked_lm_positions,
                                            [0, sent_mask_lm_token_start],
                                            [-1, max_predictions_per_seq])
    masked_lm_ids_cur_sent = tf.slice(masked_lm_ids,
                                      [0, sent_mask_lm_token_start],
                                      [-1, max_predictions_per_seq])
    masked_lm_weights_cur_sent = tf.slice(masked_lm_weights,
                                          [0, sent_mask_lm_token_start],
                                          [-1, max_predictions_per_seq])
    # Since in the processed data of smith model, the masked lm positions are
    # global indices started from the 1st token of the whole sequence, we need
    # to transform this global position to a local position for the current
    # sentence. The position index is started from 0.
    # Local_index = global_index mod max_sent_length_by_word.
    masked_lm_positions_cur_sent = tf.mod(masked_lm_positions_cur_sent,
                                          max_sent_length_by_word)
    # Shape of masked_lm_loss_cur_sent [1].
    # Shape of masked_lm_example_loss_cur_sent is [batch,
    # max_predictions_per_seq].
    (masked_lm_loss_cur_sent, masked_lm_example_loss_cur_sent,
     _) = get_masked_lm_output(bert_config, model.get_sequence_output(),
                               model.get_embedding_table(),
                               masked_lm_positions_cur_sent,
                               masked_lm_ids_cur_sent,
                               masked_lm_weights_cur_sent)
    # Output Shape: [1].
    masked_lm_loss_doc += masked_lm_loss_cur_sent
    # Output Shape: [loop_sent_number_per_doc, batch * max_predictions_per_seq].
    masked_lm_example_loss_doc.append(masked_lm_example_loss_cur_sent)
    # Output Shape: [loop_sent_number_per_doc, batch, max_predictions_per_seq].
    masked_lm_weights_doc.append(masked_lm_weights_cur_sent)
  return (input_sent_reps_doc, input_mask_doc_level, masked_lm_loss_doc,
          masked_lm_example_loss_doc, masked_lm_weights_doc)


def learn_sent_reps_normal_loop(dual_encoder_config, is_training, train_mode,
                                input_ids_1, input_mask_1,
                                masked_lm_positions_1, masked_lm_ids_1,
                                masked_lm_weights_1, input_ids_2, input_mask_2,
                                masked_lm_positions_2, masked_lm_ids_2,
                                masked_lm_weights_2, use_one_hot_embeddings):
  """Learn the sentence representations with normal loop functions."""
  input_sent_reps_doc_1 = []
  # Generate document level input masks on each sentence based on the word
  # level input mask information.
  input_mask_doc_level_1 = []
  masked_lm_loss_doc_1 = 0.0
  masked_lm_example_loss_doc_1 = []
  masked_lm_weights_doc_1 = []

  input_mask_doc_level_2 = []
  input_sent_reps_doc_2 = []
  masked_lm_loss_doc_2 = 0.0
  masked_lm_example_loss_doc_2 = []
  masked_lm_weights_doc_2 = []

  # Learn the representation for each sentence in the document.
  # Setting smaller number of loop_sent_number_per_doc can save memory for the
  # model training.
  # Shape of masked_lm_loss_doc_1 [1].
  # Shape of masked_lm_example_loss_doc_1 is [max_doc_length_by_sentence,
  # batch * max_predictions_per_seq].
  for sent_index in range(
      0, dual_encoder_config.encoder_config.loop_sent_number_per_doc):
    (input_sent_reps_doc_1, input_mask_doc_level_1, masked_lm_loss_doc_1,
     masked_lm_example_loss_doc_1,
     masked_lm_weights_doc_1) = get_sent_reps_masks_normal_loop(
         sent_index, input_sent_reps_doc_1, input_mask_doc_level_1,
         masked_lm_loss_doc_1, masked_lm_example_loss_doc_1,
         masked_lm_weights_doc_1, dual_encoder_config, is_training, train_mode,
         input_ids_1, input_mask_1, masked_lm_positions_1, masked_lm_ids_1,
         masked_lm_weights_1, use_one_hot_embeddings)
    (input_sent_reps_doc_2, input_mask_doc_level_2, masked_lm_loss_doc_2,
     masked_lm_example_loss_doc_2,
     masked_lm_weights_doc_2) = get_sent_reps_masks_normal_loop(
         sent_index, input_sent_reps_doc_2, input_mask_doc_level_2,
         masked_lm_loss_doc_2, masked_lm_example_loss_doc_2,
         masked_lm_weights_doc_2, dual_encoder_config, is_training, train_mode,
         input_ids_2, input_mask_2, masked_lm_positions_2, masked_lm_ids_2,
         masked_lm_weights_2, use_one_hot_embeddings)

  # Stack the sentence representations to learn the doc representations.
  # Output Shape: [batch, loop_sent_number_per_doc, hidden].
  input_sent_reps_doc_1_unmask = tf.stack(input_sent_reps_doc_1, axis=1)
  input_sent_reps_doc_2_unmask = tf.stack(input_sent_reps_doc_2, axis=1)

  # Output Shape:  [batch, loop_sent_number_per_doc].
  input_mask_doc_level_1_tensor = tf.stack(input_mask_doc_level_1, axis=1)
  input_mask_doc_level_2_tensor = tf.stack(input_mask_doc_level_2, axis=1)

  if (train_mode == constants.TRAIN_MODE_PRETRAIN or
      train_mode == constants.TRAIN_MODE_JOINT_TRAIN):
    # Output Shape:  [batch * max_predictions_per_seq,
    # loop_sent_number_per_doc].
    masked_lm_example_loss_doc_1 = tf.stack(
        masked_lm_example_loss_doc_1, axis=1)
    masked_lm_example_loss_doc_2 = tf.stack(
        masked_lm_example_loss_doc_2, axis=1)

    # Output Shape:  [batch, loop_sent_number_per_doc, max_predictions_per_seq].
    masked_lm_weights_doc_1 = tf.stack(masked_lm_weights_doc_1, axis=1)
    masked_lm_weights_doc_2 = tf.stack(masked_lm_weights_doc_2, axis=1)
  else:
    masked_lm_example_loss_doc_1 = tf.zeros([1])
    masked_lm_example_loss_doc_2 = tf.zeros([1])
    masked_lm_weights_doc_1 = tf.zeros([1])
    masked_lm_weights_doc_2 = tf.zeros([1])

  return (input_sent_reps_doc_1_unmask, input_mask_doc_level_1_tensor,
          input_sent_reps_doc_2_unmask, input_mask_doc_level_2_tensor,
          masked_lm_loss_doc_1, masked_lm_loss_doc_2,
          masked_lm_example_loss_doc_1, masked_lm_example_loss_doc_2,
          masked_lm_weights_doc_1, masked_lm_weights_doc_2)
