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

"""RNN value models to predict user and creator utilities given history."""


import tensorflow as tf

from recs_ecosystem_creator_rl.recommender import model_utils


class RnnValueModel:
  """Class using rnn to predict utility given history."""

  def __init__(self,
               name='rnn_value_model',
               inputs=None,
               merged_embedding=None,
               regularization_coeff=None,
               rnn_type='LSTM',
               hidden_sizes=(32, 16),
               lr=7e-4,
               model_path=None,
               seed=None):
    """Initialize a RNN value model.

    Args:
      name: String denoting the model name.
      inputs: Keras inputs of the model.
      merged_embedding: Preprocessed embedding inputs to the RNN layer.
      regularization_coeff: Float for l2 regularization coefficient.
      rnn_type: String denoting recurrent cell, 'LSTM'/'GRU'.
      hidden_sizes: Sizes of hidden layers, the first one denotes the RNN hidden
        size, and the remaining denote the size of hidden layers added to the
        RNN output. Length should be num_hidden_layers.
      lr: Float learning rate.
      model_path: String denoting the checkpoint path.
      seed: Integer, random seed.
    """
    if seed:
      tf.random.set_seed(seed)

    self._set_up(name, lr, rnn_type, hidden_sizes[0])
    # Save merged_input_model for self.get_embedding().
    self.merged_input_model = tf.keras.models.Model(
        inputs=inputs, outputs=merged_embedding)
    value_outputs = self._construct_graph(
        merged_embedding=merged_embedding,
        rnn_type=rnn_type,
        hidden_sizes=hidden_sizes,
        regularization_coeff=regularization_coeff)
    self.value_model = tf.keras.models.Model(
        inputs=inputs, outputs=value_outputs)
    self.ckpt = tf.train.Checkpoint(
        step=tf.Variable(1),
        optimizer=self.optimizer,
        value_model=self.value_model)
    self.manager = tf.train.CheckpointManager(
        self.ckpt, model_path, max_to_keep=3)
    self.ckpt.restore(self.manager.latest_checkpoint)
    if self.manager.latest_checkpoint:
      print('Restored from {}.'.format(self.manager.latest_checkpoint))
    else:
      print('Initializing from scratch.')

  def _set_up(self, name, lr, rnn_type, rnn_hidden_size):
    """Set up state size, loss, metrics, optimizer."""
    self.name = name
    self.embedding_size = rnn_hidden_size
    self.loss_object = tf.keras.losses.Huber()
    self.rnn_type = rnn_type
    # Metrics.
    self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    self.train_relative_loss = tf.keras.metrics.Mean(
        'train_loss', dtype=tf.float32)
    self.test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    self.test_relative_loss = tf.keras.metrics.Mean(
        'train_loss', dtype=tf.float32)
    # Optimizer.
    self.optimizer = tf.keras.optimizers.Adagrad(lr)

  def _construct_graph(self,
                       merged_embedding,
                       rnn_type,
                       hidden_sizes,
                       regularization_coeff=None):
    """Construct RNN to embed hidden states and predict policy value."""
    if regularization_coeff is not None:
      regularizer_obj = tf.keras.regularizers.l2(regularization_coeff)
    else:
      regularizer_obj = None
    # Save rnn_layer for embedding history.
    self.rnn_layer, whole_sequence_output, _ = model_utils.construct_rnn_layer(
        rnn_type, merged_embedding, hidden_sizes[0], regularizer_obj)
    # Save embedding_layers for later predicting utility given history.
    self.embedding_layers = []
    for i, hidden_size in enumerate(hidden_sizes[1:], 1):
      self.embedding_layers.append(
          tf.keras.layers.Dense(
              units=hidden_size,
              activation='relu',
              kernel_regularizer=regularizer_obj,
              name=f'{self.name}_hidden_layer_{i}'))
    # Add value_layer.
    self.embedding_layers.append(
        tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_regularizer=regularizer_obj,
            name=f'{self.name}_value_layer'))

    for embedding_layer in self.embedding_layers:
      whole_sequence_output = embedding_layer(whole_sequence_output)
    return whole_sequence_output

  def predict_value(self, inputs, mask=None, initial_state=None):
    """Predict utility for given history.

    First use self.get_embedding() to obtain the last rnn layer output. Then
    use self.embedding_layers() to get the last utility prediction.
    This function will be useful for creator uplift modeling.

    Args:
      inputs: List or np.array, RNN inputs, representing the given history.
      mask: A tensor of size (batch_size, trajectory_length) and type bool. Rnn
        layer will ignore the computation at mask=False.
      initial_state: Initial state of rnn layer. When rnn_type=LSTM,
        initial_state=[initial_memory_state, initial_carry_state], both have
        size (batch_size, rnn_hidden_size). When rnn_type=GRU, initial_state has
        size (batch_size, rnn_hidden_size).

    Returns:
      output: A tensor of shape (batch_size, 1), the predicted utilities for
        given trajectories.
    """
    last_rnn_output, _ = self.get_embedding(inputs, mask, initial_state)
    output = last_rnn_output
    for layer in self.embedding_layers:
      output = layer(output)
    return output

  def get_embedding(self, inputs, mask=None, initial_state=None):
    """Get embedding hidden states for given history.

    This function will be useful for embedding states and creator uplift
    modeling.

    Args:
      inputs: List or np.array, RNN inputs, representing the given history.
      mask: A tensor of size (batch_size, trajectory_length) and type bool. Rnn
        layer will ignore the computation at mask=False.
      initial_state: Initial state of rnn layer. When rnn_type=LSTM,
        initial_state=[initial_memory_state, initial_carry_state], both have
        size (batch_size, rnn_hidden_size). When rnn_type=GRU, initial_state has
        size (batch_size, rnn_hidden_size).

    Returns:
       rnn_final_output: An array of shape (batch_size, rnn_hidden_state), the
         last rnn layer output. This will be fed into actor agent as
         embedding_state.
       final_state: Final internal state(s) of rnn layer. The returned states
         are useful for initializing the rnn layer in creator_value_model for
         uplift modeling.
    """
    merged_inputs = self.merged_input_model.predict(inputs)

    if self.rnn_type == 'LSTM':
      whole_seq_output, final_memory_state, final_carry_state = self.rnn_layer(
          merged_inputs, mask=mask, initial_state=initial_state)
      return (whole_seq_output[:,
                               -1, :], [final_memory_state, final_carry_state])
    elif self.rnn_type == 'GRU':
      whole_seq_output, final_state = self.rnn_layer(
          merged_inputs, mask=mask, initial_state=initial_state)
      return (whole_seq_output[:, -1, :], final_state)
    else:
      raise NotImplementedError('Only support LSTM/GRU cells.')

  def train_step(self, inputs, targets, masks):
    """Batch train."""
    self.ckpt.step.assign_add(1)
    with tf.GradientTape() as tape:
      predictions = self.value_model(inputs, training=True)
      loss = self.loss_object(predictions, targets, sample_weight=masks)
      relative_loss_weights = masks / (
          tf.maximum(tf.abs(tf.squeeze(targets, -1)), 1e-7))
      relative_loss = self.loss_object(
          predictions, targets, sample_weight=relative_loss_weights)
    grads = tape.gradient(loss, self.value_model.trainable_variables)
    self.optimizer.apply_gradients(
        zip(grads, self.value_model.trainable_variables))
    self.train_loss(loss)
    self.train_relative_loss(relative_loss)

  def test_step(self, inputs, targets, masks):
    predictions = self.value_model(inputs)
    loss = self.loss_object(predictions, targets, sample_weight=masks)
    relative_loss_weights = masks / (
        tf.maximum(tf.abs(tf.squeeze(targets, -1)), 1e-7))
    relative_loss = self.loss_object(
        predictions, targets, sample_weight=relative_loss_weights)
    self.test_loss(loss)
    self.test_relative_loss(relative_loss)

  def save(self):
    save_path = self.manager.save()
    print('Saved checkpoint for step {}: {}'.format(
        int(self.ckpt.step), save_path))


class UserValueModel(RnnValueModel):
  """Class using rnn to predict user utility given user history."""

  def __init__(self,
               document_feature_size=None,
               creator_feature_size=None,
               user_feature_size=None,
               input_reward=None,
               model_path=None,
               lr=1e-3,
               rnn_type=None,
               regularization_coeff=None,
               hidden_sizes=(20, 20),
               seed=None):
    merged_embedding, inputs = model_utils.construct_user_rnn_inputs(
        document_feature_size, creator_feature_size, user_feature_size,
        input_reward)
    super(UserValueModel, self).__init__(
        'user_value_model',
        inputs=inputs,
        merged_embedding=merged_embedding,
        rnn_type=rnn_type,
        regularization_coeff=regularization_coeff,
        hidden_sizes=hidden_sizes,
        lr=lr,
        model_path=model_path,
        seed=seed)


class CreatorValueModel(RnnValueModel):
  """Class using rnn to predict user utility given user history."""

  def __init__(self,
               document_feature_size=None,
               creator_feature_size=None,
               model_path=None,
               lr=1e-3,
               rnn_type=None,
               regularization_coeff=None,
               hidden_sizes=(20, 20),
               num_creators=None,
               creator_id_embedding_size=0,
               trajectory_length=None,
               seed=None):
    self.document_feature_size = document_feature_size
    self.trajectory_length = trajectory_length
    self.creator_id_embedding_size = creator_id_embedding_size
    merged_embedding, inputs = model_utils.construct_creator_rnn_inputs(
        document_feature_size, creator_feature_size, num_creators,
        creator_id_embedding_size, trajectory_length)
    super(CreatorValueModel, self).__init__(
        'creator_value_model',
        inputs=inputs,
        merged_embedding=merged_embedding,
        rnn_type=rnn_type,
        regularization_coeff=regularization_coeff,
        hidden_sizes=hidden_sizes,
        lr=lr,
        model_path=model_path,
        seed=seed)
