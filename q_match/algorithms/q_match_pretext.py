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

"""Distribution match pretext.

Maintains a queue of embeddings.  For each data in the batch, corrupt it
twice into two views: view1 and view2.  Compute a list of similarities for each
view: dist1=sim(view1,queue) and dist2(view2,queue).  The loss for this algo
is then the cross entropy between the two distributions.
"""

from absl import logging
from flax.core import freeze
import jax
import jax.numpy as jnp

from q_match.algorithms.training_algo import l2_normalize
from q_match.algorithms.training_algo import PretextTrainingAlgo
from q_match.algorithms.vime_pretext_training import vime_corruption


@jax.jit
def euclidean_distance(x, y):
  """Euclidean distance between x and y.

  Args:
    x: tensor of points shape (n_points_x, 1, dim)
    y: tenosr of points with shape (1, n_points_y, dim)
  Returns:
    Tensor of eucldean distance of shape (n_points_x, n_points_y)
  """
  return jnp.sum((x - y) ** 2, axis=-1) ** .5


@jax.jit
def mse_distance(x, y):
  """Mean squared error distance between x and y.

  Args:
    x: tensor of points shape (n_points_x, 1, dim)
    y: tenosr of points with shape (1, n_points_y, dim)
  Returns:
    Tensor of mse of shape (n_points_x, n_points_y)
  """
  return jnp.mean((x - y) ** 2, axis=-1)


@jax.jit
def dist_match_loss(teacher_embs,
                    queue_embs,
                    student_embs,
                    teacher_temperature=0.04,
                    student_temperature=0.1):
  """Distribution loss.

  Usually teacher temp is lower than student temperature
  l2_normalize the embeddings along last axis before calculating loss
  The teacher emb will be the NN.

  Args:
    teacher_embs:  The embeddings to use as the target.
    queue_embs: A queue of embeddings to match against.
    student_embs: The embeddings to propogate gradients.
    teacher_temperature: Scaling temp for the teacher.
    student_temperature: Scaling temp for the student.

  Returns:
    Loss for how closes the student distribution matches the teacher
    distribution.
  """
  teacher_embs = jax.lax.stop_gradient(l2_normalize(teacher_embs))
  queue_embs = jax.lax.stop_gradient(l2_normalize(queue_embs))
  student_embs = l2_normalize(student_embs)

  teacher_logits = jnp.matmul(teacher_embs, queue_embs.T)
  teacher_logits /= teacher_temperature
  targets = jax.lax.stop_gradient(jax.nn.softmax(teacher_logits, axis=-1))

  logits = jnp.matmul(student_embs, queue_embs.T)
  logits /= student_temperature

  loss = jnp.sum(-targets * jax.nn.log_softmax(logits, axis=-1), axis=-1)
  loss = jnp.mean(loss)
  return loss


@jax.jit
def update_queue(new_embs, queue):
  """Updates the queue with the new_embs but keeps queue the same size."""
  return jnp.concatenate([new_embs, queue], axis=0)[:queue.shape[0]]


@jax.jit
def _update_ema_params(ema_params, new_params, tau):
  """Returns new EMA params."""
  return jax.tree.map(lambda x, y: x * tau + (1. - tau) * y, ema_params,
                      new_params)


class QMatchPretextTraining(PretextTrainingAlgo):
  """Q Match Training Algorithm.

  Attributes:
    logdir: location of the log directory.
    dataset: tf dataset to train.
    batch_size: batch size for training.
    model: Dictionary of models that includes
    learning_rate: the learning rate for training.
    epochs: number of epochs to train for
    params: Optional params to start training from.  If None, random params
      are initialized.
    state: Optional state to start training from.
    writer: Writer for writing to tensorboard.
    support_set_size: Size of the support set. if zero, batch mode is
      used instead.
    batch_mode: Whether to use batch mode.
    use_mse_loss: whether to use MSE loss instead of log loss.
    support_init_key: support set initialization key.
    weight_decay: weight decay on pretext params.
    corruption_p: The probability of corrupting for view1
    query_corruption_p: The probability for corruption for view 2
    student_temperature: Student temperature in distribution match loss.
  """

  def __init__(
      self,
      logdir,
      dataset,
      batch_size,
      model,
      eval_model,
      learning_rate,
      epochs,
      params=None,
      state=None,
      writer=None,
      support_set_size=4096,
      add_vime_loss=False,
      support_init_key=None,
      weight_decay=0.,
      corruption_p=.3,
      query_corruption_p=0.,
      student_temperature=0.1,
      patience=32,
      use_momentum_encoder=True,
      tau=.999,
      **kwargs
  ):

    super(QMatchPretextTraining,
          self).__init__(logdir, dataset, batch_size, model, eval_model,
                         learning_rate, epochs, params, state, writer,
                         weight_decay, patience=patience)
    self.add_vime_loss = add_vime_loss
    self.support_set_size = support_set_size
    self.support = None
    self.support_init_key = support_init_key
    self.support_keys = None
    if self.support_init_key is None:
      self.support_init_key = jax.random.PRNGKey(0)

    self.mask_key = jax.random.PRNGKey(99)
    self.corruption_p = corruption_p
    self.query_corruption_p = query_corruption_p
    self.student_temperature = student_temperature
    self.use_momentum_encoder = use_momentum_encoder
    self.tau = tau

  def _loss(self, params, state, features, support, mask_key, ema_params):
    """Loss with distribution match."""

    variables = freeze({'params': params, **state})
    if self.use_momentum_encoder:
      variables_2 = freeze({'params': ema_params, **state})
    else:
      variables_2 = freeze({'params': params, **state})

    variables_2 = jax.lax.stop_gradient(variables_2)

    ## View 1
    corrupted_features, _ = vime_corruption(features, self.corruption_p,
                                            mask_key)

    output, updated_state = self.model.apply(variables,
                                             corrupted_features,
                                             mutable=['batch_stats'],
                                             rngs=self.rngs)

    pretext_output = output['pretext']
    encoded = pretext_output['proj']
    encoded_normed = l2_normalize(encoded)  # encoded of corrupted

    ## View 2
    # Use the first key later, so pick second.
    _, new_mask_key = jax.random.split(self.mask_key)
    view_2_features, _ = vime_corruption(
        features, p=self.query_corruption_p, mask_key=new_mask_key)

    output_2, _ = self.model.apply(
        variables_2, view_2_features, mutable=['batch_stats'], rngs=self.rngs)
    pretext_output_2 = output_2['pretext']
    encoded_2 = pretext_output_2['proj']
    encoded_normed_2 = l2_normalize(encoded_2)  # encoded of corrupted

    embeddings_to_add = encoded_normed_2

    pretext_loss = dist_match_loss(
        teacher_embs=encoded_normed_2,
        queue_embs=support,
        student_embs=encoded_normed,
        student_temperature=self.student_temperature)

    return pretext_loss, (updated_state, embeddings_to_add)

  def run(self,):
    """Runs a pretext training algo."""
    params = self.params
    state = self.state
    dataset = self.dataset
    model = self.model
    ema_params = jax.tree_util.tree_map(jax.numpy.copy, params)

    example_data = jax.numpy.array(dataset.get_example_features())
    variables = freeze({'params': params, **state})
    example_output, _ = model.apply(variables,
                                    example_data,
                                    mutable=['batch_stats'],
                                    rngs=self.rngs
                                    )
    logging.debug(str(example_output))

    # initialize the support
    if self.support is None:
      encoded_dim = example_output['pretext']['proj'].shape[1]
      self.support = jax.random.normal(
          key=self.support_init_key, shape=(self.support_set_size, encoded_dim))
      self.support = l2_normalize(self.support)

    optimizer_state = self.optimizer.init(params=params)

    grad_fn = self.get_grad_fn()

    steps = 0
    for epoch in range(self.epochs):
      logging.info('Pretext Epoch: %d', epoch)
      for example in dataset.get_pretext_ds():
        features = jax.numpy.array(example['features'])

        if steps % 100 == 0:
          pretext_loss, _ = self.loss(
              params, state, features,
              self.support,
              self.mask_key,
              ema_params)
          log_train_loss_msg = f'pretext training loss {pretext_loss}'
          logging.info(log_train_loss_msg)

          metrics = {'pretext_train_loss': pretext_loss,}

          if self.writer is not None:
            self.writer.write_scalars(steps, metrics)

        gradients, (state, support_addition) = grad_fn(
            params, state, features,
            self.support, self.mask_key, ema_params)
        params, optimizer_state = self.update_model(params,
                                                    gradients,
                                                    optimizer_state)
        self.update_rngs()
        ema_params = _update_ema_params(ema_params, params, self.tau)
        self.support = update_queue(support_addition, self.support)
        self.mask_key, _ = jax.random.split(self.mask_key)
        steps += 1

      # # check validation pretext dataset if it exists
      pretext_validation_ds = dataset.get_pretext_validation_ds()
      if pretext_validation_ds is not None:
        # compute validation loss
        validation_loss = 0.
        val_seen = 0
        val_mask_key = self.mask_key
        for example in pretext_validation_ds:
          features = jax.numpy.array(example['features'])
          seen = features.shape[0]
          validation_loss += self.loss(
              params,
              state,
              features,
              self.support,
              val_mask_key,
              ema_params)[0] * seen
          val_seen += seen
          val_mask_key, _ = jax.random.split(val_mask_key)
        validation_loss /= float(val_seen)

        self.writer.write_scalars(
            epoch,
            {'pretext_validation_loss': validation_loss})
        if validation_loss < self.best_early_stop_loss:
          self.best_early_stop_loss = validation_loss
          self.early_stop_params = params
          self.early_stop_state = state
          self.patience_counter = 0
        else:
          self.patience_counter += 1

        if self.patience_counter > self.patience:
          break
      else:
        self.early_stop_params = params
        self.early_stop_state = state

    return self.early_stop_params, self.early_stop_state
