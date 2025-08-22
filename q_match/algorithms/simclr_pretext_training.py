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

"""SimCLR.

https://arxiv.org/pdf/2002.05709.pdf
"""

from absl import logging
from flax.core import freeze
import jax
import jax.numpy as jnp
import optax

from q_match.algorithms.training_algo import l2_normalize
from q_match.algorithms.training_algo import PretextTrainingAlgo
from q_match.algorithms.vime_pretext_training import vime_corruption

LARGE_NUM = 1e9


@jax.jit
def modified_loss(z_a,
                  z_b,
                  temperature=.1):
  """Modified loss.

  We use the diagonal elements of z_a @ z_b.T as the positives, and the
  off-diagonal elements as the negatives.

  A big difference is that the augmented pairs of the same example are not
  included as a negatives in this implementation.


  Args:
    z_a:  projection from view 1 (n x d).
    z_b:  projection from view 2 (n x d).
    temperature: Temperature for softmax.

  Returns:
    Loss value
  """
  z_a = l2_normalize(z_a)
  z_b = l2_normalize(z_b)

  similarity_matrix = jnp.exp(
      z_a @ z_b.T / temperature
  )  # (n x d) (d x n) = n x n

  mask = jnp.eye(similarity_matrix.shape[0], similarity_matrix.shape[1])

  positives = jnp.sum(similarity_matrix * mask, axis=1)  # (n, )
  negatives = jnp.sum(similarity_matrix * (1. - mask))  # (n, )

  logits = positives / negatives

  return jnp.mean(-jnp.log(logits))


@jax.jit
def simclr_loss(z_a, z_b, temperature=.1):
  """SimCLR loss."""
  batch_size = z_a.shape[0]
  labels = jax.nn.one_hot(jax.numpy.arange(start=0, stop=batch_size),
                          batch_size * 2)  # (n x 2n)
  masks = jax.nn.one_hot(jax.numpy.arange(start=0, stop=batch_size),
                         batch_size)  # (n x n)
  logits_aa = (z_a @ z_a.T) / temperature  # (n x n)
  logits_aa = logits_aa - masks * LARGE_NUM
  logits_bb = (z_b @ z_b.T) / temperature
  logits_bb = logits_bb - masks * LARGE_NUM
  logits_ab = z_a @ z_b.T / temperature
  logits_ba = z_b @ z_a.T / temperature

  logits_a = jax.numpy.concatenate([logits_ab, logits_aa], axis=1)  # (n x 2n)
  logits_b = jax.numpy.concatenate([logits_ba, logits_bb], axis=1)  # (n x 2n)
  loss_a = optax.softmax_cross_entropy(logits_a, labels)  # (n,)
  loss_b = optax.softmax_cross_entropy(logits_b, labels)  # (n,)

  return jax.numpy.mean(loss_a + loss_b)


class SimCLRPretextTraining(PretextTrainingAlgo):
  """SimCLR Training Algorithm.

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
    use_modified_loss: Boolean to use the modified loss function.
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
      weight_decay=0.,
      corruption_p=.3,
      patience=32,
      use_modified_loss=True,
      **kwargs
  ):

    super(SimCLRPretextTraining,
          self).__init__(logdir, dataset, batch_size, model, eval_model,
                         learning_rate, epochs, params, state, writer,
                         weight_decay, patience=patience)

    self.mask_key = jax.random.PRNGKey(99)
    self.corruption_p = corruption_p
    self.use_modified_loss = use_modified_loss

  def _loss(self, params, state,
            features, mask_key,
            ):
    """Loss with siam siam."""

    variables = freeze({'params': params, **state})

    ## View 1
    view_1_features, _ = vime_corruption(features, self.corruption_p,
                                         mask_key)
    ## View 2
    # Use the first key later, so pick second.
    _, new_mask_key = jax.random.split(self.mask_key)
    view_2_features, _ = vime_corruption(
        features, p=self.corruption_p, mask_key=new_mask_key)

    # View 1 Encoded
    output_1, updated_state = self.model.apply(variables,
                                               view_1_features,
                                               mutable=['batch_stats'],
                                               rngs=self.rngs)

    proj_1 = output_1['pretext']['proj']

    # View 2 Encoded
    output_2, updated_state = self.model.apply(variables,
                                               view_2_features,
                                               mutable=['batch_stats'],
                                               rngs=self.rngs)
    proj_2 = output_2['pretext']['proj']

    if self.use_modified_loss:
      pretext_loss = modified_loss(z_a=proj_1, z_b=proj_2)
    else:
      pretext_loss = simclr_loss(z_a=proj_1, z_b=proj_2)

    return pretext_loss, updated_state

  def run(self,):
    """Runs a pretext training algo."""
    params = self.params
    state = self.state
    dataset = self.dataset
    model = self.model

    example_data = jax.numpy.array(dataset.get_example_features())
    variables = freeze({'params': params, **state})
    example_output, _ = model.apply(variables,
                                    example_data,
                                    mutable=['batch_stats'],
                                    rngs=self.rngs,
                                    )
    logging.debug(str(example_output))

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
              self.mask_key,
              )
          log_train_loss_msg = f'pretext training loss {pretext_loss}'
          logging.info(log_train_loss_msg)

          metrics = {'pretext_train_loss': pretext_loss,}

          if self.writer is not None:
            self.writer.write_scalars(steps, metrics)

        gradients, state = grad_fn(params, state,
                                   features, self.mask_key,
                                   )

        params, optimizer_state = self.update_model(params,
                                                    gradients,
                                                    optimizer_state)
        self.update_rngs()
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
              val_mask_key,
              )[0] * seen
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
