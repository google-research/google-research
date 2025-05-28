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

"""SimSiam pretext.

https://arxiv.org/pdf/2011.10566.pdf
"""

from absl import logging
from flax.core import freeze
import jax
import jax.numpy as jnp

from q_match.algorithms.training_algo import l2_normalize
from q_match.algorithms.training_algo import PretextTrainingAlgo
from q_match.algorithms.vime_pretext_training import vime_corruption


@jax.jit
def negative_cosine_similarity(x, y):
  return -(x * y).sum(axis=-1).mean()


@jax.jit
def simsiam_loss(proj_1, proj_2, pred_1, pred_2,):
  """Simsiam loss.

  Args:
    proj_1:  view 1 proj
    proj_2:  view 2 proj.
    pred_1:  view 1 pred
    pred_2:  view 2 pred.


  Returns:
    Loss value
  """
  proj_1 = jax.lax.stop_gradient(proj_1)
  proj_2 = jax.lax.stop_gradient(proj_2)

  proj_1 = l2_normalize(proj_1)
  proj_2 = l2_normalize(proj_2)
  pred_1 = l2_normalize(pred_1)
  pred_2 = l2_normalize(pred_2)

  loss_a = negative_cosine_similarity(proj_1, pred_2)
  loss_b = negative_cosine_similarity(proj_2, pred_1)

  return jnp.mean(loss_a + loss_b) / 2


class SimSiamPretextTraining(PretextTrainingAlgo):
  """SimSiam Training Algorithm.

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
      weight_decay=0.,
      corruption_p=.3,
      patience=32,
      **kwargs
  ):

    super(SimSiamPretextTraining,
          self).__init__(logdir, dataset, batch_size, model, eval_model,
                         learning_rate, epochs, params, state, writer,
                         weight_decay, patience=patience)

    self.mask_key = jax.random.PRNGKey(99)
    self.corruption_p = corruption_p

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

    # View 1 Projection and Predictor
    output_1, updated_state = self.model.apply(variables,
                                               view_1_features,
                                               mutable=['batch_stats'],
                                               rngs=self.rngs)

    proj_1 = output_1['pretext']['siam_proj']
    pred_1 = output_1['pretext']['siam_pred']

    # View 2 Projection and Predictor
    output_2, updated_state = self.model.apply(variables,
                                               view_2_features,
                                               mutable=['batch_stats'],
                                               rngs=self.rngs)
    proj_2 = output_2['pretext']['siam_proj']
    pred_2 = output_2['pretext']['siam_pred']

    pretext_loss = simsiam_loss(proj_1=proj_1, proj_2=proj_2,
                                pred_1=pred_1, pred_2=pred_2)

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
