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

"""VICReg pretext.

https://openreview.net/pdf?id=xm6YD62D1Ub

https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
"""

from absl import logging
from flax.core import freeze
import jax
import jax.numpy as jnp

from q_match.algorithms.training_algo import PretextTrainingAlgo
from q_match.algorithms.vime_pretext_training import vime_corruption


def l2_normalize(
    x,
    axis = -1,
    epsilon = 1e-12,
):
  """L2 normalize a tensor on an axis with numerical stability."""
  norm = jnp.linalg.norm(x, ord=2, axis=axis, keepdims=True)
  return x/jnp.maximum(norm, epsilon)


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
def off_diagonal(x):
  """Returns the diagonal zero'd out."""
  n, m = x.shape
  assert n == m
  mask = jnp.ones_like(x, dtype=x.dtype) - jnp.eye(n)
  return x * mask


@jax.jit
def vicreg_loss(z_a,
                z_b,
                lamb=25.,
                mu=25.,
                nu=1.):
  """vicreg loss.


  Args:
    z_a:  encoding from view 1 (n x d).
    z_b:  encoding from view 2 (n x d).
    lamb: weight of invariance loss.
    mu: weight of variance loss.
    nu: weight of covariance loss.

  Returns:
    Loss value
  """
  n = z_a.shape[0]
  d = z_a.shape[1]

  # Invariance Loss
  sim_loss = mse_distance(z_a, z_b).mean()

  # Variance Loss
  std_z_a = jnp.sqrt(z_a.var(axis=0) + 1e-04)
  std_z_b = jnp.sqrt(z_b.var(axis=0) + 1e-04)
  std_loss = (jnp.mean(jax.nn.relu(1 - std_z_a))
              + jnp.mean(jax.nn.relu(1 - std_z_b)))

  # Covariance Loss
  z_a = z_a - z_a.mean(axis=0)
  z_b = z_b - z_b.mean(axis=0)
  cov_z_a = (z_a.T @ z_a) / (n - 1.)
  cov_z_b = (z_b.T @ z_b) / (n - 1.)
  cov_loss = (jnp.power(off_diagonal(cov_z_a), 2).sum() / d
              + jnp.power(off_diagonal(cov_z_b), 2).sum() / d)

  return lamb * sim_loss + mu * std_loss + nu * cov_loss


class VICRegPretextTraining(PretextTrainingAlgo):
  """VICReg Training Algorithm.

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

    super(VICRegPretextTraining,
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

    # View 1 Encoded
    output_1, updated_state = self.model.apply(variables,
                                               view_1_features,
                                               mutable=['batch_stats'],
                                               rngs=self.rngs)

    proj_1 = output_1['pretext']['encoded']

    # View 2 Encoded
    output_2, updated_state = self.model.apply(variables,
                                               view_2_features,
                                               mutable=['batch_stats'],
                                               rngs=self.rngs)
    proj_2 = output_2['pretext']['encoded']

    pretext_loss = vicreg_loss(z_a=proj_1, z_b=proj_2)

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
