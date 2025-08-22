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

"""Tabnet pretext training algo.

Corrupt the input x with a binary mask m, to create xt.
Pass xt through the encoder, then predict the missing values.

See paper here (page 5):
https://arxiv.org/pdf/1908.07442.pdf
"""

from absl import logging

from flax.core import freeze
import jax

from q_match.algorithms.training_algo import PretextTrainingAlgo
from q_match.algorithms.vime_pretext_training import vime_corruption


@jax.jit
def loss_reconstruction(reconstruction, original, mask, eps=.001):
  """Computes the reconstruction loss from the paper.

  Only includes components for which the orignal values were dropped.
  See details on page 5 here: https://arxiv.org/pdf/1908.07442.pdf.

  each guess is normalized by std.  see implemenatation here:
  https://github.com/dreamquark-ai/tabnet/blob/develop/pytorch_tabnet/metrics.py#L46

  Args:
    reconstruction: the reconstructed feature matrix.
    original: the original feature matrix. (Batch, Dim)
    mask: binary matrix where 1 indicates the feature was dropped.
    eps: numerical stability value

  Returns:
    Average of normalized loss.
  """
  imputation_guess = reconstruction*mask
  target = original*mask

  diff = (imputation_guess-target)
  stds = jax.numpy.std(original,
                       axis=list(range(len(original.shape)-1)))

  # replace zero std with pop mean
  zero_std_indicator = jax.numpy.array(jax.numpy.isclose(stds, 0),
                                       dtype=jax.numpy.float32)
  means = jax.numpy.mean(original,
                         axis=list(range(len(original.shape)-1)))
  stds = (1.-zero_std_indicator)*stds+zero_std_indicator*means
  stds = jax.numpy.maximum(stds, eps)
  normalized_diff = diff/stds
  return jax.numpy.sum(normalized_diff**2)/(jax.numpy.sum(mask)+eps)


def tabnet_corruption(inputs, p, mask_key):
  """Corrupts according to Tabnet."""
  if mask_key is None:
    mask = jax.numpy.zeros_like(inputs)
  else:
    mask = jax.numpy.array(
        jax.random.bernoulli(mask_key, p=p, shape=inputs.shape),
        dtype=jax.numpy.float32)

  mask_compliment = 1. - mask
  corrupted_input = inputs * mask_compliment

  return corrupted_input, mask


class TabnetPretextTraining(PretextTrainingAlgo):
  """Tabnet Pretext Training Algorithm.

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
    mask_key: The PRN key for the mask randomness.
    weight_decay: weight decay on pretext params.
  """

  def __init__(self,
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
               mask_key=None,
               weight_decay=0.,
               corruption_p=.5,
               patience=32,
               **kwargs):
    super(TabnetPretextTraining, self).__init__(logdir, dataset,
                                                batch_size, model,
                                                eval_model, learning_rate,
                                                epochs, params, state, writer,
                                                weight_decay, patience=patience)
    if mask_key is None:
      self.mask_key = jax.random.PRNGKey(0)

    self.corruption_p = corruption_p

  def _loss(self, params, state, features, mask_key):
    """Pretext loss from tabnet paper."""
    corrupted_features, mask = vime_corruption(features,
                                               self.corruption_p,
                                               mask_key=mask_key)
    variables = freeze({'params': params, **state})
    output, updated_state = self.model.apply(
        variables,
        corrupted_features,
        mutable=['batch_stats'],
        rngs=self.rngs)

    reconstruction = output['pretext']['reconstruction']

    reconstruction_loss = loss_reconstruction(reconstruction, features, mask)
    pretext_loss = reconstruction_loss
    return pretext_loss, updated_state

  def run(self,):
    """Runs a pretext training algo."""
    params = self.params
    state = self.state
    dataset = self.dataset
    mask_key = self.mask_key

    optimizer_state = self.optimizer.init(params=params)

    grad_fn = self.get_grad_fn()

    steps = 0
    for epoch in range(self.epochs):
      logging.info('Pretext Epoch: %d', epoch)
      for example in dataset.get_pretext_ds():
        features = jax.numpy.array(example['features'])

        if steps % 100 == 0:
          pretext_loss, _ = self.loss(params, state, features, mask_key)
          log_train_loss_msg = f'pretext training loss {pretext_loss}'
          logging.info(log_train_loss_msg)

          metrics = {'pretext_train_loss': pretext_loss,
                     }

          if self.writer is not None:
            self.writer.write_scalars(steps, metrics)

        gradients, state = grad_fn(params, state, features, mask_key)
        params, optimizer_state = self.update_model(params, gradients,
                                                    optimizer_state)
        self.update_rngs()
        mask_key, _ = jax.random.split(mask_key)
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
          validation_loss += self.loss(params, state, features,
                                       val_mask_key)[0]*seen
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
