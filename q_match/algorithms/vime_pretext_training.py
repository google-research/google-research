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

"""VIME pretext training algo.

Corrupt the input x with a binary mask m, to create xt.
Then pass xt through an encoder to learning a represenation z.
Then pass z through two other models, r and m
which learn to predict the reconstruction and mask respectively.

See paper here:
https://vanderschaar-lab.com/papers/NeurIPS2020_VIME.pdf

Same except we don't treat the categorical variables differently.
"""

from absl import logging

from flax.core import freeze
import jax

from q_match.algorithms.training_algo import PretextTrainingAlgo


@jax.jit
def loss_reconstruction(guess, target):
  return jax.numpy.mean((guess-target)**2)


@jax.jit
def loss_mask(guess, target):
  ce = -jax.numpy.sum(target*jax.numpy.log(guess), axis=-1)
  ce += -jax.numpy.sum((1.-target)*jax.numpy.log(1.-guess), axis=-1)
  return jax.numpy.mean(ce)


@jax.jit
def vime_corruption(inputs, p, mask_key=None):
  """Corrupts inputs according to VIME.

  Args:
    inputs: Original data (batchsize, dim)
    p: Probability of corrupting an input
    mask_key: PRNG key.  If None no corruption.

  Returns:
    Corrupted input and mask indicating which values were corrupted (1 means
      corrupted).
  """
  if mask_key is None:
    mask = jax.numpy.zeros_like(inputs)
    x_bar = jax.numpy.zeros_like(inputs)
  else:
    mask = jax.numpy.array(
        jax.random.bernoulli(mask_key, p=p, shape=inputs.shape),
        dtype=jax.numpy.float32)

    # for each column, shuffle the data
    x_bar = []
    keys = jax.random.split(mask_key,
                            inputs.shape[-1])  #  batch size many keys

    def _generate_random_permutation(key):
      return jax.random.permutation(key, inputs.shape[0])
    shuffle_idxs = jax.vmap(_generate_random_permutation)(keys)
    for i in range(inputs.shape[-1]):
      x_bar.append(inputs[shuffle_idxs[i], i])
    x_bar = jax.numpy.array(x_bar).T

  mask_compliment = 1. - mask
  corrupted_input = mask * x_bar + inputs * mask_compliment
  return corrupted_input, mask


class VimePretextTraining(PretextTrainingAlgo):
  """VIME Pretext Training Algorithm.

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
    corruption_p: The probability of corrupting the input.
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
               corruption_p=.3,
               patience=32,
               **kwargs):

    super(VimePretextTraining, self).__init__(logdir, dataset,
                                              batch_size, model,
                                              eval_model, learning_rate,
                                              epochs, params, state, writer,
                                              weight_decay, patience=patience)
    if mask_key is None:
      self.mask_key = jax.random.PRNGKey(0)
    self.corruption_p = corruption_p

  def _loss(self, params, state, features, mask_key, alpha=1.0):
    """Loss with reconstruction and mask components."""
    variables = freeze({'params': params, **state})
    corrupted_features, mask = vime_corruption(features,
                                               self.corruption_p,
                                               mask_key)
    output, updated_state = self.model.apply(variables,
                                             corrupted_features,
                                             mutable=['batch_stats'],
                                             rngs=self.rngs)

    pretext_output = output['pretext']
    reconstruction = pretext_output['reconstruction']
    mask_prediction = pretext_output['mask_guess']

    reconstruction_loss = loss_reconstruction(reconstruction, features)
    mask_loss = loss_mask(mask_prediction, mask)
    pretext_loss = alpha * reconstruction_loss + mask_loss
    return (pretext_loss, (reconstruction_loss, mask_loss, updated_state))

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
          pretext_loss, (reconstruction_loss, mask_loss, _) = self.loss(
              params, state, features, mask_key)
          log_train_loss_msg = f'pretext training loss {pretext_loss}'
          logging.info(log_train_loss_msg)

          metrics = {'pretext_train_loss': pretext_loss,
                     'reconstruction_train_loss': reconstruction_loss,
                     'mask_loss': mask_loss}

          if self.writer is not None:
            self.writer.write_scalars(steps, metrics)

        gradients, (reconstruction_loss, mask_loss, state) = grad_fn(
            params, state, features, mask_key)
        params, optimizer_state = self.update_model(params,
                                                    gradients,
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
