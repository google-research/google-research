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

"""Npair iMix Pretext Training Algorithm.

Follows MixUp and other implementations here:
https://arxiv.org/pdf/2010.08887.pdf

Model must have outputs of form:
  {'main': {'logits': ...,
            'encoded': ...,
            ...}
  }

for classification.
"""

import functools

from absl import logging
from flax.core import freeze
import jax

from q_match.algorithms.training_algo import PretextTrainingAlgo


def cross_entropy(logits, target_class):
  logprobs = jax.nn.log_softmax(logits)
  nll = jax.numpy.take_along_axis(logprobs,
                                  jax.numpy.expand_dims(target_class, axis=-1),
                                  axis=-1)
  return -nll


@functools.partial(jax.jit, static_argnums=(2,))
def mixup(features, key, alpha=2.):
  """Mixes up features."""
  lam = jax.random.beta(key=key, a=alpha, b=alpha,
                        shape=(features.shape[0], 1))  # (batchsize, 1)
  lam = jax.numpy.maximum(lam, 1.-lam)
  randidx = jax.random.permutation(key=key, x=features.shape[0])
  mixedup_features = lam*features+(1.-lam)*features[randidx]
  return mixedup_features, lam, randidx


class NPairiMixPretextTraining(PretextTrainingAlgo):
  """N-Pair iMix Pretext Training Algorithm.

  Attributes:
    logdir: location of the log directory.
    dataset: tf dataset to train.
    batch_size: batch size for training.
    model: training model
    eval_model: evaluation model
    learning_rate: the learning rate for training.
    epochs: number of epochs to train for
    params: Optional params to start training from.  If None, random params
      are initialized.
    state: Optional state to start training from.
    writer: Writer for writing to tensorboard.
    pretext_key: The PRN key for the randomness in pretext training.
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
               pretext_key=None,
               weight_decay=0.,
               patience=32,
               temperature=0.1,
               **kwargs):
    super(NPairiMixPretextTraining, self).__init__(logdir, dataset,
                                                   batch_size, model,
                                                   eval_model, learning_rate,
                                                   epochs, params, state,
                                                   writer, weight_decay,
                                                   patience=patience)
    if pretext_key is None:
      self.pretext_key = jax.random.PRNGKey(0)
    self.temperature = temperature

  def _loss(self,
            params,
            state,
            features,
            pretext_key,
            temperature=.1,
            alpha=1.):
    """Pretext loss from iMix paper."""
    # compute the MixUp of the batch
    mixedup_features, lam, randidx = mixup(features, pretext_key, alpha=alpha)

    variables = freeze({'params': params, **state})

    a_output, updated_state = self.model.apply(variables, features,
                                               mutable=['batch_stats'],
                                               rngs=self.rngs)
    a_encoded = a_output['pretext']['proj']
    b_output, _ = self.model.apply(variables, mixedup_features,
                                   mutable=['batch_stats'],
                                   rngs=self.rngs)
    b_encoded = b_output['pretext']['proj']

    a_norms = jax.numpy.linalg.norm(a_encoded, ord=2, axis=-1, keepdims=True)
    a_normalized = a_encoded/a_norms
    b_norms = jax.numpy.linalg.norm(b_encoded, ord=2, axis=-1, keepdims=True)
    b_normalized = b_encoded/b_norms

    logits = a_normalized@b_normalized.T/temperature

    cross_entropy(logits, jax.numpy.arange(features.shape[0]))

    pretext_loss = lam*cross_entropy(logits,
                                     jax.numpy.arange(features.shape[0]))
    pretext_loss += (1.-lam) * cross_entropy(logits, randidx)
    pretext_loss = jax.numpy.mean(pretext_loss)
    return pretext_loss, updated_state

  def run(self,):
    """Runs a pretext training algo."""
    params = self.params
    state = self.state
    dataset = self.dataset
    pretext_key = self.pretext_key

    optimizer_state = self.optimizer.init(params=params)

    grad_fn = self.get_grad_fn()

    steps = 0
    for epoch in range(self.epochs):
      logging.info('Pretext Epoch: %d', epoch)
      for example in dataset.get_pretext_ds():
        features = jax.numpy.array(example['features'])

        if steps % 100 == 0:
          pretext_loss, _ = self.loss(
              params,
              state,
              features,
              pretext_key,
              temperature=self.temperature)
          log_train_loss_msg = f'pretext training loss {pretext_loss}'
          logging.info(log_train_loss_msg)

          metrics = {'pretext_train_loss': pretext_loss,
                     }

          if self.writer is not None:
            self.writer.write_scalars(steps, metrics)

        gradients, state = grad_fn(params, state, features, pretext_key,
                                   temperature=self.temperature)
        params, optimizer_state = self.update_model(params, gradients,
                                                    optimizer_state)
        self.update_rngs()
        pretext_key, _ = jax.random.split(pretext_key)
        steps += 1

      # # check validation pretext dataset if it exists
      pretext_validation_ds = dataset.get_pretext_validation_ds()
      if pretext_validation_ds is not None:
        # compute validation loss
        validation_loss = 0.
        val_seen = 0
        val_mask_key = self.pretext_key
        for example in pretext_validation_ds:
          features = jax.numpy.array(example['features'])
          seen = features.shape[0]
          validation_loss += self.loss(
              params,
              state,
              features,
              val_mask_key,
              temperature=self.temperature)[0] * seen
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
