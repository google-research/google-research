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

"""DINO pretext.

Maintains a queue of embeddings.  For each data in the batch, corrupt it
twice into two views: view1 and view2.  Compute a list of similarities for each
view: dist1=sim(view1,queue) and dist2(view2,queue).  The loss for this algo
is then the cross entropy between the two distributions.
"""

from absl import logging
from flax.core import freeze
import jax
import jax.numpy as jnp

from q_match.algorithms.training_algo import PretextTrainingAlgo
from q_match.algorithms.vime_pretext_training import vime_corruption


@jax.jit
def entropy(teacher_logits, student_logits,
            teacher_temperature, student_temperature):
  teacher_logits /= teacher_temperature
  student_logits /= student_temperature

  targets = jax.nn.softmax(teacher_logits, axis=-1)
  loss = jnp.sum(-targets * jax.nn.log_softmax(student_logits, axis=-1),
                 axis=-1)
  return loss


@jax.jit
def dino_loss(teacher_embs_1,
              teacher_embs_2,
              student_embs_1,
              student_embs_2,
              center,
              teacher_temperature=0.04,
              student_temperature=0.1):
  """Dina loss.

  Usually teacher temp is lower than student temperature.

  Args:
    teacher_embs_1:  The embeddings to use as the target view 1 logits.
    teacher_embs_2:  The embeddings to use as the target view 2 logits.
    student_embs_1: The embeddings to propogate gradients view 1 logits.
    student_embs_2: The embeddings to propogate gradients view 2 logits.
    center: Centering used for teacher embeddings.
    teacher_temperature: Scaling temp for the teacher.
    student_temperature: Scaling temp for the student.

  Returns:
    Loss for how closes the student distribution matches the teacher
    distribution.
  """
  teacher_embs_1 = jax.lax.stop_gradient(teacher_embs_1 - center)
  teacher_embs_2 = jax.lax.stop_gradient(teacher_embs_2 - center)

  entropy_1 = entropy(teacher_embs_1, student_embs_2,
                      teacher_temperature, student_temperature)
  entropy_2 = entropy(teacher_embs_2, student_embs_1,
                      teacher_temperature, student_temperature)

  return jnp.mean(entropy_1 + entropy_2) / 2


@jax.jit
def _update_ema_params(ema_params, new_params, tau):
  """Returns new EMA params."""
  return jax.tree.map(lambda x, y: x * tau + (1. - tau) * y, ema_params,
                      new_params)


@jax.jit
def _update_center(center, new_center, tau):
  """Returns new Center params."""
  return jax.tree.map(lambda x, y: x * tau + (1. - tau) * y, center,
                      new_center)


class DinoPretextTraining(PretextTrainingAlgo):
  """Dino Training Algorithm.

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
      student_temperature=0.1,
      teacher_temperature=0.04,
      patience=32,
      use_momentum_encoder=True,
      tau=.999,
      **kwargs
  ):

    super(DinoPretextTraining,
          self).__init__(logdir, dataset, batch_size, model, eval_model,
                         learning_rate, epochs, params, state, writer,
                         weight_decay, patience=patience)

    self.mask_key = jax.random.PRNGKey(99)
    self.corruption_p = corruption_p
    self.student_temperature = student_temperature
    self.teacher_temperature = teacher_temperature
    self.use_momentum_encoder = use_momentum_encoder
    self.tau = tau

  def _loss(
      self, params, state, teacher_state, center, features, mask_key, ema_params
  ):
    """Loss with distribution match."""

    variables = freeze({'params': params, **state})
    if self.use_momentum_encoder:
      variables_2 = freeze({'params': ema_params, **teacher_state})
    else:
      variables_2 = freeze({'params': params, **state})

    variables_2 = jax.lax.stop_gradient(variables_2)

    ## View 1
    view_1_features, _ = vime_corruption(features, self.corruption_p,
                                         mask_key)

    # Student View 1
    output_s1, updated_state = self.model.apply(variables,
                                                view_1_features,
                                                mutable=['batch_stats'],
                                                rngs=self.rngs)

    # pretext_output = output['pretext']
    encoded_s1 = output_s1['pretext']['protos']
    student_embs_1 = encoded_s1

    # Teacher View 1
    output_t1, _ = self.model.apply(
        variables_2, view_1_features, mutable=['batch_stats'], rngs=self.rngs)

    encoded_t1 = output_t1['pretext']['protos']
    teacher_embs_1 = encoded_t1

    ## View 2
    # Use the first key later, so pick second.
    _, new_mask_key = jax.random.split(self.mask_key)
    view_2_features, _ = vime_corruption(
        features, p=self.corruption_p, mask_key=new_mask_key)

    # Student View 2
    output_s2, updated_state = self.model.apply(variables,
                                                view_2_features,
                                                mutable=['batch_stats'],
                                                rngs=self.rngs)

    encoded_s2 = output_s2['pretext']['protos']
    student_embs_2 = encoded_s2

    # Teacher View 2
    output_t2, updated_teacher_state_2 = self.model.apply(
        variables_2, view_2_features, mutable=['batch_stats'], rngs=self.rngs)

    encoded_t2 = output_t2['pretext']['protos']
    teacher_embs_2 = encoded_t2

    new_center = ((encoded_t2 + encoded_t1) / 2).mean(axis=0)

    pretext_loss = dino_loss(
        teacher_embs_1=teacher_embs_1,
        teacher_embs_2=teacher_embs_2,
        student_embs_1=student_embs_1,
        student_embs_2=student_embs_2,
        center=center,
        student_temperature=self.student_temperature,
        teacher_temperature=self.teacher_temperature)

    return pretext_loss, (updated_state, updated_teacher_state_2, new_center)

  def run(self,):
    """Runs a pretext training algo."""
    params = self.params
    state = self.state
    dataset = self.dataset
    model = self.model
    ema_params = jax.tree_util.tree_map(jax.numpy.copy, params)
    teacher_state = jax.tree_util.tree_map(jax.numpy.copy, state)

    example_data = jax.numpy.array(dataset.get_example_features())
    variables = freeze({'params': params, **state})
    example_output, _ = model.apply(variables,
                                    example_data,
                                    mutable=['batch_stats'],
                                    rngs=self.rngs,
                                    )
    logging.debug(str(example_output))

    # initialize center
    center = example_output['pretext']['protos'].mean(axis=0)
    center = jax.tree_util.tree_map(jax.numpy.copy, center)

    optimizer_state = self.optimizer.init(params=params)

    grad_fn = self.get_grad_fn()

    steps = 0
    for epoch in range(self.epochs):
      logging.info('Pretext Epoch: %d', epoch)
      for example in dataset.get_pretext_ds():
        features = jax.numpy.array(example['features'])

        if steps % 100 == 0:
          pretext_loss, _ = self.loss(
              params, state, teacher_state, center, features,
              self.mask_key,
              ema_params)
          log_train_loss_msg = f'pretext training loss {pretext_loss}'
          logging.info(log_train_loss_msg)

          metrics = {'pretext_train_loss': pretext_loss,}

          if self.writer is not None:
            self.writer.write_scalars(steps, metrics)

        gradients, (state, teacher_state,
                    new_center) = grad_fn(params, state, teacher_state,
                                          center, features, self.mask_key,
                                          ema_params)

        params, optimizer_state = self.update_model(params,
                                                    gradients,
                                                    optimizer_state)
        self.update_rngs()
        ema_params = _update_ema_params(ema_params, params, self.tau)
        center = _update_center(center, new_center, self.tau)
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
              teacher_state,
              center,
              features,
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
