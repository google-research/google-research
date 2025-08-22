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

"""Uses Jaxline for training."""

import functools

from absl import app
from absl import flags
import flax
import jax
from jaxline import experiment
from jaxline import platform
import models as adv_models
import optax
from praxis import base_hyperparams
import t5.data
import utils as adv_utils

instantiate = base_hyperparams.instantiate


def decay_hparam(param_config, global_step, total_steps):
  """Decays logarithmically from the init_value to end_value in total_steps."""
  return param_config.init_value * (param_config.end_value /
                                    param_config.init_value)**(
                                        global_step / total_steps)


def linear_decay_hparam(param_config, global_step, total_steps):
  """Decays linearly from the init_value to end_value in total_steps."""
  return param_config.init_value + global_step / total_steps * (
      param_config.end_value - param_config.init_value)


def gs_hard_hparam(soft_train_fract, global_step, total_steps):
  """Returns True if the global_step is past the soft_train_fract."""
  return global_step > soft_train_fract * total_steps


class Experiment(experiment.AbstractExperiment):
  """Updates the input to make the model predict something unsafe."""

  def __init__(self, mode, init_rng, config, model_states=None):
    """Loads the pretrained model. Also initializes the model input."""

    super().__init__(mode=mode, init_rng=init_rng)
    self.config = config

    init_rng = jax.random.fold_in(init_rng, jax.process_index())

    regular_task_p = adv_models.get_regular_task_p()
    self._regular_task = instantiate(regular_task_p)

    self._vocabulary = t5.data.PartialCodepointVocabulary(range(50258 - 3))
    vocab_mask = adv_utils.get_vocab_mask(
        self._vocabulary,
        config.exclude_tokens,
        config.exclude_no_space,
    )

    one_hot_lm_task_p = adv_models.get_onehot_task_p(dtype=config.dtype)
    self._one_hot_lm_task = instantiate(one_hot_lm_task_p)

    full_model_input = adv_utils.make_inputs(
        prefix=config.prefix,
        input_len=config.num_input_tokens,
        decode_len=config.num_output_tokens,
        input_for_classify=config.input_for_classify,
        vocabulary=self._vocabulary,
        vocab_mask=vocab_mask,
        dtype=config.dtype,
    )

    if model_states is None:
      model_states = adv_models.load_model_from_checkpoint(
          config.checkpoint_dir,
          regular_task_p,
          config.dtype,
      )

    model_states = model_states.replace(
        opt_states=optax.adam(1).init(full_model_input.logits))
    self._model_states = flax.jax_utils.replicate(model_states)

    self._prng_key = jax.random.split(init_rng, jax.local_device_count())

    if mode == 'train':
      self._opt_states = self._model_states.opt_states
      self._full_model_input = flax.jax_utils.replicate(full_model_input)

    else:
      self._full_model_input = None
      self._opt_states = None

  def step(self, *, global_step, rng, writer):
    """Decays each of the parameters and runs update_input_rep_par."""
    del rng
    global_step = global_step[0]

    self._model_states = self._model_states.replace(opt_states=self._opt_states)

    input_gs_temp = decay_hparam(self.config.training.input_gs.temp,
                                 global_step, self.config.training.steps)

    input_gs_hard = gs_hard_hparam(
        self.config.training.input_gs.soft_train_fract, global_step,
        self.config.training.steps)

    decode_gs_temp = decay_hparam(self.config.training.decode_gs.temp,
                                  global_step, self.config.training.steps)

    decode_gs_hard = gs_hard_hparam(
        self.config.training.decode_gs.soft_train_fract, global_step,
        self.config.training.steps)

    learning_rate = decay_hparam(self.config.training.learning_rate,
                                 global_step, self.config.training.steps)

    difference_loss_weight = linear_decay_hparam(
        self.config.training.difference_loss_weight,
        global_step,
        self.config.training.steps,
    )

    self._full_model_input.logits, self._opt_states, loss, self._prng_key = (
        adv_utils.update_input_rep_par(
            self._full_model_input,
            self._model_states,
            self._prng_key,
            learning_rate,
            {'temp': input_gs_temp, 'hard': input_gs_hard},
            {'temp': decode_gs_temp, 'hard': decode_gs_hard},
            difference_loss_weight,
            self.config.training.batch_size,
            adv_utils.WrappedModel(self._one_hot_lm_task.model),
        )
    )

    metrics = {
        'learning_rate': learning_rate,
        'input_gs_temp': input_gs_temp,
        'decode_gs_temp': decode_gs_temp,
        'input_gs_hard': input_gs_hard,
        'decode_gs_hard': decode_gs_hard,
        'difference_loss_weight': difference_loss_weight,
        'train_prob': -loss['loss'][0],
        'train_difference_loss': loss['difference_loss'][0],
    }

    return metrics

  def evaluate(self, global_step, rng, writer):
    """Evaluates with the most likely input. Also uses greedy decode."""
    print('-' * 30)
    global_step = global_step[0]
    print(f'Step {global_step}')

    full_model_input = flax.jax_utils.unreplicate(self._full_model_input)
    model_states = flax.jax_utils.unreplicate(self._model_states)

    eval_model_input = adv_utils.make_inputs(
        prefix=self.config.prefix,
        input_len=self.config.num_input_tokens,
        decode_len=self.config.num_eval_output_tokens,
        input_for_classify=self.config.input_for_classify,
        vocabulary=self._vocabulary,
        vocab_mask=full_model_input.vocab_mask,
        dtype=self.config.dtype,
    )
    eval_model_input.logits = full_model_input.logits

    print('Onehot Decode')
    # This does the evaluation decoding the same way as
    # decoding during training.
    unsafe_prob_onehot_dec, difference_loss, decoded_tokens_onehot_dec = (
        adv_utils.eval_label_prob(
            full_model_input,
            model_states,
            True,
            self._vocabulary,
            adv_utils.WrappedModel(self._one_hot_lm_task.model),
        )
    )

    print('Regular Decode')
    # This uses the decoding algorithm built into PAX.
    # Usually it has the same results as onehot_dec.
    # But sometimes the results are different due to floating point errors.
    unsafe_prob, decoded_tokens = adv_utils.eval_label_prob_reg_decode(
        eval_model_input,
        model_states,
        True,
        self._vocabulary,
        adv_utils.WrappedModel(self._regular_task.model),
        use_dec_enc=True,
    )

    metrics = {
        'unsafe_prob': unsafe_prob,
        'unsafe_prob_onehot_dec': unsafe_prob_onehot_dec,
        'difference_loss': difference_loss,
    }

    print(metrics)

    text_to_log = {
        'decoded_tokens':
            repr(decoded_tokens),
        'decoded_text':
            self._vocabulary.decode(decoded_tokens),
        'decoded_tokens_onehot_dec':
            repr(decoded_tokens_onehot_dec),
        'decoded_text_onehot_dec':
            self._vocabulary.decode(decoded_tokens_onehot_dec),
    }

    print(text_to_log)

    print('-' * 30)

    if writer is not None:
      writer.write_text(global_step, text_to_log)

    return metrics


if __name__ == '__main__':
  flags.mark_flag_as_required('config')
  app.run(functools.partial(platform.main, Experiment))
