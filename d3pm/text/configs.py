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

"""Defines and generates Gin configs and sweeps programmatically."""

import functools


def gin_load(hparam_fn, clear_config = False):
  """Load specified set of hyperparameter functions into gin.

  If multiple functions are specified (space separated), they will be combined.

  Args:
    hparam_fn: a string containing optionally multiple space separated names of
      configs in the configs directory which return dictionaries mapping gin
      configurable names to values.
    clear_config: whether to clear the gin config before loading.
  """

  if not hparam_fn:
    return

  # pylint: disable=g-import-not-at-top
  import gin

  if clear_config:
    gin.clear_config()

  print('=== %s ===' % hparam_fn)
  if ' ' in hparam_fn:
    for fn in hparam_fn.split():
      gin_load(fn)

    return

  hparam_fn = globals().get(hparam_fn, None)
  if hparam_fn is not None:
    hparams = hparam_fn()
  else:
    raise ValueError(f'Unable to find sweep "{hparam_fn}".')

  items = list(hparams.items())

  def _fmt(v):
    if isinstance(v, str):
      if v.startswith('@'):
        return v
      else:
        return '"%s"' % v
    return v

  bindings = ['%s = %s' % (k, _fmt(v)) for (k, v) in items]
  print('\n'.join(bindings))

  gin.parse_config(bindings)


def _transformer(
    emb_dim=512,
    num_heads=8,
    num_layers=6,
    qkv_dim=512,
    mlp_dim=2048,
    dropout_rate=None,
    attention_dropout_rate=None,
    nonlinearity='gelu',
):
  """Transformer config."""
  configs = {
      'models.build_transformer_config.emb_dim': emb_dim,
      'models.build_transformer_config.num_heads': num_heads,
      'models.build_transformer_config.num_decoder_layers': num_layers,
      'models.build_transformer_config.num_encoder_layers': num_layers,
      'models.build_transformer_config.qkv_dim': qkv_dim,
      'models.build_transformer_config.mlp_dim': mlp_dim,
      'models.build_transformer_config.mlp_activations': (nonlinearity,),
  }

  if dropout_rate is not None:
    configs['models.build_transformer_config.dropout_rate'] = dropout_rate

  if attention_dropout_rate is not None:
    configs[
        'models.build_transformer_config.attention_dropout_rate'] = attention_dropout_rate

  return configs


def _model_gpt(size=0, dropout_rate=0.0, attention_dropout_rate=0.0):
  """Configs for a variety of Transformer model sizes."""
  num_layers = [1, 3, 6, 12, 24, 36, 48][size]
  dim = [64, 128, 512, 768, 1024, 1280, 1600][size]
  num_heads = int(dim / 64)  # Always dim 64 per head
  return _transformer(
      emb_dim=dim,
      num_heads=num_heads,
      num_layers=num_layers,
      qkv_dim=dim,
      mlp_dim=dim * 4,
      dropout_rate=dropout_rate,
      attention_dropout_rate=attention_dropout_rate)


# Published GPT sizes.
gpt_extra_tiny = functools.partial(_model_gpt, size=0)
gpt_tiny = functools.partial(_model_gpt, size=1)
gpt_small = functools.partial(_model_gpt, size=2)
gpt_base = functools.partial(_model_gpt, size=3)
gpt_large = functools.partial(_model_gpt, size=4)


def lm1b():
  """Sets up diffusion to run with LM1B."""
  return {
      'run_experiment.dataset_name': 'lm1b',
      'datasets.load.max_length': 128,
      'datasets.load.pack': True,
      'discrete_diffusion_loss_fn.mask_padding': False,
      'discrete_diffusion_loss_fn.normalize_without_padding': False,
      'discrete_diffusion_predict_fn.mask_padding': False,
  }




def text8():
  """Sets up diffusion to run with LM1B."""
  params = {
      'run_experiment.dataset_name': 'text8',
      'datasets.load.max_length': 256,
      'datasets.load.sample_crop_train': True,
      'discrete_diffusion_loss_fn.mask_padding': False,
      'discrete_diffusion_loss_fn.normalize_without_padding': False,
      'discrete_diffusion_predict_fn.mask_padding': False,
  }

  return params




def diffusion_length(length=40):
  return {
      'CategoricalDiffusionModel.num_steps': length,
      'create_discrete_diffusion.num_steps': length,
  }


def uniform_diffusion():
  params = {
      'discrete_diffusion_loss_fn.hybrid_lambda': 0.0,
      'create_discrete_diffusion.kind': 'band-diagonal',
  }

  return params


def mask_diffusion():
  params = {
      'discrete_diffusion_loss_fn.hybrid_lambda': 0.01,
      'create_discrete_diffusion.kind': 'mask',
      'datasets.load.num_extra_tokens': 1,
  }

  return params


def diffusion():
  """Trains a BERT model with gradient clipping."""
  params = {
      'run_experiment.task_name':
          'diffusion',
      'run_experiment.model_cls':
          '@CategoricalDiffusionModel',
      'run_experiment.max_train_steps':
          10000,
      'CategoricalDiffusionModel.use_timestep_embeddings':
          True,
      'CategoricalDiffusionModel.use_film_layers':
          False,
      'run_experiment.batch_size_per_device':
          8,
      'discrete_diffusion_loss_fn.predict_x0':
          True,
      'discrete_diffusion_predict_fn.predict_x0':
          True,
      'discrete_diffusion_loss_fn.compute_elbo':
          True,
      'run_experiment.num_predict_steps':
          1,
      'run_experiment.num_eval_steps':
          10,
      'run_experiment.validate_every':
          25000,
      'create_discrete_diffusion.update_every':
          200,
      'trainers.Trainer.learning_rate_fn':
          '@learning_rate/utils.create_learning_rate_scheduler',
      'learning_rate/utils.create_learning_rate_scheduler.factors':
          'linear_warmup_from * constant',
      'learning_rate/utils.create_learning_rate_scheduler.base_learning_rate':
          2e-4,
      'learning_rate/utils.create_learning_rate_scheduler.warmup_steps':
          5000,
      'trainers.Trainer.grad_clip':
          0.25
  }

  return params


def lm1b_tiny():
  config = diffusion()
  config.update(gpt_extra_tiny())
  config.update(lm1b())
  config.update(mask_diffusion())
  config.update(diffusion_length(32))
  return config


def lm1b_base():
  config = diffusion()
  config.update(gpt_base())
  config.update(lm1b())
  config.update(mask_diffusion())
  config.update(diffusion_length(1000))
  return config


def text8_tiny():
  config = diffusion()
  config.update(gpt_extra_tiny())
  config.update(text8())
  config.update(mask_diffusion())
  config.update(diffusion_length(32))
  return config


def text8_base():
  config = diffusion()
  config.update(gpt_base())
  config.update(text8())
  config.update(mask_diffusion())
  config.update(diffusion_length(1000))
  return config


