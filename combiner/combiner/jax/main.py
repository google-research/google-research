# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

# pylint: skip-file
from typing import Callable, Any

import jax
from flax import optim
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from jax import lax
from flax.training.common_utils import onehot
from combiner.jax.model.util import shift_right
from combiner.jax.model.layer import AxialMixtureSelfAttLayer, SelfAttAxialRowmajorLayer, SelfAttLognLayer, SelfAttSqrtnLayer
from combiner.jax.model.net import AutoregTransformer
from combiner.jax.model.transformer_base import TransformerConfig
from functools import partial

from tqdm import tqdm

import argparse

cmd_opt = argparse.ArgumentParser(description='Argparser for combiner', allow_abbrev=False)
cmd_opt.add_argument('-model', default=None, help='combiner variants', choices=['mixture', 'axial', 'fixed', 'logn'])
cmd_args, _ = cmd_opt.parse_known_args()


class LanguageModel(nn.Module):
  vocab_size: int
  autoreg_model: Any
  emb_dim: int = 512

  @nn.compact
  def __call__(self, inputs):
    tok_embed = nn.Embed(num_embeddings=self.vocab_size,
                         features=self.emb_dim,
                         embedding_init=nn.initializers.normal(stddev=1.0))

    input_embed = tok_embed(inputs.astype('int32'))
    logits = self.autoreg_model(input_embed)
    return logits


@partial(jax.jit, static_argnums=0)
def train_step(model, x, optimizer, permute_key, dropout_key):

  def loss(params):
    x_in = shift_right(x)
    logits = model.apply(params, x_in, rngs={"permute": permute_key,
                                             "dropout": dropout_key})
    log_prob = nn.log_softmax(logits)
    x_onehot = onehot(x, num_classes=10)
    nll = -jnp.sum(x_onehot * log_prob, axis=-1)
    return jnp.mean(nll)

  l, grads = jax.value_and_grad(loss)(optimizer.target)
  optimizer = optimizer.apply_gradient(grads)
  return l, optimizer


def predict(apply_func, params, x, max_len=8):
  x_in = jnp.zeros(x.shape)
  tok_pred = x[:, 0]
  x_in = jax.ops.index_update(x_in, jax.ops.index[:, 1], tok_pred)

  list_pred = [tok_pred]
  for i in range(1, max_len):
    logits = apply_func(params, x_in)
    cur_logit = logits[:, i, :]
    tok_pred = jnp.argmax(cur_logit, axis=-1)
    if i + 1 < max_len:
      x_in = jax.ops.index_update(x_in, jax.ops.index[:, i + 1], tok_pred)
    list_pred.append(tok_pred)
  pred = jnp.array(list_pred).T
  print(pred)


if __name__ == '__main__':
  key = jax.random.PRNGKey(1)
  vocab_size = 10
  net_dim = 16
  window_size = 3
  max_len = 12
  train_cfg = TransformerConfig(
    vocab_size=vocab_size,
    output_vocab_size=vocab_size,
    num_heads=2,
    num_layers=2,
    max_len=max_len,
    emb_dim=net_dim,
    qkv_dim=net_dim,
    mlp_dim=net_dim,
    max_seg_len=4,
    dropout_rate=0,
    attention_dropout_rate=0.001,
    deterministic=False,
    seq_summary='pool-max',
    window_size=window_size,
    share_param=False
  )
  eval_config = train_cfg.replace(deterministic=True)
  if cmd_args.model == 'mixture':
    layer_cls = AxialMixtureSelfAttLayer
  elif cmd_args.model == 'axial':
    layer_cls = SelfAttAxialRowmajorLayer
  elif cmd_args.model == 'logn':
    layer_cls = SelfAttLognLayer
  elif cmd_args.model == 'fixed':
    layer_cls = SelfAttSqrtnLayer
  else:
    raise ValueError('unknown model type %s' % cmd_args.model)
  autoreg_model = lambda cfg: AutoregTransformer(config=cfg,
                                                 transformer_layer=partial(layer_cls, cfg), pred_dim=vocab_size)
  eval_model = LanguageModel(vocab_size=vocab_size,
                             autoreg_model=autoreg_model(eval_config),
                             emb_dim=16)
  x = jnp.array([[9, 8, 7, 6, 5, 4, 3, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5]])
  x_in = shift_right(x)
  key, model_key = jax.random.split(key)
  params = eval_model.init(key, x_in)

  optimizer_def = optim.Adam(0.001)
  optimizer = optimizer_def.create(params)

  pbar = tqdm(range(5000))
  pred_func = jax.jit(eval_model.apply)
  train_model = LanguageModel(vocab_size=vocab_size,
                             autoreg_model=autoreg_model(train_cfg),
                             emb_dim=16)

  for _ in pbar:
    key, permute_key, dropout_key = jax.random.split(key, num=3)
    l, optimizer = train_step(train_model, x, optimizer, permute_key, dropout_key)
    pbar.set_description('loss: %.2f' % l)
    predict(pred_func, optimizer.target, x, max_len)