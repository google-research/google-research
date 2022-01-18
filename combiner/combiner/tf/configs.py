# coding=utf-8
# Copyright 2022 The Google Research Authors.
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
from argparse import Namespace
import tensorflow.compat.v1 as tf
import copy


def validate_config(cfg):
  if cfg.model.embed_size is None:
    cfg.model.embed_size = cfg.model.model_size
  if cfg.tpu.use_bfloat16:
    cfg.model.dtype = tf.bfloat16
  else:
    cfg.model.dtype = tf.float32
  return cfg


class LM():
  """LM config."""
  cfg = Namespace()
  cfg.dataset = 'wiki40b'
  # model related params.
  cfg.model = Namespace(embed_size=768,
                   model_size=768,
                   num_heads=12,
                   dropout=0.0,
                   dropatt=0.0,
                   num_layers=12,
                   vocab_size=1024,
                   max_seq_len=2048,
                   embedding_init_std=0.02,
                   pos_sine_init=False,
                   dense_use_bias=True,
                   inner_prod=True,
                   att_type=None)
  # tpu related params.
  cfg.tpu = Namespace(iterations_per_loop=5000,
                 save_checkpoints_steps=5000,
                 keep_checkpoint_max=10,
                 use_bfloat16=True)
  # train related params.
  cfg.train = Namespace(seq_len=2048,
                   lr_max=3e-4,
                   lr_min=0.0,
                   warmup_steps=10_000,
                   batch_size=128,
                   steps=125_000,
                   weight_decay=0,
                   lr_schedule='cosine',
                   grad_clip=1.0,
                   optimizer='adamw')
  # eval related params.
  cfg.eval = Namespace(seq_len=2048,
                  batch_size=8,
                  steps=None)
  # path related params.
  cfg.path = Namespace(ckpt_dir=None)
  # initialization
  cfg.init = Namespace(warm_start_mode='restore_train',
                  warm_start_from=None)
  # validate config
  cfg = validate_config(cfg)


class AxialRowMajorLM(LM):
  """AxialRowMajor LM config."""
  cfg = copy.deepcopy(LM.cfg)
  cfg.model.att_type = 'axial_rowmajor'
  cfg.model.row_summary = 'proj'
  cfg.model.max_seg_len = 128


class SqrtFixedFullLM(LM):
  """sqrt fixed full LM config."""
  cfg = copy.deepcopy(LM.cfg)
  cfg.model.max_seg_len = 128
  cfg.model.local_summary = 'identity-max'
  cfg.model.att_type = 'sqrt_fixed_full'


class AxialMixtureLM(LM):
  """AxialMixture LM config."""
  cfg = copy.deepcopy(LM.cfg)
  cfg.model.att_type = 'axial_mixture_unidir'
  cfg.model.row_summary = 'proj'
  cfg.model.max_seg_len = 128
