# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

from parlai.scripts.train_model import setup_args, TrainLoop

if __name__ == '__main__':
  parser = setup_args()
  parser.set_defaults(
      task='airdialogue:customer',
      model='models.agents:CustomerAgent',
      model_file='outputs/customer/model',
      tensorboard_log=True,
      dict_lower=True,
      dict_tokenizer='bpe',
      n_layers=5,
      n_heads=2,
      dropout=0.20,
      ffn_size=512,
      embedding_size=256,
      log_every_n_secs=10,
      validation_patience=12,
      validation_metric='ppl',
      validation_metric_mode='min',
      validation_every_n_epochs=0.5,
      valid_size=4000,
      n_positions=512,
      truncate=512,
      learningrate=5e-4,
      warmup_updates=5000,
      clip=0.1,
      lr_scheduler='invsqrt',
      embedding_type='random',
      beam_size=1,
      skip_generation=False,
      batchsize=64,
  )
  TrainLoop(parser.parse_args()).train()
