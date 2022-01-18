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

from parlai.scripts.train_model import setup_args, TrainLoop


class MyTrainLoop(TrainLoop):

  def save_model(self, suffix=None):
    if suffix == '.checkpoint':
      suffix = f"{int(self._total_epochs)}" + suffix
    return super().save_model(suffix)


if __name__ == '__main__':
  parser = setup_args()
  parser.set_defaults(
      task='airdialogue:agent:5000',
      model='models.agents:AgentAgent',
      model_file='outputs/agent_5K/model',
      name_vec_len=10,
      save_after_valid=False,
      wei_name=1,
      tensorboard_log=True,
      dict_lower=True,
      dict_tokenizer='bpe',
      n_layers=5,
      n_heads=2,
      dropout=0.20,
      ffn_size=512,
      embedding_size=256,
      log_every_n_secs=10,
      validation_patience=5,
      validation_metric='ppl',
      validation_metric_mode='min',
      validation_every_n_epochs=1,
      valid_size=100,
      n_positions=512,
      truncate=512,
      learningrate=5e-4,
      warmup_updates=5000,
      clip=0.1,
      lr_scheduler='invsqrt',
      embedding_type='random',
      beam_size=1,
      skip_generation=True,
      batchsize=80,
  )
  MyTrainLoop(parser.parse_args()).train()
