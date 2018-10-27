# coding=utf-8
# Copyright 2018 The Google Research Authors.
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

"""QANet models for SQuAD v1.1."""

from qanet import embedding
from qanet import model_base


class BaseEmbeddingLayer(embedding.QANetEmbeddingLayer):
  """Embedding layer using word vectors."""

  @staticmethod
  def _config():
    config = embedding.QANetEmbeddingLayer._config()
    config.update({
        'mt_elmo': False,
        'use_glove': True,
        'use_char': True,
        'elmo': False,
    })
    return config


class QANet(model_base.BaseQANetModel):
  """Config for the basic QANet model for SQuAD V1 with word vectors.

  """

  @staticmethod
  def _config():
    config = model_base.BaseQANetModel._config()
    del config['initializer']
    config.update({
        'embedding': BaseEmbeddingLayer,
        'encoder_emb': model_base.QANetEncoder,
        'encoder_model': model_base.QANetEncoder,
        'max_answer_size': 30,
        'input_keep_prob': 0.9,
        'output_keep_prob': 0.9,
        'hw_keep_prob': 0.8,
    })
    return config
