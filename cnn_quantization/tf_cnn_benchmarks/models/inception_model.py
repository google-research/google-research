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

"""Inception model configuration.

Includes multiple models: inception3, inception4, inception-resnet2.

References:
  Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
  Inception-v4, Inception-ResNet and the Impact of Residual Connections on
  Learning

  Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
  Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich
  Going Deeper with Convolutions
  http://arxiv.org/pdf/1409.4842v1.pdf

  Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens,
  Zbigniew Wojna
  Rethinking the Inception Architecture for Computer Vision
  arXiv preprint arXiv:1512.00567 (2015)

  Inception v3 model: http://arxiv.org/abs/1512.00567

  Inception v4 and Resnet V2 architectures: http://arxiv.org/abs/1602.07261
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin
from cnn_quantization.tf_cnn_benchmarks.models import model


class Inceptionv3Model(model.CNNModel):
  """InceptionV3."""

  def __init__(self, auxiliary=False, params=None):
    self._auxiliary = auxiliary
    super(Inceptionv3Model, self).__init__(
        'inception3', 299, 32, 0.005, params=params)

  def add_inference(self, cnn):
    def inception_v3_a(cnn, n):
      cols = [[('conv', 64, 1, 1)], [('conv', 48, 1, 1), ('conv', 64, 5, 5)],
              [('conv', 64, 1, 1), ('conv', 96, 3, 3), ('conv', 96, 3, 3)],
              [('apool', 3, 3, 1, 1, 'SAME'), ('conv', n, 1, 1)]]
      cnn.inception_module('incept_v3_a', cols)

    def inception_v3_b(cnn):
      cols = [[('conv', 384, 3, 3, 2, 2, 'VALID')],
              [('conv', 64, 1, 1),
               ('conv', 96, 3, 3),
               ('conv', 96, 3, 3, 2, 2, 'VALID')],
              [('mpool', 3, 3, 2, 2, 'VALID')]]
      cnn.inception_module('incept_v3_b', cols)

    def inception_v3_c(cnn, n):
      cols = [[('conv', 192, 1, 1)],
              [('conv', n, 1, 1), ('conv', n, 1, 7), ('conv', 192, 7, 1)],
              [('conv', n, 1, 1), ('conv', n, 7, 1), ('conv', n, 1, 7),
               ('conv', n, 7, 1), ('conv', 192, 1, 7)],
              [('apool', 3, 3, 1, 1, 'SAME'), ('conv', 192, 1, 1)]]
      cnn.inception_module('incept_v3_c', cols)

    def inception_v3_d(cnn):
      cols = [[('conv', 192, 1, 1), ('conv', 320, 3, 3, 2, 2, 'VALID')],
              [('conv', 192, 1, 1), ('conv', 192, 1, 7), ('conv', 192, 7, 1),
               ('conv', 192, 3, 3, 2, 2, 'VALID')],
              [('mpool', 3, 3, 2, 2, 'VALID')]]
      cnn.inception_module('incept_v3_d', cols)

    def inception_v3_e(cnn, pooltype):
      cols = [[('conv', 320, 1, 1)], [('conv', 384, 1, 1), ('conv', 384, 1, 3)],
              [('share',), ('conv', 384, 3, 1)],
              [('conv', 448, 1, 1), ('conv', 384, 3, 3), ('conv', 384, 1, 3)],
              [('share',), ('share',), ('conv', 384, 3, 1)],
              [('mpool' if pooltype == 'max' else 'apool', 3, 3, 1, 1, 'SAME'),
               ('conv', 192, 1, 1)]]
      cnn.inception_module('incept_v3_e', cols)

    def incept_v3_aux(cnn):
      assert cnn.aux_top_layer is None
      cnn.aux_top_layer = cnn.top_layer
      cnn.aux_top_size = cnn.top_size
      with cnn.switch_to_aux_top_layer():
        cnn.apool(5, 5, 3, 3, mode='VALID')
        cnn.conv(128, 1, 1, mode='SAME')
        cnn.conv(768, 5, 5, mode='VALID', stddev=0.01)
        cnn.reshape([-1, 768])

    cnn.use_batch_norm = True
    cnn.conv(32, 3, 3, 2, 2, mode='VALID')   # 299 x 299 x 3
    cnn.conv(32, 3, 3, 1, 1, mode='VALID')   # 149 x 149 x 32
    cnn.conv(64, 3, 3, 1, 1, mode='SAME')    # 147 x 147 x 64
    cnn.mpool(3, 3, 2, 2, mode='VALID')      # 147 x 147 x 64
    cnn.conv(80, 1, 1, 1, 1, mode='VALID')   # 73 x 73 x 80
    cnn.conv(192, 3, 3, 1, 1, mode='VALID')  # 71 x 71 x 192
    cnn.mpool(3, 3, 2, 2, 'VALID')           # 35 x 35 x 192
    inception_v3_a(cnn, 32)                  # 35 x 35 x 256 mixed.
    inception_v3_a(cnn, 64)                  # 35 x 35 x 288 mixed_1.
    inception_v3_a(cnn, 64)                  # 35 x 35 x 288 mixed_2
    inception_v3_b(cnn)                      # 17 x 17 x 768 mixed_3
    inception_v3_c(cnn, 128)                 # 17 x 17 x 768 mixed_4
    inception_v3_c(cnn, 160)                 # 17 x 17 x 768 mixed_5
    inception_v3_c(cnn, 160)                 # 17 x 17 x 768 mixed_6
    inception_v3_c(cnn, 192)                 # 17 x 17 x 768 mixed_7
    if self._auxiliary:
      incept_v3_aux(cnn)                     # Auxillary Head logits
    inception_v3_d(cnn)                      # 17 x 17 x 1280 mixed_8
    inception_v3_e(cnn, 'avg')               # 8 x 8 x 2048 mixed_9
    inception_v3_e(cnn, 'max')               # 8 x 8 x 2048 mixed_10
    cnn.apool(8, 8, 1, 1, 'VALID')           # 8 x 8 x 2048
    cnn.reshape([-1, 2048])                  # 1 x 1 x 2048


# Stem functions
def inception_v4_sa(cnn):
  cols = [[('mpool', 3, 3, 2, 2, 'VALID')], [('conv', 96, 3, 3, 2, 2, 'VALID')]]
  cnn.inception_module('incept_v4_sa', cols)


def inception_v4_sb(cnn):
  cols = [[('conv', 64, 1, 1), ('conv', 96, 3, 3, 1, 1, 'VALID')],
          [('conv', 64, 1, 1), ('conv', 64, 7, 1), ('conv', 64, 1, 7),
           ('conv', 96, 3, 3, 1, 1, 'VALID')]]
  cnn.inception_module('incept_v4_sb', cols)


def inception_v4_sc(cnn):
  cols = [[('conv', 192, 3, 3, 2, 2, 'VALID')],
          [('mpool', 3, 3, 2, 2, 'VALID')]]
  cnn.inception_module('incept_v4_sc', cols)


# Reduction functions
def inception_v4_ra(cnn, k, l, m, n):
  cols = [
      [('mpool', 3, 3, 2, 2, 'VALID')], [('conv', n, 3, 3, 2, 2, 'VALID')],
      [('conv', k, 1, 1), ('conv', l, 3, 3), ('conv', m, 3, 3, 2, 2, 'VALID')]
  ]
  cnn.inception_module('incept_v4_ra', cols)


def inception_v4_rb(cnn):
  cols = [[('mpool', 3, 3, 2, 2, 'VALID')],
          [('conv', 192, 1, 1), ('conv', 192, 3, 3, 2, 2, 'VALID')],
          [('conv', 256, 1, 1), ('conv', 256, 1, 7), ('conv', 320, 7, 1),
           ('conv', 320, 3, 3, 2, 2, 'VALID')]]
  cnn.inception_module('incept_v4_rb', cols)


class Inceptionv4Model(model.CNNModel):
  """Inceptionv4."""

  def __init__(self, params=None):
    super(Inceptionv4Model, self).__init__(
        'inception4', 299, 32, 0.005, params=params)

  def add_inference(self, cnn):
    def inception_v4_a(cnn):
      cols = [[('apool', 3, 3, 1, 1, 'SAME'), ('conv', 96, 1, 1)],
              [('conv', 96, 1, 1)], [('conv', 64, 1, 1), ('conv', 96, 3, 3)],
              [('conv', 64, 1, 1), ('conv', 96, 3, 3), ('conv', 96, 3, 3)]]
      cnn.inception_module('incept_v4_a', cols)

    def inception_v4_b(cnn):
      cols = [[('apool', 3, 3, 1, 1, 'SAME'), ('conv', 128, 1, 1)],
              [('conv', 384, 1, 1)],
              [('conv', 192, 1, 1), ('conv', 224, 1, 7), ('conv', 256, 7, 1)],
              [('conv', 192, 1, 1), ('conv', 192, 1, 7), ('conv', 224, 7, 1),
               ('conv', 224, 1, 7), ('conv', 256, 7, 1)]]
      cnn.inception_module('incept_v4_b', cols)

    def inception_v4_c(cnn):
      cols = [[('apool', 3, 3, 1, 1, 'SAME'), ('conv', 256, 1, 1)],
              [('conv', 256, 1, 1)], [('conv', 384, 1, 1), ('conv', 256, 1, 3)],
              [('share',), ('conv', 256, 3, 1)],
              [('conv', 384, 1, 1), ('conv', 448, 1, 3), ('conv', 512, 3, 1),
               ('conv', 256, 3, 1)], [('share',), ('share',), ('share',),
                                      ('conv', 256, 1, 3)]]
      cnn.inception_module('incept_v4_c', cols)

    cnn.use_batch_norm = True
    cnn.conv(32, 3, 3, 2, 2, mode='VALID')
    cnn.conv(32, 3, 3, 1, 1, mode='VALID')
    cnn.conv(64, 3, 3)
    inception_v4_sa(cnn)
    inception_v4_sb(cnn)
    inception_v4_sc(cnn)
    for _ in xrange(4):
      inception_v4_a(cnn)
    inception_v4_ra(cnn, 192, 224, 256, 384)
    for _ in xrange(7):
      inception_v4_b(cnn)
    inception_v4_rb(cnn)
    for _ in xrange(3):
      inception_v4_c(cnn)
    cnn.spatial_mean()
    cnn.dropout(0.8)
