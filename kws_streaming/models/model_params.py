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

"""Models parameters (with toy values for testing)."""


class Flags(object):
  """Default flags for data and feature settings."""

  def __init__(self):
    self.train_dir = ''
    self.batch_size = 1
    self.wanted_words = 'yes,no,up,down,left,right,on,off,stop,go'
    self.train = 0
    self.split_data = 1
    self.sample_rate = 16000
    self.clip_duration_ms = 1000
    self.window_size_ms = 40.0
    self.window_stride_ms = 20.0
    self.preprocess = 'raw'
    self.feature_type = 'mfcc_tf'
    self.preemph = 0.0
    self.window_type = 'hann'
    self.mel_lower_edge_hertz = 20.0
    self.mel_upper_edge_hertz = 7000.0
    self.log_epsilon = 1e-12
    self.dct_num_features = 20
    self.use_tf_fft = 0
    self.mel_non_zero_only = 1
    self.fft_magnitude_squared = False
    self.mel_num_bins = 40


def att_mh_rnn_params():
  """Flags for toy multihead attention model."""
  params = Flags()
  params.model_name = 'att_rnn'
  params.cnn_filters = '3,1'
  params.cnn_kernel_size = '(5,1),(5,1)'
  params.cnn_act = "'relu','relu'"
  params.cnn_dilation_rate = '(1,1),(1,1)'
  params.cnn_strides = '(1,1),(1,1)'
  params.rnn_layers = 2
  params.rnn_type = 'gru'
  params.rnn_units = 2
  params.heads = 4
  params.dropout1 = 0.1
  params.units2 = '2,2'
  params.act2 = "'relu','linear'"
  return params


def att_rnn_params():
  """Flags for toy attention model."""
  params = Flags()
  params.model_name = 'att_rnn'
  params.cnn_filters = '3,1'
  params.cnn_kernel_size = '(5,1),(5,1)'
  params.cnn_act = "'relu','relu'"
  params.cnn_dilation_rate = '(1,1),(1,1)'
  params.cnn_strides = '(1,1),(1,1)'
  params.rnn_layers = 2
  params.rnn_type = 'gru'
  params.rnn_units = 2
  params.dropout1 = 0.1
  params.units2 = '2,2'
  params.act2 = "'relu','linear'"
  return params


def dnn_params():
  """Flags for toy dnn model."""
  params = Flags()
  params.model_name = 'dnn'
  params.units1 = '16,16'
  params.act1 = "'relu','relu'"
  params.pool_size = 2
  params.strides = 2
  params.dropout1 = 0.1
  params.units2 = '16,16'
  params.act2 = "'linear','relu'"
  return params


def crnn_params():
  """Flags for toy conv rnn model."""
  params = Flags()
  params.model_name = 'crnn'
  params.cnn_filters = '16,16'
  params.cnn_kernel_size = '(3,3),(5,3)'
  params.cnn_act = "'relu','relu'"
  params.cnn_dilation_rate = '(1,1),(1,1)'
  params.cnn_strides = '(1,1),(1,1)'
  params.gru_units = '16'
  params.return_sequences = '0'
  params.dropout1 = 0.1
  params.units1 = '16,16'
  params.act1 = "'linear','relu'"
  params.stateful = 0
  return params


def cnn_stride_params():
  """Flags for toy conv striding model."""
  params = Flags()
  params.model_name = 'cnn'
  params.cnn_filters = '16,16,16,16,16,16,16'
  params.cnn_kernel_size = '(3,3),(3,3),(3,3),(3,3),(3,3),(3,1),(3,1)'
  params.cnn_act = "'relu','relu','relu','relu','relu','relu','relu'"
  params.cnn_dilation_rate = '(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1)'
  params.cnn_strides = '(2,1),(1,1),(2,2),(1,1),(1,1),(1,1),(1,1)'
  params.dropout1 = 0.5
  params.units2 = '16,16'
  params.act2 = "'linear','relu'"
  return params


def cnn_params():
  """Flags for toy conv model."""
  params = Flags()
  params.model_name = 'cnn'
  params.cnn_filters = '16,16,16,16,16,16,16'
  params.cnn_kernel_size = '(3,3),(5,3),(5,3),(5,3),(5,2),(5,1),(10,1)'
  params.cnn_act = "'relu','relu','relu','relu','relu','relu','relu'"
  params.cnn_dilation_rate = '(1,1),(1,1),(2,1),(1,1),(2,1),(1,1),(2,1)'
  params.cnn_strides = '(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1)'
  params.dropout1 = 0.5
  params.units2 = '16,16'
  params.act2 = "'linear','relu'"
  return params


def lstm_params():
  """Flags for toy lstm model."""
  params = Flags()
  params.model_name = 'lstm'
  params.lstm_units = '32'
  params.return_sequences = '0'
  params.use_peepholes = 0
  params.num_proj = '-1'
  params.dropout1 = 0.3
  params.units1 = '32,32'
  params.act1 = "'linear','relu'"
  params.stateful = 0
  return params


def gru_params():
  """Flags for toy gru model."""
  params = Flags()
  params.model_name = 'gru'
  params.gru_units = '32'
  params.return_sequences = '0'
  params.dropout1 = 0.1
  params.units1 = '32,32'
  params.act1 = "'linear','relu'"
  params.stateful = 0
  return params


def ds_cnn_stride_params():
  """Flags for toy "depthwise convolutional neural network" stride model."""
  params = Flags()
  params.model_name = 'ds_cnn'
  params.cnn1_kernel_size = '(10,4)'
  params.cnn1_dilation_rate = '(1,1)'
  params.cnn1_strides = '(2,1)'
  params.cnn1_padding = 'same'
  params.cnn1_filters = 16
  params.cnn1_act = 'relu'
  params.bn_momentum = 0.98
  params.bn_center = 1
  params.bn_scale = 0
  params.bn_renorm = 0
  params.dw2_dilation_rate = '(1,1),(1,1),(1,1),(1,1),(1,1)'
  params.dw2_kernel_size = '(3,3),(3,3),(3,3),(3,3),(3,3)'
  params.dw2_strides = '(2,2),(1,1),(1,1),(1,1),(1,1)'
  params.dw2_padding = 'same'
  params.dw2_act = "'relu','relu','relu','relu','relu'"
  params.cnn2_filters = '16,16,16,16,16'
  params.cnn2_act = "'relu','relu','relu','relu','relu'"
  params.dropout1 = 0.2
  return params


def svdf_params():
  """Flags for toy svdf model."""
  params = Flags()
  params.mel_num_bins = 80
  params.dct_num_features = 30
  params.model_name = 'svdf'
  params.svdf_memory_size = '4,10,10,10,10,10'
  params.svdf_units1 = '16,16,16,16,16,16'
  params.svdf_act = "'relu','relu','relu','relu','relu','relu'"
  params.svdf_units2 = '16,16,16,16,16,-1'
  params.svdf_dropout = '0.0,0.0,0.0,0.0,0.0,0.0'
  params.svdf_pad = 0
  params.dropout1 = 0.0
  params.units2 = ''
  params.act2 = ''
  return params


# these are toy hotword model parameters
# with reduced dims for test latency reduction
HOTWORD_MODEL_PARAMS = {
    'svdf': svdf_params(),
    'ds_cnn_stride': ds_cnn_stride_params(),
    'gru': gru_params(),
    'lstm': lstm_params(),
    'cnn_stride': cnn_stride_params(),
    'cnn': cnn_params(),
    'crnn': crnn_params(),
    'dnn': dnn_params(),
    'att_rnn': att_rnn_params(),
    'att_mh_rnn': att_mh_rnn_params(),
}
