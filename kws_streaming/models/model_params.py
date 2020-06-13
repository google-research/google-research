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

"""Models parameters (with toy values, for testing)."""


class Params(object):
  """Default parameters for data and feature settings.

     These parameters are compatible with command line flags
  """

  def __init__(self):
    # default parameters
    self.train_dir = ''
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
    self.use_spec_augment = 0
    self.time_masks_number = 2
    self.time_mask_max_size = 10
    self.frequency_masks_number = 2
    self.frequency_mask_max_size = 5
    self.optimizer = 'adam'
    self.lr_schedule = 'linear'
    self.background_volume = 0.1
    self.l2_weight_decay = 0.0
    self.background_frequency = 0.8
    self.silence_percentage = 10.0
    self.unknown_percentage = 10.0
    self.time_shift_ms = 100.0
    self.testing_percentage = 10
    self.validation_percentage = 10
    self.how_many_training_steps = '10000,10000,10000'
    self.eval_step_interval = 400
    self.learning_rate = '0.0005,0.0001,0.00002'
    self.batch_size = 100
    self.optimizer_epsilon = 1e-08
    self.resample = 0.15
    self.volume_resample = 0.0


def att_mh_rnn_params():
  """Parameters for toy multihead attention model."""
  params = Params()
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
  """Parameters for toy attention model."""
  params = Params()
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
  """Parameters for toy dnn model."""
  params = Params()
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
  """Parameters for toy conv rnn model."""
  params = Params()
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
  """Parameters for toy conv striding model."""
  params = Params()
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
  """Parameters for toy conv model."""
  params = Params()
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


def tc_resnet_params():
  """Parameters for toy tc_resnet model."""
  params = Params()
  params.model_name = 'tc_resnet'
  params.channels = '24, 36'
  params.debug_2d = 0
  params.pool_size = ''
  params.kernel_size = '(3, 1)'
  params.pool_stride = 0
  params.bn_momentum = 0.997
  params.bn_center = 1
  params.bn_scale = 1
  params.bn_renorm = 0
  params.dropout = 0.2
  return params


def lstm_params():
  """Parameters for toy lstm model."""
  params = Params()
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
  """Parameters for toy gru model."""
  params = Params()
  params.model_name = 'gru'
  params.gru_units = '32'
  params.return_sequences = '0'
  params.dropout1 = 0.1
  params.units1 = '32,32'
  params.act1 = "'linear','relu'"
  params.stateful = 0
  return params


def ds_cnn_params():
  """Parameters for toy "depthwise convolutional neural network" stride model."""
  params = Params()
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
  """Parameters for toy svdf model."""
  params = Params()
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


def mobilenet_params():
  """Parameters for mobilenet model."""
  params = Params()
  params.model_name = 'mobilenet'
  params.cnn1_filters = 32
  params.cnn1_kernel_size = '(3,1)'
  params.cnn1_strides = '(2,2)'
  params.ds_kernel_size = '(3,1),(3,1),(3,1),(3,1)'
  params.ds_strides = '(2,2),(2,2),(1,1),(1,1)'
  params.cnn_filters = '16,32,32,32'
  params.dropout = 0.2
  params.bn_scale = 0
  return params


def mobilenet_v2_params():
  """Parameters for mobilenet v2 model."""
  params = Params()
  params.model_name = 'mobilenet_v2'
  params.cnn1_filters = 32
  params.cnn1_kernel_size = '(3,1)'
  params.cnn1_strides = '(2,2)'
  params.ds_kernel_size = '(3,1),(3,1),(3,1),(3,1)'
  params.cnn_strides = '(1,1),(2,2),(1,1),(1,1)'
  params.cnn_filters = '32,32,32,32'
  params.cnn_expansions = '1.5,1.5,1.5,1.5'
  params.dropout = 0.2
  params.bn_scale = 0
  return params


def xception_params():
  """Parameters for xception model."""
  params = Params()
  params.model_name = 'xception'
  params.cnn1_kernel_sizes = '5'
  params.cnn1_filters = '32'
  params.stride1 = 2
  params.stride2 = 2
  params.stride3 = 1
  params.stride4 = 1
  params.cnn2_kernel_sizes = '5'
  params.cnn2_filters = '32'
  params.cnn3_kernel_sizes = '5'
  params.cnn3_filters = '32'
  params.cnn4_kernel_sizes = '5'
  params.cnn4_filters = '32'
  params.units2 = '64'
  params.dropout = 0.2
  params.bn_scale = 0
  return params


def inception_params():
  """Parameters for inception model."""
  params = Params()
  params.model_name = 'inception'
  params.cnn1_filters = '24'
  params.cnn1_kernel_sizes = '5'
  params.cnn1_strides = '1'
  params.cnn2_filters1 = '10,10,16'
  params.cnn2_filters2 = '10,10,16'
  params.cnn2_kernel_sizes = '5,5,5'
  params.cnn2_strides = '2,2,1'
  params.dropout = 0.2
  params.bn_scale = 0
  return params


def inception_resnet_params():
  """Parameters for inception resnet model."""
  params = Params()
  params.model_name = 'inception_resnet'
  params.cnn1_filters = '32'
  params.cnn1_strides = '2'
  params.cnn1_kernel_sizes = '5'
  params.cnn2_scales = '0.2,0.5,1.0'
  params.cnn2_filters_branch0 = '32,32,32'
  params.cnn2_filters_branch1 = '32,32,32'
  params.cnn2_filters_branch2 = '32,32,64'
  params.cnn2_strides = '2,1,1'
  params.cnn2_kernel_sizes = '3,5,5'
  params.dropout = 0.2
  params.bn_scale = 0
  return params


def svdf_resnet_params():
  """Parameters for svdf with resnet model."""
  params = Params()
  params.model_name = 'svdf_resnet'
  params.block1_memory_size = '4,10'
  params.block2_memory_size = '10,10'
  params.block3_memory_size = '10,10'
  params.block1_units1 = '16,32'
  params.block2_units1 = '32,32'
  params.block3_units1 = '64,128'
  params.blocks_pool = '1,2,2'
  params.use_batch_norm = 1
  params.bn_scale = 0
  params.activation = 'relu'
  params.svdf_dropout = 0.0
  params.svdf_pad = 1
  params.svdf_use_bias = 0
  params.dropout1 = 0.2
  params.units2 = '64'
  params.flatten = 0
  return params


# these are toy hotword model parameters
# with reduced dims for unit test only
HOTWORD_MODEL_PARAMS = {
    'svdf': svdf_params(),
    'svdf_resnet': svdf_resnet_params(),
    'ds_cnn': ds_cnn_params(),
    'gru': gru_params(),
    'lstm': lstm_params(),
    'cnn_stride': cnn_stride_params(),
    'cnn': cnn_params(),
    'tc_resnet': tc_resnet_params(),
    'crnn': crnn_params(),
    'dnn': dnn_params(),
    'att_rnn': att_rnn_params(),
    'att_mh_rnn': att_mh_rnn_params(),
    'mobilenet': mobilenet_params(),
    'mobilenet_v2': mobilenet_v2_params(),
    'xception': xception_params(),
    'inception': inception_params(),
    'inception_resnet': inception_resnet_params(),
}
