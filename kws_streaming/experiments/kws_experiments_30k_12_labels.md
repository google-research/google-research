# Models with 30k parameters.
======================================================================================

Below models are configured to have around 30K parameters. There was no hyper parameters optimization, so there are opportunities for improvements.
Non-quantized model size will be around 100..150KB
Post training quantization is applied - quantized model size will be around 30..50KB.
All below models are trained with \
--feature_type 'mfcc_op' (speech mfcc feature extractor is using internal TFLite op ) and \
--preprocess 'raw' (so that  model is built end to end: speech feature extractor is part of the model)


## Set up python kws_streaming.

Set main folder.
```shell
# create main folder
mkdir test

# set path to a main folder
KWS_PATH=$PWD/test

cd $KWS_PATH
```

```shell
# copy content of kws_streaming to a folder /tmp/test/kws_streaming
git clone https://github.com/google-research/google-research.git
mv google-research/kws_streaming .
```

## Install tensorflow with deps.
```shell
# set up virtual env
pip install virtualenv
virtualenv --system-site-packages -p python3 ./venv3
source ./venv3/bin/activate

# install TensorFlow, correct TensorFlow version is important
pip install --upgrade pip
pip install tf_nightly==2.4.0-dev20200917
pip install tensorflow_addons
# was tested on tf_nightly-2.3.0.dev20200515-cp36-cp36m-manylinux2010_x86_64.whl

# install libs:
pip install pydot
pip install graphviz
pip install numpy
pip install absl-py
```

## Set up data sets:

There are two versions of data sets for training KWS which are well described
in [paper](https://arxiv.org/pdf/1804.03209.pdf)
[data sets V1 2017](http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz)
[data sets V2 2018](https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz)

```shell
# download and set up path to data set V2 and set it up
wget https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
mkdir data2
mv ./speech_commands_v0.02.tar.gz ./data2
cd ./data2
tar -xf ./speech_commands_v0.02.tar.gz
cd ../

# path to data sets V2
DATA_PATH=$KWS_PATH/data2
```

## Set path to models:

```shell
# set up path for model training
mkdir $KWS_PATH/models2_30k

# models trained on data V2
MODELS_PATH=$KWS_PATH/models2_30k
```

After all of these, main folder KWS_PATH should have several subfolders:
<pre><code>
  kws_streaming/
    colab/
    data/
    experiments/
    ...
  data2
    _background_noise_/
    bed/
    ...
  models2_30k/
    inception/
    ...
</code></pre>

## Models training and evaluation:

If your model is already trained then you can specify "--train 0" which will evaluate the model and produce an accuracy report with TFLite modules. If you would like to re-train model from scratch then you should: set "--train 1" and remove model subfolder inside of $MODELS_PATH

There are two options of running python script. One with bazel and another by calling python directly shown below:
```shell
# CMD_TRAIN="bazel run -c opt --copt=-mavx2 kws_streaming/train:model_train_eval --"
CMD_TRAIN="python -m kws_streaming.train.model_train_eval"
```

### svdf_resnet

parameters: 33K \
float accuracy 94.6; model size: 145KB; latency: 1.2ms \
quant accuracy 94.6; model size: 57KB; latency: 1.2ms \
```shell
$CMD_TRAIN \
--data_url '' \
--data_dir $DATA_PATH/ \
--train_dir $MODELS_PATH/svdf_resnet/ \
--mel_upper_edge_hertz 7600 \
--how_many_training_steps 20000,20000,20000,20000 \
--learning_rate 0.001,0.0005,0.0001,0.00002 \
--window_size_ms 40.0 \
--window_stride_ms 20.0 \
--mel_num_bins 80 \
--dct_num_features 40 \
--resample 0.15 \
--time_shift_ms 100 \
--feature_type 'mfcc_op' \
--fft_magnitude_squared 1 \
--preprocess 'raw' \
--train 1 \
--lr_schedule 'exp' \
svdf_resnet \
--block1_memory_size '7' \
--block2_memory_size '7' \
--block3_memory_size '11,11' \
--block1_units1 '32' \
--block2_units1 '50' \
--block3_units1 '50,128' \
--blocks_pool '2,2,1' \
--use_batch_norm 1 \
--bn_scale 1 \
--activation 'relu' \
--svdf_dropout 0.0 \
--svdf_pad 1 \
--svdf_use_bias 0 \
--dropout1 0.0 \
--units2 '64' \
--flatten 0
```

### mobilenet
parameters: 35K \
float accuracy: 92.5; model size: 141KB; latency 1.1ms \
quant accuracy: 91.5; model size: 45KB; latency 1.1ms
```shell
$CMD_TRAIN \
--data_url '' \
--data_dir $DATA_PATH/ \
--train_dir $MODELS_PATH/mobilenet/ \
--mel_upper_edge_hertz 7600 \
--how_many_training_steps 20000,20000,20000,20000 \
--learning_rate 0.001,0.0005,0.0001,0.00002 \
--window_size_ms 40.0 \
--window_stride_ms 20.0 \
--mel_num_bins 80 \
--dct_num_features 40 \
--resample 0.15 \
--time_shift_ms 100 \
--feature_type 'mfcc_op' \
--fft_magnitude_squared 1 \
--preprocess 'raw' \
--train 1 \
--lr_schedule 'exp' \
mobilenet \
--cnn1_filters 32 \
--cnn1_kernel_size '(3,1)' \
--cnn1_strides '(2,2)' \
--ds_kernel_size '(3,1),(3,1),(3,1),(3,1)' \
--ds_strides '(2,2),(2,2),(1,1),(1,1)' \
--cnn_filters '32,64,128,128' \
--dropout 0.0
```

### mobilenet_v2

parameters: 28K \
float accuracy: 95.0; model size: 118KB; latency 1.1ms \
quant accuracy: 94.1; model size: 43KB; latency 1.1ms
```shell
$CMD_TRAIN \
--data_url '' \
--data_dir $DATA_PATH/ \
--train_dir $MODELS_PATH/mobilenet_v2/ \
--mel_upper_edge_hertz 7600 \
--how_many_training_steps 20000,20000,20000,20000 \
--learning_rate 0.001,0.0005,0.0001,0.00002 \
--window_size_ms 40.0 \
--window_stride_ms 20.0 \
--mel_num_bins 80 \
--dct_num_features 30 \
--resample 0.15 \
--time_shift_ms 100 \
--train 1 \
--feature_type 'mfcc_op' \
--fft_magnitude_squared 1 \
--preprocess 'raw' \
mobilenet_v2 \
--cnn1_filters 32 \
--cnn1_kernel_size '(3,1)' \
--cnn1_strides '(2,2)' \
--ds_kernel_size '(3,1),(3,1),(3,1),(3,1)' \
--cnn_strides '(1,1),(2,2),(1,1),(1,1)' \
--cnn_filters '32,32,64,64' \
--cnn_expansions '1.5,1.5,1.5,1.5' \
--dropout 0.0
```

### inception
parameters: 30K \
float accuracy: 95.9; model size: 130KB; latency 1.2ms \
quant accuracy: 94.9; model size: 63KB; latency 1.2ms

```shell
$CMD_TRAIN \
--data_url '' \
--data_dir $DATA_PATH/ \
--train_dir $MODELS_PATH/inception/ \
--mel_upper_edge_hertz 7600 \
--how_many_training_steps 20000,20000,20000,20000 \
--learning_rate 0.001,0.0005,0.0001,0.00002 \
--window_size_ms 40.0 \
--window_stride_ms 20.0 \
--mel_num_bins 80 \
--dct_num_features 30 \
--resample 0.15 \
--time_shift_ms 100 \
--train 1 \
--feature_type 'mfcc_op' \
--fft_magnitude_squared 1 \
--preprocess 'raw' \
inception \
--cnn1_filters '32' \
--cnn1_kernel_sizes '5' \
--cnn1_strides '1' \
--cnn2_filters1 '16,16,16' \
--cnn2_filters2 '32,64,70' \
--cnn2_kernel_sizes '3,5,5'
--cnn_strides '2,2,1' \
--dropout 0.0 \
--bn_scale 0
```

### inception_resnet
parameters: 33K \
float accuracy: 95.8; model size: 140KB; latency 1.2ms \
quant accuracy: 94.8; model size: 60KB; latency 1.2ms
```shell
$CMD_TRAIN \
--data_url '' \
--data_dir $DATA_PATH/ \
--train_dir $MODELS_PATH/inception_resnet/ \
--mel_upper_edge_hertz 7600 \
--how_many_training_steps 20000,20000,20000,20000 \
--learning_rate 0.001,0.0005,0.0001,0.00002 \
--window_size_ms 40.0 \
--window_stride_ms 20.0 \
--mel_num_bins 80 \
--dct_num_features 30 \
--resample 0.15 \
--time_shift_ms 100 \
--train 1 \
--feature_type 'mfcc_op' \
--fft_magnitude_squared 1 \
--preprocess 'raw' \
inception_resnet \
--cnn1_filters '32' \
--cnn1_strides '1' \
--cnn1_kernel_sizes '5' \
--cnn2_scales '0.2,0.5,1.0' \
--cnn2_filters_branch0 '16,16,32' \
--cnn2_filters_branch1 '16,16,32' \
--cnn2_filters_branch2 '32,32,64' \
--cnn2_strides '2,2,1' \
--cnn2_kernel_sizes '3,5,5' \
--bn_scale 1 \
--dropout 0.0
```

### svdf
parameters: 32K \
float accuracy: 91.1; model size: 142KB; latency 1.2; latency stream 0.2ms \
quant accuracy: 91.1; model size: 58KB; latency 1.2; latency stream 0.2ms \
```shell
$CMD_TRAIN \
--data_url '' \
--data_dir $DATA_PATH/ \
--train_dir $MODELS_PATH/svdf/ \
--mel_upper_edge_hertz 7600 \
--how_many_training_steps 20000,20000,20000,20000 \
--learning_rate 0.001,0.0005,0.0001,0.00002 \
--window_size_ms 40.0 \
--window_stride_ms 20.0 \
--mel_num_bins 80 \
--dct_num_features 40 \
--resample 0.15 \
--time_shift_ms 100 \
--feature_type 'mfcc_op' \
--fft_magnitude_squared 1 \
--preprocess 'raw' \
--train 1 \
--lr_schedule 'exp' \
svdf \
--svdf_memory_size 4,10,10,10,10,10 \
--svdf_units1 16,32,32,32,64,128 \
--svdf_act "'relu','relu','relu','relu','relu','relu'" \
--svdf_units2 40,40,64,64,64,-1 \
--svdf_dropout 0.0,0.0,0.0,0.0,0.0,0.0 \
--svdf_pad 0 \
--dropout1 0.0 \
--units2 '' \
--act2 ''
```

### tc_resnet
parameters: 31K \
float accuracy: 95.7; model size: 127KB; latency 1.1ms \
quant accuracy: 94.6; model size: 41KB; latency 1.1ms

```shell
$CMD_TRAIN \
--data_url '' \
--data_dir $DATA_PATH/ \
--train_dir $MODELS_PATH/tc_resnet/ \
--mel_upper_edge_hertz 7600 \
--how_many_training_steps 20000,20000,20000,20000 \
--learning_rate 0.001,0.0005,0.0001,0.00002 \
--window_size_ms 40.0 \
--window_stride_ms 20.0 \
--mel_num_bins 80 \
--dct_num_features 30 \
--resample 0.15 \
--time_shift_ms 100 \
--train 1 \
--feature_type 'mfcc_op' \
--fft_magnitude_squared 1 \
--preprocess 'raw' \
tc_resnet \
--kernel_size '(3,1)' \
--channels '32, 36, 36, 40' \
--debug_2d 0 \
--pool_size '' \
--pool_stride 0 \
--bn_momentum 0.997 \
--bn_center 1 \
--bn_scale 1 \
--bn_renorm 0 \
--dropout 0.0
```

### xception
parameters: 31K \
float accuracy: 94.8; model size: 133KB; latency 1.1ms \
quant accuracy: 94.0; model size: 51KB; latency 1.1ms
```shell
$CMD_TRAIN \
--data_url '' \
--data_dir $DATA_PATH/ \
--train_dir $MODELS_PATH/xception/ \
--mel_upper_edge_hertz 7600 \
--how_many_training_steps 20000,20000,20000,20000 \
--learning_rate 0.001,0.0005,0.0001,0.00002 \
--window_size_ms 40.0 \
--window_stride_ms 20.0 \
--mel_num_bins 80 \
--dct_num_features 30 \
--resample 0.15 \
--time_shift_ms 100 \
--train 1 \
--feature_type 'mfcc_op' \
--fft_magnitude_squared 1 \
--preprocess 'raw' \
xception \
--cnn1_kernel_size '5' \
--cnn1_filters '16' \
--stride1 2 \
--stride2 2 \
--stride3 1 \
--stride4 1 \
--cnn2_kernel_sizes '5,7' \
--cnn2_filters '32,40' \
--cnn3_kernel_sizes '7' \
--cnn3_filters '64' \
--cnn4_kernel_sizes '11' \
--cnn4_filters '100' \
--units2 '64' \
--bn_scale 0 \
--dropout 0.0
```
