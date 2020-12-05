# Models from [paper](https://arxiv.org/abs/2005.06720) with quantization
======================================================================================

To enable post training model quantization, we uses \
--feature_type 'mfcc_op' which is numerically different with 'mfcc_tf' (the last one was used in [paper](https://arxiv.org/abs/2005.06720)). We did not run hyperparameters optimization with 'mfcc_op' feature extractor, so there can be some accuracy reduction. mfcc_op calls audio_spectrogram() and mfcc(). The last one expects squared fft magnitude, so we set fft_magnitude_squared 1.

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


## Set pre trained models:

```shell
# download and set up path to models trained and evaluated on data sets V2
wget https://storage.googleapis.com/kws_models/models2_q.zip
unzip ./models2_q.zip

# models trained on data V2
MODELS_PATH=$KWS_PATH/models2_q
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
  models2_q/
    svdf/
    ...
</code></pre>

## Compile TFLite benchmarking tools
Set up TFLite based neural network benchmarking on phone.
To build benchmarking tools for Android you will need [bazel](https://docs.bazel.build/versions/master/bazel-overview.html)

```shell
# build benchmarking binary
bazel build -c opt --config=android_arm64 --cxxopt='--std=c++17' \
third_party/tensorflow/lite/tools/benchmark:benchmark_model

# check that phone is connected
adb devices

# copy benchmarking binary to phone
adb push bazel-bin/third_party/tensorflow/lite/tools/benchmark/benchmark_model /data/local/tmp

# allow executing benchmarking file as a program
adb shell chmod +x /data/local/tmp/benchmark_model

# build benchmarking binary - for a case if Flex is used by neural network model
bazel build -c opt --config=android_arm64 --cxxopt='--std=c++17' \
third_party/tensorflow/lite/tools/benchmark:benchmark_model_plus_flex

# check that phone is connected
adb devices

# copy benchmarking binary to phone
adb push bazel-bin/third_party/tensorflow/lite/tools/benchmark/benchmark_model_plus_flex /data/local/tmp

# allow executing benchmarking file as a program
adb shell chmod +x /data/local/tmp/benchmark_model_plus_flex
```

## Models training and evaluation:

Now we can run below commands with "--train 0" which will evaluate the model and produce accuracy report with TFLite modules. If you would like to re-train model from scratch then you should: set "--train 0" and remove model subfolder inside of $MODELS_PATH

There are two options of running python script. One with bazel and another by calling python directly shown below:
```shell
# CMD_TRAIN="bazel run -c opt --copt=-mavx2 kws_streaming/train:model_train_eval --"
CMD_TRAIN="python -m kws_streaming.train.model_train_eval"
```

### svdf

parameters: 354K \
float accuracy: 96.0 model size: 1003KB; latency 2ms \
quant accuracy: 96.0 model size: 369KB; latency 1.6ms \
stream float: 96.0 model size: 1003KB;  latency 0.4ms \
stream quant: 96.0 model size: 406KB;  latency 0.4ms

```shell
$CMD_TRAIN \
--data_url '' \
--data_dir $DATA_PATH/ \
--train_dir $MODELS_PATH/svdf/ \
--mel_upper_edge_hertz 7000 \
--how_many_training_steps 20000,20000,20000,20000 \
--learning_rate 0.001,0.0005,0.0001,0.00002 \
--window_size_ms 40.0 \
--window_stride_ms 20.0 \
--mel_num_bins 80 \
--dct_num_features 30 \
--resample 0.15 \
--alsologtostderr \
--time_shift_ms 100 \
--train 0 \
--feature_type 'mfcc_op' \
--fft_magnitude_squared 1 \
svdf \
--svdf_memory_size 4,10,10,10,10,10 \
--svdf_units1 256,256,256,256,256,256 \
--svdf_act "'relu','relu','relu','relu','relu','relu'" \
--svdf_units2 128,128,128,128,128,-1 \
--svdf_dropout 0.0,0.0,0.0,0.0,0.0,0.0 \
--svdf_pad 0 \
--dropout1 0.0 \
--units2 '' \
--act2 ''
```

### lstm_peep

parameters: 545K \
float accuracy: 97.3 model size: 2200KB; latency 10ms \
quant accuracy: 97.3 model size: 723KB; latency 3.8ms

```shell
$CMD_TRAIN \
--data_url '' \
--data_dir $DATA_PATH/ \
--train_dir $MODELS_PATH/lstm_peep/ \
--mel_upper_edge_hertz 7000 \
--how_many_training_steps 20000,20000,20000,20000 \
--learning_rate 0.001,0.0005,0.0001,0.00002 \
--window_size_ms 40.0 \
--window_stride_ms 20.0 \
--mel_num_bins 40 \
--dct_num_features 20 \
--resample 0.15 \
--alsologtostderr \
--train 0 \
--lr_schedule 'exp' \
--use_spec_augment 1 \
--time_masks_number 2 \
--time_mask_max_size 10 \
--frequency_masks_number 2 \
--frequency_mask_max_size 5 \
--feature_type 'mfcc_op' \
--fft_magnitude_squared 1 \
lstm \
--lstm_units 500 \
--return_sequences 0 \
--use_peepholes 1 \
--num_proj 200 \
--dropout1 0.3 \
--units1 '' \
--act1 '' \
--stateful 0
```

### crnn

parameters: 467K \
float accuracy: 97.4 model size: 1800KB; latency 7ms \
quant accuracy: 97.0 model size: 593KB; latency 2.6ms

```shell
$CMD_TRAIN \
--data_url '' \
--data_dir $DATA_PATH/ \
--train_dir $MODELS_PATH/crnn/ \
--mel_upper_edge_hertz 7000 \
--how_many_training_steps 20000,20000,20000,20000 \
--learning_rate 0.001,0.0005,0.0001,0.00002 \
--window_size_ms 40.0 \
--window_stride_ms 20.0 \
--mel_num_bins 40 \
--dct_num_features 20 \
--resample 0.15 \
--alsologtostderr \
--train 0 \
--lr_schedule 'exp' \
--use_spec_augment 1 \
--time_masks_number 2 \
--time_mask_max_size 10 \
--frequency_masks_number 2 \
--frequency_mask_max_size 5 \
--feature_type 'mfcc_op' \
--fft_magnitude_squared 1 \
crnn \
--cnn_filters '16,16' \
--cnn_kernel_size '(3,3),(5,3)' \
--cnn_act "'relu','relu'" \
--cnn_dilation_rate '(1,1),(1,1)' \
--cnn_strides '(1,1),(1,1)' \
--gru_units 256 \
--return_sequences 0 \
--dropout1 0.1 \
--units1 '128,256' \
--act1 "'linear','relu'" \
--stateful 0
```

### crnn_state

parameters: 467K \
float accuracy: 97.1; model size: 1800KB; latency 7.1ms \
quant accuracy: 96.9; model size: 593KB;  latency 2.6ms \
stream float accuracy: 96.3; model size: 1700KB;  latency 0.2ms \
stream quant accuracy: 95.8; model size: 472KB;  latency 0.1ms

```shell
$CMD_TRAIN \
--data_url '' \
--data_dir $DATA_PATH/ \
--train_dir $MODELS_PATH/crnn_state/ \
--mel_upper_edge_hertz 7000 \
--how_many_training_steps 20000,20000,20000,20000 \
--learning_rate 0.001,0.0005,0.0001,0.00002 \
--window_size_ms 40.0 \
--window_stride_ms 20.0 \
--mel_num_bins 40 \
--dct_num_features 20 \
--resample 0.15 \
--alsologtostderr \
--train 0 \
--lr_schedule 'exp' \
--use_spec_augment 1 \
--time_masks_number 2 \
--time_mask_max_size 10 \
--frequency_masks_number 2 \
--frequency_mask_max_size 5 \
--feature_type 'mfcc_op' \
--fft_magnitude_squared 1 \
crnn \
--cnn_filters '16,16' \
--cnn_kernel_size '(3,3),(5,3)' \
--cnn_act "'relu','relu'" \
--cnn_dilation_rate '(1,1),(1,1)' \
--cnn_strides '(1,1),(1,1)' \
--gru_units 256 \
--return_sequences 0 \
--dropout1 0.1 \
--units1 '128,256' \
--act1 "'linear','relu'" \
--stateful 1
```

### dnn

parameters: 447K \
float accuracy: 90.4; model size: 1700KB; latency 1.2ms \
quant accuracy: 90.2; model size: 443KB; latency 1.1ms

```shell
$CMD_TRAIN \
--data_url '' \
--data_dir $DATA_PATH/ \
--train_dir $MODELS_PATH/dnn/ \
--mel_upper_edge_hertz 7000 \
--how_many_training_steps 20000,20000,20000,20000 \
--learning_rate 0.001,0.0005,0.0001,0.00002 \
--window_size_ms 40.0 \
--window_stride_ms 20.0 \
--mel_num_bins 40 \
--dct_num_features 20 \
--resample 0.15 \
--alsologtostderr \
--train 0 \
--lr_schedule 'exp' \
--use_spec_augment 1 \
--time_masks_number 2 \
--time_mask_max_size 10 \
--frequency_masks_number 2 \
--frequency_mask_max_size 5 \
--feature_type 'mfcc_op' \
--fft_magnitude_squared 1 \
dnn \
--units1 '64,128' \
--act1 "'relu','relu'" \
--pool_size 2 \
--strides 2 \
--dropout1 0.1 \
--units2 '128,256' \
--act2 "'linear','relu'"
```


### att_mh_rnn

parameters: 700K \
float accuracy: 97.9 model size: 3400KB; latency 8ms \
quant accuracy: 97.8 model size: 1300KB; latency 4ms

```shell
$CMD_TRAIN \
--data_url '' \
--data_dir $DATA_PATH/ \
--train_dir $MODELS_PATH/att_mh_rnn/ \
--mel_upper_edge_hertz 8000 \
--how_many_training_steps 20000,20000,20000,20000 \
--learning_rate 0.001,0.0005,0.0001,0.00002 \
--window_size_ms 40.0 \
--window_stride_ms 20.0 \
--mel_num_bins 40 \
--dct_num_features 20 \
--resample 0.15 \
--alsologtostderr \
--train 0 \
--lr_schedule 'exp' \
--use_spec_augment 1 \
--time_masks_number 2 \
--time_mask_max_size 10 \
--frequency_masks_number 2 \
--frequency_mask_max_size 5 \
--feature_type 'mfcc_op' \
--fft_magnitude_squared 1 \
att_mh_rnn \
--cnn_filters '10,1' \
--cnn_kernel_size '(5,1),(5,1)' \
--cnn_act "'relu','relu'" \
--cnn_dilation_rate '(1,1),(1,1)' \
--cnn_strides '(1,1),(1,1)' \
--rnn_layers 2 \
--rnn_type 'gru' \
--rnn_units 128 \
--heads 4 \
--dropout1 0.2 \
--units2 '64,32' \
--act2 "'relu','linear'"
```
