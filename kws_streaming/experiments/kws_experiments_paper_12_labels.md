# Models reported in paper [Streaming keyword spotting on mobile devices](https://arxiv.org/abs/2005.06720)
======================================================================================


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
```

```shell
# download and set up path to data set V1 and set it up
wget http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz
mkdir data1
mv ./speech_commands_v0.01.tar.gz ./data1
cd ./data1
tar -xf ./speech_commands_v0.01.tar.gz
cd ../

# Select data V1 or data V2 by setting DATA_PATH
# here we select data V2
DATA_PATH=$KWS_PATH/data2

```

## Set pre trained models:

```shell
# download and set up path to models trained and evaluated on data sets V1
wget https://storage.googleapis.com/kws_models/models1.zip
unzip ./models1.zip

# download and set up path to models trained and evaluated on data sets V2
wget https://storage.googleapis.com/kws_models/models2.zip
unzip ./models2.zip

# select pretrained models trained on data V1 or data V2
# by setting models path
# he we select models trained on data V2
MODELS_PATH=$KWS_PATH/models2

# models trained on data V1
#MODELS_PATH=$KWS_PATH/models1
```
Note that DATA_PATH and MODELS_PATH have to point to the same version of data and models accordingly.

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
  models2/
    att_rnn/
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


### ds_cnn_stride

```shell
$CMD_TRAIN \
--data_url '' \
--data_dir $DATA_PATH/ \
--train_dir $MODELS_PATH/ds_cnn_stride/ \
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
ds_cnn \
--cnn1_kernel_size "(10,4)" \
--cnn1_dilation_rate "(1,1)" \
--cnn1_strides "(2,1)" \
--cnn1_padding "same" \
--cnn1_filters 300 \
--cnn1_act 'relu' \
--bn_momentum 0.98 \
--bn_center 1 \
--bn_scale 0 \
--bn_renorm 0 \
--dw2_dilation_rate '(1,1),(1,1),(1,1),(1,1),(1,1)' \
--dw2_kernel_size '(3,3),(3,3),(3,3),(3,3),(3,3)' \
--dw2_strides '(2,2),(1,1),(1,1),(1,1),(1,1)' \
--dw2_padding "same" \
--dw2_act "'relu','relu','relu','relu','relu'" \
--cnn2_filters '300,300,300,300,300' \
--cnn2_act "'relu','relu','relu','relu','relu'" \
--dropout1 0.2
```

non stream latency[us]: 9509 7601
```shell
adb shell rm -f /data/local/tmp/model.tflite
adb push $MODELS_PATH/ds_cnn_stride/tflite_non_stream/non_stream.tflite /data/local/tmp/model.tflite
adb shell taskset f0 /data/local/tmp/benchmark_model --graph=/data/local/tmp/model.tflite --warmup_runs=1 --num_threads=1 --num_runs=1000 > $MODELS_PATH/ds_cnn_stride/tflite_non_stream/non_stream.tflite.benchmark
adb shell taskset f0 /data/local/tmp/benchmark_model --graph=/data/local/tmp/model.tflite --warmup_runs=1 --num_threads=1 --num_runs=1000 --enable_op_profiling=true > $MODELS_PATH/ds_cnn_stride/tflite_non_stream/non_stream.tflite.benchmark.profile
```

### ds_cnn_stream

```shell
$CMD_TRAIN \
--data_url '' \
--data_dir $DATA_PATH/ \
--train_dir $MODELS_PATH/ds_cnn_stream/ \
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
ds_cnn \
--cnn1_kernel_size "(3,3)" \
--cnn1_dilation_rate "(2,1)" \
--cnn1_strides "(1,1)" \
--cnn1_padding "valid" \
--cnn1_filters 300 \
--cnn1_act 'relu' \
--bn_momentum 0.98 \
--bn_center 1 \
--bn_scale 0 \
--bn_renorm 0 \
--dw2_kernel_size '(3,3),(3,3),(10,3),(5,3),(10,3)' \
--dw2_dilation_rate '(1,1),(2,2),(1,1),(2,2),(1,1)' \
--dw2_strides '(1,1),(1,1),(1,1),(1,1),(1,1)' \
--dw2_padding "valid" \
--dw2_act "'relu','relu','relu','relu','relu'" \
--cnn2_filters '300,300,300,300,300' \
--cnn2_act "'relu','relu','relu','relu','relu'" \
--dropout1 0.2
```

non stream latency[us]: 18996 15684
```shell
adb shell rm -f /data/local/tmp/model.tflite
adb push $MODELS_PATH/ds_cnn_stream/tflite_non_stream/non_stream.tflite /data/local/tmp/model.tflite
adb shell taskset f0 /data/local/tmp/benchmark_model_plus_flex --graph=/data/local/tmp/model.tflite --warmup_runs=1 --num_threads=1 --num_runs=1000 > $MODELS_PATH/ds_cnn_stream/tflite_non_stream/non_stream.tflite.benchmark
adb shell taskset f0 /data/local/tmp/benchmark_model_plus_flex --graph=/data/local/tmp/model.tflite --warmup_runs=1 --num_threads=1 --num_runs=1000 --enable_op_profiling=true > $MODELS_PATH/ds_cnn_stream/tflite_non_stream/non_stream.tflite.benchmark.profile
```
stream latency[us]: 1616 1143
```shell
adb shell rm -f /data/local/tmp/model.tflite
adb push $MODELS_PATH/ds_cnn_stream/tflite_stream_state_external/stream_state_external.tflite /data/local/tmp/model.tflite
adb shell taskset f0 /data/local/tmp/benchmark_model_plus_flex --graph=/data/local/tmp/model.tflite --warmup_runs=1 --num_threads=1 --num_runs=1000 > $MODELS_PATH/ds_cnn_stream/tflite_stream_state_external/stream_state_external.tflite.benchmark
adb shell taskset f0 /data/local/tmp/benchmark_model_plus_flex --graph=/data/local/tmp/model.tflite --warmup_runs=1 --num_threads=1 --num_runs=1000 --enable_op_profiling=true > $MODELS_PATH/ds_cnn_stream/tflite_stream_state_external/stream_state_external.tflite.benchmark.profile
```

### svdf

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

non stream latency[us]: 5028 3959
```shell
adb shell rm -f /data/local/tmp/model.tflite
adb push $MODELS_PATH/svdf/tflite_non_stream/non_stream.tflite /data/local/tmp/model.tflite
adb shell taskset f0 /data/local/tmp/benchmark_model_plus_flex --graph=/data/local/tmp/model.tflite --warmup_runs=1 --num_threads=1 --num_runs=1000 > $MODELS_PATH/svdf/tflite_non_stream/non_stream.tflite.benchmark
adb shell taskset f0 /data/local/tmp/benchmark_model_plus_flex --graph=/data/local/tmp/model.tflite --warmup_runs=1 --num_threads=1 --num_runs=1000 --enable_op_profiling=true > $MODELS_PATH/svdf/tflite_non_stream/non_stream.tflite.benchmark.profile
```
stream latency[us]: 590 622
```shell
adb shell rm -f /data/local/tmp/model.tflite
adb push $MODELS_PATH/svdf/tflite_stream_state_external/stream_state_external.tflite /data/local/tmp/model.tflite
adb shell taskset f0 /data/local/tmp/benchmark_model_plus_flex --graph=/data/local/tmp/model.tflite --warmup_runs=1 --num_threads=1 --num_runs=1000 > $MODELS_PATH/svdf/tflite_stream_state_external/stream_state_external.tflite.benchmark
adb shell taskset f0 /data/local/tmp/benchmark_model_plus_flex --graph=/data/local/tmp/model.tflite --warmup_runs=1 --num_threads=1 --num_runs=1000 --enable_op_profiling=true > $MODELS_PATH/svdf/tflite_stream_state_external/stream_state_external.tflite.benchmark.profile
```

### lstm_peep

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

non stream latency[us]: 11934 12870
```shell
adb shell rm -f /data/local/tmp/model.tflite
adb push $MODELS_PATH/lstm_peep/tflite_non_stream/non_stream.tflite /data/local/tmp/model.tflite
adb shell taskset f0 /data/local/tmp/benchmark_model_plus_flex --graph=/data/local/tmp/model.tflite --warmup_runs=1 --num_threads=1 --num_runs=1000 > $MODELS_PATH/lstm_peep/tflite_non_stream/non_stream.tflite.benchmark
adb shell taskset f0 /data/local/tmp/benchmark_model_plus_flex --graph=/data/local/tmp/model.tflite --warmup_runs=1 --num_threads=1 --num_runs=1000 --enable_op_profiling=true > $MODELS_PATH/lstm_peep/tflite_non_stream/non_stream.tflite.benchmark.profile
```
stream latency[us]: 462 474
```shell
adb shell rm -f /data/local/tmp/model.tflite
adb push $MODELS_PATH/lstm_peep/tflite_stream_state_external/stream_state_external.tflite /data/local/tmp/model.tflite
adb shell taskset f0 /data/local/tmp/benchmark_model_plus_flex --graph=/data/local/tmp/model.tflite --warmup_runs=1 --num_threads=1 --num_runs=1000 > $MODELS_PATH/lstm_peep/tflite_stream_state_external/stream_state_external.tflite.benchmark
adb shell taskset f0 /data/local/tmp/benchmark_model_plus_flex --graph=/data/local/tmp/model.tflite --warmup_runs=1 --num_threads=1 --num_runs=1000 --enable_op_profiling=true > $MODELS_PATH/lstm_peep/tflite_stream_state_external/stream_state_external.tflite.benchmark.profile
```

### gru

```shell
$CMD_TRAIN \
--data_url '' \
--data_dir $DATA_PATH/ \
--train_dir $MODELS_PATH/gru/ \
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
gru \
--gru_units 400 \
--return_sequences 0 \
--dropout1 0.1 \
--units1 128,256 \
--act1 "'linear','relu'" \
--stateful 0
```

non stream latency[us]: 11280 11767
```shell
adb shell rm -f /data/local/tmp/model.tflite
adb push $MODELS_PATH/gru/tflite_non_stream/non_stream.tflite /data/local/tmp/model.tflite
adb shell taskset f0 /data/local/tmp/benchmark_model_plus_flex --graph=/data/local/tmp/model.tflite --warmup_runs=1 --num_threads=1 --num_runs=1000 > $MODELS_PATH/gru/tflite_non_stream/non_stream.tflite.benchmark
adb shell taskset f0 /data/local/tmp/benchmark_model_plus_flex --graph=/data/local/tmp/model.tflite --warmup_runs=1 --num_threads=1 --num_runs=1000 --enable_op_profiling=true > $MODELS_PATH/gru/tflite_non_stream/non_stream.tflite.benchmark.profile
```
stream latency[us]: 531 485
```shell
adb shell rm -f /data/local/tmp/model.tflite
adb push $MODELS_PATH/gru/tflite_stream_state_external/stream_state_external.tflite /data/local/tmp/model.tflite
adb shell taskset f0 /data/local/tmp/benchmark_model_plus_flex --graph=/data/local/tmp/model.tflite --warmup_runs=1 --num_threads=1 --num_runs=1000 > $MODELS_PATH/gru/tflite_stream_state_external/stream_state_external.tflite.benchmark
adb shell taskset f0 /data/local/tmp/benchmark_model_plus_flex --graph=/data/local/tmp/model.tflite --warmup_runs=1 --num_threads=1 --num_runs=1000 --enable_op_profiling=true > $MODELS_PATH/gru/tflite_stream_state_external/stream_state_external.tflite.benchmark.profile
```

### gru_state

```shell
$CMD_TRAIN \
--data_url '' \
--data_dir $DATA_PATH/ \
--train_dir $MODELS_PATH/gru_state/ \
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
gru \
--gru_units 400 \
--return_sequences 0 \
--dropout1 0.1 \
--units1 128,256 \
--act1 "'linear','relu'" \
--stateful 1
```

non stream latency[us]: 10935 12161
```shell
adb shell rm -f /data/local/tmp/model.tflite
adb push $MODELS_PATH/gru_state/tflite_non_stream/non_stream.tflite /data/local/tmp/model.tflite
adb shell taskset f0 /data/local/tmp/benchmark_model_plus_flex --graph=/data/local/tmp/model.tflite --warmup_runs=1 --num_threads=1 --num_runs=1000 > $MODELS_PATH/gru_state/tflite_non_stream/non_stream.tflite.benchmark
adb shell taskset f0 /data/local/tmp/benchmark_model_plus_flex --graph=/data/local/tmp/model.tflite --warmup_runs=1 --num_threads=1 --num_runs=1000 --enable_op_profiling=true > $MODELS_PATH/gru_state/tflite_non_stream/non_stream.tflite.benchmark.profile
```

stream latency[us]: 532 481
```shell
adb shell rm -f /data/local/tmp/model.tflite
adb push $MODELS_PATH/gru_state/tflite_stream_state_external/stream_state_external.tflite /data/local/tmp/model.tflite
adb shell taskset f0 /data/local/tmp/benchmark_model_plus_flex --graph=/data/local/tmp/model.tflite --warmup_runs=1 --num_threads=1 --num_runs=1000 > $MODELS_PATH/gru_state/tflite_stream_state_external/stream_state_external.tflite.benchmark
adb shell taskset f0 /data/local/tmp/benchmark_model_plus_flex --graph=/data/local/tmp/model.tflite --warmup_runs=1 --num_threads=1 --num_runs=1000 --enable_op_profiling=true > $MODELS_PATH/gru_state/tflite_stream_state_external/stream_state_external.tflite.benchmark.profile
```

### cnn

```shell
$CMD_TRAIN \
--data_url '' \
--data_dir $DATA_PATH/ \
--train_dir $MODELS_PATH/cnn/ \
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
cnn \
--cnn_filters '64,64,64,64,128,64,128' \
--cnn_kernel_size '(3,3),(5,3),(5,3),(5,3),(5,2),(5,1),(10,1)' \
--cnn_act "'relu','relu','relu','relu','relu','relu','relu'" \
--cnn_dilation_rate '(1,1),(1,1),(2,1),(1,1),(2,1),(1,1),(2,1)' \
--cnn_strides '(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1)' \
--dropout1 0.5 \
--units2 '128,256' \
--act2 "'linear','relu'"
```

non stream latency[us]: 15623 12268
```shell
adb shell rm -f /data/local/tmp/model.tflite
adb push $MODELS_PATH/cnn/tflite_non_stream/non_stream.tflite /data/local/tmp/model.tflite
adb shell taskset f0 /data/local/tmp/benchmark_model --graph=/data/local/tmp/model.tflite --warmup_runs=1 --num_threads=1 --num_runs=1000 > $MODELS_PATH/cnn/tflite_non_stream/non_stream.tflite.benchmark
adb shell taskset f0 /data/local/tmp/benchmark_model --graph=/data/local/tmp/model.tflite --warmup_runs=1 --num_threads=1 --num_runs=1000 --enable_op_profiling=true > $MODELS_PATH/cnn/tflite_non_stream/non_stream.tflite.benchmark.profile
```
stream latency[us]: 1459 996
```shell
adb shell rm -f /data/local/tmp/model.tflite
adb push $MODELS_PATH/cnn/tflite_stream_state_external/stream_state_external.tflite /data/local/tmp/model.tflite
adb shell taskset f0 /data/local/tmp/benchmark_model --graph=/data/local/tmp/model.tflite --warmup_runs=1 --num_threads=1 --num_runs=1000 > $MODELS_PATH/cnn/tflite_stream_state_external/stream_state_external.tflite.benchmark
adb shell taskset f0 /data/local/tmp/benchmark_model --graph=/data/local/tmp/model.tflite --warmup_runs=1 --num_threads=1 --num_runs=1000 --enable_op_profiling=true > $MODELS_PATH/cnn/tflite_stream_state_external/stream_state_external.tflite.benchmark.profile
```

### cnn_stride

```shell
$CMD_TRAIN \
--data_url '' \
--data_dir $DATA_PATH/ \
--train_dir $MODELS_PATH/cnn_stride/ \
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
cnn \
--cnn_filters '64,64,64,128,128,128,128' \
--cnn_kernel_size '(3,3),(3,3),(3,3),(3,3),(3,3),(3,1),(3,1)' \
--cnn_act "'relu','relu','relu','relu','relu','relu','relu'" \
--cnn_dilation_rate '(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1)' \
--cnn_strides '(2,1),(1,1),(2,2),(1,1),(1,1),(1,1),(1,1)' \
--dropout1 0.5 \
--units2 '128,256' \
--act2 "'linear','relu'"
```

non stream latency[us]: 5886 4686
```shell
adb shell rm -f /data/local/tmp/model.tflite
adb push $MODELS_PATH/cnn_stride/tflite_non_stream/non_stream.tflite /data/local/tmp/model.tflite
adb shell taskset f0 /data/local/tmp/benchmark_model --graph=/data/local/tmp/model.tflite --warmup_runs=1 --num_threads=1 --num_runs=1000 > $MODELS_PATH/cnn_stride/tflite_non_stream/non_stream.tflite.benchmark
adb shell taskset f0 /data/local/tmp/benchmark_model --graph=/data/local/tmp/model.tflite --warmup_runs=1 --num_threads=1 --num_runs=1000 --enable_op_profiling=true > $MODELS_PATH/cnn_stride/tflite_non_stream/non_stream.tflite.benchmark.profile
```

### crnn

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


non stream latency[us]: 9441.95 9382
```shell
adb shell rm -f /data/local/tmp/model.tflite
adb push $MODELS_PATH/crnn/tflite_non_stream/non_stream.tflite /data/local/tmp/model.tflite
adb shell taskset f0 /data/local/tmp/benchmark_model_plus_flex --graph=/data/local/tmp/model.tflite --warmup_runs=1 --num_threads=1 --num_runs=1000 > $MODELS_PATH/crnn/tflite_non_stream/non_stream.tflite.benchmark
adb shell taskset f0 /data/local/tmp/benchmark_model_plus_flex --graph=/data/local/tmp/model.tflite --warmup_runs=1 --num_threads=1 --num_runs=1000 --enable_op_profiling=true > $MODELS_PATH/crnn/tflite_non_stream/non_stream.tflite.benchmark.profile
```
stream latency[us]: 442 474
```shell
adb shell rm -f /data/local/tmp/model.tflite
adb push $MODELS_PATH/crnn/tflite_stream_state_external/stream_state_external.tflite /data/local/tmp/model.tflite
adb shell taskset f0 /data/local/tmp/benchmark_model_plus_flex --graph=/data/local/tmp/model.tflite --warmup_runs=1 --num_threads=1 --num_runs=1000 > $MODELS_PATH/crnn/tflite_stream_state_external/stream_state_external.tflite.benchmark
adb shell taskset f0 /data/local/tmp/benchmark_model_plus_flex --graph=/data/local/tmp/model.tflite --warmup_runs=1 --num_threads=1 --num_runs=1000 --enable_op_profiling=true > $MODELS_PATH/crnn/tflite_stream_state_external/stream_state_external.tflite.benchmark.profile
```

### crnn_state

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
dnn \
--units1 '64,128' \
--act1 "'relu','relu'" \
--pool_size 2 \
--strides 2 \
--dropout1 0.1 \
--units2 '128,256' \
--act2 "'linear','relu'"
```

non stream latency[us]: 4100 3237
```shell
adb shell rm -f /data/local/tmp/model.tflite
adb push $MODELS_PATH/dnn/tflite_non_stream/non_stream.tflite /data/local/tmp/model.tflite
adb shell taskset f0 /data/local/tmp/benchmark_model_plus_flex --graph=/data/local/tmp/model.tflite --warmup_runs=1 --num_threads=1 --num_runs=1000 > $MODELS_PATH/dnn/tflite_non_stream/non_stream.tflite.benchmark
adb shell taskset f0 /data/local/tmp/benchmark_model_plus_flex --graph=/data/local/tmp/model.tflite --warmup_runs=1 --num_threads=1 --num_runs=1000 --enable_op_profiling=true > $MODELS_PATH/dnn/tflite_non_stream/non_stream.tflite.benchmark.profile
```

stream latency[us]: 609 608
```shell
adb shell rm -f /data/local/tmp/model.tflite
adb push $MODELS_PATH/dnn/tflite_stream_state_external/stream_state_external.tflite /data/local/tmp/model.tflite
adb shell taskset f0 /data/local/tmp/benchmark_model_plus_flex --graph=/data/local/tmp/model.tflite --warmup_runs=1 --num_threads=1 --num_runs=1000 > $MODELS_PATH/dnn/tflite_stream_state_external/stream_state_external.tflite.benchmark
adb shell taskset f0 /data/local/tmp/benchmark_model_plus_flex --graph=/data/local/tmp/model.tflite --warmup_runs=1 --num_threads=1 --num_runs=1000 --enable_op_profiling=true > $MODELS_PATH/dnn/tflite_stream_state_external/stream_state_external.tflite.benchmark.profile
```

### att_rnn

```shell
$CMD_TRAIN \
--data_url '' \
--data_dir $DATA_PATH/ \
--train_dir $MODELS_PATH/att_rnn/ \
--mel_upper_edge_hertz 7000 \
--how_many_training_steps 20000,20000,20000,20000 \
--learning_rate 0.001,0.0005,0.0001,0.00002 \
--window_size_ms 40.0 \
--window_stride_ms 20.0 \
--mel_num_bins 40 \
--dct_num_features 20 \
--resample 0.15 \
--alsologtostderr \
--train 1 \
--lr_schedule 'exp' \
--use_spec_augment 1 \
--time_masks_number 2 \
--time_mask_max_size 10 \
--frequency_masks_number 2 \
--frequency_mask_max_size 5 \
att_rnn \
--cnn_filters '10,1' \
--cnn_kernel_size '(5,1),(5,1)' \
--cnn_act "'relu','relu'" \
--cnn_dilation_rate '(1,1),(1,1)' \
--cnn_strides '(1,1),(1,1)' \
--rnn_layers 2 \
--rnn_type 'gru' \
--rnn_units 128 \
--dropout1 0.1 \
--units2 '64,32' \
--act2 "'relu','linear'"
```

non stream latency[us]: 9984

### att_mh_rnn

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

non stream latency[us]: 10273 10979
```shell
adb shell rm -f /data/local/tmp/model.tflite
adb push $MODELS_PATH/att_mh_rnn/tflite_non_stream/non_stream.tflite /data/local/tmp/model.tflite
adb shell taskset f0 /data/local/tmp/benchmark_model_plus_flex --graph=/data/local/tmp/model.tflite --warmup_runs=1 --num_threads=1 --num_runs=1000 > $MODELS_PATH/att_mh_rnn/tflite_non_stream/non_stream.tflite.benchmark
adb shell taskset f0 /data/local/tmp/benchmark_model_plus_flex --graph=/data/local/tmp/model.tflite --warmup_runs=1 --num_threads=1 --num_runs=1000 --enable_op_profiling=true > $MODELS_PATH/att_mh_rnn/tflite_non_stream/non_stream.tflite.benchmark.profile
```

### tc_resnet

```shell
$CMD_TRAIN \
--data_url '' \
--data_dir $DATA_PATH/ \
--train_dir $MODELS_PATH/tc_resnet/ \
--mel_upper_edge_hertz 7000 \
--how_many_training_steps 20000,20000,20000,20000 \
--learning_rate 0.1,0.05,0.02,0.01 \
--window_size_ms 40.0 \
--window_stride_ms 20.0 \
--mel_num_bins 40 \
--dct_num_features 40 \
--resample 0.15 \
--alsologtostderr \
--train 0 \
--optimizer 'momentum' \
--lr_schedule 'exp' \
--use_spec_augment 1 \
--time_masks_number 2 \
--time_mask_max_size 10 \
--frequency_masks_number 2 \
--frequency_mask_max_size 5 \
tc_resnet \
--channels '50, 50, 50, 50, 50, 72, 72' \
--debug_2d 0 \
--pool_size '' \
--pool_stride 0 \
--bn_momentum 0.997 \
--bn_center 1 \
--bn_scale 1 \
--bn_renorm 0 \
--dropout 0.2
```

non stream latency[us]: 4257
```shell
adb shell rm -f /data/local/tmp/model.tflite
adb push $MODELS_PATH/tc_resnet/tflite_non_stream/non_stream.tflite /data/local/tmp/model.tflite
adb shell taskset f0 /data/local/tmp/benchmark_model_plus_flex --graph=/data/local/tmp/model.tflite --warmup_runs=1 --num_threads=1 --num_runs=1000 > $MODELS_PATH/tc_resnet/tflite_non_stream/non_stream.tflite.benchmark
adb shell taskset f0 /data/local/tmp/benchmark_model_plus_flex --graph=/data/local/tmp/model.tflite --warmup_runs=1 --num_threads=1 --num_runs=1000 --enable_op_profiling=true > $MODELS_PATH/tc_resnet/tflite_non_stream/non_stream.tflite.benchmark.profile
```

### dnn_raw

```shell
$CMD_TRAIN \
--data_url '' \
--data_dir $DATA_PATH/ \
--train_dir $MODELS_PATH/dnn_raw/ \
--mel_upper_edge_hertz 7000 \
--how_many_training_steps 20000,20000,20000,20000 \
--learning_rate 0.001,0.0005,0.0001,0.00002 \
--window_size_ms 40.0 \
--window_stride_ms 20.0 \
--mel_num_bins 40 \
--dct_num_features 20 \
--resample 0.15 \
--alsologtostderr \
--train 1 \
--lr_schedule 'exp' \
--use_spec_augment 1 \
--time_masks_number 2 \
--time_mask_max_size 10 \
--frequency_masks_number 2 \
--frequency_mask_max_size 5 \
dnn_raw \
--units1 '64,128' \
--act1 "'relu','relu'" \
--pool_size 2 \
--strides 2 \
--dropout1 0.1 \
--units2 '128,256' \
--act2 "'linear','relu'"
```

non stream latency[us]: 621
stream latency[us]: 353


# confirm that unit tests are working:
```shell
python -m kws_streaming.data.input_data_test
python -m kws_streaming.layers.conv2d_test
python -m kws_streaming.layers.data_frame_test
python -m kws_streaming.layers.dct_test
python -m kws_streaming.layers.dense_test
python -m kws_streaming.layers.depthwiseconv1d_test
python -m kws_streaming.layers.flatten_test
python -m kws_streaming.layers.gru_test
python -m kws_streaming.layers.lstm_test
python -m kws_streaming.layers.magnitude_rdft_mel_test
python -m kws_streaming.layers.magnitude_rdft_test
python -m kws_streaming.layers.mel_spectrogram_test
python -m kws_streaming.layers.mel_table_test
python -m kws_streaming.layers.non_scaling_dropout_test
python -m kws_streaming.layers.preemphasis_test
python -m kws_streaming.layers.speech_features_test
python -m kws_streaming.layers.spectrogram_augment_test
python -m kws_streaming.layers.svdf_test
python -m kws_streaming.layers.stream_test
python -m kws_streaming.train.train_test
python -m kws_streaming.train.base_parser_test
python -m kws_streaming.models.utils_test
```
