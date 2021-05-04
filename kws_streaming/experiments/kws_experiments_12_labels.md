# Models trained on data sets speech commands v2 with 12 labels.
======================================================================================

Below models are trained on [data sets V2 2018](https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz) with 12 labels.


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
pip install tensorflow_model_optimization
# was tested on tf_nightly-2.3.0.dev20200515-cp36-cp36m-manylinux2010_x86_64.whl

# install libs:
pip install pydot
pip install graphviz
pip install numpy
pip install absl-py
```

## Set up data sets:

There are two versions of data sets for training KWS which are well described
in [paper](https://arxiv.org/pdf/1804.03209.pdf). Here we use [data sets V2 2018](https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz)

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
mkdir $KWS_PATH/models_data_v2_12_labels

# models trained on data V2
MODELS_PATH=$KWS_PATH/models_data_v2_12_labels
```

After all of these, main folder KWS_PATH should have several subfolders:
```
  kws_streaming/
    colab/
    data/
    experiments/
    ...
  data2
    _background_noise_/
    bed/
    ...
  models_data_v2_12_labels/
    ...
```

## Models training and evaluation:

There are two options of running python script. One with bazel and another by calling python directly shown below:
```shell
# CMD_TRAIN="bazel run -c opt --copt=-mavx2 kws_streaming/train:model_train_eval --"
CMD_TRAIN="python -m kws_streaming.train.model_train_eval"
```

### ds_tc_resnet - based on [MatchboxNet](https://arxiv.org/pdf/2004.08531.pdf)

parameters: 75K \
float accuracy 98.0 \
```shell
$CMD_TRAIN \
--data_url '' \
--data_dir $DATA_PATH/ \
--train_dir $MODELS_PATH/ds_tc_resnet/ \
--mel_upper_edge_hertz 7600 \
--how_many_training_steps 40000,40000,20000,20000 \
--learning_rate 0.001,0.0005,0.0002,0.0001 \
--window_size_ms 30.0 \
--window_stride_ms 10.0 \
--mel_num_bins 80 \
--dct_num_features 40 \
--resample 0.15 \
--alsologtostderr \
--train 1 \
--use_spec_augment 1 \
--time_masks_number 2 \
--time_mask_max_size 25 \
--frequency_masks_number 2 \
--frequency_mask_max_size 7 \
--pick_deterministically 1 \
ds_tc_resnet \
--activation 'relu' \
--dropout 0.0 \
--ds_filters '128, 64, 64, 64, 128, 128' \
--ds_repeat '1, 1, 1, 1, 1, 1' \
--ds_residual '0, 1, 1, 1, 0, 0' \
--ds_kernel_size '11, 13, 15, 17, 29, 1' \
--ds_stride '1, 1, 1, 1, 1, 1' \
--ds_dilation '1, 1, 1, 1, 2, 1'
```

### att_mh_rnn
parameters: 750K \
float accuracy 98.4 \

```shell
$CMD_TRAIN \
--data_url '' \
--data_dir $DATA_PATH/ \
--train_dir $MODELS_PATH/att_mh_rnn/ \
--mel_upper_edge_hertz 7600 \
--how_many_training_steps 40000,40000,20000,20000 \
--learning_rate 0.001,0.0005,0.0002,0.0001 \
--window_size_ms 30.0 \
--window_stride_ms 10.0 \
--mel_num_bins 80 \
--dct_num_features 40 \
--resample 0.15 \
--alsologtostderr \
--train 1 \
--use_spec_augment 1 \
--time_masks_number 2 \
--time_mask_max_size 25 \
--frequency_masks_number 2 \
--frequency_mask_max_size 7 \
--pick_deterministically 1 \
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
--units2 '64' \
--act2 "'relu'"
```
