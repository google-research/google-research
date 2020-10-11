# Models with 75k parameters.
======================================================================================

Below models are configured to have around 75K parameters and trained on [data sets V2 2018](https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz) with 35 labels.


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
pip install frozendict
pip install numpy
pip install absl-py
```

## Set up data sets:

We use custom set up of [data sets V2 2018](https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz) for KWS model training.
We divided data into training validation and testing data using all 35 categories and stored them in [link](https://storage.googleapis.com/kws_models/data_all_v2.zip)


```shell
# download and set up path to custom split of data set V2
wget https://storage.googleapis.com/kws_models/data_all_v2.zip
unzip ./data_all_v2.zip

# path to data sets V2
DATA_PATH=$KWS_PATH/data_all_v2
```

## Set path to models:

```shell
# set up path for model training
mkdir $KWS_PATH/models2_75k

# models trained on data V2
MODELS_PATH=$KWS_PATH/models2_75k
```

After all of these, main folder KWS_PATH should have several subfolders:
<pre><code>
  kws_streaming/
    colab/
    data/
    experiments/
    ...
  data_all_v2
    _background_noise_/
    bed/
    ...
  models2_75k/
    ...
</code></pre>

## Models training and evaluation:


There are two options of running python script. One with bazel and another by calling python directly shown below:
```shell
# CMD_TRAIN="bazel run -c opt --copt=-mavx2 kws_streaming/train:model_train_eval --"
CMD_TRAIN="python -m kws_streaming.train.model_train_eval"
```


### [MatchboxNet](https://arxiv.org/pdf/2004.08531.pdf) config with momentum

By default 'ds_padding' set 'same' \
For training streamable model 'ds_padding' has to be set 'causal' \
parameters: 75K \
accuracy 96.7
```shell
$CMD_TRAIN \
--batch_size 128 \
--split_data 0 \
--wanted_words 'visual,wow,learn,backward,dog,two,left,happy,nine,go,up,bed,stop,one,zero,tree,seven,on,four,bird,right,eight,no,six,forward,house,marvin,sheila,five,off,three,down,cat,follow,yes' \
--data_url '' \
--data_dir $DATA_PATH/ \
--train_dir $MODELS_PATH/ds_tc_resnet_momentum/ \
--mel_upper_edge_hertz 7000 \
--how_many_training_steps 20000,20000,20000,20000 \
--learning_rate 0.1,0.05,0.02,0.01 \
--window_size_ms 30.0 \
--window_stride_ms 10.0 \
--mel_num_bins 40 \
--dct_num_features 40 \
--resample 0.15 \
--alsologtostderr \
--train 1 \
--optimizer 'momentum' \
--lr_schedule 'linear' \
--use_spec_augment 1 \
--time_masks_number 2 \
--time_mask_max_size 25 \
--frequency_masks_number 2 \
--frequency_mask_max_size 7 \
ds_tc_resnet \
--ds_padding "'same', 'same', 'same', 'same', 'same', 'same'" \
--activation 'relu' \
--dropout 0.0 \
--ds_filters '128, 64, 64, 64, 128, 128' \
--ds_repeat '1, 1, 1, 1, 1, 1' \
--ds_residual '0, 1, 1, 1, 0, 0' \
--ds_kernel_size '11, 13, 15, 17, 29, 1' \
--ds_stride '1, 1, 1, 1, 1, 1' \
--ds_dilation '1, 1, 1, 1, 2, 1'
```

### [MatchboxNet](https://arxiv.org/pdf/2004.08531.pdf) config with novograd
parameters: 75K \
accuracy 96.8
```shell
$CMD_TRAIN \
--batch_size 128 \
--split_data 0 \
--novograd_beta_1 0.9 \
--novograd_beta_2 0.99 \
--novograd_weight_decay 0.001 \
--wanted_words 'visual,wow,learn,backward,dog,two,left,happy,nine,go,up,bed,stop,one,zero,tree,seven,on,four,bird,right,eight,no,six,forward,house,marvin,sheila,five,off,three,down,cat,follow,yes' \
--data_url '' \
--data_dir $DATA_PATH/ \
--train_dir $MODELS_PATH/ds_tc_resnet_novograd/ \
--mel_upper_edge_hertz 7000 \
--how_many_training_steps 20000,20000,20000,20000,20000,60000 \
--learning_rate 0.02,0.02,0.01,0.005,0.002,0.001 \
--window_size_ms 30.0 \
--window_stride_ms 10.0 \
--mel_num_bins 40 \
--dct_num_features 40 \
--resample 0.15 \
--alsologtostderr \
--train 1 \
--optimizer 'novograd' \
--lr_schedule 'exp' \
--use_spec_augment 1 \
--time_masks_number 2 \
--time_mask_max_size 25 \
--frequency_masks_number 2 \
--frequency_mask_max_size 7 \
ds_tc_resnet \
--ds_padding "'same', 'same', 'same', 'same', 'same', 'same'" \
--activation 'relu' \
--dropout 0.0 \
--ds_filters '128, 64, 64, 64, 128, 128' \
--ds_repeat '1, 1, 1, 1, 1, 1' \
--ds_residual '0, 1, 1, 1, 0, 0' \
--ds_kernel_size '11, 13, 15, 17, 29, 1' \
--ds_stride '1, 1, 1, 1, 1, 1' \
--ds_dilation '1, 1, 1, 1, 2, 1'
```
