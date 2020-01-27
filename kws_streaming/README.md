# Streaming aware neural network models for keyword spotting(KWS) with tf.keras/TFLite
======================================================================================

Neural Network (NN) model streaming is important for real time response,
high accuracy and good user experience. In this work we explore latency and
accuracy of keyword spotting (KWS) models in the streaming and non streaming modes
on mobile phone.

## Overall design
NN model conversion from non streaming mode (which is frequently
used during training) to streaming can require manual model rewriting.
We address this by designing a Keras based library which allows automatic conversion
of non streaming models to streaming one with no or minimum efforts.
We achieve this in several steps:

1. Train model using non streaming TensorFlow (TF) graph representation.
2. Convert model to streaming and non streaming inference modes.
Conversion to streaming inference mode includes TF/Keras graph traversal and
buffer insertion for the layers which have to be streamed.
3. Convert Keras model to TensorFlow Lite (TFLite)
4. Run TFLite inference on phone.

We build this library with the speech feature extractor being a part of the model
and also part of the model conversion to inference mode with TFLite.
It allows to simplify model testing on mobile devices: the developer can simply pass
audio data into the model and receive classification results.


We build streaming wrapper for layers and streaming aware layers.
It allows us to design Keras models, train them and automatically
convert them to streaming mode.
By streaming we mean streaming inference, where model processing every 20ms of audio
and return classification result.
Non streaming means that model has to receive the whole sequence (1 sec) and then return classification result.

## Inference 
KWS model in streaming mode is executed by steps:

1. Receive sample(packet) of audio data from microphone
2. Feed these data into KWS model
3. Process these data and return detection output
4. Go to next inference iteration to step 1 above.
Most of the layers are streamable by default for example activation layers:
relu, sigmoid; or dense layer. These layers does not have any state.
So we can call them stateless layers.

### State management
Where state is some buffer with data which is going to be reused in the
next iteration of the inference.
Examples of layers with states are LSTM, GRU.

Another example of layers which require state are convolutional and pooling:
To reduce latency of convolutional layer we can avoid recomputation
of the convolution on the same data.
To achieve it, we introduce a state buffer of the input data for a
convolutional layer, so that convolution is recomputed only on updated/new
data sets. We can implement such layer using two approaches:

1. with internal state - conv layer keeps state buffer as internal variable.
It receives input data, updates state buffer inside of the conv layer.
Computes convolution on state buffer and returns convolution output.
2. with external state - conv layer receives input data with state buffer
as input variables. Then it computes convolution on state buffer and
returns convolution output with updated state buffer.
The last one will be fed as input state on the next inference iteration.

A stateful model can be implemented using stateless graph (above example 2.)
because some inference engines do not support updates
of the internal buffer variables.

We implemented stateful KWS models with internal state, such models receive
input speech data and return classification results.
We also implemented stateful KWS models with external state, such models
receive input speech data and all states buffers required for model's layers
and return classification results with updated states buffers.

Models can run in several modes:

1. Non streaming training 'Modes.TRAINING'.
We receive the whole audio sequence and process it

2. Non streaming inference 'Modes.NON_STREAM_INFERENCE'.
We use the same neural net topology
as in training mode but disable regularizers (dropout etc) for inference.
We receive the whole audio sequence and process it.

3. Streaming inference with internal state
'Modes.STREAM_INTERNAL_STATE_INFERENCE'.
We change neural net topology by introducing additional buffers/states
into layers such as conv, lstm, etc.
We receive audio data in streaming mode: packet by packet.
(so we do not have access to the whole sequence).
Inference graph is stateful, so that graph has internal states which are kept
between inference invocations.

4. Streaming inference with external state
'Modes.STREAM_EXTERNAL_STATE_INFERENCE'.
We change neural net topology by introducing additional
input/output buffers/states into layers such as conv, lstm, etc.
We receive audio data in streaming mode: packet by packet.
(so we do not have access to the whole sequence).
Inference graph is stateless, so that graph has not internal state.
All states are received as inputs and after update are returned as output state

### Further information
A paper about this work is work in progress.
All experiments on KWS models presented in this paper can be reproduced by
following the steps described in
`kws_streaming/experiments/kws_experiments.txt`.
Models were trained on a desktop (Ubuntu) and tested on a Pixel4 phone.


Code directory structure:

* `colab` - examples of running KWS models
* `data` - data reading utilities
* `experiments` - command lines for model training and evaluation
* `layers` - streaming aware layers with speech feature extractor and layer tests
* `models` - KWS model definitions
* `train` - model training and evaluation

Below is an example of evaluation and training DNN model:

## Evaluation and training a DNN model.

### Set up data sets:

Download and set up path to data set V1 and set it up

```shell
wget http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz
mkdir data1
mv ./speech_commands_v0.01.tar.gz ./data1
cd ./data1
tar -xf ./speech_commands_v0.01.tar.gz
cd ../
```

Download and set up path to data set V2 and set it up

```shell
wget https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
mkdir data2
mv ./speech_commands_v0.02.tar.gz ./data2
cd ./data2
tar -xf ./speech_commands_v0.02.tar.gz
cd ../
```


### Set up models:

Download and set up path to models trained and evaluated on data sets V1

```shell
wget https://storage.googleapis.com/kws_models/models1.zip
mkdir models1
mv ./models1.zip ./models1
cd ./models1
unzip ./models1.zip
cd ../
```

Download and set up path to models trained and evaluated on data sets V2

```shell
wget https://storage.googleapis.com/kws_models/models2.zip
mkdir models2
mv ./models2.zip ./models2
cd ./models2
unzip ./models2.zip
cd ../
```

### Run only model evaluation:

```shell
python -m kws_streaming.train.model_train_eval \
--data_url '' \
--data_dir ./data1/ \
--train_dir ./models1/dnn/ \
--mel_upper_edge_hertz 7000 \
--how_many_training_steps 10000,10000,10000 \
--learning_rate 0.0005,0.0001,0.00002 \
--window_size_ms 40.0 \
--window_stride_ms 20.0 \
--mel_num_bins 40 \
--dct_num_features 20 \
--resample 0.15 \
--alsologtostderr \
--train 0 \
dnn \
--units1 '64,128' \
--act1 "'relu','relu'" \
--pool_size 2 \
--strides 2 \
--dropout1 0.1 \
--units2 '128,256' \
--act2 "'linear','relu'"
```

### Re-train dnn model from scratch on data set V1 and run evaluation:

```shell
python -m kws_streaming.train.model_train_eval \
--data_url '' \
--data_dir ./data1/ \
--train_dir ./models1/dnn_1/ \
--mel_upper_edge_hertz 7000 \
--how_many_training_steps 100,100,100 \
--learning_rate 0.0005,0.0001,0.00002 \
--window_size_ms 40.0 \
--window_stride_ms 20.0 \
--mel_num_bins 40 \
--dct_num_features 20 \
--resample 0.15 \
--alsologtostderr \
--train 1 \
dnn \
--units1 '64,128' \
--act1 "'relu','relu'" \
--pool_size 2 \
--strides 2 \
--dropout1 0.1 \
--units2 '128,256' \
--act2 "'linear','relu'"
```

Some key flags are described below:

* set `"--train 0"` to run only model evaluation
* set `"--train 1"` to run model training and model evaluation
* set `"--train_dir ./models1/dnn_1/"` to a new folder which does not exist, we will create it automatically
* set `"dnn \"` to train the dnn model
* set `"--data_dir ./data1/"` to use data sets v1


If you interested to train or evaluate models on data sets V2 just set:

* set `--data_dir ./data2/` to use the other data set, and
* set `--train_dir ./models2/dnn/` to avoid overwriting previous results.


### Training on custom data
If you want to train on your own data, you'll need to create .wavs with your
recordings, all at a consistent length, and then arrange them into subfolders
organized by label. For example, here's a possible file structure:

```
data >
  up >
    audio_0.wav
    audio_1.wav
  down >
    audio_2.wav
    audio_3.wav
  other>
    audio_4.wav
    audio_5.wav
```

You'll also need to tell the script what labels to look for, using the
"--wanted_words" argument. In this case, 'up,down' might be what you want, and
the audio in the 'other' folder would be used to train an 'unknown' category.

To pull this all together, you'd run:

```shell
python -m kws_streaming.train.model_train_eval \
--data_dir ./data \
--wanted_words up,down \
dnn
```

Above script will automatically split data into training/validation and testing.

If you prefer to split the data on your own, then you should set flag
"--split_data 0" and prepare folders with structure:

```
data >
  training >
    up >
      audio_0.wav
      audio_1.wav
    down >
      audio_2.wav
      audio_3.wav
  validation >
    up >
      audio_6.wav
      audio_7.wav
    down >
      audio_8.wav
      audio_9.wav
  testing >
    up >
      audio_12.wav
      audio_13.wav
    down >
      audio_14.wav
      audio_15.wav
  _background_noise_ >
    audio_18.wav
```

To pull this all together, you'd run:

```shell
python -m kws_streaming.train.model_train_eval \
--data_dir ./data \
--split_data 0 \
--wanted_words up,down \
dnn
```


<section class="zippy">
Description of the content of the model folder models1/dnn_1:

```
logs > - it has training/validation logs:

     train/events.out.tfevents.**.com - training loss/accuracy at every iteration

     validation/events.out.tfevents.**.com - validation loss/accuracy at every iteration

non_stream > - TF non streamable model stored in SavedModel format

stream_state_internal > TF streaming model stored in SavedModel format

tf > this folder has evaluation of tf.keras model in both streaming and non streaming modes

   model_summary_non_stream.png - non streaming model graph (picture)

   model_summary_non_stream.txt - non streaming model graph (txt)

   model_summary_stream_state_external.png - streaming model graph with external states (picture)

   model_summary_stream_state_external.txt - streaming model graph with external states (txt)

   model_summary_stream_state_internal.png - streaming model graph with internal states (picture)

   model_summary_stream_state_internal.txt - streaming model graph with internal states (txt)

   stream_state_external_model_accuracy_sub_set_reset0.txt - accuracy of streaming model with external state
      on subset of testing data (it is used to validate that TF and TFLite inference gives the same result).
      Do not use these accuracy for reporting because it is computed on subset of testing data (on 1000 samples)
      State of the model is not reseted before running inference.
      So we can see how internal state is influencing accuracy in long run.

   stream_state_external_model_accuracy_sub_set_reset1.txt - accuracy of streaming model with external state
      on subset of testing data (it is used to validate that TF and TFLite inference gives the same result).
      Do not use these accuracy for reporting because it is computed on subset of testing data (on 1000 samples)
      State of the model is reseted before running inference.
      So it is equivalent to non streaming inference (state is not kept between testing sequences).

   tf_non_stream_model_accuracy.txt - accuracy of non streaming model tested with TF

   tf_non_stream_model_sampling_stream_accuracy.txt - accuracy of non streaming model tested with TF
      Input testing data are shifted randomly in range: -time_shift_ms ... time_shift_ms

   tf_stream_state_internal_model_accuracy_sub_set.txt - accuracy of streaming model with internal state
      on subset of testing data.
      Do not use these accuracy for reporting because it is computed on subset of testing data (on 1000 samples)
      State of the model is not reseted before running inference.
      So we can see how internal state is influencing accuracy in long run.

tflite_non_stream > - TF non streaming model is converted to TFLite and stored in this folder

    non_stream.tflite - TFLite non streaming model

    tflite_non_stream_model_accuracy.txt - accuracy of TFLite non streaming model
        We report this accuracy in the paper

    non_stream.tflite.benchmark - benchmark of TFLite non streaming model on mobile phone

    non_stream.tflite.benchmark.profile - profiling of TFLite non streaming model on mobile phone

tflite_stream_state_external > TF streaming model with external state is converted to TFLite and stored in this folder

    stream_state_external.tflite - TFLite streaming model with external state

    tflite_stream_state_external_model_accuracy_reset0.txt - accuracy of TFLite streaming model with external state
       State of the model is not reseted before running inference.
       So we can see how internal state is influencing accuracy in long run.
       We report this accuracy in the paper for streaming models

    tflite_stream_state_external_model_accuracy_reset1.txt - accuracy of TFLite streaming model with external state
       State of the model is reseted before running inference.
       So it is equivalent to non streaming inference (state is not kept between testing sequences).

    stream_state_external.tflite.benchmark - benchmark of TFLite streaming model with external state on mobile phone

    stream_state_external.tflite.benchmark.profile - profiling of TFLite streaming model with external state on mobile phone

accuracy_last.txt - accuracy at the last training iteration (used for debugging)

last_weights.data - weights of the model at last training iteration (used for debugging)

flags.txt - flags which were used for model training (include all model parameters settings, paths all of it)

graph.pbtxt - TF non streaming model in graph representation

labels.txt - list of labels used for model training

best_weights.data - best model weights, these weights will be used for model evaluation with TFLite and reporting
```
</section>
