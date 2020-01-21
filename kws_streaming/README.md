# Streaming aware neural network models for keyword spotting(KWS) with tf.keras

Directory structure: \
colab - examples of running KWS models \
data - data reading utilities \
experiments - command lines for model training \
layers - streaming aware layers with speech feature extractor and layer tests \
models - KWS model definitions \
train - model training and evaluation \
\
\
We built streaming wrapper layers and streaming aware layers. \
It allows us to design Keras models, train them and automatically \
convert to streaming mode. \
By streaming we mean streaming inference. \
KWS model in streaming mode is executed by steps: \
1 Receive sample(packet) of audio data from microphone \
2 Feed these data into KWS model \
3 Process these data and return detection output \
4 Go to next inference iteration to step 1 above. \
Most of the layers are streamable by default for example activation layers: \
relu, sigmoid; or dense layer. These layers does not have any state. \
So we can call them stateless layers. \
Where state is some buffer with data which is going to be reused in the \
next iteration of the inference. \
Examples of layers with states are LSTM, GRU. \
\
Another example of layers which require state are convolutional and pooling: \
To reduce latency of convolutional layer we can avoid recomputation \
of the convolution on the same data. \
To achieve it, we introduce a state buffer of the input data for a \
convolutional layer, so that convolution is recomputed only on updated/new \
data sets. We can implement such layer using two approaches: \
a) with internal state - conv layer keeps state buffer as internal variable. \
It receives input data, updates state buffer inside of the conv layer. \
Computes convolution on state buffer and returns convolution output. \
b) with external state - conv layer receives input data with state buffer \
as input variables. Then it computes convolution on state buffer and \
returns convolution output with updated state buffer. \
The last one will be fed as input state on the next inference iteration. \
\
Stateful model can be implemented using stateless graph (above example b ) \
because some inference engines do not support updates \
of the internal buffer variables. \
\
We implemented stateful KWS models with internal state, such models receive
input speech data and return classification results. \
We also implemented stateful KWS models with external state, such models \
receive input speech data and all states buffers required for model's layers \
and return classification results with updated states buffers. \
\
There are several modes: \
1 Non streaming training 'Modes.TRAINING'. \
We receive the whole audio sequence and process it \

2 Non streaming inference 'Modes.NON_STREAM_INFERENCE'. \
We use the same neural net topology \
as in training mode but disable regularizers (dropout etc) for inference. \
We receive the whole audio sequence and process it. \

3 Streaming inference with internal state
'Modes.STREAM_INTERNAL_STATE_INFERENCE'. \
We change neural net topology by introducing additional buffers/states \
into layers such as conv, lstm, etc. \
We receive audio data in streaming mode: packet by packet. \
(so we do not have access to the whole sequence). \
Inference graph is stateful, so that graph has internal states which are kept \
between inference invocations. \

4 Streaming inference with external state
'Modes.STREAM_EXTERNAL_STATE_INFERENCE'. \
We change neural net topology by introducing additional \
input/output buffers/states into layers such as conv, lstm, etc. \
We receive audio data in streaming mode: packet by packet. \
(so we do not have access to the whole sequence). \
Inference graph is stateless, so that graph has not internal state. \
All states are received as inputs and after update are returned as output state\

Paper about this work is at: TBD.
All experiments on KWS models presented in this paper can be reproduced by \
kws_streaming/experiments/kws_experiments.txt \
Models were trained on Ubuntu and tested on Pixel4  \
