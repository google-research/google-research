# Quantization Techniques for Convolutional Neural Networks

This directory contains implementation of various techniques for quantizing
convolutional neural network models into low bit-precision.

Work in progress.

The quantization codes are implemented on top of [`tf_cnn_benchmarks`](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks),
which requires the latest nightly version of TensorFlow.

To run ResNet50 4-bit fixed-point quantized training with synthetic data
with a single GPU (from the google_research directory), run

```
python -m cnn_quantization.tf_cnn_benchmarks.tf_cnn_benchmarks --num_gpus=1 \
--batch_size=32 --model=resnet50_v2 --variable_update=parameter_server \
--quant_weight=true --quant_weight_bits=4 --use_relu_x=true --quant_act=true \
--quant_act_bits=4
```

To run on CPU, run

```
python -m cnn_quantization.tf_cnn_benchmarks.tf_cnn_benchmarks \
--data_format=NHWC --batch_size=32 --model=resnet50_v2 \
--variable_update=parameter_server --quant_weight=true --quant_weight_bits=4 \
--use_relu_x=true --quant_act=true --quant_act_bits=4
```

To see the full list of flags, run

```
python -m cnn_quantization.tf_cnn_benchmarks.tf_cnn_benchmarks --help
```

