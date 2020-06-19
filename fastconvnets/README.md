# Fast Sparse ConvNets

This directory contains links to the TF-Lite models used in the paper "Fast Sparse ConvNets".

The MobileNetV1 models have a block size of 4 in the last inverted residual block, otherwise they are unstructured. The MobileNetV2 models have a block size of 2 from inverted residual block 11 onwards, otherwise they are unstructured. The exception is the width 1.8, 80% sparse model which unstructured throughout. The first full convolution and the final fully connected layer are both dense in all models. EfficientNet models are fully unstructured and the final fully connected layer is sparse.

| Model | Top-1 Accuracy | Sparsity | Latency (ms) SD 835 | Download |
|-------|:----------------:|:----------:|:---------------------:|:----------:|
| MobileNetV1 .75  | 64.4% | 90% | 21 | [link](https://storage.googleapis.com/fast-convnets/tflite-models/mbv1_075_90_12b4_644.tflite)
| MobileNetV1 1.0  | 68.4% | 90% | 31 | [link](https://storage.googleapis.com/fast-convnets/tflite-models/mbv1_100_90_12b4_684.tflite)
| MobileNetV1 1.4  | 72.0% | 90% | 58 | [link](https://storage.googleapis.com/fast-convnets/tflite-models/mbv1_140_90_12b4_720.tflite)
| MobileNetV2 .8   | 65.2% | 85% | 26 |[link](https://storage.googleapis.com/fast-convnets/tflite-models/mbv2_080_85_11-16b2_652.tflite)
| Cache Aware MobileNetV2 1.0 | 69.7% | 85% | 33 | [link](https://storage.googleapis.com/fast-convnets/tflite-models/mbv2ca_100_85_none_697.tflite)
| MobileNetV2 1.15 | 70.2% | 85% | 40 | [link](https://storage.googleapis.com/fast-convnets/tflite-models/mbv2_115_85_11-16b2_702.tflite)
| MobileNetV2 1.4  | 72.0% | 85% | 54 | [link](https://storage.googleapis.com/fast-convnets/tflite-models/mbv2_140_85_11-16b2_720.tflite)
| MobileNetV2 1.8  | 74.9% | 80% | 102 | [link](https://storage.googleapis.com/fast-convnets/tflite-models/mbv2_180_80_none_749.tflite)
| MobileNetV2 2.0  | 74.5% | 85% | 93 | [link](https://storage.googleapis.com/fast-convnets/tflite-models/mbv2_200_85_11-16b2_744.tflite)
| EfficientNet B0  | 75.1% | 80% | 80 | [link](https://storage.googleapis.com/fast-convnets/tflite-models/enb0_80_none_751.tflite)
| EfficientNet B1  | 76.7% | 85% | 110 | [link](https://storage.googleapis.com/fast-convnets/tflite-models/enb1_85_none_767.tflite)
