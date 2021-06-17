This directory contains configs used for the paper [Pareto-Optimal Quantized ResNet Is Mostly 4-bit](https://arxiv.org/abs/2105.03536).


Other hyperparameters for the results produced in paper were obtained:
* num_epochs = 250
* lr_scheduler = LRScheduler.COSINE
* step_lr_coeff = 0.2
* step_lr_intervals = 6



#### Leaderboard


`config`  | Description                                                     | Top-1 accuracy
------------------------------------------------ | --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
`resnet50_bfloat16`                                           | No quantization                                |  76.65%
`resnet50_w8_a8_auto`                                   | Weights and activations are quantized to 8 bits                                      | 77.43%
`resnet50_w4_a4_init8_dense8_auto`                          | Weights and activations are quantized to 4 bits; first and last layer quantized to 8 bits | 77.09%

