# Sparse GPU Kernels for Deep Learning

![Sparse and Dense MobileNet Throughput v. Accuracy on V100](https://github.com/google-research/google-research/blob/master/sgk/images/sparse_mbv1.png)

This repo accompanies the paper [Sparse GPU Kernels For Deep Learning](https://arxiv.org/abs/2006.10901), published at SC'20. It includes the code and checkpoints for the sparse MobileNetV1 and Transformer models as well as the dataset of sparse matrices from deep neural networks used for benchmarking. The kernels developed in the paper are available in [Sputnik](https://github.com/google-research/sputnik), which this repo depends on.

# Sparse Neural Networks

The checkpoints for all models can downloaded [here](https://storage.googleapis.com/sgk-sc2020/sgk_models.tar.gz). The models along with their accuracies, throughputs and instructions for how to run them are included below.

### Installation

These models rely on custom TensorFlow operations for the kernels provided in Sputnik. We highly recommend you use Docker (w/ Nvidia Docker) to build and run them. After cloning the repository and entering the directory, run 

```
sudo docker build -t sgk
sudo docker run --runtime=nvidia -v /tmp/:/mount/ -it sgk
```

to build the image and launch the container. We're assuming you've downloaded and un-tarred the model checkpoints under `/tmp`, which will be made available under `/mount` inside the container.

### MobileNetV1

All throughputs measured on an Nvidia V100 GPU.

| Width | Sparsity | Top-1 Accuracy | Throughput (FPS) |
|:-----:|:--------:|:--------------:|:----------------:|
|   1   |    0%    |      72.7%     |       2,518      |
|  1.2  |    0%    |      73.8%     |       2,046      |
|  1.4  |    0%    |      74.8%     |       1,729      |
|  1.3  |    90%   |      72.9%     |       2,874      |
|  1.4  |    90%   |      73.3%     |       2,706      |
|  1.5  |    90%   |      73.8%     |       2,537      |
|  1.6  |    90%   |      74.1%     |       2,366      |
|  1.7  |    90%   |      74.4%     |       2,226      |
|  1.8  |    90%   |      74.9%     |       2,095      |

To benchmark a model from inside the container, enter the `mbv1` directory and run `bash benchmark.sh ../../sgk_models/mbv1/<model_dir> <model_width> <sparsity>`. For example, `bash benchmark.sh ../../sgk_models/mbv1/fused-sparse-18-90 1.8 0.9` benchmarks Sparse MobileNetV1 width 1.8. If you have the raw ImageNet dataset installed, you can run `bash imagenet.sh ../../sgk_models/<model_dir> <model_width> <sparsity>` to run inference on the validation set. Note that this script assumes the dataset is installed under `/tmp/data/imagenet/raw/` and is available in the container at `/mount/data/imagenet/raw/`.

### Transformer

All throughputs and memory consumption measured on an Nvidia V100 GPU. See the full paper for results on an Nvidia 1080 GPU.

|        Model       | Bits Per Dimension | Throughput (tokens/s) | Memory Usage (GB) |
|:------------------:|:------------------:|:---------------------:|:-----------------:|
| Sparse Transformer |        3.77        |         67,857        |        0.77       |
|     Transformer    |        3.76        |         32,477        |        9.88       |

To benchmark a model from inside the container, enter the `transformer` directory and run `bash benchmark.sh ../../sgk_models/transformer/<model_dir> <sparse|dense>`. For example, `bash benchmark.sh ../../sgk_models/transformer/sparse sparse` benchmarks Sparse Transformer.

# Deep Learning Matrix Collection

The dataset of sparse matrices from deep neural networks used for benchmarking sparse kernels is available for download [here](https://storage.googleapis.com/sgk-sc2020/dlmc.tar.gz). These matrices were collected from the sparse neural network models released with [The State of Sparsity in Deep Neural Networks](https://arxiv.org/abs/1902.09574). The matrices have been extracted and formatted to make benchmarking easier.

# Citation

```
@inproceedings{sgk_sc2020,
  author    = {Trevor Gale and Matei Zaharia and Cliff Young and Erich Elsen},
  title     = {Sparse {GPU} Kernels for Deep Learning},
  booktitle = {Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, {SC} 2020},
  year      = {2020},
}
```

