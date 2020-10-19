# SupCon - Supervised Contrastive Learning

####Authors: Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, Dilip Krishnan

**Corresponding author:** Aaron Sarna (sarna@google.com)

This repo contains the TensorFlow code used to train the models used in the paper [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362), presented at NeurIPS 2020.

It is implemented in TensorFlow v1 using the TPUEstimator framework, although much of the code, including the loss function is TensorFlow v2 compatible. It is intended to be run on Google Cloud TPU V3. The code will run fine on GPU or CPU as well, but they may not support the batch sizes we trained with, particularly since the batch norm and loss function implementations can only aggregate metrics across multiple chips on TPUs. We have found that adding memory, similar to [MoCo](https://arxiv.org/abs/1911.05722) can compensate for using smaller batch sizes in this setting, but that is not implemented in this repo. There is a PyTorch implementation of our paper available at https://github.com/HobbitLong/SupContrast. Note that the CIFAR10 numbers from the paper come from the PyTorch implementation, and the TensorFlow implementation currently slightly underperforms that. The ImageNet numbers in the paper come from this TensorFlow implementation.

Self-supervised contrastive learning in the style of [SimCLR](https://arxiv.org/abs/2002.05709) is essentially a special case of SupCon where the label for each sample is unique within the global batch. Therefore, this implementation also reproduces SimCLR, simply by setting `--use_labels=False`.

## Running

### Environment setup

First review the [Google Cloud TPU tutorial](https://cloud.google.com/tpu/docs/tutorials/mnist) for basic information on how to use Google Cloud TPUs.

Make sure that all dependencies are installed by running

```sh
pip install -r requirements.txt
```

Finally, once you have setup your virtual machine with Cloud TPUs, if you would like to train with ImageNet, you must [follow the instructions](https://www.tensorflow.org/datasets/catalog/imagenet2012) for downloading the dataset to be compatible with TensorFlow Datasets.

You can then set the following environment variables:

```
TPU_NAME=<tpu-name>
STORAGE_BUCKET=gs://<storage-bucket>
DATA_DIR=$STORAGE_BUCKET/<path-to-tensorflow-dataset>
MODEL_DIR=$STORAGE_BUCKET/<path-to-store-checkpoints>
```

### Training/Evaluation

The scripts directory contains a number of configurations for training SupCon, SimCLR and cross-entropy models on ImageNet and CIFAR10. Each script contains a comment at the top indicating the number of TPU cores we used to train that configuration. An example of how to run one of the scripts is:

```sh
scripts/supcon_imagenet_resnet50.sh --mode=train_then_eval \
  --tpu_name=$TPU_NAME --data_dir=$DATA_DIR --model_dir=$MODEL_DIR
```

For training on GPU or CPU, you should additionally pass `--use_tpu=False`.

The command above specified `--mode=train_then_eval`, which will train the full model and then run a single evaluation pass at the very end in the same job. You can also pass `--mode=train` and `--mode=eval` to separate invocations of the script if you would like to launch a separate continuous evaluation job.

There are many hyperparameters that can be tuned. The scripts provide the values we used for the papers, but if you would like to try others, take a look at hparams_flags.py and you can just pass those flags to the script to override the defaults it sets.

The code writes metrics summaries that can be visualized using [Tensorboard](https://www.tensorflow.org/tensorboard), setting the Tensorboard `--logdir` flag to the MODEL_DIR directory.


