# Supplementary code with ICLR 2021 paper "Extreme Memorization via Scale of Initialization"

## Download and extract datasets

-   CIFAR-10: https://www.dropbox.com/s/sstz260o6ad0ryn/cifar10.zip?dl=0
-   CIFAR-100: https://www.dropbox.com/s/czun89ff9dhlc80/cifar100.zip?dl=0
-   SVHN: https://www.dropbox.com/s/kf0p35wmp10j55x/svhn.zip?dl=0

```bash
unzip cifar10.zip
unzip cifar100.zip
unzip svhn.zip
cp -r cifar10 /tmp/
cp -r cifar100 /tmp/
cp -r svhn /tmp/
```

## Install requirements

```
pip install -r requirements.txt
```

## Main Inputs Arguments

*   `--train_input_files`: Input pattern for training tfrecords.
*   `--test_input_files`: Input pattern for test tfrecords.
*   `--dataset`: name of the dataset(options: cifar10 | cifar100 | svhn,
    default: cifar10).
*   `--model_type`: model architecture type (options: mlp | convnet, default:
    mlp)
*   `--activation`: activation function, only applicable for MLP (options: relu
    | sin | sigmoid default: relu)
*   `--loss_function`: loss function used (options: cross_entropy | hinge | l2,
    default: cross_entropy)
*   `--shuffled_labels`: should use shuffled labels ? (default: False)
*   `--custom_init`: whether to initialize w_1 with custom random normal
    initializer. Only applicable for MLP model type. (default: False)
*   `--stddev`: stddev used for random normal initializer. Only applicable when
    model type is MLP and custom_init is set to True. (default: 0.001)
*   `--num_units`: number of hidden units (default: 1024)

## Tensorboard

Each training run saves representation and gradient alignment measures as TF
summary scalars, these plots can be visualized in the Tensorboard tool as
follows.

```
tensorboard --logdir=/tmp/tensorflow/generalization
```

## Section 3 - Extreme memorization

We employ sin activation and initialize w_1 with random normal with varying
stddev - [0.001, 0.01, 0.1, 1.0, 10.0].

### Example command with stddev=0.001 on CIFAR-10 dataset.

```
python -m extreme_memorization.train \
--train_input_files=/tmp/cifar10/image_cifar10_fingerprint-train* \
--test_input_files=/tmp/cifar10/image_cifar10_fingerprint-dev* \
--activation=sin \
--custom_init=true \
--stddev=0.001
```

## Section 4 - Why should the scaling affect homogeneous activations ?

Activation function is switched to ReLU and initialize w_1 with random normal
with varying stddev - [0.001, 0.01, 0.1, 1.0, 10.0].

### Example command with stddev=0.001 on CIFAR-10 dataset

```
python -m extreme_memorization.train \
--train_input_files=/tmp/cifar10/image_cifar10_fingerprint-train* \
--test_input_files=/tmp/cifar10/image_cifar10_fingerprint-dev* \
--activation=relu \
--custom_init=true \
--stddev=0.001
```

### Example command with stddev=0.001 on CIFAR-10 dataset with *hinge loss*

```
python -m extreme_memorization.train \
--train_input_files=/tmp/cifar10/image_cifar10_fingerprint-train* \
--test_input_files=/tmp/cifar10/image_cifar10_fingerprint-dev* \
--activation=relu \
--custom_init=true \
--loss_function=hinge \
--stddev=0.001
```

### Example command with stddev=0.001 on CIFAR-10 dataset with *squared loss*

```
python -m extreme_memorization.train \
--train_input_files=/tmp/cifar10/image_cifar10_fingerprint-train* \
--test_input_files=/tmp/cifar10/image_cifar10_fingerprint-dev* \
--activation=relu \
--custom_init=true \
--loss_function=l2 \
--stddev=0.001
```

## Section 5 - Is alignment relevant more broadly?

Activation function is switched to ReLU and all variables are initialized using
glorot uniform initializer.

### Train 2-layer MLP on CIFAR-10 dataset

```
python -m extreme_memorization.train \
--train_input_files=/tmp/cifar10/image_cifar10_fingerprint-train* \
--test_input_files=/tmp/cifar10/image_cifar10_fingerprint-dev* \
--activation=relu
```

### Train 2-layer MLP on CIFAR-10 dataset with *shuffled labels*

```
python -m extreme_memorization.train \
--train_input_files=/tmp/cifar10/image_cifar10_fingerprint-train* \
--test_input_files=/tmp/cifar10/image_cifar10_fingerprint-dev* \
--activation=relu \
--shuffled_labels=true
```

### Train ConvNet on CIFAR-10 dataset

```
python -m extreme_memorization.train \
--train_input_files=/tmp/cifar10/image_cifar10_fingerprint-train* \
--test_input_files=/tmp/cifar10/image_cifar10_fingerprint-dev* \
--model_type=convnet
```
