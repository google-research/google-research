# Training Over-parameterized Models with Non-decomposable Objectives

This codebase accompanies the paper:

[Training Over-parameterized Models with Non-decomposable Objectives](https://arxiv.org/pdf/2107.04641).
Harikrishna Narasimhan, Aditya Krishna Menon.
NeurIPS 2021.

This repository was branched from [code](https://github.com/google-research/google-research/tree/master/logit_adjustment) for the paper [Menon et al., Long-tail learning via logit adjustment, ICLR 2021](https://arxiv.org/abs/2007.07314).

We provide an implementation of the method proposed in our paper for maximizing
worst-case recall (Algorithm 1) which uses the proposed logit-adjusted
cost-sensitive loss. Three baselines are also included: (i) standard
ERM, (ii) balanced (logit-adjusted) loss of Menon et al. (2021), and (iii) re-weighted
cost-sensitive loss.


## Running the code

#### 1. Setting up the Python environment
The code has been tested with Python 3.7.2. The necessary libraries can be
installed using pip with the following command:

```
# from google-research/
pip install -r non_decomp/requirements.txt
```


#### 2. Testing the code
The code may be tested on a dummy dataset using:

```
# from google-research/
python -m non_decomp.main --dataset=test --mode=baseline --train_batch_size=2 --test_batch_size=2
```

This should complete quickly without any errors.


#### 3. Downloading CIFAR-10/100 long-tail datasets
Download the CIFAR-10/100 long-tail datasets using the links provided below and
put the `.tfrecord` files in the `logit_adjustment/data/` directory. The train
datasets were created with the EXP-100 profile as detailed in the paper, the
test datasets are the same as the standard CIFAR-10/100 test datasets.

Links: [cifar10-lt_train.tfrecord](http://storage.googleapis.com/gresearch/logit_adjustment/cifar10-lt_train.tfrecord),
[cifar10_test.tfrecord](http://storage.googleapis.com/gresearch/logit_adjustment/cifar10_test.tfrecord),
[cifar100-lt_train.tfrecord](http://storage.googleapis.com/gresearch/logit_adjustment/cifar100-lt_train.tfrecord),
[cifar100_test.tfrecord](http://storage.googleapis.com/gresearch/logit_adjustment/cifar100_test.tfrecord).


#### 4. Running the code on CIFAR-10/100 long-tail datasets

You can now run the code on the CIFAR-10 long-tail dataset using the commands
below:

```
# from google-research/

# To produce results for the ERM baseline:
python -m non_decomp.main --dataset=cifar10-lt --mode=erm

# To produce results for the balanced baseline:
python -m non_decomp.main --dataset=cifar10-lt --mode=balanced

# To produce results for the re-weighted cost-sensitive baseline:
python -m non_decomp.main --dataset=cifar10-lt --mode=reweighted

# To produce results for the proposed logit-adjusted cost-sensitive approach:
python -m non_decomp.main --dataset=cifar10-lt --mode=proposed
```

Replace `cifar10-lt` above with `cifar100-lt` to obtain results for the
CIFAR-100 long-tail dataset. On each invocation, the code will print log
messages to the console. Final test accuracy will also be visible in these
log messages. You can monitor the training progress using Tensorboard:

```
# from google-research/
tensorboard --logdir=./non_decomp/log
```

Note: You may want to delete the above Tensorboard log directory before each
new invocation of the training.

<br/>
Please contact `hnarasimhan {at} google.com` if you encounter any issues with this
code.
