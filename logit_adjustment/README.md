# Long-tail learning via logit adjustment

This code accompanies the paper:

[Long-tail learning via logit adjustment](https://arxiv.org/abs/2007.07314).
Aditya Krishna Menon, Sadeep Jayasumana, Ankit Singh Rawat, Himanshu Jain, Andreas Veit, Sanjiv Kumar.
ICLR 2021.

## Running the code

#### 1. Setting up the Python environment
The code has been tested with Python 3.7.2. The necessary libraries can be
installed using pip with the following command:

```
# from google-research/
pip install -r logit_adjustment/requirements.txt
```


#### 2. Testing the code
The code may be tested on a dummy dataset using:

```
# from google-research/
python -m logit_adjustment.main --dataset=test --mode=baseline --train_batch_size=2 --test_batch_size=2
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

# To produce baseline (ERM) results:
python -m logit_adjustment.main --dataset=cifar10-lt --mode=baseline

# To produce posthoc logit-adjustment results:
python -m logit_adjustment.main --dataset=cifar10-lt --mode=posthoc

# To produce logit-adjustment loss results:
python -m logit_adjustment.main --dataset=cifar10-lt --mode=loss
```

Replace `cifar10-lt` above with `cifar100-lt` to obtain results for the
CIFAR-100 long-tail dataset. On each invocation, the code will print log
messages to the console. Final test accuracy will also be visible in these
log messages. You can monitor the training progress using Tensorboard:

```
# from google-research/
tensorboard --logdir=./logit_adjustment/log
```

Note: You may want to delete the above Tensorboard log directory before each
new invocation of the training.

## Results

Example (balanced) test accuracies obtained by running this code on a GPU device
is shown below for different datasets.

<center>

|             | Baseline (ERM) | Post-hoc logit adjustment | Logit adjustment loss |
|:------------|:--------------:|:-------------------------:|:-----------------:|
| CIFAR-10 LT |       0.7136   |          0.7732           |       0.7789      |
| CIFAR-100 LT|       0.3987   |          0.4407           |       0.4406      |

</center>

<br/>
Please contact `sadeep {at} google.com` if you encounter any issues with this
code.
