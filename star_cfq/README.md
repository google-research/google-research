# *-CFQ (Star-CFQ)

This repository contains code for training and evaluating ML
architectures on the *-CFQ dataset.

The dataset can be downloaded from the following URL:

[Download the *-CFQ dataset](https://storage.cloud.google.com/star_cfq_dataset/)

The dataset and details about its construction and use are described in this
AAAI 2021 paper: [*-CFQ: Analyzing the Scalability of Machine Learning on a
Compositional Task](http://arxiv.org/abs/2012.08266).

# Reproducing experiments

To reproduce experiments from our paper we adopted approach suggested by our
colleagues in [Compositional Generalization in Semantic
Parsing](https://github.com/google-research/google-research/tree/master/cfq_pt_vs_sa).

## Prerequisites

We would use a Google Cloud image compatible with CFQ dependencies. Please note
that the compute/zone is not important here, as the code is using TFDS for
loading the dataset, and can be chosen for the user's convenience. The
following command starts a VM with this image and a V100 GPU on Google Cloud:

```shell
gcloud config set project ${YOUR_PROJECT}
gcloud config set compute/zone europe-west4-a

VM_NAME=run-star-cfq
gcloud compute instances create $VM_NAME \
  --image-project=deeplearning-platform-release \
  --image-family=tf-1-15-cu100 \
  --machine-type=n1-standard-8 \
  --boot-disk-size=120GB \
  --maintenance-policy=TERMINATE \
  --accelerator="type=nvidia-tesla-v100,count=1" \
  --metadata="install-nvidia-driver=True"
```

Next, on that machine, get code with:

```shell
sudo apt-get install subversion -y
svn export https://github.com/google-research/google-research/trunk/star_cfq
```

Then you have to set up CFQ, which is also used to run *-CFQ code. This should
be done only once:

```shell
cd star_cfq
bash ./setup_cfq.sh
```

## Running the experiment

All the experiments should be run from `~/star_cfq` directory.

Experiments described in the paper can be run by using `run_experiment.sh`
script. It takes two parameters: a name of the experiment (required), and a
number of training steps (if omitted, default value of 500k would be used).

This script will train and test the Transformer model on all splits in the
specified experiment. For every split it will download the split, preprocess it,
train a model and finally evaluate it and report it's accuracy:

An example run (with only one split) is:

```shell
bash ./run_experiment.sh test 1100
```



