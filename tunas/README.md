# Source code for TuNAS research paper
This directory contains code to reproduce the key experiments from the research
paper
[Can weight sharing outperform random architecture search? An investigation with TuNAS](https://openaccess.thecvf.com/content_CVPR_2020/papers/Bender_Can_Weight_Sharing_Outperform_Random_Architecture_Search_An_Investigation_With_CVPR_2020_paper.pdf)
(Bender, Liu, et al., published at CVPR 2020).

**Supported experiments:** This codebase supports two types of experiments:

* **Architecture search experiments** using the three main image classification
  search spaces from the TuNAS paper:

  1. The `proxylessnas_search` space, which was called "ProxylessNAS"
  in the original TuNAS paper, contains roughly 10^21 architectures. This
  space was adapted from Cai et al.'s
  [ProxylessNAS](https://openreview.net/pdf?id=HylVB3AqYm) paper
  and built on results from the
  [MobileNetV2](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.pdf)
  and
  [MnasNet](https://openaccess.thecvf.com/content_CVPR_2019/papers/Tan_MnasNet_Platform-Aware_Neural_Architecture_Search_for_Mobile_CVPR_2019_paper.pdf)
  research papers.
  2. The larger `proxylessnas_enlarged_search` space, which was called
  "ProxylessNAS-Enlarged" in the original TuNAS paper, contains around
  10^28 architectures. This space extends `proxylessnas_search` by additionally
  searching over the number of output filters in each block of the network.
  3. The very large `mobilenet_v3_like_search` space, which was called
  "MobileNetV3-Like" in the original TuNAS paper, contains around 10^43
  architectures. It was built around the
  [MobileNetV3-Large](https://arxiv.org/abs/1905.02244)
  network architecture.

* **Stand-alone training experiments** where the best network architectures found
by a search can be trained and evaluated on ImageNet. To assist with
comparisons, we also include reproductions of baseline *reference architectures*
from the published literature, including MnasNet, ProxylessNAS-Mobile, and
MobileNetV3.

**Additional tools:** We also include tools for randomly sampling architectures
in our search spaces (to ensure that our random search baselines are
reproducible), and for analyzing the outputs of architecture search experiments.

**Released datasets:** In addition to the source code for reproducing
our experiments, we also include additional datasets, which are linked
to and described in detail in the "Released datasets" section below.

**Platform requirements:** The TuNAS codebase is written on top of TensorFlow.
It is optimized for -- and currently only runs on -- Cloud TPUs. Our experiments
were conducted using
[TPU v2-32 or TPU v3-32 instances](https://cloud.google.com/tpu/docs/types-zones).
Instructions for running experiments are included below.

## Differences from the TuNAS paper
Our experiments use separate training, validation, and test sets based on
ImageNet. Because the labels for the ImageNet test set were never released
publicly, we follow the standard practice of using the official Imagenet
validation set as our test set. In addition, we carve out a 50,0046-example
subset of the ImageNet training set for use as our validation set.

Our open-sourcing effort required us to use a different train/validation
split than we reported in our paper. As a result, the "validation set"
accuracies (from our held-out portion of the training set) in the open-source
release are systematically 0.3% lower than the ones reported in the original
TuNAS paper. This applies to searched network architectures, as well as to
randomly sampled architectures and our reproductions of published reference
models such as MobileNetV2. Based on experiments we ran prior to the code's
release, test set accuracies should match what we reported in our paper.

## Setting up an environment
Before running experiments running experiments on Cloud TPUs, we recommend
working through a published tutorial for machine learning on Cloud TPUs, such as
   Cloud's tutorial on
   [Training ResNet on Cloud TPUs](https://cloud.google.com/tpu/docs/tutorials/resnet).
   This will show you how to bring up and shut down Cloud TPU instances. It will
   also let you verify that you have a working GCP / Cloud TPU setup, and
   identify and resolve potential issues with your setup.


## Running architecture search experiments

You'll first need to set up a Cloud Storage bucket with a prepared copy of
the ImageNet dataset, as well as a VM and Cloud TPU to run the
architecture search binary. We provide brief instructions here; more detailed
ones are included in the Cloud TPU ResNet instructions linked to above.

1. Manually download and prepare the ImageNet dataset for use by the TensorFlow
   Dataset (TFDS) API, following the
   [documentation here](https://www.tensorflow.org/datasets/catalog/imagenet2012).
   The instructions below will assume that you have a directory containing a
   TFDS-friendly copy of the dataset on Google Cloud Storage (GCS).

2. Open a Cloud Shell window, or install the
   [gcloud tool](https://cloud.google.com/sdk/gcloud)
   on your computer and open a new Bash command line prompt.

3. Create a new Cloud project, or identify one you wish to reuse. In the
   shell, create a shell variable to store the id of your project. For example:

   ```bash {highlight="context:..."}
   export PROJECT_ID=...
   gcloud config set project "${PROJECT_ID}"
   ```

4. Bring up a new VM and TPU instance. For example:

   ```bash
   ctpu up \
     --zone=us-central1-a \
     --disk-size-gb=300 \
     --machine-type=e2-standard-8 \
     --tpu-size=v2-32 \
     --tf-version=2.2 \
     --name=tunas-jf32
   ```

   The zone may need to be changed (here and below) based on resource and data
   availability. For performance reasons, we strongly recommend that bringing
   up a VM and TPU that are physically close to the GCS bucket where your data
   is stored.

5. The `ctpu up` command may automatically SSH into your newly created VM and
   open up a command prompt in the new VM. If not, you'll need to run
   `gcloud compute ssh` manually. For example:

   ```bash
   gcloud compute ssh tunas-jf32 --zone=us-central1-a
   ```

   The name and zone must match the values you passed in when you brought up
   the VM.

6. Copy the TuNAS source code onto the VM. One way to do this is to follow the
   `git clone` or `svn export` instructions provided by GitHub. Another option
   is to download the code onto your local machine and then run
   `gcloud compute scp` in your local machine to copy the source code onto your
   VM. For example using the following command after making sure the highlighted part is set correctly:

   ```bash {highlight="context:~/path/to/tunas"}
   gcloud compute scp --recurse --zone=us-central1-a ~/path/to/tunas tunas-jf32:~/tunas
   ```

7. Run the TuNAS architecture search binary after making sure you set the highlighted directories correctly. For example:

   ```bash {highlight="context:gs://,1"}
   export PYTHONPATH=~/tunas:$PYTHONPATH
   export SSD=proxylessnas_search
   export SEARCH_OUTPUT_DIR=gs://path/to/output/directory
   export DATASET_DIR=gs://path/to/dataset/directory
   python3 -m tunas.mobile_architecture_search \
     --tpu=tunas-jf32 \
     --ssd="${SSD}" \
     --checkpoint_dir="${SEARCH_OUTPUT_DIR}" \
     --dataset_dir="${DATASET_DIR}"
   ```

   In the command above, `SEARCH_OUTPUT_DIR` should be replaced with the name of
   a new directory in a Cloud Storage bucket which you wish to create, and
   `DATASET_DIR` should point to the TFDS ImageNet directory that you prepared
   in Step 1.

   The SSD ("search space definition") controls which search space is used
   for experiments. Three possible choices correspond to the three main search
   spaces which were investigated in the TuNAS paper.
   * `proxylessnas_search` corresponds to ProxylessNAS search space from
     the TuNAS paper. Note that similar to the published TuNAS results but
     unlike the original ProxylessNAS implementation, this setting automatically
     enables features such as aggressive weight sharing, the absolute value
     reward function, and op and filter warmup by default.
   * `SSD=proxylessnas_enlarged_search` corresponds to the ProxylessNAS-Enlarged
     search spaces from the TuNAS paper.
   * `SSD=mobilenet_v3_like_search` corresponds to the MobileNetV3-Like search
     space from the TuNAS paper.

   The architecture search job will take several hours to finish.

8. Once a job is finished, run the `analyze_mobile_search` tool to extract a
   searched architecture from the training logs generated by
   `mobile_architecture_search.py` For example:

   ```bash
   python3 -m tunas.tools.analyze_mobile_search --dirname="${SEARCH_OUTPUT_DIR}"
   ```

  Copy the value of `indices`, which is a colon-separated list of integers,
  onto your clipboard. This is a compact specification of the final network
  architecture selected by the search; you'll need it for the next step.

  You might also want to take note of the value of `cost_model`, which can be
  interpreted as the estimated inference time of the selected network
  architecture on a Pixel 1 phone.

9. Train the network architecture from scratch. To evaluate it on our validation
   set (which is actually a held-out portion of the ImageNet training set), run
   the following command with the highlighted parts updated to the approriate values:

   ```bash {highlight="context:gs://,1 context:..."}
   export INDICES=...
   export FINAL_OUTPUT_DIR=gs://path/to/directory
   python3 -m tunas.mobile_train_eval \
     --tpu=tunas-jf32 \
     --checkpoint_dir="${FINAL_OUTPUT_DIR}" \
     --ssd="${SSD}" \
     --indices="${INDICES}" \
     --dataset_dir="${DATASET_DIR}"
   ```

   In this command, `INDICES` should be set to the colon-separated string that
   you copied in the previous step, while `FINAL_OUTPUT_DIR` should be the name
   of a new directory you wish to create.

  The job will run for a few hours (90 epochs of training), and then print a
  final accuracy measured on a held-out portion of the ImageNet training set.
  Logs containing additional information will be written to the
  `FINAL_OUTPUT_DIR`, and can be visualized using
  [TensorBoard](https://www.tensorflow.org/tensorboard).

  If you wish to instead train on the full ImageNet training set and evaluate
  on the official ImageNet validation set (which is commonly used as a test set
  in the academic literature), you can update the `FINAL_OUTPUT_DIR` and rerun
  the command above with two additional flags: `--epochs=360` (which will
  increase the number of epochs from 90 to 360) and `--use_held_out_test_set`.
  We recommend using the `--use_held_out_test_set` option sparingly, as frequent
  use may increase the risk of overfitting to the test set.


# Released datasets
In addition to the source code, we provide additional datasets: one with
information about our train-validation set split, and another which contains
experimental results from searched and random architectures.

## Train-validation split
Because the labels for the official ImageNet test set were never publicly
released, we followed the common practice of using the official ImageNet
validation set as our test set, and use a held-out portion of the training
set as our validation set.

[This file](http://storage.googleapis.com/gresearch/tunas/imagenet_valid_split_filenames.txt)
contains the ids of all the images in the held-out portion of the ImageNet
training set which we used as a validation set in our experiments.

## Searched and random architectures
We have released data from quality control experiments we ran on the
TuNAS open-source code. Our main goals were to verify that we could
reproduce the published results from the
[TuNAS paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Bender_Can_Weight_Sharing_Outperform_Random_Architecture_Search_An_Investigation_With_CVPR_2020_paper.pdf),
and to make details of these experiments available to external researchers.

The data files are:

* [random\_architectures.csv](http://storage.googleapis.com/gresearch/tunas/random_architectures.csv):
  A file containing the specifications, simulated inference times, and
  accuracies of more than 750 random architectures sampled from our three main
  search spaces.
* [searched\_architectures.csv](http://storage.googleapis.com/gresearch/tunas/searched_architectures.csv):
  A file containing the specifications and simulated inference times of searched
  network architectures found using TuNAS.

Each file contains results for the three main search space definitions (SSDs) we
investigated in our paper:

1. `proxylessnas_search`, which is referred to as the ProxylessNAS search space
   in our paper.
2. `proxylessnas_enlarged_search`, which is referred to the
   ProxylessNAS-Enlarged search space in our paper.
3. `mobilenet_v3_like_search`, which is referred to the MobileNetV3-Like
   search space in our paper.

We provide two CSV files: one for searched architectures found using TuNAS, and
another for our random search baselines, where we  sampled and evaluated random
architectures from each search space. We sampled architectures from the search
space uniformly at random, and used rejection sampling to restrict the results
to architectures whose latencies were 83-85ms for `proxylessnas_search` and
`proxylessnas_enlarged_search`, and 57-59ms for `mobilenet_v3_like_search`.
These ranges were chosen to be comparable with the reference and the searched
models.

Each CSV file contains the following columns:

* `ssd`: The search space definition used for the experiment.
* `indices`: A colon-separated list of integers that uniquely identifies a
  specific architecture in the search space. The `ssd` and `indices` can be
  passed to TuNAS' `mobile_train_eval.py` tool in order to train and evaluate a
  given architecture on ImageNet.
* `simulated_pixel1_time_ms`: The simulated inference time of the network
  architecture on a Pixel 1 phone. Simulated numbers were obtained by fitting
  a linear regression to predicted inference times obtained using the
  [NetAdapt](https://arxiv.org/abs/1804.03230) lookup tables.
* `90epoch_validation_accuracy`: The model's "validation" set accuracy after
  training for 90 epochs on ImageNet. Because the labels for the ImageNet test
  set were never released, we use a held-out subset of 50,046 examples from the
  ImageNet training set as our "validation" set, and use the remaining examples
  as our "training" set for these experiments.
* `360epoch_test_accuracy`: The model's "test" set accuracy after training for
  90 epochs on ImageNet. Because the labels for the real ImageNet test set were
  never released, we following the common practice of using the official
  ImageNet validation set as our "test" set. We train on the full ImageNet
  training set for these experiments.
