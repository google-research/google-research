# Implementation of ALOE: Learning Discrete Energy-based Models via Auxiliary-variable Local Exploration


This code implements ALOE. The following demonstration shows how to perform training and inference.

# Install

Navigate to the root of project, and perform:

    pip install -r requirements.txt
    pip install -e .


# Data and model dumps

All the data and pretrained model dumps can be downloaded via the following link:

    https://console.cloud.google.com/storage/browser/research_public_share/aloe_neurips2020

Please place the downloaded folder into `gcloud` folder, and organize the folders as:
```
aloe/
|___aloe/  # source code
|   |___common # common implementations
|   |___...
|
|___setup.py 
|
|___gcloud/
|   |___data/  # data
|   |___results/  # model dumps
|
|...
```

# Synthetic

First nagivate to the synthetic experiment folder, then run the training script

    cd aloe/synthetic
    ./run_varlen.sh

You can also modify the above script with different dataset names, by changing `data=` field. 

After training, you can draw samples from the sampler, as well as visualize the learned heat map with the following command (suppose you have trained for 500 epochs)

    ./run_varlen.sh -phase plot -epoch_load 500


# Fuzzing

The first step is to obtain raw seed inputs from [OSS-Fuzz](https://github.com/google/oss-fuzz). Please follow the instructions there to setup the docker and obtain the corresponding LibFuzzer binary and seed inputs. 

For simplicity, we show how to run our code with `libpng` target. 

## data preparation

Please create a folder with the target name under `aloe/fuzz/data/fuzz-seeds`. We have included seed inputs of `libpng` for convenience. 

Next we run the following script to cook the raw data into binary format that can be used by ALOE:

    cd aloe/fuzz
    ./run_binary_data.sh

It will create a cooked data folder under `aloe/fuzz/data/fuzz-cooked`


## training

Use the following script for training:

    cd aloe/fuzz
    ./run_varlen.sh

You can also modify the field `data=` to train for other softwares.


## perform fuzzing

Use the following script for generating inputs:

    cd aloe/fuzz
    ./run_fuzz.sh

You can also modify the script to specify the number of files to be generated.


## evaluation

You need to go through the docker image built by OSS-Fuzz to evaluate the coverage of generated inputs. Please follow the guideline here: `https://github.com/google/oss-fuzz`


# Program synthesis


## data preparation

The synthetic training/validation/test datasets are provided. Please see the content under `gcloud/data` folder.

## evaluating existing model dumps

We have provided the model dumps of the learned sampler. Please run 

    ./evaluate_editor.sh

You can also modify the `eval_method` arg to use different beam sizes, or `eval_topk` arg to evaluate different top-k accuracies. 

## pretraining q0

If you want to train the sampler from scratch, please first run the supervised learning to pretrain the initial sampler q0: 

    cd aloe/rfill
    ./run_supervised.sh

You can stop the script when the performance gets plateau.

## learning with local search

Move the pretrained models (encoder-xxx.ckpt, decoder-xxx.ckpt) into the result folder of editor. Rename the epoch index to -1, i.e., `encoder--1.ckpt`, `decoder--1.ckpt`. 

Then train the editor in the following way:

    ./run_editor.sh

It will load the pretrained q0 and train the editor part.
