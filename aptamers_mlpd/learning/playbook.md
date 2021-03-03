# Playbook for Aptamer learning code

This document covers the standard commands used to run the Aptamer TensorFlow 
code, both locally and online.  Note, that because of internal dependencies the
shared code is not runnable outside of Google.

This document assumes you have already collected sequence data as a FASTQ file,
created an experiment proto with experiment metadata and run the processing
pipeline to turn the FASTQ files into TF example protos. See the processing
playbook for details.

Before following the playbook here, check that your dataset exists in
learning/config.py. If your dataset is new, you will want to add the path 
to the data in INPUT_DATA_DIRS and add your sequencing counts to affinity 
mapping to DEFAULT_AFFINITY_TARGET_MAPS.
(The code can be run using a directory as input but it's much easier with a
defined dataset.)

[TOC]

## Run training/eval locally (for test/debug purposes)

To run training/eval, use the `train_feedforward` binary, e.g.,

    run learning/train_feedforward \
    --save_base=xxx \
    --epochs=2 \
    --run_name=$RUN_NAME \
    --num_fc_layers=1 \
    --dataset=aptitude

You can change `--dataset` to another valid dataset, e.g., `'xxx'` or `'xxx'`,
or set `--input_dir` directly.

You must set `RUN_NAME` above to some unique value for each time you run the
script. This is used to generate the path in which to save the model checkpoint
and results. E.g.,

     export RUN_NAME=xxx.debug0



### Guidance on setting number and size of epochs

By default an epoch is defined as a full iteration through the entire dataset.
After each epoch, an evaluation is run and results are reported. For larger
datasets, you may want to use the 'epoch_size' flag to limit the size of each
epoch. A good rule of thumb is to have the reporting run every hour or so of
training to get frequent enough reporting.

The model is set to run for exactly the number of epochs provided -- it will not
stop early if the model has stopped learning and currently the learning rate
does not anneal. This means that it is important to look at frequent reports of
the train and validation evaluation to make sure the model has finished training
by the end of the given number of epochs and that it hasn't over-trained.

In addition, you can set an 'eval_size' flag to determine the number of examples
to evaluate in the evaluation phase. By default, up to 1 million examples are
evaluated. Setting eval_size to 0 will evaluate the whole dataset. The
evaluation is run on a shuffled queue so each eval will grab a random subsample
if the whole dataset is not used.

    run learning/train_feedforward \
    --save_base=xxx/train_feedforward/$USER \
    --epochs=2 \
    --epoch_size=1e6 \
    --eval_size=0 \
    --run_name=$RUN_NAME \
    --num_fc_layers=1 \
    --dataset=aptitude

### Local copies of the datasets

You might also copy `input_dir` to your work station to avoid reading it from
xxx everytime (which is slow):

    [cmd removed]

Then add `--input_dir=/usr/local/google/home/$USER/data/aptamers/tf_data/xxx`
to the `train_feedforward` commmand above. If you copy the data locally,
each epoch (on the xxx dataset) should complete in about 2 minutes.

Running this way, you can watch the output on the command line. After some
setup, the code will kick off pre-training evalution and report metrics. Next
the model will train then evaluate for each epoch of the data.

## Run training/eval on the cluster

Once you are confident the model runs locally, the next step is to run the
training on the cluster. This training can either train one specific model or can be
used to search for the optimal hyper-parameters.

The Aptamer model distinguishes between meta-parameters (also called model
choices) and hyper-parameters. A meta-parameter or model choice, is a variable
that we define and do not let Vizier vary randomly, for example the number of
fully connected layers or the type of output layer. A hyper-parameter is a
variable where Vizier picks the value within a given range unless you specify
the value on the command line. For example, the learning rate or momentum. We
perform a separate hyper-parameter optimization for each combination of
meta-parameters.

The first step is to build the `.par` file:

    [cmd removed]

The script creates jobs for training/eval and exporting events. 
Typically individual training tasks
should complete in a few hours each for smaller datasets. Larger datasets or
bigger epoch sizes can take longer, especially with convolutions. The guidance
on epoch sizes above (in the section on running locally) still apply here. The
results are saved in to the `save_base` path, if no `save_base` path was
provided then the default is xxx.

### Run with fixed hyper-parameters and meta-parameter combination

The following line will kick of a single training job. This can be useful for
testing a single meta-parameter combination.

    [cmd to initiate a Vizier training run] \
    --vars="run_group=$RUN_NAME,output_layer=LATENT_AFFINITY,loss_name=CROSS_ENTROPY,num_fc_layers=2,num_conv_layers=0"

In the final steps of model choice, after picking optimal hyper-parameters, it
is recommended to kick off training with multiple replicas of the same
hyper-parameters and then evaluate the results to look at training variance for
these hyperparameters. An example of this is:

    [cmd to initiate a Vizier training run] \
    --vars="run_group=$RUN_NAME,output_layer=LATENT_AFFINITY,loss_name=CROSS_ENTROPY,num_fc_layers=2,num_conv_layers=0,num_replicas=20,hpconfig=nonlinearity='tanh',learn_rate=0.01,momentum=0.9,dropouts=[0.1,0.4,0.001]"

### Explore hyper-parameters with random search

Typically we use a Vizier random search for our hyper-parameter searching. In
other words, we kick off multiple replicas of each meta-parameter combination
but do not use one of Vizier's search algorithms to adjust these
hyper-parameters based on the results. We just kick off 100 replicas with random
values (within the supplied range) for each hyper-parameter and then manually
analyze these hyper-parameters to determine the best combination of values.

The default run file has 40 different meta-parameter combinations. The default
meta-parameter options are 0,1,2,3 fully connected layers, 5 different types of
output layers, and 2 different types of loss functions. You can override these
defaults by explictly setting these values on the command line as shown below.
Thus the command below will kick off 4000 training jobs, 100 for each of the 40
meta-parameter combinations.

     [cmd to initiate a Vizier training run] \
     --vars="run_group=$RUN_NAME,num_replicas=100,autotune=true"


## Predict 1D auxiliary features as additional tasks

To predict 1D auxiliary features such as partition function as additional tasks,
concatenate all the features to predict with comma, encapsulate it with double
quotes and provide it as the value for flag "additional_output". Valid feature
names are the keys of "feature_tensors" in learning/data.py.
Description of how the predictions are made and how the loss is handled can be
found in Predicting Affinity vs. Counts (output_layers.py) - Aptamer Tf Model
document.

    [cmd to initiate training]  \
    --save_base=xxx \
    --epochs=2 \
    --epoch_size=1e6 \
    --run_name=$RUN_NAME \
    --num_fc_layers=1 \
    --dataset=aptitude
    --additional_output="partition_function,boltzmann_probability"


