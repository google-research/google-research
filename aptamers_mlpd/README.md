# Aptamer model training and walking code

## Overview

The main steps for the code in this repository are:

1.  Data preprocessing: code in 'preprocess'
2.  Model training: code in 'learning'
3.  Sequence walking: code in 'search'
4.  Data analysis/figure generation of PD and MLPD count data: code in 'figures'

The sections below describe these steps in more detail, and point to the entry
points used to run the code for the manuscript. Steps 1-3 have been extracted
from the main Google code repository so it is not externally runnable, but all
the code has been provided for review. Steps 4 is provided as ipython notebooks
that can be run with the provided input data.

At the top of most modules is a section of imports labeled "Google internal."
The full code path for these modules has been removed. Below these are the code
references within this repository, where the code paths all start with `from
..aaa import bbb` where `aaa` is one of the directories in this repository and
`bbb` is the name of the module.

Instances of `xxx` indicate where an internal path or collaborator name has been
masked out.

The primary internal libraries used are:

* [FlumeJava](https://research.google/pubs/pub35650/) (similar to Apache beam)
  used to parallelize our sequence processing.
* [Scalable matching](https://arxiv.org/abs/1903.08690) infrastructure, used
  to find similar sequences for clustering.

## Data Preprocessing

The preprocessing code starts with the raw fastq file pairs and ends with a
matrix of sequences with the count in each pair of fastq files (i.e. each round
+ stringency level + result) so the count for a sequence in the column
round2_high_positive indicates the number of times that sequences was seen in
the fastq file from sequencing the results from round 2 at high affinity level,
in the positive (passed FACs) pool.

The output after all the preprocessing is a metadata text file describing the
files for model training and a set of 5 SSTables, each with approximately 1/5th
of the data. All members of a cluster are within a single SSTable. We set up the
5 fold to enable cross-validation. In practice we always trained with fold0 as
the test and the rest of the folds as training data.

For the manuscript, we further limited each SSTable to only contain the 'cluster
representative' (the sequence in the cluster with the highest sequence count).
We found that additional sequences in the cluster mostly appeared to be
sequencing errors, not interesting error-prone PCR exploration of the space.
Training on only the cluster max was substantially faster, and yielded models
with equivalent AUC accuracy.

The data preprocessing code is time-consuming and requires internal
dependencies. However, we have provided the raw fastq alongside the
preprocessing output so that our claims can be validated without running the
full processing code.

We used an internal pipelining tool to stitch together all the pieces of the
preprocessing. The Java steps use the internal FlumeJava API and are not 
provided. For the remaining steps, the binary provides usage information:

*   preprocess/fastq_to_sstable

    *   Converts a pair of fastqs to an SSTable
    *   Covers just one sequencing file pair (one particle display), for example
        Round2_high_positive.

*   preprocess/merge/merge_measurements_main

    *   C++ Flume code to merge the SSTables created by running fastq_to_counts
        on each sequencing pair.
    *   Most sequences are very low count and do not overlap between rounds,
        especially in Round 1. Therefore most sequences have a count of 1 in a
        single SSTable input, and have counts of zero in all other SSTables.
    *   The output from this is a single, merged SSTable where the key is the
        aptamer sequence and the value is the count in every PD round.

*   SplitMeasurementsForClustering (not provided)

    *   This FlumeJava module splits the merged SSTable into a set of sequences
        to be used in the all-pairs clustering.

*   preprocess/scam_featurize_main

    *   This C++ code converts the all-pairs and the projection sequences from
        sequence strings into vectors for ScaM to calculate cosine distance.

*   run_all_pairs and run_projections

    *   The SSTables of vectors are in the right format for the internal tool
        ScaM, described in Wu, et al 2019
        (https://arxiv.org/pdf/1903.08690.pdf). (The setup of this tool and the
        way we used it is all based on internal Google tools so the usage
        information is not provided here.)

*   RunPreprocessingPipeline (not provided)

    *   This FlumeJava pipeline takes the results of ScaM and uses them to
        output the final clustering of sequences.
    *   While ScaM is both fast and accurate, it is still an approximation. We
        run it with highly permissive settings where we expect some components
        to be connected even though their Levenshtein distance is actually
        greater than the desired distance (5 in this case). This Java Flume
        pipeline runs over all the clusters removing linkages that are longer
        than the desired Levenshtein distance. Because each individual cluster
        is now much smaller, this calculation is feasible.

## Model training

The playbook in learning/playbook.md describes the usage of this module.
Input is one SSTable for each fold and metadata, and the output of this code is
the trained models. The training can be kicked off by running
learning/train_feedforward.py, though in practice we used an internal version of
Vizier to simultaneously train with many different hyper-parameters.

Note: The initial version of this code was written against an early internal
version of tensorflow that did not yet have a lot of helper utilities. As such,
the code is far more complicated than a model written using modern Tensorflow.

We have also included a high-level overview of various architecture differences
adapted from our internal documentation
(learning/Aptamer_TF_models_for_pub.pdf). Additional parameters specific to the
manuscript for the Binned and superBin models are provided in
learning/README.md.

The specific train_feedforward parameters for the models used in the manuscript:

*   **Counts Model**:

```
--affinity_target_map=aptitude
--additional_output=partition_function
--epoch_interval_to_save_best=5
--epoch_size=200000
--epochs=50
--input_features=SEQUENCE_ONE_HOT,SEQUENCE_KMER_COUNT \
--loss_name=SQUARED_ERROR
--loss_norm=STANDARDIZE
--max_strides=1
--mbsz=64
--num_conv_layers=3
--num_fc_layers=3 \
--output_layer=LATENT_AFFINITY
--preprocess_mode=PREPROCESS_SKIP_ALL_ZERO_COUNTS \
--target_names=ALL_OUTPUTS
--total_reads_defining_positive=1000
--tuner_algorithm=RANDOM_SEARCH \
--tuner_loss=auc/true_top_1p
--tuner_target=mean
```

*   **Binned Model**:

```
--affinity_target_map=aptitude_binned
--additional_output=
--epoch_interval_to_save_best=5
--epoch_size=20000
--epochs=50
--input_features=SEQUENCE_ONE_HOT,SEQUENCE_KMER_COUNT
--loss_name=SQUARED_ERROR
--loss_norm=SKIP
--max_strides=1
--mbsz=64
--num_conv_layers=3
--num_fc_layers=0
--output_layer=FULLY_OBSERVED
--preprocess_mode=PREPROCESS_ALL_COUNTS
--target_names=ALL_OUTPUTS
--total_reads_defining_positive=0
--tuner_algorithm=RANDOM_SEARCH
--tuner_loss=auc/true_top_1p
--tuner_target=mean
```

*   **SuperBin Model**:

```
--affinity_target_map=aptitude_super_binned
--additional_output=
--dependency_norm=SKIP
--epoch_interval_to_save_best=5
--epoch_size=20000 \
--epochs=50
--input_features=SEQUENCE_ONE_HOT,SEQUENCE_KMER_COUNT
--loss_name=SQUARED_ERROR \
--loss_norm=SKIP
--mbsz=64 \
--num_conv_layers=3
--num_fc_layers=3
--output_layer=FULLY_OBSERVED \
--preprocess_mode=PREPROCESS_ALL_COUNTS \
--target_names=ALL_OUTPUTS
--total_reads_defining_positive=0
--tuner_algorithm=RANDOM_SEARCH \
--tuner_loss=auc/true_top_1p
--tuner_target=mean
```

## Sequence walking

The walking code uses the trained ML models (or a random model) to search the
local space around a seed sequence for sequences with high, relative, affinity.

For these experiments, the steps were:

*   Select seed sequences to start the search from.
    *   Experimental seeds from PD (the original particle display).
    *   Random seeds were selected by generating random sequences.
    *   ML Seeds were selected by scoring many random sequences with a machine
        learning model, retaining those with the highest score. See
        'search_inference.py' for the code used for this.
*   Perform iterative rounds of searching, where for each round we made a set of
    mutations and then scored the mutants using a model, keeping the best
    scoring sequences as 'parents' for the next round of mutations. The walking
    code is in the walkers model.

The searching code was called as follows:

```
# Assume we've created a list "seed_seqs" with the sequences to walk.
# Assume we have a variable "inferer" that is a learning/eval_feedforward.Inferer
# that uses a trained model to predict affinity for sequence(s).

# Generation counts (sequences generated at each number of steps walked from
#   a seed) should be the same for the random and non-random walkers
PER_SEED_STEPS = {1: 2, 2: 2, 3: 2, 4: 0, 5:5}
# For improving the sequences found by inference, just save 1 attempt,
#   at 3 steps out
FROM_INFERENCE_STEPS = {3:1}

# create a list to hold all the sequences to test
choice_protos = []

# create the walker that uses a random model and walk seeds with it.
random_walker = walkers.SamplingWalker(
  inferer=None, model_name='Random_Sampler', sampling_limit=1000,
  max_trials=2000, target_molecule='target', n_parents=10,
  min_distance=1, max_distance=4,
  random_seed=search_pipeline.RANDOM_SEED)
choice_protos.extend(random_walker.generate_mutants(seed_seqs, PER_SEED_STEPS))

# create the walker using the inferer, and walk seeds with it.
inference_walker = walkers.SamplingWalker(
  inferer=inferer, model_name=model_name+'_inference_walked',
  sampling_limit=1000, max_trials=2000,
  target_molecule='target', n_parents=10,
  min_distance=1, max_distance=4,
  random_seed=search_pipeline.RANDOM_SEED)
choice_protos.extend(inference_walker.generate_mutants(inference_seeds,
  FROM_INFERENCE_STEPS))
```

## Data Analysis / Figure Generation

The code/analyses used to create the manuscript figures are provided as ipython
notebooks. These can either be run locally or via
https://colab.research.google.com. We have tried to minimize the number of
inputs and depedencies.

### System Requirements

-   Python 2 (tested using 2.7.17)
-   matplotlib
-   numpy
-   pandas
-   plotnine
-   seaborn

### Installation

No specific installation (outside of ipython notebooks and the above
requirements) is required.

### Instructions for use

A summary is located at the top each notebook explaining each notebook. The
"Load in Data" section of each notebook, details the required input files. Note,
when running locally please skip the cell that loads the files onto google colab
(i.e the cell with "from google.colab import files").

Each notebook can be run independently. Once data is loaded (if uploading to
colab.research.google.com this may take a few minutes depending on connection),
each notebook runs in under 10 minutes.

Summaries of each notebook follow below:

#### Figure 2

This notebook summarizes the numbers of aptamers that appear to be enriched in
positive pools for particular particle display experiments. These values are
turned into venn diagrams and pie charts in Figure 2.

The inputs are csvs, where each row is an aptamer and columns indicate the
sequencing counts within each particle display subexperiment.

Inputs: pd_clustered_input_data_manuscript.csv, mlpd_input_data_manuscript

#### Figure 3

This notebook generates the estimated affinity values (and corresponding summary
plots) for experimental seeds (Figure 3A) and different walking strategies
(Figure 3B).

The input is a csv where each row is an aptamer and columns indicate the
sequencing counts each MLPD subexperiment.

Inputs: mlpd_input_data_manuscript.csv

#### Figure 4

This notebook loads in all the simulated truncations (with model scores) as well
as the experimentally tested subset of truncations. It then creates a plot
showing the distribution of model scores for each tested truncation.

Input: truncation_option_seed_scores_manuscript.csv
