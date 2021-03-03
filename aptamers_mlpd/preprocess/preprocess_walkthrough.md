# Preprocessing Aptamer Data

This document describes the steps between receiving a raw fastq files to having
an SSTable ready for consumption by TensorFlow. Once the SSTable of TensorFlow
examples is created, see the learning playbook (/learning/playbook.md)
for the next steps.

For now, all input is fastq files so all data is counts of the number of times a
sequence occurs. The output from preprocessing is an SSTable of TF Example 
protos containing information on the set of sequences.

## Background

Next Generation Sequence (NGS) data arrives as [fastq files]
(https://en.wikipedia.org/wiki/FASTQ_format). Our sequences are generally short,
either 40 or 45 bases. This pipeline is describing how we go from the fastq
files to input in our TensorFlow models. The final outputs are (possibly
sharded) SSTables of TensorFlow Example protos
for each cross-validation fold of data, and HDF5 files with identical content
for rapid processing in Python.

Our sequencing is usually done [paired-end]
(http://www.illumina.com/technology/next-generation-sequencing/paired-end-sequencing_assay.html),
which means the DNA sequences in the Illumina sequencer are read from both
directions. In our case, because our sequences are so short, the full sequence
is read in both directions. This means that if there are no errors, the Read2 or
Reverse Read is simply the [reverse complement]
(http://www.bx.psu.edu/old/courses/bx-fall08/definitions.html) of the Read1 or
Forward Read. The fastq files come in pairs because Read1 and Read2 are stored
in two separate parallel fastq files. In other words, both files have the same
number of fastq records where each record represents one spot of DNA within the
sequencing machine. The fastq records in the pair of files are in the same
order, so by reading through the files in parallel you can get the forward and
reverse sequence from the same spot of DNA.

Throughout this document, "Experiment" refers to the full selection experiment
with all the rounds of SELEX (or Particle Display). Each pair of fastq file
contains the sequences from one pool of DNA in the experiment. In practice, this
means that each pair of fastq files represent one round of selection in the
experiment. See Experiment Proto below for more description.

## Files and where they live

Our data files are all stored on xxx at xxx.
Each set of sequences to be trained on together should have one directory that
stores the fastq files from the run, the quality analysis reports (that are not
in colab), and any post-processing information such as SSTables.

In addition, each NGS run should be associated with a wet lab experiment
protocol which can be found in xxx.

## Experiment Proto

Each full experiment is described by a proto which lists, for each round, the
experimental conditions and the fastq files with the data. The proto
structure is at xxx/util/selection.proto.


## From Fastq To TensorFlow SSTable

We have made a pipeline to cover all the steps to convert from a set
of FASTQ files (each representing aptamer counts in a distinct condition) to a
set of SSTables of Tensorflow.Example protos
suitable for input to the machine learning models.

The high-level overview of the pipeline consists of three stages:

1. Transform raw reads into the count of reads seen for each sequence in each
   condition.
1. Cluster similar sequences together.
1. Split the counts into separate
[cross-validation folds](https://en.wikipedia.org/wiki/Cross-validation_%28statistics%29)
for model training and validation.

Each of the steps is described in more detail below.

### Transforming raw reads into read counts

The initial stage of the preprocessing pipeline is the computation of
per-condition counts of all reads in the dataset. This is relatively
straightforward.

First, the fastq_to_sstable.py
script executes separately for each input of (single- or paired-end) FASTQ files
to create a per-condition SSTable mapping from sequence (SSTable key) to the
number of times that sequence was seen in the dataset (SSTable value). This
script also handles filtering reads with poor quality.

Next, the merge_measurements_main.cc
FlumeC++ pipeline combines all the per-condition SSTables into a single
(possibly sharded) SSTable that maps from sequence (SSTable key) to the number
of times the sequence was seen in each condition in which the read was seen
exactly once (i.e., it is a sparse matrix of counts). These counts are
represented as Measurement proto (util/measurement.proto)
values in the SSTable.

### Clustering similar sequences together

The second stage of the processing pipeline annotates each sequence with a
"cluster", such that similar sequences have the same cluster ID and distinct
sequences have different cluster IDs. This has multiple benefits. First and
foremost, the segregation of data into separate cross-validation folds (the
final preprocessing stage, described below) is performed such that all sequences
from a single cluster are present in the same data fold. This
ensures that we do not overfit models by accidentally "testing on training data"
by testing on sequences that are very similar to those seen in training. Second,
this enables possible downstream processing by cluster (e.g., representing all
sequences in a single cluster by the "representative sequence" that occurs most
often) that preliminary investigations indicate has negligible impact on
model performance and can substantially reduce input data sizes.

The conceptual graph algorithm for identifying clusters is the following:

1. Add all unique sequences as nodes to a graph.
1. For each pair of nodes, calculate [Levenshtein
   distance](https://en.wikipedia.org/wiki/Levenshtein_distance) and add an
   edge to the graph if the distance is <= 5.
1. Assign a unique cluster ID to each connected component of the resultant
   graph.

However, because the input data may be quite diverse (> 1 billion sequences),
the naive algorithm presented above is intractable. We implement an
approximation of the above algorithm in the following way:

First, we split the input data into two distinct subsets using the
SplitMeasurementsForClustering.java
FlumeJava pipeline. One subset is used for an all-pairs comparison to define
initial clusters, and the other is projected into the clusters defined by the
all-pairs subset. To maximize the accuracy of all-pairs cluster detection, its
input sequences are "multi-read" sequences--i.e. those for which the sum of all
read counts in all conditions is greater than one. If this subset of sequences
is "small enough", the subset is then padded with singleton sequences until
either the entire dataset is exhausted or its size is as large as is
computationally tractable.

(Implementation note: For a dataset with a total of N reads, and an all-pairs
read count limit of R, the algorithm above takes O(NR) comparisons rather than
the naive O(N^2). In practice we set R=300,000,000, which can be computed on 10k
machines in under 2 days).

Second, we convert the two subsets of sequences to GenericFeatureVector
protos
representing the counts of all 6-mers in each sequence using the
scam_featurize
FlumeC++ pipeline. This is so distances between sequences can be approximated
efficiently using ScaM rather than calculated explicitly for
all pairs of inputs.

Third, we run ScaM on the generic feature vectors to compute approximate
neighbors, both for the all-pairs subset (comparing it to itself) and the
projection subset (projecting each of those against the all-pairs data).

Finally, we identify true neighbor sequences from the ScaM estimated neighbors
by performing explicit calculation of Levenshtein distance, use the true
neighbors in the all-pairs subset to generate a set of candidate clusters, and
then add the projected sequences into those clusters. This is all implemented as
part of the RunPreprocessingPipeline.java
FlumeJava pipeline.

### Split read counts into separate data folds

Once clustering has been computed, we can split the input data into distinct
folds in a way that ensures similar sequences always appear in the same fold.
We also attempt to split the data such that the distribution of cluster sizes is
similar across folds. This is performed by a greedy bin-filling algorithm for
all clusters with size greater than an input threshold (by default 50 sequences)
and then relying on statistical approximations for all smaller (and much more
numerous) clusters.

With data folds generated, the final pipeline stages are to link all of the
input features and transform the data to its desired final output
representations. As mentioned above, the primary output representation for the
machine learning models are Tensorflow.Example protos. All of these steps are
implemented as part of the
RunPreprocessingPipeline.java
FlumeJava pipeline.

Finally, for ease of analysis in Python, we losslessly transform the
Tensorflow.Example protos into HDF5 files containing pandas.DataFrame objects.

### Running the entire processing pipeline
The above pipeline is managed by a xxx
workflow, so that to run the entire pipeline only the experiment pbtxt file is
required.

## Analysis after clustering / counting

Further analysis on the data should be done after clustering. Exactly what to 
do here depends on the experiment details.
