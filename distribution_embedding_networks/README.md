Example code for running Distribution Embedding Networks (DEN) algorithm.

Paper: https://openreview.net/forum?id=F2rG2CXsgO (TMLR'22)

Abstract: We propose Distribution Embedding Networks (DEN) for classification
with small data. In the same spirit of meta-learning, DEN learns from a diverse
set of training tasks with the goal to generalize to unseen target tasks. Unlike
existing approaches which require the inputs of training and target tasks to
have the same dimensionality with possibly similar distributions, DEN allows
training and target tasks to live in heterogeneous input spaces. This is
especially useful for tabular-data tasks where labeled data from related tasks
are scarce. DEN uses a three-block architecture: a covariate transformation
block followed by a distribution embedding block and then a classification
block. We provide theoretical insights to show that this architecture allows the
embedding and classification blocks to be fixed after pre-training on a diverse
set of tasks; only the covariate transformation block with relatively few
parameters needs to be fine-tuned for each new task. To facilitate training, we
also propose an approach to synthesize binary classification tasks, and
demonstrate that DEN outperforms existing methods in a number of synthetic and
real tasks in numerical studies.

Example run:

`python -m distribution_embedding_networks.simulation`
