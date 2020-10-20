# Quantum Sample Learning

This directory contains reference code for paper
"Learnability and complexity of quantum distributions"

## Overview

Language modeling is a common use of LSTM where the model predicts a probability
distribution over the next input given the previous inputs so far. Typically
these inputs are words, byte sequences or individual bytes. Here we use byte as
the unit of our model for quantum sample learning. In general, autoregressive
models are expected to perform better with sequences since it's easier to
generate a sentence sequentially (word-by-word) than the entire sentence at
once. In our case, the inputs to the language model are samples of bitstrings
and the model is trained to predict the next bit given the bits observed so far,
starting with a start of sequence token. We use a standard LSTM language model
with a logistic output layer. To sample from the model, we input the start of
sequence token, sample from the output distribution then input the result as
the next timestep. This is repeated until the required number of samples is
obtained. LSTM has shown the best performance so far in four kinds of generative
models we've considered covering GANs, autoregressive GANs and Deep Bolzman
Machine. In this github, we provide original code for our implementation of
LSTM for learning quantum samples.

## Dataset

This code contains the data for 12 qubits. We will release all the data after
acceptence of the paper.

All the data locate at `data/`

### `q12c0.txt`

The theoretical probability distribution of $$2^{12}=4096$$ bitstrings.


### `experimental_samples_q12c0d14.txt`

500,000 bitstrings sampled from a 12 qubits circuit with depth 14.


### `experimental_samples_q12c0d14_test.txt`

The first 100 bitstrings in `experimental_samples_q12c0d14.txt`. This is a small
data for testing.

## Run Language Model

The default flags run a LSTM model with 256 units for 20 epoches with
batch size 64 and learning rate 0.001.

### Use Theoretically Generated Samples

```
python -m quantum_sample_learning.run_lm \
--num_qubits=12 \
--use_theoretical_distribution=True \
--probabilities_path=quantum_sample_learning/data/q12c0.txt
```

The expected output:

```
Min sampled probability 0.000000
Max sampled probability 0.002621
Mean sampled probability 0.000484
Space size 4096
Linear Fidelity: 0.982864
Logistic Fidelity: 0.979632
KL Divergence: 0.021964
theoretical_linear_xeb: 1.018109
theoretical_logistic_xeb: 1.006301
linear_xeb: 0.982864
logistic_xeb: 0.979632
kl_div: 0.021964
```

### Use 2-bit subset parity reordered Porter-Thomas

```
python -m quantum_sample_learning.run_lm \
--num_qubits=12 \
--use_theoretical_distribution=True \
--probabilities_path=quantum_sample_learning/data/q12c0.txt \
--experimental_bitstrings_path=quantum_sample_learning/data/experimental_samples_q12c0d14.txt \
--random_subset=True \
--subset_parity_size=2
```

### Use Experimentally Generated Samples

```
python -m quantum_sample_learning.run_lm \
--num_qubits=12 \
--use_theoretical_distribution=False \
--probabilities_path=quantum_sample_learning/data/q12c0.txt \
--experimental_bitstrings_path=quantum_sample_learning/data/experimental_samples_q12c0d14.txt
```

The expected output:

```
Min sampled probability 0.000000
Max sampled probability 0.002621
Mean sampled probability 0.000329
Space size 4096
Linear Fidelity: 0.347582
Logistic Fidelity: 0.343891
KL Divergence: 0.235616
theoretical_linear_xeb: 0.380668
theoretical_logistic_xeb: 0.374586
linear_xeb: 0.347582
logistic_xeb: 0.343891
kl_div: 0.235616
```

### Simulate Distribution of Random Quantum Circuit

```
python -m quantum_sample_learning.circuit
```
The output contains probability of each bit string obtained by simulating the quantum wave function at the output of a random quantum circuit. Notice that the same file contains the description of the random quantum circuit for 12 qubits.
