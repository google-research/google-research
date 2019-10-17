# Energy-Inspired Models (EIMs)
*Implemention of Energy-Inspired Models as described in [Energy-Inspired Models](https://sites.google.com/view/energy-inspired-models/home) by Dieterich Lawson* (jdl404@nyu.edu), George Tucker* (gjt@google.com), Bo Dai, Rajesh Ranganath.

Energy-based models are powerful probabilistic models, however, sampling and computing log-likelihoods is intractable due to the partition function. In practice, they rely on approximate sampling algorithms. Motivated by this, we directly build the approximate sampling algorithm into the model. This yields a class of energy-inspired models (EIMs) that incorporate learned energy functions while still providing exact samples and tractable log-likelihood lower bounds. Moreover, many recent variational bounds can be understood in a unified framework with EIMs as the variational family.

We hope that this code will be a useful starting point for future research in this area.

## Quick Start:

Requirements:
* TensorFlow (see tensorflow.org for how to install)

```
# From the deepest google-research/ run:
python -m eim.small_problems --target=checkerboard --algo=his 
python -m eim.mnist --dataset=static_mnist --proposal=rejection_sampling \
  --model=bernoulli_vae --data_dir=<data directory to cache datasets>
```
The first command runs the small problems. Check small_problems.py to see a list of target distributions and algo choices. The second command runs the larger problems. Check mnist.py to see a list of datasets, proposals, and model choices.

This is not an officially supported Google product. It is maintained by George Tucker (gjt@google.com, [@georgejtucker](https://twitter.com/georgejtucker), github user: gjtucker).
