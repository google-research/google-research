# xirl

This code accompanies the paper [XIRL: Cross-embodiment Inverse Reinforcement Learning](https://x-irl.github.io/).

If you find this code useful, consider citing our work:

```latex
@inproceedings{xxxx2021xirl
  Coming soon...
}
```

## Installation

xirl requires Python 3.8 or higher. We recommend using an [Anaconda](https://docs.anaconda.com/anaconda/install/) environment for installation. The following instructions will walk you through the setup, tested on Ubuntu 20.04.2 LTS.

```bash
# Create and activate environment.
conda create -n xirl python=3.8
conda activate xirl

# Install pytorch and torchvision.
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch

# Install dependencies.
pip install -r requirements.txt

# Install the x-MAGICAL benchmark.
git clone https://github.com/kevinzakka/x-magical.git
cd x-magical
pip install -e .

# Clone and install pytorch SAC codebase.
git clone https://github.com/kevinzakka/pytorch_sac.git
cd pytorch_sac
pip install -e .
```
