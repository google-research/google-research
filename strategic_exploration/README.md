# Introduction

**Authors**: Evan Zheran Liu, Ramtin Keramati, Sudarshan Seshadri, Kelvin Guu, Panupong Pasupat, Emma Brunskill, Percy Liang

Code accompanying our paper: Learning Abstract Models for Strategic Exploration and Fast Reward Transfer

# Setup

All of the below commands assume that the user is at the base directory of the repository.
This code depends on Python3.5.

Create a Python3.5 virtual env:

`virtualenv -p python3.5 .`
`source ./bin/activate`

Install the appropriate packages:

`pip install -r requirements.txt`

Set the `HRL_DATA` environmental variable to point to the data directory:

`export HRL_DATA="data/"`

Download the arial font `arial.ttf` (or any other font) and place it in the path `data/arial.ttf`.

# Run

The following commands reproduce the results from the paper in each of the domains.

Montezuma's Revenge:
`python -m main configs/default-systematic.txt configs/config-mixins/extra_repeat.txt`

Pitfall:
`python -m main configs/default-systematic.txt  configs/config-mixins/extra_repeat.txt configs/task-mixins/pitfall.txt -n pitfall`

Private Eye:
`python -m main configs/default-systematic.txt  configs/config-mixins/extra_repeat.txt configs/task-mixins/private_eye.txt -n private_eye`

# Results

Raw results reported in our paper can be found in the `data/paper_results` directory.
Each subdirectory contains the results of four random seeds in `0.csv`, `1.csv`, `2.csv`, and `3.csv`
Each csv file contains four columns:

* Frames: in increments of 1M.
* Reward: the reward achieved by the agent when evaluated at the given number of frames.
* Visited rooms: the number of rooms the agent has visited during training at the given number of frames.
* Learned transitions: the number of transitions the worker has learned at the given number of frames.
