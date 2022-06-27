## Imitation learning with incremental-gain stability constraints

This repository contains code to generate the tables for tunable IGS systems (Section 5.3) in the following [paper](https://arxiv.org/abs/2102.09161):

    On the Sample Complexity of Stability Constrained Imitation Learning.
    Stephen Tu, Alexander Robey, Tingnan Zhang, and Nikolai Matni.
    arXiv preprint arXiv:2102.09161, 2021.


### Requirements

```
pip3 install -r requirements.txt
```

### Running the code

Run the following command from the parent directory (after installing dependencies):
```
python3 -m incremental_gain.shift [--flags]
```

To see a list of flag options, run:
```
python3 -m incremental_gain.shift --help
```
