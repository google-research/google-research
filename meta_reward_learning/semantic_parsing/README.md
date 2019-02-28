## Learning to Generalize from Sparse and Underspecified Rewards

This directory is under construction and contains a preliminary release of code for
Meta Reward Learning (*MeRL*) used for semantic parsing experiments in our paper
[Learning to Generalize from Sparse and Underspecified Rewards](https://arxiv.org/abs/1902.07198).
It is based on the open source
implementation of Memory Augmented Program Synthesis ([*MAPO*](https://github.com/crazydonkey200/neural-symbolic-machines)).

## Running Experiments

To run an experiment, a training worker is to be launched in conjunction with an
evaluation worker as two separate processes. The `run_single.sh` shows a simple example
for launching the training worker.

Please note that the directory structure needs to be similar to the one specified
in [setup script](https://github.com/crazydonkey200/neural-symbolic-machines/blob/master/aws_setup.sh)
used by MAPO.

This code uses the deprecated [graph_replace](https://www.tensorflow.org/api_docs/python/tf/contrib/graph_editor/graph_replace)
library in tensorflow which might not be supported in the near future.

The `arg_builder.py` file provides helper functions to create the arguments to
be passed to the evaluation and training workers based on command line
arguments and semantic parsing dataset used.

Citing
------
If you use this code in your research, please cite the following paper:

> Agarwal, R., Liang, C., Schuurmans, D., & Norouzi, M. (2019).
> Learning to Generalize from Sparse and Underspecified Rewards.

    @article{agarwal2019learning,
      title={Learning to Generalize from Sparse and Underspecified Rewards},
      author={Agarwal, Rishabh and Liang, Chen and Schuurmans, Dale and Norouzi, Mohammad},
      journal={arXiv preprint arXiv:1902.07198},
      year={2019}
    }

---

*Disclaimer: This is not an official Google product.*
