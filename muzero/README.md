# Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model
Paper: https://arxiv.org/abs/1911.08265

This directory contains an implementation of the Pseudocode description of
the MuZero algorithm (https://arxiv.org/src/1911.08265v1/anc/pseudocode.py).

The implementation uses
[SEED RL](https://github.com/google-research/seed_rl) for scalable RL training.

## Pull Requests
At this time, we do not accept pull requests. We are happy to link to forks
that add interesting functionality.

## Prerequisites
We require tensorflow and other supporting libraries. Tensorflow should be
installed separately following the docs.

SEED RL should be installed following
instructions [here](https://github.com/google-research/seed_rl#prerequisites).

To install the other dependencies use

```
pip install -r requirements.txt
```

## Training
Follow instructions from the SEED repo to run
[Local Machine Training](https://github.com/google-research/seed_rl#local-machine-training-on-a-single-level)
or [Distributed Training](https://github.com/google-research/seed_rl#distributed-training-using-ai-platform).

This directory adds a `tictactoe` environment and an `atari` environment. These
can be used as the `$ENVIRONMENTS` when running the seed_rl scripts.

This directory also adds a `muzero` agent which can be used as the `$AGENTS`
when running the seed_rl scripts.
