# ConQUR

Code for ConQUR as described in "ConQUR: Mitigating Delusional Bias in Deep
Q-learning" by Andy Su, Jayden Ooi, Tyler Lu, Dale Schuurmans, and Craig
Boutilier.

The paper is available on arXiv [here](https://arxiv.org/abs/2002.12399).


## Basic Command

To run ConQUR on Pong:

```
python -m conqur.main --save_dir=$HOME/conqur --env_name=Pong --logtostderr
```
