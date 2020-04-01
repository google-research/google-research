MONET: Metadata-Orthogonal Node Embedding Training
===============================

This is the implementation accompanying the manuscript at https://arxiv.org/abs/1909.11793.


Experiment Replication:
-----------------------

Use python 2.7.17 for all scripts. Make a new virtual environment from
requirements.txt. Then run the following scripts in order:

1. polblogs_experiment.py
2. polblogs_eval.py
3. shilling_experiment.py
4. shilling_eval.py

The experiment results will appear in experiment_data/polblogs/exp_results, and
../shilling/.. for the shilling experiment.

Experiment Replication:
-----------------------

The polblogs data was downloaded from:
http://www-personal.umich.edu/~mejn/netdata/polblogs.zip
The political blogs graph data files were formatted according to the
pre-processing described in the manuscript.

The movielens data for the shilling attack was downloaded from:
https://grouplens.org/datasets/movielens/100k

Citing
------
If you find the MONET technique useful in your research, we ask that you cite
the following paper:

> Palowitch, J., Perozzi, B., (2019).
> MONET: Debiasing Graph Embeddings via the Metadata-Orthogonal Training Unit

@article{palowitch2019monet,
  title={MONET: Debiasing Graph Embeddings via the Metadata-Orthogonal Training Unit},
  author={Palowitch, John and Perozzi, Bryan},
  journal={arXiv preprint arXiv:1909.11793},
  year={2019}
}


