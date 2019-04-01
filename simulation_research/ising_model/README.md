# High Performance Monte Carlo Simulation of Ising Model on TPU Clusters

This is the implementation accompanying the paper pending submission: ["High Performance Monte Carlo Simulation of Ising Model on TPU Clusters"](https://arxiv.org/abs/1903.11714).

## Example Usage
1. Go to https://colab.research.google.com

2. In the pop-up window, select tab GITHUB and enter https://github.com/google-research/google-research, then search for ising_mcmc_tpu.ipynb and click to load the notebook into colaboratory.

3. In menu Edit -> Notebook settings -> Hardware accelerator, select TPU.

4. Then click CONNECT, and you will be able to simulate Ising model. Please be aware that the free TPU on colab is TPUv2, to repeat our experiments in the paper, TPUv3 is needed, which is avaiable on Google Cloud.

Citing
------
```none
@ARTICLE{yang2019isingtpu,
  author = {Yang, Kun and Chen, Yi-Fan and Roumpos, Georgios and Colby, Chris and Anderson, John},
  title = {High Performance Monte Carlo Simulation of Ising Model on TPU Clusters},
  journal = {arXiv preprint arXiv: 1903.11714},
  year = {2019}
}

