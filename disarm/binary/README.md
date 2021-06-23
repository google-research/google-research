# DisARM: An Antithetic Gradient Estimator for Binary Latent Variables

This python code allows you to generate results from
NeurIPS 2020 Spotlight Paper: DisARM: An Antithetic Gradient Estimator for Binary Latent Variables, by Zhe Dong, Andriy Mnih, George Tucker.

Paper link: https://arxiv.org/abs/2006.10680
NeurIPS link: https://papers.nips.cc/paper/2020/hash/d880e783834172e5ebd1868d84463d93-Abstract.html

Please find the required packages in `requirements.txt`.

The python binary for experiments on VAE with a single stochastic layer is in `experiment_launcher_singlelayer.py`, with multiple stochastic layers is in `experiment_launcher_multilayer.py`. There are 5 supported datasets are supported: `static_mnist`, `dynamic_mnist`, `fashion_mnist`, and `omniglot`.

The following `grad_type`s are supported: ARM, REINFORCE LOO, DisARM, RELAX and etc.

To launch experiments, call the binary with:

```shell
python -m disarm.experiment_launcher_singlelayer.py \
  --dataset=dynamic_mnist \
  --logdir=/tmp/disarm/dynamic_mnist \
  --grad_type=disarm \
  --encoder_type=nonlinear \
  --num_steps=1000000 \
  --demean_input \
  --initialize_with_bias
```
For multi-layer case, replace `experiment_launcher_singlelayer.py` with `experiment_launcher_multilayer.py`. The encoder and decoder structure of the VAE model, e.g. number of layers etc., can be specified in `experiment_launcher_multilayer.py`.

