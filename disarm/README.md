# DisARM: An Antithetic Gradient Estimator for Binary Latent Variables

This python code allows you to generate results from
NeurIPS 2020 submission: DisARM: An Antithetic Gradient Estimator for Binary Latent Variables.

Please find the required packages in `requirements.txt`.

The python binary is in `experiment_launcher.py`. There are three datasets are supported: `static_mnist`, `dynamic_mnist`, `fashion_mnist`, and `omniglot`.

For ELBO, the following `grad_type`s are supported: ARM, REINFORCE LOO, DisARM, RELAX and etc. For multi-sample objectives, VIMCO and local-DisARM are supported.

To launch experiments, call the binary with:

```shell
python -m disarm.experiment_launcher \
  --dataset=dynamic_mnist \
  --logdir=/tmp/disarm/dynamic_mnist \
  --grad_type=local-disarm \
  --encoder_type=nonlinear \
  --num_steps=1000000 \
  --num_pairs=10 \
  --demean_input \
  --initialize_with_bias
```


