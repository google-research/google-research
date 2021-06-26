# Coupled Gradient Estimators for Discrete Latent Variables.

This python code allows you to generate results from Coupled Gradient Estimators for Discrete Latent Variables by Zhe Dong, Andriy Mnih and George Tucker.

Paper Link: https://arxiv.org/abs/2106.08056

Please find the required packages in `requirements.txt`.

The python binary is in `experiment_launcher.py`. There are three datasets are supported: `dynamic_mnist`, `fashion_mnist`, and `omniglot`.


To launch experiments, call the binary with:

```shell
python -m experiment_launcher \
  --dataset=dynamic_mnist \
  --logdir=/tmp/logdir \
  --encoder_type=nonlinear \
  --num_steps=1000000 \
  --demean_input \
  --initialize_with_bias \
  --grad_type=disarm \ 
  --num_variables=32 \
  --num_categories=16 \
  --batch_size=200 \
  --one_hot \
  --stick_breaking \
  --importance_weight \
  --logits_order=ascending
```


