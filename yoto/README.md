# You Only Train Once

This directory contains the variational auto-encoder experiments from the paper:

[Alexey Dosovitskiy and Josip Djolonga.
 "You Only Train Once: Loss-Conditional Training of Deep Networks."
 International Conference on Learning Representations. 2019.](https://openreview.net/forum?id=HyxY6JHKwr).

## How to run
Note: All commands below have to be run from the `google_research` directory.

First, install the necessary requirements using `pip`:

```sh
pip install -r yoto/requirements.txt
```

We provide two configuration files in `yoto/config`:

* `cifar_conv_vae_yoto.gin`: Trains a variational auto-encoder using YOTO on the CIFAR-10 dataset.
* `shapes3d_conv_vae_yoto.gin`: Train a variational auto-encoder using YOTO on the Shapes3D dataset.

Feel free to have a look and modify these configs if you want to experiment with the various (hyper)parameters.
Depending on which dataset you want to train on, you have to specify the corresponding file to `yoto/main.py`.
For example, to train on the Shapes3D dataset, run (you can of course replace the
model directory by any path you want)

```sh
python -m yoto.main --model_dir=$HOME/yoto/yoto_shapes_3d 				\
                    --gin_config=yoto/config/shapes3d_conv_vae_yoto.gin     \
                    --schedule=train
```

To monitor the progress you just have to launch a [`tensorboard`](https://www.tensorflow.org/tensorboard) instance pointing to the model directory,
for example

```sh
tensorboard --logdir $HOME/yoto/yoto_s3d/
```

and then navigate your browser to http://localhost:6006.

This will train the model and save the model checkpoints to `$HOME/yoto/yoto_shapes_3d`.

Note: We are using [`tensorflow_datasets`](https://www.tensorflow.org/datasets), so if you do not have the corresponding dataset, it will be first downloaded and serialized.

Once training is done, we can evaluate the model and create a [`tensorflow_hub`](https://www.tensorflow.org/hub)-module as follows

```sh
python -m yoto.main --model_dir=$HOME/yoto/yoto_shapes_3d 				\
                    --schedule=eval
```

This will export a hub module under `$HOME/yoto/yoto_shapes_3d/hub_modules`, which can randomly sample and re-construct data, parameterized by the regularization strength in the beta-VAE (see the paper for more details).

To visualize the samples and model reconstructions (similar to Figure 3 in the paper) we have prepared a colab notebook `yoto/colabs/plot_yoto_vae.ipynb`. To see how to create a local colab runtime, please refer to [the official documentation](https://research.google.com/colaboratory/local-runtimes.html).

## Paper

If you use our code, please consider citing the paper:

```
@inproceedings{dosovitskiy2019you,
  title={You Only Train Once: Loss-Conditional Training of Deep Networks},
  author={Dosovitskiy, Alexey and Djolonga, Josip},
  booktitle={International Conference on Learning Representations},
  year={2019}
}
```

### Abstract
In many machine learning problems, loss functions are weighted sums of several terms. A typical approach to dealing with these is to train multiple separate models with different selections of weights and then either choose the best one according to some criterion or keep multiple models if it is desirable to maintain a diverse set of solutions. This is inefficient both at training and at inference time. We propose a method that allows replacing multiple models trained on one loss function each by a single model trained on a distribution of losses. At test time a model trained this way can be conditioned to generate outputs corresponding to any loss from the training distribution of losses. We demonstrate this approach on three tasks with parametrized losses: beta-VAE, learned image compression, and fast style transfer.
