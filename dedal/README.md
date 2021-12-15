# Dedal: Deep embedding and alignment of protein sequences.
This package contains all the necessary tools to reproduce the experiments presented in the [dedal paper](https://www.biorxiv.org/content/10.1101/2021.11.15.468653v1).

## Install
To install dedal, it is necessary to clone the google-research repo:
```
git clone https://github.com/google-research/google-research.git
```
From the `google_research` folder, you may install the necessary requirements by executing:
```
pip install -r dedal/requirements.txt
```

## Run
To start training a Dedal network, one can simply run, from the `google_research` folder, the following command line:
```
python3 -m dedal.main --base_dir /tmp/dedal/ --task train --gin_config dedal.gin
```
Note that the transformer architecture might be slow to train on a CPU and running on accelerators would greatly improve the training speed.

The first parameter is the folder where to write the checkpoints and to log the metrics. In the example above, it would be `/tmp/dedal/`. To visualize the logged metrics, one can simply start a tensorboard pointing to the given folder, such as:
```
%tensorboard --logdir /tmp/dedal
```

In case the training is interrupted, restarting the same command would not start the training over from scratch, but from the last available checkpoint. The frequency of checkpointing and logging can be changed from the gin config, in `base.gin`.

The `task` flag enables to either run a training, an evaluation of a downstream training with its own eval. In evaluation mode, the training checkpoints will be loaded on the fly until the last one has been reached, such that one can run
both an eval process along with a training one, so that the evaluation does not slow the training down. Alternatively, one can set `separate_eval=False` in the training loop so that eval and train will be run alternatively.

To play with the dedal configuration, for example changing a parameter, the
encoder or even add an extra head, one should get inspiration from the `base.gin`, `dedal.gin` and `substitution_matrix.gin` config files. The first one contains the configuration of the training loop, the data, metrics, losses, while the two others only contains what in the network is specific to dedal or to the substitution matrix based sequence alignment methods.

## Data
This repo does not contain real-world data. Training uses synthetic data sampled on-the-fly for illustration purposes. However, the repo does contain tools to build the datasets to be fed to Dedal for training or eval. Sequence identifiers to reproduce all Pfam-A seed splits can be downloaded [here](https://drive.google.com/file/d/11S2OdnduXM3id7F3k6kUxi8_qXJC8bav/view?usp=sharing).

## License
Licensed under the Apache 2.0 License.

## Disclaimer
This is not an official Google product.
