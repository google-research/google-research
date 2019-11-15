
# Stacked Capsule Autoencoders

This is an official Tensorflow implementation of the Stacked Capsule Autoencoder (SCAE), which was introduced in the in the following paper:
[A. R. Kosiorek](http://akosiorek.github.io/), [Sara Sabour](https://ca.linkedin.com/in/sara-sabour-63019132), [Y. W. Teh](https://www.stats.ox.ac.uk/~teh/), and [Geoffrey E. Hinton](https://vectorinstitute.ai/team/geoffrey-hinton/), ["Stacked Capsule Autoencoders"](https://arxiv.org/abs/1906.06818).

  * **Author**: Adam R. Kosiorek, Oxford Robotics Institute & Department of Statistics, University of Oxford
  * **Email**: adamk(at)robots.ox.ac.uk
  * **Webpage**: http://akosiorek.github.io/

This work was done during Adam's internship at Google Brain in Toronto.

## About the project
If you look at natural images containing objects, you will quickly see that the same object can be captured from various viewpoints. Capsule Networks are specifically designed to be robust to viewpoint changes, which makes learning more data-efficient and allows better generalization to unseen viewpoints. This project introduces a novel unsupervised version of Capsule Networks called Stacked Capsule Autoencoders (SCAE). Unlike in the original Capsules, SCAE is a generative model with an affine-aware decoder. This forces the encoder to learn image representation that is equivariant to viewpoint changes, and which leads to state-of-the-art unsupervised classification performance on MNIST and SVHN. For a more detailed description please have a look [at the paper](https://arxiv.org/abs/1906.06818) or at [Adam's blog](http://akosiorek.github.io/ml/2019/06/23/stacked_capsule_autoencoders.html).


<p align="center">
<img alt="TSNE embeddings of object-capsule presence probabilities on MNIST digits." src="https://raw.githubusercontent.com/google-research/google-research/master/stacked_capsule_autoencoders/.resources/tsne.png">
<p align="center">
<b>Fig 1:</b> TSNE embeddings of object-capsule presence probabilities on MNIST digits color-coded by digit class.
 </p>
</p>

## Dependencies
If you execute `setup_virtualenv.sh`, it will create a virtual environment and install all required dependencies. Alternatively, you can install all the dependencies using `pip install -r requirements.txt`. You can also manually install [Tensorflow v1.15](https://www.tensorflow.org/install) and the following dependencies:
  * `absl_py`>=0.8.1
  * `imageio`>=2.6.1
  * `matplotlib`>=3.0.3
  * `monty`>=3.0.2
  * `numpy`>=1.16.2
  * `Pillow`>=6.2.1
  * `scikit_learn`>=0.20.4
  * `scipy`>=1.2.1
  * `dm_sonnet`==1.35
  * `tensorflow`==1.15.0
  * `tensorflow_probability`==0.8.0
  * `tensorflow_datasets`==1.3.0

The current implementation is not compatible with `tensorflow`==2.0.

## Understanding the Code
  * The training loop is defined in `train.py`.
  * The model is built in `capsules/configs/model_config.py`.
  * The part capsule encoder and decoder are defined in `capsules/primary.py`.
  * The object capsule can be found in `capsules/capsule.py`.
  * The SCAE model is defined in `capsules/models/scae.py`.


## Running Experiments
You can train the model by invoking:

    # From google-research/
    python -m stacked_capsule_autoencoders.scripts.train --name=experiment_name

Different model configurations and datasets can be chosen by command-line flags. For an overview of the flags, have a look at the top of the training script (`train.py`) and the top of model and data configs in `capsules/configs`.

To replicate experiments reported in the paper you can run one of the `run_*.sh` files:
  1) `run_mnist.sh` runs the full MNIST model as reported in the paper.
  2) `run_mnist_coupled.sh` runs the full MNIST model, with the difference that the part mixing probabilities are constrained to be the same as pixel intensities. We used a similar setup for the SVHN experiments.
  3) `run_constellation.sh` runs the constellation experiment.


You can also pass command line arguments to the above scripts. For example,

    ./run.sh --batch_size=128

 will invoke `run.sh` script and set the batch size to 128.
 
 Model snapshots, tensorboard logs and intermediate plots (needs passing `--plot=True`) will be stored in `{logdir}/{name}` directory where `logdir` and `name` can be set as command-line arguments. `logdir` defaults to `google-research/stacked_capsule_autoencoders/checkpoints`.

## Evaluation
You can use the `eval_mnist_model.py` script to evaluate a trained (on MNIST) model snapshot. To do so, you need to invoke the script with the same command-line arguments you used to train the model. We provide `eval_mnist.sh` and `eval_mnist_coupled.sh` for evaluating snapshots produced by `run_mnist.sh` and `run_mnist_coupeld.sh`. Invoking this script will load the model snapshot and report linear classification accuracy, unsupervised classification accuracy (clustering + bipartite graph matching), and also produce a TSNE plot of capsule probabilities. The plot will be stored in the checkpoint directory.

## Citation

If you find this repo or the corresponding paper useful in your research, please consider citing:

    @inproceedings{Kosiorek2019scae,
      title={Stacked Capsule Autoencoders},
      author={Kosiorek, Adam Roman and Sabour, Sara and Teh, Yee Whye and Hinton, Geoffrey Everest},
      booktitle={Advances in Neural Information Processing Systems},
      url = {https://arxiv.org/abs/1906.06818},
      pdf = {https://arxiv.org/pdf/1906.06818},
      year={2019}
    }

## Release Notes
**Version 1.0**
* Original implementation; contains the constellation and MNIST experiments.
