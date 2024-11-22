# Agile Modeling

This is a companion codebase and dataset associated with the paper
["Agile Modeling: From Concept to Classifier in Minutes"](https://arxiv.org/abs/2302.12948).

## Introduction
In our Agile Modeling paper, we describe a system that allows users without
Machine Learning experience to create their own image classifiers from scratch,
for any concept they have in their mind.

Here we release a Colab with step-by-step instructions guiding users through
the Agile Modeling process.

Additionally, we also release all the data labeled by our users for the 14
concepts included in our experiments.

## Setting up

### Requirements
1. `Python 3.7`
2. `TensorFlow 2.13.0`
3. `scikit-image`

### Part 1: Preparing the unlabeled data
While the Agile process requires no labeled data to begin with, we have to
prepare the unlabeled pool of images from which the system selects a few images
for the user to label.

In our experiments we used the [LAION-400M open dataset](https://laion.ai/blog/laion-400-open-dataset),
but the Agile framework is not restricted to this. Any unlabeled data works as
long as it is converted to the right format. We further describe how to download
and preprocess the LAION dataset, but feel free to follow similar steps with
your dataset of choice.

Please see Demo.ipynb for instructions on how to download and process the
LAION-400M dataset use in our experiments.

### Part 2: Running the Colab
`Demo.ipynb` contains an end-to-end implementation of our prototype. Please
follow the instructions at the top to process the data, then run `Demo.ipynb`
using [Jupyter notebook](https://docs.jupyter.org/en/latest/running.html) or,
for a nice interface, [Google Colab](https://colab.google/).

## Data collected during our user study
We release all data annotated by domain experts during our user study. Please see the `data` folder to download the datasets along a README file with further details.

## Citation

If you found this codebase useful, please consider citing our paper:

```
@inproceedings{agile_modeling,
  title={Agile Modeling: From Concept to Classifier in Minutes},
  author={Stretcu, Otilia and Vendrow, Edward and Hata, Kenji and
          Viswanathan, Krishnamurthy and Ferrari, Vittorio and Tavakkol, Sasan
          and Zhou, Wenlei and Avinash, Aditya and Luo, Enming and
          Alldrin, Neil Gordon and Bateni, MohammadHossein and Berger, Gabriel
          and Bunner, Andrew and Lu, Chun-Ta and Rey, Javier A and
          DeSalvo, Giulia and Krishna, Ranjay and Fuxman, Ariel
  },
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer
             Vision, {ICCV} 2023, Paris, France, October 2-6, 2023},
  year={2023}
}