# Frechet Audio Distance

This repository provides supporting code used to compute the Fréchet Audio Distance (FAD), a reference-free evaluation metric for audio generation algorithms, in particular music enhancement.

For more details about Fréchet Audio Distance and how we verified it please check out our paper:

* K. Kilgour et. al.,
  [Fréchet Audio Distance: A Metric for Evaluating Music Enhancement Algorithms](https://arxiv.org/abs/1812.08466),

## Useage

FAD depends on:

*   [`numpy`](http://www.numpy.org/)
*   [`scipy`](http://www.scipy.org/)
*   [`tensorflow`](http://www.tensorflow.org/)
*   [`Pyton Beam`](https://beam.apache.org/documentation/sdks/python/)
*   [`Audioset model Vggish`](https://github.com/tensorflow/models/tree/master/research/audioset)

and also requires downloading a VGG model checkpoint file:

*   [VGGish model checkpoint](https://storage.googleapis.com/audioset/vggish_model.ckpt)

### Example installation and use

#### Get the FAD code

```shell
$ git clone https://github.com/google-research/google-research.git
$ cd google-research
```

#### Install dependencies
Create a virtualenv to isolate from everything else and activate it first.

```shell
# Python 2
$ virtualenv fad
# or Oython 3
$ python3 -m venv fad # (apache-beam does not yet support Python 3)
# activate the virtualenv
$ source fad/bin/activate
# Upgrade pip
$ python -m pip install --upgrade pip
# Install dependences
$ pip install apache-beam numpy scipy tensorflow
```

#### Clone TensorFlow models repo into a 'models' directory.
```shell
$ mkdir tensorflow_models
$ touch tensorflow_models/__init__.py
$ svn export https://github.com/tensorflow/models/trunk/research/audioset tensorflow_models/audioset/
$ touch tensorflow_models/audioset/__init__.py
```

#### Download data files into a data directory
```shell
$ mkdir -p data
$ curl -o data/vggish_model.ckpt https://storage.googleapis.com/audioset/vggish_model.ckpt
```

#### Create test files and file lists
This will generate a set of background test files (sine waves at different frequencies).
And two test sets of sine waves with distortions.

```shell
$ python -m frechet_audio_distance.gen_test_files --test_files "test_audio"

#Add them to file lists:
$ ls --color=never test_audio/background/*  > test_audio/test_files_background.cvs
$ ls --color=never test_audio/test1/*  > test_audio/test_files_test1.cvs
$ ls --color=never test_audio/test2/*  > test_audio/test_files_test2.cvs
```

#### Compute embeddings and eastimate multivariate Gaussians
```shell
$ mkdir -p stats
$ python -m frechet_audio_distance.create_embeddings_main --input_files test_audio/test_files_background.cvs --stats stats/background_stats
$ python -m frechet_audio_distance.create_embeddings_main --input_files test_audio/test_files_test1.cvs --stats stats/test1_stats
$ python -m frechet_audio_distance.create_embeddings_main --input_files test_audio/test_files_test2.cvs --stats stats/test2_stats
```

#### Compute the FAD from the stats
```shell
$ python -m frechet_audio_distance.compute_fad --background_stats stats/background_stats --test_stats stats/test1_stats
$ python -m frechet_audio_distance.compute_fad --background_stats stats/background_stats --test_stats stats/test2_stats
```
