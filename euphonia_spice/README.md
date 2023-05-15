# SpICE: Speech Intelligibility Classifiers from Euphonia

Paper: [Speech Intelligibility Classifiers from Half-a-Million Utterances]()

By [Subhashini Venugopalan](https://vsubhashini.github.io), Jimmy Tobin, Samuel Yang, Katie Seaver, Richard J.N. Cave, Pan-Pan Jiang, Neil Zeghidour, Rus Heywood, Jordan Green, Michael P. Brenner

# SpICE

SpICE is a family of models that takes as input speech samples and outputs
intelligibility/extent of dysarthria (0-4) mapping to (typical, mild, moderate,
severe, profound).

We developed SpICE classifiers on 551,176 disordered speech samples
 contributed by a diverse set of 468 speakers, with a range of self-reported
 speaking disorders and rated for their overall intelligibility on a five-
point scale. We trained three models following different deep
 learning approaches and evaluated them on ∼94K utterances
 from 100 speakers. We further found the models to generalize well
 (without further training) on the TORGO database
 (100% accuracy), UASpeech (0.93 correlation), ALS-TDI
 PMP (0.81 AUC) datasets as well as on a dataset of realistic unprompted
 speech we gathered (106 dysarthric and
 76 control speakers, ∼2300 samples). We open source the
 weights of one of our models to help the community advance
 research in this domain.

This repository contains notebook examples to show how to use the models.

## Models

* Keras model available on TFHub here:
  [https://tfhub.dev/google/euphonia_spice/classification/1](https://tfhub.dev/google/euphonia_spice/classification/1)
* Pytorch model - coming soon.

# Usage

* [**SpICE_cls_keras.ipynb**](https://github.com/google-research/google-research/blob/master/euphonia_spice/SpICE_cls_keras.ipynb) shows an example of how to use the huggingface wav2vec2 model with the keras version of the classifier on a sample audio file.

* [**SpICE_cls_pytorch.ipynb**]https://github.com/google-research/google-research/blob/master/euphonia_spice/SpICE_cls_pytorch.ipynb) shows an example of how to use the pytorch version of the classifier on a sample audio file.



