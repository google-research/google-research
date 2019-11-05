# Investigating Multilingual NMT Rperesentations at Scale

This directory contains a Colab recreating the plots for:

Kudugunta, S.R., Bapna, A., Caswell, I., Arivazhagan, N., & Firat, O. (2019). [Investigating Multilingual NMT Representations at Scale](https://arxiv.org/abs/1909.02197). ArXiv, abs/1909.02197.


To launch the colab, [click here](https://colab.google.com/github/google-research/google-research/blob/master//investigating-nmt/Investigating_MNMT.ipynb)


## Data

Here, we describe each file in investigating_m4.zip. More details about how the
data was created may be found in our paper.

* model_scores.csv
  - “Language Pair 1 (str),Language Pair 2 (str), Layer, Checkpoint 1 (int), Checkpoint 2 (int), SVCCA Score (float)”
  - Each row provides this data: “The activations of Language Pair 1 of Layer of the model at Checkpoint 1, when compared with the activations of Language Pair 2 of Layer of the model at Checkpoint 2 have an aggregate similarity score of SVCCA Score.”

* finetuned_relative_bleu.csv
  - "Finetuning Language Pair", "Language Pair", "Relative BLEU Change"
  - “Each row provides this data: “The relative change in BLEU of Language Pair after finetuning with Finetuning Language Pair, when compared with the activations of the model before finetuning is Relative BLEU Change.”

* finetuned_model_scores.csv
- “"Layer, Finetuning Language Pair, Language Pair, SVCCA Score"
- Each row provides this data: “The activations of Language Pair of Layer of the model after finetuning with Finetuning Language Pair, when compared with the activations of the model before finetuning have an aggregate similarity score of SVCCA Score.”

* properties.csv
- “Language Code,Language Name,Morphological typology,Family,Script,Sub Family”
- Each row provides the languages codes and some linguistic details of all the languages used in our dataset.
