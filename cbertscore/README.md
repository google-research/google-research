The code for [Clinical BERTScore: An Improved Measure of Automatic Speech Recognition
Performance in Clinical Settings](https://arxiv.org/abs/2303.05737), an
Interspeech 2023 submission.

The main contributions of this work are:

1. The Clinician Transcript Preference (CTP) dataset, consisting clinician
   preferences describing which transcript errors are less acceptable in a
   clinical context. The data can be found [here](https://osf.io/tg492/).
1. A metric of textual similarity that correlates much more strongly with
   clinician preferences, and incorporates medical knowledge. It can be used
   to train or evaluate ASR models in clinical settings.

This repo contains:

1. Code for reading the CTP.
1. Code for analyzing the CTP.
1. Code to run inference with CBERTScore.