# Momentum Contrast Training example for Flax

*NOTE* this implementation works, but does not achieve the same linear
classifier protocol result as in the paper; the paper reports 60.8% accuracy
(39.2% error) with a ResNet-50 based network; we get around 56% accuracy.

Written by Geoff French during an internship at the Google Amsterdam office
hosted by Avital Oliver and Tim Salimans.

This is an implementation of 'Momentum Contrast for Unsupervised Visual
Representation Learning' by He et al. 2019 (https://arxiv.org/abs/1911.05722).

