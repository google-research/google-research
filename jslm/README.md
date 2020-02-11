# Adaptive Language Models in JavaScript

This directory contains collection of simple adaptive language models that are
cheap enough memory- and processor-wise to train in a browser on the fly.

## Language Models

*  Prediction by Partial Matching (PPM) [model](ppm_language_model.js).

## Example

Please see a simple example usage of the model API in [example.js](example.js).

## Test Utility

A simple test driver [language_model_driver.js](language_model_driver.js) can be
used to check that the model behaves using [NodeJS](https://nodejs.org/en/). The
driver takes three parameters: the maximum order for the language model, the
training file and the test file in text format.

Example:

```shell
> node --max-old-space-size=4096 language_model_driver.js 7 training.txt test.txt
Initializing vocabulary from training.txt ...
Created vocabulary with 212 symbols.
Constructing 7-gram LM ...
Created trie with 21502513 nodes.
Running over test.txt ...
Results: numSymbols = 69302, ppl = 6.047012997396163, entropy = 2.5962226799087356 bits/char, OOVs = 0 (0%).
```
