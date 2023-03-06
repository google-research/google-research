# Coreference Resolution through a seq2seq Transition-Based System


This repository contains a reference mT5 model of the paper 'Coreference Resolution through a seq2seq Transition-Based System'. 


Most recent coreference resolution systems use search algorithms over possible spans to identify mentions and resolve coreference. We instead present a coreference resolution system that uses a text-to-text (seq2seq) paradigm to predict mentions and links jointly, which simplifies the coreference resolution by eliminating both the search for mentions and coreferences. We implemented the coreference system as a transition system and use multilingual T5 as language model. We obtained state-of-the-art accuracy with 83.3 F1-score on the CoNLL-2012 data set. 


@article{
title	= {Coreference Resolution through a seq2seq Transition-Based System},
author	= {Bernd Bohnet and Michael Collins and Chris Alberti},
journal	= {TACL}
}


## Model

The top performing mT5-model described in the paper can be downloaded following the link below.
[Coref-mT5-XXL model](https://console.cloud.google.com/storage/browser/gresearch/correference_seq2seq/checkpoint_1140000).


## Decoder

We provide a Colab notebook that runs the seq2seq transition-based model. The notebook contains code to resolve coreferences in documents. The user needs to start a mT5 server and load the provided checkpoint. The interface to the server needs to be implemented depending on the used infrastructure. It is possible to emulate the decoding of a document without running a server, which might be useful to understand the procedure in detail.
