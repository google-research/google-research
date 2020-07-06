# BERTSeq2Seq

This repository contains the code to query our best models (served as TensorFlow
Hub models) and their predictions on various academic text-generation
benchmarks from our paper
"[Leveraging Pre-trained Checkpoints for Sequence Generation Tasks](https://www.mitpressjournals.org/doi/pdf/10.1162/tacl_a_00313)"
at TACL 2020.

Please cite our paper if you use our data or models.

```
@article{rothe_tacl20,
  author = {Rothe, Sascha and Narayan, Shashi and Severyn, Aliaksei},
  title = {Leveraging Pre-trained Checkpoints for Sequence Generation Tasks},
  journal = {Transactions of the Association for Computational Linguistics},
  volume = {8},
  number = {},
  pages = {264-280},
  year = {2020}
}
```

## Introduction

Unsupervised pre-training of large neural models has recently revolutionized
Natural Language Processing. We developed a Transformer-based
sequence-to-sequence model that is compatible with publicly available
pre-trained BERT, GPT-2 and RoBERTa checkpoints and achieved new
state-of-the-art results on Machine Translation, Text Summarization, Sentence
Splitting, and Sentence Fusion. We believe that NLP researchers will find our
dataset with model predictions as a valuable resource to compare pre-trained
text generation models and to derive actionable insights.

## Predictions

The dataset consists of our sequence-to-sequence model predictions on academic
datasets for text generation: Sentence Fusion (DiscoFuse), Sentence Splitting
(WikiSplit), Summarization (XSum, CNN/DailyMail and Gigaword) and Machine
Translation (WMT 2014 and 2016). Our dataset will be a valuable resource to
compare pre-trained text generation models.

The dataset consists of json files with lists of dictionaries
```
{
  “target”: <string>,
  “prediction”: <string>
}
```

Here, “prediction” is the model generated text and “target” is the reference
text.

* MT(DE ->EN): [WMT 2014](https://storage.googleapis.com/berts2s-predictions-tacl20/de_en_newstest2014_predictions.json) and [WMT 2016](https://storage.googleapis.com/berts2s-predictions-tacl20/de_en_newstest2016_predictions.json)
* MT(EN->DE): [WMT 2014](https://storage.googleapis.com/berts2s-predictions-tacl20/en_de_newstest2014_predictions.json) and [WMT 2016](https://storage.googleapis.com/berts2s-predictions-tacl20/en_de_newstest2016_predictions.json)
* Sentence Fusion: [DiscoFuse](https://storage.googleapis.com/berts2s-predictions-tacl20/discofuse_predictions.json)
* Sentence Splitting: [WikiSplit](https://storage.googleapis.com/berts2s-predictions-tacl20/wikisplit_predictions.json)
* Summarization: [Gigaword](https://storage.googleapis.com/berts2s-predictions-tacl20/gigaword_predictions.json), [CNN/DailyMail](https://storage.googleapis.com/berts2s-predictions-tacl20/cnndm_predictions.json) and [XSum](https://storage.googleapis.com/berts2s-predictions-tacl20/bbc_xsum_predictions.json)

## TFHub Modules

Here is the code to query our best models served as TensorFlow Hub models.

```
# TF1 version
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
```

### MT(DE ->EN)

```
text_generator = hub.Module(
    'https://tfhub.dev/google/bertseq2seq/bert24_de_en/1')
de_sents = ['Satz 1', 'Satz 2']
en_sents = text_generator(en_sents)
```

### MT(EN->DE)

```
text_generator = hub.Module(
    'https://tfhub.dev/google/bertseq2seq/bert24_en_de/1')
en_sents = ['Sentence 1', 'Sentence 2']
de_sents = text_generator(en_sents)
```

### Sentence Fusion

```
text_generator = hub.Module(
    'https://tfhub.dev/google/bertseq2seq/roberta24_discofuse/1')
input_texts = ['Sentence 1a Sentence 1b',
               'Sentence 2a Sentence 2b Sentence 2c']
output_sents = text_generator(input_texts)
```

### Sentence Splitting

```
text_generator = hub.Module(
    'https://tfhub.dev/google/bertseq2seq/roberta24_wikisplit/1')
input_sentences = ['Long Sentence 1', 'Long Sentence 2']
output_texts = text_generator(input_sentences)
```

### Summarization(Title Generation)

```
text_generator = hub.Module(
    'https://tfhub.dev/google/bertseq2seq/roberta24_gigaword/1')
input_sents = ['This is the first sentence.', 'This is the second sentence.']
output_summaries = text_generator(input_sents)
```

### Summarization (Highlight Generation)

```
text_generator = hub.Module(
    'https://tfhub.dev/google/bertseq2seq/roberta24_cnndm/1')
input_documents = ['This is text from the first document.',
                   'This is text from the second document.']
output_summaries = text_generator(input_documents)
```

### Extreme Summarization

```
text_generator = hub.Module(
    'https://tfhub.dev/google/bertseq2seq/roberta24_bbc/1')
input_documents = ['This is text from the first document.',
                   'This is text from the second document.']
output_summaries = text_generator(input_documents)
```

## Contact us

If you have a technical question regarding the dataset or publication, please
create an issue in this repository. This is the fastest way to reach us.

If you would like to share feedback or report concerns, please email us at
berts2s@google.com.


