# GoEmotions

GoEmotions contains 58k carefully curated Reddit comments labeled for 27 emotion categories or Neutral.

This directory includes the data and code for data analysis scripts. We also include code for our baseline model, which involves fine-tuning a pre-trained [BERT-base model](https://github.com/google-research/bert).
For more details on the design and content of the dataset, please see our paper. # todo: add link

## Requirements

See `requirements.txt`

## Setup

Download the pre-trained BERT model from
[here](https://github.com/google-research/bert) and unzip them inside the
`pretrained_bert` directory. In the paper, we use the cased base model.

## Data

We include our data in the `data` folder.  # todo: include data

Our raw dataset, under `full_dataset.tsv`, includes a total of 58,009 examples. Each example is annotated by 3 or 5 annotators (we recruited two additional annotators when there is no agreement between the first three).

Our training data is



## Analyzing Data

See each script for more documentation and descriptive command line flags.

*   `python3 -m goemotions.analyze_data.py`: get high-level statistics of the
    data and correlation among emotion ratings.
*   `python3 -m goemotions.extract_words.py`: get the words that are significantly
    associated with each emotion, in contrast to the other emotions, based on
    their log odds ratio.
*   `python3 -m goemotions.ppca.py`: run PPCA
    (Cowen et al., 2019)[https://www.nature.com/articles/s41562-019-0533-6] on
    the data and generate plots.

## Training and evaluating models

Run `python -m goemotions.bert_classifier.py` to perform fine-tuning on top of
BERT, with added regularization. See the script and the paper for detailed
description of the flags and parameters.

## Citation

If you use this code for your publication, please cite the original paper:

```
@inproceedings{demszky2020goemotions,
 author = {Demszky, Dorottya and Movshovitz-Attias, Dana and Ko, Jeongwoo and Cowen, Alan and Nemade, Gaurav and Ravi, Sujith},
 booktitle = {58th Annual Meeting of the Association for Computational Linguistics (ACL)},
 title = {{GoEmotions: A Dataset of Fine-Grained Emotions}},
 year = {2020}
}
```

## Contact

[Dora Demszky](https://nlp.stanford.edu/~ddemszky/index.html)
