# Dataset analysis

Internal code for dataset analysis: under construction.

## Requirements

See `requirements.txt`

## Setup

Download the pre-trained BERT model from
[here](https://github.com/google-research/bert) and unzip them inside the
`pretrained_bert` directory. In the paper, we use the cased base model.

## Data

To be added.

## Analyzing Data

See each script for more documentation and descriptive command line flags.

*   `python3 -m dataset_analysis.analyze_data.py`: get high-level statistics of the
    data and correlation among targets.
*   `python3 -m dataset_analysis.extract_words.py`: get the words that are significantly
    associated with each target, in contrast to the other targets, based on
    their log odds ratio.
*   `python3 -m dataset_analysis.ppca.py`: run PPCA
    (Cowen et al., 2019)[https://www.nature.com/articles/s41562-019-0533-6] on
    the data and generate plots.

## Training and evaluating models

Run `python -m dataset_analysis.bert_classifier.py` to perform fine-tuning on top of
BERT, with added regularization. See the script and the paper for detailed
description of the flags and parameters.

## Citation

If you use this code for your publication, please cite the original paper:

`TODO: add citation`

## Contact

[Dora Demszky](https://nlp.stanford.edu/~ddemszky/index.html)
