# GoEmotions

GoEmotions contains 58k carefully curated Reddit comments labeled for 27 emotion categories or Neutral. The emotion categories are _admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise_.

This directory includes the data and code for data analysis scripts. We also include code for our baseline model, which involves fine-tuning a pre-trained [BERT-base model](https://github.com/google-research/bert).
For more details on the design and content of the dataset, please see our paper. # todo: add link

## Requirements

See `requirements.txt`

## Setup

Download the pre-trained BERT model from
[here](https://github.com/google-research/bert) and unzip them inside the
`bert` directory. In the paper, we use the cased base model.

## Data

We include our data in the `data` folder. \

Our raw dataset, under `goemotions.tsv` includes all annotations as well as metadata on the comments. Each row represents a single rater's annotation for a single example. This file includes the following columns:

* `text`: The text of the comment (with masked tokens, as described in the paper).
* `id`: The unique id of the comment.
* `author`: The Reddit username of the comment's author.
* `subreddit`: The subreddit that the comment belongs to.
* `link_id`: The link id of the comment.
* `parent_id`: The parent id of the comment.
* `created_utc`: The timestamp of the comment.
* `rater_id`: The unique id of the annotator.
* `example_very_unclear`: Whether the annotator marked the example as being very unclear or difficult to label (in this case they did not choose any emotion labels).
* + separate columns representing each of the emotion categories, with binary labels (0 or 1)

The data we used for training the models includes examples where there is agreement between at least 2 raters. Our data includes 43,410 training examples (`train.tsv`), 5426 dev examples (`dev.tsv`) and 5427 test examples (`test.tsv`). These files have _no header row_ and have the following columns:

1. text
2. comma-separated list of emotion ids (the ids are indexed based on the order of emotions in `emotions.txt`)
3. id of the comment


### Visualization

[Here](https://nlp.stanford.edu/~ddemszky/goemotions/tsne.html) you can view a TSNE projection showing a random sample of the data. The plot is generated using PPCA (see scripts below). Each point in the plot represents a single example and the text and the labels are shown on mouse-hover. The color of each point is the weighted average of the RGB values of the those emotions.


## Analyzing Data

See each script for more documentation and descriptive command line flags.

*   `python3 -m goemotions.analyze_data`: get high-level statistics of the
    data and correlation among emotion ratings.
*   `python3 -m goemotions.extract_words`: get the words that are significantly
    associated with each emotion, in contrast to the other emotions, based on
    their log odds ratio.
*   `python3 -m goemotions.ppca`: run PPCA
    [(Cowen et al., 2019)](https://www.nature.com/articles/s41562-019-0533-6) on
    the data and generate plots.

## Training and evaluating models

Run `python -m goemotions.bert_classifier` to perform fine-tuning on top of
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
