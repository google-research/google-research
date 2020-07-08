# Code for "Toward Better Storylines with Sentence-Level Language Models"
This code reproduces the experiments on ROC Stories in the ACL paper
[Toward Better Storylines with Sentence-Level Language Models](https://www.aclweb.org/anthology/2020.acl-main.666/).
It contains
scripts to download the checkpoints used to reproduce the accuracy numbers
in Tables 1 and 2 as well as Figure 1 of the paper.

## Setup
The training and eval code is Python 3 and uses Tensorflow 2.
Install all needed dependencies into a virtual environment with:

```
python3 -m venv pyenv_tf2
source pyenv_tf2/bin/activate
pip install --upgrade pip
pip3 install -R requirements.train.txt
```

## Building or Downloading the Dataset
The dataset consists of the mean BERT embedding for each sentence in the ROC
Stories dataset. These are stored as a
[TFDS](https://www.tensorflow.org/datasets/api_docs/python/tfds) dataset.

Since computing embeddings for ~400k sentences can be slow,
we have made the mean BERT embedding dataset used in the paper available for
download.

To download the dataset run the following commands from the base directory of
this repository:

```
wget https://storage.googleapis.com/gresearch/better_storylines/roc_stories_embeddings.zip
mkdir tfds_datasets
unzip roc_stories_embeddings.zip -d tfds_datasets
rm roc_stories_embeddings.zip
```

Optionally, to generate the dataset from scratch run the following command from
the base directory of this repository. If you are running locally
(without Apache Beam) it could take a long time:

```
python3 -m venv pyenv_tf1
source pyenv_tf1/bin/activate
pip install --upgrade pip
pip install -R requirements.datagen.txt
# The following line is needed only if you'd like to do frequency-weighted embs.
!wget https://storage.googleapis.com/gresearch/better_storylines/vocab_frequencies
sh scripts/build_tfds_dataset.sh
```

## Available Checkpoints
The following pre-trained checkpoints are available for download.
| Checkpoint Name               | Link |                        Description                                           |
|-------------------------------|:----:|:-----------------------------------------------------------------------------|
| `mlp_best_largescale_cl`      | [link](https://storage.googleapis.com/gresearch/better_storylines/mlp_best_largescale_cl.zip) | Best MLP checkpoint for largescale ranking task (CSLoss).            |
| `mlp_best_largescale_nocl`    | [link](https://storage.googleapis.com/gresearch/better_storylines/mlp_best_largescale_nocl.zip) | Best MLP checkpoint for largescale ranking task (no CSLoss).         |
| `mlp_best_story_cloze_cl`     | [link](https://storage.googleapis.com/gresearch/better_storylines/mlp_best_story_cloze_cl.zip) | Best MLP checkpoint for Story Cloze task. (CSLoss)                   |
| `mlp_best_story_cloze_nocl`   | [link](https://storage.googleapis.com/gresearch/better_storylines/mlp_best_story_cloze_nocl.zip) | Best MLP checkpoint for Story Cloze task. (no CSLoss)                |
| `resmlp_best_largescale_cl`   | [link](https://storage.googleapis.com/gresearch/better_storylines/resmlp_best_largescale_cl.zip) | Best residual MLP checkpoint for largescale ranking task. (CSLoss)   |
| `resmlp_best_largescale_nocl` | [link](https://storage.googleapis.com/gresearch/better_storylines/resmlp_best_largescale_nocl.zip) | Best residual MLP checkpoint for largescale ranking task. (no CSLoss)|
| `resmlp_best_story_cloze_cl`  | [link](https://storage.googleapis.com/gresearch/better_storylines/resmlp_best_story_cloze_cl.zip) | Best residual MLP checkpoint for Story Cloze ranking task. (CSLoss)  |
| `resmlp_best_story_cloze_nocl`| [link](https://storage.googleapis.com/gresearch/better_storylines/resmlp_best_story_cloze_nocl.zip) | Best residual MLP checkpoint for Story Cloze task. (no CSLoss)       |

## Running Evaluation

### Evaluation on Story Cloze task
First download a checkpoint to evaluate:

```
wget https://storage.googleapis.com/gresearch/better_storylines/mlp_best_largescale_cl.zip
mkdir trained_models
unzip mlp_best_largescale_cl.zip -d trained_models
rm mlp_best_largescale_cl.zip
```

The following script evaluates all checkpoints in the provided directory.
Validation accuracy for each checkpoint is outputted into a CSV in the file
`all_metrics.csv`. You should run this script before running any of the other
eval scripts since `all_metrics.csv` is used by the other script to select a
checkpoint.

```
sh scripts/evaluate_all_checkpoints.sh trained_models/mlp_best_largescale_cl
```

The following script outputs the accuracy of the best checkpoint in the provided
directory on each Story Cloze 2016 test set. (The 2018 test set can only be
evaluated on through submissions to the [CodaLab
leaderboard](https://competitions.codalab.org/competitions/15333).

```
sh scripts/evaluate_best_story_cloze_test.sh trained_models/mlp_best_largescale_cl
```

### Evaluation on large-scale reranking task
The following script outputs the accuracy and MRR of the best checkpoint in
the provided directory on the largescale reranking task.

```
sh scripts/evaluate_ranking_task.sh trained_models/mlp_best_largescale_cl
```

### Qualiative evaluation of large-scale reranking
To do qualitative eval you will first need to download the CSVs for the
validation and train sets, which can be requested from the [ROC Stories
website](https://www.cs.rochester.edu/nlp/rocstories/).
The following script outputs the highest-scoring next sentences on the
largescale reranking task.

```
sh scripts/evaluate_ranking_qualitative.sh path/to/rocstories/csvs trained_models/mlp_best_largescale_cl
```

## Training from scratch
The following script launches training for the residual model.

```
sh scripts/train_residual.sh
```

# Paper Citation

```
@inproceedings{ippolito2020toward,
  title={Toward Better Storylines with Sentence-Level Language Models},
  author={Ippolito, Daphne and Grangier, David and Eck, Douglas and Callison-Burch, Chris},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  year={2020}
}
```
