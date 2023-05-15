This repository includes scripts for the following paper:

["Dealing with disagreements: Looking beyond the majority vote in subjective annotations". TACL 2022](http://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl_a_00449/1986597/tacl_a_00449.pdf)

The method includes a multi-annotator approach for implementing hate speech detection in textual data.
This approach relies on every annotators' labels rather than final ground-truth labels. The training data is required to include annotators' label for each item and the predictions will similarly include several labels for each input text.

## Prerequisites
`requirements.txt` includes the list of Python package requirements.

## Data format
The training data needs to have the following format:
If there are `N` number of annotators and `X` items, the dataset should have `X` rows, and at least `N` + 1 columns. `N` columns (named 0 to `N` - 1) represent the annotators, and one column (named ``text``) represents the textual item.
Each annotator column represents annotations from a single annotator. The values in that column can be 0, 1, or numpy.nan (current code only works for binary classification but can be modified to work for multi-class tasks).

Following is an example of how the data should look for a dataset with 3 items each annotated by 2 annotators from a pool of 4 annotators.

| 0 | 1 | 2 | 3 | text                  |
|---|---|---|---|-----------------------|
| 0 | 1 |   |   | <social media post 1> |
|   | 1 |   | 0 | <social media post 2> |
| 1 |   | 1 |   | <social media post 3> |

## Data preprocessing
The `preprocess.py` includes scripts for reformating two specific datasets studied in the paper.
To replicate the experiments for one of these datasets, follow the preprocessing steps before moving on to the training section.
After running each preprocessing script, a `data` folder will be created in your current directory. Make sure to
run the `preprocess.py` script from the `multi_annotator` directory to make next steps easier.

### GoEmotions

Run the script:
`python3 preprocess.py --corpus emotions`

### GHC (Gab Hate Corpus)

1- Download the GHC dataset from ["this OSF storage"](https://osf.io/wuecz).
2- Run the script:

`python3 preprocess.py --corpus GHC --data_path <path to GabHateCorpus_annotations.tsv>`

## Running the training

To run the code for training and evaluation you should first have the preprocessed
data in the `data/<corpus>/` folder. The data should be named `<label>_multi.csv`
Running the following script initiates the training:

`python3 run_annotators_modeling.py --corpus <corpus> --model <multi_task or single> --label <label> --bert_path <path to bert-base-cased/ folder>`

### Parameters
The parameters you can define for training:

- `--corpus` you can choose the corpus from the list of [`GHC`, `emotions`]
- `--model` you can select either `single` or `multi_task`
- `--label` the name of the task in each dataset. The lable options for different datasets are as follows:
    - `GHC`: [`hate`]
    - `emotions`: [`joy`, `sadness`, `fear`, `anger`, `sadness`,
      `surprise`]
- `--bert_path` download the Hugging face bert-base-cased and pass the location. By default, the code looks for the bert-base-cased folder
in current directory. Make sure to include four main files (`vocab.txt`, `tokenizer.json`, `config.json`, and `tf_model.h5`) in this folder. You can download these file from the ["HuggingFace models"](https://huggingface.co/bert-base-cased/tree/main).



The following scripts are examples for replicating multi-task
experiments for GHC and GoEmotions dataset:

### GoEmotions

`python3 run_annotators_modeling.py --corpus emotions --model multi_task --label joy`
You can replace `joy` with each of the other 5 emotions.


### GHC

`python3 run_annotators_modeling.py --corpus GHC --model multi_task --label hate`
