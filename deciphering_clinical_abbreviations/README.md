# Deciphering Clinical Abbreviations

This repository contains data processing and evaluation code for the following
paper:

> Rajkomar, A., Loreaux, E., Liu, Y. *et al.* Deciphering clinical abbreviations with a privacy protecting machine learning system. *Nat Commun* **13**, 7456 (2022). https://doi.org/10.1038/s41467-022-35007-9

**Prerequisites:**

* To run the code, `conda` distribution must be installed on your computer. The
code in this repository was tested with `conda 4.9.1`.
* To access the released datasets, `gsutil` must be installed on your computer. Follow the installation instructions at [https://cloud.google.com/storage/docs/gsutil_install](https://cloud.google.com/storage/docs/gsutil_install).

**Setup**

1.  Clone the repository to a local directory and cd into the directory directly above it.
2.  Allow conda to be activated from a shell script by running `$ conda init bash`, swapping bash for the relevant shell name if necessary (e.g. fish, tcsh, etc.). You may need to close and restart your shell after running this.
3.  To download the data required to carry out reverse substitution and model evaluation, run `$ sh deciphering_clinical_abbreviations/download_datasets.sh`. This will download the following files to the `deciphering_clinical_abbreviations/datasets` folder:
  * `abbreviation_expansion_dictionary.csv`: The multi-source, manually curated abbreviation expansion dictionary from the paper. This dictionary is used for both reverse substitution and model evaluation.
  * `synthetic_snippets.csv`: 302 text snippets containing abbreviations written by clinicians, which is used as one of the external test sets in the paper.
  * `t5_11b_elic_outputs_synthetic_snippets.csv`: The corresponding outputs for those 302 text snippets from the t5 11B model, as reported in the paper.
  * `expansion_equivalencies.csv`: A collection of clinically equivalent expansions labeled as such by clinicians as a part of our error analysis, which is used during evaluation to determine which model expansions are correct.

## Code Overview

The code supports two main functions: the generation of reverse-substituted evaluation datasets derived from clinical notes datasets, and the evaluation of model outputs.

### Generating Reverse-substituted Evaluation Datasets

#### Libraries

The code to generate reverse-substituted evaluation datasets is organized into 2 libraries:

* `text_processing.py`: functions for breaking notes into snippets, finding words in each snippet for reverse substitution, carrying out reverse substitution, and generating labels for each snippet that can be used in downstream evaluation.
* `text_sampling.py`: logic for sampling snippets from a larger snippet dataset in a way that supports proportionality across all the substituted abbreviation-expansion pairs.

#### How to Run

The binary `run_reverse_substitution.py` processes and down-samples a collection of documents into a collection of snippets in which expansions have been substituted for their abbreviations and labeled as such. This binary takes seven arguments:

* `document_dataset_path`: The path to the document dataset csv, including a column called 'document_text' which contains the text of each document.
* `abbreviation_dictionary_path`: The path to the abbreviation dictionary csv, including the following columns:
  * abbreviation - the abbreviation string
  * expansion - the associated expansion string
* `save_filepath`: The file path to save the output files to.
* `expected_replacements_per_expansion`: The number of expected replacements per expansion. This is used to calculate a unique replacement rate for each expansion, where `rate_for_exp = min(expected_replacements_per_expansion / n_total_instances_of_exp, 1)`.
* `min_snippets_per_substitution_pair`: The minimum number of snippets containing a substitution pair that must be sampled for the final dataset. If a snippet contains any pair for which the number of previous snippets sampled containing that pair is less than this number, the snippet is sampled.
* `exclusion_strings`: Strings whose presence in a snippet should lead to the exclusion of that snippet from the dataset.
* `random_seed`: An optional random seed for determinism. Default value is 1. Setting this flag to -1 will result in non-determinism.

The binary outputs 2 files to the `save_filepath` directory:

* `dataset.csv`: the dataset storing the reverse substituted snippets
* `substitution_counts.csv`: contains the counts of abbreviation-expansion substitutions contained in the dataset

This binary was used to generate the two reverse-substituted evaluation datasets from the paper: MIMIC-III and i2b2-2014. If the path to either of these notes datasets in csv form is provided as the `document_dataset_path` argument, such that the column named "document_text" contains the note text, the datasets can be exactly replicated using the following arguments for each dataset:

* MIMIC-III:
  * `expected_replacements_per_expansion=10000`
  * `min_snippets_per_substitution_pair=2`
  * `exclusion_strings="[,]"`
  * `random_seed=1`
* i2b2-2014:
  * `expected_replacements_per_expansion=50`
  * `min_snippets_per_substitution_pair=2`
  * `exclusion_strings=""`
  * `random_seed=1`

A fake example of a reverse-substituted dataset can be generated by running `$ sh deciphering_clinical_abbreviations/run_reverse_substitution.sh`, which uses the dummy dataset provided at `deciphering_clinical_abbreviations/fake_document_dataset.csv`. Note: You may need to ensure the initialization files are loaded for the shell by including an interactive flag (`-i` for bash). Most shell sessions should load these files automatically.


### Model Evaluation

#### Libraries

The model evaluation code is organized into 4 libraries:

* `tokenizer.py`: a custom tokenizer built for evaluating clinical abbreviation expansion. Specifically, this tokenizer ensures that abbreviations are kept as single tokens, and that expansions which contain their abbreviation within them (e.g. k --> vitamin k) are kept as single tokens.
* `text_alignment.py`: a custom implementation of the Needleman-Wunsch alignment algorithm, which is a dynamic programming method for optimally aligning sequences. This algorithm has been modified to include some custom scoring rules specific to aligning abbreviations with their expansions.
* `expansion_attribution.py`: logic for taking two aligned sequences of tokens and determining how to break them up into abbreviation-expansion pairs. This method involves a few heuristics, including searching for a valid expansion match and leveraging the Needleman-Wunsch alignment itself.
* `evaluation.py`: code required to compute metrics for model outputs, including the creation of specific data structures from their paths and the manipulation of dataframes.

#### How to Run

The binary `run_model_evaluation.py`, takes four paths as arguments:

* "abbreviation_dictionary_path": A path to a csv representing the dictionary of all known abbreviation-expansion pairs. This csv must have 2 columns, "abbreviation" and "expansion", with a single row for each unique abbreviation-expansion pair.
* "input_data_path": A path to a csv representing the input data we are running the model on. This csv must have the following 3 columns:
  * "input_id": a unique string id for each raw input
  * "raw_input": the raw input string
  * "label": the abbreviation expansions label. This string follows the following label format: each abbreviation-expansion pair is separated by a comma, and within a pair, the abbreviation and expansion are separated by a space. For instance, if the original snippet is "the pt was taken to the er" and the intended expansions are "the patient was taken to the emergency room," then the label would be "pt patient, er emergency room". This implies that abbreviations that contain spaces are not currently supported; nor are expansion phrases containing commas. For abbreviations which exist in multiple places and have different expansions, every expansion must be listed alongside the duplicated abbreviation in the order they appear (e.g. "pt patient, pt physical therapy, pt patient"). If an abbreviation exists in multiple places but has the same expansion, a single entry will suffice.
* "model_outputs_path": The path to the model outputs for each raw input string. The 2 required columns within this csv are "input_id," which should act as a primary key linking model outputs to their corresponding inputs in the input data csv, and a "model_output" column containing the model output, which should be in the format of a fully expanded string.
* "expansion_equivalences_path": The path to a csv containing all known groups of expansions which should be considered clinically equivalent. The 2 required columns are as follows:
  * "abbreviation": the abbreviation
  * "equivalent_expansions": a pipe-separated list of all expansions to be considered clinically equivalent.

Run `$ sh deciphering_clinical_abbreviations/run_model_evaluation.sh` to run model eval with the synthetic snippets and corresponding model outputs downloaded with gsutil. Note: You may need to ensure the initialization files are loaded for the shell by including an interactive flag (-i for bash). Most shell sessions should load these files automatically.

## Generating C4-WSRS

We also make code available to generate C4-WSRS, a dataset which is the product of applying web-scale reverse substitution (WSRS) to the Colossal Clean Crawled Corpus (C4). This code is available through Tensorflow Datasets at [https://www.tensorflow.org/datasets/catalog/c4_wsrs](https://www.tensorflow.org/datasets/catalog/c4_wsrs). In order to generate this dataset, you must first generate C4 itself. Instructions for generating C4 can be found [here](https://github.com/google-research/text-to-text-transfer-transformer#c4). Similar to C4, we also recommend taking advantage of the Apache Beam support in TFDS, which enables distributed preprocessing of the dataset and can be run on Google Cloud Dataflow. Once you have generated C4, set the environment variable `TFDS_DATA_DIR='gs://$MY_BUCKET/tensorflow_datasets'` and run the following commands:

```
pip install tfds-nightly[c4_wsrs]
echo 'tfds-nightly[c4_wsrs]' > /tmp/beam_requirements.txt
python -m tensorflow_datasets.scripts.download_and_prepare \
  --datasets=c4_wsrs/default \
  --data_dir=$TFDS_DATA_DIR \
  --beam_pipeline_options="project=$MY_PROJECT,job_name=c4_wsrs,staging_location=gs://$MY_BUCKET/binaries,temp_location=gs://$MY_BUCKET/temp,runner=DataflowRunner,requirements_file=/tmp/beam_requirements.txt,experiments=shuffle_mode=service,region=$MY_REGION"
```


