# Benchmark for extractions tasks on visually rich documents.

## Data and Tasks
Our paper *A Benchmark for Structured Extractions from Complex Documents* can be
found at https://arxiv.org/abs/2211.15421.

The dataset consists of 2 corpora VRDU-Registration Forms and VRDU-Ad-buy Forms.
VRDU-Registration Forms consist of public documents downloaded from the
[US Department of Justice](https://www.justice.gov/nsd-fara). VRDU-Ad-buy Forms
consist of public documents from [FCC PublicFiles](https://publicfiles.fcc.gov/).
VRDU-Registration Form is the simpler of the two -- containing fewer fields,
only three distinct templates, and only simple fields. VRDU-Ad-buy Forms on the
other hand consist of more than a dozen fields, dozens of templates (distinct
layouts), and more complex fields (nested, repeated fields).

For each corpus, we provide:

*   *main/pdfs*: a directory with the raw PDFs (for convenience, you can also
    download them from the original source websites).
*   *main/dataset.jsonl*: the OCR output corresponding to each PDF, and
    structured annotations for each document obtained by asking human annotators
    to draw a bounding box around each specified field of interest.
*   *main/meta.json*: mapping from the entity names in each corpus to a
    type-specific match function (eg. DateMatch, NumericalMatch, PriceMatch,
    etc.) used to compare predictions with the ground truth.
*   *few_shot-splits/* : train/validation/test splits for various tasks provided
    through JSON files containing the filenames that should go into each bucket.

*dataset.jsonl* contains the following attributes:
* filename (name of the file in the subdirectory for which this object contains
  other data).
* ocr (output from the OCR tool that provides text detected on each page along
  with the coordinates on the page)
* annotations (list of entity names along with the text value extracted from a
  span in the document and a bounding box corresponding to the given entity,
  e.g.: "annotations": [["registration_num", [["3712\n", [0, 0.46376812,
  0.32893434, 0.5, 0.3447707], [[2380, 2385]]]]]


### Tasks
There are three kinds of tasks -- Single Template Learning (STL), Mixed Template
Learning (MTL), and Unseen Template Learning (UTL), indicated by "lv1", "lv2",
or "lv3" in the name of the split file provided in few_shot-splits:

* STL: train and test documents contain documents belonging to the same (single)
template. This task is particularly useful to understand if we need different
approaches to deal with mostly-fixed-layout documents vs. documents that vary
substantially in their layouts.

* MTL: train and test documents are drawn from the same set of templates. This
task is useful to understand if an approach can work across multiple layouts
to present the same information.

* UTL: train and test documents are drawn from disjoint sets of templates. This
task helps us understand if an approach can truly generalize to unseen
layouts and  templates. We expect this to be the hardest task, since new
templates may look substantially different from templates previously seen.

Each split file contains three list-valued fields: train/valid/split, each with
a list of filenames present in the filename attribute in dataset.jsonl. Split
files provide examples with 10, 50, 100, and 200 training instances each.
The splits are present to mitigate the large variance that may result from
sampling different training docs for the few-shot setting.


## Evaluation Tools
The ```python -m``` command assumes you are in the `google-research/` directory.

Sample invocation of the evaluation binary (on one dataset):

```bash
python -m vrdu.evaluate \
--base_dirpath='/path/to/vrdu/fara/' \
--extraction_path='/path/to/results/fara-modelFoo/' \
--eval_output_path='/path/to/results/fara-modelFoo-results.csv'
```

Note that `extraction_path` contains model outputs of JSON format. Each JSON
file corresponds to a task (split), meaning the file name starts with the split
name and end with `-test_predictions.json`.
