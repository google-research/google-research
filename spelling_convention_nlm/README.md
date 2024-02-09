# Supporting Materials for "Spelling Convention Sensitivity in NLMs"

This document provides a brief summary of materials released here in support of
the paper "Spelling convention sensitivity in neural language models", which
appears in the [Findings of the EACL, 2023](https://aclanthology.org/2023.findings-eacl.98/). The materials are in the process
of being assembled, and will be released at this URL as they become
available. All released materials will be summarized in this document.




## Data

The data subdirectory currently contains the following resources, used for
results reported in the paper. Due to size constraints at this Github location, 
larger datasets are compressed into tarballs (`.tgz`) and are split into 1M 
chunks (suffixed by `.part.X`) using the Unix `split` command. They can be 
concatenated via `cat` to reform a single file, and expanded as normal 
using `tar -xvzf`.

In `data/spelling_pairs/american_british_spelling_alternatives.tsv`, each line
contains a tab-delimited pair of words found in the [American/British English
Translator](https://github.com/hyperreality/American-British-English-Translator).
These are the 1706 word pairs mentioned in the paper, with American spelling in
the first column and British spelling in the second column. The file
`data/spelling_pairs/american_british_spelling_alternatives_used_for_prompts.tsv`
presents the 1266 pairs that were used to construct prompts.

The `data/corpora_list_pair_counts` subdirectory includes gzipped TSV files for
each of the 5 investigated corpora. Each line includes two words and a count;
and each word appears in the above-mentioned list. For example, the first line
in `BNC.all_pairs_and_counts.tsv.gz` is:

```
acclimatisation	practice	1
```

This signifies that, in the BNC corpus, the words `acclimatisation` and
`practice` cooccurred in a sentence just once.  In addition to files containing
all cooccurrences, for the C4 and OWT corpora we also include gzipped TSV files
with counts of adjacent occurrences, which are used to report results in Tables
2 and 7 of the paper.

The `data/prompts` subdirectory contains a tarball of the filled-in templates
that were scored in the paper. There are 29 prompt templates, each presented
with 16028 prompt-target word pairs, for a total of 464812 lines. See the
scripts described below to use this data to generate scores with T5 and GPT2.

The `data/score_pickles` subdirectory contains the computed raw scores summarized in the paper. Post-processing of these raw scores is shown in `scripts/normalize_scores.ipynb`.

*  GPT2
  *   `gpt2_{nonadjacent|adjacent}_scores.pickle`: Conditional and joint GPT2 scores in the
    adjacent and non-adjacent prompt-target condition.
* T5
  * `t5_{finetuned}_{conditional|joint}_{adjacent|nonadjacent}.pickle`: Conditional and joint T5 scores in the adjacent and non-adjacent prompt-target conditions.
  * `t5_conditional_adjacent_nonce.pickle`: The scores for nonce forms corresponding to Section 5.5 in the paper.

## Scripts

We provide scripts for running the scoring experiments described in the paper.
The instructions here were tested on a linux machine with 64gb ram and 10 CPU
cores. Code is provided as-is and we cannot guarantee support in your
environment. Modify as needed.

### Scoring using GPT2

You will need to
[install](https://huggingface.co/docs/transformers/installation) HuggingFace
Transformers library for PyTorch.

Scoring can then be performed with `get_gpt2_scores.py`. See comments in the
file for description of parameters that need to be set (`PROMPT_PATH`,
`ADJACENCY`, and `OUTPUT_PATH`)

### Scoring using T5X

You will need to install the [t5x](https://github.com/google-research/t5x)
package.

If using Anaconda Python:

```shell
# Create a virtual environment. As of 3/22/23, the t5x package requires Python 3.10.
conda create -n t5x python=3.10
conda activate t5x
# Clone the repository.
git clone https://github.com/google-research/t5x
# Install.
cd t5x
pip install -e .
# If the sentencepiece package fails to compile, use the conda version.
conda install sentencepiece
# You can add the scripts directory from this package to your PYTHONPATH to allow tasks.py to be imported.
export PYTHONPATH=PYTHONPATH:/scripts
```

You can grab released checkpoints and vocabulary files using
[gsutil](https://cloud.google.com/storage/docs/gsutil_install).

```shell
mkdir data
gsutil cp -r gs://t5-data/pretrained_models/t5x/mt5_large/checkpoint_1000000 ./data
gsutil cp -r gs://t5-data/vocabs/mc4.250000.100extra ./data
```

If you have the necessary packages and data, you can use `get_t5x_scores.sh` to
get scores. See comments in the file for parameters that need to be set.

### Score Postprocessing

The raw scores (NLL values) produced for GPT2 and T5X by the scripts above are
normalized into the values seen in the paper. See the `normalize_scores.ipynb`
colab notebook for details.
