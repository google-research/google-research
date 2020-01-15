# WikiSplit BLEU evaluation script

This directory contains the evaluation code for the paper
[Learning To Split and Rephrase From Wikipedia Edit History](https://aclweb.org/anthology/D18-1080).

For the WikiSplit data set release, see
https://github.com/google-research-datasets/wiki-split.

The code is implemented in Python and the required packages are listed in the
file `requirements.txt`. To install the dependencies within an active Python 3
virtual environment, do: `pip3 install -r requirements.txt`

Note: the functionality has been verified for the following version combination
only; no guarantees or support can be provided for other versions.

* Python 3.6.5
* nltk 3.4.1
* absl-py 0.7.1
* numpy 1.16.3

# Usage

The script can be run from inside the `google_research/` directory with the
command:

```
python -m wiki_split_bleu_eval.score_main \
    --pred wiki_split_bleu_eval/example_data/predictions.txt \
    --gold wiki_split_bleu_eval/example_data/gold.tsv
```

Output:

```
_gold_file	wiki_split_bleu_eval/example_data/gold.tsv
_pred_file	wiki_split_bleu_eval/example_data/predictions.txt
bleu.corpus.decomp	59.737752755430286
bleu.macro_avg_sent.decomp	59.56959114067446
counts.gold_inputs	2
counts.pred_inputs	2
counts.predictions	2
counts.references	3
lengths.simple_per_complex	3.0
lengths.tokens_per_simple	7.0
lengths.tokens_per_simple_micro	7.0
ref_lengths.simple_per_complex	2.5
ref_lengths.tokens_per_simple	9.083333333333334
refs_per_input.avg	1.5
refs_per_input.max	2
refs_per_input.min	1
uniq_refs_per_input.avg	1.5
uniq_refs_per_input.max	2
uniq_refs_per_input.min	1
```


