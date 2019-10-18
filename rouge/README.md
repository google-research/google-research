# Python ROUGE Implementation

## Overview

This is a native python implementation of ROUGE, designed to replicate results
from the original perl package.

ROUGE was originally introduced in the paper:

Lin, Chin-Yew. ROUGE: a Package for Automatic Evaluation of Summaries. In
Proceedings of the Workshop on Text Summarization Branches Out (WAS 2004),
Barcelona, Spain, July 25 - 26, 2004.

## ROUGE for Python

There are ROUGE implementations available for Python, however some are not
native python due to their dependency on the perl script, and others provide
differing results when compared with the original implementation. This makes it
difficult to directly compare with known results.

This package is designed to replicate perl results. It implements:

*   ROUGE-N (N-gram) scoring
*   ROUGE-L (Longest Common Subsequence) scoring
*   Text normalization
*   Bootstrap resampling for confidence interval calculation
*   Optional Porter stemming to remove plurals and word suffixes such as (ing,
    ion, ment).

Note that not all options provided by the original perl ROUGE script are
supported, but the subset of options that are implemented should replicate the
original functionality.

## Stopword removal

The original ROUGE perl script implemented optional stopword removal (using the
-s parameter). However, there were ~600 stopwords used by ROUGE, borrowed from
another now defunct package. This word list contained many words that may not be
suited to some tasks, such as day and month names and numbers. It also has no
clear license for redistribution. Since we are unable to replicate this
functionality precisely we do not include stopword removal.

## Two flavors of ROUGE-L
In the ROUGE paper, two flavors of ROUGE are described:

1. sentence-level: Compute longest common subsequence (LCS) between two pieces of
text. Newlines are ignored. This is called `rougeL` in this package.
2. summary-level: Newlines in the text are interpreted as sentence boundaries,
and the LCS is computed between each pair of reference and candidate sentences,
and something called union-LCS is computed. This is called `rougeLsum` in this
package. This is the ROUGE-L reported in *[Get To The Point: Summarization with
Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)*, for example.

## How to run

This package compares target files (containing one example per line) with
prediction files in the same format. It can be launched as follows (from
google-research/):

```shell
python -m rouge.rouge \
    --target_filepattern=*.targets \
    --prediction_filepattern=*.decodes \
    --output_filename=scores.csv \
    --use_stemmer=true
```

## Using pip
```
pip install rouge/requirements.txt
pip install rouge-score
```

Then in python:

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
scores = scorer.score('The quick brown fox jumps over the lazy dog',
                      'The quick brown dog jumps on the log.')
```

## License

Licensed under the
[Apache 2.0](https://github.com/google-research/google-research/blob/master/LICENSE)
License.

## Disclaimer

This is not an official Google product.
