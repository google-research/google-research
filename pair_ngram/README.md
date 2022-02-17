Pair n-gram models
==================

This is a collection of scripts for building pair n-gram models using the
OpenFst and OpenGrm libraries.

Installation
------------

[Conda](http://conda.io) is recommended for a reproducible environment. Assuming
that Conda (either Miniconda or Anaconda) is available, the following command
creates the environment `pair_ngram`.

    conda env create -f environment.yml

This only needs to be done once. The following command then activates the
environment.

    conda activate pair_ngram

This second step needs to be repeated each time you start a new shell.

Sample use
----------

Knight & Graehl (1998) use cascades of finite-state transducers to predict the
English form of Japanese [katakana](https://en.wikipedia.org/wiki/Katakana)
loanwords. Here we will use a pair n-gram model for this task, noting that many
of these loanwords are multi-word expressions in English.

1.  Create a temporary directory:

        readonly TEMPDATA="$(mktemp --directory)"

2.  Download the data files:

        curl \
            --silent \
            --output "${TEMPDATA}/pairs.tsv" \
            "https://gist.githubusercontent.com/kylebgorman/01adff5799edb0edf3bcce20187c833a/raw/fb0e66d31e021fca7adec4c2104ffea0e879f2e4/pairs.tsv"
        curl \
            --silent \
            --output "${TEMPDATA}/lexicon.txt"
            "http://cvsweb.netbsd.org/bsdweb.cgi/src/share/dict/web2?rev=1.54"

3.  Randomly split the data using [`split`](pair_ngram/split.py).

        python -m pair_ngram.split \
            --seed 10037 \
            --input "${TEMPDATA}/pairs.tsv" \
            --train "${TEMPDATA}/train.tsv" \
            --dev "${TEMPDATA}/dev.tsv" \
            --test "${TEMPDATA}/test.tsv"

    This will log the size of the data, which should be roughly 26,000 lines in
    total.

4.  Train the pair LM using [`train`](pair_ngram/train.py).

        python -m pair_ngram.train \
           --seed 10037 \
           --batch-size 128
           --max-iters 10 \
           --order 4 \
           --size 100000
           --tsv "${TEMPDATA}/train.tsv" \
           --fst "${TEMPDATA}/plm.fst"

    This script depends on OpenFst's command-line tools, its
    [`pywrapfst`](https://www.openfst.org/twiki/bin/view/FST/PythonExtension)
    Python extension (Gorman 2016), and OpenGrm's
    [Baum-Welch](https://baumwelch.opengrm.org) and
    [NGram](https://ngram.opengrm.org) command-line tools. It

    -   compiles input and output FARs,
    -   builds a zeroth-order Markov model covering grammar,
    -   repeatedly randomizes the covering grammar weights and then uses an
        online variant of Viterbi training to set the weights to maximize the
        joint probability of the data,
    -   computes the best alignments using the best aligner from the previous
        stage,
    -   encodes these alignments as FSAs,
    -   builds an n-gram model, applying smoothing and shrinking,
    -   then decodes the n-gram model to produce the final pair LM FST.

    See Gorman in preparation for further details.

5.  Predict the development set English words using `cut` and
    [`predict`](pair_ngram/predict.py):

        python -m pair_ngram.predict \
            --rule "${TEMPDATA}/plm.fst" \
            --input <(cut -f1 "${TEMPDATA}/dev.tsv") \
            --output "${TEMPDATA}/hypo.txt"

6.  Score the model using `cut` and [`error`](pair_ngram/error.py):

        python -m pair_ngram.error \
            --gold <(cut -f2 "${TEMPDATA}/dev.tsv") \
            --hypo "${TEMPDATA}/hypo.txt"

With the random seeds used in the [demo](run.sh), we obtain an error rate of
47.61 on the development set.

A better model can be had by constraining the output to consist of known English
words. For this, see [`predict_lexicon`](pair_ngram/predict_lexicon.py). With
the random seeds used in the [demo](run.sh), we obtain an error rate of 36.54 on
the development set.

Naturally, we would normally use the development set to tune hyperparameters.

Authors
-------

Scripts were created by [Kyle Gorman](mailto:kbg@google.com).

The katakana transliteration data was prepared by Yuying Ren using data from
JMDict (Breen 2004).

The English lexicon is a list of headwords from the second edition of *Webster's
New International Dictionary* (Neilson & Knott 1934) distributed by [the NetBSD
project](https://www.netbsd.org/) as
[`/usr/share/dict/web2`](http://cvsweb.netbsd.org/bsdweb.cgi/src/share/dict/web2?rev=1.54).

Contributing
------------

See [`CONTRIBUTING`](CONTRIBUTING).

Mandatory disclaimer
--------------------

This is not an officially supported Google product.

References
----------

Breen, J. 2004. JMdict: a Japanese-multilingual dictionary. In *Proceedings of
the Workshop on Multilingual Linguistic Resources*, pages 65-72.

Gorman, K. 2016. Pynini: a Python library for weighted finite-state grammar
compilation. In *Proceedings of the ACL Workshop on Statistical NLP and Weighted
Automata*, pages 75-80.

Gorman, K. In preparation. A tutorial on pair n-gram models. Ms., Google Inc.

Knight, K. and Graehl, J. 1998. Machine transliteration. *Computational
Linguistics* 24(4): 599-612.

Neilson, W.A. and Knott, T.A (editors). 1934. *Webster's New International
Dictionary*. 2nd edition. G. & C. Merriam.
