# Supporting Materials for the Perso-Arabic Script Normalization

This document provides a brief summary of Perso-Arabic script normalizer
(described in the next section) that is used in statistical language modeling
and neural machine translation experiments, also briefly outlined below
in this document.

## Script Normalization Utilities

The Perso-Arabic script normalizer uses the normalization grammars provided by
the [Nisaba](https://github.com/google-research/nisaba) finite-state script
processing library. These grammars are compiled using
[Pynini](https://www.openfst.org/twiki/bin/view/GRM/Pynini), which is a Python
finite-state grammar development toolkit.

The normalizer (implemented `normalize_text.py`) and the corresponding
Perso-Arabic normalization grammars for each of the supported languages can be
built using [Bazelisk](https://github.com/bazelbuild/bazelisk), a user-friendly
helper for the [Bazel](https://bazel.build/) build system written in Go. To
build the normalizer, follow the next steps (shown for Linux, but macOS and
Windows platforms are supported as well).

```shell
# Install Bazelisk.
BAZEL=bazelisk-lunux-amd64
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.12.0/${BAZEL}
chmod +x $BAZEL

# Build the normalizer.
./${BAZEL} build -c opt :normalize_text
```

The above build process produces the following artifacts:

1. The actual normalizer binary `bazel-bin/normalize_text`.
1. The multiple Nisaba script normalization grammars for the Perso-Arabic
   writing systems of languages supported by Nisaba stored in

   ```
   bazel-bin/normalize_text.runfiles/com_google_nisaba/nisaba/scripts/abjad_alphabet/reading_norm.far
   ```

   where the extension `far` means FST archive.

The normalizer processes either plain text input files or files compressed using
[BZip2](https://en.wikipedia.org/wiki/Bzip2) or
[GZip](https://en.wikipedia.org/wiki/Gzip) compression. As an example, to apply
the normalizer to a dump of Kashmiri Wikipedia using the Kashmiri normalization
grammar (specified by the upper-case BCP-47 language code `KS`) run

```shell
bazel-bin/normalize_text \
  --corpus_file ${INPUT_DIR}/kswiki-20211101-pages-meta-current_thr0.6.txt.bz2 \
  --far_file bazel-bin/normalize_text.runfiles/com_google_nisaba/nisaba/scripts/abjad_alphabet/reading_norm.far \
  --grammar_name KS
  --output_file ${OUTPUT_DIR}/normalized.txt.bz2
```

The normalizer proceeds token-by-token, where the sentences are split on
whitespace.  The normalizer may or may not modify the tokens, but in general the
resulting output file should have the same number of tokens. Additional
arguments are provided to store all the unique tokens that have been normalizer
in a file (`--output_token_diffs_file`) and also record the sentence/line IDs
that were changed by normalization (`--output_line_diffs_file`).

## Statistical Language Modeling

This section describes the tools used for statistical language modeling
experiments and presents some of the experiment artifacts.

### Tools

The set of tools for statistical language modeling (SLM) experiments is
reasonably simple and mostly involves utilities for data preprocessing. Unlike
the script normalizer tool described in the previous section, none of the tools
described here rely on Bazel but are run with Python interpreter directly in a
Python virtual environment.

Because the experiments focus on Wikipedia data, the first tool
`wiki_dump_to_text.py` converts publicly available Wikipedia dumps from XML
format to the regular newline-separated text files. The following example, run
in a virtual environment, will convert (imperfectly) the Kashmiri Wikipedia dump
(from [here](https://dumps.wikimedia.org/kswiki/latest/) into text format:

```shell
# Install dependencies.
pip3 install absl-py mwparserfromhell mwtypes mwxml unicodedataplus

# Run converter.
python wiki_dump_to_text.py \
  --xml_dump_file ${DATA_DIR}/wiki/kswiki-latest-pages-meta-current.xml.bz2 \
  --prob_threshold 0.6 \
  --output_file ${CORPUS_DIR}/kswiki.txt.bz2
```

When converting, there is an option to drop all the lines that are determined to
be in the script `--script_name` and have probability lower than
`--prob_threshold`. This functionality is useful for dropping as much data in
foreign scripts as possible. The above command will create a simple compressed
text file `kswiki.txt.bz` in directory specified by `${CORPUS_DIR}`.

We employ [KenLM](https://github.com/kpu/kenlm) statistical language modeling
toolkit for constructing the interpolated n-gram language models using modified
Kneser-Ney smoothing. The standard KenLM distribution was modified to support
higher-order (up to 10) n-grams (please see the discussion
[here](https://github.com/kpu/kenlm/issues/140) and
[here](https://github.com/kpu/kenlm/issues/75)). The script that trains and
evaluates the n-gram models is `ngram_train_and_eval.py`:

```shell
# Install the dependencies.
pip3 install absl-py kenlm numpy pandas

# Train the models.
python ngram_train_and_eval.py \
  --corpus_file ${CORPUS_DIR}/kswiki.txt.bz2 \
  --line_diffs_file ${CORPUS_DIR}/kswiki_line_diffs.pickle \
  --line_token_diffs_file ${CORPUS_DIR}/kswiki_token_diffs.tsv \
  --num_trials 100 \
  --order 7 \
  --output_model_dir ${RESULTS_DIR}
```

The above command will train interpolated character 7-gram models (word models
are trained when `--word_models` flag is enabled) from the Kashmiri
Wikipedia. When parsing the input text file it is randomly split into training
and test portion based on the value of `--train_test_ratio` flag (which is 0.8
by default). The model trained using the training set is validated on the test
set and perplexity for the run is remembered. This process (split, train and
test) is repeated 100 times. The result is stored in the tab-separated text file
(in tsv format) in the `${RESULTS_DIR}` directory. The file has 100 lines and
consists of three columns: number of tokens (either characters or words) in the
training and test sets, as shown by the following snippet from one of the
results files
`data/ngrams/results/pure_baselines/kswiki-20211101-pages-meta-current_thr0.6.txt_7gram_tr0.80_report.tsv`:

```shell
...
256142  59586   6.778626013067323
249408  66320   6.741281520904354
249086  66642   7.1878213833858675
257712  58016   6.233670463417501
249210  66518   6.429110205040371
...
```

Since for each language and each trial the splits are performed randomly based on the number of lines,
these lines have different numbers of tokens between the runs. This variation can be summarized using
the `describe_splits.py` utility that collects basic stats for training and test splits from the results
file(s) and computes the sufficient statistics, as in the following example:

```shell
# Install the dependencies.
> pip3 install absl-py pandas statsmodels

# Collect the statistics from a file that contains all the results from *all* the trials for *all*
# the orders, e.g., for n-grams between 3 and 10, and 100 trials, the file `${LANGUAGE}.tsv` will
# have 800 rows.
> LANGUAGE=ks
> cat data/ngrams/results/reading/00/baselines/${LANGUAGE}.*.tsv > /tmp/${LANGUAGE}.tsv
> python describe_splits.py --results_tsv_file /tmp/${LANGUAGE}.tsv

I0811 19:35:16.147488 describe_splits.py:45] Reading metrics from /tmp/ks.tsv ...
I0811 19:35:16.151791 describe_splits.py:47] Read 800 samples
I0811 19:35:16.152310 describe_splits.py:50] Train stats: mean: 275583.665 var: 6694169.4352750005 std: 2587.3093041372153
I0811 19:35:16.152624 describe_splits.py:53] Test stats: mean: 40144.335 var: 6694169.435275 std: 2587.309304137215
```

Once the perplexity results for all the runs are available, these can be
compared between the normalized and unnormalized condictions using the
`analyze_ngram_metrics.py` utility.

```shell
# Install the dependencies.
> pip3 install absl-py numpy pandas scipy statsmodels

# Compare character n-gram models.
> REPORT_DIR=data/ngrams/results/reading/00
> python analyze_ngram_metrics.py \
  --baseline_metrics_dir ${REPORT_DIR}/baselines \
  --test_metrics_dir ${REPORT_DIR}/rewrites \
  --language ks \
  --output_tex_table_file /tmp/ks.tex

...
I0811 19:47:07.412830 utils.py:54] Reading metrics from data/ngrams/results/reading/00/rewrites/ks.txt_9gram_tr0.80_report.tsv ...
I0811 19:47:07.414561 utils.py:56] Read 100 samples
t-test: -0.02223 (-0.85%), 95% CI = [-0.03999, -0.00447] t-statistic: -2.46798, p-value: +0.01444, t-dof: +197.24915
Mann-Whitney U: MannwhitneyuResult(statistic=6007.0, pvalue=0.013921951647328085)
Brunner-Munzel: BrunnerMunzelResult(statistic=-2.5097244523899853, pvalue=0.012889830729282412)
...

# Word n-gram models (similar sequence of calls to above, but note that it
# runs on different set of results in a different location).
REPORT_DIR=data/ngrams/results/reading/01
...
```

The above example does multiple pair-wise comparison between different trials
and different n-gram orders. In the case of character n-grams of orders betwen 3
and 10, and 100 random trials for each model, this corresponds to pairwise
statistical significance comparison of two sets of 800 samples each. Three types
of significance testing are performed:

1.  The classical t-test. We employ Welch’s formulation with Satterthwaite’s
    degrees of freedom,
1.  Non-parametric Mann-Whitney test,
1.  Non-parametric Brunner-Munzel test.

The results of the above signicance tests including confidence intervals,
t-statistic and p-values as well as differences in means are saved in the TeX
table in `/tmp/ks.tex`.

Finally, individual results files in tsv format can be used to generate entropy or
perplexity plots using `plot_entropy.py` utility that generates the plots of
cross-entropy or perplexity vs. the n-gram order:

```shell
# Install the depdencies.
pip3 install absl-py pandas seaborn

# Generate the plots.
python plot_entropy.py \
  --results_dir data/ngrams/results/pure_baselines \
  --language_list ks,pnb \
  --output_plots_dir /tmp
```

The above example will generate the plots of cross-entropy vs. n-gram order for
Kashmiri and Punjabi (Shahmukhi) in `/tmp/ks_ent.pdf` and `/tmp/pnb_ent.pdf`, as
well as the combined plot in `/tmp/all_ent.pdf`. The data is taken from the
individual language-specific files in the results directory
`data/ngrams/results/pure_baselines`. Perplexity plots can be generated by
enabling the `--use_perplexities` flag.

### Artifacts

The artifacts of the n-gram experiments are the results files in the following directories:

1.  `data/ngrams/results/pure_baselines/`: Character n-gram models trained on
    unsplit and unnormalized data.
1.  `data/ngrams/results/reading/00`: Character n-gram models trained on
    unnormalized (`baselines`) and normalized using reading normalization
    (`rewrites`). Orders: 3 to 10.
1.  `data/ngrams/results/reading/01`: Word n-gram models trained on
    unnormalized (`baselines`) and normalized using reading normalization
    (`rewrites`). Orders: 2 to 5.

The Perso-Arabic writing systems under investigation are Kashmiri, Malay (Jawi),
Punjabi (Shahmukhi), Sindhi, Sorani Kurdish, South Azerbaijani, Urdu, and
Uyghur.

## Neural Machine Translation

Similar to SLM experiments, the neural machine translation (NMT) experiments
study the effects on normalization on the quality of NMT models. Monolingual and
multilingual NMT systems are constructed that translate from single or multiple
languages with Perso-Arabic writing systems belonging to a subset of the
languages mentioned above (namely Sorani Kurdish, Sindhi, Urdu and Uyghur) to
English. Evaluation metrics from unnormalized and normalized systems are then
systematically compared.

### Tools

The parallel corpora were collected using the
[MTData](https://github.com/thammegowda/mtdata) tool that automates the
collection and preparation of open-source machine translation datasets. The
MTData scripts (known as *signatures* in MTData parlance) for individual
languages can be found under `data/neural/mtdata` directory. For example,
to collect the parallel Urdu-English data into `${CORPUS_DIR}`.

```shell
# Install the dependencies.
> pip3 install mtdata

# Inspect the signature.
> cat data/neural/mtdata/urd/mtdata.signature.txt
mtdata get -l urd-eng -tr OPUS-opus100_train-1-eng-urd JoshuaDec-indian_training-1-urd-eng Anuvaad-thewire-20210320-eng-urd -ts OPUS-opus100_test-1-eng-urd JoshuaDec-indian_test-1-urd-eng -dv OPUS-opus100_dev-1-eng-urd JoshuaDec-indian_dev-1-urd-eng --merge -o <out-dir>
mtdata version 0.3.5

# Collect the corresponding open-source data into the training, development and
# test sets.
> mtdata get -l urd-eng \
   -tr OPUS-opus100_train-1-eng-urd JoshuaDec-indian_training-1-urd-eng Anuvaad-thewire-20210320-eng-urd \
   -dv OPUS-opus100_dev-1-eng-urd JoshuaDec-indian_dev-1-urd-eng \
   -ts OPUS-opus100_test-1-eng-urd JoshuaDec-indian_test-1-urd-eng \
   --merge -o ${CORPUS_DIR}
```

The exact summary of the resulting training corpora can be found in
`train.stats.json` file.

The [OpenNMT](https://opennmt.net/) neural machine translation toolkit is used
to construct the translation models, in particular, we use the
[TensorFlow](https://www.tensorflow.org/)-based version of OpenNMT. We use
simple LSTM encoder-decoder architecture equipped with attention. Depending on
the amounts of training data available we use different hyperparameter
configurations: `NMTSmallV1` for smaller corpora and `NMTMediumV1` for bigger
amounts of training data (e.g., for a multilingual model). The OpenNMT recipes
are provided for each individual scenario under `data/neural/recipe`
directory. For Urdu, for example:

```shell
# Install the dependencies.
> pip3 install tensorflow OpenNMT-tf

# Inspect the skeleton for the recipe.
> cd data/neural/recipe/urd/NMTMediumV1 && ls
README.md  data.yml  tokenizer.yml  train.sh
```

The files are as follows:

1. `README.md` contains the sequence of basic OpenNMT commands required to run
   the training.
1. The configuration for the tokenizer is provided in `tokenizer.yml`.
1. The `data.yml` describes the actual training, development and test data, and
   provides some core parameters for OpenNMT. In particular, this configuration
   expects the following files to be under `data/neural/recipe/urd/NMTMediumV1`:
   *  Training features and targets (label) files `src-train.txt` and `tgt-train.txt`.
   *  Development (validation) features and targets files `src-val.txt` and
      `tgt-val.txt`.
   *  Tokenizer vocabularies for training features and targets `src-vocab.txt` and
      `tgt-vocab.txt`.

The above feature and target files can point to the original `${CORPUS_DIR}` splits
or, in the case of normalized data, to the respective files in that directory. A
sequence of commands described in `README.md` can then be used to instantiate the
training and evaluation process:

```shell
# Install the dependencies.
pip3 install tensorflow OpenNMT-tf sacrebleu

# Construct the tokenizer vocabularies.
cd data/neural/recipe/urd/NMTMediumV1
onmt-build-vocab --tokenizer_config tokenizer.yml --size 50000 --save_vocab src-vocab.txt src-train.txt
onmt-build-vocab --tokenizer_config tokenizer.yml --size 50000 --save_vocab tgt-vocab.txt tgt-train.txt

# Run the training and evaluation script.
(nohup ./train.sh > train.log 2>&1) > /dev/null 2>&1 &
```

The resulting artifacts and results can be found under `run/NMTMediumV1`
subdirectory. Note that we are using NMT evaluation scores implementation from
[SacreBleu](https://github.com/mjpost/sacrebleu) to compute the scores for
individual NMT models as well as performing the pairwise statistical
significance testing. In addition, a simple utility `analyze_nmt_metrics.py`
is provided for comparing the resulting models:

```shell
# Install the depdencies:
pip3 install absl-py numpy pandas statsmodels

# Compare the individual training epochs of two models (baseline and the test):
RESULTS_DIR=data/neural/results/ckb/NMTSmallV1/
python analyze_nmt_metrics.py \
  --baseline_metrics_dir ${RESULTS_DIR}/original/ \
  --test_metrics_dir ${RESULTS_DIR}/rewrites/ \
  --metric_file_basenames test1,test2 \
  --num_epochs 8
```

### Artifacts

The results are provided under `data/neural/results` directory for four
individual languages: Sorani Kurdish (`ckb`, but also including Kurmanji in the
training data), Sindhi (`snd`), Urdu (`urd`), Uyghur (`uig`), and the
multilingual many-to-English model (`multi`). Each directory has the following
structure:

1.  Results of Paired Approximate Randomization (PAR) test from SacreBleu for
    the final 8th epoch of the training `test?.paired-ar.epoch8` (where `test?`
    corresponds to the name of the test set).
1.  Results of Paired Bootstrap Resampling (PBS) test from SacreBleu for the
    final 8th epoch of the training `test?.paired-bs.epoch8`.
1.  Directory `data/neural/results/snd/MODEL/original/` corresponding to the
    model build on unnormalized parallel corpora, where `MODEL` corresponds
    to the model configuration (`NMTSmallV1` or `NMTMediumV1`).
1.  Directory `data/neural/results/snd/MODEL/rewrites/` corresponding to the
    model build on normalized parallel corpora.
1.  Each `original` and `rewrites` subdirectory contains
    * The hypothesis files `test?.eng.epoch?`, where `test?` is the name of the
      test set and `epoch?`  corresponds to one of the eight epochs.
    * SacreBleu scores (BLEU, chrF2 and TER) under `test?.results.epoch?`.
1.  In addition, for multilingual model only, where SacreBleu PAR and PBS
    pairwise testing is performed, there is an additional subdirectory
    `data/neural/results/multi/NMTMediumV1/pairwise` that contains the
    following:
    * The helper script `pairwise.sh` for running PBS and PAR tests on all
    the epochs of unnormalized and normalized model.
    * Individual PBS and PAR test results for each language and each epoch
    under the `test?.epoch?` subdirectory.



