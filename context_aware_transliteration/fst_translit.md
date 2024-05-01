# Running Nisaba FST-based transliteration with released models.

This document contains details on running Nisaba FST-based transliteration, as
used in the paper. [Return to main page.](README.md)

The roughly 3.6GB tar file
[models.tar](https://storage.googleapis.com/gresearch/context_aware_transliteration/models.tar)
includes three kinds of models for all 12 languages studied in the paper: 1)
native script language models trained on Wikipedia text, encoded as FSTs; 2)
Pair-6g transliteration models, encoded as FSTs; and 3) Wordpiece models,
encoded as ordered TSV files.

The scripts rely on the [Nisaba](https://github.com/google-research/nisaba)
library being installed locally (see the instructions at the bottom of [this
page](generate_tables.md)).  Let `$MYNISABA` be a variable pointing to the
Nisaba library directory, which should have a compiled `bazel-bin` subdirectory
underneath it.

Let `$MYLOCALDIR` be a variable pointing to the local directory where the
above-linked tar file will be downloaded.  Note that the tar file and
resulting models are quite large.

First get the tar file and untar it.

```
cd $MYLOCALDIR
wget https://storage.googleapis.com/gresearch/context_aware_transliteration/models.tar
tar -xf models.tar
```

This will result in three subdirectories: 1) `$MYLOCALDIR`/models/LM contains
both contextual and non-contextual language models for each language; 2)
`$MYLOCALDIR`/models/pairLM contains 25 different pair LM transliteration models
for each language; and 3) `$MYLOCALDIR`/models/wordpiece contains wordpiece
models for each language.

Note that all of the models are gzipped, so they will need to be gunzip'd prior
to use.  For the following example usages, let us assume that we are
translitering Bengali (bn) and that we are just using the transliteration model
number 00.  To romanize an input sentence in the Bengali script, and return the
two best romanizations for each word:

```
L="bn"
N="00"
PAIRLM="${MYLOCALDIR}/models/pairLM/${L}.${N}.pair6g.fst"
INPUT_SENTENCE="এটি একটি উদাহরণ বাক্য"
IFILE="/tmp/tmp.input.txt"
OFILE="/tmp/tmp.output.txt"
gunzip "${PAIRLM}.gz"
echo "${INPUT_SENTENCE}" >"${IFILE}"
"${MYNISABA}"/bazel-bin/nisaba/translit/fst/pairlm_decoder \
  --ifile="${IFILE}" \
  --ofile="${OFILE}" \
  --pairlm_config="pairlm_file: \"${PAIRLM}\"" \
  --kbest=2
gzip "${PAIRLM}"
cat "${OFILE}"
0	0	এটি		eti			0.121981
0	0	এটি		etee		2.16426
0	1	একটি	ekti		0.394668
0	1	একটি	ektee		1.12056
0	2	উদাহরণ	udaharan	0.520971
0	2	উদাহরণ	udahoron	0.901262
0	3	বাক্য	  bakyo		  0.685076
0	3	বাক্য	  bakya		  0.701284
```

In the output, the first integer is the sentence number, the second is the word
position in the sentence, the third is the input word, the fourth is the output
transliteration and the fifth is the (softmax) normalized weight (negative log
probability).

To go in the other direction, from Latin script to native script, you must
invert the model:

```
L="bn"
N="00"
PAIRLM="${MYLOCALDIR}/models/pairLM/${L}.${N}.pair6g.fst"
INPUT_SENTENCE="eti ekti udaharan bakyo"
IFILE="/tmp/tmp.input.txt"
OFILE="/tmp/tmp.output.txt"
gunzip "${PAIRLM}.gz"
echo "${INPUT_SENTENCE}" >"${IFILE}"
"${MYNISABA}"/bazel-bin/nisaba/translit/fst/pairlm_decoder \
  --ifile="${IFILE}" \
  --ofile="${OFILE}" \
  --pairlm_config="pairlm_file: \"${PAIRLM}\" invert_pairlm: true" \
  --kbest=2
gzip "${PAIRLM}"
cat "${OFILE}"
0	0	eti	এটি	0.0735731
0	0	eti	এতি	2.64603
0	1	ekti	একটি	0.0549145
0	1	ekti	এক্তি	2.9293
0	2	udaharan	উদাহরণ	0.123822
0	2	udaharan	উদাহরন	2.15018
0	3	bakyo	বাক্য	0.0340729
0	3	bakyo	বক্য	3.39624
```

To run using a language model, which in our case also requires a wordpiece
model, these must be specified in the config.  Note that language and wordpiece
models are used on the output side, and hence are only used in Latin-to-native
script transliteration:

```
L="bn"
N="00"
LMNAME="${L}.wiki-filt.train.text.shuf.ws_norm.wp"
CONTEXT_LM="${MYLOCALDIR}/models/LM/${LMNAME}.context.mod.fst"
WPMOD="${MYLOCALDIR}/models/wordpiece/${L}.wp_model.32k.ordered.tsv"
PAIRLM="${MYLOCALDIR}/models/pairLM/${L}.${N}.pair6g.fst"
INPUT_SENTENCE="eti ekti udaharan bakyo"
IFILE="/tmp/tmp.input.txt"
OFILE="/tmp/tmp.output.txt"
gunzip "${PAIRLM}.gz"
gunzip "${CONTEXT_LM}.gz"
gunzip "${WPMOD}.gz"
echo "${INPUT_SENTENCE}" >"${IFILE}"
"${MYNISABA}"/bazel-bin/nisaba/translit/fst/pairlm_decoder \
  --ifile="${IFILE}" \
  --ofile="${OFILE}" \
  --pairlm_config="pairlm_file: \"${PAIRLM}\" invert_pairlm: true \
                   lm_file: \"${CONTEXT_LM}\" oov_symbol: \"<unk>\" \
                   word_piece_model: \"${WPMOD}\"" \
  --kbest=2
gzip "${PAIRLM}"
gzip "${CONTEXT_LM}"
gzip "${WPMOD}"
cat "${OFILE}"
0	0	eti	এটি	1.86845e-05
0	0	eti	টি	11.1736
0	1	ekti	একটি	-1.51325e-05
0	1	ekti	একটিই	13.0955
0	2	udaharan	উদাহরণ	0.0361591
0	2	udaharan	উদাহরন	3.33747
0	3	bakyo	বাক্য	0.000191451
0	3	bakyo	ব্যাখ্য	8.72112
```

This took longer, and the 1-best output is no different, though the best scoring
word at each position has higher probability (lower negative log probability)
than when the language model was not used, i.e., the distribution was more
peaked.  (Note that negative weights are due to floating point precision issues,
i.e., they are approximately zero.)

If we want to use a non-contextual language model, i.e., just at the word-level
without any sentential context, closure will need to be applied to the language
model to transliterate the whole string:

```
L="bn"
N="00"
LMNAME="${L}.wiki-filt.train.text.shuf.ws_norm.wp"
NOCONTEXT_LM="${MYLOCALDIR}/models/LM/${LMNAME}.no-context.mod.fst"
WPMOD="${MYLOCALDIR}/models/wordpiece/${L}.wp_model.32k.ordered.tsv"
PAIRLM="${MYLOCALDIR}/models/pairLM/${L}.${N}.pair6g.fst"
INPUT_SENTENCE="eti ekti udaharan bakyo"
IFILE="/tmp/tmp.input.txt"
OFILE="/tmp/tmp.output.txt"
gunzip "${PAIRLM}.gz"
gunzip "${NOCONTEXT_LM}.gz"
gunzip "${WPMOD}.gz"
echo "${INPUT_SENTENCE}" >"${IFILE}"
"${MYNISABA}"/bazel-bin/nisaba/translit/fst/pairlm_decoder \
  --ifile="${IFILE}" \
  --ofile="${OFILE}" \
  --pairlm_config="pairlm_file: \"${PAIRLM}\" invert_pairlm: true \
                   lm_file: \"${NOCONTEXT_LM}\" apply_closure_to_lm: true \
                   oov_symbol: \"<unk>\" word_piece_model: \"${WPMOD}\"" \
  --kbest=2
gzip "${PAIRLM}"
gzip "${NOCONTEXT_LM}"
gzip "${WPMOD}"
cat "${OFILE}"
0	0	eti	এটি	0.00749036
0	0	eti	টি	4.91153
0	1	ekti	এটি	8.80764
0	1	ekti	একটি	-4.49489e-05
0	2	udaharan	উদাহরণ	0.0113594
0	2	udaharan	উদাহরন	4.46381
0	3	bakyo	বাক্য	-9.45233e-05
0	3	bakyo	ব্যাখ্য	9.91945
```
