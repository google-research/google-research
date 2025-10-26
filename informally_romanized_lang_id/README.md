# Improving Informally Romanized Language Identification

This directory contains details for replicating the results in the paper [Improving Informally Romanized Language Identification](https://arxiv.org/abs/2504.21540).

#### Language identifiers.

There is some conversion and sharing of data across three resources: the
Bhasha-Abhijnaanam (B-A) benchmark dataset, the Aksharantar romanization
lexicons, and the Dakshina dataset. In some cases, the methods assume that a
resource shares the same file root name, so that a romanization lexicon's
filename will, for ease of documentation, assumed to be, e.g., Assamese.tsv
instead of asm.tsv. Below are the identifiers for each resource for ease of
reference.

| B-A identifier | Aksharantar code | Dakshina code |
|:---------------|:----------------:|:-------------:|
| Assamese       | asm              |               |
| Bangla         | ben              | bn            |
| Bodo           | brx              |               |
| Gujarati       | guj              | gu            |
| Hindi          | hin              | hi            |
| Kannada        | kan              | kn            |
| Kashmiri       | kas              |               |
| Konkani        | kok              |               |
| Maithili       | mai              |               |
| Malayalam      | mal              | ml            |
| Manipuri       | mni              |               |
| Marathi        | mar              | mr            |
| Nepali         | nep              |               |
| Oriya          | ori              |               |
| Punjabi        | pan              | pa            |
| Sanskrit       | san              |               |
| Sindhi         | sid              | sd            |
| Tamil          | tam              | ta            |
| Telugu         | tel              | te            |
| Urdu           | urd              | ur            |

#### Creating supplementary romanization lexicons.

This work relies on automatic romanization of native script text based on models
learned from romanization lexicons. There is some minor lack of coverage of
characters in the native script, including, e.g., digits, punctuation and other
less common characters. The Bhasha-Abhijnaanam (B-A) benchmark contains a large
parallel (at the sentence level) resource that consists of automatically
romanized native script text. Some romanized symbols are outside of the Dakshina
and/or Aksharantar romanization lexicons. Because the B-A parallel data is
automatically romanized, the number of romanized tokens is always the same as
the number of native script tokens, allowing for straightforward extraction of
native/Latin script pairs. From such a lexicon, lack of coverage can be
repaired by checking entries against the baseline lexicon for coverage.

The ```create_supplementary_lexicons.sh``` script takes as argument a local path
to the B-A json file of the parallel dataset and a directory where baseline
romanization lexicons are provided. As noted above, these will be assumed to
have the same name as the B-A language identifier, see the example usage in the
script for more information.

For the paper, supplementary lexicons were created for each language, using the
Dakshina romanization lexicons as baseline for those languages for which they
exist, and the Aksharantar lexicons as baseline for the remainder.

As a side note, another way to get a similar result is to just train on the
baseline lexicons, then apply universal romanization (such as ```uroman```) to
the output to romanize any stray unconverted characters. For this paper, we used
the extracted supplementary lexicons, but these methods likely perform nearly
identically.

#### Training pair language model transducers.

The (relatively lengthy) ```train_pair_lm.sh``` script takes an input
romanization lexicon in TSV format and produces a pair LM transliteration model
in both automaton and transducer formats in the designated output
directory. Note that a single romanization lexicon file is requires, so if a
supplementary lexicon is generated according to the above approach, this should
be combined with the baseline lexicon into a single file. This TSV should have
two columns, so romanization lexicons (such as Dakshina) which have a third
column showing the count of the pair, should be reformatted by duplicating an
example the number of times of its count. For example, the beginning of the
training partition for Bengali in the Dakshina dataset has:

| Native script | Latin script | Count |
|:--------------|:-------------|:-----:|
| অংকিত         | angkito      | 1     |
| অংকিত         | ankit        | 1     |
| অংকিত         | ankita       | 1     |
| অংকিত         | ankito       | 3     |

Since the fourth pair has a count of three, it should be shown three times,
which allows omission of the count column:

| Native script | Latin script |
|:--------------|:-------------|
| অংকিত         | angkito      |
| অংকিত         | ankit        |
| অংকিত         | ankita       |
| অংকিত         | ankito       |
| অংকিত         | ankito       |
| অংকিত         | ankito       |

Once a single, two-column TSV is created, the script can be called, which goes
through a series of stages, resulting in two FST files, one an automaton and one
a transducer. The scripts makes use of binary executables from the
[OpenFst](https://www.openfst.org/twiki/bin/view/FST/WebHome), [OpenGrm
Baum-Welch](https://www.opengrm.org/twiki/bin/view/GRM/BaumWelch) and [OpenGrm
Ngram](https://www.opengrm.org/twiki/bin/view/GRM/NGramLibrary) libraries, so
those are expected to be installed, and the script will need to be modified to
point to those executables. Note that the [FAR
extensions](https://www.openfst.org/twiki/bin/view/FST/FstExtensions#FST%20Archives%20(FARs))
will be required from the OpenFst library. Also, version 0.3.11 or higher is
required for the OpenGrm Baum-Welch library.
