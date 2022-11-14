# Mewsli-X

Mewsli-X is a multilingual dataset of entity mentions appearing in
[WikiNews](https://www.wikinews.org/) and
[Wikipedia](https://www.wikipedia.org/) articles, that have been automatically
linked to [WikiData](https://www.wikidata.org/) entries.

The primary use case is to evaluate transfer-learning in the zero-shot
cross-lingual setting of the
[XTREME-R benchmark suite](https://sites.research.google/xtremer):

1.  Fine-tune a pretrained model on English Wikipedia examples;
2.  Evaluate on WikiNews in other languages &mdash; **given an *entity mention*
    in a WikiNews article, retrieve the correct *entity* from the predefined
    candidate set by means of its textual description.**

Mewsli-X constitutes a *doubly zero-shot* task by construction: at test time, a
model has to contend with different languages and a different set of entities
from those observed during fine-tuning.

👉 For data examples and other editions of Mewsli, see [README.md](README.md).

👉 Consider submitting to the
**[XTREME-R leaderboard](https://sites.research.google/xtremer)**. The XTREME-R
[repository](https://github.com/google-research/xtreme) includes code for
getting started with training and evaluating a baseline model in PyTorch.

👉 Please cite this paper if you use the data/code in your work: *[XTREME-R:
Towards More Challenging and Nuanced Multilingual Evaluation (Ruder et al.,
2021)](https://aclanthology.org/2021.emnlp-main.802.pdf)*.

>> _**NOTE:** New evaluation results on Mewsli-X are **not** directly comparable to those reported in the paper because the dataset required further updates, as detailed [below](#updated-dataset). This does not affect the overall findings of the paper._

```
@inproceedings{ruder-etal-2021-xtreme,
    title = "{XTREME}-{R}: Towards More Challenging and Nuanced Multilingual Evaluation",
    author = "Ruder, Sebastian  and
      Constant, Noah  and
      Botha, Jan  and
      Siddhant, Aditya  and
      Firat, Orhan  and
      Fu, Jinlan  and
      Liu, Pengfei  and
      Hu, Junjie  and
      Garrette, Dan  and
      Neubig, Graham  and
      Johnson, Melvin",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.802",
    doi = "10.18653/v1/2021.emnlp-main.802",
    pages = "10215--10245",
}
```

## Getting Started

You can obtain the Mewsli-X dataset by following these steps, which have been
verified on Linux and MacOS:

**1. Satisfy dependencies**

Environment specifications are provided for [`anaconda`](https://www.anaconda.com/).
To use them, first install `anaconda` yourself.

Alternatively, if you do not want to use `anaconda`, ensure that your system has
the following programs installed:

-   Python 3 (tested with 3.9.6)
-   bzip2
-   git
-   md5sum/md5
-   pip
-   virtualenv
-   wget
-   zip

**2. Get the code**

Either

a) grab the relevant subdirectory of the google-research repository:

```
svn export https://github.com/google-research/google-research/trunk/dense_representations_for_entity_retrieval
```

or

b) download and extract the whole google-research repository (182MB as of June
2022):

```
wget -O google-research-master.zip https://github.com/google-research/google-research/archive/refs/heads/master.zip
unzip google-research-master.zip
cd google-research-master/
```

Then change into the `mel/` subdirectory:

```
cd dense_representations_for_entity_retrieval/mel
```

**3. Create virtual environment and install dependencies**

*Option 1 (safer): Let `anaconda` handle all dependencies:*

```
bash create-env.sh conda
conda activate mewsli_env
```

*Option 2: Only install pip-packages into a new `virtualenv`:*

```
bash create-env.sh
# Then run the activation command printed to the console
```

**4. Run the shell script**

```
bash get-mewsli-x.sh
```

This script runs in 5-10 minutes. It

-   downloads the dataset archive (`mewsli-x_20220518_6.zip`, 291 MB) and public
    dumps of WikiNews (107MB);
-   extracts article text from the WikiNews dumps using a patched version of the
    third-party [`wikiextractor`](https://github.com/attardi/wikiextractor)
    tool;
-   inserts the article text into the released files
    `wikinews_mentions_no_text-{split}.jsonl` to produce the final
    `wikinews_mentions-{split}.jsonl` files.

Upon successful completion, **data files can be found at
`./mewsli_x/output/dataset/*.jsonl`**, and the console output should end with a
message and file listing similar to:

```
...
The Mewsli-X dataset is now ready in /tmp/google-research-master/dense_representations_for_entity_retrieval/mel/mewsli_x/output/dataset/:
.. 535M .. /tmp/google-research-master/dense_representations_for_entity_retrieval/mel/mewsli_x/output/dataset/candidate_set_entities.jsonl
.. 9.8M .. /tmp/google-research-master/dense_representations_for_entity_retrieval/mel/mewsli_x/output/dataset/wikinews_mentions-dev.jsonl
.. 2.9M .. /tmp/google-research-master/dense_representations_for_entity_retrieval/mel/mewsli_x/output/dataset/wikinews_mentions_no_text-dev.jsonl
..  12M .. /tmp/google-research-master/dense_representations_for_entity_retrieval/mel/mewsli_x/output/dataset/wikinews_mentions_no_text-test.jsonl
..  35M .. /tmp/google-research-master/dense_representations_for_entity_retrieval/mel/mewsli_x/output/dataset/wikinews_mentions-test.jsonl
..  24M .. /tmp/google-research-master/dense_representations_for_entity_retrieval/mel/mewsli_x/output/dataset/wikipedia_pairs-dev.jsonl
.. 283M .. /tmp/google-research-master/dense_representations_for_entity_retrieval/mel/mewsli_x/output/dataset/wikipedia_pairs-train.jsonl
```

### Usage examples

The repo includes Python dataclasses in `./mewsli_x/schema.py` that seek to make
it easier to work with the dataset. This section demonstrates their basic usage,
while the data schemas are defined in more detail in the subsequent sections.

*Here is the output from an interactive Python session, run from the parent
directory of `dense_representations_for_entity_retrieval/`*:

```
>>> from dense_representations_for_entity_retrieval.mel.mewsli_x import schema

# Load WikiNews dev set.
wikinews_dev = schema.load_jsonl(
    'dense_representations_for_entity_retrieval/mel/mewsli_x/output/dataset/wikinews_mentions-dev.jsonl',
    schema.ContextualMentions)
len(wikinews_dev)
# 2618

# Select an item to inspect. An item covers one article and possibly multiple mentions.
example_doc = wikinews_dev[999]

# You can reformat it into a single mention per item:
example = next(example_doc.unnest_to_single_mention_per_context())

# Inspect the mention string and the gold WikiData entity it resolves to.
example.mention.mention_span.text, example.mention.entity_id
# ('escala de Richter', 'Q38768')

# The full article text is available in `example.context.text`, but you may
# want to limit it to, say, just the sentence containing the mention string.
example = example.truncate(window_size=0)

# Look at the retained sentence containing the mention.
sentence = next(example.context.sentences)
sentence.text
# '26 de marzo de 2012\nUn fuerte y prolongado sismo de magnitud 7,1 grados en la escala de magnitud de momento, según el Servicio Geológico de Estados Unidos, y de 6,8 grados en la escala de Richter, según el Gobierno sudamericano, sacudió la zona centro y sur de Chile, así como la ciudad de Mendoza en Argentina.'

# Character offsets provide the correspondence between the mention and the
# context, and this is maintained by the truncate() call above, i.e.
example.context.text[example.mention.mention_span.start:example.mention.mention_span.end] == example.mention.mention_span.text
# True

# Load the candidate set
candidates = schema.load_jsonl(
    'dense_representations_for_entity_retrieval/mel/mewsli_x/output/dataset/candidate_set_entities.jsonl',
     schema.Entity)
len(candidates)
# 1003893

# Normally, you'd preprocess the information in `example` into a suitable
# form and feed it to a model tasked with finding the single correct entity
# among the `candidates`.
# In this example, just locate the gold entity by its ID to look at some of
# the information provided for an entity:
entity = next(e for e in candidates if e.entity_id == example.mention.entity_id)

# The entity description for this entity happens to be in Indonesian.
entity.description_language
# 'id'

# The entity description is in the `description` attribute and may be a few
# sentences long. Here's how you would extract the first sentence:
next(entity.sentences).text
# 'Skala Richter atau SR didefinisikan sebagai logaritma dari amplitudo maksimum, yang diukur dalam satuan mikrometer, dari rekaman gempa oleh instrumen pengukur gempa Wood-Anderson, pada jarak 100 km dari pusat gempanya.'

# Finally, here's how to load the Wikipedia example pairs.
wikipedia_dev = schema.load_jsonl(
    'dense_representations_for_entity_retrieval/mel/mewsli_x/output/dataset/wikipedia_pairs-dev.jsonl',
    schema.MentionEntityPair)
len(wikipedia_dev)
# 14051
```

## Dataset Components

The dataset includes all the components needed for the reproducible evaluation
of entity retrieval models: Wikipedia training examples, WikiNews evaluation
instances, and a fixed set of candidate entities.

***Note on versions:*** The overall data extraction process is described in the
[paper's appendix](https://aclanthology.org/2021.emnlp-main.802.pdf). In order
to ensure the quality and usability of the dataset, it was necessary to sample a
new, updated version (mewsli-x_20220518_6) compared to what was originally used
in the paper. This means results obtained on the released dataset are not
directly comparable to those reported in the paper. Changes:

-   removed text pretokenization in favor of using raw text; this supports newer
    token-free modeling approaches.
-   added more stringent text filtering to exclude noisy passages derived from
    lists, tables or reference sections.
-   ensured 1:1 mapping between entity descriptions and entity IDs.
-   added gold entities from the Wikipedia dev-set to the candidate set, to
    enable a full retrieval evaluation to be run on that dev-set.
-   improved sentence boundary annotations.

### Entity Candidate Set

The entity candidate set simulates a frozen knowledge base that entity mentions
must be linked to.

The set consists of the gold entities from the WikiNews portion (dev and test
sets) and the Wikipedia portion (dev set only), plus a random sample to reach
the target size of 1 million. The random sample is balanced across five
frequency bins following the underlying frequency distribution of hyperlinked
entities across the 50 Wikipedia language editions listed below.

#### *File & Format*

-   `candidate_set_entities.jsonl`

Text file where each line is a JSON-serialized dictionary representing one
entity:

Key                    | Definition
---------------------- | ----------
`entity_id`            | WikiData QID that uniquely identifies an entity.
`title`                | Wikipedia entity page title
`description`          | A string description for the entity &mdash; up to the first three sentences from the entity’s Wikipedia page.
`description_url`      | URL of the entity's Wikipedia page that yielded `description`.
`description_language` | ISO 639-1 language code of the entity's Wikipedia page that yielded `description`.
`sentence_spans`       | List of (`start`, `end`)-pairs denoting sentences, as character offsets relative to `description`.

#### *Languages*

`af, ar, az, bg, bn, de, el, en, es, et, eu, fa, fi, fr, gu, he, hi, ht, hu, id,
it, ja, jv, ka, kk, ko, lt, ml, mr, ms, my, nl, pa, pl, pt, qu, ro, ru, sw, ta,
te, th, tl, tr, uk, ur, vi, wo, yo, zh`.

These 50 languages were chosen to match the set of languages appearing in
XTREME-R benchmark suite. The description for each entity in the candidate set
was randomly selected from the available languages. Note, `zh` is limited to
Simplified Chinese script.

#### *Statistics*

**Description language** |               | **`af`** | **`ar`** | **`az`** | **`bg`** | **`bn`** | **`de`** | **`el`** | **`en`** | **`es`** | **`et`** | **`eu`** | **`fa`** | **`fi`** | **`fr`** | **`gu`** | **`he`** | **`hi`** | **`ht`** | **`hu`** | **`id`** | **`it`** | **`ja`** | **`jv`** | **`ka`** | **`kk`** | **`ko`** | **`lt`** | **`ml`** | **`mr`** | **`ms`** | **`my`** | **`nl`** | **`pa`** | **`pl`** | **`pt`** | **`qu`** | **`ro`** | **`ru`** | **`sw`** | **`ta`** | **`te`** | **`th`** | **`tl`** | **`tr`** | **`uk`** | **`ur`** | **`vi`** | **`wo`** | **`yo`** | **`zh`**
------------------------ | ------------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | --------
**Entities**             | **1,003,893** | 2,026    | 28,220   | 3,914    | 6,138    | 1,768    | 73,076   | 4,316    | 257,008  | 41,808   | 6,072    | 5,755    | 16,895   | 13,174   | 62,236   | 1,419    | 6,924    | 5,158    | 1,852    | 10,411   | 14,339   | 41,808   | 50,817   | 1,339    | 2,841    | 7,186    | 14,463   | 5,606    | 1,919    | 1,434    | 7,937    | 2,232    | 60,409   | 996      | 37,691   | 26,384   | 412      | 8,236    | 43,192   | 1,344    | 4,864    | 2,369    | 3,550    | 1,841    | 8,326    | 23,277   | 3,355    | 42,423   | 23       | 376      | 34,734

### WikiNews portion

Mentions in WikiNews articles are intended for **evaluation**. Mewsli-X provides
a subset sampled from articles in 11 languages.
An approximately balanced number of examples were sampled for each combination of
language and entity frequency bin, subject to data availability. (The frequency
bins are the same as used for sampling the candidate set.) This portion is
split into dev and test sets based on gold entity QID.

#### *Files & Format*

-   `wikinews_mentions-dev.jsonl`
-   `wikinews_mentions-test.jsonl`

Text files where each line is a JSON-serialized dictionary that covers one
WikiNews article and one or more selected entity mentions in the article.

(Flattened) Key            | Definition
-------------------------- | ----------
**`context`**              | Represents one WikNews article.
`context.document_id`      | A unique identifier for the article.
`context.document_url`     | Article URL.
`context.document_title`   | Article title.
`context.section_title`    | Empty, not used here.
`context.language`         | ISO 639-1 code for article's primary language.
`context.text`             | Full article text. <br> ***Note:*** This field is empty in the released data files and can be faithfully repopulated from publicly available resources by running the provided scripts.
`context.sentence_spans`   | List of (`start`, `end`)-pairs denoting sentences as character offsets relative to `context.text`.
&nbsp;                     | &nbsp;
**`mentions`**             | One more entity mentions appearing in `context`.
`mentions[i].example_id`   | Unique identifier assigned to the entity mention instance.
`mentions[i].entity_id`    | Gold WikiData QID of the mentioned entity.
`mentions[i].mention_span` | Surface mention `text` and (`start`, `end`)-pair, as character offsets relative to `context.text`.

#### *Languages*

`ar, de, en, es, fa, ja, pl, ro, ta, tr, uk`.

#### *Statistics*

&nbsp;                       | **Total**  | **ar** | **de** | **en** | **es** | **fa** | **ja** | **pl** | **ro** | **ta** | **tr** | **uk**
---------------------------- | ---------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------
**`dev`**                    |            |        |        |        |        |        |        |        |        |        |        |
**Articles**                 | **2,618**  | 244    | 316    | 307    | 309    | 52     | 279    | 294    | 119    | 266    | 204    | 228
**Entities**                 | **1,996**  | 185    | 301    | 309    | 279    | 58     | 269    | 247    | 82     | 126    | 168    | 237
**Mentions (overall)**       | **2,991**  | 318    | 326    | 316    | 311    | 72     | 310    | 304    | 145    | 312    | 262    | 315
**Mentions (cross-lingual)** | **2,285**  | 275    | 210    | 214    | 231    | 68     | 177    | 202    | 127    | 306    | 226    | 249
&nbsp;                       |            |        |        |        |        |        |        |        |        |        |        |
**`test`**                   |            |        |        |        |        |        |        |        |        |        |        |
**Articles**                 | **9,608**  | 680    | 1,361  | 1,298  | 1,315  | 167    | 1,115  | 1,343  | 308    | 819    | 590    | 612
**Entities**                 | **9,976**  | 843    | 1,402  | 1,395  | 1,381  | 309    | 1,358  | 1,325  | 394    | 674    | 857    | 1,173
**Mentions (overall)**       | **14,624** | 1,501  | 1,551  | 1,490  | 1,552  | 458    | 1,519  | 1,562  | 672    | 1,567  | 1,215  | 1,537
**Mentions (cross-lingual)** | **10,967** | 1,313  | 1,023  | 1,009  | 1,082  | 416    | 834    | 1,014  | 601    | 1,510  | 1,004  | 1,161

### Wikipedia portion

English Wikipedia examples are intended for **fine-tuning**. Each example
consists of a text fragment surrounding an entity mention, paired with
information about the entity. Both the mention context and the entity
description are in English. This portion is split into a train and dev split
based on gold entity QID.

#### *Files & Format*

-   `wikipedia_pairs-train.jsonl`
-   `wikipedia_pairs-dev.jsonl`

Text files where each line is a JSON-serialized dictionary representing
(contextual mention, entity)-pair:

(Flattened) Key                                | Definition
---------------------------------------------- | ----------
**`contextual_mention`**                       | Represents one mention in the context of a Wikipedia page.
***`contextual_mention.context`***             | Represents a fragment of a Wikipedia page.
\- `contextual_mention.context.document_id`    | WikiData QID associated with the Wikipedia page.
\- `contextual_mention.context.document_url`   | Wikipedia page URL.
\- `contextual_mention.context.document_title` | Wikipedia page title.
\- `contextual_mention.context.section_title`  | Title of the page section that yielded the fragment.
\- `contextual_mention.context.language`       | ISO 639-1 code for article's primary language. ("en")
\- `contextual_mention.context.text`           | Up to five sentences of text surrounding the entity mention.
\- `contextual_mention.context.sentence_spans` | Same semantics as specified for in the WikiNews portion.
***`contextual_mention.mention`***             | Represents the mention span, using the same semantics specified for `mentions[i]` in the WikiNews portion.
\- `contextual_mention.mention.example_id`     | Unique identifier assigned to the entity mention instance.
\- `contextual_mention.mention.entity_id`      | Gold WikiData QID of the mentioned entity.
\- `contextual_mention.mention.mention_span`   | Surface mention `text` and (`start`, `end`)-pair, as character offsets relative to `context.text`.
&nbsp;                                         | &nbsp;
**`entity`**                                   | The mentioned entity, in the same format as detailed above for the candidate set.
\- `entity.entity_id`                          | *ibid.*
\- `entity.description`                        | *ibid.*
\- ...                                         |

#### *Languages*

`en`

#### *Statistics*

&nbsp;        | **`train`** | **`dev`**
------------- | ----------- | ---------
**Documents** | 10,628      | 5,490
**Entities**  | 101,193     | 8,742
**Mentions**  | 167,719     | 14,051

## Distribution

The dataset archive is available for download from:

-   https://storage.googleapis.com/gresearch/mewsli/mewsli-x_20220518_6.zip (291
    MB; md5sum `6d3f137027c7e0146c7aa0b3c21e90f4`)

Archive contents:

```
ar/mentions.tsv
ar/docs.tsv
candidate_set_entities.jsonl
de/mentions.tsv
de/docs.tsv
en/mentions.tsv
en/docs.tsv es/
es/mentions.tsv
es/docs.tsv
fa/mentions.tsv
fa/docs.tsv
ja/mentions.tsv
ja/docs.tsv
pl/mentions.tsv
pl/docs.tsv
ro/mentions.tsv
ro/docs.tsv
ta/mentions.tsv
ta/docs.tsv
tr/mentions.tsv
tr/docs.tsv
uk/mentions.tsv
uk/docs.tsv
wikinews_mentions_no_text-dev.jsonl
wikinews_mentions_no_text-test.jsonl
wikipedia_pairs-dev.jsonl
wikipedia_pairs-train.jsonl
```

The released WikiNews files omit the article text. In order to use the dataset,
you have to repopulate the text from publicly available dumps of WikiNews. This
is all taken care of by the `get-mewsli-x.sh` script.

#### TSV-format

*This section is included for completeness but can be safely ignored when
obtaining the dataset via the `get-mewsli-x.sh` script.*

The zip-archive contains two TSV files per language (`docs.tsv` and
`mentions.tsv`). They can be joined on the `docid` column and have the structure
shown below. These files are redundant with `wikinews_*.jsonl`, but are for use
with the tools that extract and parse WikiNews article text from public dumps.

##### docs.tsv

Column   | Description
-------- | ----------------------------------------------------
docid    | Internal unique identifier for the WikiNews article.
title    | Title of the WikiNews article.
url      | Original URL of the article
language | The article language code (ISO 639-1).
curid    | Stable page ID used in WikiNews.
revid    | The version (revision) of the article that was used.
text_md5 | MD5 hash of the expected clean text.

##### mentions.tsv

Column     | Description
---------- | -----------
docid      | Internal unique identifier for the WikiNews article where mention appears.
example_id | Unique identifier for the mention instance.
mention    | Mention surface string.
position   | 0-indexed starting position of the mention in cleaned article text, in Unicode characters.
length     | Length of the mention, in Unicode characters.
entity_qid | WikiData QID of the mentioned entity.
url        | Original hyperlink taken as evidence for an entity mention.

## Updated Dataset

>>_**NOTE:** New evaluation results on Mewsli-X are **not** directly comparable to those reported in the paper because the dataset required further updates. This does not affect the overall findings of the paper._

While preparing the dataset for release, we found previously undetected quality
issues in the version that was used in the experiments for the XTREME-R paper.

For Mewsli-X evaluations to serve as a useful guide for research, we opted to
fix the issues by re-extracting and refiltering the dataset, accepting that new
results would not be directly comparable to the published numbers.

The improvements include:
- preserving the original text (rather than using legacy pre-tokenization);
- mitigating artefacts from imperfect sentence breaking: the updated version
  provides improved annotations that rely on an in-house sentence breaking model
  or [Spacy](https://spacy.io/), depending on the language and based on manual
  quality assessments;
- fully deduplicating entity descriptions in the candidate set (whereas the old
  version had a few entities with the same description);
- more extensive filtering heuristics to remove lists and tables, and to exclude
  entity mentions appearing in WikiNews
  [datelines](https://en.wikipedia.org/wiki/Dateline).
- more naturally distributed fine-tuning data, by allowing overlap between the
  _Wikipedia examples_ and the candidate set. (In the old
  version, these were defined as disjoint, but we found it heavily biased the
  Wikipedia examples toward rare entities while excluding the majority of common
  entities.) Note that the _WikiNews queries_ remain disjoint from the Wikipedia
  training examples in terms of entity IDs.


### Official Mewsli-X Baselines

The following table has the official evaluation results on the released dataset.
Please see the [XTREME-R codebase](https://github.com/google-research/xtreme)
for evaluation code and scripts for fine-tuning PyTorch implementations of mBERT and XLM-Roberta.

|  | **mBERT** |  | **XLM-R Large** |  |
|:---:|:---:|:---:|:---:|:---:|
| **Language** | **_dev_** | **_test_** | **_dev_** | **_test_** |
| **ar** | 14.00 | 18.46 | 29.96 | 34.46 |
| **de** | 66.32 | 63.97 | 72.70 | 71.42 |
| **en** | 59.27 | 55.21 | 66.97 | 62.67 |
| **es** | 57.01 | 62.33 | 63.77 | 67.39 |
| **fa** | 9.84 | 13.22 | 26.98 | 33.91 |
| **ja** | 47.37 | 48.63 | 50.78 | 52.99 |
| **pl** | 57.62 | 57.50 | 70.89 | 66.83 |
| **ro** | 40.83 | 37.74 | 49.88 | 46.56 |
| **ta** | 7.24 | 5.06 | 26.02 | 24.37 |
| **tr** | 51.01 | 50.17 | 59.63 | 59.92 |
| **uk** | 38.74 | 43.61 | 51.32 | 57.05 |
| **_Macro-average_** | **40.84** | **41.45** | **51.72** | **52.51** |

## Disclaimer

This is not an official Google product.

<!--
--->