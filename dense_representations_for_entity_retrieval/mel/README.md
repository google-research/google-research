# Entity Linking in 100 Languages

Resources accompanying the paper
_[Entity Linking in 100 Languages](https://www.aclweb.org/anthology/2020.emnlp-main.630)_
by
[Jan A. Botha](https://research.google/people/JanBotha/),
[Zifei Shan](http://www.zifeishan.org/) and
[Daniel Gillick](https://research.google/people/DanGillick/).

**To get the data, see section [Get Mewsli-9 Dataset](#get-mewsli-9-dataset).**

If you use the code/resources or discuss this topic in your work, please cite our paper:

```
@inproceedings{botha-etal-2020-entity,
    title = "{E}ntity {L}inking in 100 {L}anguages",
    author = "Botha, Jan A. and Shan, Zifei  and Gillick, Daniel",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.630",
    pages = "7833--7845",
}
```

## Mewsli-9 Dataset

Mewsli-9 is a dataset of entity mentions linked to
[WikiData](https://www.wikidata.org/), extracted from
[WikiNews](https://www.wikinews.org/) articles.
(The name is short for "**M**ultilingual Entities in N**ews**, **li**nked".)

The dataset covers 9 diverse languages, 5 language families and 6 writing
systems. It features many WikiData entities that do not appear in English
Wikipedia, thereby incentivizing research into multilingual entity linking
against WikiData at-large.

The dataset is released in the form of "dataset descriptors", plus code to
reproduce the article text from publicly available data sources.

### Statistics

| Language | Code | Docs  | Mentions | Unique Entities | Entities outside En-Wiki |
|----------|------|-------|----------|-----------------|--------------------------|
| Japanese | ja   | 3410  | 34463    | 13663           | 3384                     |
| German   | de   | 13703 | 65592    | 23086           | 3054                     |
| Spanish  | es   | 10284 | 56716    | 22077           | 1805                     |
| Arabic   | ar   | 1468  | 7367     | 2232            | 141                      |
| Serbian  | sr   | 15011 | 35669    | 4332            | 269                      |
| Turkish  | tr   | 997   | 5811     | 2630            | 157                      |
| Persian  | fa   | 165   | 535      | 385             | 12                       |
| Tamil    | ta   | 1000  | 2692     | 1041            | 20                       |
| English  | en   | 12679 | 80242    | 38697           | 14                       |
| Total    |      | 58717 | 289087   | 82162           | 8807                     |

### Schema

The dataset descriptor archive is available for download from:

- https://storage.googleapis.com/gresearch/mewsli/mewsli-9.zip (9.7 MB)

The script below will take care of downloading this.

The archive contains two TSV files per language (`docs.tsv` and `mentions.tsv`).
They can be joined on the `docid` column and have the structure shown below.

#### docs.tsv

|  Column  | Description                                          |
|----------|------------------------------------------------------|
| docid    | Internal unique identifier for the WikiNews article. |
| title    | Title of the WikiNews article.                       |
| curid    | Stable page ID used in WikiNews.                     |
| revid    | The version (revision) of the article that was used. |
| url      | Original URL of the article                          |
| text_md5 | MD5 hash of the expected clean text.                 |

#### mentions.tsv

| Column      | Description                                                                                |
|-------------|--------------------------------------------------------------------------------------------|
| docid       | Internal unique identifier for the WikiNews article where mention appears.                 |
| position    | 0-indexed starting position of the mention in cleaned article text, in Unicode characters. |
| length      | Length of the mention, in Unicode characters.                                              |
| mention     | Mention string.                                                                            |
| url         | Original hyperlink taken as evidence for an entity mention.                                |
| lang        | Language of the article where mention appears (ISO-639-1 code).                            |
| qid         | WikiData QID of the mentioned entity.                                                      |
| qid_in_refs | Whether the entity had an entry in English Wikipedia ("True" or "False")                   |
| freq_bin    | Frequency bin index (0, ..., 5) for the entity                                             |

Frequency bin index (`freq_bin`) refers to how many times an entity appeared in
the particular training set used for the paper:

- 0 -> 0 times (zero-shot)
- 1 -> 1 to 9 times
- 2 -> 10 to 99 times
- 3 -> 100 to 999 times
- 4 -> 1000 to 9,999 times
- 5 -> 10,000 or more times

#### Cleaned Article Text

For replication purposes, clean article text is extracted from the 2019-01-01
dump of WikiNews, which is available from archive.org, and pre-configured in the
script below.

### Get Mewsli-9 Dataset

**Export the relevant part of the code repository:**

```
SUBDIR=dense_representations_for_entity_retrieval
svn export https://github.com/google-research/google-research/trunk/$SUBDIR
```

**Change into the `mel/` subdirectory:**

```
cd dense_representations_for_entity_retrieval/mel
```

**Run the end-to-end script:**

```
bash get-mewsli-9.sh
```

This script runs in about 5 minutes, depending on the computing environment. It
downloads about 125MB of input data, installs a few dependencies and extracts
the clean article text that serve as mention contexts.

**Look for output under `./mewsli-9/output/dataset`.**

For example, the cleaned text for an Spanish article with docid `es-44` would be
in the file `./mewsli-9/output/dataset/en/text/es-44`.

### Using the dataset

TODO: add example.

### Dependencies

The helper scripts assume the following tools are installed:

- Python 3
- bzip2
- git
- wget
- md5sum
- pip
- virtualenv
- zip

See `wikinews_extractor/requirements.txt` for dependencies that will be
installed to a new virtual environment.

## Disclaimer

This is not an official Google product.
