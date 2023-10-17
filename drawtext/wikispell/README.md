# WikiSpell

WikiSpell data (training, dev, and test sets) can be found in the `data` directory.

## Data Preparation

This section documents how the WikiSpell data was generated, provided for documentation and reproducibility purposes. It is not necessary to run these commands since the resulting data can be found in the `wikispell/data/` directory of this repo.

### Download Wiktionary dump

```sh
wget https://dumps.wikimedia.org/enwiktionary/20220820/enwiktionary-20220820-pages-articles.xml.bz2
bunzip2 enwiktionary-20220820-pages-articles.xml.bz2
cp enwiktionary-20220820-pages-articles.xml /path/to/wikispell/data/
```

### Extract word lists from the Wiktionary dump

```sh
languages="ar,en,fi,ko,ru,th,zh"

python3 extract_wiki_wordlists.py \
   --wiktionary_dump_path="/path/to/wikispell/data/enwiktionary-20220820-pages-articles.xml" \
   --output_root="/path/to/wikispell/data/" \
   --languages="${languages}"
```

The resulting word list sizes are as follows:

```
ar   51,265
en  874,601
fi  205,177
ko   24,058
ru  395,419
th   14,650
zh  242,138
```

#### Generate the train/dev/test files

```sh
python3 make_wiki_datasets.py \
  --wikispell_root="/path/to/wikispell/data/" \
  --languages="${languages}" \
  --max_char_length=30
```

Outputs:

```
Language: ar
  Filtered out entries with >30 chars: 0/51265 (0.00%)
  Filtered out all-punctuation/symbol entries: 7/51265 (0.01%)
  Number of zero-count words: 11673/51258 (22.77%)
Language: en
  Filtered out entries with >30 chars: 63/874601 (0.01%)
  Filtered out all-punctuation/symbol entries: 187/874601 (0.02%)
  Number of zero-count words: 139934/874351 (16.00%)
Language: fi
  Filtered out entries with >30 chars: 13/205177 (0.01%)
  Filtered out all-punctuation/symbol entries: 2/205177 (0.00%)
  Number of zero-count words: 48093/205162 (23.44%)
Language: ko
  Filtered out entries with >30 chars: 0/24058 (0.00%)
  Filtered out all-punctuation/symbol entries: 4/24058 (0.02%)
  Number of zero-count words: 2765/24054 (11.49%)
Language: ru
  Filtered out entries with >30 chars: 13/395419 (0.00%)
  Filtered out all-punctuation/symbol entries: 5/395419 (0.00%)
  Number of zero-count words: 113940/395401 (28.82%)
Language: th
  Filtered out entries with >30 chars: 11/14650 (0.08%)
  Filtered out all-punctuation/symbol entries: 51/14650 (0.35%)
  Number of zero-count words: 459/14588 (3.15%)
Language: zh
  Filtered out entries with >30 chars: 0/242138 (0.00%)
  Filtered out all-punctuation/symbol entries: 87/242138 (0.04%)
  Number of zero-count words: 59581/242051 (24.62%)
```
