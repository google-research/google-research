## Miscellaneous preprocessing tools

### `bible_epitran_data_main`

This tool processes the multilingual parallel
[corpus](https://github.com/christos-c/bible-corpus) created from translations
of the Bible and generates training data consisting of pronunciation/orthography
pairs using [Epitran](https://github.com/dmort27/epitran) grapheme-to-phoneme
conversion framework.

#### Prerequisites

Following packages need to be present:

*   `absl-py`
*   `epitran`
*   `progress`

#### Examples

```shell
  > python3 -m homophonous_logography.tools.bible_epitran_data_main \
      --input_xml_file ${DIR}/bible-corpus/bibles/Dutch.xml \
      --language_id nld-Latn \
      --test_set_ids_file bible_test_ids.txt \
      --output_data_file dutch_data.tsv
```

```shell
  > python3 -m homophonous_logography.tools.bible_epitran_data_main \
      --input_xml_file ${DIR}/bible-corpus/bibles/Amharic.xml \
      --language_id amh-Ethi \
      --test_set_ids_file bible_test_ids.txt \
      --output_data_file amharic_data.tsv
```

### `parse_swedish_nb_lexicon_main`

Utility for parsing the Swedish pronunciation lexicon and preparation of the
training data.

#### Prerequisites

Package `absl-py` needs to be present in the system.

#### How to use

1.  Download the lexicon:

    ```shell
    > wget http://www.nb.no/sbfil/leksikalske_databaser/leksikon/sv.leksikon.tar.gz
    ```

1.  Uncompress:

    ```shell
    > tar xvfz sv.leksikon.tar.gz
    ```

1.  Change directory to the opened archive:

    ```shell
    > cd NST\ svensk\ leksikon/swe030224NST.pron
    ```

1.  Convert from ASCII to UTF-8:

    ```shell
    > iconv -f ISO-8859-1 -t UTF-8 swe030224NST.pron -o ${LEXICON}
    ```

1.  Run this parser:

    ```shell
    > LEXICON_SUBDIR=NST\ svensk\ leksikon/swe030224NST.pron
    > python3 parse_swedish_nb_lexicon.py \
        --input_lexicon_file ${SWEDISH}/${LEXICON_SUBDIR}/lexicon.txt \
        --output_wikipron_lexicon swedish_lexicon.txt
    ```
