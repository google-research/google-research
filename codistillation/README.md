# Codistillation

The directory contains the publicly available material for the codistillation paper cited below.

## Common Crawl Paragraph IDs

During the experimentation, we used a subset of the [Common Crawl Dataset](http://commoncrawl.org/the-data).
The list of actual paragraph IDs we used available in the following file:
[common_crawl_paragraph_ids_for_codistillation-csv-delta.tar.bz2](https://storage.cloud.google.com/codistillation-common-crawl-paragraph-ids/common_crawl_paragraph_ids_for_codistillation-csv-delta.tar.bz2) (~392MB).
The file is a `bzip2` compressed `tar` containing `CSV` files.
Each such a file has two column: `file name` and `encoded_paragraph_list`.
The column `encoded_paragraph_list` contains a list of paragraph IDs used from
the given file in [delta encoding](https://en.wikipedia.org/wiki/Delta_encoding).

The following simple decoder can be used to get the final set of IDs per file:

```
void decode(std::vector<int64>& ids) {
  int64 last = 0;
  for (int i = 0; i < ids.size(); i++) {
    int64 delta = ids[i];
    ids[i] = delta + last;
    last = ids[i];
  }
}
```

The checksum of the file (`sha256sum`) is as follows:

```
2694ae96685810d2a5c81f2dc9b3d2956d0543fa226ccbe9b26f0ffd81fa8292  common_crawl_paragraph_ids_for_codistillation-csv-delta.tar.bz2
```


If you use any of the material here please cite the following paper:

```
@inproceedings{codistillation,
  title = {Large scale distributed neural network training through online distillation},
  author = {Rohan Anil and Gabriel Pereyra and Alexandre Passos and Robert Ormandi and George E. Dahl and Geoffrey E. Hinton},
  booktitle = {International Conference on Learning Representations},
  year = {2018},
  url = {https://openreview.net/forum?id=rkr1UDeC-},
}
```

