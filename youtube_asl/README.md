## YouTube-ASL

This repository contains information about the YouTube-ASL dataset, a
large-scale, open-domain dataset of American Sign Language videos with English
captions.

### Process

Creating this dataset involved a two step process.

First, we mined YouTube for relevant ASL videos, searching through
machine-generated tags of Knowledge Graph entities that are attached to each
video. We retrieved listed public videos tagged as being related to sign
language generally or American Sign Language specifically, up to January 2022.

Second, we had native Deaf annotators go through these videos.
This was done to filter out videos with poor quality or misaligned captions.

### Dataset Characteristics

This dataset is composed of 11,093 ASL videos with 984 total hours of footage
and 610,193 English captions.

### Video IDs

As with the similar YouTube-8M dataset, we are releasing the video IDs.

These IDs can be found here
[this link](https://console.cloud.google.com/storage/browser/gresearch/youtube-asl).

### Citing YouTube-ASL

If you use YouTube-ASL, please cite our associated paper:

```
@misc{uthus2023youtubeasl,
  author = {Uthus, David and Tanzer, Garrett and Georg, Manfred},
  title = {YouTube-ASL: A Large-Scale, Open-Domain American Sign Language-English Parallel Corpus},
  year = {2023},
  eprint = {2306.15162},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2306.15162},
}
```
