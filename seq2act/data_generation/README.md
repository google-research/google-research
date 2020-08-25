# Introduction

Seq2act contains 3 datasets. Please follow the instructions below to generate the AndroidHowTo and RicoSCA dataset, and download the PixelHelp dataset.

Note: The original datasets can not be released due to the copyright. The
AndroidHowTo and PixelHelp datasets released here are re-created from a public
source for opensource purposes. They are different from the original dataset
that the paper was based on thus result in slightly different performance.

## Generate AndroidHowTo Dataset

We extracted phone related instructions from CommonCrawl Dataset.

**Download CommonCrawl WARC Files**

Download WARC files from [https://commoncrawl.org/2020/04/march-april-2020-crawl-archive-now-available/](https://commoncrawl.org/2020/04/march-april-2020-crawl-archive-now-available/) according to the manual.
Note: The `warc.paths.gz` file from the download contains 56000 files but only
3414 of them are used in our pipeline. The whitelist is provided here in file `used_warc.paths`. The WARC files are in the format of `warc.gz`. Each WARC is 1~2GB and the total size is more than several TBs, so you may need follow the manual of CommonCrawl or use your own tool to speed up the download process.

Place the downloaded WARC files under a created folder `seq2act/data/android_howto/warc/`.
If you put them in a customized path, please change the param `--input_warc_dir` in `crawl_instructions.sh` accordingly.


**Extract Instructions**

Using the downloaded WARC files to extract instructions:

```
sh seq2act/data_generation/crawl_instructions.sh
```

`crawled_instructions.json` is then generated with each line as a Json string containing one instruction.


**Download Annotation File**

From [https://github.com/google-research-datasets/seq2act/tree/master/data/android_howto](https://github.com/google-research-datasets/seq2act/tree/master/data/android_howto) download `common_crawl_annotation.csv` and put in `seq2act/data/android_howto/`, this file contains human annotations for each instruction.

Note that **not** all instructions from `crawled_instructions.json` are annotated. Instructions that are too long (>200 words) are excluded. Instructions with same content are deduped.

**Generate AndroidHowTo tfrecord**

With instructions/annotations ready, now we can generate dataset in `tfrecord` format.

```
sh seq2act/data_generation/create_android_howto.sh
```


## Generate RicoSCA Dataset

**Download Rico Public Dataset**

```
# Download dataset from http://interactionmining.org/rico#quick-downloads
# Choose 1 UI Screenshots and View Hierarchies (6 GB)
# Place the downloaded Rico .json data under folder seq2act/data/rico_sca/raw
```

**Generate RicoSCA tfrecord**

```
sh seq2act/data_generation/create_rico_sca.sh
```


## Download PixelHelp Dataset

The PixelHelp dataset is ready for download from: [https://github.com/google-research-datasets/seq2act/tree/master/data/pixel_help](https://github.com/google-research-datasets/seq2act/tree/master/data/pixel_help). Put the data in `./seq2act/data/pixel_help/`
