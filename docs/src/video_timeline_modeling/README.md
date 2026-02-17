# YouTube-News-Timeline: Video Timeline Modeling For News Story Understanding

## Introduction

We present a novel problem, namely **video timeline modeling**. Our objective is to create a video-associated timeline from a set of videos related to a specific topic, thereby facilitating the content and structure understanding of the story being told. This problem has significant potential in various real-world applications, such as news story summarization. To bootstrap research in this area, we curate a realistic benchmark dataset, **YouTube-News-Timeline**.

For more details please check our paper: https://arxiv.org/abs/2309.13446

![](https://github.com/google-research/google-research/blob/master/video_timeline_modeling/vtm.png)


## YouTube-News-Timeline Dataset

YouTube-News-Timeline consists of over 12k timelines and 300k YouTube news videos. The duration of these videos ranges from 3 seconds to 12 hours, and their average duration is around 10 minutes. We randomly split the timelines into training, validation, and testing subsets. The number of timelines, timeline nodes, and videos on training/validation/testing split in the final dataset are summarized in the table below. 

| # Timelines | # Nodes | # Videos|
| -------- | -------- | -------- |
|9936/1255/1220|74886/9325/9171|242685/30369/29930|

In the following, we show the distributions of the number of videos per node, the number of nodes per timeline, and the number of videos per timeline in the training, validation, and testing subsets.

![](https://github.com/google-research/google-research/blob/master/video_timeline_modeling/data_dist.png)

The dataset is available via [this Google Drive link](https://drive.google.com/drive/folders/1SChGxFb_Vl58Nn8jKOKTyoofxu6hz7tF?usp=sharing). Each data sample is organized in the following format.

```json
{
     "https://apnews.com/article/japan-accidents-tsunamis-earthquakes-42d4947609becd7f141e9524a8c98937":  // The URL link of the webpage where we crawl the timeline.
     [
      [
        "OhEbGK4PnZg",
        "cl19tfn33hI",
        "R0l6z0HaUAM",
        "5QhCsR-t-qM",
        "ev3FBIoHMX8"
      ],
      [
        "psAuFr8Xeqs",
        "BsRd7WQuBHc",
        "Dp_8rLL1Y18",
        "h1m7GFPAq3o"
      ],
      [
        "f4TaKPKe1gg",
        "DLlsKd-QC2o"
      ],
      [
        "ocluW1Vhvcg",
        "vusthiUFx_0",
        "vGHzuZQLYtg",
        "7XpLbhQxpLw",
        "UsPFUzXisq4"
      ],
      [
        "hA3fNK0rxcs"
      ]
    ] // The URL links of the retrieved YouTube news videos. Each list in the nested list corresponds to one node on the timeline. These nodes are ordered in the nested list.
}
```
Due to data privacy concern, we periodically refresh our dataset to remove invalid YouTube videos thus the exact size of our dataset may change slightly.


## Citation 
```
@inproceedings{
  liu2023video,
  title={Video Timeline Modeling For News Story Understanding},
  author={Liu, Meng and Zhang, Mingda and Liu, Jialu and Dai, Hanjun and Yang, Ming-Hsuan and Ji, Shuiwang and Feng, Zheyun and Gong, Boqing},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2023}
}
```

## Disclaimer
The reference timelines used to construct the dataset are crawled from the web, and the videos are sourced from YouTube. The opinions expressed in the timelines and videos do not necessarily reflect our own, and we do not endorse or promote any specific viewpoint. The dataset is intended for research and educational purposes only, and users should exercise their own judgment when interpreting and using it.
