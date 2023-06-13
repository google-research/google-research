# YouTube-News-Timeline: Video Timeline Modeling For News Story Understanding

## Introduction

We present a novel problem, namely **video timeline modeling**. Our objective is to create a video-associated timeline from a set of videos related to a specific topic, thereby facilitating the content and structure understanding of the story being told. This problem has significant potential in various real-world applications, such as news story summarization. To bootstrap research in this area, we curate a realistic benchmark dataset, **YouTube-News-Timeline**


![](https://github.com/google-research/google-research/tree/master/video_timeline_modeling/vtm.png)


## YouTube-News-Timeline Dataset

YouTube-News-Timeline consists of over 12k timelines and 300k YouTube news videos. The duration of these videos ranges from 3 seconds to 12 hours, and their average duration is around 10 minutes. We randomly split the timelines into training, validation, and testing subsets. The number of timelines, timeline nodes, and videos on training/validation/testing split in the final dataset are summarized in the table below. 

| # Timelines | # Nodes | # Videos|
| -------- | -------- | -------- |
|9936/1255/1220|74886/9325/9171|242685/30369/29930|

In the following, we show the distributions of the number of videos per node, the number of nodes per timeline, and the number of videos per timeline in the training, validation, and testing subsets.

![](https://github.com/google-research/google-research/tree/master/video_timeline_modeling/data_dist.png)

The dataset is available in the [`YouTube-News-Timeline.json`](https://github.com/google-research/google-research/tree/master/video_timeline_modeling/YouTube-News-Timeline.json) file. Each data sample is organized in the following format.

```json
{"http://www.cnn.com/2010/WORLD/asiapcf/08/13/myanmar.elections.timeline/index.html":  // The URL link of the webpage where we crawl the timeline.
    [
        ["NSe5aPNZm8c",  
        "6FnSK3GtsS8",
        "E4NjDAKLxo4",
        "G0L6g7imiXc"],
        ["K7Zh5tYyjz4",
        "YcjdUDdgqIM"],
        ["G62y6enNVh0",
        "Q8YFguRiuy0",
        "uDJ9NqrQ5Xg"],
        ["6ztk2ZqkHRc"],
        ["gDK0sKzI3UE"]
    ] /* The URL links of the retrieved YouTube news videos. Each list in the nested list corresponds to one node on the timeline. These nodes are ordered in the nested list. */
    
}
```

## Note
We are working on organizing the code to run baseline models. Will update it once it is ready.

## Citation 

Add it later

## Disclaimer
The reference timelines used to construct the dataset are crawled from the web, and the videos are sourced from YouTube. The opinions expressed in the timelines and videos do not necessarily reflect our own, and we do not endorse or promote any specific viewpoint. The dataset is intended for research and educational purposes only, and users should exercise their own judgment when interpreting and using it.
