# R4R: Instruction and Path Composition for VLN

[Room-to-Room](https://bringmeaspoon.org/) (R2R) is a pioneering dataset in
visually-grounded natural language navigation with photo-realistic environments
([Anderson et al., 2018](https://arxiv.org/abs/1711.07280)). R2R consists of an
environment and language instructions paired to reference paths. Due to the
process by which the data are generated, all R2R reference paths are
shortest-to-goal paths by construction. As such, they capture only a small
subset of the richness of navigation.

To address the lack of variety in path configurations, we propose a simple yet
effective data augmentation strategy that increases the number of training
examples and introduces paths that twist and turn, without additional human or
low-fidelity machine annotations Quite simply, the existing paths in the
dataset can be extended by joining them with other paths that start within some
threshold dth of where they end. We name this the Room-for-Room (R4R) dataset.

For further details, see the accompanying paper:
[Stay on the Path: Instruction Fidelity in Vision-and-Language Navigation](https://arxiv.org/abs/1905.12255)

## Documentation

The R4R dataset is created by joining together paths in the R2R dataset, for
which the first path ends within a thresholded distance from the start of the
second. We do not distribute the original R2R data here, and instead provide
code that constructs R4R from it. The original R2R data can be downloaded
[here](https://niessner.github.io/Matterport/).

Example usage:

```
python r4r_generate_data.py \
  --input_file_path="path/to/R2R_train.json" \
  --output_file_path="path/to/R4R_train.json" \
  --connections_dir="path/to/connections" \
  --scans_dir="path/to/data/v1/scans" \
  --distance_threshold="3.0"
```

Command line arguments for `r4r_generate_data.py`:

*   `--output_file_path`: Path to the R4R data JSON file you are generating.
*   `--input_file_path`: Path to the original R2R data JSON file, which can be
    downloaded
    [here](https://github.com/peteanderson80/Matterport3DSimulator/blob/master/tasks/R2R/data/download.sh).
*   `--connections_dir`: Path to a directory containing graph connectivity
    files, which can be downloaded
    [here](https://github.com/peteanderson80/Matterport3DSimulator/tree/master/connectivity).
*   `--scans_dir`: Path to the Matterport simulator directory, which can be
    downloaded [here](https://niessner.github.io/Matterport/).
*   `--distance_threshold`: The maximum shortest-path distance between the final
    node of first path and the first node of the second path for the two paths
    to be joined. Conventionaly this is 3.0 meters
    ([Anderson et al., 2018](https://arxiv.org/abs/1711.07280)).
*   `--heading_threshold`: The maximum absolute heading angle difference in
    radians between the final connection of first path and the initial heading
    of the second path for the two paths to be joined. Conventionaly this check
    is disabled.

Running this script on the standard R2R training and validation data with a
distance threshold of 3.0 meters and no heading threshold:

```
### R2R_train.json

******Final Results********
  Total instructions generated:    233613
  Average path distance (meters):  20.5901583255
  Average shortest path distance:  10.5022469844
  Average path length (steps):     12.0681064404
  Average shortest path length:    6.4874662553
  Total paths generated:           25930
  Total distance filtered paths:   381581
  Total heading filtered paths:    0

### R2R_val_seen.json

******Final Results********
  Total instructions generated:    1035
  Average path distance (meters):  20.3605171182
  Average shortest path distance:  11.1137253455
  Average path length (steps):     12.2173913043
  Average shortest path length:    7.0
  Total paths generated:           115
  Total distance filtered paths:   2269
  Total heading filtered paths:    0

### R2R_val_unseen.json

******Final Results********
  Total instructions generated:    45162
  Average path distance (meters):  20.222094624
  Average shortest path distance:  10.057187751
  Average path length (steps):     12.147070546
  Average shortest path length:    6.40294938222
  Total paths generated:           5018
  Total distance filtered paths:   63401
  Total heading filtered paths:    0
```

Note: this script requires NetworkX and was tested on version 2.3.

## Reference

If you use or discuss this dataset in your work, please cite our paper:

```
@InProceedings{sotp2019acl,
  title = {{Stay on the Path: Instruction Fidelity in Vision-and-Language Navigation}},
  author = {Jain, Vihan and Magalhaes, Gabriel and Ku, Alexander and Vaswani, Ashish and Ie, Eugene and Baldridge, Jason},
  booktitle = {Proc. of ACL},
  year = {2019}
}
```

## Contact

If you have a technical question regarding the dataset or publication, please
create an issue in this repository.
