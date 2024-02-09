# CANN: Constrained Approximate Nearest Neighbors

CANN ([ICCV'23 paper, ArXiv version](https://arxiv.org/abs/2306.09012)) is a
novel method for local feature-based visual localization.

**Citation**

```
@inproceedings{aiger2023iccv,
title={{Yes, we CANN: Constrained Approximate Nearest Neighbors for local feature-based visual localization}},
author={Dror Aiger and Andre Araujo and Simon Lynen},
booktitle={ICCV},
year={2023},
}
```

## Instructions to run the code

The code runs on any two sets of features d-points extracted from images and stored as a binary Eigen matrix per image where each row contains one d-dimensional vector. The number of features is arbitrary. Both index and queries data have the same format.
In this example we use FIRE features [1] but any d-dimensional vectors can be used and we assume that features were already extracted.

Download sample data from Drive: https://drive.google.com/file/d/1PHP1mEvPIJb9ZC5DfWU3pH0_ipck12Bl/view?usp=drive_link
Extract the tar in the root folder (cann). It creates the data folder '''testdata'''.
The data contains a small set for sanity check from and a large one from Baidu-Mall (one of the datasets we tested in the paper).

We provide the script `./run.sh` to run index/query search end-to-end for the data in the folder testdata. 

```bash
./run.sh
```

Our framework is an efficient indexing and retrieval scheme, independent of the local feature type. It can use ant d-points as feature vectors.
We provide data using FIRE [1] features as an example. 
We provide a small sanity test from Mapillary (https://help.mapillary.com/hc/en-us, shared under a CC-BY-SA license: https://creativecommons.org/licenses/by-sa/4.0/#) and also the Baidu-Mall dataset [3] used in the paper for verification. Only FIRE [1] descriptors are provided, not images.

In the test data we have 10 index files and 5 query files (corresponding to feature vectors extracted from images of the index and queries).

The required parameters (see run.sh, main/run_main.sh and their corresponding data) and a minimal run uses the commands:


```bash
bazel build -c opt main:colored_c_nn_random_grids_index_main
bazel-bin/main/colored_c_nn_random_grids_index_main \
  --index_descriptor_files <list of feature files> \
  --query_descriptor_files <list of feature files> \
  --pairs_file <output file name>
'''

See other parameters in the code and in the paper.


### Output

The output of the index search stage is a CSV file (called the pairs file) containing three columns: the query filename, the index filename, and the score. The file contains, for each query feature set, the top 50 matches in the index, along with their match score (a value between 0 and 1). For example, the first few lines in a pairs file might look something like:

```
gs://tmp_features/testdata/1000178804347752.jpg.desc, gs://tmp_features/testdata/1000556171133200.jpg.desc, 0.3022350
gs://tmp_features/testdata/1000178804347752.jpg.desc, gs://tmp_features/testdata/1000750897599830.jpg.desc, 0.2982784
gs://tmp_features/testdata/1000178804347752.jpg.desc, gs://tmp_features/testdata/1000582304626409.jpg.desc, 0.2845915
gs://tmp_features/testdata/1000178804347752.jpg.desc, gs://tmp_features/testdata/1000878214261175.jpg.desc, 0.2830647
```

To test baidu-mall dataset [3] (see the paper) and evaluate the results, you need to install the kapture framework[2]: https://github.com/naver/kapture-localization/blob/main/doc/benchmark.adoc
(note that the download above should be done first to obtain the actual dataset)

Then run:

```bash
main/run_main_baidu.sh
```
This may take some time depending on the number of cores you have.
To prepare the output to Kapture, run:

```bash
sed 's/testdata\/baidu_descriptors\///g' /tmp/baidu_pairs.txt | sed 's/\.desc//g' > /tmp/baidu_pairs_format.txt
```
(This removes folder names from the prefix).

The evaluation from Kapture (EWB approximation) can be applied on the resulting pair file (in this case, it is /tmp/baidu_pairs_format.txt, see the script).
the Output should look like:

==============================================================================
Custom feature matching
==============================================================================

Elapsed time: 0.001 [minutes]
Model: EWB

Found 2292 / 2292 image positions (100.00 %).
Found 2292 / 2292 image rotations (100.00 %).
Localized images: mean=(20.5261m, 40.2932 deg) / median=(5.9039m, 26.6976 deg)
All: median=(5.9039m, 26.6976 deg)
Min: 0.0786m; 1.1547 deg
Max: 246.4923m; 179.0833 deg

(0.25m, 2.0 deg): 0.00%
(0.5m, 5.0 deg): 0.13%
(5.0m, 10.0 deg): 12.30%

Model: CSI

Found 2292 / 2292 image positions (100.00 %).
Found 2292 / 2292 image rotations (100.00 %).
Localized images: mean=(20.5261m, 40.2932 deg) / median=(5.9039m, 26.6976 deg)
All: median=(5.9039m, 26.6976 deg)
Min: 0.0786m; 1.1547 deg
Max: 246.4923m; 179.0833 deg

(0.25m, 2.0 deg): 0.00%
(0.5m, 5.0 deg): 0.13%
(5.0m, 10.0 deg): 12.30%


\begin{tabular}{llll}
\hline
     & (0.25, 2.0)   & (0.5, 5.0)   & (5.0, 10.0)   \\
\hline
 EWB & 0.00\%         & 0.13\%        & 12.30\%        \\
 CSI & 0.00\%         & 0.13\%        & 12.30\%        \\
\hline
\end{tabular}

     (0.25, 2.0)    (0.5, 5.0)    (5.0, 10.0)
---  -------------  ------------  -------------
EWB  0.00%          0.13%         12.30%
CSI  0.00%          0.13%         12.30%

NOTE: The numbers are slightly different than in the paper due to code modifications. We only release the second version CANN_RG and not CANN_RS.

## Contact

If you have any questions, please reach out to Dror Aiger (aigerd@google.com).

[1] @inproceedings{superfeatures,
  title={{Learning Super-Features for Image Retrieval}},
  author={{Weinzaepfel, Philippe and Lucas, Thomas and Larlus, Diane and Kalantidis, Yannis}},
  booktitle={{ICLR}},
  year={2022}
}

[2] @article{humenberger2022investigating,
  title={Investigating the Role of Image Retrieval for Visual Localization},
  author={Humenberger, Martin and Cabon, Yohann and Pion, No{\'e} and Weinzaepfel, Philippe and Lee, Donghwan and Gu{\'e}rin, Nicolas and Sattler, Torsten and Csurka, Gabriela},
  journal={International Journal of Computer Vision},
  year={2022},
  publisher={Springer}
}

[3] @InProceedings{Sun_2017_CVPR,
author = {Sun, Xun and Xie, Yuanfan and Luo, Pei and Wang, Liang},
title = {A Dataset for Benchmarking Image-Based Localization},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {July},
year = {2017}
}