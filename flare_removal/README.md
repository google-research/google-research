# How to Train Neural Networks for Flare Removal

This repository contains code that accompanies the following paper:

> Yicheng Wu, Qiurui He, Tianfan Xue, Rahul Garg, Jiawen Chen, Ashok
> Veeraraghavan, and Jonathan T. Barron. **How to train neural networks for
> flare removal**. *Proceedings of the IEEE/CVF International Conference on
> Computer Vision (ICCV)*, 2021.

-   The paper (including the supplemental materials) is available on
    [arXiv](https://arxiv.org/abs/2011.12485) as well as
    [CVF Open Access](https://openaccess.thecvf.com/content/ICCV2021/html/Wu_How_To_Train_Neural_Networks_for_Flare_Removal_ICCV_2021_paper.html).

-   The [main project page](https://yichengwu.github.io/flare-removal/) contains
    more information, including a recorded presentation.

## Announcements

-   **1/30/2022:** It has been brought to our attention that there might be an
    issue with the training code that causes the trained model to perform worse
    than what we show on the
    [test images](https://drive.google.com/corp/drive/folders/1_gi3W8fOEusfmglJdiKCUwk3IA7B2jfQ).
    This issue was likely introduced when we cleaned up the repository prior to
    open-sourcing. We are actively investigating this issue, and will submit a
    patch to this repository as soon as possible. The issue does not affect the
    testing script (`remove_flare.py`). We can also confirm that our published
    results (both quantitative and qualitative) are accurate and reproducible
    using an older (internal) version of the code.

## Dataset

### Flare-only images

A total of 5,001 RGB flare images are
[released](https://research.google/tools/datasets/lens-flare/) via Google
Research's public dataset repository under the
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license. Among them:

-   2,001 are from lab captures (1,001 captures + interpolation between frames).
    These images are placed under the `captured` subdirectory.

-   3,000 are simulated computationally. These images are placed under the
    `simulated` subdirectory.

To obtain this data:

1.  Install [Google Cloud SDK](https://cloud.google.com/sdk/docs/quickstart).
    This should automatically install the `gsutil` tool which is required to
    access the Google Cloud Storage bucket.

2.  Run the following command:

    ```shell
    $ gsutil cp -r gs://gresearch/lens-flare /your/local/path
    ```

### Flare-free (scene) images

We use the same image dataset as
[*Single Image Reflection Removal with Perceptual Losses*](https://people.eecs.berkeley.edu/~cecilia77/project-pages/reflection.html)
(Zhang et al., CVPR 2018). Please follow
[their instructions](https://github.com/ceciliavision/perceptual-reflection-removal#dataset)
to access this data. Note that we do *not* make the distinction between the
*reflection layer* and the *transmission layer* - we shuffle the entire dataset
and treat it as a unified set of natural images. You may want to make an
appropriate train-test split before using this dataset.

## Code

### Synthesizing scattering flare

The code for synthesizing random scattering flare ("streaks") is written in
Matlab and located under the `matlab` directory. Simply execute the `main.m`
script to reproduce our results.

By default, it writes to the following directories:

-   **`matlab/apertures`**: Simulated defective apertures with dots (resembling
    dust) and polylines (resembling scratches).

-   **`matlab/streaks`**: Flare patterns resulting from the simulated defective
    apertures above. Multiple flare patterns are generated for each aperture,
    accounting for varying light source locations, defocus, and distortion.
    These images are used to further synthesize flare-corrupted photographs.

### Training a flare removal model

**WARNING:** Commands below are executed from the repository root
`google_research/`. Otherwise, Python may not be able to resolve the module
paths correctly.

The training and testing programs require certain dependencies (see
`requirements.txt`). You may create a
[virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)
and install these dependencies using `pip`, as demonstrated in `run.sh`. Note
that running `flare_removal/run.sh` directly will fail due to missing arguments
(see [below](#testing-the-model-on-images) for details), but will at least
install the correct dependencies.

The training script is `python/train.py`. A separate evaluation script
`python/evaluate.py` is also available, so an additional job can be started to
monitor the training progress in parallel (optional).

```shell
$ python3 -m flare_removal.python.train \
  --train_dir=/path/to/training/logs/dir \
  --scene_dir=/path/to/flare-free/training/image/dir \
  --flare_dir=/path/to/flare-only/image/dir

# Optional.
$ python3 -m flare_removal.python.evaluate \
  --eval_dir=/path/to/evaluation/logs/dir \
  --train_dir=/path/to/training/logs/dir \
  --scene_dir=/path/to/flare-free/evaluation/image/dir \
  --flare_dir=/path/to/flare-only/image/dir
```

A few notes on the arguments:

-   **`--train_dir`/`--eval_dir`**: This is where all training/evaluation states
    are preserved, including metrics, summary images, and model weight
    checkpoints. When a training job restarts, it will also try to pick up from
    its last state from this directory. Hence, you should use a new (empty)
    directory for each new experiment.

-   **`--scene_dir`**: Parent directory of all flare-free images. You may use
    any natural image dataset, as long as all images are RGB and have the same
    size. See [above](#flare-free-scene-images) for an example. Note that the
    scene images used for `evaluate.py` should be different from those for
    `train.py`.

-   **`--flare_dir`**: Parent directory of all flare-only images. If you
    downloaded our dataset using the instructions [above](#flare-only-images),
    the argument should be `--flare_dir=/your/local/path/lens-flare`.

-   **Other arguments** are provided to further customize the training
    configuration, e.g., alternative forms of input data, hyperparameters, etc.
    Please refer to the source code for additional documentation.

The training job will write the following contents to disk, under
`path/to/training/logs/dir`:

-   **`model/`**: latest model files.

-   **`summary/`**: training metrics and summary images, to be visualized using
    [TensorBoard](https://www.tensorflow.org/tensorboard).

-   **`ckpt-*`**: model checkpoints, for restoration of previous model weights

### Testing the model on images

We also provide a Python script to test a trained model on images in the wild.
Suppose you have followed the steps above to train a flare removal model, you
could invoke the testing script as follows:

```shell
$ python3 -m flare_removal.python.remove_flare \
  --ckpt=/path/to/training/logs/dir/model \
  --input_dir=/path/to/test/image/dir \
  --out_dir=/path/to/output/dir
```

The `--ckpt` argument locates the model directory saved by the training script.
The other arguments are self-explanatory. For more details, including additional
arguments, please refer to the source file.

## Pre-trained model

Unfortunately, due to licensing constraints, we cannot release the pre-trained
model. However, you should be able to reproduce our results using the code and
datasets described above.

## Citation

If you find this work useful, please cite:

```
@InProceedings{flareremvoal2021,
  author    = {Wu, Yicheng and He, Qiurui and Xue, Tianfan and Garg, Rahul and
               Chen, Jiawen and Veeraraghavan, Ashok and Barron, Jonathan T.},
  title     = {How To Train Neural Networks for Flare Removal},
  booktitle = {Proceedings of the IEEE/CVF International Conference on
               Computer Vision (ICCV)},
  month     = {October},
  year      = {2021},
  pages     = {2239-2247}
}
```
