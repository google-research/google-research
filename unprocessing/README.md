# Unprocessing Images for Learned Raw Denoising

Reference code for the paper [Unprocessing Images for Learned Raw Denoising](http://timothybrooks.com/tech/unprocessing).
Tim Brooks, Ben Mildenhall, Tianfan Xue, Jiawen Chen, Dillon Sharlet, Jonathan T. Barron
CVPR 2019

If you use this code, please cite our paper:

```
@inproceedings{brooks2019unprocessing,
  title={Unprocessing Images for Learned Raw Denoising},
  author={Brooks, Tim and Mildenhall, Ben and Xue, Tianfan and Chen, Jiawen and Sharlet, Dillon and Barron, Jonathan T},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019},
}
```

The code is implemented in Tensorflow and the required packages are listed in `requirements.txt`.

## Evaluation on Darmstadt Noise Dataset

In our paper, we evaluate on the [Darmstadt Noise Dataset](https://noise.visinf.tu-darmstadt.de/). Here are our [Darmstadt results](http://timothybrooks.com/tech/unprocessing/darmstadt-supp/). We highly recommend this dataset for measuring denoise performance on real photographs, as the dataset contains real noisy images, which after denoising and upon submission to the Darmstadt website will be compared against real clean ground truth. Here are instructions to [download this dataset](https://noise.visinf.tu-darmstadt.de/downloads). You'll also need to download [our trained models](https://drive.google.com/file/d/1MTFr-uaIKv5aWe7nXlhTaHBestLUiDLZ/view?usp=sharing) and unzip them into ./models/. Once downloaded, replace the provided `dnd_denoise.py` file with the version in this repository and follow the instructions below to run an unprocessing denoiser on this data.

```
/google-research$ python -m unprocessing.dnd_denoise \
    --model_ckpt=/path/to/models/unprocessing_srgb_loss/model.ckpt-3516383 \
    --data_dir=/path/to/darmstadt/data \
    --output_dir=/path/to/darmstadt/ouputs
```

Then follow instructions in the Darmstadt README file, including running `bundle_submissions.py` to prepare for submission to the Darmstadt Noise Dataset online benchmark.

## Training on MIRFlickr Images

In our paper, we train on source images from [MIRFlickr](https://press.liacs.nl/mirflickr/). We used the full MIRFLICKR-1M dataset, which includes 1 million source images, although the smaller MIRFLICKR-25000, which contains 25 thousand source images, can be used as well and is easier to download and store. Both versions are freely avaiable&mdash;here are instructions to [download this dataset](http://press.liacs.nl/mirflickr/mirdownload.html). Once downloaded, break images into `mirflickr/train` and `mirflickr/test` subdirectories. Then run the command below to train on MIRFlickr. Note that you may wish to downsample the source images prior to training to help remove JPEG compression artifacts, which we found slightly helpful in our paper.

```
/google-research$ python -m unprocessing.train \
    --model_dir=/path/to/models/unprocessing_mirflickr \
    --train_pattern=/path/to/mirflickr/train \
    --test_pattern=/path/to/mirflickr/test
```

## Training for Different Cameras

Our models are trained to work best on the Darmstadt Noise Dataset, which contains 4 cameras of various types, and low to moderate amounts of noise. They generalize well to other images, such as those from the HDR+ Burst Photography Dataset. However, if you would like to train a denoiser to work best on images from different cameras, you may wish to modify `random_ccm()`, `random_gains()` and `random_noise_levels()` in `py/unprocess.py` to best match the distribution of image metadata from your cameras. See our paper for details of how we modeled such metadata for Darmstadt. If your cameras have a special Bayer pattern outside of those supported, you will also need to modify `mosaic()` and `demosaic()` to match.

## Evaluation on Different Real Data

To run evaluation on real raw photographs outside of Darmstadt and HDR+ datasets will require loading and running the trained model similar to `dnd_denoise.py`, packing input raw images into a 4-channel image of Bayer planes, and estimating the variance from shot and read noise levels. Note that these models are only designed to work with *raw* images (not processed sRGB images).

Shot and read noise levels are sometimes included in image metadata, and may go by different names, so we recommend refering to the specification for your camera's metadata. The shot noise level is a measurement of how much variance is proportional to the input signal, and the read noise level is a measurement of how much variance is independent of the image. We calculate an approximation of variance using the input noisy image as `variance = shot_noise * noisy_img + read_noise`, and pass it as an additional input both during training and evaluation.

If shot and read noise levels are not provided in your camera's metadata, it is possible to empirically measure these noise levels by calibrating your sensor or inferring from a single image. Here is a great overview of [shot and read noise](http://people.csail.mit.edu/hasinoff/pubs/hasinoff-photon-2012-preprint.pdf), and here is an example [noise calibration script](https://android.googlesource.com/platform/cts/+/806b430/apps/CameraITS/tests/dng_noise_model/dng_noise_model.py).

## Unprocessing Images in Other Training Pipelines

If you are only looking to generate realistic raw data, you are welcome to use the unprocessing portion of our code separately from the rest of this project. Feel free to copy our `unprocess.py` file for all of your unprocessing needs!
