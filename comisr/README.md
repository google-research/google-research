# COMISR:Compression-Informed Video Super-Resolution

This repo contains the testing code for the paper in the ICCV 2021.
["COMISR: Compression-Informed Video Super-Resolution"](https://arxiv.org/abs/2105.01237)

*Disclaimer: This is not an official Google product.*

![COMISR sample](resources/comisr.png)

## Requirements

Install dependencies:
```
pip3 install -r requirements.txt
```
***or***
```
python3 -m pip install -r requirements.txt
```

**Pre-trained Model**

To install the pre-trained model, you need to first install the [gcloud sdk](https://cloud.google.com/sdk/docs/install).

The pre-trained model can be downloaded from: `gs://gresearch/comisr/model/`

Download the model files using the `gsutil` command (which is a command from the [gcloud sdk](https://cloud.google.com/sdk/docs/install)):
```shell
gsutil cp -r dir gs://gresearch/comisr/model/ model/
```

## Testing data

The vid4 testing data can be downloaded from: `gs://gresearch/comisr/data/`

The folder path should be similar to:\
.../testdata/lr_crf25/calendar\
.../testdata/lr_crf25/city\
.../testdata/lr_crf25/foliage\
.../testdata/lr_crf25/walk

.../testdata/hr/calendar\
.../testdata/hr/city\
.../testdata/hr/foliage\
.../testdata/hr/walk

## Creating compressed frames
We use [ffmpeg](https://www.ffmpeg.org/) to compress video frames. Below is one sample CLI usage.

Suppose you have a sequence of frames in im%2d.png format, e.g. calendar from vid4.

```shell
ffmpeg -framerate 10 -i im%2d.png -c:v libx264 -crf 0 lossless.mp4 \
&& ffmpeg -i lossless.mp4 -vcodec libx264 -crf 25 crf25.mp4 \
&& ffmpeg -ss 00:00:00 -t 00:00:10 -i crf25.mp4 -r 10 crf25_%2d.png
```

## Usage
```shell
python inference_and_eval.py \
--checkpoint_path=model/model.ckpt \
--input_lr_dir=lr/ \
--targets=hr/ \
--output_dir=output_dir/
```

- `--targets` is the folder containing the Ground Truth, this is used for evaluating the result
- `--output_dir` is the folder where the Upscaled image sequence goes

## Citation
If you find this code is useful for your publication, please cite the original paper:
```
@inproceedings{yli_comisr_iccv2021,
  title = {COMISR: Compression-Informed Video Super-Resolution},
  author = {Yinxiao Li and Pengchong Jin and Feng Yang and Ce Liu and Ming-Hsuan Yang and Peyman Milanfar},
  booktitle = {ICCV},
  year = {2021}
}
```


