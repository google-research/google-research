# CKD

<!-- #TODO: please let me know if you would like me to add more details here. i intentionally kept it less informative before the paper comes out-->
This repo is a work in progress. More details are coming soon.


## Setup

```
# clone this repo
SUBDIR=ckd
svn export https://github.com/google-research/google-research/trunk/$SUBDIR

# create conda environment
conda env create --name ckd --file=ckd.yml
conda activate ckd

# install pytorch separately, e.g.,
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# clone lavis repo
git clone https://github.com/salesforce/LAVIS.git

# create a copy of source files
cp -r LAVIS/lavis ./ 

# optional
rm -rf LAVIS 

# apply patches
git apply --whitespace=fix ckd.patch
```


## Datasets

Please go [here](lavis/configs/default.yaml) and update `cache_root` as you wish, this cache location will be used to store all the datasets.

### Download images

Run the following commands to download the images of the respective dataset.

```
mkdir path/to/cache/ # update with cache_root
python download_vg.py
python download_coco.py
python download_gqa.py
```

### Training files

We will upload the training annotation files soon.

<!-- #TODO: update vg_sample_negatives.py file -->

```
# create a separate env to use Gemini based on pyhon 3.10. 
# a lower python version may encounter error.

conda env create --name py310 --file=py310.yml
conda activate py310

# we extract reposne from gemini following below steps
# please make sure to update paths based on your directory

# step 1: mine the negative objects for all images
python vg_sample_negatives.py

# step 2: cache response from gemini
python vg_cache_desc.py

```


### Evaluation files

Pope evaluation annotation files are obtained from https://github.com/RUCAIBox/POPE/tree/main/output.
You can run the following commands to download them in your cache directory.

```
# coco
wget https://github.com/RUCAIBox/POPE/blob/main/output/coco/coco_pope_adversarial.json -P /path/to/cache/POPE/coco
wget https://github.com/RUCAIBox/POPE/blob/main/output/coco/coco_pope_popular.json -P /path/to/cache/POPE/coco
wget https://github.com/RUCAIBox/POPE/blob/main/output/coco/coco_pope_random.json -P /path/to/cache/POPE/coco

# aokvqa
wget https://github.com/RUCAIBox/POPE/blob/main/output/seem/aokvqa/aokvqa_pope_adversarial.json -P /path/to/cache/POPE/aokvqa
wget https://github.com/RUCAIBox/POPE/blob/main/output/seem/aokvqa/aokvqa_pope_popular.json -P /path/to/cache/POPE/aokvqa
wget https://github.com/RUCAIBox/POPE/blob/main/output/seem/aokvqa/aokvqa_pope_random.json -P /path/to/cache/POPE/aokvqa

# gqa
wget https://github.com/RUCAIBox/POPE/blob/main/output/seem/gqa/gqa_pope_adversarial.json -P /path/to/cache/POPE/gqa
wget https://github.com/RUCAIBox/POPE/blob/main/output/seem/gqa/gqa_pope_popular.json -P /path/to/cache/POPE/gqa
wget https://github.com/RUCAIBox/POPE/blob/main/output/seem/gqa/gqa_pope_random.json -P /path/to/cache/POPE/gqa
```






## Training

Here is an example to submit a training job. We used 8 A100 GPUs per job, you may adjust the number of GPUs and/or batch size as per your setup.

```
bash src/train_ckd.sh
```

## Evaluation

You can run the following commands to separately evaluate on POPE.

```
bash src/eval_hall_iblip.sh
bash src/eval_hall_gemini.sh
```


## Acknowledgements

This repository is based on [LAVIS](https://github.com/salesforce/LAVIS).


## Contact

Please feel free to contact pritam.sarkar@queensu.ca for any questions or feedback.


*Disclaimer: This is not an official Google product.*

