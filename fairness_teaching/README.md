### Introduction
  Tensorflow implementation of Teaching with Fairness project

### Requirements
1. `Python 3.6` 
2. `TensorFlow 1.14.0`
3. `scikit-image`

### Usage

#### Part 1: Download the dataset from google drive

[Google Drive](https://drive.google.com/corp/drive/folders/1eQpLvkdGAuwJ4EaNjQKTBwe7NRksM-OH)

#### Part 2: Train and test with the following code in different folder.

```Shell
# train and generate fake images with GAN
cd gan
python train.py
python generate.py
```
```Shell
# train the teacher and student model with RL
cd rl
python train.py
python restore.py
```
```Shell
# compare with the baseline 
cd baseline
python all_real.py #train with all real data
python all_fake.py #train with all fake data
python random.py #train with random combination of real and fake data
python balance.py #train with balanced combination of real and fake data
```

