# Learning to Synthesize Motion Blur

Reference code for the paper [Learning to Synthesize Motion Blur](http://timothybrooks.com/tech/motion-blur).
Tim Brooks & Jonathan T. Barron, CVPR 2019. Please [cite](https://jonbarron.info/data/BrooksBarronCVPR2019.bib)
the paper if you use this code.

First, download [the dataset we use for evaluation](https://drive.google.com/file/d/1AcxuWl2PnqkyxyTgmzepwqw48G15rT2U/view?usp=sharing),
and extract it to `./test_dataset/`. Then, download [the output of our model](https://drive.google.com/file/d/1WmpPGV2iGU6MNlwEopxANAXuyMhnC675/view?usp=sharing),
and extract it to `./models/`. This folder contains the output of three trained models used in the paper:

* `./models/motion_blur/` corresponds to the complete model proposed in the paper,
* `./models/motion_blur_direct/` corresponds to the "Direct Prediction" ablation.
* `./models/motion_blur_uniform/` corresponds to the "Uniform Weight" ablation.

Run `evaluate.py` to evaluate the performance of a trained model on the motion blur test dataset.
