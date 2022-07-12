### FLOAT: Factorized Learning of Object Attributes for Improved Multi-object Multi-part Scene Parsing

This repository contains the code for the paper:

**FLOAT: Factorized Learning of Object Attributes for Improved Multi-object Multi-part Scene Parsing** <br>
Rishubh Singh, Pranav Gupta, Pradeep Shenoy, Ravi Kiran Sarvadevabhatla <br>
*IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2022)* <br>


#### Preparing
1. Creare a new conda environment `conda create -n floatseg python=3.7.10`.
2. Activate it `conda activate floatseg`.
3. Install the requirements `pip install -r floatseg/requirements.txt`
4. Manually download all dataset versions used in this paper from [Zenodo](https://zenodo.org/record/6374908).

#### Usage
The float_part<58/108/201>_train.ipynb notebooks can be run to train the FLOAT model(s) with preset training settings.
The float_part<58/108/201>_inference.ipynb notebooks can be run to replicate our results.

#### Citation
If you find our methods useful, please cite:

```
@InProceedings{Singh_2022_CVPR,
    author    = {Singh, Rishubh and Gupta, Pranav and Shenoy, Pradeep and Sarvadevabhatla, Ravikiran},
    title     = {FLOAT: Factorized Learning of Object Attributes for Improved Multi-Object Multi-Part Scene Parsing},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {1445-1455}
}
```