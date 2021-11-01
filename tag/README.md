### Efficiently Identifying Task Groupings for Multi-Task Learning
This repository contains the code to reproduce the results described in [Efficiently Identifying Task Groupings for Multi-Task Learning](https://arxiv.org/abs/2109.04617). The code can be roughly divided into three concepts: experiments on CelebA, experiments on Taskonomy, and the logic for Network Selection. The experiments on CelebA are implemented in TensorFlow-2 with Keras while the experiments on Taskonomy use Pytorch.

#### CelebA
The code for TAG, CS, GradNorm, PCGrad, and Uncertainty Weights are encapsulated within the `google-research/tag/celeba/CelebA.ipynb` colab. Notably, one can control which method is run by the FLAGS.method variable. The mapping of methods to flags is contained within the code. Results for HOA and RG were determined by training many combinations of tasks together using the 'fast_mtl' flag.

#### Taskonomy
Our results on Taskonomy can be reproduced with the following steps:

1. Clone the [Which Tasks to Train Together in Multi-Task Learning github repository](https://github.com/tstandley/taskgrouping)
2. Revert your version to the repository version on 09.17.2021 to maintain consistency with `git reset --hard dc6c89c269021597d222860406fa0fb81b02a231`
3. Create a branch called tag in the repository: `git checkout -b tag`
4. Apply the patches in `google-research/tag/taskonomy/`
    a. `git apply gradnorm.patch`
    b. `git apply xception_taskonomy_new.patch`
    c. `git apply tag.patch`
5. Replace the full 12 TB dataset with a smaller ~2 TB dataset by overriding the buildings list in `taskgrouping/train_models.txt` with the corresponding list in `tag/taskonomy/train_models.txt`. Repeat for the validation and test datasets.
6. Follow the instructions on the [Which Tasks to Train Together in Multi-Task Learning github repository](https://github.com/tstandley/taskgrouping) to download and run the models.

#### Network Selection
The Network Selection colabs take in a dictionary of task relationships and outputs a set of task groupings. In HOA, this dictionary is the pairwise validation accuracy or loss when pairs of tasks are trained together. In TAG, this dictionary is the pairwise inter-task affinity scores averaged together throughout training and computed every 10 steps. In CS, this dictionary is the pairwise cosine similarity scores averaged throughout training and computed at every step.

The two colabs  `tag/network_selection/HOA_network_selection.ipynb` and `tag/network_selection/TAG_network_selection.ipynb` are different. The former selects groups to minimize validation error (or loss) of each task while the latter selects groups to maximize the inter-task affinity (or cosine similarity) score onto every task.
