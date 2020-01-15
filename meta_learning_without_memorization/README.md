# Meta-Learning without Memorization
*Implemention of meta-regularizers as described in [Meta-Learning without Memorization]() by Mingzhang Yin, George Tucker, Mingyuan Zhou, Sergey Levine, Chelsea Finn.*

The ability to learn new concepts with small amounts of data is a critical aspect of intelligence that has proven challenging for deep learning methods. Meta-learning has emerged as a promising technique for leveraging data from previous tasks to enable efficient learning of new tasks. However, most meta-learning algorithms implicitly require that the meta-training tasks be *mutually-exclusive*, such that no single model can solve all of the tasks at once. For example, when creating tasks for few-shot image classification, prior work uses a per-task random assignment of image classes to N-way classification labels. If this is not done, the meta-learner can ignore the task training data and learn a single model that performs all of the meta-training tasks zero-shot, but does not adapt effectively to new image classes. This requirement means that the user must take great care in designing the tasks, for example by shuffling labels or removing task identifying information from the inputs. In some domains, this makes meta-learning entirely inapplicable. In this paper, we address this challenge by designing a meta-regularization objective using information theory that places precedence on data-driven adaptation. This causes the meta-learner to decide what must be learned from the task training data and what should be inferred from the task testing input. By doing so, our algorithm can successfully use data from *non-mutually-exclusive* tasks to efficiently adapt to novel tasks. We demonstrate its applicability to both contextual and gradient-based meta-learning algorithms, and apply it in practical settings where applying standard meta-learning has been difficult. Our approach substantially outperforms standard meta-learning algorithms in these settings.

This repository:
* Provides code to generate the pose regression dataset used in the paper.
* Implements Model Agnostic Meta Learning (MAML; Finn et al. 2017) and Neural Processes (NP; Garnelo et al. 2018) with meta-regularization on the weights and activations.
We hope that this code will be a useful starting point for future research in this area.

## Generating pose regression dataset
Requirements:
* TensorFlow (see tensorflow.org for how to install)
* numpy-stl
* gym
* mujoco-py

Step 1: Download CAD models from [Beyond PASCAL: A Benchmark for 3D Object Detection in the Wild](http://cvgl.stanford.edu/projects/pascal3d.html) [ftp://cs.stanford.edu/cs/cvgl/PASCAL3D+_release1.1.zip](ftp://cs.stanford.edu/cs/cvgl/PASCAL3D+_release1.1.zip) and use the CAD folder.

We removed the two classes 'bottle' and 'train' because the objects are symmetric.

Step 2: Convert the CAD models from \*.off to \*.stl.

You can download a converter from [https://www.patrickmin.com/meshconv](https://www.patrickmin.com/meshconv). Then, run

```
chmod 755 meshconv
find ./CAD -maxdepth 2 -mindepth 2 -name "*.off" -exec meshconv -c stl {} \;
```

Step 3: Render the dataset. Using the utilities in pose_data
```
CAD_DIR=
DATA_DIR=
python mujoco_render.py --CAD_dir=${CAD_DIR} --data_dir=${DATA_DIR}
cp -r ${DATA_DIR}/rotate ${DATA_DIR}/rotate_resize
python resize_images.py --data_dir=${DATA_DIR}/rotate_resize
python data_gen.py --data_dir=${DATA_DIR}/rotate_resize
```
This generates two pickle files: train_data.pkl and val_data.pkl.

## Train models on pose regression dataset
See pose_code/run.sh for examples of training the various algorithms.

This is not an officially supported Google product. It is maintained by George Tucker (gjt@google.com, [@georgejtucker](https://twitter.com/georgejtucker), github user: gjtucker).
