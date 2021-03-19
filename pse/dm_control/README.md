
Code for the distracting DM control experiments in [Contrastive behavioral similarity embeddings
for generalization in reinforcement learning](https://agarwl.github.io/pse/), published at ICLR 2021.

To run the code, follow the following instructions:

 - Set up [Distracting Control](https://arxiv.org/abs/2101.02722) using the
   instructions [here](https://github.com/google-research/google-research/blob/master/distracting_control/README.md).
     - This requires downloading setting up DM control and downloading the DAVIS 2017 dataset.
 - For the experiments in the paper, we used 2 videos (`['bear', 'bmx-bumps']`) for the background distractions
   during training and all the [`DAVIS17_VALIDATION_VIDEOS`](https://github.com/google-research/google-research/blob/master/distracting_control/background.py) for distractions for evaluating generalization at test
   time.
 - For contrastive loss for PSEs, we use a dataset collected from a pretrained policy (using DrQ) that achieves
   good returns on the Distracting Control environment. The dataset can be downloaded from this public [GCP bucket](
   https://console.cloud.google.com/storage/browser/pse_iclr21) using [`gsutil`](https://cloud.google.com/storage/docs/gsutil).
    - To download the dataset, please run

        ```
        # All Distracting Control environments:
        gsutil -m cp -R gs://pse_iclr21/dm_control/pse_data
        # A specific `ENV` (such as 'ball_in_cup-catch'):
        gsutil -m cp -R gs://pse_iclr21/dm_control/pse_data/ENV
        ```

    - To collect data on your own domain, please use the `data_collection/collect_data` script with a pretrained checkpoint.
    - Note that the script exploits the fact that a given fixed action sequence has the same performance across different training environments in Distracting Control.
 - Launch training using the `run_train_eval.py` file and evaluate generalization using the `generalization_evaluator.py` (which reads checkpoints from the trained agent and evaluates them). 
