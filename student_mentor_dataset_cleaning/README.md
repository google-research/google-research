Usage:

How to run training for one epoch and save the models in /tmp/models:

# From google-research/
python -m student_mentor_dataset_cleaning.main --save_dir=/tmp/models \
  --student_epoch_count=1 --mentor_epoch_count=1
