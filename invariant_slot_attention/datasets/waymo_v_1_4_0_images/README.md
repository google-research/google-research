# Preprocessing Waymo Open

1. Download the training and validation splits of Waymo Open **v1.4**. You should have `individual_files/training` and `individual_files/validation`. This dataset is 300GB+, but the final TFDS will be much smaller.

2. Unfortunately, there is a conflict between the tensorflow versions required to preprocess this dataset and to run the rest of the repository. Install `requirements.txt` in this folder in a separate virtual environment.

3. `cd` your terminal into `waymo_v_1_4_0_images`.

4. Run the following scripts, possibly in parallel. We extract the data from the front-facing camera and require instance segmentation masks only for validation.
    * python preprocess.py --camera_name=1 --split_name=training --require_masks=False
    * python preprocess.py --camera_name=1 --split_name=validation --require_masks=True

5. You have successfully created the dataset in a tfrecords format located in `tfrecords`. Our Waymo Open configs already point to `datasets/waymo_v_1_4_0_images/tfrecords`.
