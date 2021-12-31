# 2.5D Visual Relationship Detection Evaluation

This directory contains script for evaluating 2.5D visual relationship detection
results. Please download the dataset to ./data directory before running the
script.

Sample command line:

```bash
python -m visual_relationship.evaluation.evaluate_vrd_main
  --groundtruth_object_path=$GROUNDTRUTH_OBJECT
  --groundtruth_vrd_path=$GROUNDTRUTH_VRD
  --prediction_path=$PREDICTION
  --output_path=$OUTPUT
```

`GROUNDTRUTH_OBJECT` and `GROUNDTRUTH_VRD` are the ground truth files provided
by the dataset.

`PREDICTION` is the predictions to be evaluated.

`OUTPUT` is where the evaluation results will be saved.

## Data Format

The prediction should be stored in a csv file. Each row in the csv file
corresponds to a predicted (object_id_1, object_id_2, predicates) triplet. The csv
file should contain the following columns:

* image_id_1: The image id for object_id_1
* entity_1: The predicted entity for object_id_1
* xmin_1, xmax_1, ymin_1, ymax_1: The bounding box coordinates for object_id_1
* image_id_2: The image id for object_id_2
* entity_2: The predicted entity for object_id_2
* xmin_2, xmax_2, ymin_2, ymax_2: The bounding box coordinates for object_id_2
* occlusion: The predicted occlusion relationship between object_id_1 and object_id_2.
  Possible values include:
  - 0: No occlusion
  - 1: object_id_1 occludes object_id_2
  - 2: object_id_2 occludes object_id_1
  - 3: object_id_1 occludes and is occluded by object_id_2
* distance: The predicted distance relationship between object_id_1 and object_id_2.
  Possible values include:
  - 0: not sure
  - 1: object_id_1 is closer to the camera than object_id_2
  - 2: object_id_2 is closer to the camera than object_id_1
  - 3: object_id_1 is about the same distance to the camera as object_id_2

Please refer to the dataset description for more information about the definition.
