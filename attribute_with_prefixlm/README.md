# ArtVLM: Attribute Recognition Through Vision-Based Prefix Language Modeling

This directory contains information on the VGARank dataset from the ECCV 2024 paper:
"ArtVLM: Attribute Recognition Through Vision-Based Prefix Language Modeling"
by William Yicheng Zhu*, Keren Ye*, Junjie Ke, Jiahui Yu, Leonidas Guibas, Peyman Milanfar, Feng Yang.

The dataset is hosted on [Kaggle](https://www.kaggle.com/datasets/googleai/vgarank).

## Dataset Description
The VGARank dataset adds two fields, attribute_prompts (VGARank-Attribute) and object_prompts (VGARank-Object) to the [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html) dataset at the instance level (one bounding box in an image, which means multiple instances could exist for a single image). Below is an example from the dataset, where the two additional fields are added to the existing annotations for one instance bounding box.

<pre><code>
  {
    "image_id": 2388999,
    "attributes": [
      {
        "synsets": [],
        "h": 44,
        "object_id": 4504988,
        "names": ["fence"],
        "w": 60,
        "attributes": ["black"],
        "y": 316,
        "x": 423,
        "attribute_prompts": [
          ["fence", "brick", 0],
          ["fence", "yellow", 0],
          ["fence", "barbed", 0],
           …
        ],
        "object_prompts": [
          ["black stripe", "black", 0],
          ["black shoe", "black", 0],
          ["label", "black", 0],
	         …
        ],
        "norm_w": 0.12,
        "norm_h": 0.12021857923497267,
        "norm_x": 0.846,
        "norm_y": 0.8633879781420765
      },
    ]
  }
</code></pre>


<b>attribute_prompts</b>:
A list of attribute-centered answer choices ([obj, att, 0 or 1]) constructed from the raw dataset distribution, where the object is fixed and the attribute is varied.
For each bounding box instance, Visual Genome provides the ground-truth object and attribute, which we use as the positive choices ([obj, att, 1]) in the question set for each instance. 
For the negative choices ([obj, att, 0]), we choose the most likely attributes (that are not in the positive set) associated with the fixed object in descending order by looking at the dataset prior, i.e. which attributes are generally associated with the object, but not true for this particular bounding box.

<b>object_prompts</b>:
Exactly the same as attribute_prompts, except the role of attributes and objects are swapped: now the attribute is fixed while the object is varied across 50 questions.


## Task Definition
Both the VGARank-Attribute and VGARank-Object tasks are defined as ranking tasks, where the objective is to rank ground truth pairs higher than false pairs. For VGARank-A, the ground truth is the object and attribute in the original VG annotation, while false pairing are created by selecting attributes that are often associated with the ground truth object but are not present in the current instance. For example, for “car is red”, false pairing attributes may be “car is blue” and “car is yellow”, which can be true statements for some cars but not for the one in question within the bounding box.

For VGARank-O, the ground truth is again from the original VG annotation, but negatives are created differently, now fixing on the ground truth attribute while varying on the object. For example, given “car is red”, negatives can be “bird is red” and “logo is red”.

## Dataset Size
We obtain a dataset with 770,721 training images, 7,997 validation images, and 32,299 testing images for each tasks. Further details regarding the preprocessing on VG are provided in the supplementary materials in the paper.

## License
CC-BY 4.0

## Citation

If you find this dataset useful for your publication, please cite the original paper:

```
@inproceedings{zhu2024artvlm,
  title = {ArtVLM: Attribute Recognition Through Vision-Based Prefix Language Modeling},
  author={Zhu, William Yicheng and Ye, Keren and Ke, Junjie and Yu, Jiahui and Guibas, Leonidas and Milanfar, Peyman and Yang, Feng},
  booktitle={to be provided},
  pages={to be provided},
  year={2024}
}
```
