## Zero Shot Cardinality Training

This repository contains the infrastructure to produce training datasets for
cardinality estimation. This process is a multi-step process:

* Create down sampled versions of databases to reduce execution cost if needed
* Collect table, column information, calculate statistics
* Generate training sql queries
* Run queries to collect actual cardinalities
* Create training datasets (querygraps)

![CardBench](training_datasets/figures/cardbench.png)


As the cost of running this pipeline is significant we plan to release the final
product (the training dataset) in addition to the code that performs all the steps.

The training datasets and their documentation can be found in the training_dataset directory.

<span style="color:red">
We are in the process of updating the CardBench code and it will be released soon.
</span>