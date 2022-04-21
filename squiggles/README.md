# Source code to generate the squiggles dataset.
The squiggles dataset allows you to generate curves
and image representations of them. 
The generative process is fully differentiable and the data
can be automatically labelled. The fully differentiable nature of the
generative process enables us to navigate the true data manifold.
For more details on the dataset, we refer to the paper. 
(Link to be added after an update when the paper is posted to arxiv.)

# Usage and caveats
*  An example to generate the dataset is given in run.sh. 


*   The basis for the filenames has to be lowercase without hyphen or underscores due to limitations in tfds metadata writing.

*   The dataset can only be generated in a dedicated folder. Writing the metadata will fail if there are files present belonging to a different dataset.

* Generating large quantities of data might be time consuming and require additional parallelization.
