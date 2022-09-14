<div id="top"></div>
<!--
*** This readme was adapted from Best-README-Template.
  https://github.com/othneildrew/Best-README-Template
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#new-data">New Dataset</a></li>
      </ul>
    </li>
    <li><a href="#structure">Project Structure</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project


A novel deep learning framework that empowers users to use cloud forecasting services with their domain knowledge.

Background:
* Existing forecasting solutions for industry partners are very passive: massive amounts of time series data being fed into complex AI models to generate future predictions.
* Users receive predictions from the model but have no way to provide their domain knowledge, or interact with the AI models in any meaningful way.
* When providing forecasting solutions to our industry partners across sectors, it is critical that we recognize users from different sections carry their unique domain knowledge (e.g., association rules, market principles, operation logistics).

Objective:
* This project aims to integrate meta-data and user feedback as knowledge graphs into deep learning, leading to intelligent, adaptive, domain-specific forecasting solutions.


<p align="right">(<a href="#top">back to top</a>)</p>



### Built With

This section should list any major frameworks/libraries used to bootstrap your project. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.

* [Pytorch](https://pytorch.org/)
*  Numpy
*  [Ray](https://docs.ray.io/en/latest/tune/index.html)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

_Below is the steps you need to take to get started._

1. Clone the latest version of the project from [Cloud Source Repositories](https://source.cloud.google.com/). See <a href="#Installation">Installation</a>.
2. Generate the dataset. For example, a set of sine functions.
3. Modify the hyperparameter tuning range in [raytune_train.py](./raytune_train.py) file
4. Run the training scripts and make predictions.

### Prerequisites

_This is the list of prerequisies that you need to use the project and how to install them._

* CUDA

* Ananconda

* Pytorch
  ```sh
  conda install pytorch torchvision -c pytorch
  ```

### Installation

_Below is how you can install and setting up the project on GCP._

1. Create a Deep Learning VM instance at [GCP](https://cloud.google.com/deep-learning-vm)
2. Change directory to the repo
   ```sh
   cd graph_temporal_ai
   ```
3. Run the script to install packages
   ```sh
   bash run.sh
   ```


### New Dataset

_Below is how you can modify the code and train with your own dataset._

1. Prepare your data in ./data folder
  * Time serie data: numpy 3D array [num_time_step x num_nodes x num_features]
  * Graph data (if using known graph): numpy 2D array  [num_nodes x num_nodes]
2. Create a config file in ./experiments/your_data.yaml
3. Run the train file as
    ```sh
    python raytune_train.py --data-config=[config_file]
    ```



<p align="right">(<a href="#top">back to top</a>)</p>


## Project Structure

_Below is the file structure of the project._

* src: source files for deep learning model architectures and utility functions

* scripts: train scripts for running the models and bash scripts for hyper-parameter tuning

* notebooks: jupyter notebooks for pre-processing the data and visualizing the predictions

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

_Below shows a simple example of forecasting multivariate sine functions with the models_

1. Generate synthetic sine functions
  ```sh
  python scripts
  ```


<p align="right">(<a href="#top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Rose Yu  - roseyu@google.com

<p align="right">(<a href="#top">back to top</a>)</p>

