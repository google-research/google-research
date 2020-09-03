FROM python:3.6

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y git

# Install Google Cloud SDK to use gsutil
RUN curl https://sdk.cloud.google.com > install.sh
RUN bash install.sh --disable-prompts --install-dir=/root

ENV PATH="/root/google-cloud-sdk/bin:${PATH}"

RUN git clone https://github.com/google-research/google-research.git --depth=1

# Download the data that is needed for running the notebooks
RUN gsutil -m cp -r gs://cloud-samples-data/research/activation_clustering/work_dir google-research/activation_clustering/examples/cifar10/

RUN gsutil -m cp -r gs://cloud-samples-data/research/activation_clustering/model.h5 google-research/activation_clustering/examples/cifar10/

RUN cd google-research/activation_clustering && pip install -e .

# Download cifar10 data and cache it in the image
RUN python -c "import tensorflow_datasets as tfds; tfds.load('cifar10', shuffle_files=False)"

ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--notebook-dir=google-research/activation_clustering/examples/cifar10"]
