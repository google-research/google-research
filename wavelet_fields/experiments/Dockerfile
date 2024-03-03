FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

# Below command creates home dir for 1000 UID user if it is not present.
RUN if ! id 1000; then useradd -m -u 1000 clouduser; fi

RUN mkdir /wavelet_repr
WORKDIR /wavelet_repr

ENV LANG=C.UTF-8

COPY entrypoint.py /wavelet_repr/

# copy wheels to install
COPY svox_wheels/pybind11-2.10.4-py3-none-any.whl /wavelet_repr/svox_wheels/pybind11-2.10.4-py3-none-any.whl
COPY svox_wheels/svox-0.2.32-cp38-cp38-linux_x86_64.whl /wavelet_repr/svox_wheels/svox-0.2.32-cp38-cp38-linux_x86_64.whl
COPY svox_wheels/swvox-0.2.32-cp38-cp38-linux_x86_64.whl /wavelet_repr/svox_wheels/swvox-0.2.32-cp38-cp38-linux_x86_64.whl

COPY plenoctree/environment.yml /wavelet_repr/plenoctree/environment.yml
COPY plenoctree/requirements.txt /wavelet_repr/plenoctree/requirements.txt
COPY my_python_utils/requirements.txt /wavelet_repr/my_python_utils/requirements.txt

RUN conda env create -f plenoctree/environment.yml

RUN conda init bash
SHELL ["/bin/bash", "-c"]
RUN conda init bash

# Activate the environment and install any required dependencies
RUN echo "conda activate plenoctree" >> ~/.bashrc

# Upgrade pip
RUN /bin/bash -c "source activate plenoctree && pip install --upgrade pip"

RUN /bin/bash -c "source activate plenoctree && pip install trimesh pysdf seaborn pyexr PyMCubes"

WORKDIR /wavelet_repr
RUN /bin/bash -c "source activate plenoctree && pip install -r plenoctree/requirements.txt"
RUN /bin/bash -c "source activate plenoctree && pip install torch==1.13.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html"

WORKDIR /wavelet_repr/my_python_utils
RUN /bin/bash -c "source activate plenoctree && pip install -r requirements.txt"
RUN /bin/bash -c "source activate plenoctree && conda install -c conda-forge opencv"
RUN /bin/bash -c "source activate plenoctree && conda install -c conda-forge ffmpeg"
WORKDIR /wavelet_repr

# Install cudatoolkit
# RUN conda install cudatoolkit-dev=11.0 -c pytorch
# RUN conda install -c conda-forge pybind11
RUN /bin/bash -c "source activate plenoctree && conda install -y -c conda-forge 'pybind11>=2.10'"
RUN /bin/bash -c "source activate plenoctree && pip install pybind11==2.10"
RUN /bin/bash -c "source activate plenoctree && pip install --upgrade jax==0.2.8 jaxlib==0.1.57+cuda110 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
RUN /bin/bash -c "source activate plenoctree && pip install --upgrade numpy==1.19.5 scipy==1.6.0 tqdm"
RUN /bin/bash -c "source activate plenoctree && conda install -c anaconda cmake"
RUN /bin/bash -c "source activate plenoctree && pip uninstall svox svox2"

RUN /bin/bash -c "source activate plenoctree && pip install seaborn"
RUN /bin/bash -c "source activate plenoctree && pip install --upgrade matplotlib==3.1.1"
# install again numpy version
RUN /bin/bash -c "source activate plenoctree && pip install --upgrade numpy==1.19.5"

# ptwt
RUN /bin/bash -c "source activate plenoctree && pip install pywavelets==1.3.0"
RUN /bin/bash -c "source activate plenoctree && pip install ptwt==0.1.6"


# once this is done, uncomment the following line, which will copy the previously built wheel to the docker image instead of building it.
WORKDIR /wavelet_repr
RUN /bin/bash -c "source activate plenoctree && pip install svox_wheels/pybind11-2.10.4-py3-none-any.whl"
RUN /bin/bash -c "source activate plenoctree && pip install svox_wheels/svox-0.2.32-cp38-cp38-linux_x86_64.whl"
RUN /bin/bash -c "source activate plenoctree && pip install svox_wheels/swvox-0.2.32-cp38-cp38-linux_x86_64.whl"

# Finally we need to repair permissions
RUN chown -R 1000:root /wavelet_repr && chmod -R 777 /wavelet_repr
RUN chown -R 1000:root /home && chmod -R 777 /home
RUN chown -R 1000:root /opt/conda/envs/plenoctree/lib/python3.8/site-packages/swvox-0.2.32.dist-info && chmod -R 777 /opt/conda/envs/plenoctree/lib/python3.8/site-packages/swvox-0.2.32.dist-info
RUN chown -R 1000:root /opt/conda/envs/plenoctree/lib/python3.8/site-packages/swvox && chmod -R 777 /opt/conda/envs/plenoctree/lib/python3.8/site-packages/swvox


ENTRYPOINT ["python", "$HOME/wavelets_code/entrypoint.py"]
