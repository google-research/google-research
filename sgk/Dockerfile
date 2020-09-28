FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

# Install tools and dependencies.
RUN apt-get -y update --fix-missing
RUN apt-get install -y \
	emacs \
	git \
	wget \
	libgoogle-glog-dev

# Install TensorFlow.
RUN apt-get install -y python3-dev python3-pip
RUN pip3 install --upgrade pip && pip install --upgrade tensorflow_gpu==2.3
RUN pip install --upgrade tensor2tensor

# Install CMake.
RUN apt-get install -y software-properties-common && \
    apt-get update && \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' && \
    apt-get update && apt-get install -y cmake

# Install Sputnik.
RUN mkdir /mount
WORKDIR /mount
RUN git clone --recursive https://github.com/google-research/sputnik.git && \
	mkdir sputnik/build
WORKDIR /mount/sputnik/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TEST=OFF -DBUILD_BENCHMARK=OFF \
	-DCUDA_ARCHS="60;70" -DCMAKE_INSTALL_PREFIX=/usr/local/sputnik && \
	make -j8 install

# Copy the source into the image.
RUN mkdir -p /mount/sgk
COPY . /mount/sgk/

# Install SGK.
RUN mkdir /mount/sgk/build
WORKDIR /mount/sgk/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCHS="60;70" \
	-DCMAKE_INSTALL_PREFIX=/usr/local/sgk && \
	make -j8 install

# Setup the environment.
ENV PYTHONPATH="/mount:${PYTHONPATH}"
ENV LD_LIBRARY_PATH="`python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())'`:${LD_LIBRARY_PATH}"
ENV LD_LIBRARY_PATH="/usr/local/sputnik/lib:${LD_LIBRARY_PATH}"

# Set the working directory.
WORKDIR /mount/sgk
