FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

# Install the basics
RUN apt-get -y update --fix-missing
RUN apt-get install -y \
	emacs \
	htop \
	python2.7 \
	python-pip \
	git
RUN pip install --upgrade pip

# Add the source-code
RUN mkdir /home/state_of_sparsity
WORKDIR /home/state_of_sparsity
COPY . .

# Install the dependencies
RUN pip install -r requirements.txt

# Set the python path
RUN export PYTHONPATH=$PYTHONPATH:`pwd`
