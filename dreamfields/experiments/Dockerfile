# FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-9
FROM nvcr.io/nvidia/pytorch:22.04-py3

RUN if ! id 1000; then useradd -m -u 1000 clouduser; fi

RUN mkdir /workdir
WORKDIR /workdir
ENV LANG=C.UTF-8
ENV TFDS_DATA_DIR=gs://tensorflow-datasets/datasets
RUN rm -f /etc/apt/sources.list.d/cuda.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

# GLIDE
RUN mkdir -p /workdir/glide_model_cache
RUN cd /workdir/glide_model_cache && wget https://openaipublic.blob.core.windows.net/diffusion/dec-2021/base.pt
RUN chown -R 1000:root /workdir/glide_model_cache
# RUN ln -s /gcs/xcloud-shared/jainajay/diffusion_3d/glide_model_cache/ /workdir/glide_model_cache

# CLIP
RUN mkdir -p /workdir/clip_model_cache
RUN cd /workdir/clip_model_cache && \
    wget https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
RUN chown -R 1000:root /workdir/clip_model_cache
# RUN ln -s /gcs/xcloud-shared/jainajay/diffusion_3d/clip_model_cache/ /workdir/clip_model_cache

RUN apt-get update && apt-get install -y git netcat htop vim less
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y ffmpeg

RUN python -m pip install --upgrade pip
# RUN conda install pytorch torchvision cudatoolkit -c pytorch
COPY requirements.txt /workdir/requirements.txt
RUN python -m pip install -r requirements.txt

RUN cd /workdir && git clone --recurse-submodules -j2 https://github.com/NVlabs/tiny-cuda-nn && \
    chown -R 1000:root /workdir/tiny-cuda-nn
COPY tcnn_setup.py ./tcnn_setup.py
RUN chown -R 1000:root ./tcnn_setup.py && chmod -R 775 ./tcnn_setup.py && \
    mv ./tcnn_setup.py /workdir/tiny-cuda-nn/bindings/torch/setup.py && \
    python -m pip install /workdir/tiny-cuda-nn/bindings/torch

COPY entrypoint.sh ./entrypoint.sh
RUN chown -R 1000:root ./entrypoint.sh && chmod -R 775 ./entrypoint.sh

COPY diffusion_3d/ /workdir/diffusion_3d
RUN chown -R 1000:root /workdir/diffusion_3d && chmod -R 775 /workdir/diffusion_3d

ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all
ENTRYPOINT ["./entrypoint.sh"]
